#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.


import sys
sys.path.append(r'C:\Users\25758\PycharmProjects\YOLOX')

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets.voc_classes import VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    # 数据类型
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    # 模型名称
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    # 数据路径，精确到图像时表示单图推理，文件夹表示批次推理
    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    # 是否保存结果
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    # 模型配置文件路径
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    # 模型权值文件路径
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    # 置信度
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    # nms阈值
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    # 测试图像尺寸
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    # 是否兼容以前版本
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def load_engine(engine_path):
    # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

class TRT():
    def __init__(self,engine):
        self.context = engine.create_execution_context()
        self.outshape = self.context.get_binding_shape(1)

    def inference(self,image):
        image=image.cpu().detach().numpy()
        image = image.ravel()  # 数据平铺
        output = np.empty((self.outshape), dtype=np.float32)
        d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
        d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
        bindings = [int(d_input), int(d_output)]
        # stream = cuda.Stream()

        cuda.memcpy_htod(d_input, image)
        self.context.execute_v2(bindings)
        cuda.memcpy_dtoh(output, d_output)
        output = torch.tensor(output.reshape(self.outshape))
        print("%"*30)
        print(output.shape)
        print("%"*30)
        # output = torch.nn.Softmax(dim=1)(torch.tensor(output))
        return output

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=VOC_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.trt_file=trt_file
        if trt_file is not None:
            # x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            # self.model(x)
            engine = load_engine(trt_file)
            model_trt=TRT(engine)
            self.model=model_trt

    def inference(self, img):
        # 推理一张图像
        # 将图像信息保存起来
        img_info = {"id": 0}
        if isinstance(img, str):
            # 获取图像的名称
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # 不论输入图像的尺寸，都统一缩放的指定尺寸，获取缩放比例，长边对齐
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        # 降维
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            if self.trt_file is not None:
                outputs = self.model.inference(img)
            else:
                outputs = self.model(img)
            # if self.decoder is not None:
                # outputs = self.decoder(outputs, dtype=outputs.type())
            # 将每一张图像的所有box信息输入，利用conf_thre,nms等过滤一部分box。
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        # 如果是文件夹，就是批量预测
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        # 推理
        outputs, img_info = predictor.inference(image_name)
        print("&"*30)
        print(outputs)
        print("&"*30)
        # 可视化
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    # 创建信息输出文件夹
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    # 写入日志
    logger.info("Args: {}".format(args))

    # 将从命令行获取的参数，赋值给实验对象
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # 获得模型对象
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    # 不是使用trt
    if not args.trt:
        if args.ckpt is None:
            # 如何我们没有给定权值文件，系统自动在输出目录寻找best_ckpt.pth
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    # 使用trt
    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        # trt路径
        if args.ckpt is None:
            trt_file = os.path.join(file_name, "model_trt.pth")
        else:
            trt_file = args.ckpt
        print(trt_file)
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    # 实例化预测器
    predictor = Predictor(
        model, exp, VOC_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    # 获取模型配置对象
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)

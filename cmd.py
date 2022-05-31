train=r"python tools/train.py -f exps/example/yolox_voc/yolox_voc_m.py -d 1 -b 32 -c weights/yolox_m.pth"
test_pth=r'python tools/demo.py image -f C:\Users\25758\PycharmProjects\YOLOX\exps\example\yolox_voc\yolox_voc_s.py -n yolox-s -c weights/best_ckpt.pth --save_result --path assets'
test_engine=r'python tools/demo.py image -f C:\Users\25758\PycharmProjects\YOLOX\exps\example\yolox_voc\yolox_voc_s.py -n yolox-s --trt -c weights/model_trt.engine --save_result --path assets'

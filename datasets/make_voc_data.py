import random
import os


if __name__=="__main__":
    train_pr=0.9
    xml_names=os.listdir(r'E:\BaiduNetdiskDownload\FaceMask\kitti1')
    xml_names2=os.listdir(r'C:\Users\25758\PycharmProjects\YOLOX\datasets\VOCdevkit\VOC2007\Annotations')
    xml_names=list(set(xml_names2) - set(xml_names))
    nums=len(xml_names)
    train_nums=int(train_pr*nums)
    list=range(nums)
    train_index=random.sample(list,train_nums)
    trainval=open('VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'a')
    # val=open('VOCdevkit/VOC2007/ImageSets/Main/val.txt', 'w')
    test=open('VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'a')
    for i in list:
        name  = xml_names[i].split('.')[0]+'\n'
        if i in train_index:
            trainval.write(name)
        else:
            test.write(name)

    trainval.close()
    test.close()


# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：trian.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO(model=r'E:\paper\YOLOv8-DeepSORT-Object-Tracking-main\ultralytics-main\ultralytics\cfg\models\12\yolo12s.yaml')
    model = YOLO(model=r'E:\paper\YOLOv8-DeepSORT-Object-Tracking-main\ultralytics-main\runs\train\yolo12s2\weights\last.pt')
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'data.yaml',
                imgsz=(480,640),
                epochs=300,
                batch=8,
                workers=8,
                device=0,
                optimizer='SGD',
                close_mosaic=10,
                resume=True,
                project='runs/train',
                name='yolo12s',
                single_cls=False,
                cache=False,
                pretrained=False
                )

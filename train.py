import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/SBF-YOLO.yaml')
    #model.load('yolov8s.pt') # loading pretrain weights
    model.train(data='ultralytics/cfg/datasets/崩边.yaml',
                cache=True,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
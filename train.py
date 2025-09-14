import torch
from ultralytics import YOLO
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # GPU 사용 가능한지 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 모델 로드 (GPU 사용 시 자동으로 CUDA로 할당됨)
    model = YOLO("yolov8m.pt")  # 기본적으로 GPU 사용 시 자동 할당됨

    # 학습
    model.train(data="dataset.yaml", epochs=50, device=device, )

from ultralytics import YOLO
import torch

# 모델 로드
model = YOLO('yolov8x.pt')  # 또는 'yolov8m.pt', 'yolov8l.pt' 등

# 모델 파인튜닝
results = model.train(
    data='path/to/your/data.yaml',  # 데이터셋 설정 파일
    epochs=50,                       # 훈련 에폭
    imgsz=640,                       # 이미지 크기
    batch=16,                        # 배치 크기 (RTX 4090에 최적화)
    device=0,                        # GPU 디바이스
    workers=8                        # 데이터 로더 워커 수
)

# 모델 평가
metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")

# 추론
results = model('path/to/image.jpg')
#yolo train model=yolov8m.pt data=dataset.yaml epochs=50 pretrained=C:/capstone/yolo/runs/detect/train17/weights/best.pt freeze=10
#yolo train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640 pretrained=best.pt
#모자 크기 우려 4521355984764928
#yolo train model=yolo11m.pt data=dataset.yaml epochs=50 pretrained=False box=5.5
#C:\capstone\yolo>yolo train model=rtdetr-x.pt data=dataset.yaml epochs=50
#yolo train model=yolo11x_resnet.yaml data=dataset.yaml amp=True augment=True mosaic=1.0 mixup=0.1 copy_paste=0.1 degrees=10.0 translate=0.2 scale=0.5 shear=0.1 perspective=0.0003 flipup=0.01 fliplr=0.5 hsv_h=0.015 hsv_s=0.4 hsv_v=0.4 erasing=0.2 lr0=0.005 cos_lr=True dropout=0.1 
#C:\capstone\yolo>yolo train model=yolo11x_resnet.yaml data=dataset.yaml batch=24 amp=True augment=True mosaic=1.0 mixup=0.1 copy_paste=0.1 degrees=10.0 translate=0.2 scale=0.5 shear=0.1 perspective=0.0003 flipud=0.01 fliplr=0.5 hsv_h=0.015 hsv_s=0.4 hsv_v=0.4 erasing=0.2 lr0=0.005 cos_lr=True dropout=0.1 
#epochs=10

#마지막 개선+ resnet 이식부분
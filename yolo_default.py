from ultralytics import RTDETR
import cv2
import numpy as np
import os

# 모델 불러오기 - RTDETR 클래스 사용
model = RTDETR("C:/capstone/yolo/runs/detect/train22/weights/best.pt")  # 또는 YOLO 클래스를 계속 사용해도 됩니다

# 이미지 경로
image_path = "C:/capstone/yolo/0407test02.jpg"
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

# 추론 진행
results = model(image_path, conf=0.40)

# 감지된 객체를 직접 처리하여 시각화
result = results[0]  # 첫 번째 이미지의 결과
boxes = result.boxes  # 바운딩 박스

# 사본 생성 (원본 이미지 보존)
output_img = image.copy()

# 클래스 이름 목록
class_names = [
    'person_all', 'head', 'face', 'eye', 'nose', 'mouth', 'ear', 'hair',
    'neck', 'body', 'arm', 'hand', 'hat', 'glasses', 'eyebrow', 'beard',
    'open_mouth_(teeth)', 'muffler', 'tie', 'ribbon', 'ear_muff', 'earring', 'necklace', 'ornament',
    'headdress', 'jewel', 'cigarette'
]

# 디버깅: 모델이 감지한 정보 출력
print("모델이 감지한 박스 정보:")
for i, box in enumerate(boxes):
    print(f"Box {i+1}:")
    print(f"  xyxy: {box.xyxy}")
    print(f"  conf: {box.conf}")
    print(f"  cls: {box.cls}")

# 바운딩 박스 그리기
for box in boxes:
    try:
        # 좌표 추출
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        
        # 좌표 정수로 변환
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 클래스 이름
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        
        print(f"그리는 박스: {class_name}, 좌표: ({x1}, {y1}, {x2}, {y2}), 신뢰도: {conf:.2f}")
        
        # 박스 색상 설정 (클래스마다 다른 색상)
        color_idx = cls_id % 10
        colors = [
            (0, 255, 0),    # 녹색
            (255, 0, 0),    # 파란색
            (0, 0, 255),    # 빨간색
            (255, 255, 0),  # 청록색
            (0, 255, 255),  # 노란색
            (255, 0, 255),  # 마젠타
            (128, 128, 0),  # 올리브
            (0, 128, 128),  # 틸
            (128, 0, 128),  # 퍼플
            (255, 165, 0)   # 오렌지
        ]
        color = colors[color_idx]
        
        # 바운딩 박스 그리기
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        
        # 텍스트 정보
        text = f"{class_name} {conf:.2f}"
        
        # 텍스트 크기 계산
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        
        # 텍스트 배경 그리기
        cv2.rectangle(output_img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        
        # 텍스트 그리기
        cv2.putText(output_img, text, (x1, y1 - 5), font, scale, (255, 255, 255), thickness)
    except Exception as e:
        print(f"박스 처리 중 오류 발생: {e}")

# 이미지 표시 및 저장
cv2.imshow("Custom Visualization", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_file = "custom_detection_" + os.path.basename(image_path)
cv2.imwrite(output_file, output_img)
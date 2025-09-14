from ultralytics import YOLO
import cv2
import torch

# 1. 학습된 모델 불러오기(경로를 실제 모델 파일로 수정)
model = YOLO("C:/capstone/yolo/runs/detect/train7/weights/best.pt")

# 2. 추론할 이미지 경로 지정
image_path = "C:/capstone/yolo/0222test02.jpg"

# 3. 모델을 통해 추론 진행
results = model.predict(source=image_path, conf=0.35)

# 4. 이미지 읽기 (OpenCV는 BGR 형식으로 읽음)
image = cv2.imread(image_path)

# 5. 클래스 이름 리스트(클래스 순서대로)
class_names = [
    'person_all', 'head', 'face', 'eye', 'nose', 'mouth', 'ear', 'hair',
    'neck', 'body', 'arm', 'hand', 'leg', 'foot', 'button', 'pocket', 'sneakers', 'man_shoes', 'woman_shoes'
]

# 6. 추론 결과를 이용해 바운딩 박스와 라벨 시각화 및 터미널에 정보 출력
for result in results:
    boxes = result.boxes  # 탐지된 모든 객체의 바운딩 박스 정보
    for box in boxes:
        # 좌표, 신뢰도, 클래스 정보 추출(tensor -> numpy로 변환)
        xyxy = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
        conf = box.conf.cpu().numpy()[0]    # 신뢰도
        cls_id = int(box.cls.cpu().numpy()[0])  # 클래스 번호
        
        # 좌표를 정수형으로 변환(OpenCV 그리기 위해)
        x1, y1, x2, y2 = map(int, xyxy)
        
        # 너비와 높이 계산
        width = x2 - x1
        height = y2 - y1
        
        # 클래스 이름 결정 (클래스 번호가 class_names 범위 내에 있을 때)
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        
        # 터미널에 정보 출력
        print(f"클래스: {label}, 신뢰도: {conf:.2f}")
        print(f"좌표: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"크기: w={width}, h={height}\n")
        
        # 바운딩 박스 그리기 (녹색 테두리, 두께 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        
        # 첫 번째 줄 텍스트: 클래스 이름, 신뢰도
        text1 = f"{label} {conf:.2f}"
        # 두 번째 줄 텍스트: 좌표, 크기 정보
        text2 = f"x1={x1}, y1={y1}, w={width}, h={height}"
        
        # 텍스트 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        
        # 텍스트 크기 계산
        (text1_width, text1_height), base1 = cv2.getTextSize(text1, font, scale, thickness)
        (text2_width, text2_height), base2 = cv2.getTextSize(text2, font, scale, thickness)
        
        # 전체 텍스트 박스 높이와 너비 계산
        box_height = text1_height + text2_height + base1 + 5
        box_width = max(text1_width, text2_width)
        
        # 텍스트 배경 박스 그리기
        cv2.rectangle(image, 
                      (x1, y1 - box_height), 
                      (x1 + box_width, y1), 
                      (0, 255, 0), 
                      -1)
        
        # 첫 번째 줄 텍스트 그리기 (하얀색 글씨)
        cv2.putText(image, text1, 
                    (x1, y1 - text2_height - base1), 
                    font, scale, (255, 255, 255), thickness)
        
        # 두 번째 줄 텍스트 그리기
        cv2.putText(image, text2, 
                    (x1, y1 - base1), 
                    font, scale, (255, 255, 255), thickness)

# 7. 결과 이미지 출력
cv2.imshow("Detection Results", image)
cv2.waitKey(0)  # 키 입력 대기, 아무 키나 누르면 창이 닫힘
cv2.destroyAllWindows()

# 또는 결과 이미지를 파일로 저장
cv2.imwrite("detection_result_02.jpg", image)

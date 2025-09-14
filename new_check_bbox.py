import json
import cv2
import os
import numpy as np

# --- 1. 클래스 이름 및 색상 팔레트 정의 --------------------------------
names = [
    'person', 'head', 'face', 'eye', 'nose', 'mouth', 'ear', 'hair', 'neck', 'upperbody',
    'arm', 'hand', 'headwear', 'glasses', 'eyebrow', 'suyeom', 'openmouth', 'muffler', 'necktie',
    'ribbon', 'earmuff', 'earring', 'neckring', 'jangsik', 'headjangsik', 'jewer', 'tabaco'
]
nc = len(names)

# 랜덤하되 재현 가능한 색상을 만듭니다
np.random.seed(42)
colors = np.random.randint(0, 255, size=(nc, 3), dtype=np.uint8)

# --- 2. 파일 경로 설정 --------------------------------
file_path = r"C:\capstone\data\train\composited_final\labels\composited_000187.txt"
img_path  = r"C:\capstone\data\train\composited_final\images\composited_000187.jpg"

# --- 3. 이미지 로드 --------------------------------
try:
    img_array = np.fromfile(img_path, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        raise Exception("이미지 디코딩 실패")
    img_h, img_w = image.shape[:2]
except Exception as e:
    print(f"이미지 로드 실패: {img_path}\n오류: {e}")
    exit()

# --- 4. 레이블 파일 처리 --------------------------------
_, ext = os.path.splitext(file_path)
if ext.lower() == '.txt':
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_c = float(parts[1]) * img_w
            y_c = float(parts[2]) * img_h
            w   = float(parts[3]) * img_w
            h   = float(parts[4]) * img_h

            # 좌상단/우하단 계산
            x1 = int(x_c - w/2);  y1 = int(y_c - h/2)
            x2 = int(x_c + w/2);  y2 = int(y_c + h/2)

            # 이 클래스 고유 색과 이름
            color = tuple(int(c) for c in colors[class_id])
            label = names[class_id]

            # 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 텍스트 배경을 반투명으로 그리고, 텍스트 표시
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    except Exception as e:
        print(f"YOLO 형식 파일 처리 오류: {e}")
        exit()
else:
    # (JSON 처리 부분은 생략—필요하다면 유사하게 색상/이름 적용)
    ...

# --- 5. 결과 표시 --------------------------------
cv2.imshow("Image with Colored Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

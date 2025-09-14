import json
import cv2
import os
import numpy as np

# 파일 경로 설정
file_path = r"C:\capstone\data\train\composited_final\labels\composited_000281.txt"
img_path = r"C:\capstone\data\train\composited_final\images\composited_000281.jpg"

# 이미지 로드
try:
    img_array = np.fromfile(img_path, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        raise Exception("이미지 디코딩 실패")
    
    # 이미지 크기 확인 (YOLO 좌표 변환에 필요)
    img_height, img_width = image.shape[:2]
    
    cv2.imshow("Loaded Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"이미지 로드 실패: {img_path}\n오류: {e}")
    exit()

# 파일 확장자 확인
_, ext = os.path.splitext(file_path)

if ext.lower() == '.txt':
    # YOLO 형식 처리
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # 각 객체의 바운딩 박스에 대해 처리
        for line in lines:
            # YOLO 형식: class x_center y_center width height (모두 0~1 사이 정규화된 값)
            parts = line.strip().split()
            if len(parts) >= 5:  # 최소 5개 값(클래스 + 좌표 4개)이 있는지 확인
                class_id = int(parts[0])
                # 정규화된 좌표를 실제 이미지 크기로 변환
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # 좌상단 좌표 계산
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                # 우하단 좌표 계산
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # 이미지에 바운딩 박스 그리기 (녹색 테두리, 두께 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 선택적: 클래스 ID를 바운딩 박스 옆에 표시
                cv2.putText(image, f"Class: {class_id}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except Exception as e:
        print(f"YOLO 형식 파일 처리 오류: {e}")
        exit()

else:  # JSON 형식 처리 (기존 코드)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # abstract_image 섹션에서 바운딩 박스 정보 추출
        abs_img = data.get("abstract_image", {})
        abs_bbox_str = abs_img.get("abs_bbox", None)
        if abs_bbox_str is None:
            print("바운딩 박스(abs_bbox) 정보가 JSON에 없습니다.")
            exit()
        try:
            bbox = json.loads(abs_bbox_str)
            # bbox 값: [x, y, w, h]
            x, y, w, h = bbox
            
            # 바운딩 박스 좌표 계산 (좌측 상단, 우측 하단) 및 정수 변환
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # 이미지에 바운딩 박스 그리기 (녹색 테두리, 두께 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as e:
            print(f"바운딩 박스 파싱 오류: {e}")
            exit()
    except Exception as e:
        print(f"JSON 파일 처리 오류: {e}")
        exit()

# 이미지 표시
cv2.imshow("Image with Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
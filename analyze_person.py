from ultralytics import YOLO
import cv2

# 모델 불러오기
model = YOLO("C:/capstone/yolo/runs/detect/train7/weights/best.pt")

# 이미지 경로
image_path = "C:/capstone/yolo/0222test02.jpg"

# 추론 진행
results = model.predict(source=image_path, conf=0.35)

# 이미지 읽기
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]
image_area = image_width * image_height

# 클래스 이름
class_names = [
    'person_all', 'head', 'face', 'eye', 'nose', 'mouth', 'ear', 'hair',
    'neck', 'body', 'arm', 'hand', 'leg', 'foot', 'button', 'pocket', 'sneakers', 'man_shoes', 'woman_shoes'
]

# 탐지 결과 저장 (각 부위별 리스트)
detections = {name: [] for name in class_names}

# 탐지 결과 처리
for result in results:
    boxes = result.boxes
    for box in boxes:
        xyxy = box.xyxy.cpu().numpy()[0]
        conf = box.conf.cpu().numpy()[0]
        cls_id = int(box.cls.cpu().numpy()[0])
        label = class_names[cls_id]
        x1, y1, x2, y2 = map(int, xyxy)
        width, height = x2 - x1, y2 - y1
        detections[label].append([x1, y1, x2, y2, width, height, conf])

# 'person_all' 크기 분석 및 치우침 판단
if 'person_all' in detections and len(detections['person_all']) > 0:
    person_box = detections['person_all'][0]
    person_area = person_box[4] * person_box[5]
    person_size = "큼" if person_area > (2/3) * image_area else "작음"
    
    # 치우침 판단
    person_center_x = (person_box[0] + person_box[2]) / 2
    image_center_x = image_width / 2
    tolerance = image_width * 0.05  # 허용 오차: 이미지 너비의 5%
    
    if person_center_x < image_center_x - tolerance:
        bias = "왼쪽으로 치우침"
    elif person_center_x > image_center_x + tolerance:
        bias = "오른쪽으로 치우침"
    else:
        bias = "중앙"
    
    detections['person_all'][0].append(person_size)
    detections['person_all'][0].append(bias)
else:
    person_size = "not 검출됨"
    bias = "not 검출됨"

# head 기준 기대하는 부위 크기 비율
if 'head' in detections and len(detections['head']) > 0:
    head_box = detections['head'][0]
    head_w, head_h = head_box[4], head_box[5]
    expected = {
        'eye':   {'w': 0.233 * head_w, 'h': 0.143 * head_h},
        'nose':  {'w': 0.067 * head_w, 'h': 0.143 * head_h},
        'mouth': {'w': 0.333 * head_w, 'h': 0.029 * head_h},
        'ear':   {'w': 0.1   * head_w, 'h': 0.229 * head_h},
        'hair':  {'w': 1.333 * head_w, 'h': 0.229 * head_h},
        'neck':  {'w': 0.067 * head_w, 'h': 0.286 * head_h},
        'face':  {'w': 0.6   * head_w, 'h': 0.7   * head_h},
        'body':  {'w': 1.5   * head_w, 'h': 0.3   * head_h},
        'arm':   {'w': 0.25  * head_w, 'h': 1.2   * head_h},
        'hand':  {'w': 0.15  * head_w, 'h': 0.15  * head_h},
    }
else:
    expected = {}

# 개별 크기 분류 함수: detected와 기대값을 비교하여 "매우 작음", "작음", "평균", "큼", "매우 큼" 반환
def classify_size(detected, expected):
    if detected < expected * 0.6:
        return "매우 작음", -2
    elif detected < expected * 0.8:
        return "작음", -1
    elif detected <= expected * 1.2:
        return "평균", 0
    elif detected <= expected * 1.4:
        return "큼", 1
    else:
        return "매우 큼", 2

# width와 height 상태를 합산하여 종합 상태를 반환하는 함수
def combine_status(score_w, score_h):
    total = score_w + score_h
    if total <= -3:
        return "매우 작음"
    elif total == -2:
        return "작음"
    elif total <= 1:
        return "보통"
    elif total == 2:
        return "큼"
    else:
        return "매우 큼"

# 부위별 크기 분류 (expected에 정의된 부위에 대해)
for part in expected.keys():
    if part in detections:
        for box in detections[part]:
            w, h = box[4], box[5]
            status_w, score_w = classify_size(w, expected[part]['w'])
            status_h, score_h = classify_size(h, expected[part]['h'])
            combined = combine_status(score_w, score_h)
            box.append(combined)

# 목의 길이와 두께 분석 (neck 부위)
if 'neck' in detections:
    for neck_box in detections['neck']:
        neck_w, neck_h = neck_box[4], neck_box[5]
        status_w, score_w = classify_size(neck_w, expected['neck']['w'])
        status_h, score_h = classify_size(neck_h, expected['neck']['h'])
        neck_status = combine_status(score_w, score_h)
        neck_box.append(neck_status)

# 시각화
for label in detections:
    for box in detections[label]:
        x1, y1, x2, y2, width, height, conf = box[:7]
        if label == 'person_all':
            status = f"{box[7]}, {box[8]}" if len(box) > 8 else ""
        else:
            status = box[7] if len(box) > 7 else ""
        text1 = f"{label} {conf:.2f} {status}" if status else f"{label} {conf:.2f}"
        text2 = f"x1={x1}, y1={y1}, w={width}, h={height}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (text1_width, text1_height), _ = cv2.getTextSize(text1, font, scale, thickness)
        (text2_width, text2_height), _ = cv2.getTextSize(text2, font, scale, thickness)
        box_height = text1_height + text2_height + 5
        box_width = max(text1_width, text2_width)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(image, (x1, y1 - box_height), (x1 + box_width, y1), (0, 255, 0), -1)
        cv2.putText(image, text1, (x1, y1 - text2_height - 5), font, scale, (255, 255, 255), thickness)
        cv2.putText(image, text2, (x1, y1 - 5), font, scale, (255, 255, 255), thickness)

# 이미지 출력 및 저장
cv2.imshow("Detection Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("detection_result_07.jpg", image)

# 분석 결과 출력
print("### 신체 부위 분석 결과 ###")
print(f"Person_all: {person_size}, {bias}")
print("Head: 존재" if 'head' in detections and len(detections['head']) > 0 else "Head: not 검출됨")

for part in ['eye', 'ear', 'arm', 'hand']:
    if part in detections and len(detections[part]) > 0:
        num = len(detections[part])
        statuses = [box[7] for box in detections[part] if len(box) > 7]
        if num == 2:
            status = f"둘 존재 - {statuses}"
        elif num == 1:
            status = f"하나 생략됨 - {statuses}"
        else:
            status = "둘 생략됨"
        print(f"{part.capitalize()}s: {status}")
    else:
        print(f"{part.capitalize()}s: 둘 생략됨")

for part in ['nose', 'mouth', 'hair', 'neck', 'face', 'body']:
    if part in detections and len(detections[part]) > 0:
        status = detections[part][0][7] if len(detections[part][0]) > 7 else "검출됨"
        print(f"{part.capitalize()}: {status}")
    else:
        print(f"{part.capitalize()}: 생략됨")
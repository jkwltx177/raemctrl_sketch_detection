from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO("C:/capstone/yolo/runs/detect/train7/weights/best.pt")

# 이미지 경로 지정
image_path = "C:/capstone/yolo/0222test02.jpg"

# 신뢰도 임계값을 설정하여 추론 수행
results = model.predict(source=image_path, conf=0.35)

# 시각화를 위해 이미지 읽기
image = cv2.imread(image_path)

# 클래스 이름 정의
class_names = [
    'person_all', 'head', 'face', 'eye', 'nose', 'mouth', 'ear', 'hair',
    'neck', 'body', 'arm', 'hand', 'leg', 'foot', 'button', 'pocket', 'sneakers', 'man_shoes', 'woman_shoes'
]

# 검출 결과를 저장할 딕셔너리 초기화
detections = {name: [] for name in class_names}

# 검출 결과 처리
for result in results:
    boxes = result.boxes
    for box in boxes:
        xyxy = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
        conf = box.conf.cpu().numpy()[0]  # 신뢰도 점수
        cls_id = int(box.cls.cpu().numpy()[0])  # 클래스 ID
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        x1, y1, x2, y2 = map(int, xyxy)
        width = x2 - x1
        height = y2 - y1
        detections[label].append((width, height, conf))

# 머리가 검출되었는지 확인하여 기준으로 사용
if 'head' in detections and len(detections['head']) > 0:
    head_w, head_h, _ = detections['head'][0]  # 머리 하나만 있다고 가정

    # 머리 크기를 기준으로 예상 비율 정의
    expected = {
        'eye': {'w': 0.233 * head_w, 'h': 0.143 * head_h},
        'nose': {'w': 0.067 * head_w, 'h': 0.143 * head_h},
        'mouth': {'w': 0.333 * head_w, 'h': 0.029 * head_h},
        'ear': {'w': 0.1 * head_w, 'h': 0.229 * head_h},
        'hair': {'w': 1.333 * head_w, 'h': 0.229 * head_h},
        'neck': {'w': 0.067 * head_w, 'h': 0.286 * head_h},
    }

    # 크기 분류 함수 정의
    def classify_size(detected, expected, tolerance=0.2):
        if detected > expected * (1 + tolerance):
            return "large"
        elif detected < expected * (1 - tolerance):
            return "small"
        else:
            return "average"

    # 각 부위 분석
    analysis = {'head': "present"}

    # 쌍으로 된 부위: 눈과 귀
    for part in ['eye', 'ear']:
        if part in detections:
            num_detected = len(detections[part])
            if num_detected == 0:
                analysis[part + 's'] = "both omitted"
            elif num_detected == 1:
                w, h, _ = detections[part][0]
                size_w = classify_size(w, expected[part]['w'])
                size_h = classify_size(h, expected[part]['h'])
                analysis[part + 's'] = f"one omitted, detected {part}: {size_w} width, {size_h} height"
            else:  # 최대 2개 검출 가정
                sizes = []
                for w, h, _ in detections[part][:2]:
                    size_w = classify_size(w, expected[part]['w'])
                    size_h = classify_size(h, expected[part]['h'])
                    sizes.append(f"{size_w} width, {size_h} height")
                analysis[part + 's'] = f"both present: {sizes[0]} and {sizes[1]}"

    # 단일 부위: 코, 입, 머리카락
    for part in ['nose', 'mouth', 'hair']:
        if part in detections and len(detections[part]) > 0:
            w, h, _ = detections[part][0]
            size_w = classify_size(w, expected[part]['w'])
            size_h = classify_size(h, expected[part]['h'])
            analysis[part] = f"{size_w} width, {size_h} height"
        else:
            analysis[part] = "omitted"

    # 목 분석
    if 'neck' in detections and len(detections['neck']) > 0:
        neck_w, neck_h, _ = detections['neck'][0]
        # 길이(높이) 분류
        if neck_h > expected['neck']['h'] * 1.2:
            length = "long"
        elif neck_h < expected['neck']['h'] * 0.8:
            length = "short"
        else:
            length = "average"
        # 두께(너비) 분류
        if neck_w > expected['neck']['w'] * 1.2:
            thickness = "thick"
        elif neck_w < expected['neck']['w'] * 0.8:
            thickness = "thin"
        else:
            thickness = "average"
        analysis['neck'] = f"{length}, {thickness}"
    else:
        analysis['neck'] = "omitted"

else:
    analysis = {'head': "omitted"}
    # 머리가 없으면 다른 부위의 크기 분류 불가, 검출 여부만 보고
    for part in ['eye', 'nose', 'mouth', 'ear', 'hair', 'neck']:
        if part in detections and len(detections[part]) > 0:
            analysis[part] = "detected but cannot classify size (head missing)"
        else:
            analysis[part] = "omitted"

# 분석 결과 출력
print("### 신체 부위 분석 결과 ###")
for part, status in analysis.items():
    print(f"{part.capitalize()}: {status}")
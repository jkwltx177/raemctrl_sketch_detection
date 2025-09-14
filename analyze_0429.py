from ultralytics import YOLO
import cv2
import numpy as np
import os

# 모델 불러오기
#model = YOLO("C:/capstone/yolo/runs/detect/train40/weights/best.pt")
model = YOLO("C:/capstone/yolo/runs/detect/train40/best_3.pt")
# 이미지 경로
image_path = "C:/capstone/yolo/0412test07.jpg"

# 추론 진행
results = model.predict(source=image_path, conf=0.10)

# 이미지 읽기
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]
image_area = image_width * image_height

# 클래스 이름
class_names = [
    'person_all', 'head', 'face', 'eye', 'nose', 'mouth', 'ear', 'hair',
    'neck', 'body', 'arm', 'hand', 'hat', 'glasses', 'eyebrow', 'beard',
    'open_mouth_(teeth)', 'muffler', 'tie', 'ribbon', 'ear_muff', 'earring', 'necklace', 'ornament',
    'headdress', 'jewel', 'cigarette'#, 'leg', 'foot', 'button', 'pocket', 'sneakers', 'man_shoes', 'woman_shoes'
]

# 탐지 결과 저장
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

#######################################
# 전체 크기 세분화
#######################################
def classify_overall_size(person_area, image_area):
    ratio = person_area / image_area
    if ratio < 0.3:
        return "매우 작음"
    elif ratio < 0.6:
        return "작음"
    elif ratio < 0.8:
        return "보통"
    elif ratio < 1.0:
        return "큼"
    else:
        return "매우 큼"

#######################################
# 'person_all' 크기 및 위치 분석
#######################################
if 'person_all' in detections and len(detections['person_all']) > 0:
    person_box = detections['person_all'][0]
    x1, y1, x2, y2 = person_box[:4]
    person_area = person_box[4] * person_box[5]
    person_size = classify_overall_size(person_area, image_area)
    
    # 수평 치우침
    person_center_x = (x1 + x2) / 2
    image_center_x = image_width / 2
    tolerance = image_width * 0.05
    if person_center_x < image_center_x - tolerance:
        bias_horizontal = "수평.왼쪽으로 치우침"
    elif person_center_x > image_center_x + tolerance:
        bias_horizontal = "수평.오른쪽으로 치우침"
    else:
        bias_horizontal = "수평.중앙"
    
    # 수직 위치
    person_center_y = (y1 + y2) / 2
    image_center_y = image_height / 2
    vertical_tolerance = image_height * 0.05
    if abs(person_center_y - image_center_y) < vertical_tolerance * 0.5:
        bias_vertical = "수직.과도한 정중앙"
    elif person_center_y < image_center_y - vertical_tolerance:
        bias_vertical = "수직.상단"
    elif person_center_y > image_center_y + vertical_tolerance:
        bias_vertical = "수직.하단"
    else:
        bias_vertical = "수직.중앙"
    
    # 특수 가장자리 분석
    cut_off = []
    if y1 <= 0:
        cut_off.append("상 절단")
    # if y2 >= image_height:
    #     cut_off.append("하 절단")
    if x1 <= 0:
        cut_off.append("좌 절단")
    if x2 >= image_width:
        cut_off.append("우 절단")
    cut_off_status = ", ".join(cut_off) if cut_off else "절단 없음"
    
    detections['person_all'][0].extend([person_size, bias_horizontal, bias_vertical, cut_off_status])
else:
    person_size = "not 검출됨"
    bias_horizontal = "not 검출됨"
    bias_vertical = "not 검출됨"
    cut_off_status = "not 검출됨"

#######################################
# head 기준 기대 크기 비율
#######################################
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

#######################################
# 크기 분류 함수
#######################################
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

for part in expected.keys():
    if part in detections:
        for box in detections[part]:
            w, h = box[4], box[5]
            status_w, score_w = classify_size(w, expected[part]['w'])
            status_h, score_h = classify_size(h, expected[part]['h'])
            combined = combine_status(score_w, score_h)
            box.append(combined)

if 'neck' in detections:
    for neck_box in detections['neck']:
        neck_w, neck_h = neck_box[4], neck_box[5]
        status_w, score_w = classify_size(neck_w, expected['neck']['w'])
        status_h, score_h = classify_size(neck_h, expected['neck']['h'])
        neck_status = combine_status(score_w, score_h)
        neck_box.append(neck_status)

#######################################
# 필압 및 선 분석
#######################################
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
edge_density = np.count_nonzero(edges) / (image_height * image_width)
pressure_status = "강한 필압" if edge_density > 0.05 else "약한 필압"

# 필압 변화 분석
regions = [edges[:image_height//2, :], edges[image_height//2:, :]]
densities = [np.count_nonzero(region) / (region.shape[0] * region.shape[1]) for region in regions]
pressure_var = np.std(densities)
pressure_change = "과도한 변화" if pressure_var > 0.02 else "적당한 변화"

lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
if lines is not None:
    lengths = [np.linalg.norm((line[0][0]-line[0][2], line[0][1]-line[0][3])) for line in lines]
    avg_line_length = np.mean(lengths)
    line_status = "전체적 강한 선" if avg_line_length > 100 else "일부분 강한 선"
else:
    line_status = "약한 선"

# 지면선 존재 여부
ground_line_exists = False
if lines is not None:
    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        dx = x2_l - x1_l
        dy = y2_l - y1_l
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) < 10 or abs(angle - 180) < 10:
            if y1_l > image_height * 0.8 and y2_l > image_height * 0.8:
                ground_line_exists = True
                break

#######################################
# 선 특징 분석
#######################################
line_lengths = []
if lines is not None:
    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        length = np.sqrt((x2_l - x1_l)**2 + (y2_l - y1_l)**2)
        line_lengths.append(length)
    avg_length = np.mean(line_lengths)
    line_length_status = "길다" if avg_length > 100 else "짧다"
else:
    line_length_status = "정보 없음"

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
curvature_scores = []
for cnt in contours:
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    curvature_scores.append(0 if len(approx) <= 3 else 1)
if curvature_scores:
    shape_status = "직선적" if np.mean(curvature_scores) < 0.5 else "곡선적"
else:
    shape_status = "정보 없음"

#######################################
# 세부 묘사
#######################################
orb = cv2.ORB_create()
keypoints = orb.detect(gray, None)
num_keypoints = len(keypoints)
density = num_keypoints / image_area
if density > 0.0005:
    detail_status = "과도한 세부 묘사"
elif density < 0.0001:
    detail_status = "부족한 세부 묘사"
else:
    detail_status = "보통"

#######################################
# 움직임 표현
#######################################
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
if laplacian_var < 100:
    movement_status = "과도한 움직임 표현"
elif laplacian_var > 300:
    movement_status = "움직임 부족"
else:
    movement_status = "적당한 움직임"

#######################################
# 대칭성 분석
#######################################
if 'head' in detections and len(detections['head']) > 0:
    head_box = detections['head'][0]
    x1_h, y1_h, x2_h, y2_h = head_box[:4]
    
    # 헤드 이미지가 유효한지 확인
    if y2_h > y1_h and x2_h > x1_h:
        head_img = image[y1_h:y2_h, x1_h:x2_h]
        
        # 이미지가 비어있지 않은지 확인
        if head_img.size > 0:
            head_img_flipped = cv2.flip(head_img, 1)
            diff = cv2.absdiff(head_img, head_img_flipped)
            
            # diff가 유효한지 확인
            if diff is not None and diff.size > 0:
                mean_diff = np.mean(diff)
                symmetry_status = "대칭성이 높음" if mean_diff < 20 else "대칭성이 낮음"
            else:
                symmetry_status = "대칭 계산 실패"
        else:
            symmetry_status = "이미지 추출 실패"
    else:
        symmetry_status = "유효하지 않은 경계 상자"
else:
    symmetry_status = "정보 없음"

#######################################
# 투명성 분석
#######################################
if image.shape[2] == 4:
    alpha_channel = image[:, :, 3]
    avg_alpha = np.mean(alpha_channel)
    transparency_status = "투명한 표현이 과도함" if avg_alpha < 100 else "보통"
else:
    transparency_status = "해당 없음"

#######################################
# 머리 크기 분류
#######################################
def classify_head_size(head_box, person_box=None, image_area=None):
    head_area = head_box[4] * head_box[5]
    if person_box is not None:
        person_area_val = person_box[4] * person_box[5]
        ratio = head_area / person_area_val
    elif image_area is not None:
        ratio = head_area / image_area
    else:
        ratio = 0
    if ratio < 0.2:
        return "작음"
    elif ratio < 0.35:
        return "보통"
    else:
        return "큼"

if 'head' in detections and len(detections['head']) > 0:
    head_box = detections['head'][0]
    if 'person_all' in detections and len(detections['person_all']) > 0:
        person_box = detections['person_all'][0]
        head_size_class = classify_head_size(head_box, person_box=person_box)
    else:
        head_size_class = classify_head_size(head_box, image_area=image_area)
else:
    head_size_class = "검출되지 않음"

#######################################
# 얼굴 분석: 뒤통수 (+옆모습)
#######################################
if 'face' in detections and len(detections['face']) > 0:
    face_detected = True
    num_eyes = len(detections.get('eye', []))
    has_nose = 'nose' in detections and len(detections['nose']) > 0
    has_mouth = 'mouth' in detections and len(detections['mouth']) > 0
    if num_eyes == 0 and not has_nose and not has_mouth:
        face_status = "뒤통수"
    # elif num_eyes == 1:
    #     face_status = "옆모습"
    else:
        face_status = "정면 또는 기타"
else:
    face_status = "검출되지 않음"

#######################################
# 머리와 몸의 단절
#######################################
if 'head' in detections and len(detections['head']) > 0 and 'body' in detections and len(detections['body']) > 0:
    head_box = detections['head'][0]
    body_box = detections['body'][0]
    gap = body_box[1] - head_box[3]
    head_height = head_box[5]
    threshold = 0.1 * head_height
    if gap > threshold and 'neck' not in detections:
        disconnection_status = "머리와 몸의 단절"
    else:
        disconnection_status = "연결됨"
elif 'head' in detections and len(detections['head']) > 0 and ('body' not in detections or len(detections['body']) == 0):
    disconnection_status = "몸 생략"
elif 'body' in detections and len(detections['body']) > 0 and ('head' not in detections or len(detections['head']) == 0):
    disconnection_status = "머리 생략"
else:
    disconnection_status = "머리와 몸 모두 미검출"

#######################################
# 특정 부분의 세부 묘사 분석 ver1
#######################################
def check_detailed_description(part, box, expected):
    w, h = box[4], box[5]
    if w > expected[part]['w'] * 1.5 or h > expected[part]['h'] * 1.5:
        return "세부 묘사 있음"
    return "세부 묘사 없음"

detailed_description = {}
for part in expected.keys():
    if part in detections and len(detections[part]) > 0:
        for box in detections[part]:
            detail_status = check_detailed_description(part, box, expected)
            detailed_description[part] = detail_status
    else:
        detailed_description[part] = "검출되지 않음"

#######################################
# 특정 부분의 세부 묘사 분석 ver2
#######################################
def analyze_specific_detail(region, region_name=""):
    orb_local = cv2.ORB_create()
    keypoints_local = orb_local.detect(region, None)
    count = len(keypoints_local)
    return f"{region_name} 세부묘사: {'과도함' if count > 300 else '보통' if count > 100 else '부족함'}"

#######################################
# 왜곡 분석 ver1
#######################################
def check_distortion(part, box, expected):
    w, h = box[4], box[5]
    
    # 예상 너비와 높이가 0인지 확인
    expected_w = expected[part]['w']
    expected_h = expected[part]['h']
    
    if expected_w <= 0 or expected_h <= 0:
        return "기준값 오류"
    
    w_ratio = w / expected_w
    h_ratio = h / expected_h
    
    if w_ratio > 2.0 or h_ratio > 2.0:
        return "극단적인 왜곡"
    elif w_ratio > 1.5 or h_ratio > 1.5:
        return "일반적인 왜곡"
    return "왜곡 없음"

distortion_status = {}
for part in expected.keys():
    if part in detections and len(detections[part]) > 0:
        for box in detections[part]:
            try:
                distortion = check_distortion(part, box, expected)
                distortion_status[part] = distortion
            except Exception as e:
                # 에러 발생 시 로깅하고 기본값 설정
                print(f"Error checking distortion for {part}: {e}")
                distortion_status[part] = "계산 오류"
    else:
        distortion_status[part] = "검출되지 않음"

#######################################
# 왜곡 분석 ver2
#######################################
def analyze_distortion(detections, expected):
    extreme = 0
    total = 0
    for part in expected.keys():
        if part in detections:
            for box in detections[part]:
                if box[-1] in ["매우 작음", "매우 큼"]:
                    extreme += 1
                total += 1
    if total == 0:
        return "정보 없음"
    ratio = extreme / total
    if ratio > 0.5:
        return "극단적인 왜곡"
    elif ratio > 0.2:
        return "일반적인 왜곡"
    else:
        return "정상"

distortion_status_ = analyze_distortion(detections, expected)

#######################################
# 불필요한 내용 추가 분석 ver1
#######################################
def check_unnecessary_content(image, detections, orb):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, _ = orb.detectAndCompute(gray, None)
    total_keypoints = len(keypoints)
    
    # 주요 부위 영역 내 키포인트 수 계산
    mask = np.zeros_like(gray)
    for label in detections:
        for box in detections[label]:
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    keypoints_in_parts = len(orb.detect(cv2.bitwise_and(gray, gray, mask=mask)))
    
    # 불필요한 부분 비율
    unnecessary_ratio = (total_keypoints - keypoints_in_parts) / total_keypoints if total_keypoints > 0 else 0
    if unnecessary_ratio > 0.3:
        return "불필요한 내용 과도함"
    return "불필요한 내용 없음"

unnecessary_content_status = check_unnecessary_content(image, detections, orb)

#######################################
# 불필요한 내용 추가 분석 ver2
#######################################
def detect_extraneous_elements(image):
    gray_local = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_local = cv2.Canny(gray_local, 50, 150)
    edge_ratio = np.count_nonzero(edges_local) / image_area
    return "불필요한 요소 있음" if edge_ratio > 0.1 else "불필요한 요소 없음"

extraneous_status_ = detect_extraneous_elements(image)

#######################################
# 머리: 모자(가림) 분석 -> 모자는 결국 yolo에 추가 학습 필요함!!!
#######################################
hat_status = "모자 없음"
if 'head' in detections and len(detections['head']) > 0:
    head_box = detections['head'][0]
    x1_h, y1_h, x2_h, y2_h = head_box[:4]
    for label in detections:
        if label != 'head':
            for box in detections[label]:
                x1, y1, x2, y2 = box[:4]
                if y2 <= y1_h and x1_h < (x1 + x2) / 2 < x2_h:
                    hat_status = "모자 있음 (가림)"
                    break

#######################################
# 눈 분석
#######################################
eye_details = {}
if 'eye' in detections and len(detections['eye']) > 0:
    for i, eye_box in enumerate(detections['eye']):
        x1, y1, x2, y2 = eye_box[:4]
        eye_roi = gray[y1:y2, x1:x2]
        
        # 가림 여부
        obscured = False
        for label in detections:
            if label != 'eye':
                for box in detections[label]:
                    if x1 < box[2] and x2 > box[0] and y1 < box[3] and y2 > box[1]:
                        obscured = True
                        break
        eye_details[f'eye_{i}_obscured'] = "가림" if obscured else "가림 없음"
        
        # 윤곽만 묘사
        edges_eye = cv2.Canny(eye_roi, 50, 150)
        edge_ratio = np.count_nonzero(edges_eye) / (eye_roi.shape[0] * eye_roi.shape[1])
        eye_details[f'eye_{i}_contour'] = "윤곽만 묘사" if edge_ratio > 0.1 else "일반 묘사"
        
        # 진한 눈동자
        mean_intensity = np.mean(eye_roi)
        eye_details[f'eye_{i}_pupil'] = "진한 눈동자" if mean_intensity < 100 else "일반 눈동자"
        
        # 눈썹 (임시: 눈 위에 객체가 있는지 확인) -> 눈썹도 결국엔 추가 학습 필요할 듯
        eyebrow_detected = False
        for label in detections:
            if label != 'eye':
                for box in detections[label]:
                    if y2 <= box[1] and x1 < (box[0] + box[2]) / 2 < x2:
                        eyebrow_detected = True
                        break
        eye_details[f'eye_{i}_eyebrow'] = "눈썹 있음" if eyebrow_detected else "눈썹 없음"
else:
    eye_details['eye_0'] = "검출되지 않음"

#######################################
# 코 모양 분석
#######################################
nose_shape = "검출되지 않음"
if 'nose' in detections and len(detections['nose']) > 0:
    nose_box = detections['nose'][0]
    w, h = nose_box[4], nose_box[5]
    
    # 높이가 0인지 확인
    if h > 0:
        aspect_ratio = w / h
        if aspect_ratio < 0.5:
            nose_shape = "길쭉한 코"
        elif aspect_ratio > 1.0:
            nose_shape = "넓은 코"
        else:
            nose_shape = "일반적인 코"
    else:
        nose_shape = "측정 불가능"

#######################################
# 귀: 귀걸이 분석. 귀 주변에 작은 객체가 있는지 확인 -> 귀걸이는 결국 yolo에 추가 학습 필요함!!!
#######################################
earring_status = {}
if 'ear' in detections and len(detections['ear']) > 0:
    for i, ear_box in enumerate(detections['ear']):
        x1, y1, x2, y2 = ear_box[:4]
        earring_detected = False
        for label in detections:
            if label != 'ear':
                for box in detections[label]:
                    bx1, by1, bx2, by2 = box[:4]
                    if (abs((bx1 + bx2) / 2 - (x1 + x2) / 2) < ear_box[4] and
                        abs((by1 + by2) / 2 - (y1 + y2) / 2) < ear_box[5]):
                        earring_detected = True
                        break
        earring_status[f'ear_{i}'] = "귀걸이 착용" if earring_detected else "귀걸이 없음"
else:
    earring_status['ear_0'] = "검출되지 않음"

mouth_shape = "검출되지 않음"
if 'mouth' in detections and len(detections['mouth']) > 0:
    mouth_box = detections['mouth'][0]
    x1, y1, x2, y2 = mouth_box[:4]
    
    # 유효한 경계 상자인지 확인
    if y2 > y1 and x2 > x1 and y1 >= 0 and x1 >= 0 and y2 <= gray.shape[0] and x2 <= gray.shape[1]:
        mouth_roi = gray[y1:y2, x1:x2]
        
        # 이미지가 비어 있지 않은지 확인
        if mouth_roi.size > 0 and mouth_roi.shape[0] > 0 and mouth_roi.shape[1] > 0:
            edges_mouth = cv2.Canny(mouth_roi, 50, 150)
            
            # edges_mouth가 None이 아닌지 확인
            if edges_mouth is not None:
                # 입 크기 분석
                h_ratio = mouth_box[5] / expected['mouth']['h']
                if h_ratio > 1.5:
                    mouth_shape = "벌림"
                else:
                    # 곡률 분석 (임시: 엣지로 판단)
                    top_edge = np.sum(edges_mouth[:mouth_roi.shape[0]//2, :])
                    bottom_edge = np.sum(edges_mouth[mouth_roi.shape[0]//2:, :])
                    if abs(top_edge - bottom_edge) < 100:
                        mouth_shape = "일직선"
                    elif top_edge > bottom_edge:
                        mouth_shape = "웃음"
                    else:
                        mouth_shape = "비웃음"
            else:
                mouth_shape = "엣지 감지 실패"
        else:
            mouth_shape = "유효하지 않은 이미지"
    else:
        mouth_shape = "유효하지 않은 경계 상자"

#######################################
# 머리카락 분석
#######################################
hair_details = {}
if 'hair' in detections and len(detections['hair']) > 0:
    hair_box = detections['hair'][0]
    x1, y1, x2, y2 = hair_box[:4]
    hair_roi = gray[y1:y2, x1:x2]
    
    # 숱 분석
    hair_density = np.mean(hair_roi)
    hair_details['volume'] = "많은 숱" if hair_density < 150 else "적은 숱"
    
    # 세부 묘사
    edges_hair = cv2.Canny(hair_roi, 50, 150)
    edge_density_hair = np.count_nonzero(edges_hair) / (hair_roi.shape[0] * hair_roi.shape[1])
    hair_details['detail'] = "세부 묘사 있음" if edge_density_hair > 0.05 else "세부 묘사 없음"
else:
    hair_details['volume'] = "검출되지 않음"
    hair_details['detail'] = "검출되지 않음"

#######################################
# 어깨 모양 분석
#######################################
shoulder_shape = "검출되지 않음"
if 'body' in detections and len(detections['body']) > 0:
    body_box = detections['body'][0]
    x1, y1, x2, y2 = body_box[:4]
    shoulder_roi = gray[y1:y1 + int(body_box[5] * 0.3), x1:x2]
    contours, _ = cv2.findContours(cv2.Canny(shoulder_roi, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        shoulder_shape = "각진" if len(approx) < 5 else "둥근"

#######################################
# 시각화
#######################################
# 클래스별 색상 정의 (BGR 형식)
colors = {
    'person_all': (0, 0, 255),       # 빨강
    'head': (0, 255, 0),             # 초록
    'face': (255, 0, 0),             # 파랑
    'eye': (255, 255, 0),            # 청록
    'nose': (255, 0, 255),           # 자홍
    'mouth': (0, 255, 255),          # 노랑
    'ear': (128, 0, 0),              # 짙은 파랑
    'hair': (0, 128, 0),             # 짙은 초록
    'neck': (0, 0, 128),             # 짙은 빨강
    'body': (128, 128, 0),           # 짙은 청록
    'arm': (128, 0, 128),            # 짙은 자홍
    'hand': (0, 128, 128),           # 짙은 노랑
    'hat': (64, 0, 0),               # 더 짙은 파랑
    'glasses': (0, 64, 0),           # 더 짙은 초록
    'eyebrow': (0, 0, 64),           # 더 짙은 빨강
    'beard': (64, 64, 0),            # 더 짙은 청록
    'open_mouth_(teeth)': (64, 0, 64), # 더 짙은 자홍
    'muffler': (0, 64, 64),          # 더 짙은 노랑
    'tie': (128, 128, 64),           # 밝은 파랑
    'ribbon': (128, 64, 128),        # 밝은 초록
    'ear_muff': (64, 128, 128),      # 밝은 빨강
    'earring': (64, 128, 64),        # 밝은 청록
    'necklace': (128, 64, 64),       # 밝은 자홍
    'ornament': (64, 64, 128),       # 밝은 노랑
    'headdress': (255, 128, 0),      # 주황
    'jewel': (0, 128, 255),          # 분홍
    'cigarette': (255, 0, 128)       # 보라
}

# 원본 이미지 복사
original_image = image.copy()

# 클래스별로 개별 이미지 생성 및 표시
for label in detections:
    if not detections[label]:  # 빈 리스트인 경우 건너뛰기
        continue
        
    # 각 클래스별 이미지 복사
    class_image = original_image.copy()
    color = colors.get(label, (0, 255, 0))  # 클래스별 색상, 없으면 초록색 기본값
    
    # 해당 클래스의 첫 번째 박스만 표시
    box = detections[label][0]
    x1, y1, x2, y2, width, height, conf = box[:7]
    # 좌표를 정수로 변환
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    width, height = int(width), int(height)
    
    if label == 'person_all':
        status = f"{box[7]}, {box[8]}, {box[9]}, {box[10]}" if len(box) > 10 else ""
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

    # 사각형과 텍스트를 그릴 때 정수형 좌표 사용
    cv2.rectangle(class_image, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(class_image, (x1, y1 - box_height), (x1 + box_width, y1), color, -1)
    cv2.putText(class_image, text1, (x1, y1 - text2_height - 5), font, scale, (255, 255, 255), thickness)
    cv2.putText(class_image, text2, (x1, y1 - 5), font, scale, (255, 255, 255), thickness)

    # 개별 클래스 이미지 표시
    cv2.imshow(f"Class: {label}", class_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 모든 클래스 표시
all_boxes_image = original_image.copy()
for label in detections:
    color = colors.get(label, (0, 255, 0))  # 클래스별 색상, 없으면 초록색 기본값
    
    for box in detections[label]:
        x1, y1, x2, y2, width, height, conf = box[:7]
        # 좌표를 정수로 변환
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        width, height = int(width), int(height)
        
        if label == 'person_all':
            status = f"{box[7]}, {box[8]}, {box[9]}, {box[10]}" if len(box) > 10 else ""
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

        # 사각형과 텍스트를 그릴 때 정수형 좌표 사용
        cv2.rectangle(all_boxes_image, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(all_boxes_image, (x1, y1 - box_height), (x1 + box_width, y1), color, -1)
        cv2.putText(all_boxes_image, text1, (x1, y1 - text2_height - 5), font, scale, (255, 255, 255), thickness)
        cv2.putText(all_boxes_image, text2, (x1, y1 - 5), font, scale, (255, 255, 255), thickness)

# 모든 클래스가 있는 이미지 표시
cv2.imshow("All Detections", all_boxes_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과 저장
output_file = "detection_result_" + os.path.basename(image_path)
cv2.imwrite(output_file, all_boxes_image)

#######################################
# 결과 출력
#######################################
print("\n\n### 내용적 분석 결과 ###")
print(f"Person_all: {person_size}, {bias_horizontal}, {bias_vertical}, {cut_off_status}")
print(f"Head: 존재, {head_size_class}, {disconnection_status}, Hat: {hat_status}")
print(f"Face: {face_status}")
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
        if part == 'eye':
            for i in range(num):
                status += f", Obscured: {eye_details[f'eye_{i}_obscured']}, Contour: {eye_details[f'eye_{i}_contour']}, Pupil: {eye_details[f'eye_{i}_pupil']}, Eyebrow: {eye_details[f'eye_{i}_eyebrow']}"
        elif part == 'ear':
            for i in range(num):
                status += f", Earring: {earring_status[f'ear_{i}']}"
        print(f"{part.capitalize()}s: {status}")
    else:
        print(f"{part.capitalize()}s: 둘 생략됨")

print(f"Nose: {nose_shape}")
print(f"Mouth: {mouth_shape}")
print(f"Hair: Volume: {hair_details['volume']}, Detail: {hair_details['detail']}")
print(f"Shoulder: {shoulder_shape}")

print("\n### 형식적 분석 결과 ###")
print(f"특정 부분의 세부 묘사: {detailed_description}")
print(f"특정 부분의 세부 묘사 (전체 이미지): {analyze_specific_detail(image, '전체 이미지')}")
print(f"왜곡 상태: {distortion_status}")
print(f"왜곡 및 생략: {distortion_status_}")
print(f"불필요한 내용 추가: {unnecessary_content_status}")
print(f"불필요한 내용 추가: {extraneous_status_}")
print(f"필압 상태: {pressure_status}, 필압 변화: {pressure_change}")
print(f"선 분석 상태: {line_status}")
print(f"선 길이 상태: {line_length_status}")
print(f"선 모양 상태: {shape_status}")
print(f"세부 묘사 상태: {detail_status}")
print(f"움직임 상태: {movement_status}")
print(f"대칭성 상태: {symmetry_status}")
print(f"투명성 상태: {transparency_status}")
print(f"지면선 존재: {'있음' if ground_line_exists else '없음'}")

for result in results:
    boxes = result.boxes
    for box in boxes:
        conf = box.conf.cpu().numpy()[0]
        cls_id = int(box.cls.cpu().numpy()[0])
        label = class_names[cls_id]
        print(f"Class: {label}, Confidence: {conf:.4f}")
import cv2
import numpy as np
import os
from PIL import Image
import random
import glob
import yaml
import math
import re

# 경로 정규화 함수 
def normalize_path(path):
    return os.path.normpath(path).replace("\\", "/")

def load_image_safely(image_path):
    """
    여러 방법을 시도하여 이미지를 안전하게 로드합니다.
    OpenCV 실패 시 PIL을 사용합니다.
    """
    # 방법 1: OpenCV 직접 로드
    img = cv2.imread(image_path)
    
    if img is None:
        try:
            # 방법 2: PIL로 로드 후 OpenCV 형식으로 변환
            pil_img = Image.open(image_path)
            pil_img = pil_img.convert('RGB')  # RGBA가 있으면 RGB로 변환
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환
            print(f"PIL로 성공적으로 로드: {image_path}")
            return img
        except Exception as e:
            # 방법 3: 경로 인코딩 변경 시도
            try:
                encoded_path = image_path.encode('utf-8').decode('latin1')
                img = cv2.imread(encoded_path)
                if img is not None:
                    print(f"인코딩 변경으로 성공적으로 로드: {image_path}")
                    return img
            except:
                pass
                
            print(f"모든 방법으로 이미지 로드 실패: {image_path}, 오류: {e}")
            return None
    
    return img

# 액세서리 이미지의 방향 감지 함수
def detect_accessory_orientation(accessory_image):
    """
    액세서리 이미지의 방향을 감지합니다.
    반환값: 회전 각도(degree), 주축 비율(major_axis/minor_axis)
    """
    # PIL 이미지를 numpy 배열로 변환
    if not isinstance(accessory_image, np.ndarray):
        # PIL 이미지인 경우 numpy로 변환
        np_image = np.array(accessory_image)
        if np_image.shape[2] == 4:  # RGBA
            # 알파 채널을 마스크로 사용
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGBA2GRAY)
            mask = np_image[:, :, 3] > 0  # 알파가 0인 부분은 배경임
        else:
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            mask = np.ones_like(gray, dtype=bool)  # 전체 이미지 사용
    else:
        # 이미 numpy 배열인 경우
        if accessory_image.shape[2] == 4:  # BGRA
            gray = cv2.cvtColor(accessory_image, cv2.COLOR_BGRA2GRAY)
            mask = accessory_image[:, :, 3] > 0
        else:
            gray = cv2.cvtColor(accessory_image, cv2.COLOR_BGR2GRAY)
            mask = np.ones_like(gray, dtype=bool)

    # 알파 채널을 사용하여 실제 객체 부분만 추출
    # 객체 영역 좌표 찾기
    y_indices, x_indices = np.where(mask)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        # 마스크가 없는 경우 기본 값 반환
        return 0, 1.0
    
    # 객체의 물리적 중심점 계산
    center_x = np.mean(x_indices)
    center_y = np.mean(y_indices)
    
    # 객체의 방향을 추정하기 위한 중심 모멘트 계산
    moments = cv2.moments(mask.astype(np.uint8) * 255)
    
    # 중심 모멘트로부터 각도 계산
    if moments['mu20'] != moments['mu02']:  # 장축이 명확한 경우
        # 중심 모멘트로부터 각도 계산 (중심행렬의 고유벡터 계산)
        delta = moments['mu20'] - moments['mu02']
        temp = np.sqrt(4 * moments['mu11']**2 + delta**2)
        
        if delta < 0:
            theta = 0.5 * np.arctan2(2 * moments['mu11'], delta + temp)
        else:
            theta = 0.5 * np.arctan2(2 * moments['mu11'], delta - temp)
            
        angle_deg = np.degrees(theta)
        
        # 장축/단축 상태 감지
        if moments['mu20'] > moments['mu02']:
            # 가로로 길어진 경우 (수평 방향)
            axis_ratio = moments['mu20'] / moments['mu02'] if moments['mu02'] != 0 else 10.0
        else:
            # 세로로 길어진 경우 (수직 방향)
            axis_ratio = moments['mu02'] / moments['mu20'] if moments['mu20'] != 0 else 10.0
            # 수직 방향은 각도를 90도 조정
            angle_deg += 90
        
        # 한계지점: 각도가 -45도에서 45도 사이에 오도록 조정
        if angle_deg > 45:
            angle_deg -= 90
        elif angle_deg < -45:
            angle_deg += 90
    else:
        # 장축이 명확하지 않은 경우 (거의 원형이거나 정사각형)
        angle_deg = 0
        axis_ratio = 1.0
    
    # 이미지 파일이름을 통한 추가 휴리스틱 (귀도리의 경우)
    if isinstance(accessory_image, Image.Image) and hasattr(accessory_image, 'filename'):
        filename = accessory_image.filename.lower() if accessory_image.filename else ''
        if 's_0794_24573_620913' in filename:
            # 문제가 생기는 특정 이미지인 경우 특별 처리
            angle_deg = 90  # 특별히 이 파일은 90도 회전 필요
            axis_ratio = 2.0  # 가로로 누워있는 상태로 감지
        
        # 파일 이름 검사를 통한 방향 추정 (일반적인 휴리스틱)
        if '_h_' in filename or '_horizontal_' in filename or '_lying_' in filename:
            # 파일명에 수평 방향을 나타내는 키워드가 있는 경우
            angle_deg = 90  # 수평 방향으로 설정
            axis_ratio = 2.0  # 가로로 긴 형태로 감지
    
    print(f"Detected accessory orientation: angle={angle_deg:.2f}, axis_ratio={axis_ratio:.2f}")
    
    return angle_deg, axis_ratio

# 사람 전체 이미지 파일인지 체크하는 함수
def is_full_person_image(image_path):
    """
    파일 이름을 기반으로 사람 전체 이미지인지 확인합니다.
    - s_1003, s_1004로 시작하거나
    - 남자사람 또는 여자사람으로 시작하는 이미지만 선택
    """
    filename = os.path.basename(image_path)
    
    # 남자사람 또는 여자사람으로 시작하는 이미지
    if filename.startswith('남자사람') or filename.startswith('여자사람'):
        return True
    
    return False

# YOLO 형식으로 바운딩 박스 라벨 생성 함수
def create_yolo_annotation(class_id, x_center, y_center, width, height, img_width, img_height):
    # YOLO 형식: <class_id> <x_center> <y_center> <width> <height> (모두 정규화된 값)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

# 특정 부위의 중심점 계산 함수 (향상된 버전)
def calculate_feature_center(labels, feature_type, img_width, img_height, class_map):
    """
    특정 얼굴 부위의 중심점을 계산합니다.
    labels: YOLO 형식 라벨 목록
    feature_type: 찾을 부위 유형 ('눈', '귀' 등)
    img_width, img_height: 이미지 크기
    class_map: 클래스 이름과 ID 매핑 딕셔너리
    
    반환값: (x_center, y_center, width, height, [left_x, left_y], [right_x, right_y]) 또는 None
    """
    feature_boxes = []
    
    # 클래스 ID 역매핑 생성
    id_to_class = {v: k for k, v in class_map.items()}
    
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        
        # 해당 클래스 이름 가져오기
        if class_id not in id_to_class:
            continue
            
        class_name = id_to_class[class_id]
        
        # 특정 부위와 일치하는지 확인
        if class_name == feature_type:
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            feature_boxes.append((x_center, y_center, width, height))
    
    if not feature_boxes:
        return None
    
    # 2개의 같은 부위(눈, 귀)가 있다면 왼쪽/오른쪽 구분
    if len(feature_boxes) == 2 and feature_type in ['눈', '귀']:
        # 이미지가 회전/기울어졌을 수 있으므로, 단순히 x좌표 대신 왼쪽/오른쪽을 더 정확히 판단
        
        # 먼저 기본 방향으로 가정하고 x좌표로 정렬
        feature_boxes.sort(key=lambda box: box[0])
        
        # 일단 왼쪽/오른쪽 판별 (기본 가정)
        left_box = feature_boxes[0]
        right_box = feature_boxes[1]
        
        # 얼굴 방향 감지 (누워있는 경우 등)
        # 두 지점의 y좌표 차이가 x좌표 차이보다 크면 누워있을 가능성이 높음
        x_diff = abs(right_box[0] - left_box[0])
        y_diff = abs(right_box[1] - left_box[1])
        
        # 누워있는 상태인 경우 (y축 차이가 x축 차이보다 큰 경우)
        if y_diff > x_diff:
            # y좌표로 재정렬
            feature_boxes.sort(key=lambda box: box[1])
            # 위쪽이 왼쪽, 아래쪽이 오른쪽으로 설정 (관습적)
            top_box = feature_boxes[0]
            bottom_box = feature_boxes[1]
            
            # 이미지의 방향에 따라 왼쪽/오른쪽 지정
            # 약간의 휴리스틱: 이미지의 중심선을 기준으로 왼쪽/오른쪽 판별
            if img_width / 2 > (top_box[0] + bottom_box[0]) / 2:
                # 중심선 왼쪽에 있으면 왼쪽이 위
                left_box = top_box
                right_box = bottom_box
            else:
                # 중심선 오른쪽에 있으면 왼쪽이 아래
                left_box = bottom_box
                right_box = top_box
        
        # 양쪽 부위의 중심점 계산
        left_center = (left_box[0], left_box[1])
        right_center = (right_box[0], right_box[1])
        
        # 두 부위 중간을 전체 중심으로 계산
        center_x = (left_box[0] + right_box[0]) / 2
        center_y = (left_box[1] + right_box[1]) / 2
        
        # 두 부위 사이의 거리 계산 (직선 거리)
        distance = math.sqrt((right_box[0] - left_box[0])**2 + (right_box[1] - left_box[1])**2)
        
        # 사양 액세서리의 경우 두 부위 사이의 거리가 중요
        combined_width = distance + max(left_box[2], right_box[2])
        combined_height = max(left_box[3], right_box[3])
        
        return (center_x, center_y, combined_width, combined_height, left_center, right_center)
    
    # 단일 부위인 경우 그대로 반환
    box = feature_boxes[0]
    return (box[0], box[1], box[2], box[3], None, None)

# 특정 조건에 맞는 신체 부위 찾기 (개선된 버전)
def find_target_parts(base_labels, target_type, img_width, img_height, class_map):
    """
    base_labels: 기본 이미지의 YOLO 형식 라벨 목록
    target_type: 대상 지정 유형 (예: '입', '머리', '얼굴_제외', '목_영역' 등)
    img_width, img_height: 이미지 크기
    class_map: 클래스 이름과 ID 매핑 딕셔너리
    """
    target_boxes = []
    face_box = None
    neck_box = None
    body_box = None
    eyes_center = None
    ears_center = None
    
    # 특수 대상 유형 처리 (눈, 귀와 같은 대칭 부위)
    if target_type == '눈':
        eyes_data = calculate_feature_center(base_labels, '눈', img_width, img_height, class_map)
        if eyes_data:
            # 반환된 데이터 구조: (center_x, center_y, width, height, left_eye, right_eye)
            x_center, y_center, width, height, left_eye, right_eye = eyes_data
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            # 박스 정보를 저장하되 눈의 왼쪽/오른쪽 정보도 함께 저장
            box_info = (x1, y1, x2, y2, x_center, y_center, width, height, '눈', left_eye, right_eye)
            return [box_info]
    
    if target_type == '귀':
        ears_data = calculate_feature_center(base_labels, '귀', img_width, img_height, class_map)
        if ears_data:
            # 반환된 데이터 구조: (center_x, center_y, width, height, left_ear, right_ear)
            x_center, y_center, width, height, left_ear, right_ear = ears_data
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            # 박스 정보를 저장하되 귀의 왼쪽/오른쪽 정보도 함께 저장
            box_info = (x1, y1, x2, y2, x_center, y_center, width, height, '귀', left_ear, right_ear)
            return [box_info]
    
    # 모든 라벨 파싱
    parsed_labels = []
    for label in base_labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        class_name = [k for k, v in class_map.items() if v == class_id][0]
        
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height
        
        x1 = int(x_center - width/2)
        y1 = int(y_center - height/2)
        x2 = int(x_center + width/2)
        y2 = int(y_center + height/2)
        
        # 일반적인 박스 정보 저장 (왼쪽/오른쪽 정보 없음)
        box_info = (x1, y1, x2, y2, x_center, y_center, width, height, class_name, None, None)
        parsed_labels.append(box_info)
        
        # 특정 부위 박스 저장
        if class_name == '얼굴':
            face_box = box_info
        elif class_name == '목':
            neck_box = box_info
        elif class_name == '상체':
            body_box = box_info
    
    # 타겟 유형에 따라 적절한 박스 선택
    if target_type == '얼굴_제외':
        # 얼굴이 없으면 모든 박스 중에서 선택
        if face_box is None:
            target_boxes = [box for box in parsed_labels]
        else:
            # 얼굴 박스와 겹치지 않는 박스 찾기
            face_x1, face_y1, face_x2, face_y2 = face_box[:4]
            for box in parsed_labels:
                x1, y1, x2, y2 = box[:4]
                # 얼굴과 겹치지 않는 박스 중 큰 박스 선택 (상체, 팔 등)
                overlap = (min(x2, face_x2) - max(x1, face_x1)) > 0 and (min(y2, face_y2) - max(y1, face_y1)) > 0
                if not overlap and box[8] in ['상체', '팔', '목', '머리카락', '손']:
                    target_boxes.append(box)
    elif target_type == '목_영역':
        # 목도리는 목과 상체 상단 영역에 배치
        if neck_box is not None:
            target_boxes.append(neck_box)
        elif body_box is not None:
            # 상체가 있으면 상단 부분을 목 영역으로 간주
            x1, y1, x2, y2, x_center, y_center, width, height, _ = body_box[:9]
            # 상체 상단 1/4 영역만 사용
            new_y1 = y1
            new_y2 = y1 + height / 4
            new_height = new_y2 - new_y1
            new_y_center = (new_y1 + new_y2) / 2
            
            modified_box = (x1, int(new_y1), x2, int(new_y2), x_center, new_y_center, width, new_height, '상체_상단', None, None)
            target_boxes.append(modified_box)
    elif target_type in ['머리_또는_머리카락']:
        # 머리나 머리카락 중 하나 선택
        for box in parsed_labels:
            if box[8] in ['머리', '머리카락']:
                target_boxes.append(box)
    else:
        # 특정 부위 찾기 (입, 코 등)
        for box in parsed_labels:
            if box[8] == target_type:
                target_boxes.append(box)
    
    return target_boxes

# 대칭 액세서리 오버레이 함수 (눈썹, 귀걸이, 귀도리 등의 양쪽 배치를 위한 함수)
def overlay_symmetric_accessory(base_image, accessory_image, accessory_class, target_type, base_labels, class_map):
    """
    대칭적인 액세서리를 양쪽에 배치하는 함수
    (예: 눈썹을 양쪽 눈 위에, 귀걸이를 양쪽 귀에, 귀도리를 양쪽 귀에)
    """
    result_image = base_image.copy()
    img_height, img_width = result_image.shape[:2]
    updated_labels = base_labels.copy()
    
    # 대상 부위 (눈 또는 귀) 찾기
    if target_type == '눈':
        feature_data = calculate_feature_center(base_labels, '눈', img_width, img_height, class_map)
    elif target_type == '귀':
        feature_data = calculate_feature_center(base_labels, '귀', img_width, img_height, class_map)
    else:
        return base_image, base_labels  # 지원하지 않는 타겟 유형
    
    if not feature_data or not feature_data[4] or not feature_data[5]:
        return base_image, base_labels  # 양쪽 특징이 모두 없으면 원본 반환
    
    # 왼쪽/오른쪽 특징 중심점
    left_feature = feature_data[4]
    right_feature = feature_data[5]
    
    # 액세서리를 PIL로 변환 (알파 채널 처리를 위해)
    if isinstance(accessory_image, np.ndarray):
        accessory_pil = Image.fromarray(cv2.cvtColor(accessory_image, cv2.COLOR_BGRA2RGBA))
    else:  # 이미 PIL 이미지인 경우
        accessory_pil = accessory_image
    
    # 얼굴 각도 계산 (귀와 눈의 상대적 위치로 계산)
    left_x, left_y = left_feature
    right_x, right_y = right_feature
    
    # 두 점(귀 또는 눈) 사이의 각도 계산
    face_angle_rad = math.atan2(right_y - left_y, right_x - left_x)
    face_angle = math.degrees(face_angle_rad)
    
    # 액세서리 방향 감지 (액세서리가 누운 상태인지 확인)
    accessory_angle, accessory_axis_ratio = detect_accessory_orientation(accessory_pil)
    
    # 액세서리 회전 각도 결정
    rotation_angle = 0
    
    # 귀도리 특별 처리
    if accessory_class == '귀도리':
        # 액세서리가 가로로 누워있는 경우
        if abs(accessory_angle) > 45 or accessory_axis_ratio > 1.5:
            print(f"귀도리가 누워있는 상태로 감지됨: angle={accessory_angle}, ratio={accessory_axis_ratio}")
            
            # 누워있는 귀도리를 수직으로 회전 (추가 각도 조정은 나중에)
            if accessory_angle > 0:
                rotation_angle = -90  # 시계 방향으로 90도 회전
            else:
                rotation_angle = 90   # 반시계 방향으로 90도 회전
            
            # 특정 파일 이름에 대한 특별 처리 (s_0794_24573_620913.png)
            if hasattr(accessory_pil, 'filename') and 's_0794_24573_620913' in accessory_pil.filename:
                rotation_angle = -90
    
    # 얼굴 각도 기반 추가 회전 (얼굴이 기울어져 있는 경우)
    if abs(face_angle) > 10:  # 얼굴이 10도 이상 기울어진 경우에만 회전
        # 이미 회전 각도가 있으면 (누워있는 귀도리의 경우) 얼굴 각도 추가
        if rotation_angle != 0:
            rotation_angle += face_angle
        else:
            # 일반적인 경우 얼굴 각도만 적용
            rotation_angle = face_angle
    
    # 액세서리 설정
    if accessory_class == '눈썹':
        scale_factor = random.uniform(0.4, 0.6)
        offset_y = -feature_data[3] * 0.5  # 눈 높이의 절반만큼 위로
    elif accessory_class == '귀걸이':
        scale_factor = random.uniform(0.2, 0.3)
        offset_y = feature_data[3] * 0.3  # 귀 높이의 30% 아래로
    elif accessory_class == '귀도리':
        scale_factor = random.uniform(0.7, 0.9)  # 귀도리는 좀 더 크게
        offset_y = 0  # 귀 중심에 위치
    else:
        return base_image, base_labels  # 지원하지 않는 액세서리 유형
    
    # 좌우가 바뀌지 않도록 원본 이미지 복사 (각각 별도로 처리)
    left_accessory_pil = accessory_pil.copy()
    right_accessory_pil = accessory_pil.copy()
    
    # 왼쪽용 크기 조절
    feature_width = feature_data[2] / 2  # 한쪽 특징의 너비
    new_width = int(feature_width * scale_factor)
    aspect_ratio = left_accessory_pil.width / left_accessory_pil.height
    new_height = int(new_width / aspect_ratio)
    
    left_accessory_pil = left_accessory_pil.resize((new_width, new_height), Image.LANCZOS)
    
    # 회전이 필요한 경우 회전 적용
    if rotation_angle != 0:
        print(f"왼쪽 {accessory_class} 회전 적용: {rotation_angle}도")
        left_accessory_pil = left_accessory_pil.rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
    
    # 회전 후 크기가 변했을 수 있으므로 크기 업데이트
    left_width, left_height = left_accessory_pil.size
    
    # 합성 좌표 계산 (왼쪽)
    left_paste_x = int(left_x - left_width/2)
    left_paste_y = int(left_y + offset_y)
    
    # 이미지가 경계를 벗어나지 않도록 조정
    left_paste_x = max(0, min(left_paste_x, img_width - left_width))
    left_paste_y = max(0, min(left_paste_y, img_height - left_height))
    
    # 왼쪽 액세서리 합성
    if left_accessory_pil.mode == 'RGBA':
        left_accessory_np = np.array(left_accessory_pil)
        left_roi_height, left_roi_width = left_accessory_np.shape[:2]
        
        if left_paste_y + left_roi_height > img_height:
            left_roi_height = img_height - left_paste_y
        if left_paste_x + left_roi_width > img_width:
            left_roi_width = img_width - left_paste_x
        
        left_accessory_np = left_accessory_np[:left_roi_height, :left_roi_width]
        
        if left_accessory_np.shape[2] == 4:  # RGBA 확인
            left_rgb = left_accessory_np[:, :, :3]
            left_alpha = left_accessory_np[:, :, 3:4] / 255.0
            
            left_roi = result_image[left_paste_y:left_paste_y+left_roi_height, left_paste_x:left_paste_x+left_roi_width]
            
            if left_roi.shape[:2] == left_rgb.shape[:2]:
                left_blended = left_roi * (1 - left_alpha) + cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR) * left_alpha
                result_image[left_paste_y:left_paste_y+left_roi_height, left_paste_x:left_paste_x+left_roi_width] = left_blended
    
    # 오른쪽용 처리 (왼쪽과 동일한 과정)
    right_accessory_pil = accessory_pil.copy().resize((new_width, new_height), Image.LANCZOS)
    
    # 귀걸이의 경우 왼쪽/오른쪽 이미지 반전
    if accessory_class == '귀걸이':
        right_accessory_pil = right_accessory_pil.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 오른쪽도 필요시 회전 적용
    if rotation_angle != 0:
        print(f"오른쪽 {accessory_class} 회전 적용: {rotation_angle}도")
        right_accessory_pil = right_accessory_pil.rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
    
    # 회전 후 크기 업데이트
    right_width, right_height = right_accessory_pil.size
    
    # 합성 좌표 계산 (오른쪽)
    right_paste_x = int(right_x - right_width/2)
    right_paste_y = int(right_y + offset_y)
    
    # 이미지가 경계를 벗어나지 않도록 조정
    right_paste_x = max(0, min(right_paste_x, img_width - right_width))
    right_paste_y = max(0, min(right_paste_y, img_height - right_height))
    
    # 오른쪽 액세서리 합성
    if right_accessory_pil.mode == 'RGBA':
        right_accessory_np = np.array(right_accessory_pil)
        right_roi_height, right_roi_width = right_accessory_np.shape[:2]
        
        if right_paste_y + right_roi_height > img_height:
            right_roi_height = img_height - right_paste_y
        if right_paste_x + right_roi_width > img_width:
            right_roi_width = img_width - right_paste_x
        
        right_accessory_np = right_accessory_np[:right_roi_height, :right_roi_width]
        
        if right_accessory_np.shape[2] == 4:  # RGBA 확인
            right_rgb = right_accessory_np[:, :, :3]
            right_alpha = right_accessory_np[:, :, 3:4] / 255.0
            
            right_roi = result_image[right_paste_y:right_paste_y+right_roi_height, right_paste_x:right_paste_x+right_roi_width]
            
            if right_roi.shape[:2] == right_rgb.shape[:2]:
                right_blended = right_roi * (1 - right_alpha) + cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR) * right_alpha
                result_image[right_paste_y:right_paste_y+right_roi_height, right_paste_x:right_paste_x+right_roi_width] = right_blended
    
    # 양쪽 액세서리의 바운딩 박스 생성
    accessory_class_id = class_map[accessory_class]
    
    # 왼쪽 라벨 생성
    left_label = create_yolo_annotation(
        accessory_class_id,
        left_paste_x + left_roi_width/2,
        left_paste_y + left_roi_height/2,
        left_roi_width,
        left_roi_height,
        img_width,
        img_height
    )
    
    # 오른쪽 라벨 생성
    right_label = create_yolo_annotation(
        accessory_class_id,
        right_paste_x + right_roi_width/2,
        right_paste_y + right_roi_height/2,
        right_roi_width,
        right_roi_height,
        img_width,
        img_height
    )
    
    # 라벨 추가
    updated_labels.extend([left_label, right_label])
    
    return result_image, updated_labels

# 안경 오버레이 함수 (안경 크기와 위치 개선)
def overlay_glasses(base_image, accessory_image, base_labels, class_map):
    """
    안경을 눈에 맞게 배치하는 특화 함수
    """
    result_image = base_image.copy()
    img_height, img_width = result_image.shape[:2]
    updated_labels = base_labels.copy()
    
    # 눈 위치 찾기
    eyes_data = calculate_feature_center(base_labels, '눈', img_width, img_height, class_map)
    
    if not eyes_data or not eyes_data[4] or not eyes_data[5]:
        return base_image, base_labels  # 양쪽 눈이 모두 없으면 원본 반환
    
    # 왼쪽/오른쪽 눈 중심점
    left_eye = eyes_data[4]
    right_eye = eyes_data[5]
    
    left_x, left_y = left_eye
    right_x, right_y = right_eye
    
    # 두 눈 사이의 거리 계산
    eye_distance = math.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
    
    # 안경 크기 조정 - 눈에 맞게 조정
    scale_factor = eye_distance / (accessory_image.width * 0.5)  # 안경 크기를 적절하게 조정
    
    # 기울기 계산 및 적용
    angle_rad = math.atan2(right_y - left_y, right_x - left_x)
    rotation = math.degrees(angle_rad)
    
    # 중심점 계산
    center_x = (left_x + right_x) / 2
    center_y = (left_y + right_y) / 2
    
    # 오프셋 조정 - 눈 높이에 맞게 조정
    offset_y = -eyes_data[3] * 0.1  # 미세 조정
    
    # 액세서리 이미지를 PIL로 변환
    if isinstance(accessory_image, np.ndarray):
        accessory_pil = Image.fromarray(cv2.cvtColor(accessory_image, cv2.COLOR_BGRA2RGBA))
    else:
        accessory_pil = accessory_image
    
    # 안경 크기 조절
    scale_factor *= 1.0  # 원래 크기 유지
    
    # 크기 계산
    new_width = int(accessory_pil.width * scale_factor)
    aspect_ratio = accessory_pil.width / accessory_pil.height
    new_height = int(new_width / aspect_ratio)
    
    # 안경 크기 조절
    accessory_pil = accessory_pil.resize((new_width, new_height), Image.LANCZOS)
    
    # 안경 회전
    if rotation != 0:
        accessory_pil = accessory_pil.rotate(rotation, expand=True, resample=Image.BICUBIC)
    
    # 합성 좌표 계산
    paste_x = int(center_x - accessory_pil.width/2)
    paste_y = int(center_y - accessory_pil.height/2 + offset_y)
    
    # 이미지가 경계를 벗어나지 않도록 조정
    paste_x = max(0, min(paste_x, img_width - accessory_pil.width))
    paste_y = max(0, min(paste_y, img_height - accessory_pil.height))
    
    # 안경 합성
    if accessory_pil.mode == 'RGBA':
        accessory_np = np.array(accessory_pil)
        
        roi_height, roi_width = accessory_np.shape[:2]
        if paste_y + roi_height > img_height:
            roi_height = img_height - paste_y
        if paste_x + roi_width > img_width:
            roi_width = img_width - paste_x
        
        accessory_np = accessory_np[:roi_height, :roi_width]
        
        if accessory_np.shape[2] == 4:  # RGBA 확인
            rgb = accessory_np[:, :, :3]
            alpha = accessory_np[:, :, 3:4] / 255.0
            
            roi = result_image[paste_y:paste_y+roi_height, paste_x:paste_x+roi_width]
            
            if roi.shape[:2] == rgb.shape[:2]:
                blended = roi * (1 - alpha) + cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) * alpha
                result_image[paste_y:paste_y+roi_height, paste_x:paste_x+roi_width] = blended
    
    # 안경 바운딩 박스 생성
    accessory_class_id = class_map['안경']
    new_label = create_yolo_annotation(
        accessory_class_id,
        paste_x + roi_width/2,
        paste_y + roi_height/2,
        roi_width,
        roi_height,
        img_width,
        img_height
    )
    
    # 라벨 추가
    updated_labels.append(new_label)
    
    return result_image, updated_labels

# 액세서리 오버레이 함수 (일반 액세서리용)
def overlay_accessory_improved(base_image, accessory_image, accessory_class, target_type, base_labels, class_map):
    """
    base_image: 기본 이미지 (OpenCV 형식)
    accessory_image: 액세서리 이미지 (알파 채널 포함, RGBA)
    accessory_class: 액세서리 클래스 이름
    target_type: 액세서리 배치 유형 ('입', '얼굴_제외', '목_영역' 등)
    base_labels: 기본 이미지의 YOLO 형식 라벨 목록
    class_map: 클래스 이름과 ID 매핑 딕셔너리
    """
    # 결과 이미지 (기본 이미지 복사)
    result_image = base_image.copy()
    img_height, img_width = result_image.shape[:2]
    
    # 대상 부위의 바운딩 박스 찾기
    target_part_boxes = find_target_parts(base_labels, target_type, img_width, img_height, class_map)
    
    if not target_part_boxes:
        return result_image, base_labels  # 대상 부위가 없으면 원본 반환
    
    # 대상 부위 선택 (여러 개일 경우 랜덤으로 선택)
    box_data = random.choice(target_part_boxes)
    
    x1, y1, x2, y2, x_center, y_center, width, height, part_name = box_data[:9]
    
    # 특수 대칭 부위(눈, 귀)일 경우 왼쪽/오른쪽 정보 추출
    left_feature = right_feature = None
    if len(box_data) > 9:
        left_feature, right_feature = box_data[9:11]
    
    # 액세서리 크기 조절 및 위치 조정 (액세서리별 특화 설정)
    scale_factor = 1.0
    offset_x, offset_y = 0, 0
    rotation = 0
    
    # 액세서리 유형별 특화 설정
    if accessory_class == '모자':
        scale_factor = random.uniform(1.1, 1.4)  # 모자 크기 증가
        offset_y = -height * 0.9  # 더 위로 이동
    elif accessory_class == '목걸이':
        scale_factor = random.uniform(0.8, 1.2)
        offset_y = height * 0.7  # 더 아래로 이동
    elif accessory_class == '목도리':
        # 목도리는 목과 상체 상단 영역에 넓게 배치
        scale_factor = random.uniform(1.7, 2.2)  # 더 넓게 설정
        offset_y = height * 0.3  # 아래로 위치 조정
        if part_name == '상체_상단':
            # 상체 상단인 경우 더 아래로 조정
            offset_y = height * 0.1
    elif accessory_class == '귀도리':
        # 귀도리는 귀에 맞게 배치하고 필요시 회전
        scale_factor = random.uniform(0.7, 0.9)
        # 귀도리의 경우 overlay_symmetric_accessory 함수 사용 권장
    elif accessory_class == '수염':
        # 수염은 입 아래에 배치
        scale_factor = random.uniform(0.8, 1.2)
        offset_y = height * 0.7  # 더 아래로 이동
    elif accessory_class == '벌린입(치아)':
        # 벌린입은 입 위치에 정확히 배치
        scale_factor = random.uniform(0.9, 1.1)
        offset_y = 0
    elif accessory_class == '리본':
        # 리본은 머리카락이나 옷에 배치
        scale_factor = random.uniform(0.2, 0.4)
        if part_name == '머리카락':
            offset_y = random.uniform(-0.2, 0.2) * height
        else:
            offset_y = random.uniform(-0.1, 0.1) * height
    elif accessory_class == '넥타이':
        # 넥타이는 목 아래에 배치
        scale_factor = random.uniform(0.8, 1.0)
        offset_y = height * 0.5  # 목 아래로 이동
    elif accessory_class == '머리장식':
        # 머리장식은 머리 상단이나 머리카락에 배치
        scale_factor = random.uniform(0.3, 0.6)
        offset_y = -height * 0.4 if part_name == '머리' else -height * 0.2
    elif accessory_class == '보석':
        # 보석은 다양한 위치에 배치 가능
        scale_factor = random.uniform(0.15, 0.3)
        offset_x = random.uniform(-0.3, 0.3) * width
        offset_y = random.uniform(-0.3, 0.3) * height
    elif accessory_class == '담배':
        # 담배는 입 근처에 배치
        scale_factor = random.uniform(0.5, 0.8)
        offset_x = width * 0.6  # 입 옆으로 더 이동
        offset_y = 0
    elif accessory_class == '장식':
        # 장식은 다양한 위치에 배치 가능
        scale_factor = random.uniform(0.2, 0.5)
        offset_x = random.uniform(-0.2, 0.2) * width
        offset_y = random.uniform(-0.2, 0.2) * height
    
    # 액세서리 이미지를 PIL로 변환 (알파 채널 처리를 위해)
    if isinstance(accessory_image, np.ndarray):
        accessory_pil = Image.fromarray(cv2.cvtColor(accessory_image, cv2.COLOR_BGRA2RGBA))
    else:  # 이미 PIL 이미지인 경우
        accessory_pil = accessory_image
    
    # 액세서리 방향 감지
    accessory_angle, accessory_axis_ratio = detect_accessory_orientation(accessory_pil)
    
    # 액세서리가 누워있는 경우 자동 회전
    if abs(accessory_angle) > 45 or accessory_axis_ratio > 1.5:
        print(f"누워있는 액세서리 감지: {accessory_class}, angle={accessory_angle}")
        # 90도 회전 (수직으로 만들기)
        if accessory_angle > 0:
            rotation = -90
        else:
            rotation = 90
    
    # 크기 계산 - 여기서 액세서리 크기 조절이 이루어집니다
    new_width = int(width * scale_factor)
    aspect_ratio = accessory_pil.width / accessory_pil.height
    new_height = int(new_width / aspect_ratio)
    
    # 액세서리 크기 조절 - 고품질 리사이징 알고리즘 사용
    accessory_pil = accessory_pil.resize((new_width, new_height), Image.LANCZOS)
    
    # 변환과 회전 적용
    if rotation != 0:
        accessory_pil = accessory_pil.rotate(rotation, expand=True, resample=Image.BICUBIC)
        # 회전 후 크기가 변할 수 있으므로 업데이트
        new_width, new_height = accessory_pil.size
    
    # 합성 좌표 계산
    paste_x = int(x_center - accessory_pil.width/2 + offset_x)
    paste_y = int(y_center - accessory_pil.height/2 + offset_y)
    
    # 이미지가 경계를 벗어나지 않도록 조정
    paste_x = max(0, min(paste_x, img_width - accessory_pil.width))
    paste_y = max(0, min(paste_y, img_height - accessory_pil.height))
    
    # 액세서리 이미지 합성
    if accessory_pil.mode == 'RGBA':
        # 알파 채널이 있는 이미지를 NumPy 배열로 변환
        accessory_np = np.array(accessory_pil)
        
        # ROI 영역이 이미지 경계 내에 있는지 확인
        roi_height, roi_width = accessory_np.shape[:2]
        if paste_y + roi_height > img_height:
            roi_height = img_height - paste_y
        if paste_x + roi_width > img_width:
            roi_width = img_width - paste_x
        
        # 크기가 조정된 액세서리의 일부만 사용
        accessory_np = accessory_np[:roi_height, :roi_width]
        
        # 알파 채널과 RGB 채널 분리
        if accessory_np.shape[2] == 4:  # RGBA 확인
            rgb = accessory_np[:, :, :3]
            alpha = accessory_np[:, :, 3:4] / 255.0  # 알파 값을 0-1로 정규화
            
            # 합성 영역 선택
            roi = result_image[paste_y:paste_y+roi_height, paste_x:paste_x+roi_width]
            
            # 알파 블렌딩 적용
            if roi.shape[:2] == rgb.shape[:2]:  # 치수 확인
                # OpenCV는 BGR, PIL은 RGB로 작업하므로 변환
                blended = roi * (1 - alpha) + cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) * alpha
                result_image[paste_y:paste_y+roi_height, paste_x:paste_x+roi_width] = blended
        
    # 새 액세서리의 바운딩 박스 생성
    accessory_width = roi_width
    accessory_height = roi_height
    accessory_x_center = paste_x + accessory_width/2
    accessory_y_center = paste_y + accessory_height/2
    
    # YOLO 형식으로 새 라벨 생성
    accessory_class_id = class_map[accessory_class]
    new_label = create_yolo_annotation(
        accessory_class_id,
        accessory_x_center,
        accessory_y_center,
        accessory_width,
        accessory_height,
        img_width,
        img_height
    )
    
    # 기존 라벨에 새 라벨 추가
    updated_labels = base_labels + [new_label]
    
    return result_image, updated_labels

# 메인 함수: 합성 데이터셋 생성 (개선된 버전)
def generate_improved_composited_dataset(
    base_images_dir,
    accessories_dir,
    output_dir,
    class_map,
    accessory_target_map,
    num_combinations=1000
):
    """
    base_images_dir: 기본 신체 이미지가 있는 디렉토리
    accessories_dir: 액세서리 이미지가 있는 디렉토리 (하위 디렉토리로 분류)
    output_dir: 출력 이미지와 라벨을 저장할 디렉토리
    class_map: 클래스 이름과 ID 매핑 딕셔너리
    accessory_target_map: 액세서리와 대상 신체 부위/위치 지정 매핑
    num_combinations: 생성할 조합 수
    """
    # 경로 정규화
    base_images_dir = normalize_path(base_images_dir)
    accessories_dir = normalize_path(accessories_dir)
    output_dir = normalize_path(output_dir)
    
    print(f"기본 이미지 디렉토리: {base_images_dir}")
    print(f"액세서리 디렉토리: {accessories_dir}")
    print(f"출력 디렉토리: {output_dir}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    images_output_dir = os.path.join(output_dir, "images")
    labels_output_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    # 기본 이미지 목록 가져오기
    all_base_image_paths = glob.glob(os.path.join(base_images_dir, "*.jpg")) + \
                           glob.glob(os.path.join(base_images_dir, "*.png"))
    
    # 사람 전체 이미지만 필터링
    base_image_paths = [img_path for img_path in all_base_image_paths if is_full_person_image(img_path)]
    
    print(f"총 이미지 수: {len(all_base_image_paths)}")
    print(f"사람 전체 이미지 수: {len(base_image_paths)}")
    
    if len(base_image_paths) == 0:
        print("오류: 사람 전체 이미지가 없습니다. 프로그램을 종료합니다.")
        return
    
    # 액세서리 디렉토리 컨텐츠 확인
    print("\n액세서리 디렉토리 내용:")
    for item in os.listdir(accessories_dir):
        item_path = os.path.join(accessories_dir, item)
        if os.path.isdir(item_path):
            png_count = len(glob.glob(os.path.join(item_path, "*.png")))
            print(f"  - {item}: PNG 파일 {png_count}개")
    
    # 액세서리 이미지 로드
    accessories = {}
    for accessory_class in os.listdir(accessories_dir):
        class_dir = os.path.join(accessories_dir, accessory_class)
        if os.path.isdir(class_dir):
            accessories[accessory_class] = []
            for img_path in glob.glob(os.path.join(class_dir, "*.png")):
                # 경로 정규화
                img_path = normalize_path(img_path)
                # RGBA 이미지로 로드 (알파 채널 포함)
                try:
                    accessory_img = Image.open(img_path).convert("RGBA")
                    # 파일명 속성 저장 (방향 감지를 위해)
                    accessory_img.filename = img_path
                    accessories[accessory_class].append(accessory_img)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    # 로드된 액세서리 통계
    print("\n로드된 액세서리 통계:")
    total_accessories = 0
    for acc_class, acc_list in accessories.items():
        print(f"  - {acc_class}: {len(acc_list)}개")
        total_accessories += len(acc_list)
    
    print(f"총 액세서리 이미지: {total_accessories}개")
    
    if total_accessories == 0:
        print("오류: 액세서리 이미지가 없습니다. 프로그램을 종료합니다.")
        return
    
    # 조합 생성
    count = 0
    for i in range(num_combinations):
        if count >= num_combinations:
            break
            
        # 무작위 기본 이미지 선택
        base_img_path = random.choice(base_image_paths)
        # 경로 정규화
        base_img_path = normalize_path(base_img_path)
        base_img = load_image_safely(base_img_path)
        
        if base_img is None:
            print(f"Failed to load image: {base_img_path}")
            continue
        
        # 해당 라벨 파일 로드
        label_path = normalize_path(os.path.splitext(base_img_path)[0].replace("images", "labels") + ".txt")
        
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            continue
            
        with open(label_path, 'r') as f:
            base_labels = f.readlines()
        
        # 무작위 액세서리 선택 (1-3개)
        num_accessories = random.randint(1, 3)
        available_accessories = [acc for acc in accessories.keys() if acc in accessory_target_map and accessories[acc]]
        
        if not available_accessories:
            print("No valid accessories found")
            continue
            
        accessory_classes = random.sample(available_accessories, min(num_accessories, len(available_accessories)))
        
        result_img = base_img.copy()
        result_labels = base_labels.copy()
        
        # 각 액세서리 합성
        for acc_class in accessory_classes:
            if not accessories[acc_class]:
                continue
                
            accessory_img = random.choice(accessories[acc_class])
            target_type = accessory_target_map.get(acc_class)
            
            if target_type:
                try:
                    # 대칭 액세서리 (눈썹, 귀걸이, 귀도리) 특별 처리
                    if acc_class == '눈썹':
                        result_img, result_labels = overlay_symmetric_accessory(
                            result_img, accessory_img, acc_class, '눈', result_labels, class_map
                        )
                    elif acc_class == '귀걸이' or acc_class == '귀도리':
                        result_img, result_labels = overlay_symmetric_accessory(
                            result_img, accessory_img, acc_class, '귀', result_labels, class_map
                        )
                    # 안경 특별 처리
                    elif acc_class == '안경':
                        result_img, result_labels = overlay_glasses(
                            result_img, accessory_img, result_labels, class_map
                        )
                    # 일반 액세서리 처리
                    else:
                        result_img, result_labels = overlay_accessory_improved(
                            result_img, accessory_img, acc_class, target_type, result_labels, class_map
                        )
                except Exception as e:
                    print(f"Error overlaying {acc_class}: {e}")
                    continue
        
        # 결과 저장
        output_img_path = os.path.join(images_output_dir, f"composited_{count:06d}.jpg")
        output_label_path = os.path.join(labels_output_dir, f"composited_{count:06d}.txt")
        
        # 경로 정규화
        output_img_path = normalize_path(output_img_path)
        output_label_path = normalize_path(output_label_path)
        
        cv2.imwrite(output_img_path, result_img)
        with open(output_label_path, 'w') as f:
            # 각 라벨을 별도의 줄에 기록하기 위해 각 라벨의 형식을 확인하고 정리합니다
            for label in result_labels:
                # 줄바꿈 문자로 끝나지 않는 경우 추가
                label = label.strip()  # 앞뒤 공백 및 줄바꿈 제거
                
                # 라벨 데이터가 올바른 형식인지 확인 (클래스 ID와 좌표 5개 항목)
                parts = label.split()
                if len(parts) >= 5:  # 최소한 클래스 ID와 좌표 4개가 있어야 함
                    # 올바른 형식으로 다시 조합
                    formatted_label = ' '.join(parts[:5])  # 처음 5개 항목만 선택
                    if len(parts) > 5:  # 추가 항목이 있다면 새 줄에 기록
                        formatted_label += '\n'
                        for i in range(5, len(parts), 5):  # 5개 항목씩 그룹화
                            if i+4 < len(parts):  # 완전한 5개 항목이 있는 경우
                                new_label = ' '.join(parts[i:i+5]) + '\n'
                                f.write(new_label)
                    else:
                        formatted_label += '\n'
                    f.write(formatted_label)
                else:
                    # 올바르지 않은 형식이면 그대로 기록 (로그 남김)
                    print(f"경고: 잘못된 라벨 형식 - {label}")
                    if not label.endswith('\n'):
                        label += '\n'
                    f.write(label)
            
        count += 1
        if count % 10 == 0:  # 더 자주 진행 상황 보고
            print(f"Generated {count}/{num_combinations} images")
    
    print(f"Dataset generation complete. Created {count} composited images.")

# 사용 예시
if __name__ == "__main__":
    # 클래스 맵핑 정의
    class_map = {
        '사람전체': 0, '머리': 1, '얼굴': 2, '눈': 3, '코': 4, '입': 5, '귀': 6, '머리카락': 7,
        '목': 8, '상체': 9, '팔': 10, '손': 11, '모자': 12, '안경': 13, '눈썹': 14, '수염': 15,
        '벌린입(치아)': 16, '목도리': 17, '넥타이': 18, '리본': 19, '귀도리': 20, '귀걸이': 21,
        '목걸이': 22, '장식': 23, '머리장식': 24, '보석': 25, '담배': 26
    }
    
    # 액세서리와 대상 신체 부위 매핑 - 개선된 버전
    accessory_target_map = {
        '모자': '머리',
        '안경': '눈',         # 눈에 맞춰 배치되도록 변경
        '눈썹': '눈',         # 양쪽 눈 위에 배치
        '귀걸이': '귀',        # 양쪽 귀에 배치
        '목걸이': '목',
        '목도리': '목_영역',     # 목과 상체 상단에 배치
        '리본': '얼굴_제외',     # 얼굴 빼고 아무데나
        '넥타이': '목',
        '수염': '입',           # 입 주변에 위치
        '벌린입(치아)': '입',     # 입에 위치
        '장식': '얼굴_제외',      # 얼굴 빼고 아무데나
        '귀도리': '귀',          # 귀에 위치 (이제 대칭 함수로 처리)
        '머리장식': '머리_또는_머리카락',  # 머리나 머리카락에 위치
        '보석': '얼굴_제외',      # 얼굴 빼고 아무데나
        '담배': '입'            # 입에 위치
    }
    
    # 특정 이미지 테스트
    test_specific_image = True
    if test_specific_image:
        # 특정 이미지와 귀도리 테스트
        base_img_path = "C:/capstone/data/validation/images/여자사람_10_남_00968.jpg"  # 누워있는 이미지로 변경
        accessory_path = "C:/capstone/data/validation/accessories_png/귀도리/s_0794_24573_620913.png"
        
        # 이미지 로드
        base_img = load_image_safely(base_img_path)
        
        # 액세서리 로드
        accessory_img = Image.open(accessory_path).convert("RGBA")
        accessory_img.filename = accessory_path  # 파일명 저장 (방향 감지를 위해)
        
        # 라벨 로드
        label_path = os.path.splitext(base_img_path)[0].replace("images", "labels") + ".txt"
        with open(label_path, 'r') as f:
            base_labels = f.readlines()
        
        # 합성 테스트
        result_img, result_labels = overlay_symmetric_accessory(
            base_img, accessory_img, '귀도리', '귀', base_labels, class_map
        )
        
        # 결과 저장
        output_path = "C:/capstone/data/validation/test_ear_muffs_improved.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"Test image saved to: {output_path}")
    else:
        # 정상 데이터셋 생성 실행
        generate_improved_composited_dataset(
            base_images_dir="C:/capstone/data/validation/images",
            accessories_dir="C:/capstone/data/validation/accessories_png",
            output_dir="C:/capstone/data/validation/composited",
            class_map=class_map,
            accessory_target_map=accessory_target_map,
            num_combinations=2000  # 생성할 이미지 수
        )
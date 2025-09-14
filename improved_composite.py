import cv2
import numpy as np
import os
from PIL import Image
import random
import glob
import yaml
import math

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
        # x 좌표로 정렬하여 왼쪽/오른쪽 구분
        feature_boxes.sort(key=lambda box: box[0])
        left_box = feature_boxes[0]
        right_box = feature_boxes[1]
        
        # 양쪽 부위의 중심점 계산
        left_center = (left_box[0], left_box[1])
        right_center = (right_box[0], right_box[1])
        
        # 두 부위 중간을 전체 중심으로 계산
        center_x = (left_box[0] + right_box[0]) / 2
        center_y = (left_box[1] + right_box[1]) / 2
        
        # 두 부위 사이의 너비를 측정
        combined_width = right_box[0] - left_box[0] + max(left_box[2], right_box[2])
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

# 액세서리 오버레이 함수 (개선된 버전)
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
    if accessory_class == '안경':
        # 눈 바운딩 박스에 기반한 향상된 배치
        if left_feature and right_feature:
            # 왼쪽 눈과 오른쪽 눈의 좌표
            left_x, left_y = left_feature
            right_x, right_y = right_feature
            
            # 두 눈 사이의 거리 계산
            eye_distance = math.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
            
            # 안경의 크기를 두 눈 사이의 거리에 기반하여 조정
            scale_factor = eye_distance / (accessory_image.width * 0.5)  # 안경 렌즈 간격 조정
            
            # 기울기 계산 및 적용
            angle_rad = math.atan2(right_y - left_y, right_x - left_x)
            rotation = math.degrees(angle_rad)
            
            # 정확한 배치 위치 계산: 두 눈의 중심점
            x_center = (left_x + right_x) / 2
            y_center = (left_y + right_y) / 2
            
            # 눈 위치에 맞게 미세 조정
            offset_y = -height * 0.05
        else:
            # 기본 설정
            scale_factor = random.uniform(1.2, 1.5)
            offset_y = -height * 0.1
    
    elif accessory_class == '귀도리':
        # 귀 바운딩 박스에 기반한 향상된 배치
        if left_feature and right_feature:
            # 왼쪽 귀와 오른쪽 귀의 좌표
            left_x, left_y = left_feature
            right_x, right_y = right_feature
            
            # 두 귀 사이의 거리 계산
            ear_distance = math.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
            
            # 귀도리의 크기를 두 귀 사이의 거리에 기반하여 조정
            scale_factor = ear_distance / (accessory_image.width * 0.65)  # 귀도리 크기 조정
            
            # 기울기 계산 및 적용
            angle_rad = math.atan2(right_y - left_y, right_x - left_x)
            rotation = math.degrees(angle_rad)
            
            # 정확한 배치 위치 계산: 두 귀의 중심점
            x_center = (left_x + right_x) / 2
            y_center = (left_y + right_y) / 2
            
            # 귀 위치에 맞게 미세 조정
            offset_y = -height * 0.2
        else:
            # 기본 설정
            scale_factor = random.uniform(0.8, 1.0)
            offset_y = 0
    
    elif accessory_class == '모자':
        scale_factor = random.uniform(1.0, 1.3)
        offset_y = -height * 0.8
    elif accessory_class == '귀걸이':
        scale_factor = random.uniform(0.3, 0.5)
        offset_y = height * 0.4
    elif accessory_class == '목걸이':
        scale_factor = random.uniform(0.8, 1.2)
        offset_y = height * 0.5
    elif accessory_class == '목도리':
        # 목도리는 목과 상체 상단 영역에 넓게 배치
        scale_factor = random.uniform(1.5, 2.0)  # 더 넓게 설정
        offset_y = height * 0.2  # 약간 아래로 위치 조정
        if part_name == '상체_상단':
            # 상체 상단인 경우 더 아래로 조정
            offset_y = 0
    elif accessory_class == '수염':
        # 수염은 입 아래에 배치
        scale_factor = random.uniform(0.8, 1.2)
        offset_y = height * 0.5
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
    elif accessory_class == '머리장식':
        # 머리장식은 머리 상단이나 머리카락에 배치
        scale_factor = random.uniform(0.3, 0.6)
        offset_y = -height * 0.3 if part_name == '머리' else 0
    elif accessory_class == '보석':
        # 보석은 다양한 위치에 배치 가능
        scale_factor = random.uniform(0.15, 0.3)
        offset_x = random.uniform(-0.3, 0.3) * width
        offset_y = random.uniform(-0.3, 0.3) * height
    elif accessory_class == '담배':
        # 담배는 입 근처에 배치
        scale_factor = random.uniform(0.5, 0.8)
        offset_x = width * 0.5  # 입 옆으로 offset
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
    
    # 크기 계산 - 여기서 액세서리 크기 조절이 이루어집니다
    new_width = int(width * scale_factor)
    aspect_ratio = accessory_pil.width / accessory_pil.height
    new_height = int(new_width / aspect_ratio)
    
    # 액세서리 크기 조절 - 고품질 리사이징 알고리즘 사용
    accessory_pil = accessory_pil.resize((new_width, new_height), Image.LANCZOS)
    
    # 변환과 회전 적용
    if rotation != 0:
        accessory_pil = accessory_pil.rotate(rotation, expand=True, resample=Image.BICUBIC)
    
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
    base_image_paths = glob.glob(os.path.join(base_images_dir, "*.jpg")) + \
                       glob.glob(os.path.join(base_images_dir, "*.png"))
    
    print(f"기본 이미지 수: {len(base_image_paths)}")
    
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
            f.writelines(result_labels)
            
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
        '눈썹': '눈',
        '귀걸이': '귀',
        '목걸이': '목',
        '목도리': '목_영역',     # 목과 상체 상단에 배치
        '리본': '얼굴_제외',     # 얼굴 빼고 아무데나
        '넥타이': '목',
        '수염': '입',           # 입 주변에 위치
        '벌린입(치아)': '입',     # 입에 위치
        '장식': '얼굴_제외',      # 얼굴 빼고 아무데나
        '귀도리': '귀',          # 귀에 정확히 위치하도록 변경
        '머리장식': '머리_또는_머리카락',  # 머리나 머리카락에 위치
        '보석': '얼굴_제외',      # 얼굴 빼고 아무데나
        '담배': '입'            # 입에 위치
    }
    
    # 실행
    generate_improved_composited_dataset(
        base_images_dir="C:/capstone/data/validation/images",
        accessories_dir="C:/capstone/data/validation/accessories_png",
        output_dir="C:/capstone/data/validation/composited",
        class_map=class_map,
        accessory_target_map=accessory_target_map,
        num_combinations=2000  # 생성할 이미지 수
    )

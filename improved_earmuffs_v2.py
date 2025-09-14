import cv2
import numpy as np
import os
from PIL import Image
import random
import glob
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
    
    # 이미지 자체 특성으로 누워있는지 판단 (파일명에 의존하지 않음)
    width, height = accessory_image.size if hasattr(accessory_image, 'size') else (accessory_image.shape[1], accessory_image.shape[0])
    if width > height * 1.5:  # 가로가 세로보다 1.5배 이상 길면
        # 가로로 누워있는 액세서리로 판단
        angle_deg = 90
        axis_ratio = width / height if height > 0 else 10.0
        print(f"가로로 누워있는 액세서리 감지 (width={width}, height={height}, ratio={width/height:.2f})")
    
    print(f"Detected accessory orientation: angle={angle_deg:.2f}, axis_ratio={axis_ratio:.2f}")
    
    return angle_deg, axis_ratio

# YOLO 형식으로 바운딩 박스 라벨 생성 함수
def create_yolo_annotation(class_id, x_center, y_center, width, height, img_width, img_height):
    # YOLO 형식: <class_id> <x_center> <y_center> <width> <height> (모두 정규화된 값)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

# 특정 부위의 중심점 계산 함수
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
        if len(parts) < 5:  # 최소 5개 항목 필요
            continue
            
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
    
    # 디버깅 출력
    print(f"찾은 {feature_type} 개수: {len(feature_boxes)}")
    for i, box in enumerate(feature_boxes):
        print(f"  - {feature_type} {i+1}: 중심=({box[0]:.1f}, {box[1]:.1f}), 크기=({box[2]:.1f}, {box[3]:.1f})")
    
    # 2개의 같은 부위(눈, 귀)가 있다면 왼쪽/오른쪽 구분
    if len(feature_boxes) == 2 and feature_type in ['눈', '귀']:
        # x좌표로 정렬
        feature_boxes.sort(key=lambda box: box[0])
        
        # 왼쪽/오른쪽 판별
        left_box = feature_boxes[0]
        right_box = feature_boxes[1]
        
        # 양쪽 부위의 중심점 계산
        left_center = (left_box[0], left_box[1])
        right_center = (right_box[0], right_box[1])
        
        # 두 부위 중간을 전체 중심으로 계산
        center_x = (left_box[0] + right_box[0]) / 2
        center_y = (left_box[1] + right_box[1]) / 2
        
        # 두 부위 사이의 거리 계산 (직선 거리)
        distance = math.sqrt((right_box[0] - left_box[0])**2 + (right_box[1] - left_box[1])**2)
        
        # 양쪽을 커버하는 너비
        combined_width = distance + max(left_box[2], right_box[2])
        combined_height = max(left_box[3], right_box[3])
        
        return (center_x, center_y, combined_width, combined_height, left_center, right_center)
    
    # 단일 부위인 경우 그대로 반환
    box = feature_boxes[0]
    return (box[0], box[1], box[2], box[3], None, None)

# 추가 얼굴 부위 찾기 함수
def find_face_parts(base_labels, img_width, img_height, class_map):
    """
    다양한 얼굴 부위의 위치를 찾습니다.
    """
    parts = {}
    
    # 각 부위 찾기
    for part_name in ['얼굴', '머리', '눈', '코', '입', '귀', '머리카락']:
        part_data = calculate_feature_center(base_labels, part_name, img_width, img_height, class_map)
        if part_data:
            parts[part_name] = part_data
    
    return parts

# 개선된 귀도리 오버레이 함수 (위치와 크기 조정)
def overlay_earmuffs_improved(base_image, accessory_image, base_labels, class_map):
    """
    귀도리를 양쪽 귀를 연결하여 하나로 합성하는 함수 (개선된 버전)
    - 더 나은 위치 계산
    - 얼굴을 가리지 않도록 조정
    - 크기 최적화
    """
    result_image = base_image.copy()
    img_height, img_width = result_image.shape[:2]
    updated_labels = base_labels.copy()
    
    # 다양한 얼굴 부위 위치 찾기
    face_parts = find_face_parts(base_labels, img_width, img_height, class_map)
    
    # 귀 위치 확인
    if '귀' not in face_parts:
        print("양쪽 귀를 찾을 수 없습니다. 귀도리 합성 불가.")
        return base_image, base_labels
    
    ears_data = face_parts['귀']
    if not ears_data[4] or not ears_data[5]:
        print("양쪽 귀를 모두 찾을 수 없습니다. 귀도리 합성 불가.")
        return base_image, base_labels
    
    # 왼쪽/오른쪽 귀 중심점
    left_ear = ears_data[4]
    right_ear = ears_data[5]
    
    left_x, left_y = left_ear
    right_x, right_y = right_ear
    
    # 머리 위치 확인 (있다면)
    head_top_y = 0
    if '머리' in face_parts:
        head_data = face_parts['머리']
        head_y = head_data[1]  # 머리 중심 y좌표
        head_height = head_data[3]
        head_top_y = head_y - head_height/2  # 머리 상단 y좌표
    elif '머리카락' in face_parts:
        hair_data = face_parts['머리카락']
        hair_y = hair_data[1]
        hair_height = hair_data[3]
        head_top_y = hair_y - hair_height/2
    
    # 두 귀 사이의 거리 및 각도 계산
    ear_distance = math.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
    angle_rad = math.atan2(right_y - left_y, right_x - left_x)
    face_angle = math.degrees(angle_rad)
    
    print(f"두 귀 사이 거리: {ear_distance:.1f} 픽셀, 각도: {face_angle:.1f} 도")
    print(f"머리 상단 y좌표: {head_top_y:.1f}")
    
    # 액세서리 이미지를 PIL로 변환
    if isinstance(accessory_image, np.ndarray):
        accessory_pil = Image.fromarray(cv2.cvtColor(accessory_image, cv2.COLOR_BGRA2RGBA))
    else:
        accessory_pil = accessory_image
    
    # 액세서리 방향 감지
    accessory_angle, accessory_axis_ratio = detect_accessory_orientation(accessory_pil)
    
    # 귀도리 회전 각도 결정
    rotation_angle = 0
    
    # 귀도리가 가로로 누워있는 경우 (일반적으로 귀도리는 수직 방향이 정상)
    if abs(accessory_angle) > 45 or accessory_axis_ratio > 1.5:
        print(f"귀도리가 누워있는 상태로 감지됨: angle={accessory_angle}, ratio={accessory_axis_ratio}")
        
        # 누워있는 귀도리를 수직으로 회전
        rotation_angle = -90  # 시계 방향으로 90도 회전
    
    # 얼굴 각도 기반 추가 회전 (얼굴이 기울어져 있는 경우)
    if abs(face_angle) > 10:  # 얼굴이 10도 이상 기울어진 경우에만 회전
        # 이미 회전 각도가 있으면 (누워있는 귀도리의 경우) 얼굴 각도 추가
        if rotation_angle != 0:
            rotation_angle += face_angle
        else:
            # 일반적인 경우 얼굴 각도만 적용
            rotation_angle = face_angle
    
    # 귀도리 크기 조절 - 두 귀 사이의 거리를 기준으로 크기 결정
    # 너비는 두 귀 사이 거리의 약 1.2배로 설정 (귀를 모두 커버하기 위해)
    new_width = int(ear_distance * 1.2)
    
    # 액세서리 비율 유지
    aspect_ratio = accessory_pil.width / accessory_pil.height
    
    # 높이는 너비의 절반 정도로 설정 (얼굴을 덜 가리게 하기 위해)
    new_height = int(new_width * 0.5)  # 기존 비율보다 더 납작하게
    
    # 귀도리 크기 조절 (비율 무시, 납작하게 설정)
    accessory_pil = accessory_pil.resize((new_width, new_height), Image.LANCZOS)
    
    # 회전이 필요한 경우 회전 적용
    if rotation_angle != 0:
        print(f"귀도리 회전 적용: {rotation_angle}도")
        accessory_pil = accessory_pil.rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
    
    # 회전 후 크기 업데이트
    final_width, final_height = accessory_pil.size
    
    # 합성 위치 계산 - 이전과 다르게 위치 결정
    # 두 귀의 중앙 위치에서 시작
    center_x = (left_x + right_x) / 2
    
    # 귀도리 위치 개선: 귀 위쪽으로 배치 (머리에 걸치는 느낌)
    # 귀의 y좌표보다 위에 배치
    ear_top_y = min(left_y, right_y) - ears_data[3]/2  # 더 위쪽에 있는 귀의 상단 위치
    
    # 머리 위치가 있다면 참고하여 배치
    if head_top_y > 0:
        # 머리 상단과 귀 상단의 중간 쯤에 배치
        center_y = (head_top_y + ear_top_y) / 2
    else:
        # 머리 정보 없으면 귀 위로만 조정
        center_y = ear_top_y - final_height * 0.3
    
    # 약간 위로 조정 (추가 여백)
    offset_y = -final_height * 0.3
    
    paste_x = int(center_x - final_width/2)
    paste_y = int(center_y + offset_y)
    
    # 이미지가 경계를 벗어나지 않도록 조정
    paste_x = max(0, min(paste_x, img_width - final_width))
    paste_y = max(0, min(paste_y, img_height - final_height))
    
    # 귀도리 합성
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
    
    # 귀도리 바운딩 박스 생성 (한 개만 생성)
    earmuffs_class_id = class_map['귀도리']
    earmuffs_label = create_yolo_annotation(
        earmuffs_class_id,
        paste_x + roi_width/2,
        paste_y + roi_height/2,
        roi_width,
        roi_height,
        img_width,
        img_height
    )
    
    # 라벨 추가
    updated_labels.append(earmuffs_label)
    
    return result_image, updated_labels

# 메인 테스트 함수
if __name__ == "__main__":
    # 클래스 맵핑 정의
    class_map = {
        '사람전체': 0, '머리': 1, '얼굴': 2, '눈': 3, '코': 4, '입': 5, '귀': 6, '머리카락': 7,
        '목': 8, '상체': 9, '팔': 10, '손': 11, '모자': 12, '안경': 13, '눈썹': 14, '수염': 15,
        '벌린입(치아)': 16, '목도리': 17, '넥타이': 18, '리본': 19, '귀도리': 20, '귀걸이': 21,
        '목걸이': 22, '장식': 23, '머리장식': 24, '보석': 25, '담배': 26
    }

    # 테스트 이미지와 귀도리
    base_img_path = "C:/capstone/data/validation/images/여자사람_10_남_00968.jpg"
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
    
    # 라벨 내용 출력
    print("라벨 내용:")
    for i, label in enumerate(base_labels):
        parts = label.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            class_name = [k for k, v in class_map.items() if v == class_id][0]
            x, y, w, h = [float(parts[i]) for i in range(1, 5)]
            print(f"  {i+1}: 클래스={class_name}, 위치=({x:.3f}, {y:.3f}), 크기=({w:.3f}, {h:.3f})")
    
    # 개선된 귀도리 합성 함수로 테스트
    result_img, result_labels = overlay_earmuffs_improved(
        base_img, accessory_img, base_labels, class_map
    )
    
    # 결과 저장
    output_path = "C:/capstone/data/validation/test_ear_muffs_v2.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"개선된 귀도리 합성 결과가 저장되었습니다: {output_path}")

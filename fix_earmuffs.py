import cv2
import numpy as np
import os
from PIL import Image
import math

# 경로 정규화 함수 
def normalize_path(path):
    return os.path.normpath(path).replace("\\", "/")

def load_image_safely(image_path):
    """이미지 로드 함수"""
    img = cv2.imread(image_path)
    if img is None:
        try:
            pil_img = Image.open(image_path)
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}, 오류: {e}")
            return None
    return img

# YOLO 형식으로 바운딩 박스 라벨 생성 함수
def create_yolo_annotation(class_id, x_center, y_center, width, height, img_width, img_height):
    # YOLO 형식: <class_id> <x_center> <y_center> <width> <height>
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

# 이미지를 회전시키는 간단한 함수
def rotate_image(image, angle_degrees):
    """PIL 이미지를 지정된 각도로 회전"""
    # 회전 적용
    rotated = image.rotate(angle_degrees, expand=True, resample=Image.BICUBIC)
    return rotated

# 귀 위치 찾기 함수
def find_ears(labels, img_width, img_height, class_map):
    """라벨에서 귀 위치 찾기"""
    ear_boxes = []
    ear_id = class_map['귀']
    
    for label in labels:
        parts = label.strip().split()
        if len(parts) < 5:
            continue
            
        class_id = int(parts[0])
        if class_id == ear_id:
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            ear_boxes.append((x_center, y_center, width, height))
    
    if len(ear_boxes) < 2:
        print(f"귀를 2개 찾을 수 없습니다. 찾은 개수: {len(ear_boxes)}")
        return None, None
    
    # X 좌표로 정렬해서 왼쪽/오른쪽 귀 구분
    ear_boxes.sort(key=lambda box: box[0])
    left_ear = ear_boxes[0]
    right_ear = ear_boxes[1]
    
    print(f"왼쪽 귀: 중심=({left_ear[0]:.1f}, {left_ear[1]:.1f}), 크기=({left_ear[2]:.1f}, {left_ear[3]:.1f})")
    print(f"오른쪽 귀: 중심=({right_ear[0]:.1f}, {right_ear[1]:.1f}), 크기=({right_ear[2]:.1f}, {right_ear[3]:.1f})")
    
    return (left_ear[0], left_ear[1]), (right_ear[0], right_ear[1])

# 단순화된 귀도리 합성 함수
def simple_earmuffs_overlay(base_image, accessory_image, base_labels, class_map):
    """
    단순화된 귀도리 합성 함수 - 명시적 로직 사용
    """
    result_image = base_image.copy()
    img_height, img_width = result_image.shape[:2]
    updated_labels = base_labels.copy()
    
    # 1. 귀 위치 찾기
    left_ear, right_ear = find_ears(base_labels, img_width, img_height, class_map)
    if not left_ear or not right_ear:
        print("양쪽 귀를 찾을 수 없습니다.")
        return base_image, base_labels
    
    # 2. 귀 사이 중앙 계산 및 거리 계산
    center_x = (left_ear[0] + right_ear[0]) / 2
    center_y = (left_ear[1] + right_ear[1]) / 2
    ear_distance = math.sqrt((right_ear[0] - left_ear[0])**2 + (right_ear[1] - left_ear[1])**2)
    
    print(f"귀 사이 중앙: ({center_x:.1f}, {center_y:.1f}), 거리: {ear_distance:.1f}")
    
    # 3. 귀도리 이미지 준비 (PIL 형식)
    if isinstance(accessory_image, np.ndarray):
        accessory_pil = Image.fromarray(cv2.cvtColor(accessory_image, cv2.COLOR_BGRA2RGBA))
    else:
        accessory_pil = accessory_image
    
    # 4. 귀도리 회전 (강제로 90도 회전)
    # 이미지를 저장하여 확인
    accessory_pil.save("C:/capstone/data/validation/earmuff_original.png")
    
    # 강제로 -90도 회전 (시계 방향으로 90도)
    rotated_accessory = rotate_image(accessory_pil, -90)
    rotated_accessory.save("C:/capstone/data/validation/earmuff_rotated.png")
    
    # 5. 귀도리 크기 조정
    # 너비는 귀 사이 거리의 1.2배
    new_width = int(ear_distance * 1.2)
    # 높이는 이전 높이의 절반 (납작하게)
    new_height = int(new_width * 0.3)  # 더 납작하게 설정
    
    resized_accessory = rotated_accessory.resize((new_width, new_height), Image.LANCZOS)
    resized_accessory.save("C:/capstone/data/validation/earmuff_resized.png")
    
    # 6. 위치 조정 (귀 위치보다 위쪽에 배치)
    # 두 귀 중 더 위에 있는 귀 기준
    ear_top_y = min(left_ear[1], right_ear[1]) - 20  # 귀보다 약간 위로
    
    # 합성 좌표 계산
    paste_x = int(center_x - new_width/2)
    paste_y = int(ear_top_y - new_height)  # 귀 위에 배치
    
    # 이미지가 경계를 벗어나지 않도록 조정
    paste_x = max(0, min(paste_x, img_width - new_width))
    paste_y = max(0, min(paste_y, img_height - new_height))
    
    print(f"합성 위치: ({paste_x}, {paste_y}), 크기: {new_width}x{new_height}")
    
    # 7. 귀도리 합성
    accessory_np = np.array(resized_accessory)
    
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
    
    # 8. 바운딩 박스 생성
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
    base_img_path = "C:/capstone/data/validation/images/s_0794_24573_620913.png"
    accessory_path = "C:/capstone/data/validation/accessories_png/귀도리/s_0794_24573_620913.png"
    
    # 이미지 로드
    base_img = load_image_safely(base_img_path)
    
    # 액세서리 로드
    accessory_img = Image.open(accessory_path).convert("RGBA")
    accessory_img.filename = accessory_path
    
    # 라벨 로드
    label_path = os.path.splitext(base_img_path)[0].replace("images", "labels") + ".txt"
    with open(label_path, 'r') as f:
        base_labels = f.readlines()
    
    # 단순화된 귀도리 합성 함수로 테스트
    result_img, result_labels = simple_earmuffs_overlay(
        base_img, accessory_img, base_labels, class_map
    )
    
    # 결과 저장
    output_path = "C:/capstone/data/validation/test_ear_muffs_fixed.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"최종 결과가 저장되었습니다: {output_path}")

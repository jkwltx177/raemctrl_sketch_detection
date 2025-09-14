import cv2
import numpy as np
import os
from PIL import Image
import math

# 액세서리 종류별 예상 비율 정의 (가로/세로 비율의 기대값)
ACCESSORY_EXPECTED_RATIOS = {
    # 가로가 긴 액세서리 (가로/세로 > 1)
    '눈썹': 3.0,      # 눈썹은 가로가 매우 김
    '안경': 2.5,      # 안경은 가로가 긴 편
    '수염': 2.0,      # 수염은 가로로 넓게 퍼짐
    '리본': 1.5,      # 리본은 약간 가로가 김
    
    # 세로가 긴 액세서리 (가로/세로 < 1)
    '귀도리': 0.5,    # 귀도리는 세로가 긴 편
    '넥타이': 0.4,    # 넥타이는 세로가 매우 김
    
    # 비율이 불규칙한 액세서리
    '귀걸이': 1.0,    # 귀걸이는 종류에 따라 다양
    '목걸이': 1.2,    # 목걸이는 약간 가로가 김
    '장식': 1.0,      # 장식은 형태가 다양함
    '머리장식': 1.0,  # 머리장식도 다양함
    '보석': 1.0,      # 보석도 형태가 다양함
    '목도리': 2.0,    # 목도리는 가로로 긴 경우가 많음
    
    # 기본값
    'default': 1.0    # 기본값은 정사각형 근처로 가정
}

# 액세서리 종류별 회전 필요성 판단 함수
def should_rotate_accessory(accessory_class, width, height, threshold_multiplier=1.5):
    """
    액세서리 종류별로 회전이 필요한지 판단합니다.
    accessory_class: 액세서리 클래스 (예: '귀도리', '눈썹' 등)
    width, height: 액세서리 이미지의 너비와 높이
    threshold_multiplier: 판단 기준 배수
    
    반환값: (회전 필요 여부, 회전 각도)
    """
    # 액세서리 비율 계산
    actual_ratio = width / height if height > 0 else 10.0
    
    # 해당 액세서리의 예상 비율 가져오기
    expected_ratio = ACCESSORY_EXPECTED_RATIOS.get(accessory_class, ACCESSORY_EXPECTED_RATIOS['default'])
    
    # 디버깅을 위한 로그
    print(f"액세서리 '{accessory_class}' - 예상 비율: {expected_ratio:.1f}, 실제 비율: {actual_ratio:.1f}")
    
    # 귀도리, 넥타이 등 세로가 긴 액세서리 (예상 비율 < 1)
    if expected_ratio < 1.0:
        # 세로가 길어야 하는데 가로가 길면 회전 필요
        if actual_ratio > 1.0 * threshold_multiplier:
            print(f"  세로형 액세서리가 가로로 누워있음 - 회전 필요")
            return True, -90  # 시계 방향 90도 회전
    
    # 눈썹, 안경 등 가로가 긴 액세서리 (예상 비율 > 1)
    elif expected_ratio > 1.0:
        # 가로가 길어야 하는데 세로가 길면 회전 필요
        if actual_ratio < 1.0 / threshold_multiplier:
            print(f"  가로형 액세서리가 세로로 서있음 - 회전 필요")
            return True, 90  # 반시계 방향 90도 회전
    
    # 그 외의 경우 회전 불필요
    return False, 0

# 액세서리 방향 감지 및 자동 회전 함수
def detect_and_fix_accessory_orientation(accessory_image, accessory_class):
    """
    액세서리 이미지의 방향을 감지하고 필요시 자동으로 회전합니다.
    accessory_image: PIL 이미지 또는 numpy 배열
    accessory_class: 액세서리 클래스명 (예: '귀도리')
    
    반환값: 올바른 방향으로 조정된 이미지, 회전 여부
    """
    # PIL 이미지로 변환
    if isinstance(accessory_image, np.ndarray):
        if accessory_image.shape[2] == 4:  # BGRA
            accessory_pil = Image.fromarray(cv2.cvtColor(accessory_image, cv2.COLOR_BGRA2RGBA))
        else:  # BGR
            accessory_pil = Image.fromarray(cv2.cvtColor(accessory_image, cv2.COLOR_BGR2RGB))
    else:
        accessory_pil = accessory_image
    
    # 이미지 크기 가져오기
    width, height = accessory_pil.size
    
    # 회전 필요 여부 판단
    should_rotate, rotation_angle = should_rotate_accessory(accessory_class, width, height)
    
    # 필요시 회전 적용
    if should_rotate:
        print(f"액세서리 '{accessory_class}' 회전 적용: {rotation_angle}도")
        rotated_image = accessory_pil.rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
        return rotated_image, True
    else:
        print(f"액세서리 '{accessory_class}' 회전 불필요")
        return accessory_pil, False

# 테스트 함수
def test_accessory_orientation_detector():
    """
    다양한 액세서리에 대한 방향 감지 테스트
    """
    # 테스트용 폴더 생성
    output_dir = "C:/capstone/data/validation/orientation_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 테스트할 액세서리 경로 및 클래스
    test_cases = [
        ("C:/capstone/data/validation/accessories_png/귀도리/s_0794_24573_620913.png", "귀도리"),
        # 다른 액세서리도 필요시 추가
    ]
    
    for i, (image_path, accessory_class) in enumerate(test_cases):
        # 이미지 로드
        try:
            accessory_img = Image.open(image_path).convert("RGBA")
            
            # 원본 이미지 저장
            original_save_path = os.path.join(output_dir, f"{i+1}_{accessory_class}_original.png")
            accessory_img.save(original_save_path)
            print(f"원본 이미지 저장: {original_save_path}")
            
            # 방향 감지 및 회전
            rotated_img, was_rotated = detect_and_fix_accessory_orientation(accessory_img, accessory_class)
            
            # 처리된 이미지 저장
            result_save_path = os.path.join(output_dir, f"{i+1}_{accessory_class}_{'rotated' if was_rotated else 'unchanged'}.png")
            rotated_img.save(result_save_path)
            print(f"결과 이미지 저장: {result_save_path}")
            
        except Exception as e:
            print(f"이미지 처리 오류: {image_path}, {e}")
    
    print("모든 테스트 완료!")

# 메인 함수
if __name__ == "__main__":
    # 방향 감지 테스트 실행
    test_accessory_orientation_detector()

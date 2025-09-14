import cv2
import numpy as np
import os
from PIL import Image
import math

# 원본 이미지와 귀도리 파일 경로
base_img_path = "C:/capstone/data/validation/images/s_0794_24573_620913.png"
accessory_path = "C:/capstone/data/validation/accessories_png/귀도리/s_0794_24573_620913.png"
label_path = base_img_path.replace("images", "labels").replace(".png", ".txt")

print(f"Base image path: {base_img_path}")
print(f"Accessory path: {accessory_path}")
print(f"Label path: {label_path}")

# 파일 존재 여부 확인
print(f"Base image exists: {os.path.exists(base_img_path)}")
print(f"Accessory exists: {os.path.exists(accessory_path)}")
print(f"Label exists: {os.path.exists(label_path)}")

# 이미지 로드
base_img = cv2.imread(base_img_path)
accessory_img = Image.open(accessory_path).convert("RGBA")

# 이미지 크기 출력
if base_img is not None:
    print(f"Base image shape: {base_img.shape}")
else:
    print("Failed to load base image")

print(f"Accessory size: {accessory_img.size}")

# 라벨 파일 내용 출력
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        labels = f.readlines()
    print(f"Labels: {labels}")
    
    # 귀 라벨 찾기
    ear_labels = [label for label in labels if label.strip().startswith('6 ')]
    print(f"Ear labels: {ear_labels}")
else:
    print("Label file not found")

# 수동으로 귀도리 합성하기
if base_img is not None and len(ear_labels) >= 1:
    # 이미지 크기
    img_height, img_width = base_img.shape[:2]
    
    # 귀 위치 계산 (첫 번째 귀 라벨 사용)
    parts = ear_labels[0].strip().split()
    ear_x = float(parts[1]) * img_width  # x center
    ear_y = float(parts[2]) * img_height  # y center
    ear_width = float(parts[3]) * img_width  # width
    ear_height = float(parts[4]) * img_height  # height
    
    print(f"Ear position: center=({ear_x:.1f}, {ear_y:.1f}), size=({ear_width:.1f}, {ear_height:.1f})")
    
    # 귀도리 크기 조절 및 회전
    scale_factor = 0.8
    new_width = int(ear_width * scale_factor * 2)  # 귀 두 개 크기 정도로
    aspect_ratio = accessory_img.width / accessory_img.height
    new_height = int(new_width / aspect_ratio)
    
    # 귀도리 크기 조절
    resized_accessory = accessory_img.resize((new_width, new_height), Image.LANCZOS)
    
    # 귀도리가 누워있는 상태이면 회전 (-90도)
    rotated_accessory = resized_accessory.rotate(-90, expand=True, resample=Image.BICUBIC)
    
    # 회전된 이미지 저장 (확인용)
    rotated_path = "C:/capstone/data/validation/rotated_earmuff.png"
    rotated_accessory.save(rotated_path)
    print(f"Rotated accessory saved to: {rotated_path}")
    
    # 회전된 귀도리 크기
    rot_width, rot_height = rotated_accessory.size
    print(f"Rotated accessory size: {rot_width}x{rot_height}")
    
    # 합성할 위치 계산
    paste_x = int(ear_x - rot_width/2)
    paste_y = int(ear_y - rot_height/2)
    
    # 이미지 경계 확인
    paste_x = max(0, min(paste_x, img_width - rot_width))
    paste_y = max(0, min(paste_y, img_height - rot_height))
    
    # 합성 (OpenCV와 PIL 간 변환)
    result_img = base_img.copy()
    
    # PIL 이미지를 NumPy 배열로 변환
    accessory_np = np.array(rotated_accessory)
    
    # ROI 영역 계산
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
        roi = result_img[paste_y:paste_y+roi_height, paste_x:paste_x+roi_width]
        
        # 알파 블렌딩 적용
        if roi.shape[:2] == rgb.shape[:2]:  # 치수 확인
            # OpenCV는 BGR, PIL은 RGB로 작업하므로 변환
            blended = roi * (1 - alpha) + cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) * alpha
            result_img[paste_y:paste_y+roi_height, paste_x:paste_x+roi_width] = blended
            
            # 결과 저장
            output_path = "C:/capstone/data/validation/test_ear_muffs_manual.jpg"
            cv2.imwrite(output_path, result_img)
            print(f"Manual test result saved to: {output_path}")
        else:
            print(f"Shape mismatch: roi={roi.shape}, rgb={rgb.shape}")
else:
    print("Cannot proceed with manual compositing due to missing data")

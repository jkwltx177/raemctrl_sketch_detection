import os
import random
import shutil
from pathlib import Path

def split_and_move_files(source_images_dir, source_labels_dir, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, split_ratio=0.8):
    # 디렉토리 경로 확인 및 생성
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.png', '.jpg'))]
    random.shuffle(image_files)  # 파일 목록을 무작위로 섞기

    # 학습과 검증 데이터 분리
    train_size = int(len(image_files) * split_ratio)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    # 학습 데이터 이동
    for img_file in train_files:
        # 이미지 파일 이동
        src_img = os.path.join(source_images_dir, img_file)
        dst_img = os.path.join(train_images_dir, img_file)
        shutil.copy2(src_img, dst_img)

        # 대응하는 레이블 파일 이동
        label_file = img_file.replace('.png', '.txt').replace('.jpg', '.txt')
        src_label = os.path.join(source_labels_dir, label_file)
        dst_label = os.path.join(train_labels_dir, label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

    # 검증 데이터 이동
    for img_file in val_files:
        # 이미지 파일 이동
        src_img = os.path.join(source_images_dir, img_file)
        dst_img = os.path.join(val_images_dir, img_file)
        shutil.copy2(src_img, dst_img)

        # 대응하는 레이블 파일 이동
        label_file = img_file.replace('.png', '.txt').replace('.jpg', '.txt')
        src_label = os.path.join(source_labels_dir, label_file)
        dst_label = os.path.join(val_labels_dir, label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

    print(f"Train set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")

# 경로 설정
source_images_dir = 'hat_output/images'  # hat_output의 images 폴더
source_labels_dir = 'hat_output/labels'  # hat_output의 labels 폴더
train_images_dir = 'C:/capstone/data/train/images'  # 기존 train 이미지 폴더
train_labels_dir = 'C:/capstone/data/train/labels'  # 기존 train 레이블 폴더
val_images_dir = 'C:/capstone/data/validation/images'  # 기존 validation 이미지 폴더
val_labels_dir = 'C:/capstone/data/validation/labels'  # 기존 validation 레이블 폴더

# 파일 분리 및 이동 실행
split_and_move_files(source_images_dir, source_labels_dir, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, split_ratio=0.8)
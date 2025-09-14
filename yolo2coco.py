import os
import json
import shutil
from PIL import Image
from pathlib import Path
import logging
import random
from sklearn.model_selection import train_test_split
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_primary_class(image_path, yolo_dataset_path):
    """
    각 이미지 파일에 해당하는 .txt 라벨 파일을 읽어 첫 번째 토큰(클래스 번호)를 반환.
    train 우선, 없으면 validation 폴더에서 찾음.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    for label_dir in [os.path.join(yolo_dataset_path, "train", "labels"),
                      os.path.join(yolo_dataset_path, "validation", "labels")]:
        label_file = os.path.join(label_dir, base + ".txt")
        if os.path.exists(label_file):
            try:
                with open(label_file, "r") as f:
                    line = f.readline().strip()
                    if line:
                        return line.split()[0]
            except Exception as e:
                logger.error(f"라벨 파일 읽기 오류 {label_file}: {e}")
    return None

def validate_bbox(x, y, w, h, img_width, img_height):
    """
    바운딩 박스가 이미지 경계 내에 있는지 확인하고 필요시 수정합니다.
    최소 크기와 최대 크기도 제한합니다.
    
    Returns:
        tuple: (is_valid, [x, y, w, h]) - 박스의 유효성과 수정된 좌표
    """
    # 작은 바운딩 박스 제외 (최소 10x10 픽셀)
    MIN_SIZE = 10
    if w < MIN_SIZE or h < MIN_SIZE:
        return False, [x, y, w, h]
    
    # 너무 큰 바운딩 박스 제외 (이미지 영역의 95% 이상)
    if w * h > 0.95 * img_width * img_height:
        return False, [x, y, w, h]
    
    # 이미지 경계 내로 클리핑
    x_new = max(0, min(x, img_width - 1))
    y_new = max(0, min(y, img_height - 1))
    
    # 박스가 이미지를 벗어나면 크기 조정
    w_new = min(w, img_width - x_new)
    h_new = min(h, img_height - y_new)
    
    # 박스 크기가 너무 작아졌는지 확인
    if w_new < MIN_SIZE or h_new < MIN_SIZE:
        return False, [x, y, w, h]
    
    return True, [x_new, y_new, w_new, h_new]

def yolo_to_coco_fixed(
    yolo_dataset_path, 
    coco_output_path, 
    class_names,
    split_ratios = {"train": 0.8, "valid": 0.1, "test": 0.1}
):
    """
    YOLO 형식 데이터셋을 COCO 형식으로 안정적으로 변환합니다.
    바운딩 박스 검증 및 ID 관리 개선 버전입니다.
    
    Args:
        yolo_dataset_path: YOLO 형식 데이터셋 경로
        coco_output_path: COCO 형식 데이터셋 저장 경로
        class_names: 클래스 이름 목록 (YOLO 클래스 ID 순서대로)
        split_ratios: 데이터 분할 비율
    """
    os.makedirs(coco_output_path, exist_ok=True)
    
    # train와 validation 폴더의 이미지 경로 모두 수집
    image_paths = []
    for split in ["train", "validation"]:
        image_dir = os.path.join(yolo_dataset_path, split, "images")
        if not os.path.exists(image_dir):
            logger.error(f"{split} 분할의 images 폴더를 찾을 수 없음")
            continue
        for ext in [".jpg", ".jpeg", ".png"]:
            image_paths.extend(list(Path(image_dir).glob(f"*{ext}")))
    
    image_paths = [str(p) for p in image_paths]
    total_images = len(image_paths)
    if total_images == 0:
        logger.error("이미지를 찾을 수 없음")
        return
    
    logger.info(f"전체 {total_images}개의 이미지를 찾았습니다.")
    
    # 각 이미지별 주 라벨 추출 (stratify를 위한 key)
    image_labels = []
    valid_images = []
    valid_labels = []
    
    for img_path in image_paths:
        primary_class = get_primary_class(img_path, yolo_dataset_path)
        # 라벨이 없는 이미지는 제외
        if primary_class is not None:
            valid_images.append(img_path)
            valid_labels.append(primary_class)
    
    logger.info(f"라벨이 있는 유효한 이미지: {len(valid_images)}개")
    
    # stratified split 적용
    try:
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            valid_images, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels
        )
        valid_paths, test_paths, valid_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
    except ValueError as e:
        logger.warning(f"Stratified split 실패: {e}, 일반 split 적용")
        # 클래스 분포가 고르지 않으면 일반 split 적용
        train_paths, temp_paths = train_test_split(
            valid_images, test_size=0.2, random_state=42
        )
        valid_paths, test_paths = train_test_split(
            temp_paths, test_size=0.5, random_state=42
        )
    
    logger.info(f"Split - Train: {len(train_paths)}, Valid: {len(valid_paths)}, Test: {len(test_paths)}")
    
    # 각 분할별로 COCO 변환 및 이미지 복사 처리
    splits = {
        "train": train_paths,
        "valid": valid_paths,
        "test": test_paths
    }
    
    # 카테고리 ID 확인 (1부터 연속적으로)
    categories = []
    for class_id, class_name in enumerate(class_names):
        categories.append({
            "id": class_id + 1,  # COCO는 1부터 시작
            "name": class_name,
            "supercategory": "none"
        })
    
    # 클래스 이름 매핑을 영어로 (선택사항)
    english_class_names = [
        'person_all', 'head', 'face', 'eye', 'nose', 'mouth', 'ear', 'hair',
        'neck', 'body', 'arm', 'hand', 'hat', 'glasses', 'eyebrow', 'beard',
        'open_mouth_(teeth)', 'muffler', 'tie', 'ribbon', 'ear_muff', 'earring',
        'necklace', 'ornament', 'headdress', 'jewel', 'cigarette'
    ]
    
    # 영어 클래스 이름으로 카테고리 재정의 (선택사항)
    categories = []
    for class_id, (korean_name, english_name) in enumerate(zip(class_names, english_class_names)):
        categories.append({
            "id": class_id + 1,
            "name": english_name,  # 영어 이름 사용
            "supercategory": "none"
        })
    
    invalid_bbox_count = 0
    fixed_bbox_count = 0
    
    for split_name, split_images in splits.items():
        output_dir = os.path.join(coco_output_path, split_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # COCO JSON 구조 초기화
        coco_dict = {
            "info": {
                "description": f"COCO format {split_name} dataset converted from YOLO format",
                "url": "",
                "version": "1.0",
                "year": 2023,
                "contributor": "Converter Script",
                "date_created": "2023"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": categories  # 미리 정의한 카테고리 사용
        }
        
        annotation_id = 0
        for image_id, image_path in enumerate(split_images):
            image_filename = os.path.basename(image_path)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                logger.error(f"이미지 열기 오류 {image_path}: {e}")
                continue
            
            # COCO JSON에 이미지 정보 추가
            coco_dict["images"].append({
                "id": image_id,  # 명확하게 관리된 이미지 ID
                "width": width,
                "height": height,
                "file_name": image_filename,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })
            
            # 이미지를 출력 디렉토리로 복사
            try:
                shutil.copy(image_path, os.path.join(output_dir, image_filename))
            except Exception as e:
                logger.error(f"이미지 복사 오류 {image_path}: {e}")
            
            # 라벨 파일 찾기: train 우선, 없으면 validation에서 찾음
            label_filename = os.path.splitext(image_filename)[0] + ".txt"
            label_path_train = os.path.join(yolo_dataset_path, "train", "labels", label_filename)
            label_path_valid = os.path.join(yolo_dataset_path, "validation", "labels", label_filename)
            label_path = None
            
            if os.path.exists(label_path_train):
                label_path = label_path_train
            elif os.path.exists(label_path_valid):
                label_path = label_path_valid
            
            if label_path and os.path.exists(label_path):
                try:
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    yolo_class = int(parts[0])
                                    
                                    # 클래스 ID가 유효한지 확인
                                    if yolo_class < 0 or yolo_class >= len(class_names):
                                        logger.warning(f"유효하지 않은 클래스 ID: {yolo_class} in {label_path}")
                                        continue
                                    
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    bbox_width = float(parts[3])
                                    bbox_height = float(parts[4])
                                    
                                    # 정규화된 좌표를 절대 좌표로 변환
                                    x = (x_center - bbox_width / 2) * width
                                    y = (y_center - bbox_height / 2) * height
                                    w_abs = bbox_width * width
                                    h_abs = bbox_height * height
                                    
                                    # 바운딩 박스 검증 및 수정
                                    is_valid, [x_new, y_new, w_new, h_new] = validate_bbox(
                                        x, y, w_abs, h_abs, width, height
                                    )
                                    
                                    if not is_valid:
                                        invalid_bbox_count += 1
                                        continue
                                    
                                    if [x_new, y_new, w_new, h_new] != [x, y, w_abs, h_abs]:
                                        fixed_bbox_count += 1
                                    
                                    # 유효한 어노테이션 추가
                                    coco_dict["annotations"].append({
                                        "id": annotation_id,
                                        "image_id": image_id,  # 이미지 ID와 일치시킴
                                        "category_id": yolo_class + 1,  # COCO는 1부터 시작
                                        "bbox": [x_new, y_new, w_new, h_new],
                                        "area": w_new * h_new,
                                        "segmentation": [],
                                        "iscrowd": 0
                                    })
                                    annotation_id += 1
                                except ValueError as e:
                                    logger.error(f"레이블 라인 파싱 오류 {line} in {label_path}: {e}")
                except Exception as e:
                    logger.error(f"레이블 파일 읽기 오류 {label_path}: {e}")
        
        # COCO JSON 저장 (ASCII 인코딩 사용)
        json_output_path = os.path.join(output_dir, "_annotations.coco.json")
        with open(json_output_path, "w") as f:
            json.dump(coco_dict, f, indent=4, ensure_ascii=True)
        
        logger.info(f"{split_name} 분할 완료: {len(coco_dict['images'])}개 이미지, {len(coco_dict['annotations'])}개 어노테이션")
    
    logger.info(f"변환 완료: {invalid_bbox_count}개 잘못된 바운딩 박스 제외, {fixed_bbox_count}개 바운딩 박스 수정")
    
    # 클래스 매핑 정보 저장 (참조용)
    mapping_info = {
        "class_mapping": {k: v for k, v in zip(class_names, english_class_names)}
    }
    with open(os.path.join(coco_output_path, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping_info, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    yolo_dataset_path = "C:/capstone/data/"
    coco_output_path = "C:/capstone/data/rf-detr_fixed_data/"
    
    # 실제 사용 클래스 이름
    class_names = [
        '사람전체', '머리', '얼굴', '눈', '코', '입', '귀', '머리카락', '목', '상체',
        '팔', '손', '모자', '안경', '눈썹', '수염', '벌린입(치아)', '목도리', '넥타이', '리본', 
        '귀도리', '귀걸이', '목걸이', '장식', '머리장식', '보석', '담배'
    ]
    
    yolo_to_coco_fixed(
        yolo_dataset_path, 
        coco_output_path, 
        class_names,
        split_ratios={"train": 0.8, "valid": 0.1, "test": 0.1}
    )
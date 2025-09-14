import os
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_coco_json_class_names(
    coco_dataset_path,
    class_name_mapping,
    output_mapping_file=None
):
    """
    이미 생성된 COCO JSON 파일의 클래스 이름을 영어로 변환하는 함수
    
    Args:
        coco_dataset_path: COCO 데이터셋 경로 (train, valid, test 폴더 포함)
        class_name_mapping: 원본 클래스 이름과 새 영어 클래스 이름 매핑 딕셔너리
        output_mapping_file: 원본 클래스 이름과 변경된 클래스 이름의 매핑 정보를 저장할 파일 경로
    """
    
    # 폴더 확인
    splits = ["train", "valid", "test"]
    found_splits = []
    
    for split in splits:
        split_path = os.path.join(coco_dataset_path, split)
        json_path = os.path.join(split_path, "_annotations.coco.json")
        
        if os.path.exists(json_path):
            found_splits.append((split, json_path))
        else:
            logger.warning(f"{json_path} 파일을 찾을 수 없습니다.")
    
    if not found_splits:
        logger.error("수정할 COCO JSON 파일을 찾을 수 없습니다.")
        return
    
    logger.info(f"발견된 분할: {[s[0] for s in found_splits]}")
    
    # 매핑 정보 저장 (나중에 참조용)
    if output_mapping_file:
        with open(output_mapping_file, "w", encoding="utf-8") as f:
            json.dump(class_name_mapping, f, indent=4, ensure_ascii=False)
        logger.info(f"클래스 매핑 정보가 {output_mapping_file}에 저장되었습니다.")
    
    # 각 JSON 파일 처리
    for split, json_path in found_splits:
        logger.info(f"{split} 분할의 JSON 파일 처리 중...")
        
        try:
            # JSON 파일 읽기
            with open(json_path, "r", encoding="utf-8") as f:
                coco_data = json.load(f)
            
            # 카테고리 이름 변경
            for category in coco_data["categories"]:
                original_name = category["name"]
                if original_name in class_name_mapping:
                    category["name"] = class_name_mapping[original_name]
                    logger.info(f"클래스 이름 변경: {original_name} -> {class_name_mapping[original_name]}")
            
            # 수정된 JSON 저장 (ASCII로 저장)
            with open(json_path, "w") as f:
                json.dump(coco_data, f, indent=4, ensure_ascii=True)
            
            logger.info(f"{split} 분할의 JSON 파일 수정 완료")
            
        except Exception as e:
            logger.error(f"{json_path} 처리 중 오류 발생: {e}")

# 사용 예시
if __name__ == "__main__":
    coco_dataset_path = "C:/capstone/data/rf-detr_data/"  # COCO 형식 데이터셋 경로
    mapping_file = "C:/capstone/data/rf-detr_data/test/coco.json"  # 매핑 정보 파일
    
    # 한글 -> 영어 클래스 이름 매핑
    class_name_mapping = {
        "사람전체": "person_all",
        "머리": "head",
        "얼굴": "face", 
        "눈": "eye", 
        "코": "nose", 
        "입": "mouth", 
        "귀": "ear", 
        "머리카락": "hair",
        "목": "neck", 
        "상체": "body",
        "팔": "arm", 
        "손": "hand", 
        "모자": "hat", 
        "안경": "glasses", 
        "눈썹": "eyebrow", 
        "수염": "beard",
        "벌린입(치아)": "open_mouth_(teeth)", 
        "목도리": "muffler", 
        "넥타이": "tie", 
        "리본": "ribbon", 
        "귀도리": "ear_muff", 
        "귀걸이": "earring", 
        "목걸이": "necklace", 
        "장식": "ornament", 
        "머리장식": "headdress", 
        "보석": "jewel", 
        "담배": "cigarette"
    }
    
    fix_coco_json_class_names(
        coco_dataset_path,
        class_name_mapping,
        output_mapping_file=mapping_file
    )
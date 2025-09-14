import os
import json
import torch
import multiprocessing
from pathlib import Path
import numpy as np
import collections
import shutil

# 환경 변수 설정
# Secrets removed for security
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def fix_coco_dataset(dataset_path):
    """COCO 데이터셋의 유효성을 철저히 검사하고 문제가 있는 어노테이션을 수정"""
    print(f"데이터셋 철저한 유효성 검사 중... ({dataset_path})")
    
    splits = ["train", "valid", "test"]
    for split in splits:
        json_path = os.path.join(dataset_path, split, "_annotations.coco.json")
        if not os.path.exists(json_path):
            print(f"경고: {json_path} 파일을 찾을 수 없습니다.")
            continue
        
        # JSON 파일 로드
        with open(json_path, "r") as f:
            coco_data = json.load(f)
        
        print(f"\n===== {split} 분할 분석 =====")
        print(f"카테고리 수: {len(coco_data['categories'])}")
        print(f"이미지 수: {len(coco_data['images'])}")
        print(f"어노테이션 수: {len(coco_data['annotations'])}")
        
        # 카테고리 ID 분석 - ID가 연속적이고 1부터 시작하는지 확인
        cat_ids = [cat["id"] for cat in coco_data["categories"]]
        cat_ids.sort()
        expected_ids = list(range(1, len(cat_ids) + 1))
        
        if cat_ids != expected_ids:
            print(f"경고: 카테고리 ID가 1부터 연속적이지 않습니다: {cat_ids}")
            print(f"카테고리 ID를 수정합니다...")
            
            # 카테고리 ID 수정
            id_mapping = {old_id: new_id for old_id, new_id in zip(cat_ids, expected_ids)}
            for cat in coco_data["categories"]:
                if cat["id"] in id_mapping:
                    cat["id"] = id_mapping[cat["id"]]
            
            # 어노테이션의 category_id도 수정
            for ann in coco_data["annotations"]:
                if ann["category_id"] in id_mapping:
                    ann["category_id"] = id_mapping[ann["category_id"]]
            
            print(f"카테고리 ID 수정 완료: {expected_ids}")
        
        # 이미지 정보 저장
        images_info = {img["id"]: {"width": img["width"], "height": img["height"], "file_name": img["file_name"]} 
                       for img in coco_data["images"]}
        
        # 유효하지 않은 이미지 ID를 가진 어노테이션 검사
        valid_img_ids = set(images_info.keys())
        invalid_img_ids = [ann["image_id"] for ann in coco_data["annotations"] 
                           if ann["image_id"] not in valid_img_ids]
        
        if invalid_img_ids:
            print(f"경고: {len(invalid_img_ids)}개의 어노테이션이 유효하지 않은 이미지 ID를 참조합니다.")
        
        # 카테고리 ID 범위 검사
        valid_cat_ids = set(cat["id"] for cat in coco_data["categories"])
        invalid_cat_ids = [ann["category_id"] for ann in coco_data["annotations"] 
                           if ann["category_id"] not in valid_cat_ids]
        
        if invalid_cat_ids:
            cat_id_counts = collections.Counter(invalid_cat_ids)
            print(f"경고: {len(invalid_cat_ids)}개의 어노테이션이 유효하지 않은 카테고리 ID를 가집니다.")
            print(f"유효하지 않은 카테고리 ID 빈도: {dict(cat_id_counts.most_common(10))}")
        
        # 바운딩 박스 크기 분석
        bbox_areas = [ann["area"] for ann in coco_data["annotations"]]
        if bbox_areas:
            print(f"바운딩 박스 면적 - 최소: {min(bbox_areas):.2f}, 최대: {max(bbox_areas):.2f}, 평균: {sum(bbox_areas)/len(bbox_areas):.2f}")
        
        # 문제가 있는 어노테이션 수정
        fixed_annotations = []
        removed_count = 0
        fixed_count = 0
        
        max_cat_id = max(cat["id"] for cat in coco_data["categories"])
        
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            
            # 유효하지 않은 이미지 ID 참조 제거
            if img_id not in valid_img_ids:
                removed_count += 1
                continue
                
            img_info = images_info[img_id]
            img_width = img_info["width"]
            img_height = img_info["height"]
            
            # 카테고리 ID 수정
            if ann["category_id"] not in valid_cat_ids:
                # 범위 밖의 카테고리 ID는 1로 대체 (첫 번째 클래스)
                ann["category_id"] = 1
                fixed_count += 1
            
            # 바운딩 박스가 없거나 형식이 잘못된 경우 제거
            if "bbox" not in ann or len(ann["bbox"]) != 4:
                removed_count += 1
                continue
                
            # 바운딩 박스 유효성 검사 및 수정
            bbox = ann["bbox"]
            try:
                x, y, w, h = map(float, bbox)
            except (ValueError, TypeError):
                removed_count += 1
                continue
                
            # 너무 작은 바운딩 박스 제거
            if w < 1 or h < 1:
                removed_count += 1
                continue
                
            # 좌표가 이미지 밖으로 나가는 경우 수정
            fixed_box = False
            if x < 0:
                w += x  # 너비 감소
                x = 0
                fixed_box = True
            if y < 0:
                h += y  # 높이 감소
                y = 0
                fixed_box = True
            if x + w > img_width:
                w = img_width - x
                fixed_box = True
            if y + h > img_height:
                h = img_height - y
                fixed_box = True
                
            # 수정된 바운딩 박스가 유효한지 최종 확인
            if w <= 0 or h <= 0:
                removed_count += 1
                continue
                
            if fixed_box:
                fixed_count += 1
                
            # 수정된 바운딩 박스 적용
            ann["bbox"] = [x, y, w, h]
            ann["area"] = w * h
            fixed_annotations.append(ann)
        
        # 수정된 어노테이션으로 업데이트
        original_count = len(coco_data["annotations"])
        coco_data["annotations"] = fixed_annotations
        
        print(f"{split} 분할: {original_count}개 중 {removed_count}개 제거, {fixed_count}개 수정됨, {len(fixed_annotations)}개 유지")
        
        # 수정된 데이터 저장
        with open(json_path, "w") as f:
            json.dump(coco_data, f, indent=4)
    
    print("\n데이터셋 유효성 검사 및 수정 완료")

def patch_rfdetr_matcher():
    """RF-DETR matcher 모듈의 인덱싱 오류를 방지하는 패치를 적용합니다"""
    try:
        import rfdetr.models.matcher as matcher_module
        original_forward = matcher_module.HungarianMatcher.forward
        
        # 패치된 forward 메서드
        def safe_forward(self, outputs, targets, group_detr=None):
            try:
                return original_forward(self, outputs, targets, group_detr)
            except RuntimeError as e:
                if "index out of bounds" in str(e) or "device-side assert triggered" in str(e):
                    print("Matcher 오류 패치 적용 중...")
                    # 안전한 인덱싱을 적용한 대체 구현
                    with torch.no_grad():
                        bs, num_queries = outputs["pred_logits"].shape[:2]
                        
                        # 원본 코드가 인덱스 오류를 일으키면 안전한 인덱싱 사용
                        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
                        out_bbox = outputs["pred_boxes"].flatten(0, 1)
                        
                        # 각 이미지와 쿼리에 대한 인덱스 리스트 생성
                        batch_indices = []
                        for i, target in enumerate(targets):
                            if len(target["labels"]) == 0:
                                # 빈 타겟은 건너뛰기
                                continue
                                
                            # 안전한 크기로 제한하여 인덱스 오류 방지
                            tgt_ids = target["labels"].long()
                            num_classes = out_prob.shape[1]
                            tgt_ids = torch.clamp(tgt_ids, 0, num_classes-1)
                            
                            # 인덱스 미리 검증
                            valid_indices = []
                            for j in range(num_queries):
                                idx = i * num_queries + j
                                if idx < out_prob.shape[0]:
                                    valid_indices.append((i, j))
                                    
                            if valid_indices:
                                batch_indices.append((i, valid_indices, tgt_ids))
                        
                        # 빈 결과 반환
                        if not batch_indices:
                            return [(torch.tensor([], dtype=torch.int64), 
                                     torch.tensor([], dtype=torch.int64)) for _ in range(bs)]
                        
                        # 안전한 방식으로 인덱스 생성
                        indices = []
                        for i in range(bs):
                            # 기본적으로 빈 매칭
                            indices.append((torch.tensor([], dtype=torch.int64, device=out_prob.device),
                                           torch.tensor([], dtype=torch.int64, device=out_prob.device)))
                            
                        # 실제 매칭이 가능한 경우에만 처리
                        for i, valid_indices, tgt_ids in batch_indices:
                            if len(valid_indices) > 0 and len(tgt_ids) > 0:
                                # 가장 간단한 매칭: 첫 N개의 쿼리를 타겟에 할당
                                src_idx = torch.tensor([idx for _, idx in valid_indices[:len(tgt_ids)]], 
                                                      dtype=torch.int64, device=out_prob.device)
                                tgt_idx = torch.arange(len(tgt_ids), device=out_prob.device)
                                indices[i] = (src_idx, tgt_idx)
                                
                        return indices
                else:
                    # 다른 종류의 오류는 그대로 전파
                    raise
        
        # 패치 적용
        matcher_module.HungarianMatcher.forward = safe_forward
        print("RF-DETR matcher 패치 적용 완료")
        return True
    except Exception as e:
        print(f"RF-DETR matcher 패치 적용 실패: {e}")
        return False

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 데이터셋 경로
    dataset_path = "C:/capstone/data/rf-detr_data"
    
    # 데이터셋 유효성 검사 및 수정
    fix_coco_dataset(dataset_path)
    
    # Matcher 모듈 패치 적용
    patch_success = patch_rfdetr_matcher()
    
    # 텐서 이상 감지 활성화 (디버깅용)
    torch.autograd.set_detect_anomaly(True)
    
    from rfdetr import RFDETRBase
    
    # 최소 설정으로 모델 훈련
    print("\n최소 설정으로 훈련 시작...")
    model = RFDETRBase()
    
    try:
        model.train(
            dataset_dir=dataset_path,
            epochs=15,
            batch_size=1,  # 최소 배치 크기
            grad_accum_steps=1,  # 누적 없음
            lr=1e-4,
            num_workers=0,  # 멀티프로세싱 워커 비활성화
            seed=42,  # 재현 가능한 결과를 위한 시드 설정
            max_size=640,  # 큰 이미지 제한
            early_stopping=True,  # 조기 종료 활성화
            early_stopping_patience=3  # 3번의 에폭 동안 개선이 없으면 종료
        )
    except Exception as e:
        print(f"\n훈련 중 오류 발생: {e}")
        
        # 오류가 계속되면 더 엄격한 데이터셋 필터링 시도
        print("\n더 엄격한 데이터셋 필터링을 시도합니다...")
        
        # 문제가 있는 어노테이션을 더 적극적으로 제거하기 위한 서브셋 생성
        subset_path = "C:/capstone/data/rf-detr_subset"
        os.makedirs(subset_path, exist_ok=True)
        
        for split in ["train", "valid", "test"]:
            split_dir = os.path.join(dataset_path, split)
            subset_split_dir = os.path.join(subset_path, split)
            os.makedirs(subset_split_dir, exist_ok=True)
            
            json_path = os.path.join(split_dir, "_annotations.coco.json")
            subset_json_path = os.path.join(subset_split_dir, "_annotations.coco.json")
            
            if not os.path.exists(json_path):
                continue
            
            # JSON 로드
            with open(json_path, "r") as f:
                coco_data = json.load(f)
            
            # 카테고리 그대로 유지
            new_data = {
                "info": coco_data.get("info", {}),
                "licenses": coco_data.get("licenses", []),
                "categories": coco_data.get("categories", []),
                "images": [],
                "annotations": []
            }
            
            # 각 이미지마다 최대 5개의 객체만 유지 (복잡한 장면 제외)
            img_ann_counts = collections.Counter([ann["image_id"] for ann in coco_data["annotations"]])
            simple_imgs = {img_id for img_id, count in img_ann_counts.items() if 1 <= count <= 5}
            
            # 적합한 이미지만 선택 (단순한 장면)
            kept_imgs = []
            for img in coco_data["images"]:
                if img["id"] in simple_imgs:
                    kept_imgs.append(img)
                    # 이미지 파일 복사
                    src_path = os.path.join(split_dir, img["file_name"])
                    dst_path = os.path.join(subset_split_dir, img["file_name"])
                    if os.path.exists(src_path):
                        shutil.copy(src_path, dst_path)
            
            # 서브셋에 포함된 이미지 ID
            kept_img_ids = {img["id"] for img in kept_imgs}
            
            # 해당 이미지의 어노테이션 추가
            kept_anns = []
            for ann in coco_data["annotations"]:
                if ann["image_id"] in kept_img_ids:
                    kept_anns.append(ann)
            
            # 새 데이터 업데이트
            new_data["images"] = kept_imgs
            new_data["annotations"] = kept_anns
            
            print(f"{split} 서브셋: {len(kept_imgs)}개 이미지, {len(kept_anns)}개 어노테이션 선택됨")
            
            # 서브셋 저장
            with open(subset_json_path, "w") as f:
                json.dump(new_data, f, indent=4)
        
        print("\n서브셋 생성 완료. 더 작은 데이터셋으로 다시 시도합니다...")
        
        # 서브셋으로 재시도
        fix_coco_dataset(subset_path)
        
        model = RFDETRBase()
        model.train(
            dataset_dir=subset_path,
            epochs=15,
            batch_size=4,
            grad_accum_steps=4,
            lr=1e-4,
            #num_workers=0,
            #seed=42,
            #max_size=640,
            early_stopping=True,
            early_stopping_patience=3
        )
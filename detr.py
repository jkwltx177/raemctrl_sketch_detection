import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection
)
from torch.optim import AdamW
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 오프라인 모드 활성화
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# COCO 형식 데이터셋을 로드하는 사용자 정의 Dataset 클래스
class CocoDetectionDataset(Dataset):
    def __init__(self, img_dir, ann_file, processor, transforms=None):
        self.img_dir = img_dir
        self.processor = processor
        self.transforms = transforms
        
        # COCO 어노테이션 파일 로드
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        
        # 이미지 ID를 인덱스에 매핑
        self.ids = []
        self.id_to_img_map = {}
        for i, img in enumerate(self.coco['images']):
            img_id = img['id']
            self.ids.append(img_id)
            self.id_to_img_map[img_id] = img
        
        # 이미지별 어노테이션 그룹화
        self.img_to_anns = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        # 카테고리 ID 매핑
        self.cat_ids = {cat['id']: i for i, cat in enumerate(self.coco['categories'])}
        self.cat_names = {cat['id']: cat['name'] for cat in self.coco['categories']}
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.id_to_img_map[img_id]
        
        # 이미지 로드
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        # 어노테이션 가져오기
        anns = self.img_to_anns.get(img_id, [])
        
        # 바운딩 박스와 클래스 라벨 추출
        boxes = []
        classes = []
        
        for ann in anns:
            # COCO 형식 바운딩 박스 [x, y, width, height]
            bbox = ann['bbox']
            # 클래스 ID를 0부터 시작하는 인덱스로 변환
            class_id = self.cat_ids[ann['category_id']]
            
            boxes.append(bbox)
            classes.append(class_id)
        
        # Albumentations 변환 적용 (이미지가 numpy 배열 필요)
        if self.transforms and boxes:
            img_array = np.array(img)
            try:
                transformed = self.transforms(image=img_array, bboxes=boxes, class_labels=classes)
                img_tensor = transformed['image']
                boxes = transformed['bboxes']
                classes = transformed['class_labels']
            except Exception as e:
                print(f"변환 오류 (스킵): {e}")
                # 변환 실패 시 기본 변환 사용
                img_tensor = T.functional.to_tensor(img)
                boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
                classes = torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long)
        else:
            # 기본 변환
            img_tensor = T.functional.to_tensor(img)
            boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long)
        
        # DETR 프로세서용 타겟 준비
        target = {
            "boxes": boxes, 
            "labels": classes, 
            "image_id": torch.tensor([img_id], dtype=torch.long)
        }
        
        # DETR 프로세서 직접 구현 (프로세서 없이도 사용 가능)
        encoding = self._prepare_for_detr(img_tensor, target)
        
        return encoding
    
    def _prepare_for_detr(self, img_tensor, target):
        """프로세서 없이 DETR 입력 형식으로 변환"""
        # 이미지 정규화 (ImageNet 통계)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = normalize(img_tensor)
        
        # 타겟 변환
        boxes = target["boxes"]
        if isinstance(boxes, torch.Tensor):
            # COCO 형식 [x, y, w, h]에서 DETR 형식 [cx, cy, w, h]로 변환
            boxes_cxcywh = boxes.clone()
            if len(boxes) > 0:
                boxes_cxcywh[:, 0] = boxes[:, 0] + boxes[:, 2] / 2  # 중심 x
                boxes_cxcywh[:, 1] = boxes[:, 1] + boxes[:, 3] / 2  # 중심 y
        else:
            # 리스트인 경우 텐서로 변환 후 처리
            if len(boxes) > 0:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                boxes_cxcywh = boxes.clone()
                boxes_cxcywh[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
                boxes_cxcywh[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            else:
                boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32)
        
        # 라벨 변환
        class_labels = target["labels"]
        if not isinstance(class_labels, torch.Tensor):
            class_labels = torch.tensor(class_labels, dtype=torch.long) if class_labels else torch.zeros((0,), dtype=torch.long)
        
        return {
            "pixel_values": img_tensor,
            "labels": {
                "class_labels": class_labels,
                "boxes": boxes_cxcywh
            }
        }

# 인스턴스화를 위한 collate 함수
def collate_fn(batch):
    # 입력과 라벨 분리
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # 라벨 수집
    labels = [{
        'class_labels': item['labels']['class_labels'],
        'boxes': item['labels']['boxes'],
    } for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }

# 간단한 DETR 프로세서 클래스 (오프라인 용)
class SimpleDetrProcessor:
    def __init__(self):
        # DETR의 기본 전처리 파라미터
        self.size = (800, 1066)  # DETR 기본 크기
        
    def __call__(self, images=None, annotations=None, return_tensors=None):
        if not isinstance(images, list):
            images = [images]
            
        processed_images = []
        processed_annotations = []
        
        for i, image in enumerate(images):
            # PIL 이미지를 텐서로 변환
            if isinstance(image, Image.Image):
                # 크기 조정
                image = image.resize(self.size)
                # 텐서로 변환
                image_tensor = T.functional.to_tensor(image)
                # 정규화
                image_tensor = T.functional.normalize(
                    image_tensor,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                processed_images.append(image_tensor)
            else:
                # 이미 텐서인 경우
                processed_images.append(image)
            
            # 어노테이션 처리
            if annotations:
                ann = annotations[i] if i < len(annotations) else None
                processed_annotations.append(ann)
        
        # 배치 차원 추가
        if return_tensors == "pt":
            batch = {
                "pixel_values": torch.stack(processed_images)
            }
            
            if annotations:
                batch["labels"] = processed_annotations
            
            return batch
        
        return processed_images

# 수동으로 만든 간단한 DETR 모델
class SimpleDetrForObjectDetection:
    def __init__(self, num_classes=91):
        # 기본 DETR ResNet-50 모델
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        
        # 클래스 수 변경
        if num_classes != 91:
            # 원래 클래스 임베딩 얻기
            original_class_embed = self.model.class_embed
            # 새 클래스 임베딩 초기화
            self.model.class_embed = torch.nn.Linear(original_class_embed.in_features, num_classes)
            # 가중치 초기화
            torch.nn.init.xavier_uniform_(self.model.class_embed.weight)
        
        # 구성 설정
        self.config = type('obj', (object,), {
            'num_labels': num_classes,
            'id2label': {i: f'class_{i}' for i in range(num_classes)},
            'label2id': {f'class_{i}': i for i in range(num_classes)},
        })
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def train(self):
        self.model.train()
        return self
    
    def eval(self):
        self.model.eval()
        return self
    
    def __call__(self, pixel_values=None, labels=None):
        # 입력 형식 변환
        if labels:
            targets = []
            for i, label in enumerate(labels):
                target = {
                    'labels': label['class_labels'],
                    'boxes': label['boxes']
                }
                targets.append(target)
            
            # DETR 형식으로 변환
            loss_dict = self.model(pixel_values, targets)
            return type('obj', (object,), {'loss': sum(loss_dict.values())})
        else:
            # 추론 모드
            outputs = self.model(pixel_values)
            return outputs

# 메인 코드
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 1. GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 장치: {device}")
    if device.type == 'cuda':
        print(f"GPU 모델: {torch.cuda.get_device_name(0)}")
        print(f"가용 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 2. 데이터셋 경로 설정
    base_path = "C:/capstone/data/rf-detr_fixed_data"
    train_img_dir = os.path.join(base_path, "train")
    val_img_dir = os.path.join(base_path, "valid")
    test_img_dir = os.path.join(base_path, "test")
    
    train_ann_file = os.path.join(train_img_dir, "_annotations.coco.json")
    val_ann_file = os.path.join(val_img_dir, "_annotations.coco.json")
    test_ann_file = os.path.join(test_img_dir, "_annotations.coco.json")
    
    # 3. 데이터 증강 설정
    try:
        train_transform = A.Compose([
            A.Resize(height=800, width=1066),  # 기본 크기만 사용
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        
        val_transform = A.Compose([
            A.Resize(height=800, width=1066),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    except Exception as e:
        print(f"알부멘테이션 설정 오류: {e}")
        # 대체 변환
        train_transform = None
        val_transform = None
    
    # 4. 프로세서 초기화 (오프라인 버전 사용)
    print("오프라인 프로세서 초기화 중...")
    processor = SimpleDetrProcessor()
    
    # 5. 데이터셋 로드
    print("데이터셋 로드 중...")
    
    try:
        train_dataset = CocoDetectionDataset(
            img_dir=train_img_dir,
            ann_file=train_ann_file,
            processor=processor,
            transforms=train_transform
        )
        
        val_dataset = CocoDetectionDataset(
            img_dir=val_img_dir,
            ann_file=val_ann_file,
            processor=processor,
            transforms=val_transform
        )
        
        print(f"훈련 데이터셋 크기: {len(train_dataset)}, 검증 데이터셋 크기: {len(val_dataset)}")
    except Exception as e:
        print(f"데이터셋 로드 오류: {e}")
        raise
    
    # 6. 데이터 로더 설정
    batch_size = 4  # 시작은 작은 배치로
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 7. 모델 로드 및 설정
    print("모델 로드 중 (오프라인 모드)...")
    
    # COCO 데이터셋에서 클래스 수 확인
    with open(train_ann_file, 'r') as f:
        coco_data = json.load(f)
    num_classes = len(coco_data['categories'])
    print(f"클래스 수: {num_classes}")
    
    # 오프라인 모델 로드
    try:
        local_model_path = "./detr-resnet-50"  # 수동으로 다운로드한 모델 경로
        if os.path.exists(local_model_path):
            model = DetrForObjectDetection.from_pretrained(
                local_model_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                local_files_only=True
            )
            print("로컬 모델 로드 성공")
        else:
            print("로컬 모델을 찾을 수 없어 torch.hub에서 로드합니다...")
            model = SimpleDetrForObjectDetection(num_classes=num_classes)
            print("torch.hub에서 모델 로드 성공")
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        print("대체 방법으로 torch.hub에서 로드 시도...")
        try:
            model = SimpleDetrForObjectDetection(num_classes=num_classes)
            print("torch.hub에서 모델 로드 성공")
        except Exception as e2:
            print(f"대체 모델 로드도 실패: {e2}")
            raise
    
    # ID2라벨 매핑 설정
    try:
        model.config.id2label = {i: cat['name'] for i, cat in enumerate(coco_data['categories'])}
        model.config.label2id = {cat['name']: i for i, cat in enumerate(coco_data['categories'])}
    except:
        print("모델 구성 업데이트 건너뜀")
    
    # 8. 모델을 GPU로 이동
    model = model.to(device)
    
    # 9. 훈련 설정
    learning_rate = 5e-5  # 더 낮은 학습률로 시작
    weight_decay = 1e-4
    epochs = 15  # 에폭 수 줄임
    
    # 옵티마이저 설정
    try:
        # 백본 파라미터와 다른 파라미터 분리
        backbone_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        # 백본에는 더 낮은 학습률 적용
        optimizer = AdamW([
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": other_params, "lr": learning_rate}
        ], lr=learning_rate, weight_decay=weight_decay)
    except:
        # 간단한 방법으로 옵티마이저 설정
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 스케줄러 설정
    num_training_steps = epochs * len(train_dataloader)
    warmup_steps = int(0.1 * num_training_steps)
    
    try:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    except:
        # 기본 스케줄러 사용
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=1e-6
        )
    
    # 10. 훈련 루프
    output_dir = "./detr_finetuned_offline"
    os.makedirs(output_dir, exist_ok=True)
    
    # 손실 추적 클래스
    class AverageMeter:
        def __init__(self):
            self.reset()
            
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
            
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
    print(f"훈련 시작...")
    best_val_loss = float('inf')
    
    try:
        for epoch in range(epochs):
            # 훈련 단계
            model.train()
            train_loss = AverageMeter()
            
            for i, batch in enumerate(train_dataloader):
                # 데이터를 GPU로 이동
                pixel_values = batch['pixel_values'].to(device)
                labels = [{k: v.to(device) for k, v in label.items()} for label in batch['labels']]
                
                # 순방향 패스
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                # 역전파
                optimizer.zero_grad()
                loss.backward()
                
                # 모델 업데이트
                optimizer.step()
                scheduler.step()
                
                # 손실 업데이트
                train_loss.update(loss.item(), pixel_values.size(0))
                
                # 로깅
                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
            
            # 검증 단계
            model.eval()
            val_loss = AverageMeter()
            
            with torch.no_grad():
                for batch in val_dataloader:
                    pixel_values = batch['pixel_values'].to(device)
                    labels = [{k: v.to(device) for k, v in label.items()} for label in batch['labels']]
                    
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    
                    val_loss.update(loss.item(), pixel_values.size(0))
            
            # 에폭 요약
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.avg:.4f}, Val Loss: {val_loss.avg:.4f}")
            
            # 최상의 모델 저장
            if val_loss.avg < best_val_loss:
                best_val_loss = val_loss.avg
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                print(f"새로운 최상의 모델 저장: {val_loss.avg:.4f}")
            
            # 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, os.path.join(output_dir, f"checkpoint-{epoch+1}.pth"))
        
        # 최종 모델 저장
        torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
        print("최종 모델 저장 완료")
        print("파인튜닝 완료!")
        
    except KeyboardInterrupt:
        print("훈련 중단됨")
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
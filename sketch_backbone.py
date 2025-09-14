from datetime import datetime
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
import os
import sys
sys.setrecursionlimit(10000)  # 기본값은 1000


# ResNet50BackboneAdapter 클래스 정의 (이전과 동일)
class ResNet50BackboneAdapter(nn.Module):
    def __init__(self, resnet_model):
        super().__init__()
        # ResNet 컴포넌트 저장
        self.stem = nn.Sequential(
            resnet_model.conv1,      # 7x7 Conv
            resnet_model.bn1,        # BatchNorm
            resnet_model.relu,       # ReLU
            resnet_model.maxpool     # MaxPool
        )
        self.layer1 = resnet_model.layer1  # 3 Bottleneck blocks (256 채널)
        self.layer2 = resnet_model.layer2  # 4 Bottleneck blocks (512 채널)
        self.layer3 = resnet_model.layer3  # 6 Bottleneck blocks (1024 채널)
        self.layer4 = resnet_model.layer4  # 3 Bottleneck blocks (2048 채널)
        
        # 채널 수 조정을 위한 1x1 컨볼루션 (ResNet -> YOLO)
        self.p5_conv = nn.Conv2d(2048, 1024, kernel_size=1)
        
        # YOLO의 SPPF 및 C2PSA 레이어
        self.sppf = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )
        
        self.c2psa = nn.Sequential(
            # YOLO의 C2PSA 구현 (간소화)
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        # 각 단계별 피처맵 추출
        
        # Stem
        x = self.stem(x)  # YOLO의 0-1 레이어 대체
        
        # Layer 1 - P2 (256 채널)
        p2 = self.layer1(x)  # YOLO의 2 레이어 대체
        
        # Layer 2 - P3 (512 채널)
        p3 = self.layer2(p2)  # YOLO의 3-4 레이어 대체
        
        # Layer 3 - P4 (1024 채널)
        p4 = self.layer3(p3)  # YOLO의 5-6 레이어 대체
        
        # Layer 4 - P5 (2048 채널)
        p5_orig = self.layer4(p4)  # YOLO의 7-8 레이어 대체
        
        # 채널 수 조정 (2048 -> 1024)
        p5 = self.p5_conv(p5_orig)
        
        # SPPF
        sppf_out = self.sppf(p5)  # YOLO의 9 레이어 대체
        
        # C2PSA
        c2psa_out = self.c2psa(sppf_out)  # YOLO의 10 레이어 대체
        
        # 모든 피처맵 반환
        return [p2, p3, p4, p5, sppf_out, c2psa_out]

# 새로운 클래스 기반 접근법: YOLO 모델 확장
class YOLOWithResNetBackbone(nn.Module):
    def __init__(self, yolo_model, resnet_model):
        super().__init__()
        # 백본 어댑터 생성
        self.backbone = ResNet50BackboneAdapter(resnet_model)
        
        # 원본 YOLO 모델 저장
        # self.yolo_model = yolo_model

         # YOLO 모델에서 필요한 부분만 직접 저장 (순환 참조 방지)
        self.detect_layer = yolo_model.model.model[-1]  # Detect 레이어만 저장
        
        # YOLO11 헤드 파트를 저장 (인덱스 11부터 끝까지)
        self.head_modules = nn.ModuleList()
        for i in range(11, len(yolo_model.model.model)):
            if isinstance(yolo_model.model.model[i], nn.Module):
                self.head_modules.append(yolo_model.model.model[i])
            else:
                # 리스트 타입 모듈 (Concat, Detect 등) 처리
                self.head_modules.append(yolo_model.model.model[i])
    
    def forward(self, x):
        # 백본에서 피처맵 추출
        features = self.backbone(x)
        
        # 피처맵 매핑 (ResNet -> YOLO)
        feature_map = {
            2: features[0],  # P2
            4: features[1],  # P3
            6: features[2],  # P4
            8: features[3],  # P5
            9: features[4],  # SPPF
            10: features[5]  # C2PSA
        }
        
        # 헤드 처리
        for i, module in enumerate(self.head_modules):
            # 인덱스 계산
            idx = i + 11
            
            # 모듈 타입에 따른 처리
            if isinstance(module, nn.Module):
                # 일반 모듈 (Conv, C3k2 등)
                x = module(x)
            elif isinstance(module, list):
                # Concat 또는 Detect 모듈의 경우
                from_indices = module[0]
                if isinstance(from_indices, list):
                    # 여러 레이어에서 입력 받는 경우 (Concat, Detect)
                    inputs = []
                    for idx in from_indices:
                        if idx == -1:
                            inputs.append(x)
                        else:
                            # 인덱스가 참조하는 피처맵 가져오기
                            if idx in feature_map:
                                inputs.append(feature_map[idx])
                            else:
                                # 이전 헤드 레이어 출력
                                inputs.append(feature_map[idx + 11])
                    
                    # 모듈 적용
                    module_type = module[2]
                    if module_type == 'Concat':
                        dim = module[3][0]
                        x = torch.cat(inputs, dim)
                    elif module_type == 'Detect':
                        # Detect 레이어는 특별 처리 필요
                        x = self.detect_layer(inputs)
                else:
                    # 단일 입력 레이어
                    idx = from_indices
                    if idx == -1:
                        # 이전 레이어 출력 사용
                        pass
                    else:
                        # 인덱스가 참조하는 피처맵 가져오기
                        if idx in feature_map:
                            x = feature_map[idx]
                        else:
                            # 이전 헤드 레이어 출력
                            x = feature_map[idx + 11]
            
            # 현재 레이어 출력 저장
            feature_map[idx] = x
        
        return x

# 새로운 방식으로 교체하는 함수
def create_model_with_resnet_backbone(yolo_model, resnet_backbone):
    """
    YOLO 모델과 ResNet50 백본을 통합한 새 모델을 생성합니다.
    
    Args:
        yolo_model: 기존 YOLO 모델
        resnet_backbone: 사전학습된 ResNet50 모델
    
    Returns:
        새로운 통합 모델
    """
    # 새 모델 생성
    new_model = YOLOWithResNetBackbone(yolo_model, resnet_backbone)
    
    # YOLO 모델의 속성을 적절히 복사
    yolo_model.model.backbone_model = new_model
    
    # forward 메서드 대체 (클래스 기반으로 변경되어 직렬화 가능)
    yolo_model.model.forward = new_model.forward
    
    return yolo_model

# 메인 실행 코드 - 백본 생성 및 저장
def main():
    # 1. ResNet50_SIN 체크포인트 로드
    print("ResNet50_SIN 체크포인트 로드 중...")
    checkpoint = torch.load('weights/resnet50_SIN.pth')
    
    # 2. 체크포인트 구조 확인 및 실제 state_dict 추출
    print("체크포인트 키:", checkpoint.keys())
    
    # 체크포인트에서 모델 state_dict 추출
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("모델 state_dict를 체크포인트에서 추출했습니다.")
    else:
        state_dict = checkpoint  # 이미 state_dict일 수도 있음
        print("체크포인트를 state_dict로 사용합니다.")
    
    # 3. ResNet50 모델 생성 및 가중치 로드
    print("ResNet50 모델 생성 및 가중치 로드 중...")
    resnet50 = models.resnet50(weights=None)  # 빈 ResNet50 모델 생성
    
    # 가중치 키가 'module.xxx' 형식인지 확인하고 필요시 수정
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # 'module.' 접두사 제거
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # 수정된 state_dict 로드
    try:
        resnet50.load_state_dict(new_state_dict)
        print("ResNet50 모델에 가중치가 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"가중치 로드 실패: {e}")
        print("모델 구조와 가중치 간의 불일치가 있을 수 있습니다.")
        print("가중치 키 예시:", list(new_state_dict.keys())[:5])
        return None
    
    # 4. YOLO 모델 로드
    print("YOLO 모델 로드 중...")
    yolo_model = YOLO('yolo11x.pt')  # 기본 모델
    
    # 5. 새로운 클래스 기반 접근법으로 모델 통합
    print("백본 교체 중...")
    modified_model = create_model_with_resnet_backbone(yolo_model, resnet50)
    
    # 6. 백본 가중치 저장
    print("백본 가중치 저장 중...")
    backbone_state_dict = modified_model.model.backbone_model.backbone.state_dict()
    torch.save({
        'backbone': backbone_state_dict,
        'model_type': 'yolo11_resnet50_SIN',
        'datetime': datetime.now().strftime("%Y%m%d_%H%M%S")
    }, 'yolo11_resnet50_SIN_backbone.pt')
    print("백본 가중치가 저장되었습니다: yolo11_resnet50_SIN_backbone.pt")
    
    return modified_model

# 새로 추가: 저장된 백본으로 모델 훈련하는 함수
def train_with_resnet_backbone(data_yaml='sketch_dataset.yaml', epochs=100, batch=16):
    """
    저장된 ResNet50-SIN 백본 가중치를 로드하여 YOLO 모델을 훈련합니다.
    
    Args:
        data_yaml: 데이터셋 설정 파일 경로
        epochs: 훈련 에폭 수
        batch: 배치 크기
    
    Returns:
        훈련된 모델과 훈련 결과
    """
    # 1. 저장된 백본 가중치 로드
    print("저장된 백본 가중치 로드 중...")
    try:
        backbone_checkpoint = torch.load('yolo11_resnet50_SIN_backbone.pt')
        print("백본 가중치를 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print("백본 가중치 파일을 찾을 수 없습니다. 먼저 main() 함수를 실행하세요.")
        return None, None
    
    # 2. ResNet50 모델 생성
    print("ResNet50 모델 생성 중...")
    resnet50 = models.resnet50(weights=None)
    
    # 3. 백본 가중치 적용
    # 저장된 형식에 맞게 로드
    if 'backbone' in backbone_checkpoint:
        backbone_state_dict = backbone_checkpoint['backbone']
        # 여기서 백본을 구성하는 ResNet50BackboneAdapter 인스턴스 생성
        backbone_adapter = ResNet50BackboneAdapter(resnet50)
        # 가중치 로드
        backbone_adapter.load_state_dict(backbone_state_dict)
        print("백본 어댑터에 가중치가 적용되었습니다.")
    else:
        print("백본 가중치 형식이 올바르지 않습니다.")
        return None, None
    
    # 4. 기본 YOLO 모델 로드
    print("YOLO 모델 로드 중...")
    yolo_model = YOLO('yolo11x.pt')
    
    # 5. 커스텀 백본으로 YOLO 모델 생성
    print("커스텀 백본 연결 중...")
    custom_model = YOLOWithResNetBackbone(yolo_model, resnet50)
    yolo_model.model.backbone_model = custom_model
    yolo_model.model.forward = custom_model.forward
    
    # 6. 모델 훈련
    print(f"ResNet50-SIN 백본으로 모델 훈련 시작 (에폭: {epochs}, 배치: {batch})...")
    try:
        results = yolo_model.train(
            data=data_yaml,               # 데이터셋 설정 파일
            epochs=epochs,                # 에폭 수
            imgsz=640,                    # 입력 이미지 크기
            batch=batch,                  # 배치 크기
            workers=8,                    # 데이터 로더 worker 수
            device='0',                   # GPU 장치
            optimizer='Adam',             # 옵티마이저
            lr0=0.001,                    # 초기 학습률
            
            # 백본 레이어 프리즈 (선택 사항)
            freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 백본 레이어 프리즈
            
            project='sketch_detection',   # 프로젝트 이름
            name=f'SIN_resnet50_backbone_{datetime.now().strftime("%Y%m%d")}',  # 실험 이름
            exist_ok=True,                # 기존 실험 덮어쓰기 허용
            
            # 데이터 증강 옵션 (선택 사항)
            augment=True,                 # 데이터 증강 사용
            degrees=0.5,                  # 회전 범위 (±도)
            translate=0.1,                # 이동 범위 (±비율)
            scale=0.5,                    # 스케일 범위 (±비율)
            shear=0.1,                    # 전단 범위 (±도)
            fliplr=0.5,                   # 좌우 반전 확률
            mosaic=1.0,                   # 모자이크 증강 확률
        )
        print("훈련이 완료되었습니다!")
        
        # 훈련된 모델 저장
        save_path = f'sketch_detection/SIN_resnet50_trained_{datetime.now().strftime("%Y%m%d")}.pt'
        try:
            # 모델 저장 (가능한 경우)
            yolo_model.save(save_path)
            print(f"훈련된 모델이 저장되었습니다: {save_path}")
        except Exception as e:
            print(f"모델 저장 실패 (직렬화 문제): {e}")
            print("모델 기능만 사용 가능합니다.")
        
        return yolo_model, results
    
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        return yolo_model, None

if __name__ == "__main__":
    # 명령줄 인수 처리
    import argparse
    parser = argparse.ArgumentParser(description='ResNet50-SIN 백본으로 YOLO 모델 생성 및 훈련')
    parser.add_argument('--mode', type=str, default='create', choices=['create', 'train', 'both'],
                       help='실행 모드: create(백본 생성), train(백본으로 훈련), both(둘 다)')
    parser.add_argument('--data', type=str, default='sketch_dataset.yaml',
                       help='훈련용 데이터셋 설정 파일')
    parser.add_argument('--epochs', type=int, default=100,
                       help='훈련 에폭 수')
    parser.add_argument('--batch', type=int, default=16,
                       help='배치 크기')
    
    args = parser.parse_args()
    
    if args.mode == 'create' or args.mode == 'both':
        print("=== ResNet50-SIN 백본 생성 모드 ===")
        model = main()
    
    if args.mode == 'train' or args.mode == 'both':
        print("\n=== ResNet50-SIN 백본으로 훈련 모드 ===")
        train_model, results = train_with_resnet_backbone(
            data_yaml=args.data,
            epochs=args.epochs,
            batch=args.batch
        )
        
        # 훈련 후 평가 (선택 사항)
        if train_model is not None:
            print("\n=== 모델 평가 ===")
            metrics = train_model.val()
            print(f"평가 결과: {metrics}")
    
    # 테스트 실행 (선택 사항)
    try:
        test_img = "path/to/test_image.jpg"  # 테스트 이미지 경로 수정 필요
        if os.path.exists(test_img):
            results = model(test_img)
            print("모델이 성공적으로 실행되었습니다.")
            results[0].show()
    except Exception as e:
        print(f"테스트 실패: {e}")
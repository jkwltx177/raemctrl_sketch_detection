import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics.nn.modules import Backbone

class ResNet50BackboneAdapter(Backbone):
    def __init__(self, backbone_path=None):
        super().__init__()
        # 백본 가중치 로드
        checkpoint = torch.load(backbone_path)
        if 'backbone' in checkpoint:
            self.backbone_weights = checkpoint['backbone']
        else:
            self.backbone_weights = checkpoint
        
        # 백본 구성
        self.setup_layers()
        
        # 가중치 적용
        self.load_weights()
    
    def setup_layers(self):
        # ResNet50 모델 생성
        resnet_model = models.resnet50(weights=None)
        
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
    
    def load_weights(self):
        # 저장된 가중치 적용
        try:
            self.load_state_dict(self.backbone_weights)
            print("백본 가중치가 성공적으로 로드되었습니다.")
        except Exception as e:
            print(f"가중치 로드 실패: {e}")
            # 가중치 키 구조 출력으로 디버깅 도움
            print("모델 키:", list(self.state_dict().keys())[:5])
            print("가중치 키:", list(self.backbone_weights.keys())[:5])
    
    def forward(self, x):
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
        
        # YOLOv8은 일반적으로 [p3, p4, p5] 형태의 피처맵을 기대함
        # 디버깅을 위한 shape 출력
        print(f"P3 shape: {p3.shape}")
        print(f"P4 shape: {p4.shape}")
        print(f"P5 shape: {p5.shape}")
        
        return [p3, p4, p5]  # YOLOv8 호환 형식

def get_backbone():
    return ResNet50BackboneAdapter
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetSINBackbone(nn.Module):
    def __init__(self, weights_path):
        super().__init__()

        self.weights_path = weights_path

        # 백본 가중치 로드
        checkpoint = torch.load(weights_path)
        self.backbone_weights = checkpoint.get('backbone', checkpoint)
            
        # ResNet50 모델 생성
        self.resnet_model = models.resnet50(weights=None)
        
        # ResNet 컴포넌트 저장
        self.stem = nn.Sequential(
            self.resnet_model.conv1,    # 7x7 Conv
            self.resnet_model.bn1,      # BatchNorm
            self.resnet_model.relu,     # ReLU
            self.resnet_model.maxpool   # MaxPool
        )
        self.layer1 = self.resnet_model.layer1  # 256 채널
        self.layer2 = self.resnet_model.layer2  # 512 채널 (P3)
        self.layer3 = self.resnet_model.layer3  # 1024 채널 (P4)
        self.layer4 = self.resnet_model.layer4  # 2048 채널
        
        # 채널 수 조정을 위한 1x1 컨볼루션
        self.p5_conv = nn.Conv2d(2048, 1024, kernel_size=1)
        
        # SPPF 및 C2PSA 레이어
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
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True)
        )
        
        # 가중치 키 매핑 후 로드
        self._load_mapped_weights()
        
    def _load_mapped_weights(self):
        """가중치 키 매핑 로직 개선"""
        mapped_weights = {}

        # 가중치의 실제 키 구조 확인
        print(f"가중치 키 구조: {list(self.backbone_weights.keys())[:5]}")

        # Stem 부분 매핑
        if 'stem.0.weight' in self.backbone_weights:
            mapped_weights['resnet_model.conv1.weight'] = self.backbone_weights['stem.0.weight']
            mapped_weights['resnet_model.bn1.weight'] = self.backbone_weights['stem.1.weight']
            mapped_weights['resnet_model.bn1.bias'] = self.backbone_weights['stem.1.bias']
            mapped_weights['resnet_model.bn1.running_mean'] = self.backbone_weights['stem.1.running_mean']
            mapped_weights['resnet_model.bn1.running_var'] = self.backbone_weights['stem.1.running_var']

        # Layer 부분 매핑
        for key, value in self.backbone_weights.items():
            if key.startswith('layer'):
                mapped_weights['resnet_model.' + key] = value

        # 가중치 로드 (strict=False로 누락된 키 허용)
        self.load_state_dict(mapped_weights, strict=False)
    
    def forward(self, x):
        # 스템
        stem_out = self.stem(x)
        
        # 레이어 1-4 순차 처리
        layer1_out = self.layer1(stem_out)
        layer2_out = self.layer2(layer1_out)  # P3
        layer3_out = self.layer3(layer2_out)  # P4
        layer4_out = self.layer4(layer3_out)  # P5
        
        # 채널 수 조정 (2048 -> 1024)
        p5_conv_out = self.p5_conv(layer4_out)
        
        # SPPF 및 C2PSA 처리
        sppf_out = self.sppf(p5_conv_out)
        c2psa_out = self.c2psa(sppf_out)
        
        # 최종 출력 (단일 텐서를 여러 피처맵으로 변형)
        p3 = nn.Conv2d(512, 3, kernel_size=1).to(x.device)(layer2_out)
        p4 = nn.Conv2d(1024, 3, kernel_size=1).to(x.device)(layer3_out)
        p5 = nn.Conv2d(1024, 3, kernel_size=1).to(x.device)(c2psa_out)
        
        # 여러 피처맵 반환 옵션 (현재 구조에 맞춰 단일 출력만 반환)
        # return [p3, p4, p5]  # 다중 출력 옵션
        
        final_conv = nn.Conv2d(1024, 3, kernel_size=1).to(x.device)
        out = final_conv(c2psa_out)
        print(f"최종 출력 shape: {out.shape}")
        
        return out
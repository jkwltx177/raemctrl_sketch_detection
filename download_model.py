import torch
import os
import requests
from tqdm import tqdm

# load_pretrained_models.py 파일에서 URL 확인
# 해당 URL 직접 사용
url = "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar"  # 예시 URL, 실제 URL 확인 필요

# 다운로드 경로 설정
download_path = 'weights/resnet50_SIN.pth'
os.makedirs(os.path.dirname(download_path), exist_ok=True)

# 직접 다운로드 구현
def download_file(url, path):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 오류 발생 시 예외 발생
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(path, 'wb') as f, tqdm(
        desc=path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))
    
    print(f"파일이 성공적으로 다운로드되었습니다: {path}")

# 파일 다운로드
download_file(url, download_path)

# 모델 직접 로드
model = torch.load(download_path)
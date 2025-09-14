import os
import shutil

# 경로 설정
data_dir = 'C:/Users/juhwa/Downloads/deepfashion/Category and Attribute Prediction Benchmark'  # DeepFashion 데이터셋이 있는 루트 경로
attr_file = os.path.join(data_dir, 'Anno_coarse/list_attr_img.txt')  # 속성 파일 경로
img_dir = os.path.join(data_dir, 'img')  # 이미지 폴더 경로
output_dir = 'C:/Users/juhwa/Downloads/deepfashion/Category and Attribute Prediction Benchmark/hat_images'  # 모자 이미지를 저장할 새로운 폴더 경로

# 출력 디렉토리 생성 (이미 존재하면 무시)
os.makedirs(output_dir, exist_ok=True)

# list_attr_img.txt 파일 읽기
with open(attr_file, 'r') as f:
    lines = f.readlines()
    attr_names = lines[1].strip().split()
    print("속성 목록:", attr_names)


# 속성 이름 목록 (두 번째 줄)
attr_names = lines[1].strip().split()

# "hat" 속성의 인덱스 찾기
try:
    hat_index = attr_names.index('hat')
    hat_index = attr_names.index('headwear')  # 'headwear'를 찾은 경우
except ValueError:
    print("속성 목록에 'hat'이 없습니다. 파일을 확인하세요.")
    exit()

# "hat" 속성이 1인 이미지 경로 추출
hat_images = []
for line in lines[2:]:  # 첫 번째 줄(이미지 수)과 두 번째 줄(속성 이름)은 건너뜀
    parts = line.strip().split()
    img_path = parts[0]  # 이미지 경로
    attrs = parts[1:]  # 속성 값들
    if len(attrs) > hat_index and attrs[hat_index] == '1':
        hat_images.append(img_path)

# 필터링된 이미지를 새로운 폴더로 복사
for img_path in hat_images:
    full_img_path = os.path.join(img_dir, img_path)  # 원본 이미지 전체 경로
    if os.path.exists(full_img_path):
        shutil.copy(full_img_path, output_dir)
    else:
        print(f"이미지 파일이 존재하지 않습니다: {full_img_path}")

print(f"총 {len(hat_images)}개의 모자 이미지를 {output_dir}에 복사 완료!")
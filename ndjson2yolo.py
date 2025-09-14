import json
import os
from PIL import Image, ImageDraw

def draw_sketch(drawing, size=(256, 256)):
    if not drawing:
        return None
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    drawn = False  # 선이 그려졌는지 추적
    for stroke in drawing:
        if len(stroke[0]) < 2 or len(stroke[1]) < 2:
            continue
        for i in range(len(stroke[0]) - 1):
            x1, y1 = stroke[0][i], stroke[1][i]
            x2, y2 = stroke[0][i + 1], stroke[1][i + 1]
            if (0 <= x1 < size[0] and 0 <= y1 < size[1] and 
                0 <= x2 < size[0] and 0 <= y2 < size[1]):
                draw.line([x1, y1, x2, y2], fill='black', width=5)
                drawn = True
    if not drawn:  # 선이 하나도 그려지지 않았으면 None 반환
        return None
    return img

def get_bounding_box(drawing):
    all_x = [x for stroke in drawing for x in stroke[0] if stroke[0]]
    all_y = [y for stroke in drawing for y in stroke[1] if stroke[1]]
    if not all_x or not all_y:
        return None
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    # 너비나 높이가 0이거나 이미지 크기를 벗어나면 None 반환
    if x_min == x_max or y_min == y_max or x_max < 0 or y_max < 0 or x_min >= 256 or y_min >= 256:
        return None
    return x_min, y_min, x_max, y_max

# NDJSON 파일 처리 함수
def process_ndjson(ndjson_file, output_dir, class_map):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    with open(ndjson_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if not data.get('recognized', False):
                continue
            drawing = data.get('drawing', [])
            # drawing이 비어 있거나 유효한 좌표가 없는 경우 건너뛰기
            if not drawing or not any(len(stroke[0]) >= 2 and len(stroke[1]) >= 2 for stroke in drawing):
                continue
            word = data.get('word', '')
            if word not in class_map:
                continue
            class_id = class_map[word]
            
            # 이미지 생성
            img = draw_sketch(drawing)
            if img is None:
                continue
            
            # 바운딩 박스 계산
            bbox = get_bounding_box(drawing)
            if bbox is None:
                continue
            x_min, y_min, x_max, y_max = bbox
            img_width, img_height = img.size
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # 이미지 저장
            img_filename = f"{data['key_id']}.jpg"
            img.save(os.path.join(output_dir, 'images', img_filename))
            
            # 레이블 파일 생성
            label_filename = f"{data['key_id']}.txt"
            with open(os.path.join(output_dir, 'labels', label_filename), 'w') as label_file:
                label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# 클래스 맵 정의 (예: 'hat': 0, 'airplane': 1 등)
class_map = {'hat': 12}  # 필요한 클래스 추가

# NDJSON 파일 경로와 출력 디렉토리 설정
ndjson_file = 'full_raw_hat.ndjson'
output_dir = 'hat_output'

# NDJSON 파일 처리
process_ndjson(ndjson_file, output_dir, class_map)
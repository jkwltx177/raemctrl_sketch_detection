import os

def fix_labels(label_dir):
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            # 좌표를 0~1 범위로 제한
            x_center = max(0, min(x_center, 1))
            y_center = max(0, min(y_center, 1))
            width = max(0, min(width, 1))
            height = max(0, min(height, 1))
            new_lines.append(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
        with open(label_path, 'w') as f:
            f.writelines(new_lines)

# 레이블 파일 수정 실행
label_dir = 'C:/capstone/data/validation/labels'
fix_labels(label_dir)
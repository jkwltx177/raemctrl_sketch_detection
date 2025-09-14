import os
import glob

def fix_label_files(labels_dir):
    # 모든 txt 파일 찾기
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    for file_path in label_files:
        fixed_lines = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            # 각 행이 클래스 ID + 4개의 좌표값으로 구성되어야 함
            if len(parts) > 5:
                # 잘못된 행 분할
                i = 0
                while i < len(parts):
                    if i + 5 <= len(parts):
                        fixed_lines.append(' '.join(parts[i:i+5]) + '\n')
                    i += 5
            else:
                fixed_lines.append(line)
        
        # 수정된 내용으로 파일 덮어쓰기
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)

if __name__ == "__main__":
    fix_label_files(r"C:\capstone\data\validation\new_new_composited\labels")
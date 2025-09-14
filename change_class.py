import os

# 현재 디렉토리
current_dir = r"C:\capstone\data\train\labels"

# 대상 접두사 목록
prefixes = ["s_0823", "s_0824"]

# 대상 파일 필터링
target_files = [f for f in os.listdir(current_dir) 
                if f.endswith(".txt") and any(f.startswith(p) for p in prefixes)]

for file_name in target_files:
    file_path = os.path.join(current_dir, file_name)
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 클래스 번호를 무조건 24로 덮어씀
    modified_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            parts[0] = "24"
        modified_lines.append(" ".join(parts) + "\n")
    
    # 덮어쓰기
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(modified_lines)

print(f"{len(target_files)}개의 파일 클래스 번호가 24로 변경 완료되었습니다.")


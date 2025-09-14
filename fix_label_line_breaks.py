import os
import glob
import argparse
import re

def normalize_path(path):
    """경로 정규화 함수"""
    return os.path.normpath(path).replace("\\", "/")

def is_valid_yolo_label(parts):
    """YOLO 라벨 형식이 유효한지 확인"""
    if len(parts) < 5:
        return False
    
    try:
        # 클래스 ID가 정수인지 확인
        class_id = int(parts[0])
        
        # 나머지 4개 값이 0-1 사이의 부동소수점인지 확인
        for i in range(1, 5):
            value = float(parts[i])
            if value < 0 or value > 1:
                return False
                
        return True
    except ValueError:
        return False

def fix_label_file(file_path, dry_run=False):
    """
    라벨 파일의 줄바꿈 오류를 수정합니다.
    
    Args:
        file_path: 수정할 라벨 파일 경로
        dry_run: True이면 실제로 파일을 수정하지 않고 수정 예정인 내용만 반환
        
    Returns:
        (수정된 라인 수, 수정된 내용 또는 None)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        fixed_lines = []
        fixed_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:  # 빈 줄 무시
                continue
                
            parts = line.split()
            
            # 정상적인 YOLO 형식 라벨인 경우 (class_id x y w h)
            if len(parts) == 5 and is_valid_yolo_label(parts):
                fixed_lines.append(' '.join(parts))
            # 여러 라벨이 한 줄에 있는 경우
            elif len(parts) > 5:
                # 5개 항목씩 그룹화하여 분리
                for i in range(0, len(parts), 5):
                    if i + 4 < len(parts):  # 완전한 5개 항목이 있는지 확인
                        group = parts[i:i+5]
                        # 유효한 YOLO 라벨인지 확인
                        if is_valid_yolo_label(group):
                            fixed_lines.append(' '.join(group))
                            fixed_count += 1
                        else:
                            # 무효한 그룹은 그대로 유지 (나중에 확인을 위해)
                            print(f"경고: {file_path}에서 무효한 라벨 그룹 발견: {' '.join(group)}")
            else:
                # 형식이 맞지 않는 라벨은 그대로 유지 (로그 출력)
                print(f"경고: {file_path}에서 형식이 맞지 않는 라벨 발견: {line}")
                fixed_lines.append(line)
        
        # 수정된 내용
        fixed_content = '\n'.join(fixed_lines)
        
        # dry_run이 아닌 경우에만 실제로 파일 수정
        if not dry_run and fixed_count > 0:
            with open(file_path, 'w') as f:
                f.write(fixed_content)
            
        return fixed_count, fixed_content if dry_run else None
        
    except Exception as e:
        print(f"오류: {file_path} 처리 중 예외 발생: {e}")
        return 0, None

def process_directory(directory_path, recursive=False, dry_run=False):
    """
    디렉토리 내의 모든 txt 파일을 처리합니다.
    
    Args:
        directory_path: 처리할 디렉토리 경로
        recursive: True이면 하위 디렉토리도 재귀적으로 처리
        dry_run: True이면 실제로 파일을 수정하지 않고 예상 결과만 출력
    """
    directory_path = normalize_path(directory_path)
    pattern = os.path.join(directory_path, "**", "*.txt") if recursive else os.path.join(directory_path, "*.txt")
    
    txt_files = glob.glob(pattern, recursive=recursive)
    total_files = len(txt_files)
    fixed_files = 0
    total_fixes = 0
    
    print(f"총 {total_files}개의 txt 파일을 찾았습니다.")
    
    if dry_run:
        print("테스트 모드입니다 - 실제 파일은 수정되지 않습니다.")
    
    for file_path in txt_files:
        fixes, _ = fix_label_file(file_path, dry_run)
        
        if fixes > 0:
            fixed_files += 1
            total_fixes += fixes
            print(f"수정됨: {file_path} (수정 항목: {fixes}개)")
    
    print(f"\n처리 완료: {total_files}개 파일 중 {fixed_files}개 파일 수정됨, 총 {total_fixes}개 항목 수정됨")
    if dry_run:
        print("이 테스트 실행에서는 파일이 실제로 수정되지 않았습니다. 실제 수정을 위해 --dry-run 옵션을 제거하세요.")

def main():
    parser = argparse.ArgumentParser(description='YOLO 라벨 파일의 줄바꿈 오류를 수정합니다.')
    parser.add_argument('directory', help='처리할 라벨 파일이 있는 디렉토리 경로')
    parser.add_argument('--recursive', '-r', action='store_true', help='하위 디렉토리까지 재귀적으로 처리')
    parser.add_argument('--dry-run', '-d', action='store_true', help='테스트 모드 - 실제 파일을 수정하지 않음')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"오류: {args.directory}는 유효한 디렉토리가 아닙니다.")
        return
    
    process_directory(args.directory, args.recursive, args.dry_run)

if __name__ == "__main__":
    main()

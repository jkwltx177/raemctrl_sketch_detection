import os
import glob

def merge_label_files():
    """
    C:\capstone\data\validation\new_new_composited 디렉토리에 있는 txt 파일의 내용을
    C:\capstone\data\validation\new_new_composited\labels 디렉토리의 동일 이름 파일에 추가합니다.
    """
    # 소스 디렉토리 (수동으로 추가한 라벨이 있는 디렉토리)
    source_dir = r"C:\capstone\data\validation\new_new_composited"
    # 대상 디렉토리 (기존 라벨 파일이 있는 디렉토리)
    target_dir = r"C:\capstone\data\validation\new_new_composited\labels"
    
    # 소스 디렉토리에서 모든 txt 파일 찾기
    source_files = glob.glob(os.path.join(source_dir, "*.txt"))
    
    # 처리된 파일 수 카운터
    processed_count = 0
    failed_count = 0
    
    print(f"소스 디렉토리에서 {len(source_files)}개의 txt 파일을 찾았습니다.")
    
    for source_file in source_files:
        # 파일 이름만 추출 (경로 제외)
        file_name = os.path.basename(source_file)
        
        # 대상 파일 경로 구성
        target_file = os.path.join(target_dir, file_name)
        
        # 대상 파일이 존재하는지 확인
        if not os.path.exists(target_file):
            print(f"경고: 대상 파일이 없습니다: {target_file}")
            failed_count += 1
            continue
        
        try:
            # 소스 파일 내용 읽기
            with open(source_file, 'r', encoding='utf-8') as f:
                source_content = f.read().strip()
            
            # 대상 파일 내용 읽기
            with open(target_file, 'r', encoding='utf-8') as f:
                target_content = f.read().strip()
            
            # 내용 병합 및 쓰기
            with open(target_file, 'w', encoding='utf-8') as f:
                # 기존 내용과 새 내용 사이에 줄바꿈이 있는지 확인
                if target_content and source_content:
                    if not target_content.endswith('\n'):
                        target_content += '\n'
                
                # 병합된 내용 쓰기
                f.write(target_content + source_content)
            
            processed_count += 1
            
            # 진행 상황 출력 (10개 파일마다)
            if processed_count % 10 == 0:
                print(f"{processed_count}개 파일 처리 완료...")
                
        except Exception as e:
            print(f"오류: {file_name} 처리 중 문제 발생: {str(e)}")
            failed_count += 1
    
    print(f"\n작업 완료: {processed_count}개 파일 성공적으로 병합, {failed_count}개 파일 실패")

def merge_label_files_with_duplicate_check():
    """
    라벨 중복을 확인하여 병합하는 향상된 버전
    """
    # 소스 디렉토리 (수동으로 추가한 라벨이 있는 디렉토리)
    source_dir = r"C:\capstone\data\validation\new_new_composited"
    # 대상 디렉토리 (기존 라벨 파일이 있는 디렉토리)
    target_dir = r"C:\capstone\data\validation\new_new_composited\labels"
    
    # 소스 디렉토리에서 모든 txt 파일 찾기
    source_files = glob.glob(os.path.join(source_dir, "*.txt"))
    
    # 처리된 파일 수 카운터
    processed_count = 0
    failed_count = 0
    
    print(f"소스 디렉토리에서 {len(source_files)}개의 txt 파일을 찾았습니다.")
    
    for source_file in source_files:
        # 파일 이름만 추출 (경로 제외)
        file_name = os.path.basename(source_file)
        
        # 대상 파일 경로 구성
        target_file = os.path.join(target_dir, file_name)
        
        # 대상 파일이 존재하는지 확인
        if not os.path.exists(target_file):
            print(f"경고: 대상 파일이 없습니다: {target_file}")
            failed_count += 1
            continue
        
        try:
            # 소스 파일 내용 읽기
            with open(source_file, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()
            
            # 대상 파일 내용 읽기
            with open(target_file, 'r', encoding='utf-8') as f:
                target_lines = f.readlines()
            
            # 각 줄을 라벨 ID 기준으로 분류
            source_labels = {}
            for line in source_lines:
                parts = line.strip().split()
                if parts:
                    label_id = parts[0]
                    if label_id not in source_labels:
                        source_labels[label_id] = []
                    source_labels[label_id].append(line)
            
            target_labels = {}
            for line in target_lines:
                parts = line.strip().split()
                if parts:
                    label_id = parts[0]
                    if label_id not in target_labels:
                        target_labels[label_id] = []
                    target_labels[label_id].append(line)
            
            # 병합된 라인 생성
            merged_lines = target_lines.copy()
            
            # 소스에서 새 라벨 추가
            for label_id, lines in source_labels.items():
                if label_id not in target_labels:
                    # 새 라벨 ID인 경우 모든 라인 추가
                    merged_lines.extend(lines)
                else:
                    # 기존 라벨 ID인 경우 중복 확인
                    for line in lines:
                        if line not in target_labels[label_id]:
                            merged_lines.append(line)
            
            # 내용 병합 및 쓰기
            with open(target_file, 'w', encoding='utf-8') as f:
                f.writelines(merged_lines)
            
            processed_count += 1
            
            # 진행 상황 출력 (10개 파일마다)
            if processed_count % 10 == 0:
                print(f"{processed_count}개 파일 처리 완료...")
                
        except Exception as e:
            print(f"오류: {file_name} 처리 중 문제 발생: {str(e)}")
            failed_count += 1
    
    print(f"\n작업 완료: {processed_count}개 파일 성공적으로 병합, {failed_count}개 파일 실패")

if __name__ == "__main__":
    print("라벨 파일 병합 유틸리티\n")
    print("1. 단순 병합 (소스 파일 내용을 대상 파일에 추가)")
    print("2. 중복 검사 병합 (라벨 ID 기준으로 중복 확인 후 병합)")
    
    choice = input("\n작업 방식을 선택하세요 (1 또는 2): ")
    
    if choice == "1":
        merge_label_files()
    elif choice == "2":
        merge_label_files_with_duplicate_check()
    else:
        print("잘못된 선택입니다. 프로그램을 종료합니다.")

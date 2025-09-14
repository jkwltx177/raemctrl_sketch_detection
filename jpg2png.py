import os
import glob
from PIL import Image
import cv2
import numpy as np

def convert_jpg_to_png_with_transparency(source_dir, target_dir):
    """
    source_dir의 모든 JPG 이미지를 PNG로 변환하고 배경을 투명하게 만들어 target_dir에 저장합니다.
    
    :param source_dir: JPG 이미지가 있는 디렉토리 경로
    :param target_dir: 변환된 PNG 이미지를 저장할 디렉토리 경로
    """
    # 타겟 디렉토리가 없으면 생성
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 모든 JPG 파일 찾기 (대소문자 구분 없이)
    jpg_files = glob.glob(os.path.join(source_dir, '*.jpg')) + \
                glob.glob(os.path.join(source_dir, '*.jpeg')) + \
                glob.glob(os.path.join(source_dir, '*.JPG')) + \
                glob.glob(os.path.join(source_dir, '*.JPEG'))
    
    count = 0
    for jpg_file in jpg_files:
        try:
            # 파일명 추출
            file_name = os.path.basename(jpg_file)
            name_without_ext = os.path.splitext(file_name)[0]
            
            # PNG 파일 경로
            png_file = os.path.join(target_dir, f"{name_without_ext}.png")
            
            # 방법 1: 간단한 변환 (배경 투명화 없음)
            # Image.open(jpg_file).save(png_file, 'PNG')
            
            # 방법 2: OpenCV를 사용한 배경 제거 및 투명화
            # 이미지 로드
            img = cv2.imread(jpg_file)
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 임계값 설정으로 마스크 생성 (밝은 배경 가정)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # 가장자리 부드럽게 처리
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # 알파 채널 생성
            b, g, r = cv2.split(img)
            rgba = [b, g, r, mask]
            dst = cv2.merge(rgba, 4)
            
            # PNG로 저장
            cv2.imwrite(png_file, dst)
            
            count += 1
            print(f"변환 완료: {jpg_file} -> {png_file}")
            
        except Exception as e:
            print(f"변환 실패: {jpg_file}, 오류: {e}")
    
    print(f"\n총 {count}개의 이미지가 변환되었습니다.")
    
def convert_simple(source_dir, target_dir, subfolder_structure=False):
    """
    JPG 이미지를 PNG로 단순 변환합니다 (포맷만 변경).
    
    :param source_dir: JPG 이미지가 있는 디렉토리 경로
    :param target_dir: 변환된 PNG 이미지를 저장할 디렉토리 경로
    :param subfolder_structure: 하위 폴더 구조를 유지할지 여부
    """
    # 타겟 디렉토리가 없으면 생성
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    if subfolder_structure:
        # 하위 폴더 구조를 포함한 모든 JPG 파일 검색
        jpg_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    jpg_files.append(os.path.join(root, file))
    else:
        # 현재 폴더의 JPG 파일만 검색
        jpg_files = glob.glob(os.path.join(source_dir, '*.jpg')) + \
                    glob.glob(os.path.join(source_dir, '*.jpeg')) + \
                    glob.glob(os.path.join(source_dir, '*.JPG')) + \
                    glob.glob(os.path.join(source_dir, '*.JPEG'))
    
    count = 0
    for jpg_file in jpg_files:
        try:
            # 상대 경로 계산
            rel_path = os.path.relpath(jpg_file, source_dir)
            dir_path = os.path.dirname(rel_path)
            
            # 파일명 추출
            file_name = os.path.basename(jpg_file)
            name_without_ext = os.path.splitext(file_name)[0]
            
            # 대상 디렉토리 생성
            if subfolder_structure and dir_path:
                target_subdir = os.path.join(target_dir, dir_path)
                if not os.path.exists(target_subdir):
                    os.makedirs(target_subdir)
                png_file = os.path.join(target_subdir, f"{name_without_ext}.png")
            else:
                png_file = os.path.join(target_dir, f"{name_without_ext}.png")
            
            # 이미지 로드 및 변환
            img = Image.open(jpg_file)
            
            # RGBA로 변환 (알파 채널 추가)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # PNG로 저장
            img.save(png_file, 'PNG')
            
            count += 1
            if count % 10 == 0:
                print(f"{count}개 변환 완료...")
            
        except Exception as e:
            print(f"변환 실패: {jpg_file}, 오류: {e}")
    
    print(f"\n총 {count}개의 이미지가 변환되었습니다.")

# 실행 예제
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='JPG 이미지를 PNG로 변환')
    parser.add_argument('--source', type=str, default='.', help='소스 디렉토리 경로')
    parser.add_argument('--target', type=str, default='./png_output', help='타겟 디렉토리 경로')
    parser.add_argument('--mode', type=str, default='simple', choices=['simple', 'transparency'], 
                        help='변환 모드: simple(단순 변환) 또는 transparency(배경 투명화 시도)')
    parser.add_argument('--subfolders', action='store_true', help='하위 폴더 구조 유지')
    
    args = parser.parse_args()
    
    print(f"변환 시작: {args.source} -> {args.target}")
    print(f"모드: {args.mode}, 하위 폴더 유지: {args.subfolders}")
    
    if args.mode == 'transparency':
        convert_jpg_to_png_with_transparency(args.source, args.target)
    else:
        convert_simple(args.source, args.target, args.subfolders)
import json
import os
import glob

# 입력 JSON 파일들이 있는 폴더 경로 (실제 경로로 수정)
input_dir = r"C:\049.스케치, 아이콘 인식용 다양한 추상 이미지 데이터\01.데이터\2.Validation\라벨링데이터\VL1\ABSTRACT_SKETCH\L1_6\L2_46\L3_816"
# 수정된 JSON 파일을 저장할 출력 폴더 경로 (실제 경로로 수정)
output_dir = r"C:\049.스케치, 아이콘 인식용 다양한 추상 이미지 데이터\01.데이터\2.Validation\라벨링데이터\VL1\ABSTRACT_SKETCH\L1_6\L2_46\L3_816_hlaf_width"
os.makedirs(output_dir, exist_ok=True)

modified_count = 0
json_files = glob.glob(os.path.join(input_dir, "*.json"))

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # category 정보 확인: '귀걸이'인 경우에만 처리
    category = data.get("category", {})
    ctg_nm_level3 = category.get("ctg_nm_level3", "")
    if ctg_nm_level3 != "귀걸이":
        continue
    
    # abstract_image의 abs_bbox 값 추출 (예: "[78,41.48957824707031,142,213]")
    abs_img = data.get("abstract_image", {})
    abs_bbox_str = abs_img.get("abs_bbox", None)
    if abs_bbox_str is None:
        continue
    
    try:
        # 문자열을 리스트로 파싱 (형식: [x, y, w, h])
        bbox = json.loads(abs_bbox_str)
        if len(bbox) != 4:
            print(f"{json_file} - 예상과 다른 bbox 형식입니다.")
            continue
        
        x, y, w, h = bbox
        # 가로 길이(w)를 절반으로 줄임
        new_w = w / 2.0
        new_bbox = [x, y, new_w, h]
        
        # 수정된 bbox를 다시 문자열로 저장
        abs_img["abs_bbox"] = json.dumps(new_bbox)
        data["abstract_image"] = abs_img
        
    except Exception as e:
        print(f"{json_file} 처리 중 오류 발생: {e}")
        continue
    
    # 출력 폴더에 같은 파일명으로 저장
    output_file_path = os.path.join(output_dir, os.path.basename(json_file))
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    modified_count += 1
    print(f"변환 완료: {json_file} -> {output_file_path}")

print(f"총 {modified_count}개의 '귀걸이' JSON 파일이 업데이트되었습니다.")

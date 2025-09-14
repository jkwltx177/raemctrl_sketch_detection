import json
import os
import glob

# 최종 통합 데이터셋에서 '안경' 클래스 번호 (dataset.yaml에서 'x'는 n-1번)
class_mapping = {
    "담배": 26
}

# 두 디렉토리 (안경 JSON과 선글라스 JSON 파일이 있는 경로)
input_dirs = [
    #r"C:\049.스케치, 아이콘 인식용 다양한 추상 이미지 데이터\01.데이터\2.Validation\라벨링데이터\VL1\ABSTRACT_SKETCH\L1_8\L2_58\L3_1003",
    r"C:\049.스케치, 아이콘 인식용 다양한 추상 이미지 데이터\01.데이터\2.Validation\라벨링데이터\VL1\ABSTRACT_SKETCH\L1_3\L2_24\L3_305"
]

#output_dir = "C:/capstone/data/train/labels"
output_dir = "C:/capstone/data/validation/labels"
os.makedirs(output_dir, exist_ok=True)

# 라벨 변환 규칙: "안경테"와 "선글라스" 모두 "안경"으로 변환
# def map_category(label):
#     if label in ["스냅백", "베레모", "중절모", "비니", "털모자", "모자(캡)", "카우보이모자", "학사모", "왕관"]:
#         return "모자"
#     return label
# def map_category(label):
#     if label in ["입술/입", "이빨/치아"]:
#         return "벌린입(치아)"
#     return label
# def map_category(label):
#     if label in ["앉아있는 사람", "걷는 사람"]:
#         return "사람전체"
#     return label
def map_category(label):
    if label == "담배":
        return "담배"
    return label

# 변환된 파일 개수를 카운트할 변수
converted_count = 0

# 각 입력 디렉토리의 JSON 파일들을 순회
for input_dir in input_dirs:
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # abstract_image 정보에서 이미지 해상도와 경로, 바운딩 박스 추출
        abs_img = data.get("abstract_image", {})
        abs_width = abs_img.get("abs_width", 300)
        abs_height = abs_img.get("abs_height", 300)
        abs_path = abs_img.get("abs_path", "default.jpg")
        img_file_name = os.path.basename(abs_path)
        txt_file_name = os.path.splitext(img_file_name)[0] + ".txt"
        output_file_path = os.path.join(output_dir, txt_file_name)
        
        # abs_bbox는 문자열 형태로 제공되므로 파싱 (예: "[20,57.48957824707031,263,177]")
        abs_bbox_str = abs_img.get("abs_bbox", None)
        if abs_bbox_str is None:
            continue
        try:
            bbox_values = json.loads(abs_bbox_str)
            x, y, w, h = bbox_values
        except Exception as e:
            print(f"Error parsing bbox in {json_file}: {e}")
            continue
        
        # 중심 좌표 계산 및 정규화 (YOLO 포맷: [cx, cy, w, h] - 모두 0~1 범위)
        cx = x + w / 2.0
        cy = y + h / 2.0
        cx_norm = cx / abs_width
        cy_norm = cy / abs_height
        w_norm = w / abs_width
        h_norm = h / abs_height
        
        # category 정보에서 레벨3 라벨 추출 및 변환
        cat = data.get("category", {})
        ctg_label = cat.get("ctg_nm_level3", None)
        if ctg_label is None:
            continue
        
        mapped_label = map_category(ctg_label)
        if mapped_label not in class_mapping:
            continue  # 최종 통합 클래스에 없다면 무시
        
        class_id = class_mapping[mapped_label]
        
        # YOLO 형식의 txt 파일로 저장
        with open(output_file_path, "w", encoding="utf-8") as f_out:
            f_out.write(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        converted_count += 1
        print(f"변환 완료: {json_file} -> {output_file_path}")

print(f"총 {converted_count}개의 파일이 변환되었습니다.")
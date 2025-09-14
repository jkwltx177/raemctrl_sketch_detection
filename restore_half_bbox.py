import json
import os

# 사용자가 복구할 JSON 파일 경로 지정
input_file = input("복구할 JSON 파일의 경로를 입력하세요: ")
# 복구된 JSON 파일을 저장할 출력 폴더
output_dir = r"C:\049.스케치, 아이콘 인식용 다양한 추상 이미지 데이터\01.데이터\1.Training\라벨링데이터\TL3\ABSTRACT_SKETCH\L1_6\L2_46\복구"
os.makedirs(output_dir, exist_ok=True)

try:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # category 정보 확인
    category = data.get("category", {})
    ctg_nm_level3 = category.get("ctg_nm_level3", "")
    
    # "귀걸이" 카테고리인지 확인
    if ctg_nm_level3 != "귀걸이":
        print("지정된 파일은 '귀걸이' 카테고리가 아닙니다.")
    else:
        # abs_bbox 복구
        abs_img = data.get("abstract_image", {})
        abs_bbox_str = abs_img.get("abs_bbox", None)
        
        if abs_bbox_str is not None:
            # 문자열을 리스트로 변환
            bbox = json.loads(abs_bbox_str)
            if len(bbox) != 4:
                print("bbox 형식이 잘못되었습니다.")
            else:
                x, y, w, h = bbox
                # 가로 길이(w)를 두 배로 복구
                restored_w = w * 2.0
                restored_bbox = [x, y, restored_w, h]

                # 복구된 bbox를 다시 JSON에 반영
                abs_img["abs_bbox"] = json.dumps(restored_bbox)
                data["abstract_image"] = abs_img

                # 복구된 파일 저장
                output_file_path = os.path.join(output_dir, os.path.basename(input_file))
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

                print(f"복구 완료: {input_file} -> {output_file_path}")
        else:
            print("abs_bbox 정보가 없습니다.")

except FileNotFoundError:
    print("지정한 JSON 파일을 찾을 수 없습니다.")
except json.JSONDecodeError:
    print("JSON 파일 형식이 잘못되었습니다.")
except Exception as e:
    print(f"오류 발생: {e}")

import os
import json


def clamp(value, low, high):
    """간단한 clamp 함수."""
    return max(low, min(value, high))


def validate_and_sort_quad(points, w, h):
    """
    points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    w,h: 이미지 폭/높이
    1) 각 점을 [0..w-1], [0..h-1] 범위로 clamp
    2) minX/minY, maxX/maxY로 사각형 bounding box 계산
    3) 면적이 1px 이하라면 None
    4) [ (minX,minY), (maxX,minY), (maxX,maxY), (minX,maxY) ] 순서로 반환
    """
    if len(points) != 4:
        return None

    # 1) 클리핑
    clipped = []
    for (x, y) in points:
        cx = clamp(x, 0, w - 1)
        cy = clamp(y, 0, h - 1)
        clipped.append((cx, cy))

    # 2) min/max
    xs = [p[0] for p in clipped]
    ys = [p[1] for p in clipped]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # 3) 면적 체크
    if (maxx - minx) < 1 or (maxy - miny) < 1:
        # 면적이 1px 미만이면 무효로 처리
        return None

    # 4) 사각형 좌표를 TL->TR->BR->BL 순서로 구성
    final_quad = [
        [minx, miny],
        [maxx, miny],
        [maxx, maxy],
        [minx, maxy]
    ]
    return final_quad


def convert_segments_with_clamp_and_remove_xxx(input_path, output_path):
    """
    기존 JSON에서:
    1) 'segments' -> 'polygon'(4점) -> clamp/reorder/면적검사
    2) text == 'xxx'인 segment 제거
    3) 'points' 키에 2D 리스트 저장 (ICDAR 형식)

    결과 JSON 구조:
    {
      "annotations": [
        {
          "id": int,
          "text": str,
          "image_id": int or str,
          "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        }, ...
      ],
      "cropLabels": [],
      "images": [
        {
          "file_name": ...,
          "date_created": ...,
          "width": w,
          "id": image_id,
          "height": h
        }
      ],
      "info": {...},
      "metadata": []
    }
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)

    new_data = {
        "annotations": [],
        "cropLabels": [],
        "images": [],
        "info": {
            "date_created": "",
            "name": "",
            "description": ""
        },
        "metadata": []
    }

    # 이미지 정보 추출
    try:
        img_id = int(old_data["image_id"])
    except:
        img_id = old_data.get("image_id", "unknown_id")

    img_w = old_data.get("width", 0)
    img_h = old_data.get("height", 0)

    # images
    new_data["images"].append({
        "file_name": old_data.get("file_name", ""),
        "date_created": "",
        "width": img_w,
        "id": img_id,
        "height": img_h
    })

    ann_id = 1
    old_annotations = old_data.get("annotations", [])

    for ann in old_annotations:
        segments = ann.get("segments", [])
        for seg in segments:
            text_val = seg.get("char", "")
            poly = seg.get("polygon", None)

            # 1) "xxx" 제거
            if text_val == "xxx":
                continue

            # 2) polygon(4점) -> clamp + reorder
            if poly and isinstance(poly, list) and len(poly) == 4:
                sorted_quad = validate_and_sort_quad(poly, img_w, img_h)
                if sorted_quad is None:
                    continue  # 면적이 너무 작거나 invalid

                # 유효하면 annotations에 추가
                new_data["annotations"].append({
                    "id": ann_id,
                    "text": text_val,
                    "image_id": img_id,
                    "points": sorted_quad
                })
                ann_id += 1

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"[완료] {output_path} 처리 후 저장")


def process_json_folder(folder_path):
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith('.json'):
                json_path = os.path.join(dirpath, filename)
                convert_segments_with_clamp_and_remove_xxx(json_path, json_path)


if __name__ == "__main__":
    folder = "/opt/project/datasets/gen_OCR/output/labels/"
    process_json_folder(folder)


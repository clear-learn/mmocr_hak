#!/usr/bin/env python3
# json_fix_area_filter.py
#  · 'xxx'·'###' 박스 삭제
#  · bbox→poly 보강, area 90 % 필터
#  · 최종으로 **원본/제거 박스 개수** 집계 출력

import os, json
from pathlib import Path
from typing import Dict, List, Tuple

IGNORE_TAG  = "xxx"
CATEGORY_ID = 1
FRAC_THRESH = 0.9          # 90 %

def bbox_to_flat_poly(b: List[float]) -> List[float]:
    x, y, w, h = b
    return [x, y, x+w, y, x+w, y+h, x, y+h]

def valid_bbox(b) -> bool:
    return (
        isinstance(b, list) and len(b) == 4
        and all(isinstance(v, (int, float)) and v is not None for v in b)
        and b[2] > 0 and b[3] > 0
    )

# --------------------------------------------------
def fix_json_file(jpath: Path) -> Tuple[bool, int, int]:
    """
    하나의 JSON을 수정하고 저장.
    Returns
    -------
    changed : bool
    orig_cnt: int  (원본 annotation 개수)
    drop_cnt: int  (삭제된 개수)
    """
    try:
        data: Dict = json.loads(jpath.read_text("utf-8"))
    except json.JSONDecodeError:
        print(f"[ERR] JSON 파싱 오류: {jpath}")
        return False, 0, 0

    orig_cnt = len(data.get("annotations", []))   # ★ 원본 개수
    norm_anns: List[Dict] = []
    changed = False
    drop_cnt = 0                                 # ★ 삭제 개수

    # ---------- 1차: 정규화 ----------
    for ann in data.get("annotations", []):
        # 0) ignore 표기면 삭제
        if ann.get("text") in {IGNORE_TAG, "###"}:
            drop_cnt += 1
            changed = True
            continue

        # 1) segments 삭제
        if ann.pop("segments", None) is not None:
            changed = True

        # 2) bbox → polygon
        bbox = ann.pop("bbox", None)
        if bbox is not None:
            if valid_bbox(bbox):
                poly = bbox_to_flat_poly(bbox)
                ann.update({"polygon": [poly], "segmentation": [poly],
                            "area": bbox[2]*bbox[3]})
                changed = True
            else:
                print(f"[WARN] skip invalid bbox {bbox} @ {jpath}")
                drop_cnt += 1
                changed = True
                continue

        # 3) area 보강
        if "area" not in ann:
            flat = ann.get("polygon", ann.get("segmentation", [[0]*8]))[0]
            xs, ys = flat[0::2], flat[1::2]
            ann["area"] = (max(xs)-min(xs))*(max(ys)-min(ys))

        # 4) COCO 필드 보강
        ann.setdefault("category_id", CATEGORY_ID)
        ann.setdefault("iscrowd", 0)

        norm_anns.append(ann)

    # ---------- 2차: 90 % 필터 (박스 ≥2개) ----------
    if len(norm_anns) >= 2:
        total_area = sum(a["area"] for a in norm_anns)
        if total_area > 0:
            kept = [a for a in norm_anns if a["area"]/total_area < FRAC_THRESH]
            drop_cnt += len(norm_anns) - len(kept)
            if len(kept) != len(norm_anns):
                changed = True
            norm_anns = kept

    # ---------- 저장 ----------
    if changed:
        data["annotations"] = norm_anns
        jpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

    return changed, orig_cnt, drop_cnt

# --------------------------------------------------
def fix_json_in_folder(folder_path: str):
    fixed = unchanged = 0
    total_orig = total_drop = 0

    for root, _, files in os.walk(folder_path):
        for fn in files:
            if fn.lower().endswith(".json"):
                jp = Path(root) / fn
                changed, o_cnt, d_cnt = fix_json_file(jp)
                total_orig += o_cnt
                total_drop += d_cnt
                if changed:
                    fixed += 1
                    print(f"[FIXED] {jp}   (orig {o_cnt}, dropped {d_cnt})")
                else:
                    unchanged += 1

    print("\n[SUMMARY]")
    print(f"  files fixed   : {fixed:,}")
    print(f"  files unchanged: {unchanged:,}")
    print(f"  total boxes   : {total_orig:,}")
    print(f"  boxes removed : {total_drop:,}")
    print(f"  boxes kept    : {total_orig-total_drop:,}")

# --------------------------------------------------
if __name__ == "__main__":
    fix_json_in_folder("/opt/project/datasets/OCR_outdoor/OCR_outdoor/train/labels_")

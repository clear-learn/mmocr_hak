#!/usr/bin/env python
# make_rec_from_cocotext_v2_split.py
"""
 1. cocotext.v2.json 하나를 읽어 crop 생성
 2. 전체 crop 중 무작위 5 000장을 val 세트로 분리
"""
import json, cv2, random, shutil
from pathlib import Path

# ───────── 사용자 설정 ───────────────────────────────
json_file  = Path("/opt/project/datasets/cocotext.v2.json")
image_dir  = Path("/opt/project/datasets/train2014")          # COCO 이미지
out_root   = Path("/opt/project/datasets/OCR_outdoor/test")   # 출력 루트
train_dir  = out_root / "coco_text_train"      # 학습 crop
val_dir    = out_root / "coco_text_test"      # 검증 crop
train_gt   = out_root / "gt.txt"
val_gt     = out_root / "val.txt"
holdout_n  = 5_000                         # 검증 개수
img_ext    = ".jpg"
# ───────────────────────────────────────────────────

# 출력 초기화
for d in (train_dir, val_dir):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
for f in (train_gt, val_gt):
    f.write_text("", encoding="utf-8")

def id2path(i: int) -> Path:
    return image_dir / f"COCO_train2014_{i:012d}{img_ext}"

print("🔄 JSON 로드…")
with json_file.open(encoding="utf-8") as f:
    coco = json.load(f)

anns = list(coco["anns"].items())
random.shuffle(anns)                # 무작위 섞기

img_cache, train_cnt, val_cnt = {}, 0, 0

for ann_id, ann in anns:
    txt = (ann.get("utf8_string") or "").strip()
    if txt == "" or ann.get("legibility") != "legible":
        continue

    img_id = ann["image_id"]
    if img_id not in img_cache:
        p = id2path(img_id)
        if not p.is_file():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_cache[img_id] = img
    else:
        img = img_cache[img_id]

    h, w = img.shape[:2]
    xy = ann.get("mask");  xs, ys = xy[0::2], xy[1::2]
    x0, x1 = max(0, int(min(xs))), min(w, int(max(xs)))
    y0, y1 = max(0, int(min(ys))), min(h, int(max(ys)))
    if x1 <= x0 or y1 <= y0:
        continue
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        continue

    is_val = val_cnt < holdout_n
    save_dir = val_dir if is_val else train_dir
    save_name = f"{img_id}_{ann_id}.jpg"
    save_path = save_dir / save_name
    cv2.imwrite(str(save_path), crop)

    rel = save_path.relative_to(out_root).as_posix()
    gt_file = val_gt if is_val else train_gt
    gt_file.open("a", encoding="utf-8").write(f"{rel}\t{txt}\n")
    if is_val: val_cnt += 1
    else:      train_cnt += 1

print(f"✓ train crop {train_cnt}장 ➜ {train_dir}")
print(f"✓ val   crop {val_cnt}장   ➜ {val_dir}")
print(f"✓ gt.txt  : {train_gt}")
print(f"✓ val.txt : {val_gt}")
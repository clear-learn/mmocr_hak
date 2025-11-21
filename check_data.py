#!/usr/bin/env python3
# convert_outdoor_to_yoloseg.py  (pic·labels 분리 없이 저장, 겉박스 IoU≥0.95 제거)

import json, shutil
from pathlib import Path

# ───────── 사용자 설정 ─────────
SRC_ROOT = Path('/opt/project/datasets/OCR_outdoor/OCR_outdoor')
DST_ROOT = Path('/opt/project/datasets/OCR_outdoor/outdoor_yolo')
SPLITS   = ['train', 'val']
THR_IOU  = 0.95            # global_bb 와 IoU 0.95 이상이면 skip
# ──────────────────────────────

# ---------- 보조 함수 ----------
def bbox_to_poly(b):  # [x,y,w,h] → 8 point
    x, y, w, h = b
    return [x, y, x + w, y, x + w, y + h, x, y + h]

def poly2bbox(p):
    xs, ys = p[::2], p[1::2]
    return min(xs), min(ys), max(xs), max(ys)

def iou(bb1, bb2):
    xA, yA = max(bb1[0], bb2[0]), max(bb1[1], bb2[1])
    xB, yB = min(bb1[2], bb2[2]), min(bb1[3], bb2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    a1 = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    a2 = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    return inter / (a1 + a2 - inter)

def yolo_line(poly, w, h):
    n = [poly[i] / (w if i % 2 == 0 else h) for i in range(8)]
    return '0 ' + ' '.join(f'{v:.6f}' for v in n) + '\n'

# ---------- JSON 하나 변환 ----------
def convert(json_fp: Path, split: str):
    try:
        data = json.loads(json_fp.read_text())
    except Exception as e:
        print(f'[SKIP] {json_fp} (JSON error: {e})')
        return

    imgs = {im['id']: im for im in data['images']}
    # rel_dir = ex) 1.간판     (labels/1.간판/xxx.json → '1.간판')
    rel_dir = json_fp.parent.relative_to(SRC_ROOT / split / 'labels')

    # 이미지별 폴리곤 모으기
    polys_by_img = {}
    for ann in data['annotations']:
        im = imgs.get(ann['image_id'])
        if not im:
            continue
        poly = (ann['polygon'][0] if ann.get('polygon')
                else bbox_to_poly(ann['bbox']) if 'bbox' in ann else None)
        if poly and len(poly) >= 8:
            polys_by_img.setdefault(im['id'], []).append(poly)

    for img_id, polys in polys_by_img.items():
        im = imgs[img_id]
        src_img = SRC_ROOT / split / 'pic' / rel_dir / im['file_name']
        if not src_img.exists():
            print(f'[MISS] {src_img}')
            continue

        # 최외곽 bbox
        xs = [p[i] for p in polys for i in range(0, 8, 2)]
        ys = [p[i] for p in polys for i in range(1, 8, 2)]
        global_bb = (min(xs), min(ys), max(xs), max(ys))

        # 출력 디렉터리 (train/1.간판)
        out_dir = DST_ROOT / split / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = (out_dir / im['file_name']).with_suffix('.txt')

        with txt_path.open('w') as f:
            for poly in polys:
                if len(polys) > 1 and iou(poly2bbox(poly), global_bb) >= THR_IOU:
                    continue          # 겉박스 skip
                f.write(yolo_line(poly, im['width'], im['height']))

        dst_img = out_dir / im['file_name']
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

# ---------- split 루프 ----------
def run(split):
    for jf in (SRC_ROOT / split / 'labels').rglob('*.json'):
        convert(jf, split)

    out_dir = DST_ROOT / split
    n_img = sum(1 for p in out_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    n_txt = sum(1 for _ in out_dir.rglob('*.txt'))
    print(f'[{split}] images {n_img:,}  txt {n_txt:,}')

if __name__ == '__main__':
    for sp in SPLITS:
        run(sp)
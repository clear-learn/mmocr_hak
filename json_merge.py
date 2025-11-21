import json
import warnings
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm

# ------------------------------------------------------------
LABEL_ROOT  = Path('/opt/project/datasets/OCR_outdoor/OCR_outdoor/val/labels')
PIC_ROOT    = Path('/opt/project/datasets/OCR_outdoor/OCR_outdoor/val/pic')
OUTPUT_JSON = '/opt/project/datasets/OCR_outdoor/OCR_outdoor/coco_val.json'

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

merged = {
    'images': [],
    'annotations': [],
    'categories': [{'id': 1, 'name': 'text'}]
}

img_id_ctr = 1
ann_id_ctr = 1
skip_no_img = skip_no_poly = 0
patch_seg2pol = patch_box2pol = patch_pol2box = 0
dup_name_warned = set()

# ―― 안전: 깨진 JPG 로딩 허용 ――
ImageFile.LOAD_TRUNCATED_IMAGES = True


def rect_poly_from_box(x, y, w, h):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def bbox_from_poly(flat):
    xs, ys = flat[0::2], flat[1::2]
    x0, y0 = min(xs), min(ys)
    return [int(x0), int(y0), int(max(xs) - x0), int(max(ys) - y0)]


def flatten(pts):
    """[[x,y], …]  또는 [[x1…x8]] → 납작 리스트"""
    return pts[0] if len(pts) == 1 and isinstance(pts[0], (list, tuple)) else [c for p in pts for c in p]


# ------------------------------------------------------------
print('[INFO] indexing images …')
img_index, rel_index = {}, {}
for p in PIC_ROOT.rglob('*'):
    if p.suffix.lower() in VALID_EXT and p.is_file():
        if p.name in img_index and p.name not in dup_name_warned:
            warnings.warn(f'[WARN] Duplicate image name detected: {p.name}')
            dup_name_warned.add(p.name)
        img_index.setdefault(p.name, p)                 # 첫 번째 발견 경로를 우선
        rel_index.setdefault(str(p.relative_to(PIC_ROOT)), p)
print(f'[INFO] {len(img_index):,} images indexed')


def find_img(pstr):            # 파일명 or 상대경로 → Path
    return rel_index.get(pstr) if '/' in pstr else img_index.get(Path(pstr).name)


# ------------------------------------------------------------
def process_json(jfp: Path):
    """JSON 한 개 파일 처리 → merged 에 누적"""
    global img_id_ctr, ann_id_ctr
    global skip_no_img, skip_no_poly
    global patch_seg2pol, patch_box2pol, patch_pol2box

    data = json.loads(jfp.read_text(encoding='utf-8'))
    img_info = data.get('images', [{}])[0]
    fname = img_info.get('file_name', '').strip()
    if not fname:
        return

    img_path = find_img(fname)
    if img_path is None:
        skip_no_img += 1
        return

    w, h = img_info.get('width'), img_info.get('height')
    if w is None or h is None:
        with Image.open(img_path) as im:
            w, h = im.size

    img_id = img_id_ctr
    img_id_ctr += 1
    merged['images'].append({
        'id': img_id,
        'file_name': str(img_path.relative_to(PIC_ROOT)),
        'width': int(w),
        'height': int(h)
    })

    # ---------- annotation ----------
    for ann in data.get('annotations', []):
        # ---- 텍스트 / ignore 처리 (원본 ann 단위이므로 아래에서 그대로 복사) ----
        txt = (ann.get('text') or ann.get('transcription') or '').strip()
        iscrowd_flag = 1 if txt.lower() in {'###', 'xxx'} else 0

        # ---- polygon 후보 수집 ----
        poly_candidates = []

        # 1) points (대개 단일 폴리곤)
        if 'points' in ann and len(ann['points']) >= 4:
            poly_candidates.append(flatten(ann['points']))

        # 2) segmentation (다중 폴리곤 가능) → 폴리곤마다 분리
        elif ann.get('segmentation'):
            segs = ann['segmentation']
            if isinstance(segs, list):
                for seg in segs:
                    # COCO 형식: [x1, y1, x2, y2, ...]
                    if isinstance(seg, (list, tuple)) and seg and isinstance(seg[0], (int, float)):
                        poly_candidates.append(list(seg))             # 그대로
                    # DBNet 형식: [[x,y], [x,y], ...]
                    elif isinstance(seg, (list, tuple)) and seg and isinstance(seg[0], (list, tuple)):
                        poly_candidates.append(flatten(seg))
            patch_seg2pol += 1

        # 3) bbox → 사각 폴리곤
        elif 'bbox' in ann and len(ann['bbox']) == 4:
            x, y, wb, hb = ann['bbox']
            if None in (x, y, wb, hb) or wb <= 0 or hb <= 0:
                skip_no_poly += 1
                continue
            poly_candidates.append(rect_poly_from_box(x, y, wb, hb))
            patch_box2pol += 1

        # ---- 각 폴리곤 → 개별 annotation 등록 ----
        for poly in poly_candidates:
            if poly is None or len(poly) < 8:
                skip_no_poly += 1
                continue

            # bbox 재계산 (폴리곤 단위)
            bbox_vals = bbox_from_poly(poly)
            patch_pol2box += 1

            merged['annotations'].append({
                'id': ann_id_ctr,
                'image_id': img_id,
                'category_id': 1,
                'bbox': bbox_vals,
                'polygon': [poly],            # 하나의 폴리곤만
                'segmentation': [poly],       # COCO 규격
                'area': bbox_vals[2] * bbox_vals[3],
                'iscrowd': iscrowd_flag,
                'text': txt
            })
            ann_id_ctr += 1


# ------------------------------------------------------------
def main():
    for jf in tqdm(list(LABEL_ROOT.rglob('*.json')), desc='Merging'):
        process_json(jf)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        # 가독성보다 용량·속도를 우선 → indent 생략
        json.dump(merged, f, ensure_ascii=False, separators=(',', ':'))

    print(f'\n[DONE] {OUTPUT_JSON}')
    print(f'  images               : {len(merged["images"]):,}')
    print(f'  annotations          : {len(merged["annotations"]):,}')
    print(f'  seg→poly patched     : {patch_seg2pol:,}')
    print(f'  bbox→poly patched    : {patch_box2pol:,}')
    print(f'  poly→bbox patched    : {patch_pol2box:,}')
    print(f'  skip(no poly)        : {skip_no_poly:,}')
    print(f'  skip(no image)       : {skip_no_img:,}')


if __name__ == '__main__':
    main()
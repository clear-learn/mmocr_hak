"""
blur_outside_labels_yolo_parallel.py
────────────────────────────────────
YOLO-Seg txt ↔ 이미지가 같은 폴더에 있을 때
라벨 박스 周 margin 은 그대로, 나머지는 가우시안 블러.

멀티프로세스 병렬 처리 버전
"""

import cv2, random, json, os
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ── 설정값 ─────────────────────────────────────── #
SRC_ROOT   = Path("/opt/project/datasets/OCR_outdoor/outdoor_yolo_fix/val_")
DST_ROOT   = Path("/opt/project/datasets/OCR_outdoor/outdoor_yolo_fix/blur_val")

MARGIN_PX        = 15
SIGMA_BLUR_RANGE = (5.0, 10.0)
APPLY_PROB       = 0.8
EXTS_IMG         = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
NUM_WORKERS      = max(cpu_count() - 1, 1)   # 전 코어-1
CHUNKSIZE        = 50                        # 파일당 task 묶음
# ──────────────────────────────────────────────── #


def yolo_txt_to_polys(txt_path: Path, img_w: int, img_h: int):
    polys = []
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 9:                 # cls + 최소 4점
            continue
        coords = list(map(float, parts[1:]))
        pts = [[coords[i] * img_w, coords[i + 1] * img_h]
               for i in range(0, len(coords), 2)]
        polys.append(pts)
    return polys


def blur_outside(img_bgr: np.ndarray, polys, margin, sigma):
    h, w = img_bgr.shape[:2]
    keep = np.zeros((h, w), np.uint8)

    for poly in polys:
        if len(poly) >= 3:
            cv2.fillPoly(keep, [np.int32(poly)], 255)

    if margin > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        keep = cv2.dilate(keep, k, iterations=1)

    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigma, sigma)
    out = img_bgr.copy()
    out[keep == 0] = blur[keep == 0]
    return out


# ── 워커 함수 ───────────────────────────────────── #
def process_one(img_path_str: str):
    img_path = Path(img_path_str)
    if img_path.suffix.lower() not in EXTS_IMG:
        return
    if random.random() > APPLY_PROB:
        return

    txt_path = img_path.with_suffix(".txt")
    if not txt_path.exists():
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] read fail {img_path}")
        return

    polys = yolo_txt_to_polys(txt_path, img.shape[1], img.shape[0])
    if not polys:
        return

    sigma = random.uniform(*SIGMA_BLUR_RANGE)
    out_img = blur_outside(img, polys, MARGIN_PX, sigma)

    # 저장
    rel      = img_path.relative_to(SRC_ROOT)
    out_i_pt = DST_ROOT / rel
    out_i_pt.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_i_pt), out_img)

    out_t_pt = DST_ROOT / txt_path.relative_to(SRC_ROOT)
    out_t_pt.parent.mkdir(parents=True, exist_ok=True)
    out_t_pt.write_text(txt_path.read_text("utf-8"), "utf-8")

    print(f"[SAVED] {out_i_pt}  (σ={sigma:.2f})")


def main():
    img_files = [str(p) for p in SRC_ROOT.rglob("*")
                 if p.suffix.lower() in EXTS_IMG]

    with Pool(NUM_WORKERS, maxtasksperchild=100) as pool:
        pool.map(process_one, img_files, chunksize=CHUNKSIZE)

    print("All done.")


if __name__ == "__main__":
    # 각 워커에서 OpenCV 멀티스레드를 끄면 과도한 스레드 폭주 방지
    cv2.setNumThreads(0)
    main()
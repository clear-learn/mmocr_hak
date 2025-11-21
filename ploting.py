import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------
JSON_PATH = Path(
    '/opt/project/datasets/OCR_outdoor/OCR_outdoor/val/labels/01.총류/책표지_총류_002109.json'
)
IMG_PATH  = Path(
    '/opt/project/datasets/OCR_outdoor/OCR_outdoor/val/pic/01.총류/책표지_총류_002109.jpg'
)
SAVE_PATH = IMG_PATH.with_stem(IMG_PATH.stem + '_vis')  # 원본이름+_vis.jpg
# 시각화 설정
POINT_SIZE  = 6
LINE_THICK  = 2
POINT_COLOR = (0, 255, 0)   # 초록 (BGR)
LINE_COLOR  = (255, 0, 0)   # 파랑 (BGR)
# ----------------------------------------------------------------

def draw_polygons(img, polygons):
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(img, [pts], True, LINE_COLOR, LINE_THICK, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(img, (x, y), POINT_SIZE, POINT_COLOR, -1, cv2.LINE_AA)
    return img

def main():
    # 1) JSON 읽기
    data = json.loads(JSON_PATH.read_text(encoding='utf-8'))

    # 2) 이미지 한 장 로드
    if not IMG_PATH.exists():
        raise FileNotFoundError(IMG_PATH)
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        raise RuntimeError(f'cv2.imread 실패: {IMG_PATH}')

    # 3) 폴리곤 시각화
    for ann in data['annotations']:
        for poly in ann['polygon']:
            draw_polygons(img, [poly])

    # 4) 저장
    cv2.imwrite(str(SAVE_PATH), img)
    print(f'[SAVED] {SAVE_PATH}')

    # 5) 화면에도 확인(선택)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    main()
import os
import json
import cv2

LABEL_ROOT = "/opt/project/datasets/야외데이터/야외_데이터/train/labels"
PIC_ROOT = "/opt/project/datasets/야외데이터/야외_데이터/train/pic"

OUTPUT_DIR = "/opt/project/datasets/OCR/train_crops"  # 잘라낸 이미지를 저장할 폴더
OUTPUT_LABEL = "/opt/project/datasets/OCR/train_label.txt"  # 인식용 라벨 파일 (MMOCR가 읽을 것)

def is_json_file(filename):
    return filename.lower().endswith('.json')

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_LABEL, 'w', encoding='utf-8') as label_f:

        for root, dirs, files in os.walk(LABEL_ROOT):
            for filename in files:
                if not is_json_file(filename):
                    continue

                json_path = os.path.join(root, filename)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # images[]
                images_arr = data.get("images", [])
                if not images_arr:
                    print(f"[WARN] No images in {json_path}")
                    continue
                img_info = images_arr[0]
                old_file_name = img_info.get("file_name", "")
                if not old_file_name:
                    print(f"[WARN] image file_name missing in {json_path}")
                    continue

                # 원본 이미지 경로 (상대 폴더 고려)
                rel_json_path = os.path.relpath(json_path, LABEL_ROOT)
                dir_part = os.path.dirname(rel_json_path)  # ex) folderA
                image_rel_path = os.path.join(dir_part, old_file_name)  # folderA/image.jpg
                real_img_path = os.path.join(PIC_ROOT, image_rel_path)

                if not os.path.isfile(real_img_path):
                    print(f"[WARN] Image not found: {real_img_path}")
                    continue

                img = cv2.imread(real_img_path)
                if img is None:
                    print(f"[WARN] Failed to read {real_img_path}")
                    continue

                annotations = data.get("annotations", [])
                if not annotations:
                    print(f"[WARN] No annotations in {json_path}")
                    continue

                base_id = os.path.splitext(old_file_name)[0]

                ann_idx = 0
                H, W = img.shape[:2]

                for ann in annotations:
                    text = ann.get("text", "").strip()
                    if not text:
                        continue
                    if text == "xxx":
                        print(f"[INFO] Skip annotation with text='xxx' in {json_path}")
                        continue

                    points = ann.get("points", None)
                    if not points or len(points) != 4:
                        print(f"[WARN] Invalid or missing 'points' in {json_path}: {points}")
                        continue

                    # points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    minx, maxx = min(xs), max(xs)
                    miny, maxy = min(ys), max(ys)

                    # 정수 변환
                    try:
                        minx, maxx = int(minx), int(maxx)
                        miny, maxy = int(miny), int(maxy)
                    except ValueError:
                        print(f"[WARN] points not integer-convertible in {json_path}: {points}")
                        continue

                    if minx < 0 or miny < 0 or maxx > W or maxy > H:
                        print(f"[WARN] points out of range in {real_img_path}: {points}")
                        continue

                    w = maxx - minx
                    h = maxy - miny
                    if w <= 0 or h <= 0:
                        print(f"[WARN] zero or negative area: {points}")
                        continue

                    crop = img[miny:maxy, minx:maxx]
                    if crop.size == 0:
                        continue

                    crop_filename = f"{base_id}_{ann_idx}.jpg"
                    save_path = os.path.join(OUTPUT_DIR, crop_filename)
                    cv2.imwrite(save_path, crop)
                    ann_idx += 1

                    label_line = f"{OUTPUT_DIR}/{crop_filename}\t{text}\n"
                    label_f.write(label_line)

    print(f"Done! Cropped images saved in '{OUTPUT_DIR}' and label file: '{OUTPUT_LABEL}'")

if __name__ == "__main__":
    main()

# make_rec_dataset.py
import json, cv2, shutil
from pathlib import Path

# ────────────────────────────────────────────────────────────
# 1️⃣  경로만 환경에 맞게 수정하세요
root_dir      = Path("/opt/project/datasets/OCR_outdoor/OCR_outdoor/val")
label_root    = root_dir / "labels"     # labels/…/*.json
image_root    = root_dir / "pic"        # pic/…/*.jpg
out_img_root  = root_dir / "rec_images" # 잘린 패치 저장
gt_txt_path   = root_dir / "gt.txt"     # <경로>\t<라벨>
skip_token    = "xxx"                   # 건너뛸 라벨
# ────────────────────────────────────────────────────────────

# 이전 결과 삭제 후 새로 생성
if out_img_root.exists():
    shutil.rmtree(out_img_root)
out_img_root.mkdir(parents=True, exist_ok=True)

lines = []
for json_file in label_root.rglob("*.json"):
    rel_subdir = json_file.parent.relative_to(label_root)   # e.g. 1.title
    img_subdir = image_root / rel_subdir                    # pic/1.title

    with json_file.open(encoding="utf-8") as f:
        coco = json.load(f)
    img_map = {d["id"]: d["file_name"] for d in coco["images"]}

    for ann in coco["annotations"]:
        # ① 라벨 필터
        if ann.get("text", "").strip() == skip_token:
            continue

        img_path = img_subdir / img_map.get(ann["image_id"], "")
        if not img_path.is_file():
            print("⚠️  이미지 없음:", img_path);  continue

        # ② 좌표 소스 선택: polygon → segmentation → bbox
        if "polygon" in ann:
            xy = ann["polygon"][0]
        elif "segmentation" in ann and ann["segmentation"]:
            xy = ann["segmentation"][0]
        elif "bbox" in ann:
            x, y, w_box, h_box = ann["bbox"]
            xy = [x, y, x + w_box, y, x + w_box, y + h_box, x, y + h_box]
        else:
            print("⚠️  좌표 없음 SKIP:", json_file, ann["id"]);  continue

        xs, ys = xy[0::2], xy[1::2]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)

        # ③ 이미지 읽기
        img = cv2.imread(str(img_path))
        if img is None:
            print("⚠️  읽기 실패:", img_path);  continue
        h, w = img.shape[:2]

        # ④ 음수 클램프(좌·상) + 빈 bbox 체크
        x0 = max(0, x0)
        y0 = max(0, y0)
        if x1 <= x0 or y1 <= y0:
            print("⚠️  무효 bbox:", img_path, ann["id"]);  continue

        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            print("⚠️  빈 crop :", img_path, ann["id"]);  continue

        # ⑤ 저장
        crop_dir = out_img_root / rel_subdir
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_name = f"{img_path.stem}_{ann['id']}.jpg"
        crop_path = crop_dir / crop_name
        if not cv2.imwrite(str(crop_path), crop):
            print("⚠️  저장 실패:", crop_path);  continue

        # ⑥ gt.txt 라인 추가
        rel_path = crop_path.relative_to(root_dir).as_posix()
        lines.append(f"{rel_path}\t{ann['text']}")

# ⑦ gt.txt 저장
with gt_txt_path.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✓ 잘린 패치: {len(lines)} 장  ➜  {out_img_root}")
print(f"✓ GT 목록   : {gt_txt_path}")
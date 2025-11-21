#!/usr/bin/env python
# merge_gt_jsonl_files.py
# ─────────────────────────────────────────
"""
예시 라인
{"filename": "rec_images/가로형간판/간판_가로형간판_169453_1.jpg", "text": "영어"}

● in_paths  : 합칠 입력 파일 목록(갯수 제한 없음)
● out_path  : 결과를 쓸 경로
  – 이미 존재하면 덮어씀
  – 중복 정의( filename·text 완전 동일 )은 한 번만 남김
  – filename 기준으로 정렬해 저장
"""
import json
from pathlib import Path

# ─── 입력 · 출력 파일 지정 ────────────────────────
in_paths = [
    Path("/opt/project/datasets/OCR_outdoor/recog_outdoor/val/labels/gt_jsonl.txt"),
    Path("/opt/project/datasets/OCR_outdoor/test/gt_jsonl.txt"),
]
out_path = Path("/opt/project/datasets/OCR_outdoor/recog_outdoor/val/gt_jsonl.txt")
# ───────────────────────────────────────────────

seen = set()
merged = []

for fp in in_paths:
    with fp.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️  잘못된 JSON : {fp} → {line[:50]}");  continue

            key = (obj.get("filename"), obj.get("text"))
            if key in seen:
                continue

            seen.add(key)
            merged.append(obj)

# filename 기준 정렬 (필요 없으면 다음 줄 삭제)
merged.sort(key=lambda o: o["filename"])

# 저장
with out_path.open("w", encoding="utf-8") as f:
    for obj in merged:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✓ 입력 {len(in_paths)} 파일 → 합쳐진 {len(merged)} 라인")
print(f"✓ 결과 파일 : {out_path}")
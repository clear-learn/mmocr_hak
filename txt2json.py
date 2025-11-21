# txt2jsonl.py
import json
from pathlib import Path

# 1) 📂 루트 디렉터리
root = Path('/opt/project/datasets/OCR_outdoor/test/')  # ← 맨 앞 / 확인!

# 2) 입력/출력 파일
in_txt  = root / 'val.txt'
out_txt = root / 'gt_jsonl.txt'

# 3) 변환
with in_txt.open(encoding='utf-8') as fin, \
     out_txt.open('w', encoding='utf-8') as fout:
    for line in fin:
        if not line.strip():
            continue            # 빈 줄 skip
        path_, text = line.rstrip('\n').split('\t', 1)
        obj = dict(filename=path_, text=text)
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f'saved → {out_txt}')

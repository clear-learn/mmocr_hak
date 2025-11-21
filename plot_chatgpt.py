import matplotlib
matplotlib.use('Agg')

import json
import matplotlib.pyplot as plt
from PIL import Image

# JSON 파일 불러오기
with open('/opt/project/datasets/OCR_outdoor/test/test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 이미지 불러오기
img = Image.open('/opt/project/datasets/OCR_outdoor/test/229791537.jpg')

# matplotlib으로 이미지 띄우기
fig, ax = plt.subplots()
ax.imshow(img)

# annotation 순회하며 점 그리기
for ann in data["annotations"]:
    # center 좌표 가져오기
    y_center, x_center = ann["center"]

    # 점 그리기 (marker='o'로 빨간 점 표시)
    ax.plot(x_center, y_center, marker='o', color='blue', markersize=5)

# 축 제거
plt.axis('off')

# 파일로 저장
plt.savefig('points_only.png', bbox_inches='tight', dpi=200)
print("결과 이미지가 points_only.png에 저장되었습니다.")
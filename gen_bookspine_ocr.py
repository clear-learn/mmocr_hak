import os
import random
import re
import string
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
from pathlib import Path
from math import cos, sin, pi
import colorsys, random

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

SAVE_JSON = False

OUT_PIC_DIR   = Path('/opt/project/datasets/gen_OCR/output/pic')
OUT_LABEL_DIR = Path('/opt/project/datasets/gen_OCR/output/labels')

CLS_ID    = 0

pattern_allow = re.compile(r'^[\uAC00-\uD7A3a-zA-Z0-9\s.,!?@#&\-_=+\'"<>;:|`~\[\]{}]*$')
html_drop = re.compile(r'&#\d+;')

def int_to_roman(n: int) -> str:
    """
    1 ≤ n ≤ 3999  →  Roman numeral 문자열
    범위를 벗어나면 그냥 str(n) 반환
    """
    if n <= 0 or n > 3999:
        return str(n)

    vals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"),  (90,  "XC"), (50,  "L"), (40,  "XL"),
        (10,  "X"),  (9,   "IX"), (5,   "V"), (4,   "IV"),
        (1,   "I")
    ]
    res = []
    for v, sym in vals:
        while n >= v:
            res.append(sym)
            n -= v
    return ''.join(res)

def biased_random_num(p_small=0.35, small_max=30, big_max=999):

    if random.random() < p_small:
        return random.randint(1, small_max)
    else:
        return random.randint(small_max+1, big_max)

def random_shape_and_text_color(min_contrast=4.5):
    """
    WCAG 대비비(contrast ratio) ≥ min_contrast 를 만족하는
    (shape_color, text_color) 한 쌍을 랜덤으로 반환.
    """
    def luminance(rgb):
        def f(c):
            c = c/255
            return c/12.92 if c<=0.03928 else ((c+0.055)/1.055)**2.4
        r,g,b = map(f, rgb)
        return 0.2126*r + 0.7152*g + 0.0722*b

    def contrast(c1, c2):
        L1, L2 = luminance(c1), luminance(c2)
        if L1 < L2: L1, L2 = L2, L1
        return (L1+0.05)/(L2+0.05)

    # ① 랜덤한 shape 색 하나 뽑기 (HSV → RGB)
    h = random.random()
    s = 0.6 + random.random()*0.4        # 채도 0.6~1.0
    v = 0.6 + random.random()*0.4        # 밝기 0.6~1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    shape = tuple(int(x*255) for x in (r, g, b))

    # ② 흑/백 먼저 시험
    for cand in [(0,0,0), (255,255,255)]:
        if contrast(shape, cand) >= min_contrast:
            return shape, cand

    # ③ 안 되면 Hue 반대(보색)로 계산
    h2 = (h + 0.5) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h2, s, 1-v+0.2)  # 밝기 반전
    text = tuple(int(x*255) for x in (r2, g2, b2))

    # 여전히 대비가 부족하면 밝기·채도를 살짝 보정
    if contrast(shape, text) < min_contrast:
        if luminance(text) > 0.5: text = (0,0,0)
        else: text = (255,255,255)
    return shape, text


def draw_shape_number(num_str,
                      font,
                      shape="circle",           # "circle" | "star" | "heart" | "triangle"
                      pad=12,                   # 숫자와 테두리 간격(px)
                      fill_shape=(255,255,255), # 도형 채움색
                      outline_shape=None,       # 도형 외곽선색
                      fill_num=(0,0,0),         # 숫자 색
                      stroke_width=1):
    """
    num_str   : "7" 또는 "123" 등 문자열
    font      : PIL.ImageFont
    반환      : (PIL.Image, polygon 4pts, seg list)
    """
    draw_tmp = ImageDraw.Draw(Image.new("RGBA",(1,1)))
    l,t,r,b  = draw_tmp.textbbox((0,0), num_str, font=font, stroke_width=stroke_width)
    w_txt, h_txt = r-l, b-t

    # 도형 크기: 텍스트 바깥으로 pad 만큼
    w_box = w_txt + 2*pad
    h_box = h_txt + 2*pad
    size  = max(w_box, h_box)          # 정사각형 캔버스(도형용)
    img   = Image.new("RGBA", (size, size), (0,0,0,0))
    draw  = ImageDraw.Draw(img)

    cx, cy = size//2, size//2
    r_in   = size//2

    if shape == "circle":
        draw.ellipse([cx-r_in, cy-r_in, cx+r_in, cy+r_in],
                     fill=fill_shape, outline=outline_shape)
    elif shape == "triangle":
        pts = [(cx, cy-r_in),
               (cx+r_in*cos(pi/6), cy+r_in*sin(pi/6)),
               (cx-r_in*cos(pi/6), cy+r_in*sin(pi/6))]
        draw.polygon(pts, fill=fill_shape, outline=outline_shape)
    elif shape == "star":
        pts=[]
        for i in range(10):
            ang = pi/2 + i*pi/5
            radius = r_in if i%2==0 else r_in*0.4
            pts.append((cx+radius*cos(ang), cy-radius*sin(ang)))
        draw.polygon(pts, fill=fill_shape, outline=outline_shape)
    elif shape == "heart":
        path = []
        for t_ in range(0,361,5):
            t_rad = pi*t_/180
            x = 16*sin(t_rad)**3
            y = 13*cos(t_rad) - 5*cos(2*t_rad) - 2*cos(3*t_rad) - cos(4*t_rad)
            path.append((cx + x*r_in/18, cy - y*r_in/18))
        draw.polygon(path, fill=fill_shape, outline=outline_shape)
    else:
        draw.rectangle([0,0,size,size], fill=fill_shape)

    # 숫자 그리기 중앙 정렬
    txt_x = (size - w_txt)//2
    txt_y = (size - h_txt)//2
    draw.text((txt_x, txt_y), num_str, font=font,
              fill=fill_num, stroke_width=stroke_width, stroke_fill=fill_num)

    # YOLO/세그용 폴리곤(전체 사각)
    poly = [[0,0],[size,0],[size,size],[0,size]]
    segs = [{"char": num_str, "polygon": poly}]
    return img, poly, segs

def _oriented_bbox_from_segs(segs):
    """
    세그먼트(글자) 좌표들을 이용해
    cv2.minAreaRect 로 가장 작은 회전 사각형 반환.
    """
    if not segs:
        return []
    pts = np.concatenate([np.array(s["polygon"], np.float32) for s in segs])
    rect = cv2.minAreaRect(pts)            # (ctr(x,y), (w,h), angle)
    box  = cv2.boxPoints(rect)             # 4×2
    return box.tolist()

def _crop_preserve_oriented(img, segs):
    """
    투명 여백을 crop 하고, segs 좌표를 동일 오프셋만큼 이동.
    (block_poly 는 호출한 쪽에서 다시 만들도록 반환하지 않음)
    """
    bbox = img.getbbox()      # (l,t,r,b) or None
    if not bbox:
        return img, segs, []
    l, t, r, b = bbox
    img = img.crop(bbox)
    dx, dy = -l, -t

    for s in segs:
        s["polygon"] = [[x + dx, y + dy] for x, y in s["polygon"]]

    new_block = _oriented_bbox_from_segs(segs)
    return img, segs, new_block

# ─── 1) helper 추가 ──────────────────────────────────────────────
def _transform_polygon(poly, M_aff2x3, M_persp):
    """단일 폴리곤(리스트)을 Affine→Perspective 순서로 변환"""
    if not poly:       # 빈 리스트 처리f
        return []
    pts = cv2.transform(
            np.float32(poly).reshape(-1,1,2),
            M_aff2x3)
    pts = cv2.perspectiveTransform(pts, M_persp)
    return pts.reshape(-1, 2).tolist()

def apply_affine_and_perspective(img, polys,
                                 max_rot=2, max_shear=2,
                                 max_scale=0.05, max_shift=4,
                                 max_disp=40):
    """
    1) 작은 Affine 변형
    2) 원근 Perspective 변형
    img    : PIL.Image (RGBA)
    polys  : list of 4-point list [[x1,y1],…] for title/author/publisher
    반환   : (new_img, new_polys)
    """
    w, h = img.size

    # ───────────────────────────
    # (1) Affine 변형 파라미터
    ang   = np.deg2rad(np.random.uniform(-max_rot, max_rot))
    shear = np.deg2rad(np.random.uniform(-max_shear, max_shear))
    sx = sy = 1.0 + np.random.uniform(-max_scale, max_scale)
    tx = np.random.uniform(-max_shift, max_shift)
    ty = np.random.uniform(-max_shift, max_shift)

    # Affine 3×3 행렬
    M_rot   = np.array([[ np.cos(ang), -np.sin(ang), 0],
                        [ np.sin(ang),  np.cos(ang), 0],
                        [          0,           0, 1]], np.float32)
    M_scale = np.array([[sx,  0, 0],
                        [ 0, sy, 0],
                        [ 0,  0, 1]], np.float32)
    M_shear = np.array([[1, np.tan(shear), 0],
                        [0,            1, 0],
                        [0,            0, 1]], np.float32)
    M_trans = np.array([[1, 0, tx],
                        [0, 1, ty],
                        [0, 0, 1]], np.float32)

    M_affine = M_trans @ M_shear @ M_scale @ M_rot      # 3×3
    M_aff2x3 = M_affine[:2]                             # cv2.warpAffine 용

    # ───────────────────────────
    # (2) Perspective 변형
    # 원본 사각형
    src = np.float32([[0,0], [w,0], [w,h], [0,h]])
    # 각 코너를 max_disp 픽셀 이내로 랜덤 이동
    dst = src + np.random.uniform(-max_disp, max_disp, src.shape).astype(np.float32)
    M_persp = cv2.getPerspectiveTransform(src, dst)    # 3×3

    # ───────────────────────────
    # (3) 이미지에 먼저 Affine → 이어서 Perspective
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    warped_aff = cv2.warpAffine(arr, M_aff2x3, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0,0,0,0))
    warped = cv2.warpPerspective(warped_aff, M_persp, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0,0,0,0))
    new_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA))

    # ───────────────────────────
    # (4) 폴리곤에도 동일하게
    new_polys = []
    for poly in polys:
        if not poly:
            new_polys.append([])
            continue

        pts = np.float32(poly).reshape(-1, 1, 2)
        aff_pts   = cv2.transform(pts, M_aff2x3)
        persp_pts = cv2.perspectiveTransform(aff_pts, M_persp)
        new_polys.append(persp_pts.reshape(-1, 2).tolist())

    return new_img, new_polys, M_aff2x3, M_persp     # ← 행렬 2개 추가

def push_inside(poly, W, H):
    # 폴리곤이 없으면 그대로 반환
    if not poly:
        return poly, (0, 0)

    xs = [x for x, _ in poly]
    ys = [y for _, y in poly]

    dx = -min(0, min(xs))
    dy = -min(0, min(ys))
    dx2 = max(0, max(xs) + dx - (W - 1))
    dy2 = max(0, max(ys) + dy - (H - 1))
    dx -= dx2
    dy -= dy2

    return [[x + dx, y + dy] for x, y in poly], (dx, dy)

# ── NEW: 작은 랜덤 기하 변형 ─────────────────────────
def jitter_image_and_polygons(pil_img,
                              segs,
                              block_poly,         # ← 블록은 버려도 OK, 새로 만들 예정
                              max_angle=5,
                              max_shift=3,
                              pad=50):
    """
    pad → 회전/이동 → crop → oriented bbox 재계산
    반환: (새 이미지, 세그 리스트, 회전된 tight 블록 폴리곤)
    """
    if pil_img is None or pil_img.width <= 1 or pil_img.height <= 1:
        return pil_img, segs, block_poly

    # ─── ① pad 캔버스 ─────────────────────────────
    W0, H0 = pil_img.size
    padded = Image.new("RGBA", (W0 + 2 * pad, H0 + 2 * pad), (0, 0, 0, 0))
    padded.paste(pil_img, (pad, pad))

    def add_pad(poly): return [[x + pad, y + pad] for x, y in poly]
    segs = [{"char": s["char"], "polygon": add_pad(s["polygon"])} for s in segs]

    # ─── ② 회전 + 평행이동 ────────────────────────
    ang = random.uniform(-max_angle, max_angle)
    dx  = random.randint(-max_shift, max_shift)
    dy  = random.randint(-max_shift, max_shift)

    W, H = padded.size
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy

    arr_rot = cv2.warpAffine(np.array(padded), M, (W, H),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    rot_img = Image.fromarray(arr_rot)

    def transform(poly):
        if not poly:
            return []
        pts = cv2.transform(np.float32(poly).reshape(-1, 1, 2), M)
        return pts.reshape(-1, 2).tolist()

    segs = [{"char": s["char"], "polygon": transform(s["polygon"])} for s in segs]

    # ─── ③ crop + tight oriented box 재계산 ───────
    rot_img, segs, block_poly = _crop_preserve_oriented(rot_img, segs)

    return rot_img, segs, block_poly            # 순서: img, segs, poly

def shift(segs, off):
    ox, oy = off
    for s in segs:
        s["polygon"] = [[x+ox, y+oy] for x, y in s["polygon"]]
    return segs

def tight_bbox_from_segments(segments):
    """segments가 비면 None 반환; 아니면 4점 리스트."""
    if not segments:
        return None       # ← None으로 명확히 표현
    xs = [x for seg in segments for x, _ in seg["polygon"]]
    ys = [y for seg in segments for _, y in seg["polygon"]]
    return [[min(xs), min(ys)],
            [max(xs), min(ys)],
            [max(xs), max(ys)],
            [min(xs), max(ys)]]

def clamp_poly(poly, w, h):
    return [[max(0, min(x, w-1)), max(0, min(y, h-1))] for x, y in poly]

# ── NEW: crop + 좌표보정 헬퍼 ───────────────────────
def _crop_and_fix_coords(img, overall_poly, seg_anns):
    bbox = img.getbbox()
    if not bbox:                     # 비어 있으면 그대로
        return img, overall_poly, seg_anns

    l, t, r, b = bbox
    img = img.crop(bbox)
    dx, dy = -l, -t

    # 글자(segments) 좌표 평행 이동
    for s in seg_anns:
        s["polygon"] = [[x + dx, y + dy] for x, y in s["polygon"]]

    if seg_anns:
        overall_poly = tight_bbox_from_segments(seg_anns)
    else:
        w, h = img.size
        overall_poly = [[0, 0], [w, 0], [w, h], [0, h]]

    return img, overall_poly, seg_anns


# ── YOLO-Seg helper (bbox+poly 8점) ─────────────
def poly_to_yolo_seg_line(poly, img_w, img_h):
    """
    YOLO-Seg 라벨 한 줄
      0  x1 y1 x2 y2 … xN yN    (좌표는 0~1 정규화, N은 짝수)
    """
    flat_norm = []
    for x, y in poly:              # poly 안의 모든 점 사용
        flat_norm.extend([x / img_w, y / img_h])
    return f"{CLS_ID} " + ' '.join(f"{v:.6f}" for v in flat_norm) + '\n'

def is_allowed(row):
    text = ' '.join(map(str, row.values))
    if html_drop.search(text):
        return False
    return bool(pattern_allow.fullmatch(text))

def get_values(path):
    df = pd.read_csv(path, engine="python")
    filtered_df = df[df['몰타입(branchtype)'].isin(['국내도서', '외국도서'])]
    sample_en = filtered_df[filtered_df['몰타입(branchtype)'] == '외국도서']
    sample_en = sample_en[sample_en.apply(is_allowed, axis=1)].sample(n=100000)
    sample_kr = filtered_df[filtered_df['몰타입(branchtype)'] == '국내도서']
    sample_kr = sample_kr[sample_kr.apply(is_allowed, axis=1)].sample(n=100000)

    df = pd.concat([sample_en, sample_kr]).reset_index(drop=True)
    df = df[['ItemId','제목(title)', '부제(subtitle)', '저자(author)', '출판사(publisher)']]
    df.rename(columns={
        'ItemId': 'itemid',
        '제목(title)': 'title',
        '부제(subtitle)': 'subtitle',
        '저자(author)': 'author',
        '출판사(publisher)': 'publisher'
    }, inplace=True)

    df['titles'] = df['title'].fillna('') + ' ' + df['subtitle'].fillna('')
    df['publisher'] = df['publisher'].fillna('').str.replace(r'\s*\([^)]*\)', '', regex=True)
    df['author'] = df['author'].fillna('')

    new_df = df[['itemid', 'titles', 'author', 'publisher']]
    return new_df


def load_all_images_in_subfolders(root_folder):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(valid_exts):
                full_path = os.path.join(dirpath, filename)
                image_paths.append(full_path)
    return image_paths

def load_random_bg(root_folder):
    image_paths = load_all_images_in_subfolders(root_folder)
    if not image_paths:
        raise FileNotFoundError(f"No valid image found in subfolders of {root_folder}")
    chosen = random.choice(image_paths)
    bg_img = Image.open(chosen).convert("RGBA")
    return bg_img

def pick_contrasting_color(bg_img, low_range=90, high_range=195):
    np_bg = np.array(bg_img)
    if np_bg.shape[2] == 4:
        np_bg = np_bg[:, :, :3]
    r_avg = np_bg[:, :, 0].mean()
    g_avg = np_bg[:, :, 1].mean()
    b_avg = np_bg[:, :, 2].mean()
    brightness = 0.299*r_avg + 0.587*g_avg + 0.114*b_avg

    if brightness >= 128:
        return (
            random.randint(0, low_range),
            random.randint(0, low_range),
            random.randint(0, low_range)
        )
    else:
        return (
            random.randint(high_range, 255),
            random.randint(high_range, 255),
            random.randint(high_range, 255)
        )

def draw_horizontal_text_field_wrapped(text, font,
                                       max_width,
                                       letter_spacing=2,
                                       inter_line_spacing=10,
                                       padding=1,
                                       fill_color=(0, 0, 0),
                                       stroke_width=0):
    """
    가로쓰기 + 줄바꿈
    """
    if not text.strip():
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), [], []

    draw_temp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    words = re.split(r'\s+', text.strip())
    lines = []
    current_line = []
    current_line_width = 0

    word_dims = []
    for word in words:
        if not word:
            continue
        l, t, r, b = draw_temp.textbbox((0, 0), word, font=font, stroke_width=stroke_width)
        w = r - l
        h = b - t
        word_dims.append((word, w, h))

    for (word, w, h) in word_dims:
        if current_line and (current_line_width + w + letter_spacing) > max_width:
            lines.append(current_line)
            current_line = []
            current_line_width = 0
        if current_line:
            current_line_width += letter_spacing
        current_line.append((word, w, h))
        current_line_width += w

    if current_line:
        lines.append(current_line)

    if not lines:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), [], []

    line_heights = [max(h for (_, _, h) in line) for line in lines]
    total_height = sum(line_heights) + inter_line_spacing*(len(lines)-1)
    final_img_width = max_width + 2*padding
    final_img_height = total_height + 2*padding

    img = Image.new("RGBA", (final_img_width, final_img_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    seg_anns = []
    cur_y = padding

    for line_idx, line in enumerate(lines):
        lh = line_heights[line_idx]
        line_width = sum(w for (_, w, _) in line) + letter_spacing*(len(line)-1)
        cur_x = padding + (max_width - line_width)//2

        for (word, w, h) in line:
            draw.text((cur_x, cur_y), word, font=font,
                      fill=fill_color,
                      stroke_width=stroke_width,
                      stroke_fill=fill_color)
            l, t, r, b = draw.textbbox((cur_x, cur_y), word, font=font, stroke_width=stroke_width)
            seg_anns.append({
                "char": word,
                "polygon": [[l,t],[r,t],[r,b],[l,b]]
            })
            cur_x += w + letter_spacing

        cur_y += lh + inter_line_spacing

    overall_poly = [
        [padding,padding],
        [final_img_width-padding,padding],
        [final_img_width-padding, final_img_height-padding],
        [padding, final_img_height-padding]
    ]
    return _crop_and_fix_coords(img, overall_poly, seg_anns)

def draw_vertical_text_field_wrapped(text, font,
                                     max_height,
                                     letter_spacing=2,
                                     inter_column_spacing=10,
                                     word_spacing=10,
                                     padding=5,
                                     fill_color=(0, 0, 0),
                                     stroke_width=0):
    """
    세로로 '단어 단위' 칼럼 쌓기 (vertical wrap)
    """
    if not text.strip():
        return Image.new("RGBA",(1,1),(0,0,0,0)), [], []

    draw_temp = ImageDraw.Draw(Image.new("RGBA",(1,1)))
    words = re.split(r'\s+', text.strip())
    word_infos = []
    for word in words:
        if not word:
            continue
        l, t, r, b = draw_temp.textbbox((0,0), word, font=font, stroke_width=stroke_width)
        w = r-l
        h = b-t
        word_infos.append((word,w,h))

    if not word_infos:
        return Image.new("RGBA",(1,1),(0,0,0,0)), [], []

    columns = []
    current_col = []
    current_height = 0

    for (word,w,h) in word_infos:
        needed = h if not current_col else (word_spacing + h)
        if current_height+needed>max_height:
            columns.append(current_col)
            current_col=[]
            current_height=0
        if current_col:
            current_height+=word_spacing
        current_col.append((word,w,h))
        current_height+=h

    if current_col:
        columns.append(current_col)

    if not columns:
        return Image.new("RGBA",(1,1),(0,0,0,0)), [], []

    col_dims=[]
    for col in columns:
        cw = 0
        ch = 0
        for i,(word,w,h) in enumerate(col):
            ch += h
            cw = max(cw,w)
            if i<(len(col)-1):
                ch += word_spacing
        col_dims.append((cw,ch))

    total_img_height = max(d[1] for d in col_dims) if col_dims else 0
    total_img_width = sum(d[0] for d in col_dims)
    if len(col_dims)>1:
        total_img_width += inter_column_spacing*(len(col_dims)-1)

    final_width = total_img_width + 2*padding
    final_height = total_img_height + 2*padding

    img = Image.new("RGBA",(final_width, final_height),(0,0,0,0))
    draw = ImageDraw.Draw(img)
    seg_anns = []

    cur_x = padding
    for cidx,col in enumerate(columns):
        cw,ch = col_dims[cidx]
        y_offset = padding + (total_img_height - ch)//2
        cur_y = y_offset

        for i,(word,w,h) in enumerate(col):
            dx = cur_x + (cw-w)//2
            draw.text((dx,cur_y), word, font=font,
                      fill=fill_color,
                      stroke_width=stroke_width,
                      stroke_fill=fill_color)
            l,t,r,b = draw.textbbox((dx,cur_y), word, font=font, stroke_width=stroke_width)
            seg_anns.append({
                "char": word,
                "polygon": [[l,t],[r,t],[r,b],[l,b]]
            })
            cur_y+=h
            if i<(len(col)-1):
                cur_y+=word_spacing

        if cidx<(len(columns)-1):
            cur_x+= cw+inter_column_spacing

    overall_poly=[
        [padding,padding],
        [final_width-padding,padding],
        [final_width-padding,final_height-padding],
        [padding,final_height-padding]
    ]
    return _crop_and_fix_coords(img, overall_poly, seg_anns)


# -------------------------------------------------------
# 세로쓰기(글자 단위x, 전체를 한 박스) + 내부 padding/여유분 증가
# -------------------------------------------------------
def draw_vertical_text_as_one_box(text,
                                  font,
                                  max_height,
                                  line_spacing=5,
                                  # padding 크게 (예: 20)
                                  padding=20,
                                  fill_color=(0,0,0),
                                  stroke_width=0):
    """
    한 문장(혹은 단어)을 세로로 배치해도 최종 bounding box는 '하나'만 생성.
    글자가 잘리는 현상을 줄이기 위해 padding/extra_margin을 크게 잡음.
    """
    if not text.strip():
        return Image.new("RGBA",(1,1),(0,0,0,0)), [], []

    # 폰트 bbox와 실제 렌더링 오차 방지를 위해 여유
    extra_margin = 3

    draw_temp = ImageDraw.Draw(Image.new("RGBA",(1,1)))
    chars = list(text)  # 실제 글자 분해
    char_infos = []
    for ch in chars:
        l,t,r,b = draw_temp.textbbox((0,0), ch, font=font, stroke_width=stroke_width)
        # 각 글자에 extra_margin을 추가
        w = (r - l) + extra_margin
        h = (b - t) + extra_margin
        char_infos.append((ch, w, h))

    # 세로로 모든 글자를 배치했을 때의 총 높이, 최대 폭 계산
    total_char_height = sum(info[2] for info in char_infos)
    if len(char_infos) > 1:
        total_char_height += line_spacing * (len(char_infos) - 1)
    max_char_width = max(info[1] for info in char_infos) if char_infos else 0

    final_w = max_char_width + 2*padding
    final_h = total_char_height + 2*padding
    if final_h > max_height:
        final_h = max_height

    img = Image.new("RGBA",(final_w, final_h),(0,0,0,0))
    draw = ImageDraw.Draw(img)

    cur_y = padding
    for idx, (ch, w, h) in enumerate(char_infos):
        x_pos = padding + (max_char_width - w)//2
        # 실제 그릴 땐 extra_margin 이전 크기보다 조금 줄여서 위치 맞추기
        draw.text((x_pos, cur_y),
                  ch,
                  font=font,
                  fill=fill_color,
                  stroke_width=stroke_width,
                  stroke_fill=fill_color)

        # h에는 extra_margin이 더해져 있으므로 그대로 쌓아 올림
        cur_y += h

        if idx < len(char_infos)-1:
            cur_y += line_spacing

    overall_poly = [
        [0,0],
        [final_w,0],
        [final_w,final_h],
        [0,final_h]
    ]
    seg_anns = [{
        "char": text,
        "polygon": overall_poly
    }]

    non_empty = img.getbbox()        # (l,t,r,b)
    if non_empty:
        l,t,r,b = non_empty
        img = img.crop(non_empty)

        # 모든 좌표를 같은 만큼 당겨줌
        dx, dy = -l, -t
        overall_poly = [[x+dx, y+dy] for x,y in overall_poly]
        for s in seg_anns:
            s["polygon"] = [[x+dx, y+dy] for x,y in s["polygon"]]

    overall_poly = tight_bbox_from_segments(seg_anns)
    return img, overall_poly, seg_anns


def scale_image_and_polygons(img, segments, max_w, max_h, extra_polys=None):
    """
    img           : 경사된 세로 글자 + 빈 여백
    segments      : 글자 단위 세그
    extra_polys   : [block_poly]  4점 (회전 포함)
    ----------------------------------------------------------------------------
    스케일은 block_poly 의 높이/너비 기준으로만 계산
    """
    if not extra_polys or not extra_polys[0]:
        return img, segments, extra_polys

    block = extra_polys[0]              # 4×2
    xs = [x for x, _ in block]
    ys = [y for _, y in block]
    block_w = max(xs) - min(xs)
    block_h = max(ys) - min(ys)

    # ── (1) 필요한 축소비율 계산 ──────────────
    scale_factor = min(max_w / block_w, max_h / block_h, 1.0)
    if scale_factor == 1.0:
        return img, segments, extra_polys

    # ── (2) 이미지 실제 축소 ──────────────────
    new_w = max(1, int(img.width * scale_factor))
    new_h = max(1, int(img.height * scale_factor))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # ── (3) 세그·블록 모두 동일 비율로 축소 ───
    for seg in segments:
        seg["polygon"] = [[x*scale_factor, y*scale_factor] for x, y in seg["polygon"]]
    for poly in extra_polys:
        for pt in poly:
            pt[0] *= scale_factor
            pt[1] *= scale_factor

    return img, segments, extra_polys

def rotate_90_with_polygons(
        original_img: Image.Image,
        original_poly: list[list[float]],
        original_segments: list[dict]
    ):
    """
    -90° 회전 ↗ 여백 crop ↗ 폴리곤·세그 좌표 동기 이동
    반환: (rot_img, rot_poly, rot_segments)
    """
    # ── 빈·점 이미지면 그대로 ────────────────────────
    if original_img.width <= 1 or original_img.height <= 1:
        return original_img, original_poly, original_segments

    # ── OpenCV 로 -90° 회전 ─────────────────────────
    arr   = np.array(original_img)                     # RGBA
    w0, h0 = original_img.size
    center = (w0 / 2.0, h0 / 2.0)

    M = cv2.getRotationMatrix2D(center, -90, 1.0)      # 시계방향 90°
    cos_, sin_ = abs(M[0, 0]), abs(M[0, 1])
    new_w = max(int(h0 * sin_ + w0 * cos_), 1)
    new_h = max(int(h0 * cos_ + w0 * sin_), 1)

    # 평행이동 보정
    M[0, 2] += new_w / 2.0 - center[0]
    M[1, 2] += new_h / 2.0 - center[1]

    rotated_arr = cv2.warpAffine(
        arr, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0))      # 투명 채움

    rot_img = Image.fromarray(rotated_arr)

    # ── 폴리곤/세그도 동일 행렬 적용 ───────────────
    pts = np.array(original_poly, np.float32).reshape(-1, 1, 2)
    rot_poly = cv2.transform(pts, M).reshape(-1, 2).tolist()

    rot_segments = []
    for seg in original_segments:
        sp = np.array(seg["polygon"], np.float32).reshape(-1, 1, 2)
        rot_sp = cv2.transform(sp, M).reshape(-1, 2).tolist()
        rot_segments.append({"char": seg["char"], "polygon": rot_sp})

    # ── (★) 투명 여백 crop 후 동일 오프셋 적용 ────
    crop = rot_img.getbbox()            # 내용 있는 영역 bbox; None 이면 그대로
    if crop:
        l, t, r, b = crop
        rot_img = rot_img.crop(crop)
        dx, dy = -l, -t

        rot_poly= [[x + dx, y + dy] for x, y in rot_poly]
        for s in rot_segments:
            s["polygon"] = [[x + dx, y + dy] for x, y in s["polygon"]]

    # rot_poly = tight_bbox_from_segments(rot_poly)

    return rot_img, rot_poly, rot_segments


# ---------------------------------------------------------
# 핵심 추가: 글자수가 20자를 초과하면 앞 20자까지만 남김
# ---------------------------------------------------------
def limit_text_to_20(text):
    """
    만약 text가 20자를 초과하면, 앞 20자까지만 사용하고 나머지는 버린다.
    """
    text = text.strip()
    if len(text) > 8:
        return text[:8]
    return text

def create_book_spine(itemid, title, author, publisher, bg_folder):

    mode = random.choices(
        ["arabic", "roman"], weights=[0.7, 0.3], k=1
    )[0]  # 30 % 확률로 Roman 사용
    num_val = biased_random_num()  # 1 ~ 999 까지 편향 랜덤

    if mode == "roman":
        rand_num = int_to_roman(num_val)
    else:
        rand_num = str(num_val)

    OUT_PIC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    bg = load_random_bg(bg_folder)
    fixed_w, fixed_h = 200, 1200
    bg = bg.resize((fixed_w,fixed_h), resample=Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(20))
    bg_w, bg_h = bg.size

    # ------------------------------------------------
    # 20자 넘어가면 버리는 로직 (limit_text_to_20)
    # ------------------------------------------------
    if title:    title = limit_text_to_20(title)
    if author:   author = limit_text_to_20(author)
    if publisher:publisher = limit_text_to_20(publisher)

    if not title.strip() and not author.strip() and not publisher.strip():
        return

    base_min=35
    base_max=70
    scale_w = bg_w/200.0
    scale_h = bg_h/1200.0
    scale_factor = max(scale_w, scale_h)
    scale_factor = max(0.3, min(2.0, scale_factor))

    title_font_size = random.randint(int(base_min*scale_factor), int(base_max*scale_factor))
    num_font_size = int(title_font_size * 0.7)
    author_font_size= int(title_font_size*0.8)
    publisher_font_size= int(title_font_size*0.6)

    font_dir = '/opt/project/datasets/gen_OCR/font/'
    valid_exts = ('.ttf','.otf')
    font_files = [os.path.join(font_dir,f) for f in os.listdir(font_dir) if f.lower().endswith(valid_exts)]
    if not font_files:
        raise Exception("No valid font found.")

    title_font = ImageFont.truetype(random.choice(font_files), title_font_size)
    num_font = ImageFont.truetype(random.choice(font_files), num_font_size)
    author_font = ImageFont.truetype(random.choice(font_files), author_font_size)
    publisher_font = ImageFont.truetype(random.choice(font_files), publisher_font_size)

    t_color = pick_contrasting_color(bg)
    # n_color = pick_contrasting_color(bg)
    a_color = pick_contrasting_color(bg)
    p_color = pick_contrasting_color(bg)

    # 책등 레이아웃에서 타이틀/저자/출판사에 각각 할당할 세로 비율
    ratio_title = 0.4
    ratio_number = 0.1
    ratio_author = 0.25
    ratio_publisher = 0.25

    # 각 구간의 (높이) 계산
    title_region_h = int(bg_h * ratio_title)
    num_region_h = int(bg_h * ratio_number)
    author_region_h = int(bg_h * ratio_author)
    pub_region_h = int(bg_h * ratio_publisher)

    # 전체에서 문구가 너무 커지지 않도록 80% 정도로 제한
    max_dim_for_text = int(bg_h * 0.8)

    # ---------------------------
    # 1) 타이틀 이미지 생성
    # ---------------------------
    title_mode = random.choice(['direct','rotated','v_rotated'])
    if title_mode == 'direct':
        t_img, t_poly, t_segs = draw_vertical_text_as_one_box(
            text=title,
            font=title_font,
            max_height=max_dim_for_text,
            line_spacing=10,
            fill_color=t_color,
            stroke_width=1
        )
    elif title_mode=='rotated':
        horz_img, horz_poly, horz_segs = draw_horizontal_text_field_wrapped(
            text=title,
            font=title_font,
            max_width=max_dim_for_text,
            fill_color=t_color,
            stroke_width=1
        )
        t_img, t_poly, t_segs = rotate_90_with_polygons(horz_img, horz_poly, horz_segs)
    else:  # v_rotated
        vert_img, vert_poly, vert_segs = draw_vertical_text_field_wrapped(
            text=title,
            font=title_font,
            max_height=max_dim_for_text,
            fill_color=t_color,
            stroke_width=1
        )
        t_img, t_poly, t_segs = rotate_90_with_polygons(vert_img, vert_poly, vert_segs)

    t_img, t_segs, t_poly = jitter_image_and_polygons(t_img,
                                                      t_segs,
                                                      t_poly)

    shape_choice = random.choice([None, "circle", "star", "heart", "triangle"])
    shape_color, text_color = random_shape_and_text_color()

    if shape_choice is None:
        # (A) 도형 없이 그냥 숫자
        # ───────────────────────
        n_img, n_poly, n_segs = draw_vertical_text_as_one_box(
            text=rand_num,
            font=num_font,
            max_height=int(bg_h * 0.8),
            line_spacing=8,
            fill_color=text_color,  # ← 숫자 색
            stroke_width=1
        )
    else:
        # (B) 도형 + 숫자
        # ───────────────────────
        n_img, n_poly, n_segs = draw_shape_number(
            rand_num,
            font=num_font,
            shape=shape_choice,  # "circle" | "star" | …
            pad=int(num_font_size * 0.4),
            fill_shape=shape_color,  # 도형 색
            fill_num=text_color,  # 숫자 색
            stroke_width=1
        )

    n_img, n_segs, n_poly = jitter_image_and_polygons(n_img, n_segs, n_poly)

    # 2) 저자 이미지 생성
    author_mode = random.choice(['direct','rotated','v_rotated'])
    if author_mode=='direct':
        a_img, a_poly, a_segs = draw_vertical_text_as_one_box(
            text=author,
            font=author_font,
            max_height=max_dim_for_text,
            line_spacing=8,
            fill_color=a_color,
            stroke_width=1
        )
    elif author_mode=='rotated':
        horz_img2, horz_poly2, horz_segs2 = draw_horizontal_text_field_wrapped(
            text=author,
            font=author_font,
            max_width=int(bg_w*0.8),
            fill_color=a_color,
            stroke_width=1
        )
        a_img, a_poly, a_segs = rotate_90_with_polygons(horz_img2, horz_poly2, horz_segs2)
    else:
        vert_img2, vert_poly2, vert_segs2 = draw_vertical_text_field_wrapped(
            text=author,
            font=author_font,
            max_height=int(bg_h*0.8),
            fill_color=a_color,
            stroke_width=1
        )
        a_img, a_poly, a_segs = rotate_90_with_polygons(vert_img2, vert_poly2, vert_segs2)

    a_img, a_segs, a_poly = jitter_image_and_polygons(a_img,
                                                      a_segs,
                                                      a_poly)

    # 3) 출판사 이미지 생성
    publisher_mode = random.choice(['direct','rotated','v_rotated'])
    if publisher_mode=='direct':
        p_img, p_poly, p_segs = draw_vertical_text_as_one_box(
            text=publisher,
            font=publisher_font,
            max_height=max_dim_for_text,
            line_spacing=8,
            fill_color=p_color,
            stroke_width=1
        )
    elif publisher_mode=='rotated':
        horz_img3, horz_poly3, horz_segs3 = draw_horizontal_text_field_wrapped(
            text=publisher,
            font=publisher_font,
            max_width=int(bg_w*0.8),
            fill_color=p_color,
            stroke_width=1
        )
        p_img, p_poly, p_segs = rotate_90_with_polygons(horz_img3, horz_poly3, horz_segs3)
    else:
        vert_img3, vert_poly3, vert_segs3 = draw_vertical_text_field_wrapped(
            text=publisher,
            font=publisher_font,
            max_height=int(bg_h*0.8),
            fill_color=p_color,
            stroke_width=1
        )
        p_img, p_poly, p_segs = rotate_90_with_polygons(vert_img3, vert_poly3, vert_segs3)

    p_img, p_segs, p_poly = jitter_image_and_polygons(p_img,
                                                      p_segs,
                                                      p_poly)

    # ------------------------------------------------
    # 각 구간 별로, 이미지가 정해진 영역보다 크면 줄이기
    # scale_image_and_polygons: (width, height) 제한
    # ------------------------------------------------
    # 제목


    t_img, t_segs, [t_poly] = scale_image_and_polygons(
        t_img, t_segs, bg_w, title_region_h, extra_polys=[t_poly])

    n_img, n_segs, [n_poly] = scale_image_and_polygons(
        n_img, n_segs, bg_w, num_region_h, extra_polys=[n_poly])

    # 저자
    a_img, a_segs, [a_poly] = scale_image_and_polygons(
        a_img, a_segs, bg_w, author_region_h, extra_polys=[a_poly])

    # 출판사
    p_img, p_segs, [p_poly] = scale_image_and_polygons(
        p_img, p_segs, bg_w, pub_region_h, extra_polys=[p_poly])

    # ---------------------------
    # 배치 좌표(오프셋) 계산 및 보정
    # ---------------------------
    # 제목
    t_off_y = (title_region_h - t_img.height)//2
    t_off_y = max(t_off_y, 0)  # 음수 방지
    t_off_x = (bg_w - t_img.width)//2
    t_off_x = max(t_off_x, 0)

    n_start = title_region_h
    n_off_y = n_start + (num_region_h - n_img.height) // 2
    n_off_y = max(n_off_y, n_start)  # 음수/영역 초과 방지
    n_off_x = (bg_w - n_img.width) // 2
    n_off_x = max(n_off_x, 0)

    # 저자
    a_start = title_region_h + num_region_h
    a_off_y = a_start + (author_region_h - a_img.height)//2
    a_off_y = max(a_off_y, a_start)  # 음수/영역 초과 방지
    a_off_x = (bg_w - a_img.width)//2
    a_off_x = max(a_off_x, 0)

    # 출판사
    p_start = title_region_h + num_region_h + author_region_h
    p_off_y = p_start + (pub_region_h - p_img.height)//2
    p_off_y = max(p_off_y, p_start)
    p_off_x = (bg_w - p_img.width)//2
    p_off_x = max(p_off_x, 0)

    t_off = (t_off_x, t_off_y)
    n_off = (n_off_x, n_off_y)
    a_off = (a_off_x, a_off_y)
    p_off = (p_off_x, p_off_y)

    def offset_polygon(poly_list, ox, oy):
        return [[x+ox,y+oy] for (x,y) in poly_list]

    # 제목 offsets
    t_poly_off = offset_polygon(t_poly, *t_off)
    t_segs_off=[]
    for seg in t_segs:
        poly_off = offset_polygon(seg["polygon"], *t_off)
        t_segs_off.append({"char": seg["char"], "polygon": poly_off})

    n_poly_off = offset_polygon(n_poly, *n_off)
    n_segs_off = [{"char": s["char"],
                   "polygon": offset_polygon(s["polygon"], *n_off)}
                  for s in n_segs]

    # 저자 offsets
    a_poly_off = offset_polygon(a_poly, *a_off)
    a_segs_off=[]
    for seg in a_segs:
        poly_off = offset_polygon(seg["polygon"], *a_off)
        a_segs_off.append({"char": seg["char"], "polygon": poly_off})

    # 출판사 offsets
    p_poly_off = offset_polygon(p_poly, *p_off)
    p_segs_off=[]
    for seg in p_segs:
        poly_off = offset_polygon(seg["polygon"], *p_off)
        p_segs_off.append({"char": seg["char"], "polygon": poly_off})

    # 최종 합성
    bg.alpha_composite(t_img, dest=t_off)
    bg.alpha_composite(n_img, dest=n_off)
    bg.alpha_composite(a_img, dest=a_off)
    bg.alpha_composite(p_img, dest=p_off)

    t_segs_off = shift(t_segs, t_off)
    n_segs_off = shift(n_segs, n_off)
    a_segs_off = shift(a_segs, a_off)
    p_segs_off = shift(p_segs, p_off)

    # ② NEW: tight 박스 → 이미지 경계로 clamp

    t_poly_off, (dx_t, dy_t) = push_inside(offset_polygon(t_poly, *t_off), bg_w, bg_h)
    n_poly_off, (dx_n, dy_n) = push_inside(offset_polygon(n_poly, *n_off), bg_w, bg_h)
    a_poly_off, (dx_a, dy_a) = push_inside(offset_polygon(a_poly, *a_off), bg_w, bg_h)
    p_poly_off, (dx_p, dy_p) = push_inside(offset_polygon(p_poly, *p_off), bg_w, bg_h)

    def shift_segs(segs, dx, dy):
        for s in segs:
            s["polygon"] = [[x + dx, y + dy] for x, y in s["polygon"]]

    shift_segs(t_segs_off, dx_t, dy_t)
    shift_segs(n_segs_off, dx_n, dy_n)
    shift_segs(a_segs_off, dx_a, dy_a)
    shift_segs(p_segs_off, dx_p, dy_p)

    all_polys = [t_poly_off,n_poly_off, a_poly_off, p_poly_off]
    bg, all_polys, M_aff, M_persp = apply_affine_and_perspective(bg, all_polys)
    t_poly_off, n_poly_off, a_poly_off, p_poly_off = all_polys

    # ─── 새로 추가 ─────────────────────────────
    def is_valid_poly(poly):
        return poly and len(poly) >= 3
    if not (is_valid_poly(t_poly_off) or
            is_valid_poly(n_poly_off) or
            is_valid_poly(a_poly_off) or
            is_valid_poly(p_poly_off)):
        return

    for seg in t_segs_off:
        seg["polygon"] = _transform_polygon(seg["polygon"], M_aff, M_persp)

    for seg in n_segs_off:
        seg["polygon"] = _transform_polygon(seg["polygon"], M_aff, M_persp)

    # ── 저자 세그먼트 변환
    for seg in a_segs_off:
        seg["polygon"] = _transform_polygon(seg["polygon"], M_aff, M_persp)

    # ── 출판사 세그먼트 변환
    for seg in p_segs_off:
        seg["polygon"] = _transform_polygon(seg["polygon"], M_aff, M_persp)

    base_filename = str(itemid)
    out_path = OUT_PIC_DIR / f"{base_filename}.jpg"
    bg_final = bg.convert("RGB")
    bg_final.save(out_path)


    if SAVE_JSON:
        # COCO JSON
        annots = [
            {"label":"title","text":title,"polygon":t_poly_off,"segments":t_segs_off},
            {"label": "title", "text": rand_num, "polygon": n_poly_off, "segments": n_segs_off},
            {"label":"author","text":author,"polygon":a_poly_off,"segments":a_segs_off},
            {"label":"publisher","text":publisher,"polygon":p_poly_off,"segments":p_segs_off},
        ]
        data = {
            "image_id": str(itemid),
            "file_name": out_path.name,
            "width": bg_w,
            "height": bg_h,
            "annotations": annots
        }
        (OUT_LABEL_DIR / f"{base_filename}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')

    else:
        lines = []
        for poly in (t_poly_off,n_poly_off, a_poly_off, p_poly_off):
            if poly:
                lines.append(poly_to_yolo_seg_line(poly, bg_w, bg_h))

        (OUT_PIC_DIR / f"{base_filename}.txt").write_text(''.join(lines), 'utf-8')

def generate(path, bg_folder='../data/bg_images'):
    df = get_values(path)
    # 필요에 따라 수정 가능
    samples = df.sample(n=5000).reset_index(drop=True)

    args=[]
    for i in range(len(samples)):
        itemid = samples.loc[i,'itemid']
        title  = samples.loc[i,'titles']
        author = samples.loc[i,'author']
        publisher = samples.loc[i,'publisher']
        args.append((itemid, title, author, publisher, bg_folder))

    with Pool() as pool:
        pool.starmap(create_book_spine, args)

if __name__=="__main__":
    generate(
        '/opt/project/datasets/gen_OCR/Result_328.csv',
        bg_folder='/opt/project/datasets/public_book_aladin/standard_spine_images_240909/'
    )
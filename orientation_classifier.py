import cv2, glob, numpy as np
from pathlib import Path
from skimage.feature import hog
from sklearn.svm import SVC
from joblib import dump, load

# ───────────────────── ① 데이터 수집 ──────────────────────
# 폴더 구조 예:
#   data/0/*.jpg      # 0° (가로)
#   data/90/*.jpg     # 시계방향 90°
#   data/180/*.jpg
#   data/270/*.jpg
data_root = Path("data")
angles = [0, 90, 180, 270]          # 라벨 순서

def to_gray32(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (32, 32))

X, y = [], []
for lbl, ang in enumerate(angles):
    for fp in (data_root / str(ang)).glob("*.jpg"):
        X.append(to_gray32(fp))
        y.append(lbl)

# ───────────────────── ② HOG 특징 추출 ────────────────────
def hog32(img):
    return hog(img,
               orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(1, 1),
               visualize=False, feature_vector=True)

X = np.array([hog32(img) for img in X])
y = np.array(y)

# ───────────────────── ③ SVM 학습 ─────────────────────────
clf = SVC(kernel="rbf", C=10, gamma=0.04, probability=False)
clf.fit(X, y)
dump(clf, "orientation_svm.joblib")
print("✔ SVM 학습 완료 — 모델 저장: orientation_svm.joblib")

# ───────────────────── ④ 추론/보정 함수 ──────────────────
def orient_correct(img_bgr, clf_model):
    """
    img_bgr : (H,W,3) BGR 또는 grayscale 이미지
    return  : (rot_img, angle_deg)
    """
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    feat = hog32(cv2.resize(gray, (32, 32)))
    pred = int(clf_model.predict([feat])[0])   # 0,1,2,3
    angle = angles[pred]                       # 0/90/180/270
    rot_k = pred                               # np.rot90 k값과 일치
    rot_img = np.rot90(img_bgr, k=rot_k)       # 시계반대 기준
    return rot_img, angle

# ─────────────── ⑤ 데모 (단일 이미지 테스트) ───────────────
if __name__ == "__main__":
    svm = load("orientation_svm.joblib")
    test_path = "demo.jpg"
    src  = cv2.imread(test_path)
    corr, ang = orient_correct(src, svm)
    print(f"예측 각도: {ang}° → 보정 완료")
    cv2.imshow("corrected", corr); cv2.waitKey(0)
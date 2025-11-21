# clean_broken_by_cv2.py
import cv2, shutil, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ────────── ①  본인 데이터 루트만 고쳐 주세요 ──────────
ROOT = Path("/opt/project/datasets/OCR_outdoor/recog_outdoor/val").resolve()
# ─────────────────────────────────────────────────────

GT_PATH     = ROOT / "gt.txt"
BACKUP_PATH = ROOT / "gt_backup.txt"
N_WORKERS   = os.cpu_count() or 4      # 프로세스 수

def _is_broken(rel_path: str) -> bool:
    """cv2로 완전 디코딩 → 깨진 그림이면 True"""
    img_path = ROOT / rel_path
    img = cv2.imread(str(img_path))
    return img is None or img.size == 0

def main():
    # 0) gt.txt 백업
    if not BACKUP_PATH.exists():
        shutil.copy(GT_PATH, BACKUP_PATH)
        print(f"[✓] gt.txt 백업 → {BACKUP_PATH}")

    # 1) gt.txt 에 기록된 모든 이미지 경로 수집
    with GT_PATH.open(encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    rel_paths = [ln.split("\t", 1)[0] for ln in lines]
    print(f"[i] 검사 대상: {len(rel_paths):,} 개")

    # 2) 병렬로 깨진 파일 탐색
    bad_set = set()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        fut2p = {ex.submit(_is_broken, p): p for p in rel_paths}
        for fut in as_completed(fut2p):
            if fut.result():
                bad_set.add(fut2p[fut])

    if not bad_set:
        print("[✓] 깨진 이미지 없음! 종료")
        return
    print(f"[!] 깨진 이미지 발견: {len(bad_set):,} 개")

    # 3) gt.txt 갱신
    kept_lines = [ln for ln in lines if ln.split("\t", 1)[0] not in bad_set]
    with GT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(kept_lines))

    # 4) 실제 파일 삭제
    for rel in bad_set:
        (ROOT / rel).unlink(missing_ok=True)

    print(f"[✓] 삭제 완료, gt.txt 라인 수: {len(kept_lines):,}")

if __name__ == "__main__":
    main()
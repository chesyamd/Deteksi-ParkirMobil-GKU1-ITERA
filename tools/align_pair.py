"""Selaraskan (alignment) citra latar kosong ke frame citra terisi.

Pipeline proyek mengasumsikan kamera TETAP: latar kosong sudah sejajar persis
dengan citra uji. Pada data lapangan kita foto diambil handheld (tanpa tripod),
sehingga sudut kamera sedikit berubah antar-pengambilan. Tool ini memakai
SIFT + homografi (RANSAC) untuk memetakan citra KOSONG ke kerangka citra TERISI,
lalu menyimpan hasil warp supaya bisa dipakai langsung oleh pipeline yang sudah ada
(tanpa mengubah kode inti).

Contoh (citra asli diletakkan di data/source/, tidak di-commit):
    python tools/align_pair.py \
        --fill  data/source/IMG_2846_terisi.jpg \
        --empty data/source/IMG_2883_kosong.jpg \
        --out-fill data/raw/IMG_2846.jpg \
        --out-background data/background/IMG_2883_aligned.jpg \
        --diag data/outputs/align_2846_2883.jpg

Ukuran kerja mengikuti TARGET_SIZE di src/config.py (4:3, tidak memelintir citra).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TARGET_SIZE  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SIFT = cv2.SIFT_create(4000)
MATCHER = cv2.BFMatcher()


def _resize(img, size):
    if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
        return cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return img


def match_homography(gray_src, gray_dst, ratio=0.75, ransac=3.0):
    """Homografi dari citra sumber -> citra tujuan via SIFT + BFMatcher + RANSAC."""
    k_src, d_src = SIFT.detectAndCompute(gray_src, None)
    k_dst, d_dst = SIFT.detectAndCompute(gray_dst, None)
    if d_src is None or d_dst is None:
        raise SystemExit("SIFT gagal menemukan fitur (citra terlalu polos?)")
    matches = MATCHER.knnMatch(d_src, d_dst, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    if len(good) < 8:
        raise SystemExit(f"Pasangan fitur terlalu sedikit ({len(good)}). Cari foto sudut yang lebih mirip.")
    src = np.float32([k_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac)
    if H is None:
        raise SystemExit("findHomography gagal")
    mask = mask.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst.reshape(-1, 2), axis=1)[mask]
    stats = {
        "good": len(good),
        "inlier": int(mask.sum()),
        "mean_err_px": float(err.mean()) if err.size else float("nan"),
        "median_err_px": float(np.median(err)) if err.size else float("nan"),
    }
    return H, stats, (k_src, k_dst, good, mask)


def warp_to(src_bgr, H, size):
    """Warp citra sumber ke kerangka tujuan; area tak terliput diisi hitam."""
    return cv2.warpPerspective(src_bgr, H, tuple(size))


def valid_mask(H, size):
    ones = np.full((size[1], size[0]), 255, np.uint8)
    return cv2.warpPerspective(ones, H, tuple(size))


def _draw_diag(fill, warped, kp_pair, stats, out):
    k_src, k_dst, good, mask = kp_pair
    m = cv2.drawMatches(warped, k_src, fill, k_dst, good, None,
                        matchColor=(0, 200, 0), singlePointColor=(120, 120, 120),
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    txt = (f"aligned: inlier={stats['inlier']}/{stats['good']}  "
           f"err rata2={stats['mean_err_px']:.2f}px  median={stats['median_err_px']:.2f}px")
    cv2.rectangle(m, (0, 0), (m.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(m, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), m)


def main():
    ap = argparse.ArgumentParser(description="Selaraskan citra kosong ke frame citra terisi")
    ap.add_argument("--fill", required=True, help="citra terisi (menjadi kerangka acuan)")
    ap.add_argument("--empty", required=True, help="citra kosong ATAU folder berisi citra kosong")
    ap.add_argument("--out-fill", required=True, help="tujuan simpan citra terisi (ukuran kerja)")
    ap.add_argument("--out-background", required=True, help="tujuan simpan latar hasil warp")
    ap.add_argument("--size", default=f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
                    help="ukuran kerja WxH (default dari TARGET_SIZE)")
    ap.add_argument("--diag", default=None, help="berkas citra diagnostik pencocokan (opsional)")
    ap.add_argument("--min-inlier", type=int, default=30,
                    help="ambang minimal inlier agar dianggap selaras (default 30)")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    size = (w, h)

    fill = _resize(cv2.imread(args.fill), size)
    if fill is None:
        raise SystemExit(f"Citra terisi tidak ditemukan: {args.fill}")
    gray_fill = cv2.cvtColor(fill, cv2.COLOR_BGR2GRAY)

    empty_path = Path(args.empty)
    if empty_path.is_dir():
        empties = sorted(p for p in empty_path.iterdir() if p.suffix.lower() in IMAGE_EXT)
    else:
        empties = [empty_path]
    if not empties:
        raise SystemExit("Tidak ada citra kosong yang ditemukan")

    warped_list, stats_list = [], []
    for ep in empties:
        empty = _resize(cv2.imread(str(ep)), size)
        if empty is None:
            print(f"  lewati (tak terbaca): {ep}")
            continue
        gray_empty = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
        H, stats, kp = match_homography(gray_empty, gray_fill)
        stats["empty"] = ep.name
        warped_list.append(warp_to(empty, H, size))
        stats_list.append(stats)
        print(f"  {ep.name}: inlier={stats['inlier']} good={stats['good']} "
              f"err={stats['mean_err_px']:.2f}px")
        if args.diag and len(empties) == 1:
            _draw_diag(fill, warped_list[0], kp, stats, Path(args.diag))

    if not warped_list:
        raise SystemExit("Tidak ada citra kosong yang berhasil diproses")

    best = max(stats_list, key=lambda s: s["inlier"])
    if best["inlier"] < args.min_inlier:
        raise SystemExit(
            f"Kualitas selaras rendah (inlier terbaik={best['inlier']} < {args.min_inlier}). "
            "Ambil citra kosong dari sudut yang lebih mirip dengan citra terisi."
        )
    print(f"Penyelarasan terbaik: {best['empty']} (inlier={best['inlier']})")

    if len(warped_list) == 1:
        background = warped_list[0]
    else:
        background = np.median(np.stack(warped_list), axis=0).astype(np.uint8)
        print(f"Latar = median {len(warped_list)} citra kosong yang sudah diselaraskan")

    Path(args.out_fill).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_background).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out_fill, fill)
    cv2.imwrite(args.out_background, background)
    print("citra terisi ->", args.out_fill)
    print("latar selaras ->", args.out_background)


if __name__ == "__main__":
    main()

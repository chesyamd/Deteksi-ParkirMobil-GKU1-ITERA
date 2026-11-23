"""Jalankan pipeline pada SATU citra (untuk demo / cek cepat).

Contoh:
    python -m src.main --image data/raw/gku1_000.jpg \
        --background data/background --roi data/roi/roi.json \
        --out data/outputs/gku1_000_overlay.jpg
"""

import argparse
import json
from pathlib import Path

import cv2

from .config import VERSIONS
from .pipeline import detect
from .preprocess import load_image
from .roi import draw_overlay, load_slots
from .segment import compute_background, load_reference_frame

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser(description="Deteksi slot parkir pada satu citra")
    ap.add_argument("--image", required=True)
    ap.add_argument("--background", required=True, help="folder citra latar ATAU satu berkas")
    ap.add_argument("--roi", required=True)
    ap.add_argument("--version", default="usulan", choices=list(VERSIONS))
    ap.add_argument("--out", default=None, help="berkas overlay keluaran (opsional)")
    args = ap.parse_args()

    p = VERSIONS[args.version]
    slots = load_slots(args.roi)

    bg_path = Path(args.background)
    if bg_path.is_dir():
        bg_paths = sorted(x for x in bg_path.iterdir() if x.suffix.lower() in IMAGE_EXT)
    else:
        bg_paths = [bg_path]
    if not bg_paths:
        raise SystemExit("Tidak ada citra latar")

    reference = (compute_background(bg_paths, p) if p.method == "bgsub"
                 else load_reference_frame(bg_paths[0], p))

    bgr = load_image(args.image, p.resize)
    result = detect(bgr, reference, slots, p)

    print(json.dumps({
        "image": Path(args.image).name,
        "version": args.version,
        "threshold": round(result["threshold"], 2),
        "slot_tersedia": result["free"],
        "slot_total": result["total"],
        "slots": result["states"],
    }, indent=2, ensure_ascii=False))

    if args.out:
        overlay = draw_overlay(bgr, result["states"])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, overlay)
        print("overlay ->", args.out)


if __name__ == "__main__":
    main()

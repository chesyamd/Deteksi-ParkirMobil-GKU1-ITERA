"""Jalankan pipeline pada SATU citra uji dengan database adegan.

Contoh::

    python -m src.main --image data/raw/gku1_000.jpg \
        --database data/database --version usulan \
        --out data/outputs/gku1_000_overlay.jpg
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2

from .config import VERSIONS
from .database import SceneDatabase
from .pipeline import detect_photo
from .preprocess import load_image
from .roi import draw_overlay


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deteksi slot parkir pada satu citra")
    ap.add_argument("--image", required=True)
    ap.add_argument("--database", default="data/database",
                    help="folder database adegan (hasil src.database build)")
    ap.add_argument("--version", default="usulan", choices=list(VERSIONS))
    ap.add_argument("--out", default=None, help="berkas overlay keluaran (opsional)")
    ap.add_argument("--save-mask", default=None,
                    help="berkas masker biner keluaran (opsional)")
    ap.add_argument("--size", default="1280x960",
                    help="ukuran kerja WxH (default 1280x960)")
    args = ap.parse_args(argv)

    w, h = (int(v) for v in args.size.lower().split("x"))
    size = (w, h)
    p = replace(VERSIONS[args.version], resize=size)
    db = SceneDatabase(args.database, size=size)
    if not db.scenes:
        raise SystemExit(
            f"Database adegan kosong di {args.database}. Jalankan "
            "`python -m src.database build` setelah mengisi refs/."
        )

    bgr = load_image(args.image)
    result = detect_photo(bgr, db, p)

    payload = {
        "image": Path(args.image).name,
        "version": args.version,
        "recognized": result["recognized"],
        "scene": result.get("scene_id"),
        "scene_match": result.get("scene_match"),
        "threshold": (round(result["threshold"], 2)
                      if result.get("threshold") is not None else None),
        "slot_tersedia": result["free"],
        "slot_total": result["total"],
        "slots": result["states"],
    }
    if not result["recognized"]:
        payload["pesan"] = result.get("reason", "foto tidak dikenali")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.out and result["recognized"]:
        overlay = draw_overlay(bgr, result["states"])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, overlay)
        print("overlay ->", args.out)
    if args.save_mask and result["recognized"]:
        Path(args.save_mask).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.save_mask, result["mask"])
        print("mask ->", args.save_mask)

    return 0 if result["recognized"] else 2


if __name__ == "__main__":
    sys.exit(main())

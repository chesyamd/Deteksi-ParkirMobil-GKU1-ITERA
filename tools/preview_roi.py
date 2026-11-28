"""Pratinjau ROI slot di atas foto (tanpa menjalankan deteksi).

Contoh::

    python tools/preview_roi.py --image data/database/<adegan>/canonical.jpg \
        --roi data/database/<adegan>/roi.json --out data/outputs/roi_preview.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TARGET_SIZE  # noqa: E402
from src.roi import load_slots, slot_points  # noqa: E402

COLORS = [(0, 200, 0), (255, 160, 0), (0, 180, 255), (200, 0, 200),
          (255, 255, 0), (0, 255, 255)]


def draw_roi_preview(img, slots, alpha=0.22):
    """Gambar semua slot + label + luas. ``img`` BGR."""
    out = img.copy()
    overlay = img.copy()
    for i, slot in enumerate(slots):
        color = COLORS[i % len(COLORS)]
        pts = slot_points(slot).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
        x, y = slot_points(slot)[0]
        cv2.putText(out, f"{slot['name']}", (int(x) + 4, int(y) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    # garis digambar ulang agar tetap tegas setelah blending
    for i, slot in enumerate(slots):
        color = COLORS[i % len(COLORS)]
        pts = slot_points(slot).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
        x, y = slot_points(slot)[0]
        cv2.putText(out, slot["name"], (int(x) + 4, int(y) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"{len(slots)} slot ROI", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pratinjau ROI di atas foto")
    ap.add_argument("--image", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--out", default="data/outputs/roi_preview.jpg")
    ap.add_argument("--size", default=f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
                    help="ukuran kerja WxH (default 1280x960)")
    ap.add_argument("--alpha", type=float, default=0.22)
    args = ap.parse_args(argv)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Citra tidak ditemukan: {args.image}")
    w, h = (int(v) for v in args.size.lower().split("x"))
    if (img.shape[1], img.shape[0]) != (w, h):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    slots = load_slots(args.roi)
    out = draw_roi_preview(img, slots, args.alpha)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"{len(slots)} slot -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

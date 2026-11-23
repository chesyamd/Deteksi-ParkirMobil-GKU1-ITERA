"""Penanda ROI slot parkir (klik-dan-tarik) - dijalankan sekali per posisi kamera.

Contoh:
    python tools/roi_editor.py data/background/bg_00.jpg --out data/roi/roi.json

Cara pakai: tarik kotak pada setiap slot di jendela yang muncul,
tekan ENTER/SPACE untuk konfirmasi tiap kotak, dan ESC untuk selesai.
Nama slot otomatis slot_1, slot_2, ... (urut sesuai urutan penandaan).
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TARGET_SIZE  # noqa: E402
from src.roi import save_slots  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="citra area parkir (idealnya kondisi kosong)")
    ap.add_argument("--out", default="data/roi/roi.json")
    ap.add_argument("--prefix", default="slot")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Citra tidak ditemukan: {args.image}")
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    slots = []
    print("Tarik kotak di tiap slot. ENTER/SPACE = simpan kotak, ESC = selesai.")
    while True:
        x, y, w, h = cv2.selectROI(f"ROI editor ({len(slots)} slot tersimpan)",
                                   img, showCrosshair=False, fromCenter=False)
        if w == 0 or h == 0:
            break
        name = f"{args.prefix}_{len(slots) + 1}"
        slots.append({"name": name, "x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0), 2)
        cv2.putText(img, name, (int(x) + 4, int(y) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.destroyWindow(f"ROI editor ({len(slots) - 1} slot tersimpan)")

    if slots:
        save_slots(args.out, slots, size=TARGET_SIZE, note=Path(args.image).name)
        print(f"Tersimpan {len(slots)} slot -> {args.out}")
    else:
        print("Tidak ada slot yang disimpan.")


if __name__ == "__main__":
    main()

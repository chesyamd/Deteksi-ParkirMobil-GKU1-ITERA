"""Penanda ROI slot parkir - dijalankan sekali per posisi kamera.

Dua mode:
- **kotak** (`--shape rect`, bawaan): tarik kotak pada setiap slot.
- **poligon** (`--shape polygon`): klik titik-titik sudut slot (cocok untuk slot
  yang tampak trapesium pada foto miring). Klik kiri = tambah titik,
  klik kanan = tutup poligon, ESC/Q = selesai.

Contoh:
    python tools/roi_editor.py data/background/bg_00.jpg --out data/roi/roi.json
    python tools/roi_editor.py data/background/bg_00.jpg --shape polygon \
        --out data/roi/roi.json

Nama slot otomatis slot_1, slot_2, ... (urut sesuai urutan penandaan).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TARGET_SIZE  # noqa: E402
from src.roi import save_slots  # noqa: E402

GREEN = (0, 200, 0)
ORANGE = (0, 180, 255)


def _label(img, slot, color):
    pts = (slot["points"] if "points" in slot
           else [[slot["x"], slot["y"]], [slot["x"] + slot["w"], slot["y"] + slot["h"]]])
    lx = int(min(p[0] for p in pts)) + 4
    ly = int(min(p[1] for p in pts)) + 18
    cv2.putText(img, slot["name"], (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def select_rects(img, prefix):
    """Mode kotak: tarik kotak tiap slot. ENTER/SPACE simpan, ESC selesai."""
    slots = []
    print("Tarik kotak di tiap slot. ENTER/SPACE = simpan kotak, ESC = selesai.")
    while True:
        x, y, w, h = cv2.selectROI(f"ROI editor ({len(slots)} slot tersimpan)",
                                   img, showCrosshair=False, fromCenter=False)
        if w == 0 or h == 0:
            break
        slot = {"name": f"{prefix}_{len(slots) + 1}",
                "x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        slots.append(slot)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), GREEN, 2)
        _label(img, slot, GREEN)
        cv2.destroyWindow(f"ROI editor ({len(slots) - 1} slot tersimpan)")
    return slots


def select_polygons(img, prefix):
    """Mode poligon: klik kiri = titik, klik kanan = tutup, ESC/Q = selesai."""
    slots = []
    pending = []
    base = img.copy()
    win = "ROI poligon"

    def render():
        disp = base.copy()
        for s in slots:
            pts = np.array(s["points"], np.int32).reshape(-1, 1, 2)
            cv2.polylines(disp, [pts], True, GREEN, 2)
            _label(disp, s, GREEN)
        if pending:
            p = np.array(pending, np.int32).reshape(-1, 1, 2)
            cv2.polylines(disp, [p], False, ORANGE, 2)
            for pt in pending:
                cv2.circle(disp, pt, 3, ORANGE, -1)
        cv2.putText(disp, f"klik kiri: titik | klik kanan: tutup | ESC: selesai "
                          f"({len(slots)} slot, {len(pending)} titik)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, disp)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and len(pending) >= 3:
            slots.append({"name": f"{prefix}_{len(slots) + 1}",
                          "points": [[int(px), int(py)] for px, py in pending]})
            pending.clear()
        render()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    render()
    print("Klik kiri = titik sudut, klik kanan = tutup poligon, ESC/Q = selesai.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyWindow(win)
    return slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="citra area parkir (idealnya kondisi kosong)")
    ap.add_argument("--out", default="data/roi/roi.json")
    ap.add_argument("--prefix", default="slot")
    ap.add_argument("--shape", choices=["rect", "polygon"], default="rect",
                    help="bentuk ROI: kotak atau poligon/trapesium")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Citra tidak ditemukan: {args.image}")
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    if args.shape == "polygon":
        slots = select_polygons(img, args.prefix)
    else:
        slots = select_rects(img, args.prefix)

    if slots:
        note = f"{Path(args.image).name} ({args.shape})"
        save_slots(args.out, slots, size=TARGET_SIZE, note=note)
        print(f"Tersimpan {len(slots)} slot -> {args.out}")
    else:
        print("Tidak ada slot yang disimpan.")


if __name__ == "__main__":
    main()

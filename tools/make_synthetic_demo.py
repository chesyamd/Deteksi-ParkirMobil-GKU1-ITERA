"""Bangkitkan DATA SINTETIS untuk menguji pipeline tanpa data lapangan.

Ini BUKAN data proyek (data proyek wajib primer dari GKU 1 ITERA). Fungsinya
hanya smoke test: memastikan kode jalan end-to-end dan alur evaluasi benar.
Data sintetis diberi derau (gradien cahaya, bintik, bayangan awan, mobil yang
tidak sepenuhnya rapi) agar perbedaan baseline vs usulan vs ablasi terlihat.

Contoh:
    python tools/make_synthetic_demo.py
    python -m src.evaluate --images sandbox/raw --background sandbox/background \
        --roi sandbox/roi/roi.json --gt sandbox/ground_truth/ground_truth.csv
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TARGET_SIZE  # noqa: E402
from src.roi import save_slots  # noqa: E402

W, H = TARGET_SIZE
RNG = np.random.default_rng(42)

SLOT_W, SLOT_H = 180, 250
GAP_X, GAP_Y = 40, 50
COLS, ROWS = 5, 2
Y0 = 70
N_BACKGROUND = 12
N_FRAMES = 12
ASPHALT = 70

# Warna mobil (BGR) dengan tingkat kecerahan berbeda-beda.
# Satu warna sengaja dibuat gelap mendekati aspal -> kasus gagal realistis.
CAR_COLORS = [(35, 35, 40), (185, 185, 190), (120, 80, 60), (215, 215, 220)]


def slot_origins():
    grid_w = COLS * SLOT_W + (COLS - 1) * GAP_X
    x0 = (W - grid_w) // 2
    for r in range(ROWS):
        for c in range(COLS):
            yield x0 + c * (SLOT_W + GAP_X), Y0 + r * (SLOT_H + GAP_Y)


def base_image(brightness=0.0, gradient=0.0, speckle=0.0, rng=RNG):
    """Aspal + marka slot, dengan gradien cahaya dan derau bintik."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    field = (xx / W - 0.5) * gradient + (yy / H - 0.5) * gradient * 0.5
    gray = np.clip(ASPHALT + field + float(brightness), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for (x, y) in slot_origins():
        cv2.rectangle(img, (x, y), (x + SLOT_W, y + SLOT_H), (205, 205, 205), 3)

    if speckle > 0:
        n = int(speckle * H * W)
        idx = rng.integers(0, H * W, n)
        img.reshape(-1, 3)[idx] = rng.choice([0, 255], size=(n, 3)).astype(np.uint8)
    return img


def draw_car(img, slot_x, slot_y, color, rng):
    """Mobil di dalam slot + bayangan + kaca gelap, posisi sedikit bergeser."""
    m = 40
    dx = float(rng.normal(0, 0.08)) * SLOT_W
    dy = float(rng.normal(0, 0.06)) * SLOT_H
    x0 = int(slot_x + m + dx)
    x1 = int(slot_x + SLOT_W - m + dx)
    y0 = int(slot_y + SLOT_H * 0.14 + dy)
    y1 = int(slot_y + SLOT_H * 0.86 + dy)

    shade = tuple(int(v) for v in np.clip(np.array(color, np.float32) * 0.5, 0, 255))
    cv2.rectangle(img, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), shade, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)

    # Kaca/atap berwarna aspal -> jadi "lubang" di mask (uji morfologi closing).
    cy0 = y0 + int((y1 - y0) * 0.34)
    cy1 = y0 + int((y1 - y0) * 0.66)
    cv2.rectangle(img, (x0 + 4, cy0), (x1 - 4, cy1), (ASPHALT, ASPHALT, ASPHALT), -1)


def add_cloud_shadow(img, rng):
    """Bayangan awan lembut menutupi 1-2 slot -> sumber false positive realistis."""
    cx = int(rng.integers(int(0.2 * W), int(0.8 * W)))
    cy = int(rng.integers(int(0.2 * H), int(0.8 * H)))
    mask = np.zeros((H, W), np.float32)
    cv2.ellipse(mask, (cx, cy), (int(rng.integers(140, 260)), int(rng.integers(100, 180))),
                int(rng.integers(0, 180)), 0, 360, 1.0, -1)
    k = int(rng.choice([31, 51]))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    out = img.astype(np.float32) - mask[..., None] * float(rng.uniform(30, 50))
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    root = Path(__file__).resolve().parents[1]
    raw, bg_dir = root / "sandbox/raw", root / "sandbox/background"
    roi_dir, gt_dir = root / "sandbox/roi", root / "sandbox/ground_truth"
    for d in (raw, bg_dir, roi_dir, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    slots = []
    for i, (x, y) in enumerate(slot_origins(), start=1):
        slots.append({"name": f"slot_{i}", "x": x + 8, "y": y + 8,
                      "w": SLOT_W - 16, "h": SLOT_H - 16})
    save_slots(roi_dir / "roi.json", slots, size=TARGET_SIZE,
               note="ROI sintetis (smoke test)")

    for k in range(N_BACKGROUND):
        cv2.imwrite(str(bg_dir / f"bg_{k:02d}.jpg"),
                    base_image(brightness=RNG.normal(0, 3),
                               gradient=RNG.normal(0, 6), speckle=0.0008))

    rows = []
    for k in range(N_FRAMES):
        img = base_image(brightness=RNG.normal(0, 6),
                         gradient=RNG.normal(0, 22), speckle=0.003)
        occupied = RNG.random(len(slots)) < 0.5
        for s, occ in zip(slots, occupied):
            if occ:
                draw_car(img, s["x"] - 8, s["y"] - 8,
                         CAR_COLORS[int(RNG.integers(len(CAR_COLORS)))], RNG)
        if RNG.random() < 0.35:
            img = add_cloud_shadow(img, RNG)
        name = f"gku1_{k:03d}.jpg"
        cv2.imwrite(str(raw / name), img)
        rows.append([name] + ["terisi" if o else "kosong" for o in occupied])

    with open(gt_dir / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image"] + [s["name"] for s in slots])
        w.writerows(rows)

    print(f"OK: {N_BACKGROUND} citra latar, {N_FRAMES} citra uji di {raw}")


if __name__ == "__main__":
    main()

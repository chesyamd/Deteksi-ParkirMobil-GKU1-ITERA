"""Bangkitkan DATA SINTETIS untuk menguji pipeline tanpa data lapangan.

**INI BUKAN DATA PROYEK.** Data proyek wajib foto primer GKU 1 ITERA. Data di
``sandbox/`` hanya untuk smoke test: memastikan database adegan, registrasi
SIFT+homografi, segmentasi, evaluasi, dan penolakan foto berjalan end-to-end.

Yang dibangkitkan:
- 2 adegan sintetis dengan tekstur paving acak dan tata letak slot berbeda;
- foto kosong (refs) dengan pergeseran kamera acak;
- foto uji dengan mobil persegi + pergeseran kamera acak;
- satu foto "bukan parkiran" untuk menguji penolakan adegan;
- ``scene.json``, ``roi.json``, ``ground_truth.csv``.

Contoh::

    python tools/make_synthetic_demo.py
    python -m src.database build --database sandbox/database
    python -m src.evaluate --database sandbox/database --images sandbox/raw \
        --gt sandbox/ground_truth.csv --out sandbox/outputs --save-overlays
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TARGET_SIZE  # noqa: E402
from src.roi import save_slots  # noqa: E402

W, H = TARGET_SIZE
IMAGE_EXT = ".jpg"
N_REFS = 7
N_TESTS = 8

CAR_COLORS = [
    (235, 235, 235),  # putih
    (40, 40, 190),    # merah
    (190, 110, 40),   # biru
    (160, 160, 165),  # silver
    (35, 35, 40),     # gelap (kasus gagal realistis)
]

SCENES = [
    {"id": "sintetis_timur", "seed": 101, "cols": 3, "rows": 2,
     "slot_w": 180, "slot_h": 250, "gap_x": 45, "gap_y": 55, "y0": 70},
    {"id": "sintetis_barat", "seed": 202, "cols": 4, "rows": 1,
     "slot_w": 160, "slot_h": 230, "gap_x": 50, "gap_y": 45, "y0": 300},
]


# --------------------------------------------------------------------------- #
# Tekstur & objek
# --------------------------------------------------------------------------- #
def paving_texture(w, h, rng, base=78):
    """Paving block acak: ubin, nat, retakan, dan derau."""
    img = np.full((h, w, 3), base, np.uint8)
    tile = 64
    for y in range(0, h + tile, tile):
        for x in range(0, w + tile, tile):
            g = int(rng.integers(52, 112))
            dx, dy = int(rng.integers(-4, 5)), int(rng.integers(-4, 5))
            x0, y0 = x + dx, y + dy
            x1, y1 = x + tile - 2 + dx, y + tile - 2 + dy
            cv2.rectangle(img, (x0, y0), (x1, y1), (g, g, g), -1)
            cv2.line(img, (x0, y1), (x1, y1), (35, 35, 35), 1)
            cv2.line(img, (x1, y0), (x1, y1), (35, 35, 35), 1)

    for _ in range(160):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (p1[0] + int(rng.integers(-35, 36)),
              p1[1] + int(rng.integers(-35, 36)))
        g = int(rng.integers(28, 55))
        cv2.line(img, p1, p2, (g, g, g), 1, cv2.LINE_AA)

    noise = rng.normal(0, 5.0, (h, w, 1)).repeat(3, axis=2)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def make_slots(cfg):
    cols, rows = cfg["cols"], cfg["rows"]
    sw, sh = cfg["slot_w"], cfg["slot_h"]
    gx, gy = cfg["gap_x"], cfg["gap_y"]
    grid_w = cols * sw + (cols - 1) * gx
    x0 = (W - grid_w) // 2
    slots = []
    i = 1
    for r in range(rows):
        for c in range(cols):
            slots.append({
                "name": f"slot_{i}",
                "x": x0 + c * (sw + gx),
                "y": cfg["y0"] + r * (sh + gy),
                "w": sw,
                "h": sh,
            })
            i += 1
    return slots


def draw_slot_markings(img, slots):
    for s in slots:
        cv2.rectangle(img, (s["x"], s["y"]),
                      (s["x"] + s["w"], s["y"] + s["h"]), (225, 225, 225), 3)


def draw_car(img, slot, color, rng):
    """Mobil persegi + bayangan + kaca gelap, posisi sedikit bergeser."""
    margin_x = int(slot["w"] * 0.16)
    margin_y = int(slot["h"] * 0.13)
    dx = int(rng.normal(0, 0.04) * slot["w"])
    dy = int(rng.normal(0, 0.03) * slot["h"])
    x0 = slot["x"] + margin_x + dx
    x1 = slot["x"] + slot["w"] - margin_x + dx
    y0 = slot["y"] + margin_y + dy
    y1 = slot["y"] + slot["h"] - margin_y + dy

    shade = tuple(int(v * 0.45) for v in color)
    cv2.rectangle(img, (x0 - 7, y0 - 7), (x1 + 7, y1 + 7), shade, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)

    cy0 = y0 + int((y1 - y0) * 0.33)
    cy1 = y0 + int((y1 - y0) * 0.67)
    cv2.rectangle(img, (x0 + 6, cy0), (x1 - 6, cy1), (45, 45, 50), -1)
    cv2.line(img, (x0 + 6, (cy0 + cy1) // 2), (x1 - 6, (cy0 + cy1) // 2),
             (90, 90, 95), 2)


# --------------------------------------------------------------------------- #
# Kamera handheld sintetis
# --------------------------------------------------------------------------- #
def random_homography(w, h, rng, max_shift=45, max_rot=2.5,
                      max_scale=0.030, max_persp=12):
    """Homografi kamera acak (translasi, rotasi, skala, perspektif kecil)."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    center = (w / 2.0, h / 2.0)
    angle = float(rng.uniform(-max_rot, max_rot))
    scale = float(1.0 + rng.uniform(-max_scale, max_scale))
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += float(rng.uniform(-max_shift, max_shift))
    M[1, 2] += float(rng.uniform(-max_shift, max_shift))
    affine = cv2.perspectiveTransform(
        src.reshape(-1, 1, 2), np.vstack([M, [0, 0, 1]])
    ).reshape(-1, 2)
    jitter = rng.normal(0, max_persp, size=(4, 2))
    dst = (affine + jitter).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def warp_frame(base, Hmat, rng, lighting=True):
    frame = cv2.warpPerspective(base, Hmat, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
    if lighting:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        field = ((xx / W - 0.5) * float(rng.normal(0, 18))
                 + (yy / H - 0.5) * float(rng.normal(0, 12)))
        frame = np.clip(frame.astype(np.float32) + field[..., None]
                        + float(rng.normal(0, 5)), 0, 255).astype(np.uint8)
    return frame


def outside_image(rng):
    """Foto 'bukan parkiran': gradien + bentuk acak, tekstur berbeda."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.stack([
        120 + 60 * np.sin(xx / 90.0) + 30 * np.cos(yy / 70.0),
        90 + 50 * np.cos(xx / 50.0),
        70 + 40 * np.sin((xx + yy) / 60.0),
    ], axis=2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(25):
        c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        r = int(rng.integers(20, 120))
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.circle(img, c, r, color, int(rng.integers(2, 8)))
    return img


# --------------------------------------------------------------------------- #
# Generator utama
# --------------------------------------------------------------------------- #
def main(seed=42):
    root = Path(__file__).resolve().parents[1]
    sandbox = root / "sandbox"
    db_root = sandbox / "database"
    raw_dir = sandbox / "raw"
    gt_path = sandbox / "ground_truth.csv"
    for d in (db_root, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    all_slot_names, rows = [], []

    for cfg in SCENES:
        scene_id = cfg["id"]
        scene_dir = db_root / scene_id
        refs_dir = scene_dir / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        for old in refs_dir.glob("*.jpg"):
            old.unlink()

        scene_rng = np.random.default_rng(cfg["seed"])
        slots = make_slots(cfg)
        all_slot_names = sorted(set(all_slot_names) | {s["name"] for s in slots})

        base = paving_texture(W, H, scene_rng, base=78 + cfg["seed"] % 17)
        draw_slot_markings(base, slots)

        # refs: satu kanonik (identitas) + pergeseran kamera acak.
        # ref_00 sengaja diberi satu mobil transien agar median terbukti
        # lebih tahan daripada satu frame acuan (ablasi tanpa median).
        for k in range(N_REFS):
            if k == 0:
                frame = base.copy()
                draw_car(frame, slots[0], CAR_COLORS[1], scene_rng)
            else:
                Hk = random_homography(W, H, scene_rng)
                frame = warp_frame(base, Hk, scene_rng)
            cv2.imwrite(str(refs_dir / f"ref_{k:02d}{IMAGE_EXT}"), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        # metadata adegan
        (scene_dir / "scene.json").write_text(json.dumps({
            "id": scene_id,
            "name": f"{scene_id} (SINTETIS)",
            "note": ("Adegan sintetis untuk smoke test. BUKAN data laporan. "
                     "Tekstur paving acak + mobil persegi + pergeseran kamera. "
                     "ref_00 sengaja memuat satu mobil transien untuk menguji median."),
            "captured_at": "",
            "camera": "sintetis",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        save_slots(scene_dir / "roi.json", slots, size=TARGET_SIZE,
                   note=f"{scene_id} (SINTETIS, bukan data laporan)")

        # foto uji: mobil ditempel di bingkai kanonik lalu kamera digeser
        for k in range(N_TESTS):
            occupied = scene_rng.random(len(slots)) < 0.5
            frame_base = base.copy()
            for slot, occ in zip(slots, occupied):
                if occ:
                    color = CAR_COLORS[int(scene_rng.integers(len(CAR_COLORS)))]
                    draw_car(frame_base, slot, color, scene_rng)
            Hk = random_homography(W, H, scene_rng)
            frame = warp_frame(frame_base, Hk, scene_rng)
            name = f"{scene_id}_{k:03d}.jpg"
            cv2.imwrite(str(raw_dir / name), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            row = {"image": name, "scene": scene_id}
            for slot, occ in zip(slots, occupied):
                row[slot["name"]] = "terisi" if occ else "kosong"
            rows.append(row)

    # foto di luar database -> harus ditolak
    outside_name = "asing_bukan_parkiran.jpg"
    cv2.imwrite(str(raw_dir / outside_name), outside_image(rng),
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    rows.append({"image": outside_name, "scene": "luar_database"})

    fields = ["image", "scene"] + all_slot_names
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    (sandbox / "README.md").write_text(
        "# Data SINTETIS (bukan data laporan)\n\n"
        "Folder ini dibangkitkan oleh `tools/make_synthetic_demo.py`. Isinya\n"
        "hanya untuk smoke test pipeline dan evaluasi. **Jangan memasukkan\n"
        "angka dari folder ini ke laporan sebagai hasil penelitian.**\n\n"
        "Perintah:\n\n"
        "```bash\n"
        "python tools/make_synthetic_demo.py\n"
        "python -m src.database build --database sandbox/database\n"
        "python -m src.evaluate --database sandbox/database --images sandbox/raw \\\n"
        "    --gt sandbox/ground_truth.csv --out sandbox/outputs --save-overlays\n"
        "```\n",
        encoding="utf-8",
    )

    print(f"OK: {len(SCENES)} adegan sintetis di {db_root}")
    print(f"    {N_REFS} refs/adegan, {len(SCENES) * N_TESTS} foto uji + 1 foto luar")
    print(f"    GT: {gt_path}")
    print("    Ingat: SINTETIS, bukan data laporan.")


if __name__ == "__main__":
    main()

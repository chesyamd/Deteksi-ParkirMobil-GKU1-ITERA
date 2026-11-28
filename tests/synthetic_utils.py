"""Helper data sintetis kecil untuk unit test (bukan data laporan)."""

import json
from pathlib import Path

import cv2
import numpy as np

from src.roi import save_slots


def make_texture(w=400, h=300, seed=0):
    """Tekstur acak berfitur banyak agar SIFT punya keypoint."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 72, np.uint8)
    for _ in range(220):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        color = tuple(int(v) for v in rng.integers(25, 230, size=3))
        if rng.random() < 0.5:
            x2 = x + int(rng.integers(8, 60))
            y2 = y + int(rng.integers(8, 60))
            cv2.rectangle(img, (x, y), (x2, y2), color, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(4, 22)), color, -1)
    for _ in range(80):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, tuple(int(v) for v in rng.integers(10, 80, size=3)), 1)
    noise = rng.normal(0, 4, (h, w, 1)).repeat(3, axis=2)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def translation(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def affine_h(w, h, rng, max_shift=15, max_rot=1.0, max_scale=0.02):
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getRotationMatrix2D((w / 2, h / 2),
                                float(rng.uniform(-max_rot, max_rot)),
                                float(1 + rng.uniform(-max_scale, max_scale)))
    M[0, 2] += float(rng.uniform(-max_shift, max_shift))
    M[1, 2] += float(rng.uniform(-max_shift, max_shift))
    dst = cv2.perspectiveTransform(
        src.reshape(-1, 1, 2), np.vstack([M, [0, 0, 1]])
    ).reshape(-1, 2).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def warp(img, H, size):
    return cv2.warpPerspective(img, H, tuple(size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))


def draw_slots(img, slots):
    for s in slots:
        cv2.rectangle(img, (s["x"], s["y"]),
                      (s["x"] + s["w"], s["y"] + s["h"]), (225, 225, 225), 2)


def draw_cars(img, slots, occupied):
    for s, occ in zip(slots, occupied):
        if not occ:
            continue
        mx, my = int(s["w"] * 0.18), int(s["h"] * 0.18)
        cv2.rectangle(img, (s["x"] + mx, s["y"] + my),
                      (s["x"] + s["w"] - mx, s["y"] + s["h"] - my),
                      (220, 220, 225), -1)


def make_scene(root, scene_id="adegan_a", size=(400, 300), slots=None,
               n_refs=4, seed=0):
    """Buat folder adegan berisi refs/, roi.json, scene.json. Kembalikan base."""
    root = Path(root)
    scene = root / scene_id
    refs = scene / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    if slots is None:
        slots = [{"name": "slot_1", "x": 40, "y": 60, "w": 130, "h": 170}]
    rng = np.random.default_rng(seed)
    base = make_texture(size[0], size[1], seed)
    draw_slots(base, slots)
    for k in range(n_refs):
        H = np.eye(3) if k == 0 else affine_h(size[0], size[1], rng)
        cv2.imwrite(str(refs / f"ref_{k:02d}.jpg"), warp(base, H, size))
    save_slots(scene / "roi.json", slots, size=size, note="uji sintetis")
    (scene / "scene.json").write_text(json.dumps({
        "id": scene_id, "name": scene_id, "note": "unit test sintetis",
    }), encoding="utf-8")
    return base, slots


def make_query(base, slots, occupied, size=(400, 300), seed=1):
    rng = np.random.default_rng(seed)
    frame = base.copy()
    draw_cars(frame, slots, occupied)
    H = affine_h(size[0], size[1], rng)
    return warp(frame, H, size)

"""Utilitas Region of Interest (ROI) per slot parkir + penggambar overlay.

ROI slot dapat berupa:
- **kotak**: `{name, x, y, w, h}`
- **poligon/trapesium**: `{name, points: [[x, y], ...]}`

Poligon lebih akurat pada foto miring (oblik): slot tampak trapesium, sehingga
kotak akan ikut menghitung aspal/jalan di luar marka slot. Kedua bentuk
diperlakukan sama oleh pipeline.
"""

import json
from pathlib import Path

import cv2
import numpy as np

GREEN = (0, 200, 0)   # kosong
RED = (0, 0, 255)     # terisi


def load_slots(path):
    """Baca daftar slot dari JSON. Mengembalikan list {name, x, y, w, h} atau
    {name, points}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["slots"] if isinstance(data, dict) else data


def save_slots(path, slots, size=None, note=""):
    """Tulis daftar slot ke JSON (sekali kalibrasi per posisi kamera)."""
    payload = {
        "image_size": list(size) if size else None,
        "note": note,
        "slots": slots,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def is_polygon(slot):
    """True bila slot didefinisikan sebagai poligon (`points`)."""
    return bool(slot.get("points"))


def slot_points(slot):
    """Titik sudut slot sebagai array int32 Nx2 (poligon, atau 4 sudut kotak)."""
    if is_polygon(slot):
        return np.array(slot["points"], dtype=np.int32).reshape(-1, 2)
    x, y, w, h = int(slot["x"]), int(slot["y"]), int(slot["w"]), int(slot["h"])
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)


def slot_bbox(slot):
    """Kotak batas (bounding box) slot sebagai (x, y, w, h)."""
    pts = slot_points(slot)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def slot_rect(slot):
    """Kotak batas slot sebagai (x, y, w, h); nama lama yang tetap didukung."""
    return slot_bbox(slot)


def slot_area(slot):
    """Luas slot dalam piksel (kotak = w*h, poligon = luas kontur)."""
    if is_polygon(slot):
        return float(abs(cv2.contourArea(slot_points(slot).reshape(-1, 1, 2))))
    x, y, w, h = slot_bbox(slot)
    return float(w * h)


def slot_mask(slot, shape):
    """Masker uint8 (0/255) berbentuk slot, seukuran `shape` (tinggi, lebar)."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [slot_points(slot).reshape(-1, 1, 2)], 255)
    return mask


def draw_overlay(img, states):
    """Gambar kotak/poligon hijau (kosong) / merah (terisi) + hitungan slot."""
    out = img.copy()
    for s in states:
        color = RED if s["status"] == "terisi" else GREEN
        pts = s.get("points")
        if pts:
            cv2.polylines(out, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                          True, color, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (s["x"], s["y"]),
                          (s["x"] + s["w"], s["y"] + s["h"]), color, 2)
        cv2.putText(out, s["name"], (s["x"] + 4, s["y"] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    free = sum(1 for s in states if s["status"] == "kosong")
    cv2.putText(out, f"Slot tersedia: {free}/{len(states)}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out

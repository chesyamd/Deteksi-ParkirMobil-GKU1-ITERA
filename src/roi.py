"""Utilitas Region of Interest (ROI) per slot parkir + penggambar overlay."""

import json
from pathlib import Path

import cv2

GREEN = (0, 200, 0)   # kosong
RED = (0, 0, 255)     # terisi


def load_slots(path):
    """Baca daftar slot dari JSON. Mengembalikan list {name, x, y, w, h}."""
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


def slot_rect(slot):
    return int(slot["x"]), int(slot["y"]), int(slot["w"]), int(slot["h"])


def draw_overlay(img, states):
    """Gambar kotak hijau/merah + hitungan slot tersedia di atas citra asli."""
    out = img.copy()
    for s in states:
        color = RED if s["status"] == "terisi" else GREEN
        cv2.rectangle(out, (s["x"], s["y"]), (s["x"] + s["w"], s["y"] + s["h"]), color, 2)
        cv2.putText(out, s["name"], (s["x"] + 4, s["y"] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    free = sum(1 for s in states if s["status"] == "kosong")
    cv2.putText(out, f"Slot tersedia: {free}/{len(states)}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out

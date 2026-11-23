"""Tahap 5 - klasifikasi status slot dari rasio piksel kendaraan dalam ROI."""

import numpy as np

from .config import Params
from .roi import slot_rect


def occupancy_ratio(mask, slot):
    """Rasio piksel putih (kendaraan) di dalam kotak ROI slot."""
    x, y, w, h = slot_rect(slot)
    roi = mask[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / float(roi.size)


def classify_slot(mask, slot, p: Params):
    r = occupancy_ratio(mask, slot)
    x, y, w, h = slot_rect(slot)
    return {
        "name": slot["name"],
        "x": x, "y": y, "w": w, "h": h,
        "ratio": round(r, 4),
        "status": "terisi" if r >= p.ratio_threshold else "kosong",
    }

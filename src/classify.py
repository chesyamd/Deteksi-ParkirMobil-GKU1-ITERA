"""Tahap 5 - klasifikasi status slot dari rasio piksel kendaraan dalam ROI.

ROI boleh berupa kotak (`x,y,w,h`) atau poligon (`points`). Untuk poligon,
hanya piksel DI DALAM poligon yang dihitung, sehingga aspal di luar marka slot
tidak ikut menaikkan rasio (penting pada foto miring).
"""

import cv2
import numpy as np

from .config import Params
from .roi import is_polygon, slot_bbox, slot_mask, slot_points


def occupancy_ratio(mask, slot):
    """Rasio piksel putih (kendaraan) di dalam ROI slot (0..1)."""
    if is_polygon(slot):
        m = slot_mask(slot, mask.shape)
        area = int(cv2.countNonZero(m))
        if area == 0:
            return 0.0
        inside = int(cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=m)))
        return inside / float(area)

    x, y, w, h = slot_bbox(slot)
    roi = mask[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / float(roi.size)


def classify_slot(mask, slot, p: Params):
    r = occupancy_ratio(mask, slot)
    x, y, w, h = slot_bbox(slot)
    state = {
        "name": slot["name"],
        "x": x, "y": y, "w": w, "h": h,
        "ratio": round(r, 4),
        "status": "terisi" if r >= p.ratio_threshold else "kosong",
    }
    if is_polygon(slot):
        state["points"] = slot_points(slot).tolist()
    return state

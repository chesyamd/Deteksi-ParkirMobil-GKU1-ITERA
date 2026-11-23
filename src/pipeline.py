"""Pipeline lengkap: satu citra -> status tiap slot + citra antara."""

from .classify import classify_slot
from .config import Params
from .segment import binarize, clean_mask, difference


def detect(current_bgr, reference_gray, slots, p: Params):
    """Jalankan seluruh tahap. Mengembalikan dict berisi states + citra antara.

    current_bgr    : citra BGR (belum di-resize; preprocess yang mengurus)
    reference_gray : citra latar/frame acuan dalam grayscale (sudah di-resize)
    slots          : list {name, x, y, w, h}
    p              : Params (versi percobaan)
    """
    diff = difference(current_bgr, reference_gray, p)
    mask, thr = binarize(diff, p)
    if p.use_morph:
        mask = clean_mask(mask, p)
    states = [classify_slot(mask, s, p) for s in slots]
    free = sum(1 for s in states if s["status"] == "kosong")
    return {
        "states": states,
        "mask": mask,
        "diff": diff,
        "threshold": thr,
        "free": free,
        "total": len(states),
    }

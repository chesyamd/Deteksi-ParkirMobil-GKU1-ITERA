"""Pipeline lengkap: satu citra -> status tiap slot + citra antara.

Dua level API:
- ``detect``        : inti, membandingkan citra uji dengan latar yang SUDAH
                      berada di kerangka yang sama (dipakai unit test).
- ``detect_scene``  : menambahkan registrasi latar adegan ke foto uji.
- ``detect_photo``  : memilih adegan dari database lalu menjalankan pipeline.
"""

import cv2
import numpy as np

from .classify import classify_slot
from .config import Params
from .registration import (match_homography, to_gray, transform_slots,
                           warp_image, warp_valid_mask)
from .segment import binarize, clean_mask, difference


def detect(current_bgr, reference_gray, slots, p: Params, valid_mask=None):
    """Jalankan seluruh tahap. Mengembalikan dict berisi states + citra antara.

    current_bgr    : citra BGR (belum di-resize; preprocess yang mengurus)
    reference_gray : citra latar/frame acuan dalam grayscale (sudah di-resize)
    slots          : list {name, x, y, w, h} atau {name, points}
    p              : Params (versi percobaan)
    valid_mask     : opsional, area yang boleh dibandingkan (hasil registrasi)
    """
    diff = difference(current_bgr, reference_gray, p, valid_mask)
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
        "recognized": True,
    }


def _resize(img, size):
    if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
        return cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return img


def _scaled_slots(slots, sx, sy):
    if sx == 1.0 and sy == 1.0:
        return slots
    H = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    return transform_slots(slots, H)


def _unrecognized(reason, match=None):
    return {
        "recognized": False,
        "reason": reason,
        "scene_id": None,
        "scene_name": None,
        "scene_match": match.as_dict() if match is not None else None,
        "registration": None,
        "states": [],
        "mask": None,
        "diff": None,
        "threshold": None,
        "free": 0,
        "total": 0,
    }


def detect_scene(current_bgr, scene, p: Params, match=None):
    """Deteksi slot pada foto uji dengan latar dari satu adegan.

    Bila ``p.use_registration`` aktif, latar adegan diselaraskan ke foto uji
    dengan homografi. ROI slot ikut ditransformasi ke kerangka foto uji.
    Bila registrasi gagal, hasil ``recognized=False`` (bukan ditebak).
    """
    if p.resize is not None:
        target = tuple(p.resize)
    else:
        target = (scene.background_bgr.shape[1], scene.background_bgr.shape[0])

    current = _resize(current_bgr, target)
    bg = _resize(scene.background_bgr, target)
    can = _resize(scene.canonical_bgr, target)
    scene_valid = _resize(scene.valid_mask, target)

    sx = target[0] / float(scene.background_bgr.shape[1])
    sy = target[1] / float(scene.background_bgr.shape[0])
    slots = _scaled_slots(scene.slots, sx, sy)

    base_bgr = bg if p.use_median else can
    base_valid = scene_valid if p.use_median else np.full(scene_valid.shape, 255, np.uint8)

    if p.use_registration:
        if match is None or match.homography is None:
            match = match_homography(to_gray(can), to_gray(current), p)
        if not match.success:
            return _unrecognized(
                f"registrasi latar ke foto gagal ({match.reason}); "
                "perlu foto latar baru untuk sudut ini",
                match=None,
            )
        H = match.homography
        bg_aligned = warp_image(base_bgr, H, target)
        valid = warp_valid_mask(H, target)
        if p.use_median:
            valid = cv2.bitwise_and(valid, warp_image(base_valid, H, target))
        slots = transform_slots(slots, H)
        reference = to_gray(bg_aligned)
        result = detect(current, reference, slots, p, valid)
        result["registration"] = match.as_dict()
    else:
        result = detect(current, to_gray(base_bgr), slots, p, base_valid)
        result["registration"] = None

    result["scene_id"] = scene.id
    result["scene_name"] = scene.meta.get("name", scene.id)
    result["slots"] = slots
    return result


def detect_photo(image_bgr, db, p: Params, match=None):
    """Pilih adegan dari database lalu deteksi. Menolak foto tak dikenal."""
    if match is None:
        match = db.match(image_bgr, p)
    if not match.recognized:
        return _unrecognized(match.reason, match)
    result = detect_scene(image_bgr, match.scene, p, match=match.result)
    result["scene_match"] = match.as_dict()
    return result

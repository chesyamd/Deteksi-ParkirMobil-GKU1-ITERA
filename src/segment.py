"""Tahap 2-4 - background subtraction, thresholding Otsu, morfologi."""

import cv2
import numpy as np

from .config import Params
from .preprocess import enhance, load_image, preprocess, to_gray


def compute_background(paths, p: Params):
    """Citra latar = median N citra area kosong (mode "bgsub" / metode usulan).

    Median dipakai (bukan rata-rata) agar tahan terhadap satu-dua citra yang
    masih terlanjur mengandung kendaraan/gerakan.
    """
    paths = [str(x) for x in paths]
    if not paths:
        raise ValueError("Tidak ada citra latar yang diberikan")
    grays = [to_gray(load_image(x, p.resize)) for x in paths]
    if len(grays) == 1:
        return grays[0]
    return np.median(np.stack(grays), axis=0).astype(np.uint8)


def load_reference_frame(path, p: Params):
    """Satu frame acuan (mode "framediff" / baseline)."""
    return to_gray(load_image(path, p.resize))


def difference(current_bgr, reference_gray, p: Params, valid_mask=None):
    """|preprocess(citra kini) - enhance(citra referensi)| dalam grayscale.

    Citra referensi dipastikan seukuran citra kini agar selisih selalu valid,
    walau referensi diberikan pada resolusi yang berbeda. Bila ``valid_mask``
    diberikan (hasil warp registrasi), piksel di luar area valid dibuat nol
    agar border hitam tidak dianggap kendaraan.
    """
    cur = preprocess(current_bgr, p)
    ref = enhance(reference_gray, p)
    if ref.shape != cur.shape:
        ref = cv2.resize(ref, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(cur, ref)
    if valid_mask is not None:
        if valid_mask.shape != diff.shape:
            valid_mask = cv2.resize(valid_mask, (diff.shape[1], diff.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        diff = cv2.bitwise_and(diff, diff, mask=valid_mask)
    return diff


def binarize(diff, p: Params):
    """Ubah citra selisih menjadi biner. Otsu mencari ambang otomatis."""
    if p.use_otsu:
        thr, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thr = float(p.fixed_threshold)
        _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    return mask, float(thr)


def clean_mask(mask, p: Params):
    """Opening (buang bintik kecil) lalu closing (tutup lubang di bodi mobil)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_kernel, p.morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

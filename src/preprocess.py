"""Tahap 1 - pra-pemrosesan citra: resize, grayscale, Gaussian blur, CLAHE."""

import cv2
import numpy as np

from .config import Params


def load_image(path, size=None):
    """Baca citra dari disk (BGR). Opsional langsung di-resize."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Citra tidak dapat dibaca: {path}")
    if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return img


def to_gray(img):
    """BGR -> grayscale (tahan terhadap citra yang sudah grayscale)."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray, k=5):
    """Peredam noise: rata-rata berbobot piksel tetangga dalam jendela k x k."""
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def apply_clahe(gray, clip=2.0, tile=8):
    """CLAHE: perbaikan kontras LOKAL per tile (bukan global)."""
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return clahe.apply(gray)


def enhance(gray, p: Params):
    """Blur + CLAHE pada citra grayscale (tanpa resize)."""
    if p.use_blur:
        gray = gaussian_blur(gray, p.blur_kernel)
    if p.use_clahe:
        gray = apply_clahe(gray, p.clahe_clip, p.clahe_tile)
    return gray


def preprocess(img, p: Params):
    """Citra mentah (BGR) -> grayscale siap diselisihkan."""
    if p.resize is not None and (img.shape[1], img.shape[0]) != tuple(p.resize):
        img = cv2.resize(img, tuple(p.resize), interpolation=cv2.INTER_AREA)
    return enhance(to_gray(img), p)

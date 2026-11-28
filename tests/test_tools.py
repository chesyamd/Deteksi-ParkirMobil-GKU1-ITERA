"""Unit test alat bantu: crop/resize konversi dan pratinjau ROI."""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from convert_heic import center_crop_to_aspect, convert_one  # noqa: E402
from preview_roi import draw_roi_preview  # noqa: E402


def test_center_crop_ke_rasio_target():
    img = Image.fromarray(np.zeros((300, 600, 3), np.uint8))  # 2:1
    out = center_crop_to_aspect(img, (400, 300))  # 4:3
    w, h = out.size
    assert abs((w / h) - (400 / 300)) < 1e-6
    assert h == 300


def test_convert_one_png_ke_jpg_ukuran_kerja(tmp_path):
    src = tmp_path / "masukan.png"
    dst = tmp_path / "keluaran.jpg"
    cv2.imwrite(str(src), np.full((240, 320, 3), 128, np.uint8))
    via = convert_one(src, dst, size=(320, 240), quality=90, prefer="pillow")
    assert via == "pillow"
    out = cv2.imread(str(dst))
    assert out is not None
    assert (out.shape[1], out.shape[0]) == (320, 240)


def test_draw_roi_preview_tidak_mengubah_ukuran():
    img = np.full((240, 320, 3), 60, np.uint8)
    slots = [{"name": "slot_1", "x": 20, "y": 30, "w": 100, "h": 120}]
    out = draw_roi_preview(img, slots)
    assert out.shape == img.shape
    assert out.dtype == np.uint8

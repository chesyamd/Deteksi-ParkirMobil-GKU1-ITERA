"""Unit test registrasi SIFT + homografi."""

import cv2
import numpy as np

from src.config import Params
from src.registration import (corner_error, match_homography,
                              transform_slots, warp_valid_mask)
from synthetic_utils import make_texture, translation


def test_registrasi_memulihkan_pergeseran_diketahui():
    w, h = 400, 300
    src = make_texture(w, h, seed=1)
    H_true = translation(17, -11)
    dst = cv2.warpPerspective(src, H_true, (w, h),
                              borderMode=cv2.BORDER_REFLECT_101)

    p = Params(resize=None, min_good_matches=8, min_inliers=10)
    res = match_homography(src, dst, p)

    assert res.success, res.reason
    assert res.inliers >= 20
    assert res.homography is not None
    assert corner_error(res.homography, H_true, (w, h)) < 2.0


def test_registrasi_memulihkan_rotasi_dan_skala():
    w, h = 400, 300
    src = make_texture(w, h, seed=2)
    H_true = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.04)
    H_true = np.vstack([H_true, [0, 0, 1]])
    dst = cv2.warpPerspective(src, H_true, (w, h),
                              borderMode=cv2.BORDER_REFLECT_101)

    p = Params(resize=None, min_good_matches=8, min_inliers=10)
    res = match_homography(src, dst, p)

    assert res.success, res.reason
    assert corner_error(res.homography, H_true, (w, h)) < 3.0


def test_transform_slots_menggeser_poligon():
    slot = {"name": "s", "points": [[10, 10], [50, 10], [50, 40], [10, 40]]}
    out = transform_slots([slot], translation(5, 7))
    assert out[0]["name"] == "s"
    assert out[0]["points"] == [[15, 17], [55, 17], [55, 47], [15, 47]]


def test_warp_valid_mask_mengisi_area_terliput():
    mask = warp_valid_mask(np.eye(3), (100, 80))
    assert mask.shape == (80, 100)
    assert int(mask.min()) == 255

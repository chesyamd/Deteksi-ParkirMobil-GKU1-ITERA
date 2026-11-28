"""Registrasi citra handheld: SIFT + pencocokan fitur + homografi RANSAC.

Foto lapangan diambil dengan ponsel tanpa tripod, sehingga posisi kamera
berubah antar-pengambilan. Modul ini menyelaraskan dua citra dari sudut
pandang berbeda dengan homografi:

    H : koordinat citra SUMBER  ->  koordinat citra TUJUAN

Untuk pipeline, sumber = bingkai kanonik adegan (foto kosong acuan), tujuan =
foto uji. Latar adegan di-warp dengan H agar sejajar dengan foto uji, lalu
ROI slot (yang didefinisikan pada bingkai kanonik) ikut ditransformasi.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Params
from .roi import slot_points


@dataclass
class RegistrationResult:
    """Hasil pencocokan dua citra."""

    homography: np.ndarray | None
    good_matches: int
    inliers: int
    inlier_ratio: float
    mean_err_px: float
    success: bool
    reason: str = ""

    def as_dict(self):
        return {
            "success": self.success,
            "good_matches": self.good_matches,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "mean_err_px": (round(self.mean_err_px, 3)
                            if np.isfinite(self.mean_err_px) else None),
            "reason": self.reason,
        }


def to_gray(img):
    """BGR/gray -> grayscale uint8."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_features(gray, nfeatures=4000):
    """Deteksi keypoint + deskriptor SIFT."""
    sift = cv2.SIFT_create(nfeatures=int(nfeatures))
    keypoints, descriptors = sift.detectAndCompute(to_gray(gray), None)
    return keypoints, descriptors


def _ratio_good_matches(kp_src, des_src, kp_dst, des_dst, ratio):
    if des_src is None or des_dst is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(des_src, des_dst, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def match_homography_from_features(kp_src, des_src, kp_dst, des_dst, p: Params):
    """Hitung homografi sumber -> tujuan dari fitur yang sudah dideteksi."""
    good = _ratio_good_matches(kp_src, des_src, kp_dst, des_dst, p.sift_ratio)
    if len(good) < p.min_good_matches:
        return RegistrationResult(
            None, len(good), 0, 0.0, float("nan"), False,
            f"pasangan fitur terlalu sedikit ({len(good)} < {p.min_good_matches})",
        )

    src = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, float(p.ransac_thresh))
    if H is None or mask is None:
        return RegistrationResult(
            None, len(good), 0, 0.0, float("nan"), False,
            "findHomography gagal (fitur tidak konsisten)",
        )
    if not np.all(np.isfinite(H)):
        return RegistrationResult(
            None, len(good), 0, 0.0, float("nan"), False,
            "homografi mengandung nilai tak hingga",
        )

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / float(len(good)) if good else 0.0

    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst.reshape(-1, 2), axis=1)[inlier_mask]
    mean_err = float(errors.mean()) if errors.size else float("nan")

    reason = ""
    success = True
    if inliers < p.min_inliers:
        success = False
        reason = f"inlier terlalu sedikit ({inliers} < {p.min_inliers})"
    elif inlier_ratio < p.min_inlier_ratio:
        success = False
        reason = (f"rasio inlier terlalu rendah "
                  f"({inlier_ratio:.2f} < {p.min_inlier_ratio:.2f})")

    return RegistrationResult(
        H, len(good), inliers, inlier_ratio, mean_err, success, reason
    )


def match_homography(src, dst, p: Params):
    """Deteksi fitur lalu hitung homografi sumber -> tujuan."""
    kp_src, des_src = detect_features(src, p.sift_features)
    kp_dst, des_dst = detect_features(dst, p.sift_features)
    return match_homography_from_features(kp_src, des_src, kp_dst, des_dst, p)


def warp_image(src_bgr, H, size):
    """Warp citra sumber ke kerangka tujuan; area tak terliput diisi hitam."""
    w, h = int(size[0]), int(size[1])
    return cv2.warpPerspective(src_bgr, H, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_valid_mask(H, size):
    """Masker area yang benar-benar terliput warp (bukan border hitam)."""
    w, h = int(size[0]), int(size[1])
    ones = np.full((h, w), 255, np.uint8)
    return cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def transform_points(points, H):
    """Transformasi array titik Nx2 dengan homografi H."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def transform_slots(slots, H):
    """Transformasi semua ROI slot (kotak/poligon) ke kerangka tujuan."""
    out = []
    for slot in slots:
        pts = transform_points(slot_points(slot), H)
        new_slot = {"name": slot["name"], "points": pts.astype(int).tolist()}
        out.append(new_slot)
    return out


def corner_error(H_est, H_true, size):
    """Galat rata-rata (piksel) empat sudut citra setelah transformasi.

    Dipakai unit test untuk memastikan homografi yang dipulihkan mendekati
    homografi yang diketahui.
    """
    w, h = int(size[0]), int(size[1])
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    est = cv2.perspectiveTransform(corners, H_est).reshape(-1, 2)
    true = cv2.perspectiveTransform(corners, H_true).reshape(-1, 2)
    return float(np.linalg.norm(est - true, axis=1).mean())

"""Konfigurasi & preset parameter pipeline deteksi slot parkir.

Semua parameter yang dituning (dan dilaporkan di Bab 3/4) terpusat di sini
supaya eksperimen baseline vs usulan vs ablasi memakai definisi yang sama persis.

Desain data:
- Kamera ponsel TIDAK tetap (handheld). Karena itu tidak ada asumsi "satu
  kamera + satu roi.json". Setiap adegan (satu sudut pandang) punya database
  sendiri di ``data/database/<id_adegan>/`` berisi foto kosong di ``refs/``,
  ``scene.json``, dan ``roi.json`` pada bingkai kanonik.
- Foto uji dicocokkan ke adegan (SIFT + RANSAC), lalu latar diselaraskan ke
  foto uji dengan homografi sebelum diselisihkan.
"""

from dataclasses import dataclass

# Ukuran kerja seragam untuk semua citra (lebar, tinggi).
# 1280x960 = rasio 4:3, sama dengan rasio kamera/lensa ponsel (tidak memelintir).
TARGET_SIZE = (1280, 960)

# Ambang rasio yang disapu pada analisis sensitivitas (level slot).
SENSITIVITY_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)


@dataclass(frozen=True)
class Params:
    """Parameter pipeline. Satu objek = satu "versi" percobaan.

    Semua field punya nilai bawaan = metode usulan penuh. Preset di bawah
    mengubah field tertentu untuk baseline/ablasi.
    """

    # --- Tahap 0: geometri & registrasi (kamera handheld) ---
    resize: tuple | None = TARGET_SIZE
    use_registration: bool = True       # selaraskan latar ke foto uji (SIFT+homografi)
    use_median: bool = True             # latar = median banyak refs (bukan satu frame)
    sift_features: int = 4000           # batas jumlah keypoint SIFT
    sift_ratio: float = 0.75            # Lowe ratio test
    ransac_thresh: float = 3.0          # ambang reproyeksi RANSAC (piksel)
    min_good_matches: int = 10          # minimal pasangan fitur lolos ratio test
    min_inliers: int = 30               # minimal inlier agar adegan dianggap cocok
    min_inlier_ratio: float = 0.15      # minimal inlier/good matches

    # --- Tahap 1: pra-pemrosesan ---
    use_blur: bool = True
    blur_kernel: int = 5                # Gaussian blur 5x5
    use_clahe: bool = True
    clahe_clip: float = 2.0             # clipLimit CLAHE
    clahe_tile: int = 8                 # tileGridSize 8x8

    # --- Tahap 3: binerisasi ---
    use_otsu: bool = True
    fixed_threshold: int = 40           # dipakai hanya bila use_otsu = False

    # --- Tahap 4: morfologi ---
    use_morph: bool = True
    morph_kernel: int = 5               # kernel elips 5x5

    # --- Tahap 5: klasifikasi per ROI ---
    ratio_threshold: float = 0.30       # rasio piksel kendaraan >= ambang -> "terisi"


# --------------------------------------------------------------------------- #
# Preset versi percobaan
# --------------------------------------------------------------------------- #
# Usulan: registrasi + latar median + blur + CLAHE + Otsu + morfologi.
PROPOSED = Params()

# Baseline klasik: satu frame latar, tanpa registrasi, tanpa pra-pemrosesan,
# tanpa morfologi. Tetap memakai Otsu agar perbandingan adil di tahap binerisasi.
BASELINE = Params(
    use_registration=False,
    use_median=False,
    use_blur=False,
    use_clahe=False,
    use_morph=False,
)

# Ablasi: hilangkan satu tahap dari metode usulan.
ABL_NO_REGISTRATION = Params(use_registration=False)
ABL_NO_MEDIAN = Params(use_median=False)
ABL_NO_CLAHE = Params(use_clahe=False)
ABL_NO_BLUR = Params(use_blur=False)
ABL_NO_MORPH = Params(use_morph=False)

VERSIONS = {
    "baseline": BASELINE,
    "usulan": PROPOSED,
    "ablasi_tanpa_registrasi": ABL_NO_REGISTRATION,
    "ablasi_tanpa_median": ABL_NO_MEDIAN,
    "ablasi_tanpa_clahe": ABL_NO_CLAHE,
    "ablasi_tanpa_blur": ABL_NO_BLUR,
    "ablasi_tanpa_morfologi": ABL_NO_MORPH,
}

# Urutan tampilan tabel evaluasi (baseline dan usulan di depan).
VERSION_ORDER = list(VERSIONS)

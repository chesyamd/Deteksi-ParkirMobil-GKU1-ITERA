"""Konfigurasi & preset parameter pipeline deteksi slot parkir.

Semua parameter yang nanti dituning (dan dilaporkan di Bab 3/4) terpusat di sini
supaya eksperimen baseline vs usulan vs ablasi memakai definisi yang sama persis.
"""

from dataclasses import dataclass

# Ukuran kerja seragam untuk semua citra (lebar, tinggi).
# Koordinat ROI slot didefinisikan pada ukuran ini, jadi kamera harus tetap
# pada posisi yang sama selama satu sesi pengambilan.
TARGET_SIZE = (1280, 720)


@dataclass
class Params:
    """Parameter pipeline. Satu objek = satu "versi" percobaan."""

    # --- Tahap 1: pra-pemrosesan ---
    resize: tuple | None = TARGET_SIZE
    use_blur: bool = True
    blur_kernel: int = 5          # Gaussian blur 5x5
    use_clahe: bool = True
    clahe_clip: float = 2.0       # clipLimit CLAHE
    clahe_tile: int = 8           # tileGridSize 8x8

    # --- Tahap 2: background subtraction ---
    #   "bgsub"     = metode usulan (latar = median banyak citra kosong)
    #   "framediff" = baseline (latar = SATU frame acuan, tanpa pra-pemrosesan)
    method: str = "bgsub"

    # --- Tahap 3: binerisasi ---
    use_otsu: bool = True
    fixed_threshold: int = 40     # dipakai hanya bila use_otsu = False

    # --- Tahap 4: morfologi ---
    use_morph: bool = True
    morph_kernel: int = 5         # kernel elips 5x5

    # --- Tahap 5: klasifikasi per ROI ---
    ratio_threshold: float = 0.30  # rasio piksel kendaraan >= ambang -> "terisi"


# Preset versi percobaan
PROPOSED = Params()
BASELINE = Params(method="framediff", use_blur=False, use_clahe=False, use_morph=False)
ABL_NO_CLAHE = Params(use_clahe=False)
ABL_NO_MORPH = Params(use_morph=False)
ABL_NO_BLUR = Params(use_blur=False)

VERSIONS = {
    "baseline": BASELINE,
    "usulan": PROPOSED,
    "ablasi_tanpa_clahe": ABL_NO_CLAHE,
    "ablasi_tanpa_morfologi": ABL_NO_MORPH,
    "ablasi_tanpa_blur": ABL_NO_BLUR,
}

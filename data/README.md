# Struktur folder data

```
data/
    source/          # foto mentah HEIC/4032x3024 (TIDAK di-commit)
    raw/             # foto uji JPG 1280x960 (TIDAK di-commit)
    database/        # database per adegan/sudut pandang (lihat README di dalam)
    ground_truth/    # label manual per slot (template di-commit)
    outputs/         # hasil evaluasi & overlay (TIDAK di-commit)
    roi/             # ROI pilot lama (dipertahankan untuk kompatibilitas)
```

## Aturan

- **Jangan commit foto mentah besar** (HEIC/4032x3024). Simpan di `data/source/`
  atau bagikan lewat Drive. `.gitignore` sudah mengecualikan folder ini.
- Foto kerja 1280x960 (4:3) hasil konversi masuk `data/raw/` atau
  `data/database/<adegan>/refs/`; keduanya juga tidak di-commit.
- Yang **di-commit** hanya metadata kecil: `scene.json`, `roi.json`,
  `ground_truth.csv`, dan template.
- Data sintetis ada di `sandbox/` (di luar `data/`) dan **bukan data laporan**.

## Urutan kerja setelah foto tersedia

1. `python tools/convert_heic.py --input data/source --out data/source/jpg`
2. Salin foto kosong ke `data/database/<adegan>/refs/`, isi `scene.json`.
3. `python -m src.database build`
4. Tandai ROI dengan `tools/roi_editor.py`, cek dengan `tools/preview_roi.py`.
5. Salin foto uji ke `data/raw/`, isi `data/ground_truth/ground_truth.csv`.
6. `python -m src.evaluate --database data/database --images data/raw ...`

# Data multi-scene (paket I-07)

Folder ini untuk **beberapa titik kamera pada satu area** (mis. parkiran mobil
GKU 1). Setiap scene punya latar dan ROI sendiri, lalu metrik digabung (pooled)
oleh `python -m src.evaluate --scenes data/scenes`.

## Tata letak

```
data/scenes/
└── <nama_scene>/            # mis. pagi_timur, siang_barat
    ├── raw/                 # citra uji scene ini      (TIDAK di-commit)
    ├── background/          # citra area kosong scene   (TIDAK di-commit)
    ├── roi.json             # ROI slot scene ini        (di-commit)
    └── ground_truth.csv     # label per slot            (di-commit)
```

Aturan: **satu scene = satu posisi kamera**. Koordinat `roi.json` hanya sah
untuk citra dari posisi itu. Jangan mencampur citra dari posisi berbeda dalam
satu scene — buat subfolder baru.

## Contoh

```bash
mkdir -p data/scenes/pagi/raw data/scenes/pagi/background
cp <citra-uji>/*.jpg          data/scenes/pagi/raw/
cp <citra-kosong>/*.jpg       data/scenes/pagi/background/
cp data/roi/roi.json          data/scenes/pagi/roi.json          # lalu sesuaikan
cp data/ground_truth/ground_truth.csv data/scenes/pagi/ground_truth.csv

python -m src.evaluate --scenes data/scenes --out data/outputs --save-overlays
```

Hasil: `results.csv` (gabungan), `results_per_scene.csv`, `results.md`
(gabungan + rincian per scene), `per_image.json`, dan `overlays/<scene>/`.

> Isi `raw/` dan `background/` **tidak** di-commit (ukuran + privasi); dibagikan
> lewat Drive. `roi.json` dan `ground_truth.csv` kecil → ikut ke repo.

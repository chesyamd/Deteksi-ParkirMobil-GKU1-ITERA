# Data multi-scene (legacy)

> **Catatan:** desain utama sekarang memakai `data/database/<id_adegan>/`
> dengan pencocokan adegan SIFT+homografi dan registrasi latar per foto
> (lihat `data/database/README.md`). Folder `data/scenes/` dipertahankan hanya
> sebagai arsip/kompatibilitas fungsi lama. CLI `src.evaluate --scenes` sudah
> tidak dipakai; gunakan `src.evaluate --database`.

Tata letak lama:

```
data/scenes/<nama_scene>/
    raw/                 # citra uji (TIDAK di-commit)
    background/          # citra kosong (TIDAK di-commit)
    roi.json             # ROI slot (di-commit)
    ground_truth.csv     # label per slot (di-commit)
```

Untuk data baru, gunakan:

```
data/database/<id_adegan>/
    refs/                # foto kosong (TIDAK di-commit)
    scene.json           # metadata (di-commit)
    roi.json             # ROI pada bingkai kanonik (di-commit)
```

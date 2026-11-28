# Database adegan parkir GKU 1 ITERA

Kamera ponsel **tidak tetap** (handheld), jadi setiap **sudut pandang** adalah
satu adegan (`scene`) dengan database sendiri. Jangan memakai satu `roi.json`
global untuk semua foto.

## Struktur satu adegan

```
data/database/<id_adegan>/
    refs/                 # foto parkiran KOSONG (JPG 1280x960)
    scene.json            # metadata adegan (di-commit)
    roi.json              # polygon/kotak slot pada bingkai kanonik (di-commit)
    canonical.jpg         # dibuat build: ref pertama = bingkai kanonik
    background.jpg        # dibuat build: median refs terselaraskan
    valid_mask.png        # dibuat build: area valid hasil warp
    build.json            # dibuat build: ringkasan proses
```

Foto di `refs/`, `canonical.jpg`, `background.jpg`, `valid_mask.png`, dan
`build.json` **tidak di-commit** (lihat `.gitignore`). `scene.json` dan
`roi.json` kecil dan wajib ikut repo.

## Langkah mengisi data (setelah foto diambil)

1. **Konversi HEIC** ke JPG ukuran kerja (1280x960, 4:3):
   ```bash
   python tools/convert_heic.py --input data/source --out data/source/jpg
   ```
   Pasang `pillow-heif` (disarankan) atau ImageMagick sebagai fallback.
2. **Buat folder adegan** dan masukkan foto kosong:
   ```
   data/database/gku1_pagi_timur/refs/*.jpg
   ```
   Ambil 7-20 foto area kosong dari sudut yang sama, sedikit berbeda posisi.
3. **Buat `scene.json`** (contoh):
   ```json
   {
     "id": "gku1_pagi_timur",
     "name": "GKU 1 - pagi, sisi timur",
     "note": "Handheld, 08.00 WIB, cuaca cerah.",
     "captured_at": "2026-11-30",
     "camera": "Ponsel X, mode foto, tanpa tripod"
   }
   ```
4. **Tandai ROI** pada `canonical.jpg` (atau ref pertama) dengan editor:
   ```bash
   python -m src.database build --database data/database   # buat canonical dulu
   python tools/roi_editor.py data/database/gku1_pagi_timur/canonical.jpg \
       --shape polygon --out data/database/gku1_pagi_timur/roi.json
   python tools/preview_roi.py \
       --image data/database/gku1_pagi_timur/canonical.jpg \
       --roi data/database/gku1_pagi_timur/roi.json \
       --out data/outputs/roi_preview.jpg
   ```
   ROI boleh kotak `{x,y,w,h}` atau poligon `{points:[[x,y],...]}`. Untuk foto
   miring, poligon lebih akurat.
5. **Build ulang** latar median:
   ```bash
   python -m src.database build --database data/database
   ```
6. **Labeli ground truth** foto uji di
   `data/ground_truth/ground_truth.csv` (lihat README di folder itu).

## Perintah

```bash
python -m src.database build      # bangun semua adegan
python -m src.database list       # daftar adegan siap pakai
python -m src.database match --image data/raw/foto.jpg
```

Foto yang tidak cocok ke adegan mana pun akan ditolak dengan pesan
"perlu foto latar baru", bukan ditebak. Tambahkan adegan baru dari sudut itu.

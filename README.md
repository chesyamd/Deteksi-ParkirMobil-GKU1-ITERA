# Deteksi Ketersediaan Slot Parkir Mobil GKU 1 ITERA

Tugas Besar Pengolahan Citra Digital - Kelompok **KASKOS**, Kelas RB
(IF25-40410), Program Studi Teknik Informatika, Institut Teknologi Sumatera.

Pipeline: **pencocokan adegan (SIFT+RANSAC) -> registrasi homografi ->
pra-pemrosesan (blur + CLAHE) -> background subtraction (median refs) ->
thresholding Otsu -> morfologi -> klasifikasi per ROI slot** -> status tiap
slot (kosong/terisi), jumlah slot kosong, masker, dan overlay.

Foto lapangan diambil **handheld** dengan ponsel (HEIC, 4032x3024, 4:3) dari
banyak sudut. Karena kamera tidak tetap, sistem **tidak** memakai satu
`roi.json` global. Setiap sudut pandang adalah satu **adegan** dengan database
sendiri.

---

## Struktur repo

```
src/
    config.py        # SEMUA parameter + preset baseline/usulan/ablasi
    preprocess.py    # resize, grayscale, Gaussian blur, CLAHE
    registration.py  # SIFT + homografi RANSAC, warp, transformasi ROI
    database.py      # database adegan, build latar median, match foto
    segment.py       # background subtraction, Otsu, morfologi
    classify.py      # rasio piksel kendaraan -> kosong/terisi
    roi.py           # ROI kotak/poligon + overlay
    pipeline.py      # rangkaian tahap + deteksi per adegan
    evaluate.py      # baseline vs usulan vs ablasi, sensitivitas, kasus gagal
    main.py          # deteksi satu foto (CLI)
app/                 # demo Streamlit
tools/               # konversi HEIC, editor/pratinjau ROI, generator sintetis
tests/               # unit test (pytest)
docs/                # kerangka laporan, protokol, bacaan, naskah video
data/                # struktur data (foto tidak di-commit), lihat data/README.md
sandbox/             # data SINTETIS untuk smoke test (tidak di-commit)
.github/             # CI + templat issue/PR
requirements.txt
requirements-app.txt
requirements-tools.txt
```

## Instalasi (reproduksi)

Python 3.11+ (diuji 3.12). Disarankan memakai venv.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Opsional:
- `pip install -r requirements-app.txt` untuk aplikasi Streamlit.
- `pip install -r requirements-tools.txt` untuk konversi HEIC cepat
  (`pillow-heif`); tanpa itu, `tools/convert_heic.py` memakai ImageMagick.

## Menjalankan

### 1) Smoke test data sintetis (BUKAN data laporan)

```bash
python tools/make_synthetic_demo.py
python -m src.database build --database sandbox/database
python -m src.evaluate --database sandbox/database --images sandbox/raw \
    --gt sandbox/ground_truth.csv --out sandbox/outputs --save-overlays
```

> Angka dari `sandbox/` hanya untuk memastikan kode berjalan. **Jangan
> dimasukkan ke laporan sebagai hasil penelitian.**

### 2) Data asli (setelah foto tersedia)

```bash
# HEIC -> JPG 1280x960
python tools/convert_heic.py --input data/source --out data/source/jpg

# isi data/database/<id_adegan>/refs/ + scene.json, lalu:
python -m src.database build
python -m src.database list

# tandai ROI pada bingkai kanonik
python tools/roi_editor.py data/database/<id_adegan>/canonical.jpg \
    --shape polygon --out data/database/<id_adegan>/roi.json
python tools/preview_roi.py --image data/database/<id_adegan>/canonical.jpg \
    --roi data/database/<id_adegan>/roi.json --out data/outputs/roi_preview.jpg

# build ulang setelah ROI siap
python -m src.database build

# evaluasi
python -m src.evaluate --database data/database --images data/raw \
    --gt data/ground_truth/ground_truth.csv --out data/outputs --save-overlays
```

### 3) Deteksi satu foto

```bash
python -m src.main --image data/raw/gku1_000.jpg --database data/database \
    --version usulan --out data/outputs/gku1_000_overlay.jpg
```

### 4) Aplikasi demo

```bash
pip install -r requirements.txt -r requirements-app.txt
streamlit run app/streamlit_app.py
```

Aplikasi menampilkan adegan terpilih, overlay hijau (kosong)/merah (terisi),
hitungan slot, dan masker. Foto yang tidak cocok ke adegan mana pun ditolak
dengan pesan "perlu foto latar baru".

## Konsep pengolahan citra yang dipakai

| Tahap | Konsep | Berkas |
|---|---|---|
| 0. Pencocokan adegan | SIFT + rasio Lowe + RANSAC, inlier terbanyak | `registration.py`, `database.py` |
| 0. Registrasi | homografi, warpPerspective, valid mask | `registration.py` |
| 1. Pra-pemrosesan | resize 4:3, grayscale, Gaussian blur, CLAHE | `preprocess.py` |
| 2. Segmentasi | background subtraction, median banyak refs | `segment.py`, `database.py` |
| 3. Binerisasi | thresholding Otsu | `segment.py` |
| 4. Pembersihan | morfologi opening + closing | `segment.py` |
| 5. Klasifikasi | rasio piksel kendaraan dalam ROI kotak/poligon | `classify.py`, `roi.py` |

## Versi percobaan (`src/config.py`)

| Nama | Isi |
|---|---|
| `baseline` | satu frame latar, **tanpa** registrasi/pra-proses/morfologi |
| `usulan` | registrasi + latar median + blur + CLAHE + Otsu + morfologi |
| `ablasi_tanpa_registrasi` | usulan tanpa penyelarasan latar |
| `ablasi_tanpa_median` | latar = satu frame kanonik (bukan median) |
| `ablasi_tanpa_clahe` | usulan tanpa CLAHE |
| `ablasi_tanpa_blur` | usulan tanpa Gaussian blur |
| `ablasi_tanpa_morfologi` | usulan tanpa opening/closing |

Semua parameter terpusat di `Params` (`src/config.py`). Menambah versi baru
cukup menambah preset di `VERSIONS`.

## Keluaran evaluasi

`python -m src.evaluate` menghasilkan di folder `--out`:

- `results.csv` - satu baris per versi: akurasi, precision/recall/F1 (makro),
  TP/FP/FN/TN, jumlah slot, jumlah citra, jumlah ditolak.
- `results.md` - tabel baseline vs usulan vs ablasi, confusion matrix,
  sensitivitas ambang, rekognisi adegan, daftar kasus gagal.
- `sensitivity.csv` - akurasi/precision/recall/F1 vs ambang rasio (usulan).
- `failures.csv` - slot salah klasifikasi dan foto ditolak.
- `per_image.json` - prediksi per citra tiap versi.
- `overlays/` - overlay metode usulan (opsional `--save-overlays`).

Metrik dihitung pada **level slot**. Foto yang tidak cocok ke adegan mana pun
ditolak dan dicatat, tidak ditebak.

## Data & privasi

- Foto mentah besar (HEIC/4032x3024) **tidak di-commit**; lihat `.gitignore`.
- Foto kerja 1280x960 juga tidak di-commit; bagikan lewat Drive.
- Yang di-commit: `scene.json`, `roi.json`, `ground_truth.csv`, template,
  kode, tes, dan dokumen.
- Data sintetis ada di `sandbox/` dan tidak di-commit.

## Dokumen

Kerangka dokumen ada di `docs/`: outline laporan 5 bab, template tabel hasil,
protokol pengambilan foto, daftar bacaan, catatan progres mingguan, dan naskah
video demo. Isi yang bertanda `TODO` wajib dilengkapi tim setelah data asli
masuk.

## Tim

| Peran | Nama (NIM) |
|---|---|
| Project Lead + Evaluation & QA Analyst | Chesya Margaretha Deto (124140098) |
| Data Engineer & Annotator | Jelita Ayu Resito Siregar (124140056) |
| Computer Vision Engineer | Kezia Adelina Tamba (124140046) |
| Application Engineer | Ribka Deuina Manik (124140014) |

Dosen pengampu: Mohamad Idris, S.Si., M.Sc.

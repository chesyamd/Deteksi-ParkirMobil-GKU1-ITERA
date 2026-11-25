# Deteksi Ketersediaan Slot Parkir Mobil GKU 1 ITERA

Tugas Besar Pengolahan Citra Digital — Kelompok **KASKOS**, Kelas RB (IF25-40410),
Program Studi Teknik Informatika, Institut Teknologi Sumatera.

Pipeline: **pra-pemrosesan → background subtraction → thresholding Otsu →
morfologi → klasifikasi per ROI slot** → status setiap slot (kosong/terisi)
beserta citra overlay.

---

## Ringkasan

Sistem menerima dua masukan: **citra latar** (area parkir kosong) dan **citra uji**
(area parkir terisi). Selisih keduanya digunakan untuk menemukan piksel kendaraan,
lalu setiap slot parkir dinilai dari **rasio piksel kendaraan** di dalam ROI-nya.
Keluaran berupa status tiap slot dan jumlah slot yang tersedia.

## Struktur repo

```
.
├── src/                 # pipeline inti
│   ├── config.py        # parameter + preset versi percobaan
│   ├── preprocess.py    # tahap 1: resize, grayscale, Gaussian blur, CLAHE
│   ├── segment.py       # tahap 2-4: background subtraction, Otsu, morfologi
│   ├── roi.py           # definisi ROI slot + penggambar overlay
│   ├── classify.py      # tahap 5: rasio piksel kendaraan -> kosong/terisi
│   ├── pipeline.py      # rangkaian seluruh tahap
│   ├── evaluate.py      # baseline vs usulan vs ablasi + metrik
│   └── main.py          # pipeline pada satu citra (CLI)
├── app/                 # aplikasi demo (Streamlit)
├── tools/               # penanda ROI, penyelaras citra handheld, generator sintetis
├── tests/               # unit test
├── data/                # struktur data (citra tidak disertakan)
├── .github/             # CI + templat issue/pull request
├── requirements.txt
└── requirements-app.txt
```

`sandbox/` (hasil data sintetis) dan `data/` (citra asli) **tidak di-commit**.

## Instalasi

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v          # 13 test
```

## Menjalankan

```bash
# 1) Smoke test pada data sintetis (memastikan kode berjalan end-to-end)
python tools/make_synthetic_demo.py
python -m src.evaluate --images sandbox/raw --background sandbox/background \
    --roi sandbox/roi/roi.json --gt sandbox/ground_truth/ground_truth.csv \
    --out sandbox/outputs --save-overlays

# 2) Deteksi pada satu citra
python -m src.main --image data/raw/gku1_pagi_001.jpg --background data/background \
    --roi data/roi/roi.json --out data/outputs/gku1_pagi_001_overlay.jpg

# 3) Aplikasi demo
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```

> **Penting:** isi `sandbox/` adalah data **sintetis** dan hanya untuk memastikan
> program berjalan. Angkanya tidak dilaporkan sebagai hasil proyek. Hasil resmi
> hanya dari citra asli.

## Alur kerja pada data asli

1. **Simpan citra asli** (terisi & kosong) ke `data/source/` — **tidak** di-commit.
2. **Selaraskan** citra kosong ke kerangka citra terisi. Foto di lapangan diambil
   *handheld* (tanpa tripod), sehingga sudut kamera bergeser antar-pengambilan dan
   background subtraction mentah akan penuh galat. Penyelarasan memakai SIFT +
   homografi (RANSAC):
   ```bash
   python tools/align_pair.py \
       --fill  data/source/IMG_2846_terisi.jpg \
       --empty data/source/IMG_2883_kosong.jpg \
       --out-fill data/raw/IMG_2846.jpg \
       --out-background data/background/IMG_2883_aligned.jpg \
       --diag data/outputs/align_2846_2883.jpg
   ```
   `--empty` juga boleh berupa **folder** berisi beberapa citra kosong; semuanya
   disejajarkan lalu diambil nilai **median** sebagai latar.
3. **Tandai ROI**:
   ```bash
   python tools/roi_editor.py data/background/bg_00.jpg --out data/roi/roi.json
   ```
4. **Isi ground truth**: `data/ground_truth/ground_truth.csv`
   (kolom `image` + satu kolom per slot, nama kolom sama dengan `roi.json`).
5. **Evaluasi**:
   ```bash
   python -m src.evaluate --images data/raw --background data/background \
       --roi data/roi/roi.json --gt data/ground_truth/ground_truth.csv \
       --out data/outputs --save-overlays
   ```
   Hasil: `results.csv`, `results.md`, `per_image.json`, dan `overlays/`.

## Pemetaan tahap pipeline → berkas

| Tahap | Konsep pengolahan citra | Fungsi |
|---|---|---|
| 1. Pra-pemrosesan | resize, grayscale, Gaussian blur 5×5, CLAHE (clip 2.0, tile 8×8) | `preprocess.py` |
| 2. Segmentasi | background subtraction (median N citra kosong / frame acuan) | `segment.compute_background`, `segment.difference` |
| 3. Binerisasi | thresholding Otsu | `segment.binarize` |
| 4. Pembersihan | morfologi opening + closing (kernel elips 5×5) | `segment.clean_mask` |
| 5. Klasifikasi slot | rasio piksel kendaraan dalam ROI vs ambang | `classify.py` |

## Versi percobaan (`src/config.py`)

| Nama | Isi |
|---|---|
| `baseline` | frame acuan tunggal + Otsu, **tanpa** blur/CLAHE/morfologi |
| `usulan` | pipeline penuh (blur + CLAHE + background median + Otsu + morfologi) |
| `ablasi_tanpa_clahe` | usulan tanpa CLAHE |
| `ablasi_tanpa_morfologi` | usulan tanpa morfologi |
| `ablasi_tanpa_blur` | usulan tanpa Gaussian blur |

Baseline memakai **satu citra acuan** (analogi *frame differencing* pada citra
statis), sedangkan metode usulan memakai **median 10–20 citra kosong** sebagai latar.

## Catatan tentang data

Citra asli tidak disertakan dalam repo karena ukuran dan privasi (wajah/plat
nomor). Dataset tersedia atas permintaan. Hasil evaluasi resmi dihitung dari
citra asli tersebut, bukan dari data sintetis.

## Tim

| Peran | Nama (NIM) |
|---|---|
| Project Lead + Evaluation & QA Analyst | Chesya Margaretha Deto (124140098) |
| Data Engineer & Annotator | Jelita Ayu Resito Siregar (124140056) |
| Computer Vision Engineer | Kezia Adelina Tamba (124140046) |
| Application Engineer | Ribka Deuina Manik (124140014) |

Dosen pengampu: Mohamad Idris, S.Si., M.Sc.
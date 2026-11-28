# TODO yang Butuh Manusia

Daftar ini sengaja dipisahkan dari kode. Kerjakan setelah foto tersedia.

## Data

- [ ] Ambil foto primer di GKU 1 ITERA (beberapa adegan/sudut pandang).
- [ ] Konversi HEIC -> JPG 1280x960 (`tools/convert_heic.py`).
- [ ] Isi `data/database/<adegan>/refs/` (7-20 foto kosong per adegan).
- [ ] Isi `scene.json` tiap adegan (waktu, cuaca, kamera, catatan).
- [ ] Build database (`python -m src.database build`).
- [ ] Kalibrasi ROI manual per adegan (`tools/roi_editor.py`).
- [ ] Cek ROI (`tools/preview_roi.py`).
- [ ] Labeli `data/ground_truth/ground_truth.csv` (minimal 2 pelabel).
- [ ] Catat kondisi pengambilan (protokol di `docs/protokol_pengambilan_foto.md`).

## Eksperimen

- [ ] Jalankan evaluasi pada data asli (`src.evaluate`).
- [ ] Periksa `results.md`, `sensitivity.csv`, `failures.csv`.
- [ ] Tuning parameter di `src/config.py` (jika perlu), catat alasannya.
- [ ] Pastikan angka laporan berasal dari data asli, bukan `sandbox/`.

## Dokumen & luaran

- [ ] Isi proposal (`docs/proposal.md`).
- [ ] Isi laporan 5 bab (`docs/laporan_outline.md`, `docs/template_hasil.md`).
- [ ] Lengkapi daftar pustaka (`docs/bacaan.md`).
- [ ] Buat slide presentasi.
- [ ] Rekam video demo (`docs/naskah_video.md`).
- [ ] Uji aplikasi Streamlit dengan data asli.
- [ ] Isi catatan progres mingguan (`docs/progres_mingguan.md`).
- [ ] Bersihkan data sensitif (wajah/plat) sebelum berbagi.

## Verifikasi akhir

- [ ] `python -m pytest -q` lulus.
- [ ] `python -m src.database build` berhasil untuk semua adegan.
- [ ] `python -m src.evaluate ...` menghasilkan semua berkas.
- [ ] Tidak ada foto mentah besar yang ter-commit.
- [ ] README dan perintah reproduksi sudah dicoba dari awal.

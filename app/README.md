# Aplikasi Demo

`streamlit_app.py` - demo interaktif:

1. Unggah foto parkiran (JPG/PNG/HEIC).
2. Sistem mencocokkan foto ke database adegan (SIFT+RANSAC).
3. Bila cocok: tampil adegan terpilih, overlay slot hijau/merah, hitungan slot
   kosong, dan masker biner.
4. Bila tidak cocok: pesan jelas "perlu foto latar baru" beserta statistik
   inlier terbaik.

## Menjalankan

```bash
pip install -r requirements.txt -r requirements-app.txt
python -m src.database build --database data/database
streamlit run app/streamlit_app.py
```

Folder database bisa diubah di sidebar. Bila `data/database` kosong dan
`sandbox/database` ada, aplikasi menawarkan database sintetis (ditandai jelas
"SINTETIS, bukan data laporan").

```bash
# contoh memakai database sintetis
python tools/make_synthetic_demo.py
python -m src.database build --database sandbox/database
set PARKIR_DATABASE=sandbox/database   # PowerShell: $env:PARKIR_DATABASE=...
streamlit run app/streamlit_app.py
```

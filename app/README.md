# Aplikasi Demo

`streamlit_app.py` — demo interaktif: unggah citra latar + ROI + citra uji →
tampil masker biner dan overlay (hijau = kosong, merah = terisi) + hitungan slot.

## Menjalankan

```bash
pip install -r requirements.txt -r requirements-app.txt
streamlit run app/streamlit_app.py
```

Tanpa unggahan apa pun, aplikasi otomatis memakai contoh di `sandbox/`
(jalankan dulu `python tools/make_synthetic_demo.py` bila folder itu kosong).

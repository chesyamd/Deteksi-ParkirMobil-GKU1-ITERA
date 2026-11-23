# Judul perubahan

<!-- Ringkas: apa yang diubah dan mengapa. Satu-dua baris saja. -->

## Jenis perubahan
- [ ] Fitur baru (`feat`)
- [ ] Perbaikan bug (`fix`)
- [ ] Dokumentasi (`docs`)
- [ ] Percobaan/eksperimen (`exp`)
- [ ] Perapian kode (`chore`)

## Bagian yang tersentuh
- [ ] `src/` pipeline
- [ ] `tools/` alat bantu data
- [ ] `app/` aplikasi demo
- [ ] `tests/` pengujian
- [ ] data / ROI / ground truth

## Cara menguji
<!-- Perintah yang dijalankan + hasil singkat, mis: -->
<!-- python -m unittest discover -s tests -v  ->  OK (15 tests) -->

## Checklist
- [ ] `python -m unittest discover -s tests -v` lulus
- [ ] Tidak mengubah parameter di `src/config.py` tanpa menjelaskan alasannya
- [ ] Tidak meng-commit `data/raw/`, `data/background/`, atau `sandbox/`
- [ ] Sudah meminta review dari code owner terkait (`CODEOWNERS`)

# Naskah Video Demo

Durasi target: 5-8 menit. Ganti `TODO` dengan data asli. Jangan menampilkan
angka sintetis sebagai hasil penelitian.

## Persiapan

- [ ] Database adegan asli sudah di-build (`python -m src.database build`).
- [ ] Aplikasi Streamlit berjalan (`streamlit run app/streamlit_app.py`).
- [ ] Siapkan 2-3 foto uji: satu dikenali, satu sudut baru (ditolak).
- [ ] Siapkan slide singkat: masalah, pipeline, hasil asli.
- [ ] Rekam layar + suara; pastikan teks terbaca.

## Struktur & naskah

### 0:00-0:30 - Pembuka
"Kami dari Kelompok KASKOS. Tugas besar Pengolahan Citra Digital. Topik kami
deteksi slot parkir kosong di GKU 1 ITERA dari foto ponsel yang diambil
handheld."

### 0:30-1:30 - Masalah
"Kamera tidak tetap, jadi latar dan ROI tidak bisa diasumsikan sama. Kami
membuat database per sudut pandang dan menyelaraskan latar dengan SIFT dan
homografi."

Tampilkan diagram pipeline (TODO: gambar).

### 1:30-2:30 - Data & anotasi
"Satu adegan berisi 7-20 foto kosong. Kami menandai slot dengan poligon.
Ground truth dilabeli manual oleh dua anggota."
Tampilkan `roi.json` + `preview_roi`.

### 2:30-4:00 - Demo aplikasi (foto dikenali)
1. Unggah foto uji.
2. Tunjukkan adegan terpilih + jumlah inlier.
3. Tunjukkan overlay hijau/merah, hitungan slot kosong, dan masker.
4. Jelaskan ambang rasio dan versi metode.

### 4:00-5:00 - Demo penolakan
1. Unggah foto dari sudut yang belum punya adegan.
2. Tunjukkan pesan "perlu foto latar baru" dan statistik inlier.
3. Jelaskan sistem tidak menebak.

### 5:00-6:00 - Hasil asli
Tampilkan tabel `results.md`: baseline vs usulan vs ablasi. Sebutkan angka
**asli** (TODO) dan satu contoh kasus gagal dari `failures.csv`.

### 6:00-7:00 - Kesimpulan & penutup
"Kesimpulan: registrasi dan median membantu pada data handheld; ablasi
menunjukkan kontribusi tiap tahap. Keterbatasan: jumlah adegan dan kondisi
cahaya. Terima kasih."

## Checklist akhir

- [ ] Tidak ada angka sintetis.
- [ ] Semua klaim sesuai `results.md` asli.
- [ ] Video menunjukkan aplikasi benar-benar berjalan.
- [ ] Durasi sesuai ketentuan.

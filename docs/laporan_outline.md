# Outline Laporan Akhir (5 Bab)

> TODO: isi bagian bertanda TODO dengan data asli. Jangan memakai angka dari
> `sandbox/` (data sintetis).

## Bab 1 - Pendahuluan

1.1 Latar belakang (parkiran kampus, kebutuhan slot kosong).
1.2 Rumusan masalah (kamera handheld, sudut berubah, pencahayaan, slot miring).
1.3 Tujuan dan manfaat.
1.4 Batasan masalah.
1.5 Sistematika penulisan.

## Bab 2 - Landasan Teori

2.1 Pengolahan citra digital dan ruang warna.
2.2 Registrasi citra: SIFT, pencocokan fitur, ratio test, RANSAC, homografi.
2.3 Background subtraction dan median banyak frame.
2.4 Thresholding Otsu.
2.5 Morfologi citra (opening/closing).
2.6 CLAHE dan Gaussian blur.
2.7 Metrik evaluasi level slot: akurasi, precision, recall, F1, confusion matrix.
2.8 Penelitian terkait (lihat `bacaan.md`).

## Bab 3 - Metodologi

3.1 Gambaran umum pipeline (diagram alur TODO).
3.2 Akuisisi data primer:
    - perangkat, waktu, cuaca, sudut, jumlah foto;
    - kondisi handheld dan variasi sudut.
3.3 Struktur database adegan (`refs/`, `scene.json`, `roi.json`).
3.4 Pembangunan latar median (SIFT+homografi antar-refs).
3.5 Pencocokan foto uji ke adegan + penolakan foto tak dikenal.
3.6 Registrasi latar ke foto uji dan transformasi ROI.
3.7 Pra-pemrosesan, segmentasi, morfologi.
3.8 Klasifikasi slot berbasis rasio piksel.
3.9 Baseline, metode usulan, dan skenario ablasi (tabel TODO).
3.10 Parameter (rujuk `src/config.py`) dan alasan pemilihan.
3.11 Metrik dan protokol evaluasi.

## Bab 4 - Hasil dan Pembahasan

4.1 Ringkasan dataset (jumlah adegan, foto, slot, distribusi kosong/terisi).
4.2 Hasil baseline vs usulan vs ablasi (tabel dari `results.csv`).
4.3 Confusion matrix dan metrik per kelas.
4.4 Sensitivitas ambang rasio (`sensitivity.csv`).
4.5 Rekognisi adegan dan foto ditolak.
4.6 Analisis kasus gagal (`failures.csv`): contoh citra + penyebab.
4.7 Diskusi: kontribusi tiap tahap, keterbatasan, ancaman validitas.
4.8 Perbandingan dengan penelitian terkait.

## Bab 5 - Kesimpulan dan Saran

5.1 Kesimpulan (jawab rumusan masalah dengan angka asli).
5.2 Saran (adegan tambahan, anotasi, pengembangan ML/DL, video, dll).

## Lampiran

- Daftar berkas data & kondisi pengambilan.
- `roi.json` dan cuplikan `ground_truth.csv`.
- Perintah reproduksi.
- Tabel hasil lengkap.

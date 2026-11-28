# Kerangka Proposal

> TODO: lengkapi setiap bagian. Panjang proposal mengikuti panduan dosen.

## 1. Judul

Deteksi Ketersediaan Slot Parkir Mobil dari Foto Handheld di GKU 1 ITERA
Menggunakan Registrasi Citra dan Background Subtraction.

## 2. Latar Belakang

- Kebutuhan informasi slot parkir kosong di area kampus.
- Tantangan: foto diambil handheld, sudut kamera berubah, pencahayaan tidak
  seragam, kendaraan beragam warna, slot miring/trapesium.
- Mengapa pendekatan klasik (tanpa DL) layak: data terbatas, dapat dijelaskan,
  modular, dan menjadi baseline yang kuat.

## 3. Rumusan Masalah

1. Bagaimana mencocokkan foto uji ke adegan/sudut pandang yang tepat?
2. Bagaimana menyelaraskan latar kosong ke foto uji yang diambil handheld?
3. Bagaimana menentukan status kosong/terisi tiap slot secara terukur?
4. Seberapa besar kontribusi tiap tahap (registrasi, median, CLAHE, blur,
   morfologi) terhadap kinerja?

## 4. Tujuan

- Membangun pipeline deteksi slot parkir yang data-agnostik terhadap sudut
  pandang.
- Menyediakan baseline klasik dan metode usulan yang dibandingkan secara adil.
- Melakukan ablasi dan analisis kasus gagal dengan metrik level slot.

## 5. Batasan

- Data primer: foto sendiri di parkiran GKU 1 ITERA.
- Satu adegan = satu sudut pandang; foto dari sudut baru perlu adegan baru.
- Fokus slot mobil, bukan motor.
- Tidak melakukan tracking video; input berupa foto.

## 6. Metodologi (ringkas)

1. Konversi HEIC -> JPG 1280x960.
2. Database adegan: refs foto kosong, `scene.json`, `roi.json`.
3. Build latar: SIFT+homografi antar-refs -> median.
4. Foto uji: SIFT+RANSAC match ke adegan (inlier terbanyak), tolak jika gagal.
5. Registrasi latar ke foto uji, transformasi ROI.
6. Pra-pemrosesan: Gaussian blur + CLAHE.
7. Background subtraction -> Otsu -> morfologi.
8. Rasio piksel kendaraan per ROI -> status slot.
9. Evaluasi level slot + ablasi + sensitivitas ambang.

## 7. Jadwal

| Minggu | Kegiatan | Bukti |
|---|---|---|
| TODO | Pengambilan foto & konversi | daftar berkas |
| TODO | Anotasi ROI & ground truth | `roi.json`, `ground_truth.csv` |
| TODO | Eksperimen & evaluasi | `results.csv` |
| TODO | Laporan, slide, video, demo | berkas luaran |

## 8. Luaran

- Proposal, laporan akhir 5 bab, kode, slide, video demo, aplikasi demo.
- Data primer terdokumentasi (kondisi pengambilan).

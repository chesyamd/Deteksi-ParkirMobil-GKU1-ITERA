# Protokol Pengambilan Foto (Data Primer)

> TODO: sesuaikan dengan jadwal dan perangkat tim. Catat kondisi sebenarnya,
> bukan kondisi ideal.

## Tujuan

Mendapatkan foto parkiran GKU 1 ITERA yang cukup untuk:
- membangun latar tiap adegan (foto kosong),
- menguji deteksi slot (foto dengan kendaraan),
- mengevaluasi ketahanan terhadap sudut, cahaya, dan bayangan.

## Perangkat & format

- Ponsel dengan kamera 4:3 (contoh 4032x3024), format HEIC.
- Tanpa tripod (handheld) agar sesuai kondisi nyata.
- Jangan memakai zoom digital berlebihan.
- Sertakan metadata: tanggal, jam, cuaca, arah hadap, perkiraan tinggi kamera.

## Aturan per adegan (satu sudut pandang)

1. Pilih satu posisi berdiri dan arah hadap. Catat patokan (mis. "dekat tiang
   lampu, menghadap barat").
2. Ambil **7-20 foto kosong** dari posisi itu. Geser badan sedikit (10-30 cm)
   dan ubah sudut kecil; jangan berpindah jauh.
3. Ambil **10-20 foto terisi** dari posisi yang sama, pada waktu yang sama
   atau berbeda (catat). Usahakan ada campuran slot kosong/terisi.
4. Untuk setiap adegan baru (sudut berbeda), buat folder baru:
   `data/database/<id_adegan>/refs/`.
5. Foto uji disimpan di `data/raw/` dan dicatat di `ground_truth.csv`.

## Kondisi yang perlu dicakup

- Pagi, siang, sore (jika memungkinkan) untuk menguji CLAHE/robust cahaya.
- Mendung/cerah, bayangan pohon/bangunan.
- Mobil warna terang dan gelap.
- Slot terhalang sebagian (catat; boleh tidak dilabeli).
- Sudut pandang agak miring agar slot tampak trapesium (uji ROI poligon).

## Etika & privasi

- Hindari memotret wajah/orang secara sengaja.
- Jangan menyebarkan plat nomor yang terbaca jelas; blur bila perlu.
- Foto mentah tidak di-commit ke repo.

## Setelah memotret (checklist)

- [ ] Konversi HEIC -> JPG 1280x960:
      `python tools/convert_heic.py --input data/source --out data/source/jpg`
- [ ] Salin foto kosong ke `data/database/<adegan>/refs/`.
- [ ] Isi `scene.json` (id, nama, catatan, waktu, kamera).
- [ ] `python -m src.database build`
- [ ] Tandai ROI (`tools/roi_editor.py`) dan cek (`tools/preview_roi.py`).
- [ ] Salin foto uji ke `data/raw/`.
- [ ] Isi `data/ground_truth/ground_truth.csv` (minimal 2 pelabel).
- [ ] Jalankan evaluasi dan simpan `results.csv`/`results.md`.

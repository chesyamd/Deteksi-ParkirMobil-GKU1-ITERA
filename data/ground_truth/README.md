# Ground truth level slot

Satu baris = satu foto uji. Kolom:

- `image` : nama berkas di `data/raw/` (harus sama persis).
- `scene` : id adegan di `data/database/<id>/` tempat foto itu diambil.
  Boleh dikosongkan; sistem akan mencoba mencocokkan otomatis. Untuk evaluasi
  yang adil, isi manual.
- `slot_N`: `terisi` atau `kosong` untuk tiap slot di `roi.json` adegan.
  Boleh juga `1`/`0`. Sel kosong berarti slot itu tidak dilabeli (tidak dihitung).

## Aturan pelabelan manual

1. Buka foto di `data/raw/` dan `roi.json` adegan yang sesuai (gunakan
   `tools/preview_roi.py`).
2. Tentukan status tiap slot: **terisi** bila ada kendaraan yang menempati
   sebagian besar slot; **kosong** bila tidak ada kendaraan.
3. Jangan menebak slot yang terhalang/terpotong; kosongkan selnya dan catat
   alasannya di catatan tim.
4. Minimal dua anggota memeriksa silang (double annotation) untuk foto sulit.
5. Jangan mengubah `roi.json` setelah pelabelan dimulai tanpa melabeli ulang.

Mulai dari `ground_truth_template.csv`, lalu simpan sebagai
`ground_truth.csv`. File ini kecil dan **di-commit**; fotonya tidak.

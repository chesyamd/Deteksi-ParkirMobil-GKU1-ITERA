# Template Tabel Hasil

> TODO: salin angka dari `results.csv`, `sensitivity.csv`, dan `failures.csv`
> hasil data ASLI. Jangan mengisi dengan angka sintetis.

## Tabel 1 - Ringkasan dataset

| Item | Jumlah | Catatan |
|---|---|---|
| Adegan (sudut pandang) | TODO | |
| Foto kosong (refs) | TODO | |
| Foto uji berlabel | TODO | |
| Slot total (label) | TODO | |
| Slot terisi / kosong | TODO / TODO | |
| Waktu pengambilan | TODO | pagi/siang/sore |
| Cuaca | TODO | |

## Tabel 2 - Baseline vs usulan vs ablasi (level slot)

| Versi | Akurasi | Precision (makro) | Recall (makro) | F1 (makro) | TP | FP | FN | TN | Ditolak |
|---|---|---|---|---|---|---|---|---|---|
| baseline | TODO | TODO | TODO | TODO | | | | | |
| usulan | TODO | TODO | TODO | TODO | | | | | |
| ablasi_tanpa_registrasi | TODO | TODO | TODO | TODO | | | | | |
| ablasi_tanpa_median | TODO | TODO | TODO | TODO | | | | | |
| ablasi_tanpa_clahe | TODO | TODO | TODO | TODO | | | | | |
| ablasi_tanpa_blur | TODO | TODO | TODO | TODO | | | | | |
| ablasi_tanpa_morfologi | TODO | TODO | TODO | TODO | | | | | |

## Tabel 3 - Confusion matrix (positif = terisi)

| Versi | TP | FP | FN | TN |
|---|---|---|---|---|
| baseline | TODO | TODO | TODO | TODO |
| usulan | TODO | TODO | TODO | TODO |

## Tabel 4 - Sensitivitas ambang (usulan)

| Ambang rasio | Akurasi | Precision (makro) | Recall (makro) | F1 (makro) |
|---|---|---|---|---|
| 0.10 | TODO | TODO | TODO | TODO |
| ... | TODO | TODO | TODO | TODO |

## Tabel 5 - Rekognisi adegan

| Item | Jumlah |
|---|---|
| Citra ber-GT adegan | TODO |
| Cocok benar | TODO |
| Ditolak | TODO |
| Salah adegan | TODO |

## Tabel 6 - Contoh kasus gagal

| Citra | Adegan | Slot | GT | Pred | Rasio | Dugaan penyebab |
|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## Panduan analisis

- Jelaskan **mengapa** registrasi/median/CLAHE/blur/morfologi membantu atau
  tidak pada data Anda.
- Untuk tiap kasus gagal, kaitkan dengan kondisi nyata (bayangan, mobil gelap,
  slot terhalang, sudut ekstrem, perubahan cahaya).
- Sebutkan keterbatasan: jumlah adegan, variasi cuaca, akurasi anotasi manual.

# Persiapan Dataset

Sumber: korpus Fase 1 (`Korpus_Bahasa_Palembang/`), commit `3742461a1125232e56e89789c24610da24e1daa0`.
Hash file sumber disimpan di `data/source_snapshot.json`.

## Data yang dipakai

- 1313 ujaran teranotasi dari 11 penutur; 1227 dipakai dan 86 dikeluarkan.
- Alasan dikeluarkan (satu ujaran bisa memiliki lebih dari satu alasan): transkripsi tidak jelas 6, suara orang lain (OTHER) 25, rekaman berlabel Reject 83. Rincian per ujaran ada di `data/manifests/excluded.csv`.
- Semua audio berformat WAV 16 kHz, mono, 16-bit. Kesalahan format: 0.

## Pembagian data

| Split | Speaker | Ujaran | Durasi (menit) |
|---|---|---:|---:|
| train | S01, S02, S04, S07, S08, S09, S11 | 762 | 58.48 |
| validation | S03, S06 | 235 | 16.43 |
| test | S05, S10 | 230 | 16.84 |

Pembagian bersifat speaker-independent: semua ujaran dari satu penutur berada di split yang sama.
Komposisi 7/2/2 dipilih dari semua kombinasi penutur secara deterministik. Setiap split wajib
memuat penutur asli dan bukan asli serta laki-laki dan perempuan, lalu dipilih kombinasi yang
proporsi durasi dan jumlah ujarannya paling mendekati proporsi penutur. Pemilihan tidak memakai
hasil model. Pembagian disimpan di `configs/split_v1.json`.

Tidak ada penutur maupun audio (hash PCM) yang sama antar-split.

## Keterbatasan

- Hanya 11 penutur, dengan 2 penutur di validation dan 2 di test, sehingga hasil peka terhadap pilihan penutur.
- Penutur asli berbicara spontan dan penutur bukan asli membaca naskah, sehingga pengaruh kedua faktor tidak dapat dipisahkan.
- Beberapa kalimat identik muncul di lebih dari satu split karena naskah yang sama dibaca beberapa penutur (train/validation 5, train/test 2, validation/test 2).

## Format manifest

`data/manifests/{train,validation,test}.csv` berisi satu baris per ujaran: `sample_id`, `audio_path`,
`start_sec`/`end_sec` (posisi ujaran dalam rekaman), `transcript` (sesuai anotasi), metadata penutur,
dan `pcm_sha256` (hash audio ujaran).

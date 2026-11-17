# Metode dan baseline ASR — bahan laporan Orang 3

## Model dan kontrak inference

Implementasi menggunakan Whisper multilingual small melalui faster-whisper
1.2.1 dan CTranslate2 4.9.0. Checkpoint
`Systran/faster-whisper-small` dipatok pada revision
`536b0662742c02347bc0e980a01041f333bce120`; bobot, konfigurasi dan tokenizer
diambil dari snapshot yang sama dan hash berkas disimpan untuk setiap run.
Pemilihan ini disetujui pengguna untuk implementasi. Konfirmasi koordinasi kelas
dan kelompok pembanding belum dibuktikan oleh pekerjaan teknis ini.

Whisper memakai arsitektur Transformer encoder-decoder: audio diubah menjadi
fitur log-Mel, lalu decoder menghasilkan token teks. Implementasi mempertahankan
prosesor dan tokenizer pretrained. Adapter menerima waveform mono float32
16 kHz dari preprocessing Orang 2. Untuk manifest, hanya rentang anotasi
`[start_sec, end_sec)` yang diproses; rekaman topik tidak diberikan seluruhnya
sebagai satu ujaran. Audio dan transkripsi Fase 1 tetap utuh.

## Konfigurasi baseline dan eksperimen

Baseline berjalan di CPU dengan empat thread dan permintaan INT8; backend lokal
melaporkan tipe aktual `int8_float32`. Konfigurasi memakai hint bahasa Indonesia
(`id`), tugas transkripsi, beam size 1 dan temperature 0. Tidak ada prompt,
conditioning antar-ujaran, atau VAD tambahan. Hint Indonesia tidak menyatakan
bahwa Whisper menyediakan token bahasa Palembang. Referensi tidak dipakai
sebagai masukan decoder. Prediksi mentah dipertahankan untuk scorer Orang 5.

Dua konfigurasi pembanding disediakan: beam size 5 dengan faktor lain tetap,
dan beam size 1 dengan paket preprocessing RMS + trimming yang sudah disediakan
Orang 2. Pembanding kedua menguji paket transformasi, sehingga efek RMS dan
trimming tidak dapat dipisahkan. Pilihan konfigurasi akhir memerlukan WER/CER
validation dengan kebijakan scoring yang sama; data test tidak digunakan.
Run smoke hanya menguji jalur eksekusi, bukan memilih konfigurasi terbaik.

## Bukti dan batasan

Bukti eksekusi terdapat dalam `phase2/results/baseline/`; setiap direktori run
menyimpan konfigurasi, status, versi paket, perangkat, waktu aktual, commit kode,
hash manifest/preprocessing, snapshot data dan hash artefak model. Raw
`predictions.csv` disimpan lokal dan diabaikan Git karena model menghasilkan
nama pada ujaran dengan referensi `[nama]`. Contoh publik dipilih dari ujaran
tanpa identitas; pemilihan tersebut bukan estimasi akurasi korpus.

Validator manifest lama memiliki masalah portabilitas LF/CRLF. Runner mencatat
perbedaan yang terbukti hanya line ending CSV/JSON dan menolak perubahan isi
lain; PCM tiap segmen juga diperiksa terhadap hash manifest sebelum decoding.
Tidak ada penulisan ulang sumber atau snapshot. Review provenance oleh Orang 1
dan review leakage/scoring oleh Orang 5 tetap diperlukan.

Transkripsi referensi Fase 1 berasal dari draft Whisper large-v3 yang kemudian
direview. Proses penyusunan referensi ini perlu dilaporkan karena dapat
memengaruhi independensinya terhadap keluarga model yang dievaluasi. Pekerjaan
ini belum memberikan bukti fine-tuning, WER/CER final, atau keunggulan salah satu
konfigurasi. Tidak ada listening baru untuk menyimpulkan penyebab kesalahan.

## Sumber implementasi

- [Whisper resmi](https://github.com/openai/whisper): arsitektur dan penggunaan model.
- [faster-whisper v1.2.1](https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1): backend inference dan parameter decoding.
- [Checkpoint yang dipakai](https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120): konversi model dan artefak tokenizer.

# Eksperimen perbaikan Whisper small

Eksperimen memakai checkpoint `ft_lr1e-5_001` epoch 4, pembagian penutur yang sudah ada, dan set validasi yang sama. Kontrolnya adalah konfigurasi model final dengan beam 5. Setiap kandidat mengubah satu pengaturan. RMS dan trimming diperlakukan sebagai satu paket preprocessing karena keduanya diubah oleh konfigurasi yang sama.

| Kandidat | Perubahan terhadap kontrol |
| --- | --- |
| `improve_ft_rms_trim.json` | Gunakan preprocessing RMS + trimming yang sudah ada |
| `improve_ft_beam10.json` | Naikkan beam dari 5 ke 10 |
| `improve_ft_length_penalty.json` | Turunkan `length_penalty` dari nilai bawaan 1 menjadi 0,5 |

Kandidat harus mengurangi WER validasi utama setidaknya 0,5 poin persentase dan tidak menaikkan CER utama atau WER salah satu penutur. Skor utama mengecualikan referensi dengan tag `[nama]` seperti evaluasi test. Skor untuk 235 ujaran lengkap tetap dicatat agar bisa dibaca berdampingan dengan eksperimen sebelumnya. Baseline pretrained beam 5 pada test dijalankan terpisah untuk membandingkan bobot pada pengaturan dekode yang setara. Skor test tidak dipakai untuk menyeleksi varian.

Semua konfigurasi kandidat memakai CPU INT8 agar bisa dijalankan dengan setup repo. Penyaringan lokal bisa dilakukan dengan mengubah `device` menjadi `mps` dan `compute_type` menjadi `float16` pada salinan sementara semua konfigurasi, termasuk kontrol. Bandingkan hanya run dengan perangkat dan presisi yang sama. Kandidat dari penyaringan perlu dikonfirmasi dengan run CPU sebelum mengganti default.

Dari root repo, siapkan dependensi dan checkpoint seperti di README Fase 2. Gunakan ID run baru setiap kali.

```sh
python -m phase2.src.infer --config phase2/configs/experiments/finetuned_ft_lr1e-5_001_beam5.json --run-id improve_cpu_control_001
python -m phase2.src.infer --config phase2/configs/experiments/improve_ft_rms_trim.json --run-id improve_cpu_rms_trim_001
python -m phase2.src.infer --config phase2/configs/experiments/improve_ft_beam10.json --run-id improve_cpu_beam10_001
python -m phase2.src.infer --config phase2/configs/experiments/improve_ft_length_penalty.json --run-id improve_cpu_length_penalty_001
python -c 'from phase2.src.evaluate import run; import sys; print(run(sys.argv[1]))' phase2/results/experiments/improve_cpu_control_001/predictions.csv
python -m phase2.scripts.run_test --config phase2/configs/experiments/whisper_small_beam5.json --run-id improvement_pretrained_beam5_001
```

Gunakan perintah evaluator yang sama untuk setiap file prediksi validasi. Prediksi lengkap disimpan lokal karena dapat memuat nama penutur.

## Hasil penyaringan dan validasi penuh

Pilot mengambil maksimal empat ujaran tanpa tag `[nama]` per file rekaman validasi. Ujaran dipilih dalam urutan manifest dengan jarak indeks merata, termasuk ujaran pertama dan terakhir. Hasilnya adalah 24 ujaran dari delapan rekaman dan dua penutur. Semua varian pilot dan pasangan validasi penuh dijalankan pada Apple MPS dengan FP16. Daftar ID pilot, konfigurasi efektif, versi paket, hash masukan, dan jumlah edit disimpan di [pilot_metrics.json](pilot_metrics.json) dan [full_mps_metrics.json](full_mps_metrics.json).

| Eksperimen | Ujaran utama | WER | CER |
| --- | ---: | ---: | ---: |
| Pilot kontrol | 24 | 26,61% | 8,99% |
| Pilot RMS + trim | 24 | 26,21% | 8,86% |
| Pilot beam 10 | 24 | 26,21% | 8,86% |
| Pilot penalti panjang 0,5 | 24 | 27,02% | 9,06% |
| Validasi penuh, kontrol | 231 | 25,79% | 8,78% |
| Validasi penuh, RMS + trim | 231 | 25,66% | 8,76% |

RMS + trim diteruskan ke validasi penuh karena hasil pilotnya setara dengan beam 10 dan ukuran beam tetap 5. Sebelum pembulatan, WER turun 0,12 poin, atau tiga edit kata dari 2.451 kata referensi. WER S03 tetap 30,10% dan WER S06 turun dari 21,33% ke 21,08%. Penurunan agregat berada di bawah ambang 0,5 poin. Konfigurasi default tetap fine-tuned beam 5 dengan `preprocess_v1`. Beam 10 dan penalti panjang hanya memiliki bukti pilot. Konfirmasi CPU untuk kandidat tidak diperlukan karena tidak ada kandidat yang lolos validasi penuh.

## Perbandingan test pada beam 5

Kedua konfigurasi tetap memakai `preprocess_v1`, parameter dekode yang sama, CPU dengan komputasi aktual `int8_float32`, dan versi paket yang sama. Kedua run memproses 230 ujaran. Skor utama memakai 228 ujaran setelah mengecualikan dua referensi dengan `[nama]`. Skenario ini menguji perbandingan baseline dan checkpoint fine-tuned dengan beam yang disamakan. Run ini tidak dipakai untuk memilih konfigurasi.

| Model | WER | CER | Substitusi | Penghapusan | Penyisipan |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained, beam 5 | 36,42% | 12,13% | 823 | 138 | 57 |
| Fine-tuned, beam 5 | 23,51% | 7,22% | 540 | 48 | 69 |

WER turun 12,92 poin (35,46% relatif). CER turun 4,91 poin. WER turun untuk kedua penutur, dari 36,93% ke 20,05% pada S05 dan dari 35,98% ke 26,50% pada S10. Substitusi dan penghapusan berkurang, sementara penyisipan naik dari 57 menjadi 69. Dua penutur belum cukup untuk mengukur generalisasi di luar kondisi korpus.

- Pretrained beam 5 [catatan run](../test/improvement_pretrained_beam5_001/run.json) dan [metrik](../test/improvement_pretrained_beam5_001/evaluation/metrics.json).
- Fine-tuned beam 5 [catatan run](../test/improvement_ft_control_beam5_001/run.json) dan [metrik](../test/improvement_ft_control_beam5_001/evaluation/metrics.json).

Ulangi run fine-tuned dengan perintah `run_test` pada README Fase 2 dan ID baru. Simpan versi paket, perangkat, konfigurasi, dan hash manifest dan prediksi bersama angka WER/CER. Skor fine-tuned pada pengulangan ini sedikit berbeda dari arsip lama (23,43% WER dan 7,20% CER). Untuk melaporkan pengaruh fine-tuning, pakai pasangan run beam 5 dari tabel yang sama.

## Contoh transkripsi

Tiga contoh dipilih untuk memperlihatkan keberhasilan dan kesalahan yang masih muncul. WER/CER pada kolom terakhir adalah skor model fine-tuned per ujaran.

| Ujaran | Referensi | Pretrained | Fine-tuned | WER / CER final |
| --- | --- | --- | --- | ---: |
| S05_T04_0017 | cak kalian tu harus nyobo | Cak kalen torsnya obo. | cak kalian tu harus nyobo | 0,00% / 0,00% |
| S05_T04_0004 | jadi nasi tu wajib nian kau kasih | Jadi nasi itu wajib nyen kau kasih. | jadi nasi itu wajib nian kau kasih | 14,29% / 3,03% |
| S10_T02_0019 | semuonyo perlu latihan sedikit demi sedikit | Semuanya perlu latihan sedikit demi sedikit. | semuannyo perlu hatihan sedikit demi sedikit | 33,33% / 6,98% |

Pada contoh kedua, fine-tuned memulihkan `nian` tetapi mengganti `tu` menjadi `itu`. Pada contoh terakhir, model masih salah menulis `semuonyo` dan mengubah `latihan` menjadi `hatian`. Observasi ini berasal dari perbandingan teks. Aksen, noise, dan akurasi anotasi perlu diperiksa dengan mendengarkan audio sebelum dipakai sebagai penjelasan.

Spesifikasi IF4071 Fase 2 (bagian 5 dan 10) meminta evaluasi WER/CER, variasi eksperimen yang dapat dijelaskan, contoh keluaran, serta analisis kesalahan. Rubriknya tidak menentukan ambang WER/CER. Bukti di sini mendukung pemakaian model untuk demonstrasi dan pelaporan eksperimen, dengan keterbatasan pada dua penutur uji. Pemakaian korpus kelompok lain dengan teknik yang sama dan koordinasi antarkelompok perlu dikonfirmasi sesuai bagian 2 spesifikasi.

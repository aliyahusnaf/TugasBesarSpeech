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

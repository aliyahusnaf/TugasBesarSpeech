# Dokumentasi Persiapan Dataset

<!-- BEGIN GENERATED -->

## Audit korpus Fase 1

Snapshot commit: `3742461a1125232e56e89789c24610da24e1daa0`. Hash sumber tersedia di `phase2/data/source_snapshot.json`; hash ini mengikat isi file lokal yang diaudit.

- WAV ditemukan: 57; metadata: 46 file.
- Anotasi: 1313 segmen; diterima: 1227; dikecualikan: 86.
- Speaker utama: 11; nama asli tidak dimasukkan ke manifest.
- File sumber panjang yang tidak dipakai langsung: S01.wav, S02.wav, S03.wav, S04.wav, S05.wav, S06.wav, S07.wav, S08.wav, S09.wav, S10.wav, S11.wav.
- Semua WAV dibaca penuh untuk cek header, payload, sinyal nol, dan hash.
- Kesalahan format/audio: 0; warning durasi anotasi: 0.

Rekaman panjang dan rekaman per topik tidak dihitung sebagai sampel terpisah. Anotasi menunjuk rekaman per topik. `documentation/qc/segment_offsets.csv` menyediakan pemetaan ke rekaman panjang; pemeriksaan kecocokan sampel ada di `phase2/data/source_relations.csv` setelah menjalankan audit tambahan.

### Kebijakan pengecualian

Reject dikeluarkan satu file beserta seluruh segmennya. OTHER dikeluarkan karena pengguna mengonfirmasi bahwa itu orang lain, bukan S04. Segmen dengan transkripsi tidak jelas ([unclear]/event unclear), kosong, hanya tag non-ucapan, batas invalid, atau audio rusak/nol tidak dipakai. Semua alasan per ujaran ada di `data/manifests/excluded.csv`; alasan dapat lebih dari satu, jangan dijumlahkan sebagai jumlah segmen. Duplikat PCM satu speaker hanya memakai ID pertama; jika speaker berbeda, semua anggota grup ditahan karena identitas ambigu.

`[nama]`, code-switch, dan event cough/noise yang masih punya kata dipertahankan dan ditandai, bukan dihapus otomatis. Orang 2/5 perlu menyepakati penanganan tag dan ejaan untuk training/scoring; transkripsi asli tidak diubah di tahap ini.

| Split      | Speaker                           | Ujaran | Menit segmen |
| ---------- | --------------------------------- | -----: | -----------: |
| train      | S01, S02, S04, S07, S08, S09, S11 |    762 |        58.48 |
| validation | S03, S06                          |    235 |        16.43 |
| test       | S05, S10                          |    230 |        16.84 |

Durasi dihitung dari jumlah sampel pada batas anotasi, bukan panjang file penuh. Jumlah durasi segmen dapat menghitung bagian overlap lebih dari sekali. Audit teknis tidak menjamin kebenaran transkripsi; belum ada listening baru. QC lama ada di `Korpus_Bahasa_Palembang/documentation/qc/`.

### Catatan sumber panjang vs file per topik

Audit sampel ulang menemukan 41/46 file topik sama persis dengan rentang sumber panjang. Empat file berbeda satu sampel (S01_T05, S02_T01, S03_T04, S06_T02). S05_T05 cocok pada bagian awal, tetapi 307748 sampel terakhir (sekitar 19.23 detik) melewati akhir S05.wav. Jangan mengganti path anotasi dengan rekaman panjang atau menganggap keduanya identik. File per topik menjadi sumber kanonik manifest karena itulah yang dirujuk annotation.csv; masalah hubungan sumber ini dicatat untuk review Orang 2. Rincian terukur ada di source_relations.csv.

Sembilan segmen diterima berdurasi di bawah 250 ms; tidak otomatis ditolak karena ujaran kata pendek dapat valid. Daftar review juga mencakup tag/event yang masih punya transkripsi. Tidak ada segmen diterima yang beririsan dengan OTHER.

## Dataset Palembang internal v1

Sumber: https://github.com/aliyahusnaf/TugasBesarSpeech, commit `3742461a1125232e56e89789c24610da24e1daa0`. Manifest memakai data Fase 1 yang lolos kebijakan di bagian audit di dokumen ini. Versi split: `palembang_internal_v1`. File Fase 1 tetap utuh.

| Split      | Speaker                           | Ujaran | Menit segmen |
| ---------- | --------------------------------- | -----: | -----------: |
| train      | S01, S02, S04, S07, S08, S09, S11 |    762 |        58.48 |
| validation | S03, S06                          |    235 |        16.43 |
| test       | S05, S10                          |    230 |        16.84 |

### Strategi split

Pemisahan berdasarkan speaker, termasuk validation. Semua segmen dari file/speaker yang sama tetap satu split. Setiap split memuat native dan non-native serta L/P. Pilih kombinasi 7/2/2 dengan enumerasi deterministik, meminimalkan jumlah kuadrat selisih porsi durasi dan jumlah ujaran terhadap porsi speaker (7/11, 2/11, 2/11). Jika skor sama, gunakan urutan ID. Tidak memakai prediksi, WER, atau CER untuk memilih speaker. Tidak ada seed karena tidak ada random sampling. Pembagian disimpan di `configs/split_v1.json` dan dipakai ulang, bukan diacak lagi.

### Batasan

Hanya 11 speaker dan dua speaker per held-out set; hasil peka terhadap pilihan speaker. Native berjenis spontaneous, non-native read: kedua faktor terikat dan pengaruhnya tidak bisa dipisahkan begitu saja. Kalimat sama dapat muncul pada read speech antar-speaker; overlap teks dicatat di `data/dataset_stats.json`, bukan dianggap generalisasi ke kalimat baru. Ini split internal, belum mencakup korpus kelompok lain. Review Orang 5 masih diperlukan sebelum eksperimen final.

README Fase 1 menyatakan informed consent dan penggunaan akademik non-komersial. Formulir consent tidak disalin ke Fase 2; belum dilakukan penilaian ulang ruang lingkup izin distribusi. Konfirmasi dahulu untuk korpus tambahan atau publikasi baru.

Kecocokan metadata speaker tidak membuktikan tidak ada suara latar/orang lain yang belum dianotasi. Tidak ada verifikasi identitas secara biometrik/listening baru. Pemisahan speaker disimpulkan dari ID dan anotasi yang tersedia.

### Kontrak data

Satu baris = satu ujaran. `audio_path` relatif root repo TugasBesarSpeech. `start_sec`/`end_sec` relatif file itu, akhir eksklusif; konversi batas sampel dengan `round(seconds * sample_rate)`. Jangan beri seluruh file ke model dengan transkripsi satu segmen. `transcript` persis sumber; normalisasi milik Orang 2. `pcm_sha256` adalah hash byte PCM rentang segmen. Label native/gender/speech_type diambil dari metadata, bukan ditebak dari suara. Kolom `text_tags`/`events` diteruskan agar kebijakan scoring bisa ditetapkan secara terbuka.

<!-- END GENERATED -->

## Keputusan data dan handoff

### Sudah ditetapkan

- Korpus Fase 1 tetap utuh; semua file baru di `phase2/`.
- Manifest per ujaran mengikuti Utterance_ID, transkripsi asli, dan Start/End.
- Reject tidak dimasukkan, termasuk semua segmen S04_T04_001.wav.
- OTHER dikonfirmasi pengguna sebagai orang lain, bukan S04; semua segmennya tidak dimasukkan. Tidak digabung menjadi satu speaker baru.
- Segmen dengan [unclear]/event unclear dikecualikan sebagai transkripsi bermasalah.
- Split internal 7/2/2 dipilih dari data, disimpan tetap di configs/split_v1.json. Split ini belum memasukkan korpus tambahan. Tidak ada pemindahan audio.
- Tidak ada pemilihan ASR atau perubahan ejaan/normalisasi pada tahap Orang 1.

### Masih perlu koordinasi

- Orang 5: review split, kebocoran, dan kebijakan pengecualian sebelum eksperimen.
- Orang 2/5: aturan [nama], [noise], [cough], code-switch, dan tanda baca untuk training/scoring. Tag tetap dipertahankan di manifest asli. Tidak boleh memakai hasil model untuk mengganti transkripsi referensi.
- Orang 2: review segmen sangat pendek, kualitas rekaman, dan cocoknya suara-teks.
- Orang 3/4 dan koordinator kelas: teknik ASR, checkpoint, GPU, serta kelompok pasangan bahasa/teknik sama. Belum ada keputusan resmi yang tercatat.
- Penggabungan korpus lain: tunggu data, izin, metadata, dan pemetaan speaker lintas kelompok. Jangan asal menambahkan ke train; buat versi split baru.
- Konfirmasi ketentuan kelompok 6 orang dan ruang lingkup izin berbagi dataset.

### Bagian laporan Orang 1

Gunakan bagian audit dan dataset di dokumen ini sebagai bahan: asal korpus dan snapshot, native/read speech, jumlah/durasi aktual, alasan pengecualian, strategi speaker-disjoint, dan keterbatasan. Orang 6 menggabungkan tulisan akhir. Angka audit bukan hasil WER/CER atau bukti kualitas transkripsi secara listening.

## Bahan laporan - Persiapan dataset

### Tujuan dan sumber

Fase 2 bertujuan membangun ASR untuk mentranskripsikan ucapan Bahasa Palembang. Data internal berasal dari korpus Fase 1 di Korpus_Bahasa_Palembang. Versi sumber dicatat menggunakan commit Git dan hash SHA-256 audio, anotasi, metadata, serta berkas QC. Salinan asli tidak diubah selama persiapan data Fase 2.

### Audit dan pemilihan data

Korpus memiliki 11 speaker utama, 57 WAV, dan 46 entri metadata rekaman per topik. Sebelas WAV lainnya merupakan rekaman panjang per speaker dan tidak dihitung ulang sebagai sampel tambahan. Sebanyak 1.313 anotasi ujaran tersedia dengan waktu awal, akhir, dan transkripsi. Semua 57 WAV lolos pemeriksaan teknis PCM 16-bit, mono, 16 kHz dan pembacaan payload. Kebenaran isi transkripsi tidak dapat disimpulkan dari pemeriksaan teknis saja.

Sebanyak 86 ujaran tidak digunakan setelah menggabungkan alasan pengecualian: 83 berada dalam file berlabel Reject, 25 berlabel OTHER, dan 6 memiliki transkripsi tidak jelas. Kategori tersebut tumpang tindih. OTHER telah dikonfirmasi sebagai suara orang lain, bukan penutur utama S04. Rekaman dan anotasi yang dikecualikan tetap disimpan di korpus asli. Daftar alasan per ujaran tersedia di excluded.csv. Teks referensi pada data yang diterima tidak dinormalisasi atau diterjemahkan ke Bahasa Indonesia baku.

Manifest akhir internal memuat 1.227 ujaran dengan total durasi segmen sekitar 91,75 menit. Satu baris manifest menunjuk satu rentang waktu pada rekaman per topik, bukan seluruh rekaman. ID ujaran, speaker, path, transkripsi, waktu awal/ akhir, sumber, kualitas, serta pembagian dataset disimpan bersama.

### Pembagian dataset

| Set        | Speaker                           | Ujaran | Durasi segmen (menit) |
| ---------- | --------------------------------- | -----: | --------------------: |
| Train      | S01, S02, S04, S07, S08, S09, S11 |    762 |                 58,48 |
| Validation | S03, S06                          |    235 |                 16,43 |
| Test       | S05, S10                          |    230 |                 16,84 |

Pemisahan dilakukan berdasarkan speaker untuk ketiga set. Semua kandidat pembagian 7/2/2 dievaluasi secara deterministik. Tiap set diwajibkan memiliki native/non-native dan L/P. Dari kandidat yang memenuhi syarat, dipilih yang meminimalkan jumlah kuadrat selisih porsi durasi dan jumlah ujaran terhadap porsi speaker 7/11, 2/11, dan 2/11. Prediksi model dan skor evaluasi tidak digunakan untuk pemilihan ini. Konfigurasi disimpan agar eksperimen memakai split tetap.

Validasi tidak menemukan ID ujaran ganda, PCM segmen identik, maupun speaker yang berada di lebih dari satu split. Tidak ada segmen diterima yang beririsan waktu dengan anotasi OTHER. Sembilan pengujian otomatis lulus dan pembuatan ulang manifest menghasilkan file identik. Ini membuktikan konsistensi terhadap sumber dan aturan kode, bukan verifikasi identitas semua suara secara manual.

### Keterbatasan dan pekerjaan lanjutan

Jumlah speaker kecil dan tiap held-out set hanya berisi dua speaker. Native memakai spontaneous speech, sedangkan non-native memakai read speech, sehingga pengaruh kedua faktor tersebut tidak dapat dipisahkan. Ada transkripsi identik antar-split; evaluasi speaker baru tidak sama dengan evaluasi kalimat baru.

Empat file topik berbeda satu sampel dari rekaman panjang, dan bagian akhir S05_T05_001.wav tidak sepenuhnya ada di S05.wav. Manifest mengikuti rekaman per topik yang dirujuk anotasi; perbedaan ini dicatat untuk review. Sembilan ujaran di bawah 250 ms dan tag [nama]/[cough]/[noise] diteruskan untuk pemeriksaan Orang 2. Kebijakan normalisasi training/scoring harus disepakati Orang 2 dan 5 sebelum eksperimen. Belum ada listening ulang atau hasil WER/CER pada tahap ini.

Pembagian ini berlaku untuk korpus internal. Dataset kelompok lain, identitas speaker lintas kelompok, izin penggunaan, dan review Orang 5 masih perlu diselesaikan. Jika korpus atau kebijakan berubah, gunakan versi manifest/split baru dan jelaskan perubahan sebelum melakukan evaluasi akhir.

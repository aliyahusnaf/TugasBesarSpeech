# Korpus Suara Bahasa Palembang

Tugas Besar Fase 1 — IF4071 Pemrosesan Ucapan, STEI ITB, 2026/2027

Korpus ucapan bahasa Palembang ragam sehari-hari (*baso sari-sari*) untuk membangun sistem pengenal ucapan (ASR).

## Ringkasan

| | |
|---|---|
| Penutur | 11 orang (6 laki-laki, 5 perempuan), usia 19–22 tahun |
| Jenis ucapan | 5 native: spontaneous speech (bercerita berdasarkan prompt)<br>6 non-native: read speech (membaca naskah) |
| Topik | T01 pilihan studi, T02 makanan favorit, T03 hobi dan kegemaran |
| Durasi | ±10–15 menit per penutur |
| Lingkungan | Indoor, relatif tenang |
| Perekaman | Aplikasi perekam bawaan ponsel, dikonversi dengan ffmpeg |
| Format audio | WAV, PCM, 16 kHz, 16-bit, mono |

## Penamaan File

`S01_T01_001.wav` → `S01` = Speaker ID, `T01` = kode topik, `001` = nomor utterance

## Struktur Folder

```
Korpus_Bahasa_Palembang/
├── README.md
├── audio/S01 ... audio/S11    file audio per penutur
├── metadata/metadata.csv
├── annotation/annotation.csv
├── consent_form/              formulir informed consent
└── documentation/             laporan
```

**metadata.csv:** File, Speaker, Gender, Age, Language/Dialect, Origin, Native, Speech_Type, Language_Experience, Topic, Duration, Device, Environment, Quality

**annotation.csv:** File, Speaker, Start, End, Transcription, Language/Dialect, Topic, Events

## Label Kualitas

| Label | Arti |
|---|---|
| Good | Semua kriteria QC terpenuhi |
| Acceptable | Ada gangguan ringan, ucapan tetap jelas |
| Reject | Ucapan sulit dipahami, clipping parah, atau tertutup suara lain |

## Tanda dalam Transkripsi

| Tanda | Arti |
|---|---|
| `[unclear]` | Kata tidak jelas |
| `[noise]` | Noise |
| `[laugh]` | Tawa |
| `[cough]` | Batuk |
| `[overlap]` | Suara bertumpuk |

## Etika

Seluruh penutur telah menandatangani informed consent. Penutur hanya diidentifikasi dengan Speaker ID, dan nama penutur tidak dicantumkan di korpus. Data hanya digunakan untuk keperluan akademik non-komersial.

## Keterbatasan

Variasi usia penutur terbatas (19–22 tahun), dan jenis ucapan berbeda antara penutur native (spontaneous) dan non-native (read).
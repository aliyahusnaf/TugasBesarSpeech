# Fase 2 - ASR Bahasa Palembang

Bagian Orang 1: audit korpus, manifest per ujaran, split speaker, dan statistik. Script ini memakai Python 3.9+ dan standard library; tidak perlu model/GPU. Korpus `Korpus_Bahasa_Palembang/` hanya dibaca, tidak dipindah atau diubah.

Jalankan dari root repo `TugasBesarSpeech`:

```powershell
python -m phase2.scripts.prepare_manifest
python -m phase2.scripts.audit_phase1
python -m phase2.scripts.validate_manifest
python -m unittest discover -s phase2/tests -v
```

`prepare_manifest` membangun audit dan manifest. Pilihan speaker disimpan dalam `phase2/configs/split_v1.json`. Run berikutnya memakai pilihan itu. Jika hash sumber berubah, proses berhenti: tinjau perubahan dan buat versi dataset baru, jangan hapus konfigurasi hanya supaya proses lolos.

## File untuk anggota lain

- `data/manifests/train.csv`, `validation.csv`, `test.csv`: data yang dipakai.
- `data/manifests/excluded.csv`: ujaran yang tidak dipakai beserta alasannya.
- `data/speakers.csv`, `data/dataset_stats.json`: komposisi dan durasi split.
- `data/audio_inventory.csv`, `data/source_snapshot.json`: audit format/hash.
- `data/source_relations.csv`, `data/audit_details.json`: hubungan file panjang dan potongan, serta hal yang perlu ditinjau Orang 2.
- [Dokumentasi persiapan dataset](docs/dataset_preparation.md): audit, dataset card, keputusan, catatan koordinasi, dan bahan laporan Orang 1.
- `data/validation_report.json`: hasil pemeriksaan manifest terakhir.

## Membaca satu ujaran

**Satu baris menunjuk segmen, bukan seluruh file WAV.** Path relatif root repo. Contoh pembacaan array untuk pipeline Orang 2-4 (NumPy milik pipeline model):

```python
import csv
from pathlib import Path
import wave
import numpy as np

root = Path.cwd()  # Root repo TugasBesarSpeech.
with (root / "phase2/data/manifests/train.csv").open(encoding="utf-8", newline="") as f:
    row = next(csv.DictReader(f))

with wave.open(str(root / row["audio_path"]), "rb") as audio:
    sample_rate = audio.getframerate()
    start = round(float(row["start_sec"]) * sample_rate)
    end = round(float(row["end_sec"]) * sample_rate)
    audio.setpos(start)
    signal = np.frombuffer(audio.readframes(end - start), dtype="<i2").astype(np.float32) / 32768
text = row["transcript"]  # Teks asli; normalisasi dilakukan terpisah oleh Orang 2.
```

Orang 2 boleh membuat WAV per ujaran di lokasi baru, tetapi harus mempertahankan `sample_id`, sumber, split, dan transkripsi asli. Jangan trimming rekaman panjang sebelum memakai batas anotasi karena offset bisa bergeser. Orang 3/4 melatih di train dan memilih konfigurasi di validation. Test disimpan untuk evaluasi final Orang 5. Pipeline model, preprocessing, dan metrik belum diimplementasikan di pekerjaan ini.

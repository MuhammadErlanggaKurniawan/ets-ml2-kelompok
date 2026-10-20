# Belajar dengan Label Mahal

Proyek ETS **Machine Learning II** (Sains Data, UPN "Veteran" Jawa Timur), Oktober 2026.

Kelompok:
- Muhammad Erlangga Kurniawan (23083010045)
- Michael Valencio Erasmo Bagas Ganendra (23083010042)
- Rijal Apuila Khufi (23083010033)

Ketiga topik skripsi kami punya masalah yang sama: **labelnya mahal**.
- Kesegaran makanan dinilai lewat panel sensori dan penyimpanan sampai 48 jam.
- Kelarutan obat harus diukur di laboratorium.
- Mutu kuning telur puyuh hanya bisa diukur dengan memecah telurnya.

Proyek ini membangun satu pipeline (akuisisi → wrangling → augmentasi/imbalance → ensemble →
*active learning* berbasis *multi-armed bandit* → interpretasi). Pipeline itu diuji pada dua dataset publik,
dan slot untuk data citra telur sudah disiapkan.

## Notebook (jalankan di Colab)

| Notebook | Isi | Colab |
|---|---|---|
| `01_track_A_enose_kesegaran` | E-nose daging sapi: leakage, ensemble, imbalance, augmentasi, SHAP, kurva batch | [buka](https://colab.research.google.com/github/MuhammadErlanggaKurniawan/ets-ml2-kelompok/blob/main/notebooks/01_track_A_enose_kesegaran.ipynb) |
| `02_track_B_kelarutan_aqsoldb` | AqSolDB: scaffold split, ensemble, Optuna, imbalanced regression, SHAP | [buka](https://colab.research.google.com/github/MuhammadErlanggaKurniawan/ets-ml2-kelompok/blob/main/notebooks/02_track_B_kelarutan_aqsoldb.ipynb) |
| `03_reinforcement_active_learning` | Agen bandit (ε-greedy, UCB1) yang memilih strategi kueri, untuk kedua track | [buka](https://colab.research.google.com/github/MuhammadErlanggaKurniawan/ets-ml2-kelompok/blob/main/notebooks/03_reinforcement_active_learning.ipynb) |
| `04_track_C_telur_puyuh_slot` | EfficientNet-B0 + XGBoost untuk Yolk Index, **menunggu data** | [buka](https://colab.research.google.com/github/MuhammadErlanggaKurniawan/ets-ml2-kelompok/blob/main/notebooks/04_track_C_telur_puyuh_slot.ipynb) |

Notebook dibangun dari `scripts/build_notebooks.py`. Ubah kodenya di sana, lalu jalankan `python3 scripts/build_notebooks.py`.

## Versi skrip (lokal)

```bash
uv sync
uv run python scripts/download_data.py        # unduh + verifikasi SHA-256
uv run python scripts/run_enose.py            # Track A
uv run python scripts/run_enose_shap.py
uv run python scripts/run_enose_series_curve.py
uv run python scripts/run_solubility.py       # Track B
uv run python scripts/run_active_learning.py solubility
uv run python scripts/run_active_learning.py enose
uv run python scripts/make_report_data.py     # semua angka slide -> deck/data.json
```

Hasil mentah ada di `results/`.

## Data

| Dataset | Sumber | Lisensi |
|---|---|---|
| AqSolDB, 9.982 senyawa | Sorkun, Khetan & Er (2019), *Scientific Data* 6:143; mirror `maomlab/AqSolDB` | MIT (mirror) |
| E-nose daging sapi, 5 seri × 2.160 menit | Wijaya, Sarno & Zulaika (2018), *Data in Brief* 21:2414–2420; doi:10.17632/mwmhh766fc | CC BY 4.0 |

Data mentah tidak di-commit. Skrip mengunduhnya dan mencocokkan checksum di `data/raw/checksums_sha256.json`.

## Prinsip validasi

Semua prapemrosesan (imputasi, scaling, SMOTE, augmentasi, tuning) di-fit **hanya pada fold latih**.
Unit generalisasinya adalah **seri** untuk e-nose (*Leave-One-Series-Out*) dan **scaffold Bemis–Murcko**
untuk molekul. Split baris acak dilaporkan hanya sebagai pembanding untuk menunjukkan *data leakage*.

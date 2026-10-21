# Prediksi Kesegaran Daging dan Kelarutan Senyawa dengan Machine Learning

Proyek ETS **Machine Learning II** (Sains Data, UPN "Veteran" Jawa Timur), Oktober 2026.

Kelompok:
- Muhammad Erlangga Kurniawan (23083010045)
- Michael Valencio Erasmo Bagas Ganendra (23083010042)
- Rijal Apuila Khufi (23083010033)

Ketiga topik skripsi kami punya masalah yang sama: data berlabel lama dan mahal didapat.
- Kesegaran makanan dinilai lewat panel sensori dan penyimpanan sampai 48 jam.
- Kelarutan senyawa obat harus diukur di laboratorium.
- Mutu kuning telur puyuh hanya bisa diukur dengan memecah telurnya.

Proyek ini membangun satu pipeline (akuisisi → pengolahan → pemodelan → evaluasi jujur → penambahan label oleh agen
*reinforcement learning*). Pipeline itu diuji pada dua dataset publik: e-nose daging sapi dan AqSolDB.
Empat aspek ML II yang dikerjakan: augmentasi, *imbalanced data*, *ensemble learning*, dan *reinforcement learning*
(*multi-armed bandit* untuk *active learning*).

Slide presentasi: `deck/ETS_ML2_Presentasi.pptx` (13 slide, 15 menit), dibangun dari `deck/build.js`.

## Notebook (satu notebook, jalankan di Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuhammadErlanggaKurniawan/ets-ml2-kelompok/blob/main/notebooks/ETS_ML2_Pipeline_Lengkap.ipynb)

`notebooks/ETS_ML2_Pipeline_Lengkap.ipynb` berisi 13 bagian berurutan:
1. Persiapan
2. Akuisisi data
3. EDA
4. Wrangling
5. Validasi jujur
6. Ensemble + Optuna
7. Imbalance
8. Augmentasi
9. Bandit active learning
10. Tambah batch vs baris
11. SHAP
12. Slot citra telur (opsional)
13. Ringkasan

Pilih **Runtime → Run all**. Mode `"lengkap"` (sama dengan slide) butuh waktu ±60–90 menit di CPU Colab. Mode `"cepat"` untuk demo.

Notebook dibangun dari `scripts/build_notebook.py`. Ubah kodenya di sana, lalu jalankan `python3 scripts/build_notebook.py`.

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

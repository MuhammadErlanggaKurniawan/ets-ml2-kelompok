# Deteksi Kesegaran Daging Ayam dengan Sensor Gas dan Machine Learning

Proyek ETS **Machine Learning II** (Sains Data, UPN "Veteran" Jawa Timur), Oktober 2026.

Kelompok:
- Muhammad Erlangga Kurniawan (23083010045)
- Michael Valencio Erasmo Bagas Ganendra (23083010042)
- Rijal Apuila Khufi (23083010033)

## Ringkasnya

Daging ayam yang mulai busuk mengeluarkan gas. Enam sensor gas murah (seri MQ) dan satu sensor kelembapan
membentuk **hidung elektronik (e-nose)**. Proyek ini menebak kondisi ayam (**Fresh**, **Semi-fresh**, atau
**Spoiled**) hanya dari bacaan sensor.

- **Satu dataset:** e-nose daging ayam dari Anwar & Anwar (2025), 15 sesi rekaman, 218.318 bacaan.
- **Tiga teknik ML II:** augmentasi data, ensemble learning, reinforcement learning.
- **Cara uji:** Leave-One-Session-Out. Model selalu diuji pada sesi ayam yang belum pernah dilihatnya.

## Notebook (jalankan di Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuhammadErlanggaKurniawan/ets-ml2-kelompok/blob/main/notebooks/ETS_ML2_Kesegaran_Ayam.ipynb)

`notebooks/ETS_ML2_Kesegaran_Ayam.ipynb` berisi 11 bagian berurutan, dari akuisisi data sampai kesimpulan.
Pilih **Runtime → Run all** (sekitar 5 menit, tanpa GPU). File di GitHub sudah berisi hasil run, jadi bisa
dibaca tanpa menjalankan ulang.

Notebook dibangun dari `scripts/build_notebook.py`. Ubah kodenya di sana, lalu jalankan
`python3 scripts/build_notebook.py`.

## Hasil utama

| Pertanyaan | Hasil |
|---|---|
| Data dibagi acak (bocor) vs diuji per sesi (jujur) | akurasi 96,9% vs 77,6% |
| Ensemble terbaik vs model tunggal terbaik | Random Forest macro-F1 0,782 vs KNN 0,736 |
| Efek pengolahan data pada Random Forest | 0,681 → 0,782 |
| Efek augmentasi pada Random Forest | jitter −0,009, variasi kepekaan −0,015 |
| Agen RL (reward di sesi uji) | 0,61 vs "selalu ikut model" 0,44 |
| Ayam busuk yang lolos dijual (aturan agen vs ikut model) | 0 vs 254, dengan 47% kasus dicek manual |
| Tambah sesi latih 1 → 14 | macro-F1 0,58 → 0,75 |

Semua angka ada di `hasil/ringkasan.json`, dan grafiknya ada di folder `hasil/`.

## Data

Anwar, H., & Anwar, T. (2025). Quality assessment of chicken using machine learning and electronic nose.
*Sensing and Bio-Sensing Research, 47*, 100739. https://doi.org/10.1016/j.sbsr.2025.100739

Data diunduh langsung dari repositori penulis (`talhaanwarch/chicken_assessment_e-nose`, commit `e7e3ca6`) dan
dicek dengan SHA-256. Repositori itu tidak mencantumkan lisensi, jadi data mentah **tidak** disimpan di repo ini.

## Eksperimen lanjutan

| Notebook | Isi | Penanggung jawab |
|---|---|---|
| [`eksperimen_michael_ensemble.ipynb`](https://colab.research.google.com/github/MuhammadErlanggaKurniawan/ets-ml2-kelompok/blob/main/notebooks/eksperimen_michael_ensemble.ipynb) | Extra Trees, LightGBM, jumlah pohon, dan akurasi per sesi | Michael |
| [`eksperimen_rijal_rl.ipynb`](https://colab.research.google.com/github/MuhammadErlanggaKurniawan/ets-ml2-kelompok/blob/main/notebooks/eksperimen_rijal_rl.ipynb) | Seberapa peka agen RL terhadap reward, ambang yakin, dan ε | Rijal |

Kedua notebook memakai data dan fitur yang sama dengan notebook utama, dan dibangun dari `scripts/build_starters.py`.

## Slide

`deck/ETS_ML2_Kesegaran_Ayam.pptx`, dibangun dari `deck/build.js` dengan angka dari `hasil/ringkasan.json`.

## Struktur repo

| Folder | Isi |
|---|---|
| `notebooks/` | notebook utama + dua notebook eksperimen lanjutan |
| `hasil/` | grafik dan `ringkasan.json` dari run notebook utama |
| `deck/` | slide presentasi dan skrip pembuatnya |
| `scripts/` | pembuat notebook (`build_notebook.py`, `build_starters.py`) |

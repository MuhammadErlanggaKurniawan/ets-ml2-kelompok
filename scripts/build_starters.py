"""Generate the two follow-up experiment notebooks (ensemble for Michael, RL for Rijal).

Both reuse the main notebook's data and feature steps so their baselines match the slides.
Run:  python3 scripts/build_starters.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "MuhammadErlanggaKurniawan/ets-ml2-kelompok"


def notebook(cells):
    nb = {"cells": [], "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}}
    for i, (kind, src) in enumerate(cells):
        lines = src.strip("\n").split("\n")
        cell = {"cell_type": kind, "metadata": {}, "id": f"cell-{i:02d}",
                "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    return nb


def header(name, title, owner, slides, topic):
    return ("markdown", rf"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name})

# {title}
**Dikerjakan oleh:** {owner} · lanjutan dari notebook utama `ETS_ML2_Kesegaran_Ayam.ipynb` (slide {slides})

Notebook ini menguji **{topic}** lebih jauh. Data dan fiturnya sama persis dengan notebook utama, jadi angka
awalnya harus sama dengan slide.

## Cara mengerjakan
1. Jalankan sel dari atas ke bawah (**Shift + Enter**), atau pilih Runtime → Run all.
2. Di sel bertanda **✏️ UBAH DI SINI**, coba ganti angkanya, jalankan lagi, dan perhatikan hasilnya.
3. Di sel bertanda **✍️ TULIS DI SINI**, tulis pengamatanmu dengan kata-katamu sendiri (klik dua kali sel
   itu untuk mengedit).
4. Setiap selesai satu langkah, simpan ke GitHub: **File → Save a copy in GitHub**. Pilih repo
   `{REPO}`, branch `main`, path `notebooks/{name}`, lalu tulis pesan commit singkat (contoh di tiap langkah).
   Satu kali simpan = satu commit atas nama akunmu.
""")


SETUP_MD = ("markdown", r"""
---
## Langkah 0 — Siapkan data (sama dengan notebook utama)
Sel ini mengunduh data, membuat label, merangkum per 5 menit, dan membuat 19 fitur. Cukup dijalankan, tidak perlu
diubah. Butuh ±10 detik.
""")
SETUP_CODE = ("code", r"""
import hashlib, urllib.request, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")
SEED = 42
CLASSES = ["Fresh", "Semi-fresh", "Spoiled"]
URL = ("https://raw.githubusercontent.com/talhaanwarch/chicken_assessment_e-nose/"
       "e7e3ca686bce0ef3e5c59f9833c85a3190c36e6e/complete_data.csv")
SHA256 = "7c9920fcfe523af1ee2855d53f13334ddba93db5135d7868b9f63c5cd488d953"
path = Path("data/raw/complete_data.csv")
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, path)
assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256, "Isi file berbeda dari sumber asli"

SENSOR_NAMES = {"s1": "MQ-135", "s2": "MQ-2", "s3": "MQ-3", "s4": "MQ-6", "s5": "MQ-7", "s6": "MQ-9",
                "s8": "Kelembapan"}
raw = pd.read_csv(path).rename(columns={**SENSOR_NAMES, "hour": "jam", "group": "sesi"})
MQ = ["MQ-135", "MQ-2", "MQ-3", "MQ-6", "MQ-7", "MQ-9"]
SENSORS = MQ + ["Kelembapan"]
long_session = raw.groupby("sesi")["jam"].transform("max") > 15
raw["label"] = np.select([raw.jam <= np.where(long_session, 8, 5), raw.jam <= np.where(long_session, 16, 10)],
                         [0, 1], 2)


def make_windows(readings):
    r = readings.copy()
    r["urutan"] = r.groupby(["sesi", "jam"]).cumcount()
    r["n"] = r.groupby(["sesi", "jam"])["jam"].transform("size")
    r["jendela"] = r.urutan * 12 // r.n
    return r.groupby(["sesi", "jam", "jendela"]).agg(**{c: (c, "mean") for c in SENSORS},
                                                    label=("label", "first")).reset_index()


def make_features(win):
    level = np.log1p(win[MQ].clip(lower=0))
    reference = level.where(win.jam == 1).groupby(win.sesi).transform("mean")
    keep = (win.jam >= 2).to_numpy()
    rows = win[keep].reset_index(drop=True)
    level, reference = level[keep].reset_index(drop=True), reference[keep].reset_index(drop=True)
    gas = rows[MQ].clip(lower=0)
    features = pd.concat([level.add_prefix("level_"), gas.div(gas.sum(axis=1) + 1, axis=0).add_prefix("porsi_"),
                          (level - reference).add_prefix("naik_"), rows[["Kelembapan"]]], axis=1)
    return features, rows


features, data = make_features(make_windows(raw))
X, y, groups = features.to_numpy(), data.label.to_numpy(), data.sesi.to_numpy()


def loso(make_model, X_=None):
    # every window is predicted by a model that never saw its session
    X_ = X if X_ is None else X_
    pred, proba = np.empty_like(y), np.zeros((len(y), 3))
    for train, test in LeaveOneGroupOut().split(X_, y, groups):
        model = make_model().fit(X_[train], y[train])
        pred[test], proba[test] = model.predict(X_[test]), model.predict_proba(X_[test])
    return pred, proba


def score(pred):
    return {"akurasi": round(accuracy_score(y, pred), 3), "macro_f1": round(f1_score(y, pred, average="macro"), 3)}


make_rf = lambda: RandomForestClassifier(n_estimators=300, max_depth=6, random_state=SEED, n_jobs=-1)
print(f"Data siap: {len(data)} jendela, {X.shape[1]} fitur, {len(np.unique(groups))} sesi.")
""")


def write_md(text):
    return ("markdown", "### ✍️ TULIS DI SINI\n" + text + "\n\n*(klik dua kali sel ini, hapus teks petunjuk, lalu tulis jawabanmu)*")


# =====================================================================================
michael = notebook([
    header("eksperimen_michael_ensemble.ipynb", "Eksperimen Ensemble Tambahan", "Michael Valencio Erasmo Bagas Ganendra",
           "5–7", "ensemble learning"),
    SETUP_MD, SETUP_CODE,
    ("markdown", r"""
---
## Langkah 1 — Ulangi hasil Random Forest di slide
Hasilnya harus **akurasi 0,776 dan macro-F1 0,782** (slide 6). Kalau beda sedikit di Colab, itu karena versi
library; catat saja.

💾 Pesan commit contoh: `ulangi baseline Random Forest`
"""),
    ("code", r"""
pred_rf, _ = loso(make_rf)
hasil = {"Random Forest (300 pohon)": score(pred_rf)}
print(hasil)
"""),
    write_md("Apakah angkanya sama dengan slide 6? Kalau beda, berapa bedanya?"),
    ("markdown", r"""
---
## Langkah 2 — Extra Trees (ensemble bagging yang lebih acak)
Extra Trees mirip Random Forest, tapi titik potong di tiap pohon dipilih **acak**. Kadang ini membuat model lebih
tahan terhadap perbedaan antar sesi.

💾 Pesan commit contoh: `coba Extra Trees`
"""),
    ("code", r"""
from sklearn.ensemble import ExtraTreesClassifier

N_POHON = 300      # ✏️ UBAH DI SINI (coba 100 atau 500)
KEDALAMAN = 6      # ✏️ UBAH DI SINI (coba 4 atau 10)

pred_et, _ = loso(lambda: ExtraTreesClassifier(n_estimators=N_POHON, max_depth=KEDALAMAN, random_state=SEED, n_jobs=-1))
hasil[f"Extra Trees ({N_POHON} pohon, kedalaman {KEDALAMAN})"] = score(pred_et)
display(pd.DataFrame(hasil).T)
"""),
    write_md("Extra Trees lebih baik atau lebih buruk dari Random Forest? Kenapa menurutmu?"),
    ("markdown", r"""
---
## Langkah 3 — LightGBM (ensemble boosting lain)
LightGBM mirip XGBoost: pohon dibuat berurutan, tiap pohon memperbaiki kesalahan sebelumnya, tapi cara
menumbuhkan pohonnya berbeda (per daun, bukan per tingkat).

💾 Pesan commit contoh: `coba LightGBM`
"""),
    ("code", r"""
from lightgbm import LGBMClassifier

N_POHON_LGBM = 200     # ✏️ UBAH DI SINI
LEARNING_RATE = 0.05   # ✏️ UBAH DI SINI (coba 0.01 atau 0.1)

pred_lgbm, _ = loso(lambda: LGBMClassifier(n_estimators=N_POHON_LGBM, num_leaves=15, learning_rate=LEARNING_RATE,
                                           random_state=SEED, verbose=-1))
hasil[f"LightGBM ({N_POHON_LGBM} pohon, lr {LEARNING_RATE})"] = score(pred_lgbm)
display(pd.DataFrame(hasil).T)
"""),
    write_md("Bandingkan LightGBM dengan XGBoost di slide 6 (macro-F1 0,765). Mana yang lebih baik?"),
    ("markdown", r"""
---
## Langkah 4 — Apakah lebih banyak pohon selalu lebih baik?
Random Forest dicoba dengan jumlah pohon berbeda.

💾 Pesan commit contoh: `uji jumlah pohon Random Forest`
"""),
    ("code", r"""
JUMLAH_POHON = [10, 50, 100, 300]   # ✏️ UBAH DI SINI (boleh tambah, misalnya 500)

f1_per_pohon = {}
for n in JUMLAH_POHON:
    pred, _ = loso(lambda: RandomForestClassifier(n_estimators=n, max_depth=6, random_state=SEED, n_jobs=-1))
    f1_per_pohon[n] = score(pred)["macro_f1"]
    print(f"{n:>4} pohon → macro-F1 {f1_per_pohon[n]}")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(list(f1_per_pohon), list(f1_per_pohon.values()), marker="o", color="#0F5257", lw=2)
ax.set(xlabel="jumlah pohon", ylabel="macro-F1 (LOSO)", title="Random Forest: jumlah pohon vs skor")
plt.show()
"""),
    write_md("Mulai berapa pohon skornya tidak naik lagi? Apa artinya untuk ensemble?"),
    ("markdown", r"""
---
## Langkah 5 — Model mana yang paling stabil antar sesi?
Akurasi dihitung per sesi untuk tiap model. Model yang bagus bukan cuma rata-ratanya tinggi, tapi juga tidak
terlalu jelek di sesi tertentu.

💾 Pesan commit contoh: `bandingkan akurasi per sesi`
"""),
    ("code", r"""
per_sesi = pd.DataFrame({
    nama: [accuracy_score(y[groups == s], pred[groups == s]) for s in np.unique(groups)]
    for nama, pred in [("Random Forest", pred_rf), ("Extra Trees", pred_et), ("LightGBM", pred_lgbm)]
}, index=np.unique(groups))
per_sesi.index.name = "sesi"
display(per_sesi.round(3))
print("Akurasi terendah tiap model:")
print(per_sesi.min().round(3))

per_sesi.plot.bar(figsize=(10, 4), color=["#0F5257", "#9AA5A6", "#E3A33B"])
plt.ylabel("akurasi")
plt.title("Akurasi per sesi")
plt.ylim(0, 1)
plt.show()
"""),
    write_md("Sesi mana yang paling sulit untuk semua model? Model mana yang paling stabil?"),
    ("markdown", r"""
---
## Langkah 6 — Kesimpulan
💾 Pesan commit contoh: `tulis kesimpulan eksperimen ensemble`
"""),
    ("code", r"""
display(pd.DataFrame(hasil).T.sort_values("macro_f1", ascending=False))
"""),
    write_md("Tulis 3–4 kalimat: model ensemble mana yang terbaik, apakah hasilnya mengubah kesimpulan di slide 6, "
             "dan apa yang kamu pelajari."),
])

# =====================================================================================
rijal = notebook([
    header("eksperimen_rijal_rl.ipynb", "Eksperimen Reinforcement Learning: Seberapa Peka Agen terhadap Reward?",
           "Rijal Apuila Khufi", "8–9", "agen reinforcement learning"),
    SETUP_MD, SETUP_CODE,
    ("markdown", r"""
---
## Langkah 1 — Siapkan tebakan Random Forest dan agen
Agen melihat tebakan Random Forest (yang tidak pernah melihat sesi itu) dan memilih **Jual / Cek manual / Buang**.
Fungsi `jalankan_agen` di bawah sama persis dengan notebook utama, tapi reward dan pengaturannya bisa diubah.
"""),
    ("code", r"""
_, proba_rf = loso(make_rf)
predicted = proba_rf.argmax(axis=1)
layak = y <= 1                                   # Fresh or Semi-fresh = still fine to sell
ACTIONS = ["Jual", "Cek manual", "Buang"]
STATE_NAMES = [f"{c} ({'ragu' if u else 'yakin'})" for c in CLASSES for u in (False, True)]
# same fixed session split as the main notebook
RL_TEST_SESSIONS = [[0, 10, 11], [2, 8, 13], [1, 6, 9], [4, 7, 14], [3, 5, 12]]
FOLDS = [(np.flatnonzero(~np.isin(groups, t)), np.flatnonzero(np.isin(groups, t))) for t in RL_TEST_SESSIONS]


def jalankan_agen(jual_layak=1.0, jual_busuk=-5.0, cek=0.5, buang_layak=-1.0, buang_busuk=1.0,
                  ambang_yakin=0.7, epsilon=0.1, iterasi=3000, seeds=5):
    state = predicted * 2 + (proba_rf.max(axis=1) < ambang_yakin)
    reward = np.column_stack([np.where(layak, jual_layak, jual_busuk), np.full(len(y), cek),
                              np.where(layak, buang_layak, buang_busuk)])
    curves, policies = [], []
    for train_idx, test_idx in FOLDS:
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            Q, N, curve = np.zeros((6, 3)), np.zeros((6, 3)), []
            for t in range(1, iterasi + 1):
                i = rng.choice(train_idx)
                s = state[i]
                a = rng.integers(3) if rng.random() < epsilon else int(np.argmax(Q[s]))
                N[s, a] += 1
                Q[s, a] += (reward[i, a] - Q[s, a]) / N[s, a]
                if t == 1 or t % 100 == 0:
                    curve.append(reward[test_idx, Q.argmax(axis=1)[state[test_idx]]].mean())
            curves.append(curve)
            policies.append(Q.argmax(axis=1))
    policy = [int(np.bincount([p[s] for p in policies], minlength=3).argmax()) for s in range(6)]
    action = np.array(policy)[state]
    return {"reward_akhir": round(float(np.mean(curves, axis=0)[-1]), 3),
            "aturan": dict(zip(STATE_NAMES, [ACTIONS[a] for a in policy])),
            "busuk_terjual": int(((action == 0) & ~layak).sum()),
            "layak_terbuang": int(((action == 2) & layak).sum()),
            "persen_cek": round(100 * float((action == 1).mean()), 1),
            "kurva": np.mean(curves, axis=0)}


def tabel_aturan(hasil_per_setting):
    return pd.DataFrame({k: v["aturan"] for k, v in hasil_per_setting.items()})


def tabel_dampak(hasil_per_setting):
    cols = ["reward_akhir", "busuk_terjual", "layak_terbuang", "persen_cek"]
    return pd.DataFrame({k: {c: v[c] for c in cols} for k, v in hasil_per_setting.items()}).T


print("Siap.")
"""),
    ("markdown", r"""
---
## Langkah 2 — Ulangi hasil di slide
Dengan reward asli, hasilnya harus: **reward akhir 0,61**, aturan Fresh → Jual, Semi-fresh → Cek manual,
Spoiled yakin → Buang, Spoiled ragu → Cek manual, dan **0 ayam busuk terjual**.

💾 Pesan commit contoh: `ulangi hasil agen RL`
"""),
    ("code", r"""
asli = jalankan_agen()
print("Reward akhir:", asli["reward_akhir"], "| busuk terjual:", asli["busuk_terjual"],
      "| cek manual:", asli["persen_cek"], "%")
display(pd.Series(asli["aturan"], name="aksi").to_frame())
"""),
    write_md("Apakah hasilnya sama dengan slide 8 dan 9?"),
    ("markdown", r"""
---
## Langkah 3 — Kalau denda menjual ayam busuk diubah
Di slide, menjual ayam busuk diberi **−5**. Bagaimana kalau dendanya lebih ringan atau lebih berat?

💾 Pesan commit contoh: `uji denda jual ayam busuk`
"""),
    ("code", r"""
DENDA = [-2, -5, -10]   # ✏️ UBAH DI SINI

hasil_denda = {f"jual busuk {d}": jalankan_agen(jual_busuk=d) for d in DENDA}
display(tabel_aturan(hasil_denda))
display(tabel_dampak(hasil_denda))
"""),
    write_md("Aturan mana yang berubah saat denda diubah? Berapa ayam busuk yang terjual di tiap setting?"),
    ("markdown", r"""
---
## Langkah 4 — Kalau biaya cek manual diubah
Di slide, cek manual diberi **+0,5** (aman, tapi ada biaya). Makin kecil angkanya, makin mahal cek manual.

💾 Pesan commit contoh: `uji biaya cek manual`
"""),
    ("code", r"""
NILAI_CEK = [0.2, 0.5, 0.8]   # ✏️ UBAH DI SINI

hasil_cek = {f"cek {c}": jalankan_agen(cek=c) for c in NILAI_CEK}
display(tabel_aturan(hasil_cek))
display(tabel_dampak(hasil_cek))
"""),
    write_md("Kalau cek manual makin mahal, apakah agen makin jarang mengecek? Apa risikonya?"),
    ("markdown", r"""
---
## Langkah 5 — Kalau ambang "yakin" diubah
Model dianggap **yakin** kalau peluang tebakannya ≥ ambang. Di slide ambangnya 0,7.

💾 Pesan commit contoh: `uji ambang yakin`
"""),
    ("code", r"""
AMBANG = [0.6, 0.7, 0.8]   # ✏️ UBAH DI SINI

hasil_ambang = {f"ambang {a}": jalankan_agen(ambang_yakin=a) for a in AMBANG}
display(tabel_aturan(hasil_ambang))
display(tabel_dampak(hasil_ambang))
"""),
    write_md("Apa efek menaikkan atau menurunkan ambang terhadap jumlah cek manual dan keamanan?"),
    ("markdown", r"""
---
## Langkah 6 — Kalau ε (seberapa sering agen coba-coba) diubah
ε = 0,1 artinya 10% aksi dipilih acak untuk eksplorasi.

💾 Pesan commit contoh: `uji epsilon agen`
"""),
    ("code", r"""
EPSILON = [0.01, 0.1, 0.3]   # ✏️ UBAH DI SINI

fig, ax = plt.subplots(figsize=(8, 4))
langkah = np.r_[1, np.arange(100, 3001, 100)]
for e in EPSILON:
    h = jalankan_agen(epsilon=e)
    ax.plot(langkah, h["kurva"], marker="o", ms=3, label=f"ε = {e} (akhir {h['reward_akhir']})")
ax.set(xlabel="iterasi belajar agen", ylabel="rata-rata reward di sesi uji", title="Pengaruh ε terhadap belajar agen")
ax.legend()
plt.show()
"""),
    write_md("ε mana yang belajarnya paling cepat? Mana yang hasil akhirnya paling baik?"),
    ("markdown", r"""
---
## Langkah 7 — Kesimpulan
💾 Pesan commit contoh: `tulis kesimpulan eksperimen RL`
"""),
    write_md("Tulis 3–4 kalimat: seberapa peka agen terhadap angka reward, setting mana yang menurutmu paling masuk "
             "akal untuk toko sungguhan, dan apa batasannya."),
])

for name, nb in [("eksperimen_michael_ensemble.ipynb", michael), ("eksperimen_rijal_rl.ipynb", rijal)]:
    (ROOT / "notebooks" / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", name, len(nb["cells"]), "cells")

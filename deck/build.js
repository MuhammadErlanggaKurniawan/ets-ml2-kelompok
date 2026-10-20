// Build the ETS deck from deck/data.json:  node deck/build.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const OUT = path.join(__dirname, "ETS_ML2_Belajar_dengan_Label_Mahal.pptx");

const THEME = {
  name: "Label Mahal",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16262B", lt1: "FFFFFF", dk2: "0F3D3E", lt2: "EEF4F2",
    accent1: "0F5257", accent2: "E3A33B", accent3: "9AA5A6", accent4: "C4553B",
    accent5: "6C8F7D", accent6: "2E6F95", hlink: "2E6F95", folHlink: "6C8F7D",
  },
};
const H = THEME.colors; // hex values for chart options that need hex

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Belajar dengan Label Mahal";
pres.author = "Kelompok ETS ML II";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const W = 13.333, M = 0.6;
const fmt = (x, n = 3) => Number(x).toFixed(n).replace(".", ",");
const pct = (x) => `${Math.round(x * 100)}%`;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "DARK",
  background: { color: H.dk2 },
  objects: [],
  slideNumber: { x: W - 1.0, y: 7.0, w: 0.5, h: 0.3, color: "CADBD7", fontSize: 10, fontFace: "Calibri" },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: H.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.85,
        fontFace: "Cambria", fontSize: 32, bold: true, color: C.text2, valign: "middle", margin: 0 },
      text: "" } },
    { text: { text: "ETS Machine Learning II · Belajar dengan Label Mahal",
        options: { x: M, y: 7.0, w: 7, h: 0.3, fontSize: 10, color: "7A8A8C", margin: 0 } } },
  ],
  slideNumber: { x: W - 1.0, y: 7.0, w: 0.5, h: 0.3, color: "7A8A8C", fontSize: 10, fontFace: "Calibri" },
});

function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function text(s, t, o) {
  s.addText(t, { isTextBox: true, fontSize: 15, color: C.text1, valign: "top", margin: 0, ...o });
}
function card(s, x, y, w, h, fill = C.background2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill },
    line: { color: fill, width: 0 } });
}
function chip(s, x, y, label, fill = C.accent1, color = C.background1, d = 0.5) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 } });
  s.addText(label, { isTextBox: true, x, y, w: d, h: d, align: "center", valign: "middle", fontSize: 16,
    bold: true, color, margin: 0 });
}
function stat(s, x, y, w, value, label, color = C.accent1) {
  s.addText(value, { isTextBox: true, x, y, w, h: 0.95, fontFace: "Cambria", fontSize: 54, bold: true, color,
    margin: 0, valign: "bottom" });
  s.addText(label, { isTextBox: true, x, y: y + 1.0, w, h: 0.6, fontSize: 13, color: "56676A", margin: 0 });
}
const AX = { catAxisLabelColor: "56676A", valAxisLabelColor: "56676A", catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: H.dk1,
  valGridLine: { color: "E1E8E6", size: 0.75 }, catGridLine: { style: "none" },
  titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: H.dk2, legendFontFace: "+mn-lt", legendFontSize: 11 };

// ------------------------------------------------------------------ data helpers
const byModel = (arr, k = "model") => Object.fromEntries(arr.map((r) => [r[k], r]));
const enB = byModel(D["enose_models_binary"]);
const en4 = byModel(D["enose_models_4-class"]);
const sol = byModel(D.sol_models);
const split = (m, l) => D.enose_split.find((r) => r.model === m && r.label === l);
const solSplit = (sp, f) => D.sol_split.find((r) => r.model === "LightGBM" && r.split === sp && r.features === f);
const alS = D.al_solubility, alE = D.al_enose;
const SINGLE_CLS = ["LogReg", "SVM-RBF", "KNN", "DecisionTree"];
const ENS_CLS = ["RandomForest", "XGBoost", "LightGBM", "Stacking"];

// ================================================================== 1. title
pres.addSection({ title: "Pembuka" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Pembuka" });
  chip(s, M, 0.9, "₳", C.accent2, C.text2, 0.7);
  s.addText("Belajar dengan Label Mahal", { isTextBox: true, x: M, y: 1.9, w: 11.5, h: 1.3, fontFace: "Cambria",
    fontSize: 54, bold: true, color: C.background1, margin: 0 });
  s.addText("Pipeline ensemble dan active learning berbasis multi-armed bandit untuk penilaian mutu pangan dan senyawa obat",
    { isTextBox: true, x: M, y: 3.25, w: 10.5, h: 1.0, fontSize: 20, color: "CADBD7", margin: 0 });
  s.addText([
    { text: "Muhammad Erlangga Kurniawan · 23083010045", options: { breakLine: true } },
    { text: "Michael Valencio Erasmo Bagas Ganendra · 23083010042", options: { breakLine: true } },
    { text: "Rijal Apuila Khufi · 23083010033" },
  ], { isTextBox: true, x: M, y: 4.75, w: 8, h: 1.2, fontSize: 15, color: C.background1, margin: 0,
    paraSpaceAfter: 4 });
  s.addText("Tugas ETS Machine Learning II · Program Studi Sains Data · UPN \"Veteran\" Jawa Timur · Oktober 2026",
    { isTextBox: true, x: M, y: 6.55, w: 11, h: 0.4, fontSize: 12, color: "9FB8B3", margin: 0 });
  s.addNotes("Pembuka. Kami bertiga membawa tiga topik skripsi yang berbeda: kesegaran makanan, kelarutan obat, dan mutu kuning telur puyuh. Proyek ini mencari irisan ketiganya: data yang labelnya mahal.");
}

// ================================================================== 2. three theses
{
  const s = content("Tiga skripsi, satu masalah yang sama", "Pembuka");
  const cols = [
    ["Erlangga", "Kesegaran makanan", "MQ-135, pH, DHT22 pada Raspberry Pi",
     "Panel sensori 3 penilai, pengamatan sampai 48 jam", "LR · RF · SVM · XGBoost"],
    ["Michael", "Kelarutan air obat", "Deskriptor molekul 2D dari SMILES",
     "Pengukuran laboratorium (LogS)", "Gradient boosting + Optuna + SHAP"],
    ["Rijal", "Mutu kuning telur puyuh", "Citra RGB kuning telur",
     "Yolk Index: telur harus dipecah (destruktif)", "EfficientNet-B0 + XGBoost + BO"],
  ];
  const cw = 3.85, gap = 0.3, y0 = 1.45;
  cols.forEach((c, i) => {
    const x = M + i * (cw + gap);
    card(s, x, y0, cw, 4.0);
    s.addText(c[0].toUpperCase(), { isTextBox: true, x: x + 0.3, y: y0 + 0.25, w: cw - 0.6, h: 0.3, fontSize: 12,
      bold: true, color: C.accent1, charSpacing: 2, margin: 0 });
    s.addText(c[1], { isTextBox: true, x: x + 0.3, y: y0 + 0.6, w: cw - 0.6, h: 0.5, fontFace: "Cambria",
      fontSize: 21, bold: true, color: C.text2, margin: 0 });
    const rows = [["Input", c[2]], ["Label", c[3]], ["Model inti", c[4]]];
    rows.forEach((r, j) => {
      const yy = y0 + 1.3 + j * 0.85;
      s.addText(r[0], { isTextBox: true, x: x + 0.3, y: yy, w: cw - 0.6, h: 0.25, fontSize: 11, color: "56676A",
        margin: 0 });
      s.addText(r[1], { isTextBox: true, x: x + 0.3, y: yy + 0.25, w: cw - 0.6, h: 0.55, fontSize: 14,
        bold: r[0] === "Label", color: r[0] === "Label" ? C.accent4 : C.text1, margin: 0 });
    });
  });
  card(s, M, 5.7, W - 2 * M, 0.95, C.text2);
  s.addText([
    { text: "Irisan:  ", options: { bold: true, color: C.accent2 } },
    { text: "label mahal  ·  model inti berbasis boosting  ·  validasi rawan kebocoran (data berulang per batch, molekul mirip per scaffold)",
      options: { color: C.background1 } },
  ], { isTextBox: true, x: M + 0.35, y: 5.7, w: W - 2 * M - 0.7, h: 0.95, fontSize: 16, valign: "middle", margin: 0 });
  s.addNotes("Ketiga topik punya pola yang sama. Labelnya mahal: penilai sensori dan waktu tunggu, uji laboratorium, atau telur yang harus dipecah. Model intinya sama-sama berbasis boosting, dan validasinya mudah bocor. Pertanyaan proyek kami: bagaimana mendapatkan model yang jujur dan bagus dengan label sesedikit mungkin?");
}

// ================================================================== 3. questions -> 4 aspects
{
  const s = content("Empat pertanyaan, empat aspek ML II", "Pembuka");
  const q = [
    ["1", "Wrangling & augmentasi", "Apakah augmentasi deret waktu sensor membantu model mengenali batch baru?"],
    ["2", "Imbalanced data", "Apakah class weight, SMOTE, atau undersampling memperbaiki kelas minoritas?"],
    ["3", "Ensemble learning", "Apakah bagging, boosting, dan stacking mengungguli model tunggal pada validasi yang jujur?"],
    ["4", "Reinforcement", "Dapatkah agen bandit memilih strategi kueri sehingga label yang dibutuhkan lebih sedikit?"],
  ];
  q.forEach((r, i) => {
    const y = 1.5 + i * 1.22;
    chip(s, M, y + 0.1, r[0]);
    s.addText(r[1], { isTextBox: true, x: M + 0.75, y, w: 3.4, h: 0.7, fontSize: 18, bold: true, color: C.text2,
      valign: "middle", margin: 0 });
    s.addText(r[2], { isTextBox: true, x: M + 4.25, y, w: 4.6, h: 0.7, fontSize: 15, color: C.text1,
      valign: "middle", margin: 0 });
  });
  card(s, 9.75, 1.5, 2.98, 4.36, C.text2);
  s.addText("4", { isTextBox: true, x: 9.75, y: 1.85, w: 2.98, h: 1.3, align: "center", fontFace: "Cambria",
    fontSize: 80, bold: true, color: C.accent2, margin: 0 });
  s.addText("aspek ML II dikerjakan", { isTextBox: true, x: 9.95, y: 3.2, w: 2.58, h: 0.4, align: "center",
    fontSize: 15, color: C.background1, margin: 0 });
  s.addText("≥ 3 aspek → nilai minimum kelompok 75", { isTextBox: true, x: 9.95, y: 4.1, w: 2.58, h: 1.2,
    align: "center", fontSize: 14, color: "CADBD7", margin: 0 });
  s.addNotes("Kami mengerjakan empat aspek, jadi sesuai ketentuan nilai minimum kelompok menjadi 75. Deep learning EfficientNet-B0 di track telur juga terhitung sebagai aspek ensemble menurut ketentuan dosen, tetapi track itu masih menunggu data.");
}

// ================================================================== 4. pipeline
{
  const s = content("Satu pipeline untuk ketiga topik", "Pembuka");
  const steps = [
    ["Akuisisi", "Unduh dataset publik + cek SHA-256; jalur Raspberry Pi Erlangga"],
    ["Wrangling", "Fitur kausal sensor, scaffold Bemis–Murcko, 207 deskriptor RDKit"],
    ["Validasi jujur", "Leave-One-Series-Out dan scaffold split"],
    ["Augmentasi & imbalance", "Jitter, gain scaling, class weight, SMOTE"],
    ["Ensemble", "RF, XGBoost, LightGBM, stacking, Optuna (TPE)"],
    ["Loop bandit", "Agen memilih strategi kueri active learning"],
  ];
  const bw = 1.88, gap = 0.17, y = 1.75;
  steps.forEach((st, i) => {
    const x = M + i * (bw + gap);
    card(s, x, y, bw, 2.75, i === 5 ? C.text2 : C.background2);
    chip(s, x + 0.2, y + 0.2, String(i + 1), i === 5 ? C.accent2 : C.accent1, i === 5 ? C.text2 : C.background1, 0.45);
    s.addText(st[0], { isTextBox: true, x: x + 0.2, y: y + 0.8, w: bw - 0.4, h: 0.65, fontSize: 16, bold: true,
      color: i === 5 ? C.background1 : C.text2, margin: 0, valign: "top" });
    s.addText(st[1], { isTextBox: true, x: x + 0.2, y: y + 1.45, w: bw - 0.4, h: 1.2, fontSize: 12.5,
      color: i === 5 ? "CADBD7" : C.text1, margin: 0 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 4.85, w: W - 2 * M, h: 0.85, rectRadius: 0.12,
    fill: { color: C.background1 }, line: { color: C.accent1, width: 1.25 } });
  s.addText([
    { text: "Aturan emas: ", options: { bold: true, color: C.accent1 } },
    { text: "semua yang belajar dari data (imputasi, scaling, SMOTE, augmentasi, tuning, reward bandit) di-fit hanya pada data latih/validasi. Data uji hanya untuk pelaporan." },
  ], { isTextBox: true, x: M + 0.3, y: 4.85, w: W - 2 * M - 0.6, h: 0.85, fontSize: 15, valign: "middle", margin: 0,
    color: C.text1 });
  s.addText("Kode: github.com/MuhammadErlanggaKurniawan/ets-ml2-kelompok · 4 notebook Colab",
    { isTextBox: true, x: M, y: 6.0, w: W - 2 * M, h: 0.4, fontSize: 13, color: "56676A", margin: 0 });
  s.addNotes("Pipeline berjalan dari akuisisi sampai loop bandit. Satu aturan dipegang di semua tahap: apa pun yang belajar dari data hanya boleh melihat data latih atau validasi. Semua kode ada di repo GitHub dalam bentuk empat notebook Colab.");
}

// ================================================================== 5. data
pres.addSection({ title: "Data & validasi" });
{
  const s = content("Data yang dipakai dan yang belum ada", "Data & validasi");
  const rows = [
    ["Track", "Dataset", "Ukuran", "Label", "Status"],
    ["A · proksi Erlangga", "E-nose daging sapi (Wijaya dkk., 2018)", `5 seri × 2.160 menit = ${D.enose_rows.toLocaleString("id-ID")} baris; 9 sensor MOS termasuk MQ135`, "TVC 4 kelas; biner segar/tidak segar", "Dipakai"],
    ["B · Michael", "AqSolDB (Sorkun dkk., 2019)", `${D.sol_meta.n.toLocaleString("id-ID")} senyawa, ${D.sol_meta.n_scaffolds.toLocaleString("id-ID")} scaffold, ${D.sol_meta.n_rdkit_features} deskriptor`, "LogS (regresi)", "Dipakai"],
    ["C · Rijal", "Citra kuning telur puyuh", "Rencana 372 butir (2 suhu × 31 hari × 6)", "Kelas Yolk Index", "Kode siap, menunggu data"],
    ["Pilot Erlangga", "Raspberry Pi, protokol PILOT_V1", "6 baris, 1 batch; pH valid 3/6", "Panel sensori", "Belum cukup untuk validasi"],
  ];
  const head = { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13 };
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: i === 0 ? head :
      { fontSize: 13, color: j === 4 && i >= 3 ? C.accent4 : C.text1, bold: j === 4,
        fill: { color: i % 2 ? C.background1 : C.background2 } } }))),
    { x: M, y: 1.45, w: W - 2 * M, colW: [1.9, 2.9, 3.6, 2.2, 1.53], rowH: 0.7, margin: 0.08,
      border: { type: "solid", color: "DCE5E2", pt: 0.75 }, valign: "middle" });
  text(s, "Kami tidak memakai data pilot 6 baris untuk klaim performa: dengan satu batch, GroupKFold tidak bisa dijalankan, dan angka dari sana adalah artefak. Dataset e-nose dipilih karena memakai MQ135, kelembapan, dan suhu, serta punya 5 seri yang berperan seperti batch_id.",
    { x: M, y: 5.25, w: W - 2 * M, h: 1.0, fontSize: 14, color: "3A4A4D" });
  s.addNotes("Data skripsi Erlangga baru pilot satu batch, dan citra telur Rijal belum diambil. Karena itu kami memakai dataset publik yang strukturnya sama dan menyebutnya terus terang sebagai proksi. AqSolDB adalah data skripsi Michael sendiri.");
}

// ================================================================== 6. leakage
{
  const s = content("Temuan 1: validasi acak menipu", "Data & validasi");
  const r1 = split("LightGBM", "random/engineered"), r2 = split("LightGBM", "loso/engineered");
  const a1 = solSplit("random", "RDKit-2D"), a2 = solSplit("scaffold", "RDKit-2D");
  card(s, M, 1.45, 5.9, 3.5);
  s.addText("E-NOSE · macro-F1 4 kelas", { isTextBox: true, x: M + 0.35, y: 1.65, w: 5.2, h: 0.3, fontSize: 12,
    bold: true, color: C.accent1, charSpacing: 2, margin: 0 });
  stat(s, M + 0.35, 2.05, 2.6, fmt(r1.f1), "split baris acak", C.accent4);
  stat(s, M + 3.1, 2.05, 2.6, fmt(r2.f1), "Leave-One-Series-Out", C.accent1);
  text(s, "Menit yang bertetangga hampir identik.", { x: M + 0.35, y: 4.2, w: 5.2, h: 0.5, fontSize: 13, color: "56676A" });
  const x2 = M + 6.23;
  card(s, x2, 1.45, 5.9, 3.5);
  s.addText("ASQOLDB · RMSE LogS (↓)".replace("ASQOLDB", "AQSOLDB"), { isTextBox: true, x: x2 + 0.35, y: 1.65, w: 5.2, h: 0.3,
    fontSize: 12, bold: true, color: C.accent1, charSpacing: 2, margin: 0 });
  stat(s, x2 + 0.35, 2.05, 2.6, fmt(a1.rmse), "split acak", C.accent4);
  stat(s, x2 + 3.1, 2.05, 2.6, fmt(a2.rmse), "scaffold split", C.accent1);
  text(s, `Uji scaffold: ${D.sol_meta.test_scaffolds_seen_in_train} scaffold yang muncul di data latih.`,
    { x: x2 + 0.35, y: 4.2, w: 5.2, h: 0.5, fontSize: 13, color: "56676A" });
  text(s, [
    { text: "Akurasi ≈ 99% pada dataset e-nose ini hanya dapat kami reproduksi dengan split baris acak. ", options: { bold: true } },
    { text: "Relevansi: inilah alasan GroupKFold per batch_id (Erlangga) dan scaffold split (Michael) wajib dipakai." },
  ], { x: M, y: 5.3, w: W - 2 * M, h: 1.0, fontSize: 16 });
  s.addNotes(`Model dan data yang sama memberi macro-F1 ${fmt(r1.f1)} dengan split acak, tetapi hanya ${fmt(r2.f1)} saat diuji pada potongan daging yang belum pernah dilihat. Pada AqSolDB, RMSE naik dari ${fmt(a1.rmse)} ke ${fmt(a2.rmse)}. Penyebabnya kebocoran: menit yang bertetangga dan molekul satu scaffold terlalu mirip. Kami tidak menuduh paper mana pun; kami hanya menunjukkan angka 99% bisa didapat dari split acak.`);
}

// ================================================================== 7. augmentation & imbalance
pres.addSection({ title: "Aspek ML II" });
{
  const s = content("Augmentasi & imbalance: efeknya bergantung model", "Aspek ML II");
  const variants = ["none", "jitter", "scaling", "jitter+scaling", "jitter+scaling x4"];
  const labels = ["tanpa", "jitter", "scaling", "jitter+scaling", "×4"];
  const val = (m, v) => D.enose_augment.find((r) => r.model === m && r.variant === v).f1;
  s.addChart(pres.charts.BAR, [
    { name: "LogReg", labels, values: variants.map((v) => val("LogReg", v)) },
    { name: "LightGBM", labels, values: variants.map((v) => val("LightGBM", v)) },
  ], { x: M, y: 1.35, w: 7.2, h: 4.4, barDir: "col", barGrouping: "clustered", chartColors: [H.accent3, H.accent1],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 0.7,
    showTitle: true, title: "Augmentasi (train only) · macro-F1 LOSO 4 kelas", showLegend: true, legendPos: "b", ...AX });
  const im = (m, v) => D.enose_imbalance.find((r) => r.model === m && r.variant === v);
  const x = 8.2;
  card(s, x, 1.35, W - x - M, 4.4);
  s.addText("IMBALANCE · LightGBM", { isTextBox: true, x: x + 0.3, y: 1.55, w: 4, h: 0.3, fontSize: 12, bold: true,
    color: C.accent1, charSpacing: 2, margin: 0 });
  const rows = [["Variasi", "F1", "Recall excellent"]].concat(
    [["tanpa", "class_weight=None"], ["class weight", "class_weight=balanced"], ["SMOTE", "SMOTE"], ["undersampling", "RandomUnder"]]
      .map(([lab, v]) => [lab, fmt(im("LightGBM", v).f1), fmt(im("LightGBM", v).recall_excellent)]));
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { fontSize: 13, bold: i === 0, color: C.text1 } }))),
    { x: x + 0.3, y: 2.0, w: W - x - M - 0.6, colW: [1.7, 0.85, 1.38], rowH: 0.45, margin: 0.05,
      border: { type: "solid", color: "DCE5E2", pt: 0.5 } });
  text(s, "Selisih ±0,02 masih di dalam simpangan baku antar seri (0,06–0,13).", { x: x + 0.3, y: 4.45, w: W - x - M - 0.6,
    h: 1.0, fontSize: 13, color: "56676A" });
  text(s, [
    { text: "Bacaan: ", options: { bold: true } },
    { text: `gain scaling membantu model linear (+${fmt(val("LogReg", "scaling") - val("LogReg", "none"), 3)}) tetapi menurunkan LightGBM. Augmentasi dan resampling bukan obat universal; manfaatnya harus diuji dengan validasi yang benar.` },
  ], { x: M, y: 6.0, w: W - 2 * M, h: 0.85, fontSize: 15 });
  s.addNotes("Augmentasi dibuat pada sinyal mentah data latih: jitter mensimulasikan derau sensor, gain scaling mensimulasikan perbedaan sensitivitas antar unit sensor. Data uji tetap asli. Model linear terbantu, LightGBM justru turun. Penanganan imbalance memberi perubahan kecil yang masih di dalam simpangan baku, jadi kami tidak mengklaim ada teknik yang lebih unggul secara bermakna.");
}

// ================================================================== 8. ensemble e-nose
{
  const s = content("Ensemble e-nose: boosting paling stabil lintas seri", "Aspek ML II");
  const names = SINGLE_CLS.concat(ENS_CLS);
  const disp = { LogReg: "Logistic Reg.", "SVM-RBF": "SVM-RBF", KNN: "KNN", DecisionTree: "Decision Tree",
    RandomForest: "Random Forest", XGBoost: "XGBoost", LightGBM: "LightGBM", Stacking: "Stacking" };
  s.addChart(pres.charts.BAR, [
    { name: "Model tunggal", labels: names.map((n) => disp[n]), values: names.map((n) => (SINGLE_CLS.includes(n) ? enB[n].f1 : 0)) },
    { name: "Ensemble", labels: names.map((n) => disp[n]), values: names.map((n) => (ENS_CLS.includes(n) ? enB[n].f1 : 0)) },
  ], { x: M, y: 1.35, w: 7.6, h: 4.9, barDir: "col", barGrouping: "stacked", chartColors: [H.accent3, H.accent1],
    showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: "0.00;;;", dataLabelColor: H.lt1,
    valAxisMinVal: 0, valAxisMaxVal: 1, showTitle: true, title: "Biner segar vs tidak segar · macro-F1 LOSO (5 seri)",
    showLegend: true, legendPos: "b", ...AX, dataLabelColor: H.lt1 });
  const x = 8.6;
  card(s, x, 1.35, W - x - M, 4.9, C.text2);
  s.addText(fmt(enB.XGBoost.f1), { isTextBox: true, x: x + 0.35, y: 1.6, w: 3.4, h: 1.0, fontFace: "Cambria", fontSize: 56,
    bold: true, color: C.accent2, margin: 0 });
  s.addText(`XGBoost, sd ${fmt(enB.XGBoost.sd)}`, { isTextBox: true, x: x + 0.35, y: 2.6, w: 3.4, h: 0.4, fontSize: 14,
    color: C.background1, margin: 0 });
  s.addText([
    { text: `Model tunggal terbaik: Decision Tree ${fmt(enB.DecisionTree.f1)} (sd ${fmt(enB.DecisionTree.sd)}).`, options: { breakLine: true } },
    { text: `Simpangan baku XGBoost ${Math.round(enB.DecisionTree.sd / enB.XGBoost.sd)}× lebih kecil: stabil di setiap potongan daging.`, options: { breakLine: true } },
    { text: `Pada 4 kelas, keunggulan tidak jelas: LightGBM ${fmt(en4.LightGBM.f1)} vs Decision Tree ${fmt(en4.DecisionTree.f1)}.` },
  ], { isTextBox: true, x: x + 0.35, y: 3.25, w: W - x - M - 0.7, h: 2.8, fontSize: 14, color: "CADBD7", margin: 0,
    paraSpaceAfter: 8 });
  s.addNotes("Pada tugas biner, label yang sama dengan skripsi Erlangga, XGBoost unggul jelas dan simpangan bakunya paling kecil, artinya performanya stabil di kelima potongan daging. Meta-learner stacking juga dilatih dengan LOSO di dalam fold latih supaya tidak bocor. Pada 4 kelas kami sampaikan terus terang bahwa ensemble tidak unggul jelas.");
}

// ================================================================== 9. ensemble solubility
{
  const s = content("Ensemble kelarutan: stacking terbaik, Optuna tidak", "Aspek ML II");
  const names = ["Ridge", "KNN", "SVR-RBF", "DecisionTree", "RandomForest", "XGBoost", "LightGBM", "Stacking", "LightGBM-Optuna"];
  const singles = ["Ridge", "KNN", "SVR-RBF", "DecisionTree"];
  const disp = { DecisionTree: "Decision Tree", RandomForest: "Random Forest", "LightGBM-Optuna": "LightGBM+Optuna" };
  const lab = names.map((n) => disp[n] || n);
  s.addChart(pres.charts.BAR, [
    { name: "Model tunggal", labels: lab, values: names.map((n) => (singles.includes(n) ? sol[n].rmse : 0)) },
    { name: "Ensemble", labels: lab, values: names.map((n) => (!singles.includes(n) ? sol[n].rmse : 0)) },
  ], { x: M, y: 1.35, w: 7.6, h: 4.9, barDir: "bar", barGrouping: "stacked", chartColors: [H.accent3, H.accent1],
    showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: "0.000;;;", valAxisMinVal: 0, valAxisMaxVal: 1.8,
    catAxisOrientation: "maxMin", showTitle: true, title: "RMSE uji scaffold (lebih kecil lebih baik)", showLegend: true,
    legendPos: "b", ...AX, dataLabelColor: H.lt1 });
  const x = 8.6, w = W - x - M;
  card(s, x, 1.35, w, 2.25);
  s.addText(fmt(sol.Stacking.rmse), { isTextBox: true, x: x + 0.3, y: 1.5, w: 2.2, h: 0.9, fontFace: "Cambria", fontSize: 44,
    bold: true, color: C.accent1, margin: 0 });
  s.addText(`Stacking (Ridge+SVR+RF+LGBM → Ridge), R² ${fmt(sol.Stacking.r2)}`, { isTextBox: true, x: x + 0.3, y: 2.45,
    w: w - 0.6, h: 0.9, fontSize: 13, color: C.text1, margin: 0 });
  card(s, x, 3.8, w, 2.45, C.text2);
  s.addText("Optuna (40 trial TPE)", { isTextBox: true, x: x + 0.3, y: 3.95, w: w - 0.6, h: 0.35, fontSize: 14, bold: true,
    color: C.accent2, margin: 0 });
  s.addText([
    { text: `CV RMSE ${fmt(D.sol_meta.optuna_best_cv_rmse)} → uji ${fmt(sol["LightGBM-Optuna"].rmse)}`, options: { breakLine: true, bold: true } },
    { text: "Data uji berisi scaffold langka (distribution shift). Tuning tidak bisa menggantikan data yang mewakili domain target." },
  ], { isTextBox: true, x: x + 0.3, y: 4.35, w: w - 0.6, h: 1.8, fontSize: 13, color: C.background1, margin: 0,
    paraSpaceAfter: 6 });
  s.addNotes(`Stacking memberi RMSE terendah ${fmt(sol.Stacking.rmse)}, sesuai teori bahwa stacking menggabungkan model heterogen yang galatnya berbeda. Optuna menurunkan RMSE validasi silang menjadi ${fmt(D.sol_meta.optuna_best_cv_rmse)}, tetapi di data uji hasilnya ${fmt(sol["LightGBM-Optuna"].rmse)}, setara LightGBM default. Fold CV masih berisi scaffold yang umum, sedangkan data uji berisi scaffold langka.`);
}

// ================================================================== 10. imbalanced regression
{
  const s = content("Imbalanced regression: ekstrem membaik, tengah memburuk", "Aspek ML II");
  const lab = ["< −6", "−6..−4", "−4..−2", "−2..0", "≥ 0"];
  s.addChart(pres.charts.BAR, [
    { name: "tanpa bobot", labels: lab, values: D.sol_per_bin.map((r) => r.mae_unweighted) },
    { name: "bobot frekuensi-invers", labels: lab, values: D.sol_per_bin.map((r) => r.mae_weighted) },
  ], { x: M, y: 1.35, w: 7.6, h: 4.6, barDir: "col", barGrouping: "clustered", chartColors: [H.accent3, H.accent2],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1.7,
    showTitle: true, title: "MAE per kategori LogS (data uji scaffold)", showLegend: true, legendPos: "b", ...AX });
  const x = 8.6, w = W - x - M;
  card(s, x, 1.35, w, 4.6);
  const ex = D.sol_per_bin;
  s.addText([
    { text: "Kategori sangat tidak larut", options: { bold: true, breakLine: true } },
    { text: `MAE ${fmt(ex[0].mae_unweighted)} → ${fmt(ex[0].mae_weighted)}`, options: { color: C.accent1, breakLine: true } },
    { text: "Kategori sangat larut", options: { bold: true, breakLine: true } },
    { text: `MAE ${fmt(ex[4].mae_unweighted)} → ${fmt(ex[4].mae_weighted)}`, options: { color: C.accent1, breakLine: true } },
    { text: "RMSE total", options: { bold: true, breakLine: true } },
    { text: `${fmt(sol["LightGBM-Optuna"].rmse)} → ${fmt(D.sol_weighted_overall.rmse)}`, options: { color: C.accent4 } },
  ], { isTextBox: true, x: x + 0.3, y: 1.6, w: w - 0.6, h: 3.0, fontSize: 15, color: C.text1, margin: 0, paraSpaceAfter: 4 });
  text(s, "Pilih sesuai tujuan: untuk screening senyawa yang sangat tidak larut, bobot ini berguna.",
    { x: x + 0.3, y: 4.7, w: w - 0.6, h: 1.1, fontSize: 13, color: "56676A" });
  s.addNotes("Untuk regresi, imbalance berarti nilai ekstrem jarang muncul. Kami memberi bobot sampel sebanding kebalikan frekuensi kategori kelarutannya. Galat di kedua ekstrem turun, galat di tengah naik, dan RMSE total sedikit naik. Ini trade-off yang harus dipilih sesuai tujuan aplikasi.");
}

// ================================================================== 11. bandit concept
pres.addSection({ title: "Reinforcement" });
{
  const s = content("Reinforcement: agen bandit memilih strategi kueri", "Reinforcement");
  const loop = [
    ["Keadaan", "Model + data berlabel saat ini"],
    ["Aksi", "Pilih arm: random · uncertainty · diversity"],
    ["Oracle", "Beri label pada satu batch"],
    ["Reward", "Perbaikan relatif skor validasi"],
  ];
  const cx = M, cy = 1.5, bw = 2.7, bh = 1.25;
  const pos = [[cx, cy], [cx + 3.2, cy], [cx + 3.2, cy + 2.0], [cx, cy + 2.0]];
  loop.forEach((l, i) => {
    const [x, y] = pos[i];
    card(s, x, y, bw, bh, i === 1 ? C.text2 : C.background2);
    s.addText(l[0], { isTextBox: true, x: x + 0.25, y: y + 0.15, w: bw - 0.5, h: 0.4, fontSize: 17, bold: true,
      color: i === 1 ? C.accent2 : C.text2, margin: 0 });
    s.addText(l[1], { isTextBox: true, x: x + 0.25, y: y + 0.55, w: bw - 0.5, h: 0.65, fontSize: 13,
      color: i === 1 ? C.background1 : C.text1, margin: 0 });
  });
  const arrow = (x, y, w, h, flipH = false, flipV = false) => s.addShape(pres.shapes.LINE,
    { x, y, w, h, flipH, flipV, line: { color: H.accent1, width: 2, endArrowType: "triangle" } });
  arrow(cx + bw + 0.05, cy + bh / 2, 0.4, 0);
  arrow(cx + 3.2 + bw / 2, cy + bh + 0.05, 0, 0.65);
  arrow(cx + bw + 0.05, cy + 2.0 + bh / 2, 0.4, 0, true);
  arrow(cx + bw / 2, cy + bh + 0.05, 0, 0.65, false, true);
  const x = 7.1, w = W - x - M;
  card(s, x, 1.5, w, 3.25);
  s.addText([
    { text: "ε-greedy + Monte Carlo", options: { bold: true, color: C.accent1, breakLine: true } },
    { text: "Q(a) ← Q(a) + (r − Q(a)) / n(a);  eksplorasi dengan peluang ε = 0,2", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "UCB1", options: { bold: true, color: C.accent1, breakLine: true } },
    { text: "pilih argmax  Q(a) + c·√(2 ln N / n(a))" },
  ], { isTextBox: true, x: x + 0.3, y: 1.7, w: w - 0.6, h: 2.9, fontSize: 15, color: C.text1, margin: 0 });
  text(s, [
    { text: "Kenapa disebut reinforcement: ", options: { bold: true } },
    { text: "ini RL tanpa state (multi-armed bandit), jembatan bandit → RL di materi kuliah. Gagasan mengikuti Active Learning by Learning (Hsu & Lin, AAAI 2015). Data uji tidak pernah dipakai untuk reward." },
  ], { x: M, y: 5.25, w: W - 2 * M, h: 1.3, fontSize: 15 });
  s.addNotes("Setiap ronde agen memilih satu strategi kueri: acak, paling tidak yakin, atau paling beragam. Oracle memberi label, model dilatih ulang, dan reward adalah perbaikan skor validasi. Agen memperbarui estimasi nilai tiap arm dengan rata-rata Monte Carlo seperti di materi kuliah, lalu memilih dengan epsilon-greedy atau UCB1. Ini multi-armed bandit, bukan deep RL, dan kami menyebutnya demikian.");
}

// ================================================================== 12. AL solubility
{
  const s = content("Kelarutan: bandit menghemat ≈ setengah label", "Reinforcement");
  const pols = ["random", "uncertainty", "diversity", "bandit-eps", "bandit-ucb"];
  const names = { random: "random", uncertainty: "uncertainty", diversity: "diversity", "bandit-eps": "bandit ε-greedy", "bandit-ucb": "bandit UCB1" };
  const labels = alS.n_labelled.map((n) => n.toLocaleString("id-ID"));
  s.addChart(pres.charts.LINE, pols.map((p) => ({ name: names[p], labels, values: alS.curves[p] })),
    { x: M, y: 1.35, w: 7.9, h: 4.9, chartColors: [H.accent3, H.accent6, H.accent5, H.accent2, H.accent4],
      lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 4, valAxisMinVal: 1.3, valAxisMaxVal: 1.9,
      valAxisLabelFormatCode: "0.0", showTitle: true, title: "RMSE uji scaffold vs jumlah label (rata-rata 5 seed)",
      showLegend: true, legendPos: "b", ...AX, catAxisLabelFontSize: 10 });
  const x = 8.9, w = W - x - M;
  const save = 1 - alS.labels_to_match_random["bandit-ucb"] / alS.labels_to_match_random.random;
  card(s, x, 1.35, w, 4.9, C.text2);
  s.addText(pct(save), { isTextBox: true, x: x + 0.3, y: 1.55, w: w - 0.6, h: 1.0, fontFace: "Cambria", fontSize: 56,
    bold: true, color: C.accent2, margin: 0 });
  s.addText(`label lebih hemat: UCB1 menyamai RMSE akhir random (${fmt(alS.target_random_final)}) dengan ${alS.labels_to_match_random["bandit-ucb"].toLocaleString("id-ID")} label, bukan ${alS.labels_to_match_random.random.toLocaleString("id-ID")}`,
    { isTextBox: true, x: x + 0.3, y: 2.6, w: w - 0.6, h: 1.2, fontSize: 13, color: C.background1, margin: 0 });
  s.addText([
    { text: `2.100 label: bandit ${fmt(alS.final["bandit-ucb"].mean)} vs random ${fmt(alS.final.random.mean)}`, options: { breakLine: true } },
    { text: `Seluruh pool (${alS.pool_size.toLocaleString("id-ID")} label): ${fmt(alS.full_pool_test)}`, options: { breakLine: true } },
    { text: "Bandit setara strategi terbaik tanpa perlu tahu strategi itu sebelumnya." },
  ], { isTextBox: true, x: x + 0.3, y: 3.95, w: w - 0.6, h: 2.2, fontSize: 13, color: "CADBD7", margin: 0, paraSpaceAfter: 6 });
  s.addNotes(`Mulai dari 100 label, tiap ronde menambah 100, selama 20 ronde dengan 5 seed. Bandit UCB1 sudah menyamai kualitas akhir sampling acak pada ${alS.labels_to_match_random["bandit-ucb"]} label, sehingga hemat sekitar ${pct(save)} label. Dengan sepertiga pool, RMSE-nya hanya terpaut sedikit dari model yang memakai seluruh pool. Nilai jual bandit: tidak perlu menebak strategi terbaik di awal.`);
}

// ================================================================== 13. AL e-nose + batches
{
  const s = content("E-nose: yang kurang adalah batch, bukan label", "Reinforcement");
  const pols = ["random", "uncertainty", "diversity", "bandit-eps", "bandit-ucb"];
  const names = { random: "random", uncertainty: "uncertainty", diversity: "diversity", "bandit-eps": "bandit ε-greedy", "bandit-ucb": "bandit UCB1" };
  s.addChart(pres.charts.LINE, pols.map((p) => ({ name: names[p], labels: alE.n_labelled.map(String), values: alE.curves[p] })),
    { x: M, y: 1.35, w: 6.2, h: 4.55, chartColors: [H.accent3, H.accent6, H.accent5, H.accent2, H.accent4], lineSize: 2,
      lineDataSymbol: "circle", lineDataSymbolSize: 4, valAxisMinVal: 0.7, valAxisMaxVal: 0.95, valAxisLabelFormatCode: "0.00",
      showTitle: true, title: "Active learning · macro-F1 seri uji vs jumlah label", showLegend: true, legendPos: "b", ...AX,
      catAxisLabelFontSize: 10 });
  const sc = D.series_curve;
  s.addChart(pres.charts.BAR, [{ name: "macro-F1", labels: sc.map((r) => r.label), values: sc.map((r) => r.f1) }],
    { x: 7.05, y: 1.35, w: W - 7.05 - M, h: 4.55, barDir: "bar", chartColors: [H.accent1], showValue: true,
      dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, catAxisOrientation: "maxMin",
      showTitle: true, title: "Tambah batch vs tambah baris (XGBoost biner)", showLegend: false, ...AX, catAxisLabelFontSize: 11 });
  const b4_160 = sc.find((r) => r.label.startsWith("4 seri · 160")).f1, b3 = sc.find((r) => r.label.startsWith("3 seri")).f1;
  text(s, [
    { text: `160 baris dari 4 seri (${fmt(b4_160, 2)}) mengalahkan 6.480 baris dari 3 seri (${fmt(b3, 2)}). `, options: { bold: true } },
    { text: `AL memilih baris, padahal masalahnya pergeseran domain antar seri; random tetap paling aman (${fmt(alE.final.random.mean)}). Implikasi: rancangan 16 batch Erlangga sudah tepat.` },
  ], { x: M, y: 6.05, w: W - 2 * M, h: 0.85, fontSize: 14 });
  s.addNotes("Pada e-nose, active learning tidak menang: 40 label sudah memberi F1 sekitar 0,83 dan kurvanya hampir datar. Grafik kanan menjelaskan sebabnya. Menambah seri menaikkan F1 tajam, sedangkan menambah baris dari seri yang sama hampir tidak berpengaruh. Untuk skripsi Erlangga, ini argumen empiris bahwa 16 batch lebih penting daripada banyak bacaan per batch.");
}

// ================================================================== 14. SHAP
pres.addSection({ title: "Diskusi" });
{
  const s = content("Apa yang dipelajari model", "Diskusi");
  const st = Object.entries(D.sol_shap_top).slice(0, 8);
  s.addChart(pres.charts.BAR, [{ name: "mean |SHAP|", labels: st.map((r) => r[0]), values: st.map((r) => r[1]) }],
    { x: M, y: 1.35, w: 5.9, h: 4.3, barDir: "bar", chartColors: [H.accent1], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0.00", catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 0.7, showTitle: true,
      title: "Kelarutan · LightGBM+Optuna", showLegend: false, ...AX });
  const se = Object.entries(D.enose_shap_sensors);
  s.addChart(pres.charts.BAR, [{ name: "mean |SHAP|", labels: se.map((r) => r[0]), values: se.map((r) => r[1]) }],
    { x: 6.83, y: 1.35, w: W - 6.83 - M, h: 4.3, barDir: "bar", chartColors: [H.accent2], showValue: true,
      dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 5,
      showTitle: true, title: "E-nose biner · XGBoost, dijelaskan pada seri uji", showLegend: false, ...AX });
  text(s, [
    { text: "MolLogP paling berpengaruh, ", options: { bold: true } },
    { text: "konsisten dengan kimia: lipofilisitas berbanding terbalik dengan kelarutan air. " },
    { text: `Pada e-nose, MQ4 dan MQ8 dominan, sedangkan MQ135 hanya ${fmt(D.enose_shap_sensors.MQ135, 2)}. `, options: { bold: true } },
    { text: "Domainnya daging mentah, jadi ini hipotesis untuk alat Erlangga, bukan kesimpulan." },
  ], { x: M, y: 5.85, w: W - 2 * M, h: 1.0, fontSize: 14 });
  s.addNotes("SHAP pada model kelarutan menempatkan MolLogP di urutan pertama, sesuai kimia. Pada e-nose, sensor metana MQ4 dan hidrogen MQ8 paling menentukan, sedangkan MQ135 kecil. Karena domainnya daging sapi mentah, ini kami ajukan sebagai hipotesis: apakah satu sensor MQ-135 cukup untuk makanan matang?");
}

// ================================================================== 15. discussion grid
{
  const s = content("Analisis dan diskusi", "Diskusi");
  const g = [
    ["Batasan", ["Hanya 5 seri e-nose; sd besar", "Track A proksi: TVC & daging mentah ≠ panel sensori & makanan matang", "AL disimulasikan dari label yang sudah ada", "Track C belum punya data"]],
    ["Trade-off", ["Augmentasi: bantu linear, rugikan boosting", "Bobot imbalance: ekstrem membaik, tengah memburuk", "Stacking: terbaik di kelarutan, mahal komputasi", "Bandit: hemat label, tambah latih ulang"]],
    ["Sumber galat", ["Pergeseran domain antar seri (ambien tak terkontrol)", "Scaffold langka di data uji", "Variasi pengukuran LogS lintas 9 sumber", "Asumsi baseline: pemantauan mulai saat segar"]],
    ["Pengembangan", ["Data 16 batch Erlangga dengan pipeline yang sama", "Uji sensor MQ4/MQ8 tambahan", "AL tingkat batch: agen memilih sesi, bukan baris", "Citra Rijal: hemat label = hemat telur dipecah"]],
  ];
  const cw = (W - 2 * M - 0.3) / 2, ch = 2.55;
  g.forEach((b, i) => {
    const x = M + (i % 2) * (cw + 0.3), y = 1.35 + Math.floor(i / 2) * (ch + 0.25);
    card(s, x, y, cw, ch);
    s.addText(b[0], { isTextBox: true, x: x + 0.3, y: y + 0.18, w: cw - 0.6, h: 0.4, fontSize: 18, bold: true,
      color: C.accent1, margin: 0 });
    s.addText(b[1].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < b[1].length - 1 } })),
      { isTextBox: true, x: x + 0.3, y: y + 0.65, w: cw - 0.6, h: ch - 0.8, fontSize: 14, color: C.text1, margin: 0,
        paraSpaceAfter: 3 });
  });
  s.addNotes("Batasan utama: jumlah seri kecil dan track A adalah proksi. Trade-off muncul di augmentasi, bobot imbalance, stacking, dan bandit. Sumber galat terbesar adalah pergeseran domain. Pengembangan paling penting adalah menjalankan pipeline ini pada data 16 batch Erlangga dan data citra Rijal, serta active learning di tingkat batch.");
}

// ================================================================== 16. conclusion
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Diskusi" });
  s.addText("Kesimpulan", { isTextBox: true, x: M, y: 0.6, w: 8, h: 0.9, fontFace: "Cambria", fontSize: 40, bold: true,
    color: C.background1, margin: 0 });
  const k = [
    ["Validasi lebih menentukan daripada model", `Split acak ${fmt(split("LightGBM", "random/engineered").f1, 2)} vs LOSO ${fmt(split("LightGBM", "loso/engineered").f1, 2)}`],
    ["Boosting kuat dan stabil; stacking menang tipis", `XGBoost biner ${fmt(enB.XGBoost.f1, 2)} · stacking RMSE ${fmt(sol.Stacking.rmse, 2)}`],
    ["Bandit hemat label saat masalahnya kekurangan label", `≈ ${pct(1 - alS.labels_to_match_random["bandit-ucb"] / alS.labels_to_match_random.random)} lebih sedikit label di AqSolDB`],
    ["Saat masalahnya domain shift, tambah batch", "Tidak cukup sekadar menambah baris; ini relevan untuk 16 batch Erlangga"],
  ];
  k.forEach((r, i) => {
    const y = 1.85 + i * 1.1;
    chip(s, M, y + 0.05, String(i + 1), C.accent2, C.text2, 0.55);
    s.addText(r[0], { isTextBox: true, x: M + 0.85, y, w: 7.2, h: 0.45, fontSize: 19, bold: true, color: C.background1, margin: 0 });
    s.addText(r[1], { isTextBox: true, x: M + 0.85, y: y + 0.45, w: 7.2, h: 0.4, fontSize: 14, color: "CADBD7", margin: 0 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.95, y: 1.85, w: W - 8.95 - M, h: 4.25, rectRadius: 0.12,
    fill: { color: "15494A" }, line: { color: "15494A", width: 0 } });
  s.addText([
    { text: "Langkah berikutnya", options: { bold: true, color: C.accent2, breakLine: true, fontSize: 17 } },
    { text: "Erlangga: ambil 16 batch, jalankan GroupKFold per batch", options: { bullet: true, breakLine: true } },
    { text: "Michael: fingerprint + domain shift scaffold", options: { bullet: true, breakLine: true } },
    { text: "Rijal: citra pilot → embedding → bandit AL", options: { bullet: true } },
  ], { isTextBox: true, x: 9.2, y: 2.05, w: W - 8.95 - M - 0.5, h: 3.9, fontSize: 14, color: C.background1, margin: 0,
    paraSpaceAfter: 8 });
  s.addNotes("Empat pesan penutup. Validasi menentukan. Boosting adalah baseline yang kuat. Bandit menghemat label ketika masalahnya memang kekurangan label. Ketika masalahnya pergeseran domain, yang dibutuhkan adalah batch baru. Terima kasih, kami siap menjawab pertanyaan.");
}

// ================================================================== 17. references
{
  const s = content("Referensi", "Diskusi");
  const refs = [
    "Hsu, W.-N., & Lin, H.-T. (2015). Active learning by learning. Proc. AAAI, 29(1).",
    "Sorkun, M. C., Khetan, A., & Er, S. (2019). AqSolDB, a curated reference set of aqueous solubility and 2D descriptors. Scientific Data, 6, 143.",
    "Wijaya, D. R., Sarno, R., & Zulaika, E. (2018). Electronic nose dataset for beef quality monitoring in uncontrolled ambient conditions. Data in Brief, 21, 2414–2420.",
    "Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. J. Med. Chem., 39(15), 2887–2893.",
    "Chawla, N. V., dkk. (2002). SMOTE: Synthetic minority over-sampling technique. JAIR, 16, 321–357.",
    "Akiba, T., dkk. (2019). Optuna: A next-generation hyperparameter optimization framework. Proc. KDD.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proc. KDD.",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.",
    "Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47, 235–256.",
    "Materi kuliah Machine Learning II: Ensemble_Learning.pptx, reinforcement.pptx (github.com/jendralhxr/machinelearning).",
  ];
  s.addText(refs.map((t, i) => ({ text: t, options: { breakLine: i < refs.length - 1 } })),
    { isTextBox: true, x: M, y: 1.35, w: W - 2 * M, h: 5.4, fontSize: 13, color: C.text1, margin: 0, paraSpaceAfter: 6,
      valign: "top" });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  // optional: write THEME colours into the theme part (helper lives outside the repo)
  if (process.env.PPTX_SKILL) {
    const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
})();

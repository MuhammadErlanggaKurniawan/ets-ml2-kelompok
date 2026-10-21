// Build the ETS deck from deck/data.json:  node deck/build.js
// 13 slides for a 15-minute talk, ordered by the lecturer's rubric. Each content slide names its speaker.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const OUT = path.join(__dirname, "ETS_ML2_Presentasi.pptx");
const TITLE = "Prediksi Kesegaran Daging dan Kelarutan Senyawa dengan Machine Learning";

const THEME = {
  name: "ETS ML2",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16262B", lt1: "FFFFFF", dk2: "0F3D3E", lt2: "EEF4F2",
    accent1: "0F5257", accent2: "E3A33B", accent3: "9AA5A6", accent4: "C4553B",
    accent5: "6C8F7D", accent6: "2E6F95", hlink: "2E6F95", folHlink: "6C8F7D",
  },
};
const H = THEME.colors; // hex values for options that need hex

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = TITLE;
pres.author = "Kelompok ETS ML II";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const W = 13.333, M = 0.6;
const fmt = (x, n = 3) => Number(x).toFixed(n).replace(".", ",");
const num = (x) => Number(x).toLocaleString("id-ID");
const pct = (x) => `${Math.round(x * 100)}%`;
const GREY = "56676A", LIGHT = "CADBD7";

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "DARK",
  background: { color: H.dk2 },
  objects: [],
  slideNumber: { x: W - 1.0, y: 7.0, w: 0.5, h: 0.3, color: LIGHT, fontSize: 10, fontFace: "Calibri" },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: H.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M - 2.3, h: 0.85,
        fontFace: "Cambria", fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 },
      text: "" } },
    { text: { text: "ETS Machine Learning II · Prediksi Kesegaran Daging & Kelarutan Senyawa",
        options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: "7A8A8C", margin: 0 } } },
  ],
  slideNumber: { x: W - 1.0, y: 7.0, w: 0.5, h: 0.3, color: "7A8A8C", fontSize: 10, fontFace: "Calibri" },
});

function content(title, section, speaker) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (speaker) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: W - M - 2.0, y: 0.55, w: 2.0, h: 0.45, rectRadius: 0.22,
      fill: { color: C.background2 }, line: { color: C.background2, width: 0 } });
    s.addText(`Pembicara: ${speaker}`, { isTextBox: true, x: W - M - 2.0, y: 0.55, w: 2.0, h: 0.45, align: "center",
      valign: "middle", fontSize: 12, bold: true, color: C.accent1, margin: 0 });
  }
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
function label(s, t, x, y, w, color = C.accent1) {
  s.addText(t.toUpperCase(), { isTextBox: true, x, y, w, h: 0.3, fontSize: 12, bold: true, color, charSpacing: 2,
    margin: 0 });
}
function stat(s, x, y, w, value, caption, color = C.accent1) {
  s.addText(value, { isTextBox: true, x, y, w, h: 0.95, fontFace: "Cambria", fontSize: 54, bold: true, color,
    margin: 0, valign: "bottom" });
  s.addText(caption, { isTextBox: true, x, y: y + 1.0, w, h: 0.6, fontSize: 14, color: GREY, margin: 0 });
}
function bar(s, runs, y, h = 0.8, fill = C.text2) {
  card(s, M, y, W - 2 * M, h, fill);
  s.addText(runs, { isTextBox: true, x: M + 0.35, y, w: W - 2 * M - 0.7, h, fontSize: 16, valign: "middle",
    color: C.background1, margin: 0 });
}
// Charts are PNGs from scripts/make_figures.py: Google Slides drops native chart category labels.
const FIG = path.join(__dirname, "figures");
function img(s, name, x, y, w, h) {
  const file = path.join(FIG, `${name}.png`);
  const buf = fs.readFileSync(file);
  const pw = buf.readUInt32BE(16), ph = buf.readUInt32BE(20); // PNG IHDR
  const scale = Math.min(w / pw, h / ph);
  const iw = pw * scale, ih = ph * scale;
  s.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih });
}

// ------------------------------------------------------------------ data helpers
const byModel = (arr) => Object.fromEntries(arr.map((r) => [r.model, r]));
const enB = byModel(D["enose_models_binary"]);
const sol = byModel(D.sol_models);
const split = (m, l) => D.enose_split.find((r) => r.model === m && r.label === l);
const solSplit = (sp) => D.sol_split.find((r) => r.model === "LightGBM" && r.split === sp && r.features === "RDKit-2D");
const aug = (m, v) => D.enose_augment.find((r) => r.model === m && r.variant === v).f1;
const imb = (v) => D.enose_imbalance.find((r) => r.model === "LightGBM" && r.variant === v);
const alS = D.al_solubility, alE = D.al_enose;
const need = alS.labels_to_match_random;
const SAVE = 1 - need["bandit-ucb"] / need.random;
const sc = Object.fromEntries(D.series_curve.map((r) => [r.label, r.f1]));
const leakRandom = split("LightGBM", "random/engineered").f1, leakLoso = split("LightGBM", "loso/engineered").f1;
const bestSingleSol = ["Ridge", "KNN", "SVR-RBF", "DecisionTree"].map((m) => sol[m]).sort((a, b) => a.rmse - b.rmse)[0];
const rows3 = 3 * 2160;

// ================================================================== 1. title (Erlangga)
pres.addSection({ title: "Erlangga: latar belakang & data" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Erlangga: latar belakang & data" });
  label(s, "Tugas ETS · Machine Learning II", M, 0.95, 8, C.accent2);
  s.addText([
    { text: "Prediksi Kesegaran Daging dan", options: { breakLine: true } },
    { text: "Kelarutan Senyawa", options: { breakLine: true } },
    { text: "dengan Machine Learning" },
  ], { isTextBox: true, x: M, y: 1.35, w: 11.6, h: 2.3, fontFace: "Cambria", fontSize: 40, bold: true,
    color: C.background1, margin: 0, valign: "top" });
  s.addText("Satu pipeline dari akuisisi data sampai model: augmentasi, imbalance, ensemble, dan reinforcement learning",
    { isTextBox: true, x: M, y: 3.8, w: 10.5, h: 0.8, fontSize: 19, color: LIGHT, margin: 0 });
  s.addText([
    { text: "Muhammad Erlangga Kurniawan · 23083010045", options: { breakLine: true } },
    { text: "Michael Valencio Erasmo Bagas Ganendra · 23083010042", options: { breakLine: true } },
    { text: "Rijal Apuila Khufi · 23083010033" },
  ], { isTextBox: true, x: M, y: 4.95, w: 8, h: 1.2, fontSize: 15, color: C.background1, margin: 0,
    paraSpaceAfter: 4 });
  s.addText("Program Studi Sains Data · UPN \"Veteran\" Jawa Timur · Oktober 2026",
    { isTextBox: true, x: M, y: 6.55, w: 11, h: 0.4, fontSize: 12, color: "9FB8B3", margin: 0 });
  s.addNotes(`[Erlangga] Selamat pagi. Kami kelompok Erlangga, Michael, dan Rijal. Judul proyek kami: ${TITLE}. Pembagiannya: saya membuka dengan latar belakang, alur kerja, dan data. Michael menjelaskan imbalance, augmentasi, dan ensemble. Rijal menjelaskan reinforcement learning, diskusi, dan kesimpulan.`);
}

// ================================================================== 2. background
{
  const s = content("Kenapa topik ini?", "Erlangga: latar belakang & data", "Erlangga");
  const cols = [
    ["Erlangga", "Kesegaran makanan dari sensor gas", "dinilai panelis, menunggu sampai 48 jam",
     "cek kesegaran cukup dengan sensor murah"],
    ["Michael", "Kelarutan senyawa obat dalam air", "harus diukur di laboratorium",
     "saring senyawa sebelum diuji di lab"],
    ["Rijal", "Mutu kuning telur puyuh dari foto", "telurnya harus dipecah",
     "nilai mutu tanpa merusak telur"],
  ];
  const cw = 3.85, gap = 0.3, y0 = 1.45, ch = 3.85;
  cols.forEach((c, i) => {
    const x = M + i * (cw + gap);
    card(s, x, y0, cw, ch);
    label(s, `Skripsi ${c[0]}`, x + 0.3, y0 + 0.25, cw - 0.6);
    s.addText(c[1], { isTextBox: true, x: x + 0.3, y: y0 + 0.6, w: cw - 0.6, h: 0.85, fontFace: "Cambria",
      fontSize: 20, bold: true, color: C.text2, margin: 0, valign: "top" });
    [["Label (jawaban benar) didapat dari", c[2], C.accent4], ["Manfaat kalau ada model", c[3], C.accent1]]
      .forEach((r, j) => {
        const yy = y0 + 1.6 + j * 1.05;
        s.addText(r[0], { isTextBox: true, x: x + 0.3, y: yy, w: cw - 0.6, h: 0.28, fontSize: 12, color: GREY,
          margin: 0 });
        s.addText(r[1], { isTextBox: true, x: x + 0.3, y: yy + 0.3, w: cw - 0.6, h: 0.65, fontSize: 15, bold: true,
          color: r[2], margin: 0, valign: "top" });
      });
  });
  bar(s, [
    { text: "Masalah yang sama: ", options: { bold: true, color: C.accent2 } },
    { text: "data berlabel lama dan mahal didapat.  Tujuan: model yang akurat, diuji secara jujur, dan hemat label." },
  ], 5.55, 1.0);
  s.addNotes("[Erlangga] Proyek ini berangkat dari tiga topik skripsi kami. Saya meneliti kesegaran makanan dengan sensor gas, Michael meneliti kelarutan senyawa obat, Rijal meneliti mutu kuning telur puyuh dari foto. Ternyata masalahnya sama: untuk mendapat jawaban yang benar, atau label, kita harus menunggu panelis sampai 48 jam, menguji di laboratorium, atau memecahkan telurnya. Jadi tujuan kami: model yang akurat, diuji secara jujur, dan sebisa mungkin hemat label. Manfaatnya: kesegaran bisa dicek dengan sensor murah, dan senyawa bisa disaring sebelum diuji di laboratorium.");
}

// ================================================================== 3. pipeline + data
{
  const s = content("Alur kerja dan data", "Erlangga: latar belakang & data", "Erlangga");
  const steps = [
    ["Akuisisi", "Unduh 2 dataset publik, cek keaslian file (SHA-256)", null],
    ["Pengolahan", "Bersihkan data, ubah jadi fitur angka", "Augmentasi · Imbalance"],
    ["Pemodelan", "Bandingkan model tunggal vs gabungan", "Ensemble"],
    ["Evaluasi jujur", "Uji pada data yang benar-benar baru", null],
    ["Tambah label", "Agen memilih data yang perlu dilabeli", "Reinforcement"],
  ];
  const gap = 0.35, bw = (W - 2 * M - 4 * gap) / 5, y = 1.4, bh = 2.3;
  steps.forEach((st, i) => {
    const x = M + i * (bw + gap);
    card(s, x, y, bw, bh);
    chip(s, x + 0.2, y + 0.2, String(i + 1), C.accent1, C.background1, 0.45);
    s.addText(st[0], { isTextBox: true, x: x + 0.75, y: y + 0.2, w: bw - 0.9, h: 0.45, fontSize: 16, bold: true,
      color: C.text2, margin: 0, valign: "middle" });
    s.addText(st[1], { isTextBox: true, x: x + 0.2, y: y + 0.8, w: bw - 0.4, h: 0.9, fontSize: 13, color: C.text1,
      margin: 0 });
    if (st[2]) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.15, y: y + bh - 0.55, w: bw - 0.3, h: 0.4, rectRadius: 0.2,
        fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 } });
      s.addText(st[2], { isTextBox: true, x: x + 0.15, y: y + bh - 0.55, w: bw - 0.3, h: 0.4, align: "center",
        valign: "middle", fontSize: 11.5, bold: true, color: C.text2, margin: 0 });
    }
    if (i < 4) s.addShape(pres.shapes.LINE, { x: x + bw + 0.05, y: y + bh / 2, w: gap - 0.1, h: 0,
      line: { color: H.accent1, width: 2, endArrowType: "triangle" } });
  });
  const dw = (W - 2 * M - 0.3) / 2, dy = 3.95, dh = 2.15;
  const ds = [
    ["E-nose daging sapi", "Wijaya dkk. (2018), Data in Brief",
     `${num(D.enose_rows)} baris = 5 potongan daging × 2.160 menit, 9 sensor gas + kelembapan`,
     "Label: segar / tidak segar (dari 4 kelas mutu)"],
    ["AqSolDB", "Sorkun dkk. (2019), Scientific Data",
     `${num(D.sol_meta.n)} senyawa, ditulis sebagai teks SMILES`,
     "Label: kelarutan LogS (angka, jadi regresi)"],
  ];
  ds.forEach((d, i) => {
    const x = M + i * (dw + 0.3);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: dy, w: dw, h: dh, rectRadius: 0.12,
      fill: { color: C.background1 }, line: { color: C.accent1, width: 1.25 } });
    s.addText(d[0], { isTextBox: true, x: x + 0.3, y: dy + 0.2, w: dw - 0.6, h: 0.4, fontFace: "Cambria", fontSize: 19,
      bold: true, color: C.text2, margin: 0 });
    s.addText(d[1], { isTextBox: true, x: x + 0.3, y: dy + 0.62, w: dw - 0.6, h: 0.3, fontSize: 12, color: GREY,
      margin: 0 });
    s.addText([{ text: d[2], options: { breakLine: true } }, { text: d[3], options: { bold: true, color: C.accent1 } }],
      { isTextBox: true, x: x + 0.3, y: dy + 1.0, w: dw - 0.6, h: 1.05, fontSize: 14, color: C.text1, margin: 0,
        paraSpaceAfter: 4, valign: "top" });
  });
  text(s, "Data skripsi Erlangga masih pilot (1 batch) dan foto telur Rijal belum diambil, jadi kami memakai dataset publik yang strukturnya mirip.",
    { x: M, y: 6.3, w: W - 2 * M, h: 0.5, fontSize: 13, color: GREY });
  s.addNotes(`[Erlangga] Ini alur kerja kami, lima tahap. Akuisisi: kami mengunduh dua dataset publik dan mengecek keaslian filenya. Pengolahan: data dibersihkan dan diubah menjadi fitur angka; di sini masuk augmentasi dan penanganan imbalance. Pemodelan: kami membandingkan model tunggal dengan ensemble. Evaluasi: model diuji pada data yang benar-benar baru. Terakhir, agen reinforcement learning memilih data mana yang perlu dilabeli. Jadi ada empat aspek ML II yang kami kerjakan. Datanya dua: e-nose daging sapi, ${num(D.enose_rows)} baris dari 5 potongan daging yang diukur 2.160 menit dengan 9 sensor gas; dan AqSolDB, ${num(D.sol_meta.n)} senyawa dengan nilai kelarutannya. Data skripsi saya baru pilot dan foto telur Rijal belum diambil, jadi kami memakai dataset publik yang strukturnya mirip.`);
}

// ================================================================== 4. wrangling
{
  const s = content("Pengolahan data (wrangling)", "Erlangga: latar belakang & data", "Erlangga");
  const colsW = [
    ["E-nose daging sapi", [
      ["Tiap sensor punya nilai awal berbeda", "Dibagi rata-rata 10 menit pertama (rasio, mirip Rs/R0)"],
      ["Sinyal naik-turun, berisik", "Dihaluskan: rata-rata 15 menit terakhir"],
      ["Arah perubahan gas itu penting", "Kemiringan (slope) 15 menit, hanya dari data masa lalu"],
    ]],
    ["AqSolDB", [
      ["Molekul berupa teks (SMILES)", `Diubah jadi ${D.sol_meta.n_rdkit_features} fitur angka dengan RDKit`],
      ["Ada nilai kosong dan ekstrem", "Kolom yang banyak kosong dibuang, ekstrem dipotong, sisanya diisi median"],
      ["Banyak molekul bersaudara", "Dikelompokkan per kerangka (scaffold) agar uji adil"],
    ]],
  ];
  const cw = (W - 2 * M - 0.3) / 2, y0 = 1.4, ch = 4.4;
  colsW.forEach((c, i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, y0, cw, ch);
    s.addText(c[0], { isTextBox: true, x: x + 0.3, y: y0 + 0.2, w: cw - 0.6, h: 0.45, fontFace: "Cambria", fontSize: 19,
      bold: true, color: C.text2, margin: 0 });
    s.addText("MASALAH", { isTextBox: true, x: x + 0.3, y: y0 + 0.75, w: 2.2, h: 0.25, fontSize: 11, bold: true,
      color: GREY, charSpacing: 2, margin: 0 });
    s.addText("YANG KAMI LAKUKAN", { isTextBox: true, x: x + 2.85, y: y0 + 0.75, w: 2.7, h: 0.25, fontSize: 11,
      bold: true, color: C.accent1, charSpacing: 2, margin: 0 });
    c[1].forEach((r, j) => {
      const yy = y0 + 1.1 + j * 1.07;
      s.addText(r[0], { isTextBox: true, x: x + 0.3, y: yy, w: 2.2, h: 0.9, fontSize: 14, color: C.text1, margin: 0,
        valign: "middle" });
      s.addText("→", { isTextBox: true, x: x + 2.5, y: yy, w: 0.35, h: 0.9, fontSize: 18, color: C.accent2, bold: true,
        margin: 0, valign: "middle", align: "center" });
      s.addText(r[1], { isTextBox: true, x: x + 2.85, y: yy, w: cw - 3.15, h: 0.9, fontSize: 14, bold: true,
        color: C.text2, margin: 0, valign: "middle" });
    });
  });
  bar(s, [
    { text: "Prinsip: ", options: { bold: true, color: C.accent2 } },
    { text: "semua langkah yang \"belajar\" dari data (median, skala, augmentasi, tuning) dihitung dari data latih saja." },
  ], 6.0, 0.75);
  s.addNotes(`[Erlangga] Pengolahan data. Pada e-nose, setiap sensor punya nilai awal berbeda, jadi kami membaginya dengan rata-rata 10 menit pertama. Sinyal yang berisik dihaluskan dengan rata-rata 15 menit, dan kami menambah kemiringan untuk melihat arah perubahan gas. Semua fitur hanya memakai data masa lalu, tidak mengintip masa depan. Pada AqSolDB, molekul yang berbentuk teks SMILES diubah menjadi ${D.sol_meta.n_rdkit_features} fitur angka dengan RDKit. Kolom yang banyak kosong dibuang, nilai ekstrem dipotong, sisanya diisi median. Molekul yang bersaudara dikelompokkan berdasarkan kerangkanya. Prinsipnya: semua yang belajar dari data hanya dihitung dari data latih.`);
}

// ================================================================== 5. leakage
{
  const s = content("Jebakan akurasi 99%", "Erlangga: latar belakang & data", "Erlangga");
  const a1 = solSplit("random"), a2 = solSplit("scaffold");
  const cw = (W - 2 * M - 0.33) / 2;
  [[M, "E-nose · skor F1 (↑ lebih baik)", fmt(leakRandom), "data dibagi acak", fmt(leakLoso),
    "diuji di potongan daging baru"],
   [M + cw + 0.33, "AqSolDB · error RMSE (↓ lebih baik)", fmt(a1.rmse), "data dibagi acak", fmt(a2.rmse),
    "diuji di kerangka molekul baru"]].forEach(([x, head, v1, c1, v2, c2]) => {
    card(s, x, 1.45, cw, 3.3);
    label(s, head, x + 0.35, 1.65, cw - 0.7);
    stat(s, x + 0.35, 2.1, 2.6, v1, c1, C.accent4);
    stat(s, x + 3.1, 2.1, 2.6, v2, c2, C.accent1);
    s.addText("terlalu bagus", { isTextBox: true, x: x + 0.35, y: 3.8, w: 2.6, h: 0.35, fontSize: 13, italic: true,
      color: C.accent4, margin: 0 });
    s.addText("angka jujur", { isTextBox: true, x: x + 3.1, y: 3.8, w: 2.6, h: 0.35, fontSize: 13, italic: true,
      color: C.accent1, margin: 0 });
  });
  text(s, [
    { text: "Kenapa? ", options: { bold: true } },
    { text: "Kalau dibagi acak, menit ke-100 masuk data latih dan menit ke-101 masuk data uji. Nilainya hampir sama, jadi model seperti ujian dengan soal yang sudah bocor." },
  ], { x: M, y: 5.0, w: W - 2 * M, h: 0.8, fontSize: 15 });
  bar(s, [
    { text: "Semua hasil berikutnya memakai uji jujur: ", options: { bold: true, color: C.accent2 } },
    { text: "Leave-One-Series-Out (e-nose) dan scaffold split (AqSolDB)." },
  ], 5.95, 0.75);
  s.addNotes(`[Erlangga] Ini temuan pertama dan paling penting. Dengan model yang sama, kalau data dibagi acak, F1-nya ${fmt(leakRandom)}, hampir sempurna. Tapi kalau diuji pada potongan daging yang belum pernah dilihat, hanya ${fmt(leakLoso)}. Penyebabnya kebocoran data: menit ke-100 ada di data latih, menit ke-101 ada di data uji, dan nilainya hampir sama. Seperti ujian yang soalnya sudah bocor. Di AqSolDB polanya sama: error naik dari ${fmt(a1.rmse)} ke ${fmt(a2.rmse)} ketika diuji pada kerangka molekul baru. Karena itu semua hasil berikutnya memakai cara uji yang jujur. Selanjutnya Michael.`);
}

// ================================================================== 6. imbalance + augmentation (Michael)
pres.addSection({ title: "Michael: imbalance, augmentasi, ensemble" });
{
  const s = content("Imbalance dan augmentasi", "Michael: imbalance, augmentasi, ensemble", "Michael");
  img(s, "s07_augment", M, 1.35, 6.5, 4.4);
  const x = 7.35, w = W - x - M;
  card(s, x, 1.35, w, 4.4);
  const c = D.enose_classes;
  const blocks = [
    ["Kenapa perlu?", [
      `Kelas tidak seimbang: "busuk" ${num(c.spoiled)} baris, "sangat segar" ${num(c.excellent)} baris.`,
      "Hanya 5 potongan daging, variasinya sedikit.",
    ]],
    ["Bagaimana?", [
      "Imbalance: class weight, SMOTE, undersampling.",
      "Augmentasi: jitter = tambah derau kecil; scaling = sinyal dikalikan, seolah sensor beda kepekaan. Hanya di data latih.",
    ]],
    ["Hasilnya", [
      `Scaling: Logistic Regression naik ${fmt(aug("LogReg", "none"))} → ${fmt(aug("LogReg", "scaling"))}, LightGBM turun ${fmt(aug("LightGBM", "none"))} → ${fmt(aug("LightGBM", "scaling"))}.`,
      `Imbalance: F1 hanya bergeser ±0,02; undersampling menaikkan recall "sangat segar" ${fmt(imb("class_weight=None").recall_excellent)} → ${fmt(imb("RandomUnder").recall_excellent)}.`,
    ]],
  ];
  const runs = [];
  blocks.forEach((b, i) => {
    runs.push({ text: b[0], options: { bold: true, color: C.accent1, fontSize: 14, breakLine: true } });
    b[1].forEach((t, j) => runs.push({ text: t, options: { bullet: true,
      breakLine: !(i === blocks.length - 1 && j === b[1].length - 1) } }));
  });
  s.addText(runs, { isTextBox: true, x: x + 0.3, y: 1.55, w: w - 0.6, h: 4.05, fontSize: 13, color: C.text1, margin: 0,
    paraSpaceAfter: 4, valign: "top" });
  bar(s, [
    { text: "Kesimpulan: ", options: { bold: true, color: C.accent2 } },
    { text: "augmentasi dan imbalance bukan obat mujarab; manfaatnya bergantung model dan harus dibuktikan dengan uji jujur." },
  ], 6.0, 0.75);
  s.addNotes(`[Michael] Terima kasih Erlangga. Saya mulai dari imbalance dan augmentasi. Kenapa perlu? Kelasnya tidak seimbang: kelas busuk ${num(c.spoiled)} baris, kelas sangat segar hanya ${num(c.excellent)}. Dan kita hanya punya 5 potongan daging, jadi variasinya sedikit. Untuk imbalance kami coba class weight, SMOTE, dan undersampling. Untuk augmentasi kami pakai jitter, yaitu menambah derau kecil, dan scaling, yaitu mengalikan sinyal seolah sensornya berbeda kepekaan. Augmentasi hanya diberikan ke data latih. Hasilnya: scaling membantu Logistic Regression dari ${fmt(aug("LogReg", "none"))} ke ${fmt(aug("LogReg", "scaling"))}, tetapi justru menurunkan LightGBM. Imbalance hanya menggeser F1 sekitar 0,02, tetapi undersampling menaikkan recall kelas sangat segar dari ${fmt(imb("class_weight=None").recall_excellent)} ke ${fmt(imb("RandomUnder").recall_excellent)}. Jadi teknik ini bukan obat mujarab; manfaatnya harus dibuktikan.`);
}

// ================================================================== 7. ensemble
{
  const s = content("Ensemble: gabungan model menang", "Michael: imbalance, augmentasi, ensemble", "Michael");
  text(s, [
    { text: "Ensemble = gabungan banyak model:  ", options: { bold: true } },
    { text: "Random Forest (bagging) · XGBoost, LightGBM (boosting) · Stacking (model penggabung)" },
  ], { x: M, y: 1.3, w: W - 2 * M, h: 0.4, fontSize: 15 });
  const cw = (W - 2 * M - 0.33) / 2;
  img(s, "s08_enose_ensemble", M, 1.8, cw, 3.9);
  img(s, "s09_sol_ensemble", M + cw + 0.33, 1.8, cw, 3.9);
  const ratio = Math.round(enB.DecisionTree.sd / enB.XGBoost.sd);
  const drop = Math.round((1 - sol.Stacking.rmse / bestSingleSol.rmse) * 100);
  [[M, [
    { text: `XGBoost ${fmt(enB.XGBoost.f1)}`, options: { bold: true, color: C.accent2 } },
    { text: ` vs model tunggal terbaik (Decision Tree) ${fmt(enB.DecisionTree.f1)}; naik-turunnya ${ratio}× lebih kecil antar potongan daging` },
  ]], [M + cw + 0.33, [
    { text: `Stacking ${fmt(sol.Stacking.rmse)}`, options: { bold: true, color: C.accent2 } },
    { text: ` vs model tunggal terbaik (SVR) ${fmt(bestSingleSol.rmse)}; error turun ±${drop}%` },
  ]]].forEach(([x, runs]) => {
    card(s, x, 5.85, cw, 0.9, C.text2);
    s.addText(runs, { isTextBox: true, x: x + 0.3, y: 5.85, w: cw - 0.6, h: 0.9, fontSize: 14, color: C.background1,
      valign: "middle", margin: 0 });
  });
  s.addNotes(`[Michael] Berikutnya ensemble, yaitu menggabungkan banyak model. Random Forest menggabungkan banyak pohon yang dilatih paralel. XGBoost dan LightGBM membangun pohon bertahap, setiap pohon memperbaiki kesalahan pohon sebelumnya. Stacking memakai satu model untuk menggabungkan prediksi model lain. Di e-nose, XGBoost mencapai F1 ${fmt(enB.XGBoost.f1)}, jauh di atas model tunggal terbaik, Decision Tree, ${fmt(enB.DecisionTree.f1)}, dan naik-turunnya antar potongan daging ${ratio} kali lebih kecil. Di kelarutan, stacking memberi error terendah ${fmt(sol.Stacking.rmse)} dibanding model tunggal terbaik, SVR, ${fmt(bestSingleSol.rmse)}, turun sekitar ${drop} persen. Jadi ensemble lebih baik di kedua data.`);
}

// ================================================================== 8. SHAP
{
  const s = content("Apa yang dipelajari model? (SHAP)", "Michael: imbalance, augmentasi, ensemble", "Michael");
  const cw = (W - 2 * M - 0.33) / 2;
  img(s, "s14_shap_sol", M, 1.35, cw, 4.3);
  img(s, "s14_shap_enose", M + cw + 0.33, 1.35, cw, 4.3);
  text(s, [
    { text: "Kelarutan: ", options: { bold: true, color: C.accent1 } },
    { text: "MolLogP (seberapa \"suka lemak\" molekulnya) paling berpengaruh. Makin suka lemak, makin sulit larut di air: cocok dengan teori kimia.", options: { breakLine: true } },
    { text: "E-nose: ", options: { bold: true, color: C.accent1 } },
    { text: `sensor MQ4 (metana) dan MQ8 (hidrogen) paling menentukan; MQ135 yang dipakai skripsi Erlangga kecil (${fmt(D.enose_shap_sensors.MQ135, 2)}). Ini masukan untuk pemilihan sensor.` },
  ], { x: M, y: 5.8, w: W - 2 * M, h: 1.05, fontSize: 14, paraSpaceAfter: 4 });
  s.addNotes(`[Michael] Terakhir dari saya: apa yang sebenarnya dipelajari model. Kami memakai SHAP untuk melihat fitur yang paling berpengaruh. Di kelarutan, MolLogP paling berpengaruh. MolLogP mengukur seberapa suka molekul pada lemak; makin suka lemak, makin sulit larut di air. Ini cocok dengan teori kimia, jadi modelnya masuk akal. Di e-nose, sensor MQ4 untuk metana dan MQ8 untuk hidrogen paling menentukan, sedangkan MQ135 yang dipakai di skripsi Erlangga pengaruhnya kecil. Ini jadi masukan untuk pemilihan sensor. Selanjutnya Rijal.`);
}

// ================================================================== 9. reinforcement (Rijal)
pres.addSection({ title: "Rijal: reinforcement, diskusi, kesimpulan" });
{
  const s = content("Reinforcement: agen memilih data untuk dilabeli", "Rijal: reinforcement, diskusi, kesimpulan", "Rijal");
  const lw = 5.0;
  card(s, M, 1.35, lw, 4.45);
  label(s, "Diulang 20 putaran", M + 0.3, 1.55, lw - 0.6);
  const steps = [
    ["Agen memilih strategi", "acak · paling membuat model ragu · paling berbeda"],
    ["Minta label 100 data", "diambil dari data yang belum berlabel"],
    ["Latih ulang model", "model makin akurat"],
    ["Hitung reward", "error turun banyak = strategi bagus, makin sering dipilih"],
  ];
  steps.forEach((st, i) => {
    const y = 2.0 + i * 0.92;
    chip(s, M + 0.3, y + 0.05, String(i + 1), i === 3 ? C.accent2 : C.accent1, i === 3 ? C.text2 : C.background1, 0.48);
    s.addText(st[0], { isTextBox: true, x: M + 0.95, y, w: lw - 1.2, h: 0.34, fontSize: 15, bold: true, color: C.text2,
      margin: 0 });
    s.addText(st[1], { isTextBox: true, x: M + 0.95, y: y + 0.34, w: lw - 1.2, h: 0.5, fontSize: 12.5, color: C.text1,
      margin: 0, valign: "top" });
  });
  img(s, "al_hemat", M + lw + 0.3, 1.35, W - 2 * M - lw - 0.3, 4.45);
  bar(s, [
    { text: `Hemat ±${pct(SAVE)} label: `, options: { bold: true, color: C.accent2 } },
    { text: `agen bandit cukup ${num(need["bandit-ucb"])} label untuk menyamai hasil ${num(need.random)} label yang dipilih acak.` },
  ], 5.95, 0.8);
  s.addNotes(`[Rijal] Terima kasih Michael. Saya menjelaskan bagian reinforcement learning. Pertanyaannya: kalau label mahal, data mana yang sebaiknya dilabeli dulu? Kami membuat agen yang belajar menjawabnya. Setiap putaran, agen memilih satu strategi: acak, data yang paling membuat model ragu, atau data yang paling berbeda. Lalu 100 data diberi label, model dilatih ulang, dan agen mendapat reward sebesar penurunan error. Strategi yang sering memberi reward besar akan makin sering dipilih. Ini disebut multi-armed bandit, bentuk paling sederhana dari reinforcement learning, mirip memilih mesin slot yang paling sering menang; algoritmanya UCB1. Hasilnya di grafik: agen bandit cukup ${num(need["bandit-ucb"])} label untuk menyamai hasil ${num(need.random)} label yang dipilih acak. Hemat sekitar ${pct(SAVE)} label. Untuk penelitian saya, ini berarti lebih sedikit telur yang harus dipecah.`);
}

// ================================================================== 10. more data
{
  const s = content("Tambah data: yang penting variasinya", "Rijal: reinforcement, diskusi, kesimpulan", "Rijal");
  img(s, "s13_series_curve", M, 1.35, 5.8, 4.5);
  const x = 6.75, w = W - x - M;
  const items = [
    [`${fmt(sc["1 seri"], 2)} → ${fmt(sc["4 seri"], 2)}`, "F1 naik saat potongan daging (seri) ditambah dari 1 ke 4"],
    [`${fmt(sc["4 seri · 160 baris"], 2)} > ${fmt(sc["3 seri"], 2)}`,
      `160 baris dari 4 seri mengalahkan ${num(rows3)} baris dari 3 seri`],
    [`acak ${fmt(alE.final.random.mean, 2)}`, "Di e-nose, pemilihan acak tetap terbaik: masalahnya beda kondisi antar seri, bukan kurang label"],
  ];
  items.forEach((it, i) => {
    const y = 1.35 + i * 1.53;
    card(s, x, y, w, 1.38, i === 2 ? C.text2 : C.background2);
    s.addText(it[0], { isTextBox: true, x: x + 0.3, y: y + 0.12, w: 2.4, h: 1.1, fontFace: "Cambria", fontSize: 30,
      bold: true, color: i === 2 ? C.accent2 : C.accent1, margin: 0, valign: "middle", fit: "shrink" });
    s.addText(it[1], { isTextBox: true, x: x + 2.8, y: y + 0.12, w: w - 3.05, h: 1.14, fontSize: 14,
      color: i === 2 ? C.background1 : C.text1, margin: 0, valign: "middle" });
  });
  bar(s, [
    { text: "Implikasi untuk skripsi Erlangga: ", options: { bold: true, color: C.accent2 } },
    { text: "perbanyak batch pengambilan data, bukan memperpanjang bacaan per batch." },
  ], 6.0, 0.75);
  s.addNotes(`[Rijal] Rubrik reinforcement bertanya: apakah tambahan data dan iterasi meningkatkan performa? Di e-nose jawabannya tergantung jenis datanya. Menambah potongan daging dari 1 ke 4 menaikkan F1 dari ${fmt(sc["1 seri"], 2)} ke ${fmt(sc["4 seri"], 2)}. Tetapi hanya 160 baris dari 4 potongan daging sudah memberi ${fmt(sc["4 seri · 160 baris"], 2)}, lebih baik daripada ${num(rows3)} baris dari 3 potongan daging, yaitu ${fmt(sc["3 seri"], 2)}. Artinya variasi lebih penting daripada jumlah baris. Karena itu active learning di e-nose tidak mengalahkan pemilihan acak: masalahnya perbedaan kondisi antar potongan daging, bukan kurang label. Untuk skripsi Erlangga, implikasinya: perbanyak batch pengambilan data.`);
}

// ================================================================== 11. discussion
{
  const s = content("Analisis dan diskusi", "Rijal: reinforcement, diskusi, kesimpulan", "Rijal");
  const g = [
    ["Batasan", ["E-nose hanya 5 potongan daging, hasil naik-turun", "Data e-nose = daging mentah, bukan data skripsi Erlangga", "Active learning masih simulasi (label sudah ada)"]],
    ["Trade-off", ["Augmentasi bantu model sederhana, rugikan boosting", `Bobot imbalance: kelas ekstrem membaik, error total naik (${fmt(sol["LightGBM-Optuna"].rmse)} → ${fmt(D.sol_weighted_overall.rmse)})`, "Stacking paling akurat, tapi paling lama dilatih"]],
    ["Sumber galat", ["Beda kondisi ruang antar potongan daging", "Senyawa uji punya kerangka yang belum pernah dilihat", `Tuning Optuna terlalu pas di validasi: ${fmt(D.sol_meta.optuna_best_cv_rmse)} vs uji ${fmt(sol["LightGBM-Optuna"].rmse)}`]],
    ["Pengembangan", ["Jalankan pipeline di 16 batch data Erlangga", "Active learning untuk foto telur: lebih sedikit telur dipecah", "Uji sensor MQ4/MQ8 tambahan"]],
  ];
  const cw = (W - 2 * M - 0.3) / 2, ch = 2.55;
  g.forEach((b, i) => {
    const x = M + (i % 2) * (cw + 0.3), y = 1.35 + Math.floor(i / 2) * (ch + 0.25);
    card(s, x, y, cw, ch);
    s.addText(b[0], { isTextBox: true, x: x + 0.3, y: y + 0.18, w: cw - 0.6, h: 0.4, fontSize: 18, bold: true,
      color: C.accent1, margin: 0 });
    s.addText(b[1].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < b[1].length - 1 } })),
      { isTextBox: true, x: x + 0.3, y: y + 0.68, w: cw - 0.6, h: ch - 0.85, fontSize: 15, color: C.text1, margin: 0,
        paraSpaceAfter: 5, valign: "top" });
  });
  s.addNotes(`[Rijal] Analisis dan diskusi. Batasan: e-nose hanya 5 potongan daging sehingga hasilnya naik-turun; data ini daging mentah, bukan data skripsi Erlangga; dan active learning masih simulasi karena labelnya sudah ada. Trade-off: augmentasi membantu model sederhana tetapi merugikan boosting; bobot imbalance memperbaiki kelas ekstrem tetapi error total sedikit naik; stacking paling akurat tetapi paling lama dilatih. Sumber galat: beda kondisi ruang antar potongan daging, senyawa uji yang kerangkanya belum pernah dilihat, dan tuning Optuna yang terlalu pas di validasi: ${fmt(D.sol_meta.optuna_best_cv_rmse)} di validasi, ${fmt(sol["LightGBM-Optuna"].rmse)} di uji. Pengembangan: menjalankan pipeline ini pada 16 batch data Erlangga, active learning untuk foto telur supaya lebih sedikit telur yang dipecah, dan menguji sensor MQ4 dan MQ8.`);
}

// ================================================================== 12. conclusion
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Rijal: reinforcement, diskusi, kesimpulan" });
  s.addText("Kesimpulan", { isTextBox: true, x: M, y: 0.6, w: 8, h: 0.9, fontFace: "Cambria", fontSize: 40, bold: true,
    color: C.background1, margin: 0 });
  const k = [
    ["Uji yang jujur lebih penting dari model canggih", `${fmt(leakRandom)} itu palsu, ${fmt(leakLoso)} yang jujur`],
    ["Ensemble menang di kedua data", `XGBoost ${fmt(enB.XGBoost.f1, 2)} vs Decision Tree ${fmt(enB.DecisionTree.f1, 2)} · Stacking error terendah ${fmt(sol.Stacking.rmse, 2)}`],
    ["Agen bandit menghemat label", `±${pct(SAVE)} lebih sedikit label di data kelarutan`],
    ["Beda kondisi antar batch? Tambah batch", "bukan tambah baris dari batch yang sama"],
  ];
  k.forEach((r, i) => {
    const y = 1.85 + i * 1.1;
    chip(s, M, y + 0.05, String(i + 1), C.accent2, C.text2, 0.55);
    s.addText(r[0], { isTextBox: true, x: M + 0.85, y, w: 7.0, h: 0.45, fontSize: 19, bold: true, color: C.background1,
      margin: 0 });
    s.addText(r[1], { isTextBox: true, x: M + 0.85, y: y + 0.45, w: 7.0, h: 0.4, fontSize: 14, color: LIGHT, margin: 0 });
  });
  const bx = 8.6, bw2 = W - bx - M;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: 1.85, w: bw2, h: 4.25, rectRadius: 0.12,
    fill: { color: "15494A" }, line: { color: "15494A", width: 0 } });
  s.addText([
    { text: "Terima kasih", options: { bold: true, color: C.accent2, breakLine: true, fontSize: 28, fontFace: "Cambria" } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Kode (GitHub):", options: { breakLine: true, color: LIGHT } },
    { text: "github.com/MuhammadErlanggaKurniawan", options: { bold: true, breakLine: true } },
    { text: "/ets-ml2-kelompok", options: { bold: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Notebook Colab (13 bagian):", options: { breakLine: true, color: LIGHT } },
    { text: "ETS_ML2_Pipeline_Lengkap.ipynb", options: { bold: true } },
  ], { isTextBox: true, x: bx + 0.3, y: 2.1, w: bw2 - 0.6, h: 3.8, fontSize: 13, color: C.background1, margin: 0,
    paraSpaceAfter: 4, valign: "top" });
  s.addNotes(`[Rijal] Kesimpulan kami ada empat. Pertama, cara uji yang jujur lebih menentukan daripada model yang canggih: ${fmt(leakRandom)} itu palsu, ${fmt(leakLoso)} yang jujur. Kedua, ensemble menang di kedua data. Ketiga, agen bandit menghemat sekitar ${pct(SAVE)} label di data kelarutan. Keempat, kalau masalahnya perbedaan kondisi antar batch, solusinya menambah batch, bukan menambah baris. Semua kode ada di GitHub dan bisa dijalankan di Colab. Terima kasih, kami siap menerima pertanyaan.`);
}

// ================================================================== 13. references (backup)
{
  const s = content("Lampiran: referensi", "Rijal: reinforcement, diskusi, kesimpulan");
  const refs = [
    "Wijaya, D. R., Sarno, R., & Zulaika, E. (2018). Electronic nose dataset for beef quality monitoring in uncontrolled ambient conditions. Data in Brief, 21, 2414–2420.",
    "Sorkun, M. C., Khetan, A., & Er, S. (2019). AqSolDB, a curated reference set of aqueous solubility and 2D descriptors. Scientific Data, 6, 143.",
    "Hsu, W.-N., & Lin, H.-T. (2015). Active learning by learning. Proc. AAAI, 29(1).",
    "Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47, 235–256.",
    "Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. J. Med. Chem., 39(15), 2887–2893.",
    "Chawla, N. V., dkk. (2002). SMOTE: Synthetic minority over-sampling technique. JAIR, 16, 321–357.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proc. KDD.",
    "Akiba, T., dkk. (2019). Optuna: A next-generation hyperparameter optimization framework. Proc. KDD.",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.",
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

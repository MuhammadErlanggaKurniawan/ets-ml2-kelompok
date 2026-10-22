// Build the ETS deck from hasil/ringkasan.json and the notebook figures:  node deck/build.js
// 20 slides for a 15-minute talk, following the rubric order: latar belakang, data wrangling & augmentation,
// ensemble learning, reinforcement learning, hasil dan analisis, celah pengembangan, simpulan.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.join(__dirname, "..");
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "hasil", "ringkasan.json"), "utf8"));
const OUT = path.join(__dirname, "ETS_ML2_Kesegaran_Ayam.pptx");
const TITLE = "Klasifikasi Kesegaran Daging Ayam Berbasis Ensemble dan Reinforcement Learning";

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
const H = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = TITLE;
pres.author = "Kelompok ETS ML II";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const W = 13.333, M = 0.6, CW = W - 2 * M;
const GREY = "56676A", LIGHT = "CADBD7", ROW = "F6F9F8", LINE = "DCE5E2";
const fmt = (x, n = 3) => Number(x).toFixed(n).replace(".", ",").replace(/^-/, "−");
const pct = (x) => `${(100 * x).toFixed(1).replace(".", ",")}%`;
const num = (x) => Number(x).toLocaleString("id-ID");
const signed = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3).replace(".", ",");

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "DARK",
  background: { color: H.dk2 },
  objects: [],
  slideNumber: { x: W - 1.0, y: 7.0, w: 0.5, h: 0.3, color: LIGHT, fontSize: 10, fontFace: "Calibri" },
});
pres.defineSlideMaster({
  title: "PLAIN",
  background: { color: H.lt1 },
  objects: [],
  slideNumber: { x: W - 1.0, y: 7.0, w: 0.5, h: 0.3, color: "7A8A8C", fontSize: 10, fontFace: "Calibri" },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: H.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: CW - 2.3, h: 0.85,
        fontFace: "Cambria", fontSize: 26, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 },
      text: "" } },
    { text: { text: "ETS Machine Learning II · Klasifikasi Kesegaran Daging Ayam",
        options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: "7A8A8C", margin: 0 } } },
  ],
  slideNumber: { x: W - 1.0, y: 7.0, w: 0.5, h: 0.3, color: "7A8A8C", fontSize: 10, fontFace: "Calibri" },
});

function speaker(s, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: W - M - 2.0, y: 0.55, w: 2.0, h: 0.45, rectRadius: 0.22,
    fill: { color: H.lt2 }, line: { color: H.lt2, width: 0 } });
  s.addText(`Pembicara: ${name}`, { isTextBox: true, x: W - M - 2.0, y: 0.55, w: 2.0, h: 0.45, align: "center",
    valign: "middle", fontSize: 12, bold: true, color: H.accent1, margin: 0 });
}
function content(title, section, name) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (name) speaker(s, name);
  return s;
}
function card(s, x, y, w, h, fill = H.lt2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill },
    line: { color: fill, width: 0 } });
}
function chip(s, x, y, label, fill = H.accent1, color = H.lt1, d = 0.5) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 } });
  s.addText(label, { isTextBox: true, x, y, w: d, h: d, align: "center", valign: "middle", fontSize: 16,
    bold: true, color, margin: 0 });
}
function square(s, x, y, label, d = 0.62, fill = H.dk2) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 } });
  s.addText(label, { isTextBox: true, x, y, w: d, h: d, align: "center", valign: "middle", fontSize: 20,
    bold: true, color: H.lt1, margin: 0 });
}
function label(s, t, x, y, w, color = H.accent1) {
  s.addText(t.toUpperCase(), { isTextBox: true, x, y, w, h: 0.3, fontSize: 12, bold: true, color, charSpacing: 2,
    margin: 0 });
}
function stat(s, x, y, w, value, caption, color = H.accent1, size = 50) {
  s.addText(value, { isTextBox: true, x, y, w, h: 0.9, fontFace: "Cambria", fontSize: size, bold: true, color,
    margin: 0, valign: "bottom" });
  s.addText(caption, { isTextBox: true, x, y: y + 0.95, w, h: 0.6, fontSize: 13, color: GREY, margin: 0,
    valign: "top" });
}
function bar(s, runs, y, h = 0.8, fill = H.dk2) {
  card(s, M, y, CW, h, fill);
  s.addText(runs, { isTextBox: true, x: M + 0.35, y, w: CW - 0.7, h, fontSize: 15, valign: "middle",
    color: H.lt1, margin: 0 });
}
// rows[0] is the header; `bold` marks [row, col] cells, `hi` marks highlighted rows
function table(s, rows, { x, y, w, colW, rowH = 0.42, fontSize = 13, bold = [], hi = [], align } = {}) {
  const k = w / colW.reduce((a, b) => a + b, 0);
  const isBold = (r, c) => bold.some(([br, bc]) => br === r && (bc === c || bc === "*"));
  const data = rows.map((row, r) => row.map((t, c) => ({ text: t, options: r === 0
    ? { bold: true, color: H.lt1, fill: { color: H.dk2 }, fontSize: fontSize - 0.5, align: "center" }
    : { color: H.dk1, fontSize, bold: isBold(r, c), align: c === 0 ? "left" : (align || "center"),
        fill: { color: hi.includes(r) ? "FBEFD7" : (r % 2 ? H.lt1 : ROW) } } })));
  s.addTable(data, { x, y, w, colW: colW.map((v) => v * k), rowH, margin: [0.03, 0.08, 0.03, 0.08],
    border: { type: "solid", color: LINE, pt: 0.75 }, valign: "middle", fontFace: "Calibri" });
}
// Charts are the notebook's PNGs: Google Slides drops native chart category labels on import.
function img(s, name, x, y, w, h) {
  const file = path.join(ROOT, "hasil", `${name}.png`);
  const buf = fs.readFileSync(file);
  const pw = buf.readUInt32BE(16), ph = buf.readUInt32BE(20); // PNG IHDR
  const scale = Math.min(w / pw, h / ph);
  const iw = pw * scale, ih = ph * scale;
  s.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih });
}
function bullets(items, opts = {}) {
  return items.map((t, j) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, i) => ({ text: r.text, options: { ...(r.options || {}), ...(i === 0 ? { bullet: opts.bullet ?? true } : {}),
      breakLine: i === runs.length - 1 && j < items.length - 1 } }));
  }).flat();
}
const b = (text) => ({ text, options: { bold: true } });

// ------------------------------------------------------------------ numbers
const model = Object.fromEntries(R.models.map((m) => [m.model, m]));
const single = R.models.filter((m) => m.jenis === "tunggal");
const ensemble = R.models.filter((m) => m.jenis === "ensemble");
const bestSingle = single.reduce((a, c) => (c.macro_f1 > a.macro_f1 ? c : a));
const bestEns = ensemble.reduce((a, c) => (c.macro_f1 > a.macro_f1 ? c : a));
const bestRecall = R.models.reduce((a, c) => (c.recall_spoiled > a.recall_spoiled ? c : a));
const mean = (arr) => arr.reduce((a, c) => a + c, 0) / arr.length;
const wr = Object.values(R.wrangling);
const aug = (m, k) => R.augmentation.find((r) => r.model === m && r.augmentasi === k).macro_f1;
const rl = R.rl, imp = rl.impact;
const sc = R.session_curve;
const leak = R.leakage;
const cc = R.class_counts;
const nAll = R.n_windows;
const sensorRank = Object.keys(R.sensor_importance);
const mq135Rank = sensorRank.indexOf("MQ-135") + 1;
const hard = R.hardest_sessions;
const medianPeak = 535; // median MQ-135 peak over the 15 sessions (notebook, Bagian 7)

const S1 = "Erlangga: pendahuluan, data, wrangling";
const S2 = "Michael: validasi dan ensemble";
const S3 = "Rijal: reinforcement learning, analisis";
const S4 = "Erlangga: pengembangan dan simpulan";

// ================================================================== 1. title
pres.addSection({ title: S1 });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: S1 });
  label(s, "Machine Learning II · Evaluasi Tengah Semester", M, 0.85, 9, H.accent2);
  s.addText([
    { text: "KLASIFIKASI KESEGARAN DAGING AYAM", options: { breakLine: true } },
    { text: "BERBASIS ENSEMBLE DAN", options: { breakLine: true } },
    { text: "REINFORCEMENT LEARNING" },
  ], { isTextBox: true, x: M, y: 1.3, w: 12, h: 2.3, fontFace: "Cambria", fontSize: 38, bold: true,
    color: H.lt1, margin: 0, valign: "top" });
  s.addText("Studi kasus data electronic nose (sensor gas seri MQ)", { isTextBox: true, x: M, y: 3.75, w: 11,
    h: 0.5, fontSize: 20, italic: true, color: LIGHT, margin: 0 });
  s.addText([
    { text: "Disusun oleh:", options: { breakLine: true, color: H.accent2, bold: true } },
    { text: "Muhammad Erlangga Kurniawan (23083010045)", options: { breakLine: true } },
    { text: "Michael Valencio Erasmo Bagas Ganendra (23083010042)", options: { breakLine: true } },
    { text: "Rijal Apuila Khufi (23083010033)" },
  ], { isTextBox: true, x: M, y: 4.55, w: 8, h: 1.6, fontSize: 15, color: H.lt1, margin: 0, paraSpaceAfter: 3 });
  s.addShape(pres.shapes.LINE, { x: M, y: 6.4, w: CW, h: 0, line: { color: "3E6466", width: 1 } });
  s.addText("Program Studi Sains Data · UPN \"Veteran\" Jawa Timur", { isTextBox: true, x: M, y: 6.55, w: 8,
    h: 0.35, fontSize: 12, color: "9FB8B3", margin: 0 });
  s.addText("Oktober 2026", { isTextBox: true, x: W - M - 3, y: 6.55, w: 3, h: 0.35, fontSize: 12, align: "right",
    color: "9FB8B3", margin: 0 });
  s.addNotes(`[Erlangga] Selamat pagi. Kami dari kelompok yang beranggotakan Muhammad Erlangga Kurniawan, Michael Valencio Erasmo Bagas Ganendra, dan Rijal Apuila Khufi. Kami akan mempresentasikan proyek berjudul ${TITLE}, dengan studi kasus data electronic nose.`);
}

// ================================================================== 2. contents
{
  const s = pres.addSlide({ masterName: "PLAIN", sectionTitle: S1 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 4.6, h: 7.5, fill: { color: H.dk2 }, line: { color: H.dk2, width: 0 } });
  s.addText([{ text: "Daftar", options: { breakLine: true } }, { text: "Isi" }], { isTextBox: true, x: M, y: 4.6,
    w: 3.8, h: 2.2, fontFace: "Cambria", fontSize: 54, bold: true, color: H.lt1, margin: 0, valign: "bottom" });
  const items = ["Latar Belakang", "Data Wrangling & Augmentation", "Ensemble Learning", "Reinforcement Learning",
    "Hasil dan Analisis", "Celah Pengembangan", "Simpulan"];
  items.forEach((t, i) => {
    const y = 0.75 + i * 0.86;
    square(s, 5.6, y, String(i + 1), 0.6);
    s.addText(t, { isTextBox: true, x: 6.5, y, w: 6.2, h: 0.6, fontSize: 22, color: H.dk1, margin: 0,
      valign: "middle" });
  });
  s.addNotes("[Erlangga] Presentasi kami terdiri dari tujuh bagian: latar belakang, data wrangling dan augmentation, ensemble learning, reinforcement learning, hasil dan analisis, celah pengembangan, dan simpulan. Saya menyampaikan latar belakang dan pengolahan data. Michael menyampaikan skema validasi dan ensemble learning. Rijal menyampaikan reinforcement learning serta hasil dan analisis. Terakhir, saya menutup dengan celah pengembangan dan simpulan.");
}

// ================================================================== 3. background + dataset
{
  const s = content("Latar Belakang", S1);
  speaker(s, "Erlangga");
  const lw = 5.35;
  s.addText([
    { text: "Daging ayam mudah mengalami pembusukan (spoilage) akibat aktivitas mikroba selama penyimpanan. Proses ini menghasilkan senyawa volatil seperti amonia (NH₃) dan senyawa organik volatil (VOC).", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Uji laboratorium (TVB-N, Total Plate Count) akurat, tetapi lama, mahal, dan merusak sampel. " },
    { text: "Electronic nose (e-nose)", options: { bold: true } },
    { text: ", yaitu rangkaian sensor gas seri MQ, menjadi alternatif yang cepat, murah, dan non-destruktif." },
  ], { isTextBox: true, x: M, y: 1.4, w: lw, h: 2.35, fontSize: 14.5, color: H.dk1, margin: 0, valign: "top",
    align: "justify" });
  card(s, M, 3.95, lw, 2.4);
  label(s, "Irisan skripsi anggota", M + 0.3, 4.12, lw - 0.6);
  const theses = [
    ["Muhammad Erlangga Kurniawan", "kesegaran makanan dengan sensor MQ-135"],
    ["Rijal Apuila Khufi", "mutu produk unggas (kuning telur puyuh)"],
    ["Michael Valencio Erasmo Bagas Ganendra", "model gradient boosting (XGBoost)"],
  ];
  s.addText(theses.map(([n, t], i) => [
    { text: n, options: { bold: true, color: H.accent1, breakLine: true } },
    { text: t, options: { breakLine: i < theses.length - 1 } },
  ]).flat(), { isTextBox: true, x: M + 0.3, y: 4.5, w: lw - 0.6, h: 1.75, fontSize: 12.5, color: H.dk1, margin: 0,
    valign: "top", paraSpaceAfter: 2 });

  // example values = medians of complete_data.csv (sensor values are ADC readings minus the session's first reading)
  const x = M + lw + 0.35, w = W - M - x;
  table(s, [
    ["Parameter Dataset", "Contoh Sampel", "Keterangan"],
    ["s1 · MQ-135", "106", "Amonia (NH₃), CO₂, benzena"],
    ["s2 · MQ-2", "84", "LPG, metana, asap"],
    ["s3 · MQ-3", "74", "Alkohol (etanol)"],
    ["s4 · MQ-6", "38", "LPG, butana"],
    ["s5 · MQ-7", "20", "Karbon monoksida (CO)"],
    ["s6 · MQ-9", "56", "CO dan gas mudah terbakar"],
    ["s8 · Kelembapan", "0,3", "Perubahan kelembapan relatif"],
    ["hour", "0–27 jam", "Lama penyimpanan (dasar label)"],
    ["group", `${R.n_sessions} sesi`, "Sesi rekaman (sampel ayam)"],
    ["Label [Target]", "Fresh", "Fresh · Semi-fresh · Spoiled"],
  ], { x, y: 1.4, w, colW: [2.0, 1.4, 3.1], rowH: 0.41, fontSize: 12.5, hi: [10], bold: [[10, "*"]] });
  s.addText(`${num(R.n_readings)} baris dari ${R.n_sessions} sesi rekaman. Nilai sensor = selisih bacaan ADC terhadap awal rekaman (median). Sumber: Anwar & Anwar (2025) [1].`,
    { isTextBox: true, x, y: 6.0, w, h: 0.5, fontSize: 11, color: GREY, margin: 0, valign: "top" });
  s.addNotes(`[Erlangga] Daging ayam mudah mengalami pembusukan selama penyimpanan karena aktivitas mikroba. Pembusukan menghasilkan senyawa volatil seperti amonia. Uji laboratorium memang akurat, tetapi lama, mahal, dan merusak sampel. Karena itu kami menggunakan electronic nose, yaitu rangkaian sensor gas seri MQ yang cepat, murah, dan tidak merusak sampel. Topik ini merupakan irisan skripsi anggota kelompok: skripsi saya tentang kesegaran makanan dengan sensor MQ-135, Rijal tentang mutu produk unggas, dan Michael tentang XGBoost. Dataset yang kami gunakan berasal dari Anwar dan Anwar tahun 2025, berisi ${num(R.n_readings)} baris dari ${R.n_sessions} sesi rekaman. Tabel di kanan menunjukkan parameternya: enam sensor gas MQ, kelembapan, lama penyimpanan, dan sesi rekaman. Targetnya tiga kelas kesegaran: Fresh, Semi-fresh, dan Spoiled.`);
}

// ================================================================== 4. why not a fixed threshold
{
  const s = content("Mengapa Aturan Ambang Tetap (If-Else) Tidak Cukup?", S1);
  speaker(s, "Erlangga");
  const lw = 6.95;
  s.addText(bullets([
    [b("Variasi antar sesi: "), { text: `puncak bacaan MQ-135 berkisar ${R.peak_min.toFixed(0)}–${R.peak_max.toFixed(0)}, sehingga satu nilai ambang tidak berlaku untuk semua sampel.` }],
    [b("False Negative (ayam busuk lolos): "), { text: "Spoiled diprediksi layak → berisiko bagi kesehatan konsumen." }],
    [b("False Positive (ayam layak terbuang): "), { text: "Fresh/Semi-fresh diprediksi Spoiled → kerugian ekonomi." }],
  ]), { isTextBox: true, x: M, y: 1.4, w: lw, h: 2.4, fontSize: 15, color: H.dk1, margin: 0, paraSpaceAfter: 8,
    valign: "top" });
  const p = (k) => pct(cc[k] / nAll);
  table(s, [
    ["Kelas", "Waktu simpan*", "Jumlah", "Proporsi", "Status"],
    ["Fresh", "≤ 8 jam", num(cc.Fresh), p("Fresh"), "Layak jual"],
    ["Semi-fresh", "8–16 jam", num(cc["Semi-fresh"]), p("Semi-fresh"), "Layak, mendekati batas"],
    ["Spoiled", "> 16 jam", num(cc.Spoiled), p("Spoiled"), "Tidak layak"],
    ["Total", "", num(nAll), "100%", ""],
  ], { x: M, y: 3.95, w: lw, colW: [1.35, 1.45, 1.05, 1.1, 2.0], rowH: 0.4, fontSize: 13, bold: [[4, "*"], [3, 4]] });
  s.addText(`*Sesi > 15 jam. Sesi ≤ 15 jam: ≤ 5 / 5–10 / > 10 jam (mengikuti paper asli). Rasio kelas terbesar : terkecil = ${fmt(R.class_ratio, 2)}.`,
    { isTextBox: true, x: M, y: 6.05, w: lw, h: 0.5, fontSize: 11, color: GREY, margin: 0, valign: "top" });
  img(s, "01_mq135_per_sesi", M + lw + 0.25, 1.35, W - M - (M + lw + 0.25), 4.9);
  s.addNotes(`[Erlangga] Mengapa tidak cukup memakai aturan ambang tetap, misalnya jika nilai MQ-135 di atas angka tertentu maka ayam dianggap busuk? Grafik di kanan menunjukkan bacaan MQ-135 setiap sesi. Polanya sama-sama naik, tetapi levelnya sangat berbeda: puncaknya berkisar ${R.peak_min.toFixed(0)} sampai ${R.peak_max.toFixed(0)}. Satu nilai ambang tidak akan berlaku untuk semua sampel. Selain itu, jenis kesalahan punya konsekuensi berbeda. False negative, yaitu ayam busuk diprediksi layak, berisiko bagi kesehatan konsumen. False positive, yaitu ayam layak diprediksi busuk, menimbulkan kerugian ekonomi. Label kelas dibuat dari lama penyimpanan mengikuti paper asli. Hasilnya ${num(nAll)} sampel dengan distribusi kelas yang seimbang.`);
}

// ================================================================== 5. wrangling + augmentation (methods)
{
  const s = content("Data Wrangling & Augmentation", S1);
  speaker(s, "Erlangga");
  const cw = (CW - 0.35) / 2;
  const cols = [
    ["Data Wrangling", "Proses pembersihan, transformasi, dan restrukturisasi data mentah menjadi format yang siap dianalisis.", [
      ["A. Verifikasi integritas data", "checksum SHA-256 dan pengurutan per sesi"],
      ["B. Agregasi window 5 menit", `${num(R.n_readings)} baris → ${num(R.n_windows_all)} segmen`],
      ["C. Penghapusan periode warm-up sensor", `2 jam pertama dihapus → ${num(R.n_windows)} segmen`],
      [`D. Rekayasa fitur (${R.n_features} fitur)`, "log-level, proporsi relatif antar sensor, perubahan terhadap baseline (analog Rs/R0), kelembapan"],
      ["E. Pencegahan data leakage", "variabel waktu tidak dijadikan fitur karena label dibuat dari waktu"],
    ]],
    ["Data Augmentation", "Teknik menambah variasi data latih secara sintetis untuk meningkatkan ketahanan (robustness) model terhadap variasi sensor.", [
      ["A. Gaussian noise jittering", "derau acak dengan σ = 5% simpangan baku tiap sensor"],
      ["B. Simulasi variasi sensitivitas sensor", "bacaan tiap sensor dikalikan faktor log-normal (σ = 0,3) per sesi"],
      ["Penerapan", "hanya pada data latih, 2 salinan sintetis per sesi"],
      ["SMOTE tidak digunakan", `distribusi kelas sudah seimbang (rasio ${fmt(R.class_ratio, 2)})`],
    ]],
  ];
  cols.forEach(([head, def, steps], i) => {
    const x = M + i * (cw + 0.35);
    card(s, x, 1.35, cw, 5.35);
    label(s, head, x + 0.3, 1.55, cw - 0.6);
    s.addText(def, { isTextBox: true, x: x + 0.3, y: 1.9, w: cw - 0.6, h: 0.75, fontSize: 13.5, italic: true,
      color: GREY, margin: 0, valign: "top" });
    s.addText(steps.map(([t, d], j) => [
      { text: t, options: { bold: true, color: H.dk2, breakLine: true } },
      { text: d, options: { breakLine: j < steps.length - 1 } },
    ]).flat(), { isTextBox: true, x: x + 0.3, y: 2.75, w: cw - 0.6, h: 3.8, fontSize: 13.5, color: H.dk1,
      margin: 0, valign: "top", paraSpaceAfter: 4 });
  });
  s.addNotes(`[Erlangga] Data wrangling adalah proses membersihkan dan mengubah data mentah menjadi format yang siap dianalisis. Ada lima tahap. Pertama, integritas file diverifikasi dengan SHA-256. Kedua, bacaan sensor dirata-ratakan per 5 menit, sehingga ${num(R.n_readings)} baris menjadi ${num(R.n_windows_all)} segmen. Ketiga, dua jam pertama dihapus karena sensor masih dalam masa warm-up, sehingga tersisa ${num(R.n_windows)} segmen. Keempat, kami membuat ${R.n_features} fitur: log-level gas, proporsi relatif antar sensor, perubahan terhadap baseline jam pertama yang analog dengan Rs per R0, dan kelembapan. Kelima, variabel waktu tidak dijadikan fitur karena label dibuat dari waktu; jika dipakai akan terjadi data leakage. Untuk data augmentation, kami menguji dua teknik: Gaussian noise jittering dan simulasi variasi sensitivitas sensor. Augmentasi hanya diterapkan pada data latih. SMOTE tidak digunakan karena distribusi kelas sudah seimbang. Selanjutnya Michael akan menjelaskan skema validasi dan hasilnya.`);
}

// ================================================================== 6. validation (Michael)
pres.addSection({ title: S2 });
{
  const s = content("Skema Validasi: Leave-One-Session-Out (LOSO)", S2);
  speaker(s, "Michael");
  card(s, M, 1.4, 6.4, 3.35);
  label(s, "Akurasi Random Forest", M + 0.35, 1.6, 5.8);
  stat(s, M + 0.35, 2.0, 2.9, pct(leak.acak.akurasi), "Random split\n(terdampak data leakage)", H.accent4);
  stat(s, M + 3.3, 2.0, 2.9, pct(leak.loso.akurasi), "Leave-One-Session-Out\n(estimasi realistis)", H.accent1);
  const x = M + 6.7, w = W - M - x;
  card(s, x, 1.4, w, 3.35, H.dk2);
  label(s, "Prosedur LOSO", x + 0.3, 1.6, w - 0.6, H.accent2);
  s.addText(bullets(["Satu sesi dijadikan data uji.", `Model dilatih dengan ${R.n_sessions - 1} sesi lainnya.`,
    `Diulang ${R.n_sessions} kali hingga setiap sesi pernah menjadi data uji.`], { bullet: { type: "number" } }),
  { isTextBox: true, x: x + 0.3, y: 2.0, w: w - 0.6, h: 1.6, fontSize: 15, color: H.lt1, margin: 0, paraSpaceAfter: 6,
    valign: "top" });
  s.addText("Sesuai kondisi penggunaan: alat diterapkan pada sampel ayam baru.", { isTextBox: true, x: x + 0.3,
    y: 3.85, w: w - 0.6, h: 0.7, fontSize: 13, color: LIGHT, margin: 0 });
  s.addText([
    b("Alasan: "),
    { text: "segmen yang berurutan dalam satu sesi sangat mirip. Pada random split, segmen dari sesi yang sama masuk ke data latih dan data uji sekaligus, sehingga akurasi terlalu tinggi (overestimate)." },
  ], { isTextBox: true, x: M, y: 4.95, w: CW, h: 0.85, fontSize: 15, color: H.dk1, margin: 0, valign: "top" });
  bar(s, [
    { text: "Temuan serupa pada paper asli [1]: ", options: { bold: true, color: H.accent2 } },
    { text: "Random Forest 100% dengan random split, tetapi 69% dengan pembagian non-acak." },
  ], 6.0, 0.7);
  s.addNotes(`[Michael] Terima kasih. Sebelum membandingkan model, kami menentukan skema validasi. Segmen yang berurutan dalam satu sesi isinya sangat mirip. Jika data dibagi secara acak, segmen dari sesi yang sama masuk ke data latih dan data uji sekaligus, sehingga akurasinya terlalu tinggi. Dengan random split, Random Forest mencapai ${pct(leak.acak.akurasi)}. Dengan Leave-One-Session-Out, akurasinya ${pct(leak.loso.akurasi)}. Pada LOSO, satu sesi dijadikan data uji, model dilatih dengan ${R.n_sessions - 1} sesi lainnya, dan proses ini diulang ${R.n_sessions} kali. Skema ini sesuai kondisi nyata, karena alat akan dipakai untuk sampel ayam baru. Paper aslinya juga menemukan hal serupa: 100 persen dengan random split dan 69 persen dengan pembagian non-acak. Seluruh hasil berikutnya menggunakan LOSO.`);
}

// ================================================================== 7. wrangling + augmentation results
{
  const s = content("Hasil Data Wrangling & Augmentation", S2);
  speaker(s, "Michael");
  const lw = 6.7;
  label(s, "Efek rekayasa fitur (Random Forest, LOSO)", M, 1.4, lw);
  table(s, [
    ["Tahap fitur", "Jumlah fitur", "F1-Macro"],
    ["Rata-rata mentah", "7", fmt(wr[0])],
    ["+ Log-level & proporsi relatif", "13", fmt(wr[1])],
    ["+ Perubahan terhadap baseline", String(R.n_features), fmt(wr[2])],
  ], { x: M, y: 1.75, w: lw, colW: [3.7, 1.5, 1.5], rowH: 0.42, fontSize: 13.5, hi: [3], bold: [[3, "*"]] });
  label(s, "Efek augmentasi (F1-Macro, LOSO)", M, 3.7, lw);
  table(s, [
    ["Skema", "Decision Tree", "Random Forest"],
    ["Tanpa augmentasi", fmt(aug("Decision Tree", "tanpa")), fmt(aug("Random Forest", "tanpa"))],
    ["Gaussian jitter", fmt(aug("Decision Tree", "jitter")), fmt(aug("Random Forest", "jitter"))],
    ["Variasi sensitivitas sensor", fmt(aug("Decision Tree", "sensitivitas")), fmt(aug("Random Forest", "sensitivitas"))],
  ], { x: M, y: 4.05, w: lw, colW: [3.2, 1.75, 1.75], rowH: 0.42, fontSize: 13.5, bold: [[1, 2]] });
  img(s, "05_efek_wrangling", M + lw + 0.25, 1.35, W - M - (M + lw + 0.25), 4.5);
  const d = (k) => signed(aug("Random Forest", k) - aug("Random Forest", "tanpa"));
  bar(s, [
    { text: "Temuan: ", options: { bold: true, color: H.accent2 } },
    { text: `rekayasa fitur menaikkan F1-Macro ${signed(wr[2] - wr[0])}. Augmentasi tidak meningkatkan Random Forest (jitter ${d("jitter")}, sensitivitas ${d("sensitivitas")}): variasi sintetis belum mewakili variasi nyata antar sesi.` },
  ], 5.95, 0.85);
  s.addNotes(`[Michael] Tabel atas menunjukkan efek rekayasa fitur pada Random Forest. Dengan rata-rata mentah, F1-macro ${fmt(wr[0])}. Setelah ditambah log-level dan proporsi relatif, menjadi ${fmt(wr[1])}. Setelah ditambah perubahan terhadap baseline, menjadi ${fmt(wr[2])}. Jadi tahap wrangling memberi peningkatan terbesar, yaitu ${fmt(wr[2] - wr[0])}. Tabel bawah menunjukkan efek augmentasi. Pada Random Forest, jitter menurunkan skor ${fmt(aug("Random Forest", "tanpa") - aug("Random Forest", "jitter"))} dan variasi sensitivitas menurunkan ${fmt(aug("Random Forest", "tanpa") - aug("Random Forest", "sensitivitas"))}. Pada Decision Tree hasilnya campuran. Kesimpulannya, augmentasi sintetis belum mewakili variasi nyata antar sesi, sehingga tidak meningkatkan performa.`);
}

// ================================================================== 8. ensemble (definition + methods)
{
  const s = content("Ensemble Learning", S2);
  speaker(s, "Michael");
  s.addText([
    { text: "Ensemble learning", options: { bold: true } },
    { text: " adalah metode yang menggabungkan beberapa model (base learner) untuk menghasilkan prediksi yang lebih akurat dan stabil dibandingkan model tunggal [2]. Model tunggal cenderung mengalami overfitting dan sensitif terhadap noise, sehingga penggabungan model membantu menurunkan kesalahan." },
  ], { isTextBox: true, x: M, y: 1.35, w: CW, h: 1.1, fontSize: 15, color: H.dk1, margin: 0, valign: "top",
    align: "justify" });
  const cw = (CW - 0.35) / 2;
  const boxes = [
    ["Model pembanding (tunggal)", H.lt2, H.accent1, H.dk1, [
      ["A. Logistic Regression", "batas keputusan linear"],
      ["B. K-Nearest Neighbors", "k = 15, voting tetangga terdekat"],
      ["C. Decision Tree", "satu pohon, kedalaman maksimum 6"],
    ]],
    ["Model ensemble", H.dk2, H.accent2, H.lt1, [
      ["A. Random Forest [3]", "bagging: 300 pohon (kedalaman 6), hasil di-voting"],
      ["B. XGBoost [4]", "boosting: 200 pohon berurutan, tiap pohon memperbaiki kesalahan sebelumnya"],
      ["C. Soft Voting Classifier", "rata-rata probabilitas Logistic Regression + Random Forest + XGBoost"],
    ]],
  ];
  boxes.forEach(([head, fill, hc, tc, items], i) => {
    const x = M + i * (cw + 0.35);
    card(s, x, 2.65, cw, 3.3, fill);
    label(s, head, x + 0.3, 2.85, cw - 0.6, hc);
    s.addText(items.map(([t, d], j) => [
      { text: t, options: { bold: true, breakLine: true } },
      { text: d, options: { breakLine: j < items.length - 1 } },
    ]).flat(), { isTextBox: true, x: x + 0.3, y: 3.3, w: cw - 0.6, h: 2.55, fontSize: 14, color: tc, margin: 0,
      valign: "top", paraSpaceAfter: 6 });
  });
  s.addText("Hyperparameter ditetapkan sejak awal, tidak di-tuning menggunakan data uji.  [2] Zhou (2012) · [3] Breiman (2001) · [4] Chen & Guestrin (2016)",
    { isTextBox: true, x: M, y: 6.15, w: CW, h: 0.45, fontSize: 11.5, color: GREY, margin: 0 });
  s.addNotes("[Michael] Ensemble learning adalah metode yang menggabungkan beberapa model untuk menghasilkan prediksi yang lebih akurat dan stabil dibandingkan model tunggal. Model tunggal cenderung mengalami overfitting dan sensitif terhadap noise. Kami membandingkan tiga model tunggal, yaitu Logistic Regression, KNN, dan Decision Tree, dengan tiga model ensemble. Random Forest menggunakan teknik bagging dengan 300 pohon. XGBoost menggunakan boosting, yaitu pohon dibangun berurutan dan setiap pohon memperbaiki kesalahan pohon sebelumnya. Soft Voting merata-ratakan probabilitas dari Logistic Regression, Random Forest, dan XGBoost. Seluruh hyperparameter ditetapkan sejak awal dan tidak di-tuning menggunakan data uji.");
}

// ================================================================== 9. ensemble results table
{
  const s = content("Hasil Ensemble Learning", S2);
  speaker(s, "Michael");
  const cols = ["akurasi", "macro_f1", "f1_weighted", "recall_spoiled", "akurasi_sesi_terburuk"];
  const best = Object.fromEntries(cols.map((k) => [k, R.models.reduce((a, c) => (c[k] > a[k] ? c : a)).model]));
  const rows = R.models.map((m) => [m.model, m.jenis === "ensemble" ? "Ensemble" : "Tunggal", pct(m.akurasi),
    fmt(m.macro_f1), fmt(m.f1_weighted), pct(m.recall_spoiled), pct(m.akurasi_sesi_terburuk)]);
  const bold = [];
  R.models.forEach((m, i) => cols.forEach((k, j) => { if (best[k] === m.model) bold.push([i + 1, j + 2]); }));
  table(s, [["Model", "Jenis", "Akurasi", "F1-Macro", "F1-Weighted", "Recall Spoiled", "Akurasi Sesi Terburuk"], ...rows],
    { x: M, y: 1.4, w: CW, colW: [2.5, 1.4, 1.4, 1.4, 1.6, 1.8, 2.0], rowH: 0.48, fontSize: 14,
      hi: [R.models.indexOf(model["Random Forest"]) + 1], bold });
  s.addText("Validasi Leave-One-Session-Out. Angka tebal = terbaik per kolom.", { isTextBox: true, x: M, y: 4.85,
    w: CW, h: 0.3, fontSize: 11, color: GREY, margin: 0 });
  const facts = [
    [`${fmt(mean(ensemble.map((m) => m.macro_f1)))} vs ${fmt(mean(single.map((m) => m.macro_f1)))}`, "rata-rata F1-Macro ensemble vs model tunggal"],
    [`${fmt(model["Random Forest"].macro_f1)} vs ${fmt(model["Decision Tree"].macro_f1)}`, "Random Forest vs Decision Tree (kedalaman pohon sama)"],
    [pct(bestRecall.recall_spoiled), `recall Spoiled tertinggi (${bestRecall.model})`],
  ];
  const fw = (CW - 0.5) / 3;
  facts.forEach(([v, t], i) => {
    const x = M + i * (fw + 0.25);
    card(s, x, 5.3, fw, 1.4, i === 0 ? H.dk2 : H.lt2);
    s.addText(v, { isTextBox: true, x: x + 0.3, y: 5.38, w: fw - 0.6, h: 0.65, fontFace: "Cambria", fontSize: 26,
      bold: true, color: i === 0 ? H.accent2 : H.accent1, margin: 0, valign: "middle" });
    s.addText(t, { isTextBox: true, x: x + 0.3, y: 6.05, w: fw - 0.6, h: 0.55, fontSize: 13,
      color: i === 0 ? H.lt1 : H.dk1, margin: 0, valign: "top" });
  });
  s.addNotes(`[Michael] Tabel ini menunjukkan hasil enam model dengan validasi LOSO. Ketiga model ensemble berada di atas seluruh model tunggal. Rata-rata F1-macro ensemble ${fmt(mean(ensemble.map((m) => m.macro_f1)))}, sedangkan model tunggal ${fmt(mean(single.map((m) => m.macro_f1)))}. Model terbaik adalah ${bestEns.model} dengan akurasi ${pct(bestEns.akurasi)} dan F1-macro ${fmt(bestEns.macro_f1)}. Perbandingan paling adil adalah Random Forest dengan Decision Tree, karena kedalaman pohonnya sama: ${fmt(model["Random Forest"].macro_f1)} berbanding ${fmt(model["Decision Tree"].macro_f1)}. Untuk mendeteksi ayam busuk, ${bestRecall.model} memiliki recall Spoiled tertinggi, yaitu ${pct(bestRecall.recall_spoiled)}.`);
}

// ================================================================== 10. confusion matrix
{
  const s = content("Hasil Ensemble Learning: Confusion Matrix", S2);
  speaker(s, "Michael");
  img(s, "06_confusion_rf", M, 1.35, 5.6, 5.3);
  const x = M + 5.9, w = W - M - x;
  const rc = R.recall_rf;
  label(s, "Interpretasi (Random Forest, LOSO)", x, 1.45, w);
  s.addText(bullets([
    "Tidak ada sampel Fresh yang diprediksi Spoiled, maupun sebaliknya.",
    "Kesalahan terkonsentrasi pada kelas Semi-fresh, yang berada di antara dua kelas lain.",
    [b("Recall per kelas: "), { text: `Fresh ${pct(rc.Fresh)} · Semi-fresh ${pct(rc["Semi-fresh"])} · Spoiled ${pct(rc.Spoiled)}` }],
  ]), { isTextBox: true, x, y: 1.85, w, h: 2.3, fontSize: 15, color: H.dk1, margin: 0, paraSpaceAfter: 8, valign: "top" });
  card(s, x, 4.35, w, 2.3, H.dk2);
  label(s, "Eksperimen lanjutan (notebook Michael)", x + 0.3, 4.55, w - 0.6, H.accent2);
  s.addText(bullets([
    "Extra Trees (500 pohon): F1-Macro 0,787, setara dengan Random Forest (selisih 0,005).",
    "Penambahan pohon jenuh di ±300: 300 pohon 0,782; 500 pohon 0,781.",
  ]), { isTextBox: true, x: x + 0.3, y: 4.95, w: w - 0.6, h: 1.6, fontSize: 14, color: H.lt1, margin: 0,
    paraSpaceAfter: 6, valign: "top" });
  s.addNotes(`[Michael] Ini confusion matrix Random Forest. Tidak ada sampel Fresh yang diprediksi Spoiled, dan sebaliknya. Kesalahan terkonsentrasi pada kelas Semi-fresh, dengan recall ${pct(rc["Semi-fresh"])}, karena kelas ini berada di antara dua kelas lainnya. Pada eksperimen lanjutan di notebook saya, Extra Trees dengan 500 pohon mencapai F1-macro 0,787, setara dengan Random Forest. Penambahan pohon di atas 300 juga tidak lagi meningkatkan performa. Selanjutnya Rijal akan menjelaskan reinforcement learning.`);
}

// ================================================================== 11. RL definition + method (Rijal)
pres.addSection({ title: S3 });
{
  const s = content("Reinforcement Learning", S3);
  speaker(s, "Rijal");
  const cw = (CW - 0.5) / 2;
  const blocks = [
    ["1", "Definisi", [
      { text: "Reinforcement learning merupakan pendekatan machine learning di mana agen belajar mengambil keputusan melalui interaksi trial and error dengan lingkungan. Setiap aksi menghasilkan reward atau penalti, dan agen berupaya memaksimalkan total reward [5].", options: { breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 6 } },
      { text: "Pada proyek ini, agen tidak hanya mengklasifikasikan, tetapi memutuskan tindakan: " },
      { text: "Jual, Cek manual, atau Buang.", options: { bold: true } },
    ]],
    ["2", "Metode yang Digunakan", [
      { text: "Contextual multi-armed bandit dengan strategi ε-greedy (ε = 0,1):", options: { bold: true, breakLine: true } },
      ...bullets([
        "Exploitation 90%: memilih aksi dengan nilai Q tertinggi",
        "Exploration 10%: memilih aksi secara acak",
        "Pembaruan nilai: Q ← Q + (r − Q) / n (rata-rata inkremental)",
        "3.000 iterasi × 5 seed; dilatih pada 12 sesi, dievaluasi pada 3 sesi lain (5 fold)",
      ]),
    ]],
  ];
  blocks.forEach(([n, head, runs], i) => {
    const x = M + i * (cw + 0.5);
    square(s, x, 1.45, n, 0.8);
    s.addText(head, { isTextBox: true, x: x + 1.05, y: 1.45, w: cw - 1.05, h: 0.8, fontSize: 20, bold: true,
      color: H.dk2, margin: 0, valign: "middle" });
    s.addText(runs, { isTextBox: true, x, y: 2.5, w: cw, h: 3.6, fontSize: 15, color: H.dk1, margin: 0,
      valign: "top", paraSpaceAfter: 5, align: i === 0 ? "justify" : "left" });
  });
  s.addText("[5] Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.",
    { isTextBox: true, x: M, y: 6.3, w: CW, h: 0.35, fontSize: 11.5, color: GREY, margin: 0 });
  s.addNotes("[Rijal] Terima kasih. Reinforcement learning adalah pendekatan di mana agen belajar mengambil keputusan melalui trial and error. Setiap aksi menghasilkan reward atau penalti, dan agen berusaha memaksimalkan total reward. Tujuan sistem kami tidak hanya mengklasifikasikan, tetapi juga memutuskan tindakan: jual, cek manual, atau buang. Metode yang kami gunakan adalah contextual multi-armed bandit dengan strategi epsilon-greedy. Sembilan puluh persen agen memilih aksi dengan nilai tertinggi, dan sepuluh persen memilih aksi acak untuk eksplorasi. Nilai setiap aksi diperbarui dengan rata-rata reward yang diterima. Agen dilatih 3.000 iterasi pada 12 sesi dan dievaluasi pada 3 sesi lainnya.");
}

// ================================================================== 12. agent design
{
  const s = content("Desain Agen: State, Aksi, dan Reward", S3);
  speaker(s, "Rijal");
  const lw = 6.8;
  table(s, [
    ["Komponen", "Definisi pada proyek ini"],
    ["Agen", "Pengambil keputusan penanganan daging ayam"],
    ["State", "Prediksi Random Forest × tingkat keyakinan: yakin (probabilitas ≥ 0,7) atau ragu (< 0,7) → 6 state"],
    ["Aksi", "Jual · Cek manual · Buang"],
    ["Reward", "Nilai dari tabel di kanan, ditetapkan berdasarkan risiko"],
  ], { x: M, y: 1.4, w: lw, colW: [1.4, 5.4], rowH: 0.62, fontSize: 14, align: "left", bold: [[1, 0], [2, 0], [3, 0], [4, 0]] });
  const x = M + lw + 0.35, w = W - M - x;
  label(s, "Tabel reward", x, 1.4, w);
  table(s, [
    ["Aksi", "Ayam layak*", "Ayam busuk"],
    ["Jual", "+1", "−5"],
    ["Cek manual", "+0,5", "+0,5"],
    ["Buang", "−1", "+1"],
  ], { x, y: 1.75, w, colW: [1.9, 1.6, 1.6], rowH: 0.52, fontSize: 15, bold: [[1, 2]] });
  s.addText("*Layak = Fresh atau Semi-fresh. Cek manual selalu aman, tetapi membutuhkan biaya.", { isTextBox: true,
    x, y: 3.95, w, h: 0.6, fontSize: 12, color: GREY, margin: 0, valign: "top" });
  bar(s, [
    { text: "Penalti terbesar (−5) ", options: { bold: true, color: H.accent2 } },
    { text: "diberikan untuk menjual ayam busuk, karena False Negative berisiko bagi kesehatan konsumen. Membuang ayam layak (False Positive) diberi −1 karena menimbulkan kerugian ekonomi." },
  ], 5.0, 1.1);
  s.addNotes("[Rijal] Komponen agen kami sebagai berikut. State adalah prediksi Random Forest yang dikombinasikan dengan tingkat keyakinan model: yakin jika probabilitasnya minimal 0,7, dan ragu jika di bawahnya. Jadi ada enam state. Aksinya tiga: jual, cek manual, dan buang. Reward ditetapkan berdasarkan risiko. Menjual ayam layak bernilai plus satu, sedangkan menjual ayam busuk diberi penalti minus lima karena berisiko bagi konsumen. Cek manual selalu bernilai plus 0,5 karena aman tetapi membutuhkan biaya. Membuang ayam layak bernilai minus satu, dan membuang ayam busuk bernilai plus satu.");
}

// ================================================================== 13. learned policy
{
  const s = content("Hasil Reinforcement Learning: Kebijakan Agen", S3);
  speaker(s, "Rijal");
  const names = Object.keys(rl.state_reward);
  const acts = ["Jual", "Cek manual", "Buang"];
  const bold = [];
  const rows = names.map((n, i) => {
    const r = rl.state_reward[n];
    const bestA = acts.reduce((a, c) => (r[c] > r[a] ? c : a));
    bold.push([i + 1, 3 + acts.indexOf(bestA)], [i + 1, 6]);
    return [n, num(rl.state_counts[n]), pct(r["proporsi Spoiled"]), ...acts.map((a) => fmt(r[a], 2)), rl.policy[n]];
  });
  label(s, "Rata-rata reward per aksi di setiap state", M, 1.4, CW);
  table(s, [["State (prediksi model)", "Jumlah sampel", "Proporsi Spoiled", "Jual", "Cek manual", "Buang", "Aksi terpilih"], ...rows],
    { x: M, y: 1.75, w: CW, colW: [2.6, 1.5, 1.7, 1.3, 1.5, 1.3, 2.2], rowH: 0.45, fontSize: 14, bold, hi: [3] });
  const semi = rl.state_reward["Semi-fresh (yakin)"];
  bar(s, [
    { text: "Temuan: ", options: { bold: true, color: H.accent2 } },
    { text: `sampel Semi-fresh (yakin) mengandung ${pct(semi["proporsi Spoiled"])} ayam Spoiled, sehingga rata-rata reward Jual (${fmt(semi.Jual, 2)}) lebih rendah daripada Cek manual (${fmt(semi["Cek manual"], 2)}). Agen menemukan aturan ini sendiri melalui trial and error, tanpa diberi aturan.` },
  ], 5.15, 1.1);
  s.addText("Dihitung pada seluruh sampel dengan prediksi out-of-fold Random Forest (notebook Bagian 9).", { isTextBox: true,
    x: M, y: 6.35, w: CW, h: 0.3, fontSize: 11, color: GREY, margin: 0 });
  const sp = rl.state_reward["Spoiled (ragu)"];
  s.addNotes(`[Rijal] Tabel ini menunjukkan rata-rata reward setiap aksi di setiap state, yaitu nilai yang dipelajari agen melalui trial and error. Untuk prediksi Fresh, jual selalu bernilai plus satu, sehingga agen memilih jual. Yang menarik adalah Semi-fresh dengan keyakinan tinggi: ${pct(semi["proporsi Spoiled"])} di antaranya sebenarnya Spoiled, sehingga rata-rata reward jual turun menjadi ${fmt(semi.Jual, 2)}, lebih rendah dari cek manual ${fmt(semi["Cek manual"], 2)}. Agen menemukan sendiri bahwa Semi-fresh lebih aman dicek terlebih dahulu. Untuk Spoiled yang yakin, agen memilih buang. Untuk Spoiled yang ragu, ${pct(1 - sp["proporsi Spoiled"])} sebenarnya masih layak, sehingga agen memilih cek manual.`);
}

// ================================================================== 14. policy comparison
{
  const s = content("Hasil Reinforcement Learning: Perbandingan Kebijakan", S3);
  speaker(s, "Rijal");
  table(s, [
    ["Kebijakan", "Reward rata-rata (sesi uji)", "Ayam busuk terjual", "Ayam layak terbuang", "Cek manual"],
    ["Agen RL (ε-greedy)", fmt(rl.reward_final, 2), String(imp.busuk_terjual_agen), String(imp.layak_terbuang_agen), pct(imp.porsi_cek)],
    ["Selalu ikuti prediksi model", fmt(rl.trust_model, 2), String(imp.busuk_terjual_ikut_model), String(imp.layak_terbuang_ikut_model), "0%"],
    ["Selalu cek manual", fmt(rl.always_check, 2), "0", "0", "100%"],
    ["Aksi acak", fmt(rl.random_action, 2), "—", "—", "—"],
  ], { x: M, y: 1.4, w: CW, colW: [3.3, 2.4, 2.1, 2.2, 2.1], rowH: 0.5, fontSize: 14.5, hi: [1], bold: [[1, "*"]] });
  s.addText(`Jumlah ayam dihitung pada seluruh ${num(nAll)} sampel (${num(imp.n_busuk)} Spoiled, ${num(imp.n_layak)} layak).`,
    { isTextBox: true, x: M, y: 4.0, w: CW, h: 0.3, fontSize: 11, color: GREY, margin: 0 });
  const st = [
    [String(imp.busuk_terjual_agen), `ayam busuk terjual (vs ${imp.busuk_terjual_ikut_model} jika mengikuti prediksi model)`, H.accent1],
    [fmt(rl.reward_final, 2), `reward agen (vs ${fmt(rl.trust_model, 2)} jika mengikuti prediksi model)`, H.accent1],
    [pct(imp.porsi_cek), "kasus memerlukan cek manual (trade-off)", H.accent4],
  ];
  const fw = (CW - 0.5) / 3;
  st.forEach(([v, t, col], i) => {
    const x = M + i * (fw + 0.25);
    card(s, x, 4.5, fw, 2.1);
    stat(s, x + 0.3, 4.6, fw - 0.6, v, t, col, 44);
  });
  s.addNotes(`[Rijal] Kebijakan agen kami bandingkan dengan kebijakan lain pada sesi uji. Agen memperoleh rata-rata reward ${fmt(rl.reward_final, 2)}, lebih tinggi dari selalu mengikuti prediksi model yang hanya ${fmt(rl.trust_model, 2)} dan selalu cek manual ${fmt(rl.always_check, 2)}. Jika kebijakan agen diterapkan pada seluruh data, tidak ada ayam busuk yang terjual, dibandingkan ${imp.busuk_terjual_ikut_model} jika hanya mengikuti prediksi model. Konsekuensinya, ${pct(imp.porsi_cek)} kasus memerlukan cek manual dan ${imp.layak_terbuang_agen} ayam layak terbuang.`);
}

// ================================================================== 15. more iterations, more data
{
  const s = content("Hasil RL: Penambahan Iterasi dan Data", S3);
  speaker(s, "Rijal");
  const cw = (CW - 0.33) / 2;
  img(s, "08_rl_kurva", M, 1.3, cw, 3.75);
  img(s, "09_tambah_sesi", M + cw + 0.33, 1.3, cw, 3.75);
  [[M, [
    { text: "Penambahan iterasi: ", options: { bold: true, color: H.accent2 } },
    { text: `reward agen ${fmt(rl.reward_start, 2)} (iterasi 1) → ${fmt(rl.reward_after_100, 2)} (iterasi 100) → ${fmt(rl.reward_final, 2)} (iterasi 3.000).` },
  ]], [M + cw + 0.33, [
    { text: "Penambahan data: ", options: { bold: true, color: H.accent2 } },
    { text: `F1-Macro Random Forest ${fmt(sc[0].mean, 2)} (1 sesi latih) → ${fmt(sc[sc.length - 1].mean, 2)} (14 sesi latih). Data asli lebih efektif daripada augmentasi sintetis.` },
  ]]].forEach(([x, runs]) => {
    card(s, x, 5.2, cw, 1.4, H.dk2);
    s.addText(runs, { isTextBox: true, x: x + 0.3, y: 5.2, w: cw - 0.6, h: 1.4, fontSize: 14, color: H.lt1,
      valign: "middle", margin: 0 });
  });
  s.addNotes(`[Rijal] Rubrik reinforcement menanyakan bagaimana tambahan iterasi dan data meningkatkan performa. Grafik kiri menunjukkan penambahan iterasi: reward agen naik dari ${fmt(rl.reward_start, 2)} pada iterasi pertama menjadi ${fmt(rl.reward_after_100, 2)} setelah 100 iterasi, lalu stabil di ${fmt(rl.reward_final, 2)}. Grafik kanan menunjukkan penambahan data: F1-macro Random Forest naik dari ${fmt(sc[0].mean, 2)} dengan satu sesi latih menjadi ${fmt(sc[sc.length - 1].mean, 2)} dengan 14 sesi latih. Penambahan data asli terbukti lebih efektif daripada augmentasi sintetis.`);
}

// ================================================================== 16. analysis
{
  const s = content("Hasil dan Analisis", S3);
  speaker(s, "Rijal");
  const [h1, h2] = hard;
  const cards = [
    ["Batasan", [`Hanya ${R.n_sessions} sesi dari satu penelitian.`, "Label berdasarkan lama penyimpanan, bukan uji mikrobiologi.", "Nilai reward ditetapkan berdasarkan asumsi risiko."]],
    ["Sumber Galat", [`Kesalahan terbanyak pada Semi-fresh (recall ${pct(R.recall_rf["Semi-fresh"])}).`,
      `Sesi ${h1.sesi} dan ${h2.sesi} tersulit (akurasi ${pct(h1.akurasi)} dan ${pct(h2.akurasi)}): puncak MQ-135 hanya ${h1.puncak_MQ135.toFixed(0)} dan ${h2.puncak_MQ135.toFixed(0)} (median ${medianPeak}).`,
      "Kesalahan pada sesi tersebut memprediksi kondisi lebih segar dari sebenarnya."]],
    ["Trade-off", [`Agen menekan ayam busuk terjual dari ${imp.busuk_terjual_ikut_model} menjadi ${imp.busuk_terjual_agen}.`,
      `Konsekuensinya, ${pct(imp.porsi_cek)} kasus perlu cek manual dan ${imp.layak_terbuang_agen} ayam layak terbuang.`,
      "Penalti −2 (eksperimen Rijal) membuat agen menjual Semi-fresh: 71 ayam busuk terjual."]],
    ["Insight", [`Random split melebih-lebihkan akurasi (${pct(leak.acak.akurasi)} vs ${pct(leak.loso.akurasi)}).`,
      "Rekayasa fitur lebih berpengaruh daripada pemilihan model.", "Data asli lebih efektif daripada augmentasi sintetis.",
      `Sensor paling berpengaruh: ${sensorRank[0]} dan ${sensorRank[1]}; MQ-135 peringkat ke-${mq135Rank}.`]],
  ];
  const cw = (CW - 0.75) / 4;
  cards.forEach(([head, items], i) => {
    const x = M + i * (cw + 0.25);
    card(s, x, 1.4, cw, 5.25, H.dk2);
    s.addText(String(i + 1), { isTextBox: true, x: x + 0.3, y: 1.55, w: 1, h: 0.75, fontFace: "Cambria",
      fontSize: 40, color: H.accent2, margin: 0, valign: "middle" });
    s.addText(head, { isTextBox: true, x: x + 0.3, y: 2.35, w: cw - 0.6, h: 0.45, fontSize: 18, bold: true,
      color: H.lt1, margin: 0, italic: head === "Trade-off" });
    s.addText(bullets(items), { isTextBox: true, x: x + 0.3, y: 2.9, w: cw - 0.5, h: 3.65, fontSize: 12.5,
      color: LIGHT, margin: 0, paraSpaceAfter: 6, valign: "top" });
  });
  s.addNotes(`[Rijal] Dari seluruh hasil, ada empat poin analisis. Batasan: dataset hanya ${R.n_sessions} sesi, label berasal dari lama penyimpanan, dan nilai reward merupakan asumsi. Sumber galat: kesalahan terbanyak pada kelas Semi-fresh. Sesi ${h1.sesi} dan ${h2.sesi} paling sulit karena puncak MQ-135-nya sangat rendah, ${h1.puncak_MQ135.toFixed(0)} dan ${h2.puncak_MQ135.toFixed(0)}, jauh di bawah median ${medianPeak}, sehingga model memprediksi kondisinya lebih segar dari sebenarnya. Trade-off: agen menekan ayam busuk terjual menjadi nol, dengan konsekuensi ${pct(imp.porsi_cek)} kasus perlu cek manual. Pada eksperimen saya, jika penalti diturunkan menjadi minus dua, agen mulai menjual Semi-fresh dan 71 ayam busuk terjual. Insight: random split melebih-lebihkan akurasi, rekayasa fitur lebih berpengaruh daripada pemilihan model, data asli lebih efektif daripada augmentasi, dan sensor paling berpengaruh adalah ${sensorRank[0]} dan ${sensorRank[1]}. Selanjutnya Erlangga akan menutup presentasi.`);
}

// ================================================================== 17. future work (Erlangga)
pres.addSection({ title: S4 });
{
  const s = pres.addSlide({ masterName: "PLAIN", sectionTitle: S4 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 4.6, h: 7.5, fill: { color: H.dk2 }, line: { color: H.dk2, width: 0 } });
  s.addText([{ text: "Celah", options: { breakLine: true } }, { text: "Pengembangan" }], { isTextBox: true, x: M - 0.1,
    y: 4.4, w: 4.1, h: 2.4, fontFace: "Cambria", fontSize: 34, bold: true, color: H.lt1, margin: 0, valign: "bottom" });
  speaker(s, "Erlangga");
  const items = [
    ["Penambahan data", "jumlah sesi dan variasi kondisi penyimpanan (suhu, jenis potongan daging)"],
    ["Label berbasis laboratorium", "uji TVB-N atau Total Plate Count sebagai ground truth"],
    ["Penerapan pada data skripsi", "16 batch data sensor MQ-135 dan DHT22"],
    ["Reinforcement learning multi-step", "misalnya Q-learning untuk menjadwalkan pemeriksaan selama penyimpanan"],
    ["Deep learning sekuensial", "1D-CNN atau LSTM untuk memanfaatkan pola temporal sensor"],
  ];
  items.forEach(([t, d], i) => {
    const y = 1.3 + i * 1.08;
    square(s, 5.3, y, String.fromCharCode(65 + i), 0.55);
    s.addText([{ text: t, options: { bold: true, color: H.dk2, breakLine: true } }, { text: d }], { isTextBox: true,
      x: 6.1, y: y - 0.05, w: 6.6, h: 0.9, fontSize: 15, color: H.dk1, margin: 0, valign: "top" });
  });
  s.addNotes("[Erlangga] Terima kasih Rijal. Terdapat beberapa celah pengembangan. Pertama, menambah jumlah sesi dan variasi kondisi penyimpanan. Kedua, menggunakan label dari uji laboratorium seperti TVB-N atau Total Plate Count. Ketiga, menerapkan pipeline ini pada 16 batch data skripsi saya yang menggunakan sensor MQ-135 dan DHT22. Keempat, mengembangkan reinforcement learning multi-step, misalnya Q-learning, untuk menjadwalkan pemeriksaan selama penyimpanan. Kelima, menggunakan deep learning sekuensial seperti 1D-CNN atau LSTM.");
}

// ================================================================== 18. conclusion
{
  const s = pres.addSlide({ masterName: "PLAIN", sectionTitle: S4 });
  s.addText("Simpulan", { isTextBox: true, x: M, y: 2.9, w: 3.6, h: 1.3, fontFace: "Cambria", fontSize: 48, bold: true,
    color: H.dk2, margin: 0, valign: "middle" });
  speaker(s, "Erlangga");
  const k = [
    ["Skema validasi menentukan keandalan hasil.", `LOSO memberi estimasi realistis (${pct(leak.loso.akurasi)}); random split terdampak data leakage (${pct(leak.acak.akurasi)}).`],
    ["Data wrangling memberi peningkatan terbesar.", `Fitur relatif terhadap baseline menaikkan F1-Macro ${fmt(wr[0])} → ${fmt(wr[2])}; augmentasi sintetis tidak meningkatkan performa.`],
    ["Ensemble learning mengungguli model tunggal.", `Rata-rata F1-Macro ${fmt(mean(ensemble.map((m) => m.macro_f1)))} vs ${fmt(mean(single.map((m) => m.macro_f1)))}; terbaik ${bestEns.model} (${fmt(bestEns.macro_f1)}).`],
    ["Reinforcement learning menghasilkan keputusan yang lebih aman.", `Reward ${fmt(rl.reward_start, 2)} → ${fmt(rl.reward_final, 2)} seiring iterasi; ayam busuk terjual ${imp.busuk_terjual_ikut_model} → ${imp.busuk_terjual_agen}. Tambahan sesi latih menaikkan F1-Macro ${fmt(sc[0].mean, 2)} → ${fmt(sc[sc.length - 1].mean, 2)}.`],
  ];
  k.forEach(([t, d], i) => {
    const y = 1.2 + i * 1.38;
    chip(s, 4.6, y + 0.05, String(i + 1), H.accent1, H.lt1, 0.55);
    s.addText([{ text: t, options: { bold: true, color: H.dk2, breakLine: true, fontSize: 17 } }, { text: d }],
      { isTextBox: true, x: 5.4, y, w: 7.3, h: 1.2, fontSize: 14, color: H.dk1, margin: 0, valign: "top" });
    if (i < k.length - 1) s.addShape(pres.shapes.LINE, { x: 5.4, y: y + 1.27, w: 7.3, h: 0, line: { color: LINE, width: 1 } });
  });
  s.addNotes(`[Erlangga] Sebagai simpulan. Pertama, skema validasi menentukan keandalan hasil: LOSO memberi estimasi yang realistis sebesar ${pct(leak.loso.akurasi)}, sedangkan random split terdampak data leakage. Kedua, data wrangling memberi peningkatan terbesar: fitur relatif terhadap baseline menaikkan F1-macro dari ${fmt(wr[0])} menjadi ${fmt(wr[2])}, sedangkan augmentasi sintetis tidak meningkatkan performa. Ketiga, ensemble learning mengungguli model tunggal, dengan ${bestEns.model} sebagai model terbaik. Keempat, reinforcement learning menghasilkan keputusan yang lebih aman: reward naik seiring iterasi, ayam busuk yang terjual turun dari ${imp.busuk_terjual_ikut_model} menjadi nol, dan penambahan sesi latih menaikkan performa model.`);
}

// ================================================================== 19. references
{
  const s = content("Daftar Pustaka", S4);
  const refs = [
    "[1] Anwar, H., & Anwar, T. (2025). Quality assessment of chicken using machine learning and electronic nose. Sensing and Bio-Sensing Research, 47, 100739. https://doi.org/10.1016/j.sbsr.2025.100739",
    "[2] Zhou, Z.-H. (2012). Ensemble Methods: Foundations and Algorithms. CRC Press.",
    "[3] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
    "[4] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.",
    "[5] Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.",
    "Materi kuliah Machine Learning II: Ensemble_Learning.pptx dan reinforcement.pptx (github.com/jendralhxr/machinelearning).",
    "Data dan kode penulis: github.com/talhaanwarch/chicken_assessment_e-nose (commit e7e3ca6).",
  ];
  s.addText(refs.map((t, i) => ({ text: t, options: { breakLine: i < refs.length - 1 } })),
    { isTextBox: true, x: M, y: 1.35, w: CW, h: 5.3, fontSize: 14, color: H.dk1, margin: 0, paraSpaceAfter: 9,
      valign: "top" });
  s.addNotes("[Erlangga] Referensi yang kami gunakan tercantum pada slide ini.");
}

// ================================================================== 20. thank you
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: S4 });
  label(s, "Machine Learning II", M, 0.85, 6, H.accent2);
  s.addText("TERIMA KASIH", { isTextBox: true, x: M, y: 2.2, w: CW, h: 1.5, fontFace: "Cambria", fontSize: 66,
    bold: true, color: H.lt1, margin: 0, align: "center", valign: "middle" });
  s.addText("Sesi tanya jawab", { isTextBox: true, x: M, y: 3.75, w: CW, h: 0.5, fontSize: 20, italic: true,
    color: LIGHT, margin: 0, align: "center" });
  s.addText([
    { text: "Kode dan notebook: ", options: { color: LIGHT } },
    { text: "github.com/MuhammadErlanggaKurniawan/ets-ml2-kelompok", options: { bold: true } },
  ], { isTextBox: true, x: M, y: 4.7, w: CW, h: 0.45, fontSize: 15, color: H.lt1, margin: 0, align: "center" });
  s.addShape(pres.shapes.LINE, { x: M, y: 6.4, w: CW, h: 0, line: { color: "3E6466", width: 1 } });
  s.addText("Muhammad Erlangga Kurniawan · Michael Valencio Erasmo Bagas Ganendra · Rijal Apuila Khufi",
    { isTextBox: true, x: M, y: 6.55, w: 10, h: 0.35, fontSize: 12, color: "9FB8B3", margin: 0 });
  s.addNotes("[Erlangga] Demikian presentasi dari kelompok kami. Seluruh kode dan notebook dapat diakses di GitHub. Terima kasih, kami persilakan Bapak/Ibu dan teman-teman untuk bertanya.");
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

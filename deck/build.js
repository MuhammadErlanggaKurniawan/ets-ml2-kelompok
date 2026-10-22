// Build the ETS deck from hasil/ringkasan.json and the notebook figures:  node deck/build.js
// 12 slides for a 15-minute talk, ordered by the lecturer's rubric. Each content slide names its speaker.
// Wording uses standard ML terms (data leakage, LOSO, segmen, baseline); structure matches the team briefing.
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

const W = 13.333, M = 0.6;
const GREY = "56676A", LIGHT = "CADBD7";
const fmt = (x, n = 3) => Number(x).toFixed(n).replace(".", ",");
const pct = (x) => `${(100 * x).toFixed(1).replace(".", ",")}%`;
const num = (x) => Number(x).toLocaleString("id-ID");

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
    { text: { text: "ETS Machine Learning II · Klasifikasi Kesegaran Daging Ayam",
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
// Charts are the notebook's PNGs: Google Slides drops native chart category labels on import.
function img(s, name, x, y, w, h) {
  const file = path.join(ROOT, "hasil", `${name}.png`);
  const buf = fs.readFileSync(file);
  const pw = buf.readUInt32BE(16), ph = buf.readUInt32BE(20); // PNG IHDR
  const scale = Math.min(w / pw, h / ph);
  const iw = pw * scale, ih = ph * scale;
  s.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih });
}
function bullets(items) {
  return items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } }));
}

// ------------------------------------------------------------------ numbers
const model = Object.fromEntries(R.models.map((m) => [m.model, m]));
const single = R.models.filter((m) => m.jenis === "tunggal");
const ensemble = R.models.filter((m) => m.jenis === "ensemble");
const bestSingle = single.reduce((a, b) => (b.macro_f1 > a.macro_f1 ? b : a));
const bestEns = ensemble.reduce((a, b) => (b.macro_f1 > a.macro_f1 ? b : a));
const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
const wr = Object.values(R.wrangling);
const aug = (m, k) => R.augmentation.find((r) => r.model === m && r.augmentasi === k).macro_f1;
const rl = R.rl;
const sc = R.session_curve;
const leak = R.leakage;
const cc = R.class_counts;
const sensorRank = Object.keys(R.sensor_importance);
const mq135Rank = sensorRank.indexOf("MQ-135") + 1;

const S1 = "Erlangga: latar belakang, data, wrangling";
const S2 = "Michael: validasi, ensemble, augmentasi";
const S3 = "Rijal: reinforcement, analisis, simpulan";

// ================================================================== 1. title
pres.addSection({ title: S1 });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: S1 });
  label(s, "Machine Learning II · Evaluasi Tengah Semester", M, 0.95, 9, C.accent2);
  s.addText([
    { text: "Klasifikasi Kesegaran Daging Ayam", options: { breakLine: true } },
    { text: "Berbasis Ensemble dan", options: { breakLine: true } },
    { text: "Reinforcement Learning" },
  ], { isTextBox: true, x: M, y: 1.35, w: 11.6, h: 2.3, fontFace: "Cambria", fontSize: 40, bold: true,
    color: C.background1, margin: 0, valign: "top" });
  s.addText("Studi kasus data electronic nose · tiga aspek ML II: data augmentation, ensemble learning, dan reinforcement learning",
    { isTextBox: true, x: M, y: 3.8, w: 10.5, h: 0.8, fontSize: 19, color: LIGHT, margin: 0 });
  s.addText([
    { text: "Muhammad Erlangga Kurniawan · 23083010045", options: { breakLine: true } },
    { text: "Michael Valencio Erasmo Bagas Ganendra · 23083010042", options: { breakLine: true } },
    { text: "Rijal Apuila Khufi · 23083010033" },
  ], { isTextBox: true, x: M, y: 4.95, w: 8, h: 1.2, fontSize: 15, color: C.background1, margin: 0,
    paraSpaceAfter: 4 });
  s.addText("Program Studi Sains Data · UPN \"Veteran\" Jawa Timur · Oktober 2026",
    { isTextBox: true, x: M, y: 6.55, w: 11, h: 0.4, fontSize: 12, color: "9FB8B3", margin: 0 });
  s.addNotes(`[Erlangga] Selamat pagi. Kami kelompok Erlangga, Michael, dan Rijal. Judul proyek kami: ${TITLE}. Kami menggunakan satu dataset dan tiga aspek Machine Learning II: data augmentation, ensemble learning, dan reinforcement learning. Saya membuka dengan latar belakang, dataset, dan data wrangling. Michael menjelaskan skema validasi, ensemble learning, dan data augmentation. Rijal menjelaskan reinforcement learning, hasil dan analisis, serta simpulan.`);
}

// ================================================================== 2. background
{
  const s = content("Latar Belakang", S1, "Erlangga");
  const lw = 6.3;
  card(s, M, 1.4, lw, 4.15, C.text2);
  label(s, "Permasalahan", M + 0.35, 1.65, lw - 0.7, C.accent2);
  const flow = [
    ["Daging ayam yang membusuk menghasilkan senyawa volatil", "seperti amonia (NH₃) dan senyawa organik volatil (VOC)"],
    ["Sensor gas seri MQ dapat mendeteksi senyawa tersebut", "rangkaian beberapa sensor gas disebut electronic nose (e-nose)"],
    ["Dapatkah machine learning mengklasifikasikan kesegaran ayam?", "Fresh · Semi-fresh · Spoiled"],
  ];
  flow.forEach((f, i) => {
    const y = 2.1 + i * 1.1;
    chip(s, M + 0.35, y + 0.05, String(i + 1), C.accent2, C.text2, 0.48);
    s.addText(f[0], { isTextBox: true, x: M + 1.0, y, w: lw - 1.3, h: 0.4, fontSize: 16, bold: true,
      color: C.background1, margin: 0 });
    s.addText(f[1], { isTextBox: true, x: M + 1.0, y: y + 0.42, w: lw - 1.3, h: 0.4, fontSize: 13, color: LIGHT,
      margin: 0 });
  });
  const x = M + lw + 0.3, w = W - M - x;
  label(s, "Irisan skripsi anggota", x, 1.45, w);
  const theses = [
    ["Muhammad Erlangga Kurniawan", "Kesegaran makanan dengan sensor MQ-135, sensor yang juga ada di dataset ini."],
    ["Rijal Apuila Khufi", "Mutu produk unggas (kuning telur puyuh)."],
    ["Michael Valencio Erasmo Bagas Ganendra", "Model gradient boosting (XGBoost), dipakai pada bagian ensemble."],
  ];
  theses.forEach((t, i) => {
    const y = 1.85 + i * 1.25;
    card(s, x, y, w, 1.1);
    s.addText(t[0], { isTextBox: true, x: x + 0.3, y: y + 0.12, w: w - 0.6, h: 0.32, fontSize: 15, bold: true,
      color: C.accent1, margin: 0 });
    s.addText(t[1], { isTextBox: true, x: x + 0.3, y: y + 0.45, w: w - 0.6, h: 0.6, fontSize: 13, color: C.text1,
      margin: 0, valign: "top" });
  });
  bar(s, [
    { text: "Manfaat: ", options: { bold: true, color: C.accent2 } },
    { text: "pengujian kesegaran yang cepat, murah, dan non-destruktif, misalnya di toko atau rantai distribusi." },
  ], 5.85, 0.8);
  s.addNotes("[Erlangga] Daging ayam yang mulai busuk menghasilkan senyawa volatil seperti amonia. Sensor gas seri MQ dapat mendeteksi senyawa tersebut, dan rangkaian beberapa sensor gas disebut electronic nose atau e-nose. Pertanyaan kami: dari data sensor saja, dapatkah machine learning mengklasifikasikan daging ayam sebagai Fresh, Semi-fresh, atau Spoiled? Topik ini merupakan irisan skripsi kami: saya meneliti kesegaran makanan dengan sensor MQ-135, Rijal meneliti mutu produk unggas yaitu kuning telur puyuh, dan Michael menggunakan gradient boosting atau XGBoost yang kami pakai pada bagian ensemble. Manfaatnya: pengujian kesegaran yang cepat, murah, dan tidak merusak sampel.");
}

// ================================================================== 3. data + pipeline
{
  const s = content("Dataset dan Alur Kerja", S1, "Erlangga");
  const steps = [
    ["Akuisisi", "Unduh data, verifikasi checksum SHA-256", null],
    ["Wrangling", "Window 5 menit, warm-up, 19 fitur", "Augmentation"],
    ["Pemodelan", "Model tunggal vs ensemble", "Ensemble"],
    ["Validasi", "LOSO: diuji pada sesi yang tidak dipakai saat latih", null],
    ["Keputusan", "Agen memilih: jual, cek manual, buang", "Reinforcement"],
  ];
  const gap = 0.35, bw = (W - 2 * M - 4 * gap) / 5, y = 1.4, bh = 2.25;
  steps.forEach((st, i) => {
    const x = M + i * (bw + gap);
    card(s, x, y, bw, bh);
    chip(s, x + 0.2, y + 0.2, String(i + 1), C.accent1, C.background1, 0.45);
    s.addText(st[0], { isTextBox: true, x: x + 0.75, y: y + 0.2, w: bw - 0.9, h: 0.45, fontSize: 16, bold: true,
      color: C.text2, margin: 0, valign: "middle" });
    s.addText(st[1], { isTextBox: true, x: x + 0.2, y: y + 0.8, w: bw - 0.4, h: 0.85, fontSize: 13, color: C.text1,
      margin: 0 });
    if (st[2]) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.15, y: y + bh - 0.55, w: bw - 0.3, h: 0.4, rectRadius: 0.2,
        fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 } });
      s.addText(st[2], { isTextBox: true, x: x + 0.15, y: y + bh - 0.55, w: bw - 0.3, h: 0.4, align: "center",
        valign: "middle", fontSize: 12, bold: true, color: C.text2, margin: 0 });
    }
    if (i < 4) s.addShape(pres.shapes.LINE, { x: x + bw + 0.05, y: y + bh / 2, w: gap - 0.1, h: 0,
      line: { color: H.accent1, width: 2, endArrowType: "triangle" } });
  });
  const dy = 3.9, dh = 2.55;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: dy, w: 7.4, h: dh, rectRadius: 0.12,
    fill: { color: C.background1 }, line: { color: C.accent1, width: 1.25 } });
  s.addText("E-nose daging ayam", { isTextBox: true, x: M + 0.3, y: dy + 0.2, w: 6.8, h: 0.4, fontFace: "Cambria",
    fontSize: 19, bold: true, color: C.text2, margin: 0 });
  s.addText("Anwar & Anwar (2025), Sensing and Bio-Sensing Research 47, 100739", { isTextBox: true, x: M + 0.3,
    y: dy + 0.62, w: 6.8, h: 0.3, fontSize: 12, color: GREY, margin: 0 });
  s.addText(bullets([
    `${R.n_sessions} sesi rekaman, ${num(R.n_readings)} baris data`,
    "7 sensor: MQ-135, MQ-2, MQ-3, MQ-6, MQ-7, MQ-9, kelembapan",
    "Label dari lama penyimpanan: Fresh · Semi-fresh · Spoiled",
  ]), { isTextBox: true, x: M + 0.3, y: dy + 1.05, w: 6.8, h: 1.4, fontSize: 14, color: C.text1, margin: 0,
    paraSpaceAfter: 4, valign: "top" });
  const x2 = M + 7.7, w2 = W - M - x2;
  card(s, x2, dy, w2, dh);
  label(s, "Distribusi kelas", x2 + 0.3, dy + 0.2, w2 - 0.6);
  [["Fresh", cc.Fresh, "13866B"], ["Semi-fresh", cc["Semi-fresh"], H.accent2], ["Spoiled", cc.Spoiled, H.accent4]]
    .forEach((c, i) => {
      const yy = dy + 0.65 + i * 0.45;
      s.addText(c[0], { isTextBox: true, x: x2 + 0.3, y: yy, w: 1.6, h: 0.38, fontSize: 14, color: C.text1, margin: 0,
        valign: "middle" });
      s.addText(num(c[1]), { isTextBox: true, x: x2 + 1.9, y: yy, w: 1.2, h: 0.38, fontSize: 15, bold: true,
        color: c[2], margin: 0, valign: "middle" });
    });
  s.addText(`Seimbang (rasio ${fmt(R.class_ratio, 2)}), sehingga SMOTE tidak diperlukan.`, { isTextBox: true,
    x: x2 + 0.3, y: dy + 2.0, w: w2 - 0.6, h: 0.45, fontSize: 12.5, color: GREY, margin: 0 });
  s.addNotes(`[Erlangga] Alur kerja kami terdiri dari lima tahap. Akuisisi: data diunduh dari repositori penulis dan diverifikasi dengan SHA-256. Data wrangling: data diringkas dan diubah menjadi fitur. Pemodelan: kami membandingkan model tunggal dengan ensemble. Validasi: model selalu diuji pada sesi yang tidak dipakai saat pelatihan, yaitu Leave-One-Session-Out. Terakhir, agen reinforcement learning memilih keputusan. Datanya satu dataset: e-nose daging ayam dari Anwar dan Anwar 2025, ${R.n_sessions} sesi rekaman dengan ${num(R.n_readings)} baris dari tujuh sensor. Labelnya dibuat dari lama penyimpanan. Distribusi kelasnya seimbang, sehingga SMOTE tidak diperlukan.`);
}

// ================================================================== 4. wrangling
{
  const s = content("Data Wrangling", S1, "Erlangga");
  const lw = 6.2;
  const steps = [
    ["Agregasi window 5 menit", `Sensor membaca ±12 kali per menit. ${num(R.n_readings)} baris → ${num(R.n_windows_all)} segmen.`],
    ["Penghapusan periode warm-up (2 jam)", `Sensor perlu waktu stabil. Rata-rata jam ke-1 menjadi baseline; tersisa ${num(R.n_windows)} segmen.`],
    [`Rekayasa fitur (${R.n_features} fitur)`, "Log-level gas · proporsi relatif antar sensor · perubahan terhadap baseline (analog Rs/R0) · kelembapan."],
  ];
  steps.forEach((st, i) => {
    const y = 1.45 + i * 1.4;
    chip(s, M, y + 0.05, String(i + 1));
    s.addText(st[0], { isTextBox: true, x: M + 0.7, y, w: lw - 0.7, h: 0.4, fontSize: 17, bold: true, color: C.text2,
      margin: 0 });
    s.addText(st[1], { isTextBox: true, x: M + 0.7, y: y + 0.42, w: lw - 0.7, h: 0.85, fontSize: 14, color: C.text1,
      margin: 0, valign: "top" });
  });
  img(s, "05_efek_wrangling", M + lw + 0.2, 1.3, W - 2 * M - lw - 0.2, 4.4);
  bar(s, [
    { text: "Mengapa waktu tidak dijadikan fitur? ", options: { bold: true, color: C.accent2 } },
    { text: "Label dibuat dari waktu; jika dipakai, terjadi data leakage (model mempelajari lama simpan, bukan pola gas)." },
  ], 6.0, 0.75);
  s.addNotes(`[Erlangga] Data wrangling terdiri dari tiga langkah. Pertama, sensor membaca sekitar 12 kali per menit, sehingga kami melakukan agregasi per 5 menit: ${num(R.n_readings)} baris menjadi ${num(R.n_windows_all)} segmen. Kedua, dua jam pertama merupakan periode warm-up karena sensor perlu stabil, dan rata-rata jam pertama dipakai sebagai baseline. Ketiga, kami membuat ${R.n_features} fitur: log-level gas, proporsi relatif antar sensor, dan perubahan terhadap baseline, yang analog dengan Rs per R0 di skripsi saya. Grafik di kanan menunjukkan efeknya: F1-macro Random Forest naik dari ${fmt(wr[0])} menjadi ${fmt(wr[2])}. Satu hal penting: variabel waktu tidak kami jadikan fitur, karena label dibuat dari waktu; jika dipakai akan terjadi data leakage. Selanjutnya Michael.`);
}

// ================================================================== 5. validation (Michael)
pres.addSection({ title: S2 });
{
  const s = content("Skema Validasi: Random Split vs LOSO", S2, "Michael");
  card(s, M, 1.45, 6.4, 3.4);
  label(s, "Akurasi Random Forest", M + 0.35, 1.65, 5.8);
  stat(s, M + 0.35, 2.1, 2.9, pct(leak.acak.akurasi), "random split", C.accent4);
  stat(s, M + 3.3, 2.1, 2.9, pct(leak.loso.akurasi), "Leave-One-Session-Out", C.accent1);
  s.addText("overestimate (data leakage)", { isTextBox: true, x: M + 0.35, y: 3.8, w: 2.6, h: 0.35, fontSize: 13, italic: true,
    color: C.accent4, margin: 0 });
  s.addText("estimasi realistis", { isTextBox: true, x: M + 3.3, y: 3.8, w: 2.6, h: 0.35, fontSize: 13, italic: true,
    color: C.accent1, margin: 0 });
  const x = M + 6.7, w = W - M - x;
  card(s, x, 1.45, w, 3.4, C.text2);
  label(s, "Prosedur Leave-One-Session-Out", x + 0.3, 1.65, w - 0.6, C.accent2);
  s.addText(bullets(["Satu sesi dijadikan data uji", "Model dilatih dengan 14 sesi lainnya",
    "Diulang 15 kali hingga setiap sesi pernah diuji"]), { isTextBox: true, x: x + 0.3, y: 2.1, w: w - 0.6, h: 1.4,
    fontSize: 15, color: C.background1, margin: 0, paraSpaceAfter: 6, valign: "top" });
  s.addText("Sesuai kondisi penggunaan: alat diterapkan pada sampel ayam baru.", { isTextBox: true, x: x + 0.3, y: 3.75,
    w: w - 0.6, h: 0.8, fontSize: 13, color: LIGHT, margin: 0 });
  text(s, [
    { text: "Alasan: ", options: { bold: true } },
    { text: "segmen 5 menit yang berurutan sangat mirip. Pada random split, segmen dari sesi yang sama masuk ke data latih dan data uji sekaligus (data leakage)." },
  ], { x: M, y: 5.05, w: W - 2 * M, h: 0.75, fontSize: 15 });
  bar(s, [
    { text: "Temuan serupa pada paper asli: ", options: { bold: true, color: C.accent2 } },
    { text: "Random Forest 100% dengan random split, tetapi 69% dengan pembagian non-acak." },
  ], 6.0, 0.75);
  s.addNotes(`[Michael] Terima kasih Erlangga. Sebelum membandingkan model, kami menentukan skema validasi. Dengan random split, Random Forest mendapat akurasi ${pct(leak.acak.akurasi)}, hampir sempurna. Namun jika diuji pada sesi yang tidak dipakai saat pelatihan, akurasinya ${pct(leak.loso.akurasi)}. Penyebabnya, segmen 5 menit yang berurutan isinya hampir sama, sehingga pada random split segmen dari sesi yang sama masuk ke data latih dan data uji sekaligus. Kondisi ini disebut data leakage. Paper aslinya menemukan hal serupa: 100 persen dengan random split dan 69 persen dengan pembagian non-acak. Karena itu kami memakai Leave-One-Session-Out: satu sesi dijadikan data uji, model dilatih dengan 14 sesi lainnya, dan diulang 15 kali.`);
}

// ================================================================== 6. ensemble
{
  const s = content("Ensemble Learning", S2, "Michael");
  img(s, "04_ensemble", M, 1.3, 7.6, 4.5);
  const x = M + 7.85, w = W - M - x;
  card(s, x, 1.35, w, 2.55);
  label(s, "Jenis ensemble", x + 0.3, 1.5, w - 0.6);
  s.addText([
    { text: "Random Forest (bagging): ", options: { bold: true } },
    { text: "300 pohon, lalu voting", options: { breakLine: true } },
    { text: "XGBoost (boosting): ", options: { bold: true } },
    { text: "pohon dibuat berurutan, tiap pohon memperbaiki kesalahan pohon sebelumnya", options: { breakLine: true } },
    { text: "Soft Voting: ", options: { bold: true } },
    { text: "rata-rata probabilitas dari 3 model berbeda" },
  ], { isTextBox: true, x: x + 0.3, y: 1.9, w: w - 0.6, h: 1.95, fontSize: 13, color: C.text1, margin: 0,
    paraSpaceAfter: 5, valign: "top" });
  card(s, x, 4.05, w, 1.75, C.text2);
  s.addText([
    { text: `${fmt(model["Random Forest"].macro_f1)} vs ${fmt(model["Decision Tree"].macro_f1)}`,
      options: { bold: true, color: C.accent2, fontSize: 24, fontFace: "Cambria", breakLine: true } },
    { text: "Random Forest vs Decision Tree. Kedalaman pohon sama (6); perbedaannya hanya jumlah pohon." },
  ], { isTextBox: true, x: x + 0.3, y: 4.15, w: w - 0.6, h: 1.55, fontSize: 13, color: C.background1, margin: 0,
    valign: "middle" });
  bar(s, [
    { text: "Rata-rata F1-macro: ", options: { bold: true, color: C.accent2 } },
    { text: `ensemble ${fmt(mean(ensemble.map((m) => m.macro_f1)))} vs model tunggal ${fmt(mean(single.map((m) => m.macro_f1)))}. Terbaik: ${bestEns.model} ${fmt(bestEns.macro_f1)} (akurasi ${pct(bestEns.akurasi)}).` },
  ], 6.0, 0.75);
  s.addNotes(`[Michael] Ini bagian ensemble learning, yaitu penggabungan beberapa model agar prediksi lebih akurat dan stabil. Kami membandingkan tiga model tunggal, yaitu Logistic Regression, KNN, dan Decision Tree, dengan tiga ensemble. Random Forest menggabungkan 300 pohon lalu voting. XGBoost membangun pohon secara berurutan, dan setiap pohon memperbaiki kesalahan pohon sebelumnya. Soft Voting merata-ratakan probabilitas dari tiga model berbeda. Hasilnya, ketiga ensemble berada di atas seluruh model tunggal. Terbaik Random Forest dengan F1-macro ${fmt(bestEns.macro_f1)}, sedangkan model tunggal terbaik ${bestSingle.model} ${fmt(bestSingle.macro_f1)}. Perbandingan paling adil adalah Random Forest ${fmt(model["Random Forest"].macro_f1)} dan Decision Tree ${fmt(model["Decision Tree"].macro_f1)}, dengan kedalaman pohon yang sama; perbedaannya hanya satu pohon dibanding 300 pohon. Hyperparameter seluruh model ditetapkan sejak awal dan tidak di-tuning menggunakan data uji.`);
}

// ================================================================== 7. augmentation
{
  const s = content("Data Augmentation", S2, "Michael");
  const lw = 5.6;
  card(s, M, 1.35, lw, 4.45);
  const blocks = [
    ["Tujuan", ["Data hanya 15 sesi; sensor pada alat lain dapat berbeda sensitivitas dan noise-nya."]],
    ["Metode", ["Gaussian noise jittering (σ = 5% simpangan baku)", "Simulasi variasi sensitivitas sensor (faktor ±30% per sesi)",
      "2 salinan sintetis per sesi, hanya pada data latih"]],
  ];
  const runs = [];
  blocks.forEach((b, i) => {
    runs.push({ text: b[0], options: { bold: true, color: C.accent1, fontSize: 15, breakLine: true } });
    b[1].forEach((t, j) => runs.push({ text: t, options: { bullet: true,
      breakLine: !(i === blocks.length - 1 && j === b[1].length - 1) } }));
  });
  s.addText(runs, { isTextBox: true, x: M + 0.3, y: 1.55, w: lw - 0.6, h: 4.1, fontSize: 14, color: C.text1, margin: 0,
    paraSpaceAfter: 6, valign: "top" });
  img(s, "07_augmentasi", M + lw + 0.2, 1.3, W - 2 * M - lw - 0.2, 4.5);
  const d = (m, k) => (aug(m, k) - aug(m, "tanpa")).toFixed(3).replace(".", ",").replace(/^(\d)/, "+$1");
  bar(s, [
    { text: "Hasil: ", options: { bold: true, color: C.accent2 } },
    { text: `performa tidak meningkat (Random Forest: jitter ${d("Random Forest", "jitter")}, sensitivitas ${d("Random Forest", "sensitivitas")}). Variasi sintetis belum mewakili variasi nyata antar sesi.` },
  ], 6.0, 0.75);
  s.addNotes(`[Michael] Terakhir dari saya: data augmentation. Data kami hanya 15 sesi, padahal sensor pada alat lain dapat berbeda sensitivitas dan noise-nya. Kami membuat data latih sintetis dengan dua teknik. Gaussian noise jittering menambahkan derau acak kecil. Simulasi variasi sensitivitas mengalikan bacaan tiap sensor sekitar plus minus 30 persen, seolah-olah berasal dari alat lain. Augmentasi hanya diterapkan pada data latih, sedangkan data uji selalu asli. Hasilnya, performa tidak meningkat: Random Forest turun sedikit, dan Decision Tree hanya naik tipis dengan variasi sensitivitas. Artinya, variasi sintetis belum mewakili variasi nyata antar sesi. Rijal akan menunjukkan bahwa penambahan sesi asli jauh lebih efektif. Selanjutnya Rijal.`);
}

// ================================================================== 8. RL agent (Rijal)
pres.addSection({ title: S3 });
{
  const s = content("Reinforcement Learning: Agen Keputusan", S3, "Rijal");
  const lw = 6.3;
  card(s, M, 1.35, lw, 4.45);
  label(s, "Komponen RL", M + 0.3, 1.55, lw - 0.6);
  const rows = [
    ["State", "prediksi Random Forest × keyakinan model: yakin (≥ 0,7) / ragu → 6 state"],
    ["Aksi", "Jual · Cek manual · Buang"],
    ["Reward", "nilai dari tabel di kanan (asumsi berbasis risiko)"],
    ["Metode", "ε-greedy (contextual bandit): 90% exploitation, 10% exploration; nilai aksi = rata-rata reward"],
  ];
  rows.forEach((r, i) => {
    const y = 2.0 + i * 0.92;
    s.addText(r[0], { isTextBox: true, x: M + 0.3, y, w: 1.3, h: 0.8, fontSize: 15, bold: true, color: C.text2,
      margin: 0, valign: "top" });
    s.addText(r[1], { isTextBox: true, x: M + 1.65, y, w: lw - 1.95, h: 0.8, fontSize: 14, color: C.text1, margin: 0,
      valign: "top" });
  });
  const x = M + lw + 0.3, w = W - M - x;
  label(s, "Tabel reward", x, 1.4, w);
  const head = { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13, align: "center" };
  const cell = (t, color = C.text1, bold = false) => ({ text: t, options: { fontSize: 14, color, bold, align: "center" } });
  s.addTable([
    [{ text: "Aksi", options: head }, { text: "Ayam layak", options: head }, { text: "Ayam busuk", options: head }],
    [cell("Jual", C.text1, true), cell("+1"), cell("−5", H.accent4, true)],
    [cell("Cek manual", C.text1, true), cell("+0,5"), cell("+0,5")],
    [cell("Buang", C.text1, true), cell("−1"), cell("+1")],
  ], { x, y: 1.8, w, colW: [w * 0.38, w * 0.31, w * 0.31], rowH: 0.42, margin: 0.05,
    border: { type: "solid", color: "DCE5E2", pt: 0.75 }, valign: "middle" });
  label(s, "Kebijakan hasil pembelajaran", x, 3.75, w);
  const pol = rl.policy;
  s.addText([
    { text: "Prediksi Fresh → ", options: {} }, { text: pol["Fresh (yakin)"], options: { bold: true, color: C.accent1, breakLine: true } },
    { text: "Prediksi Semi-fresh → ", options: {} }, { text: pol["Semi-fresh (yakin)"], options: { bold: true, color: C.accent2, breakLine: true } },
    { text: "Spoiled & model yakin → ", options: {} }, { text: pol["Spoiled (yakin)"], options: { bold: true, color: C.accent4, breakLine: true } },
    { text: "Spoiled & model ragu → ", options: {} }, { text: pol["Spoiled (ragu)"], options: { bold: true, color: C.accent2 } },
  ], { isTextBox: true, x, y: 4.15, w, h: 1.6, fontSize: 14, color: C.text1, margin: 0, paraSpaceAfter: 3,
    valign: "top" });
  bar(s, [
    { text: "Tujuan sistem: ", options: { bold: true, color: C.accent2 } },
    { text: "tidak hanya mengklasifikasikan, tetapi mengambil keputusan yang aman dan efisien." },
  ], 6.0, 0.75);
  s.addNotes(`[Rijal] Terima kasih Michael. Bagian saya reinforcement learning. Tujuan sistem kami tidak hanya mengklasifikasikan, tetapi juga mengambil keputusan: jual, cek manual, atau buang. Agen belajar melalui trial and error. State-nya adalah prediksi Random Forest ditambah tingkat keyakinan model, yakin atau ragu, sehingga ada enam state. Aksinya tiga. Reward-nya dari tabel ini, berdasarkan asumsi risiko: menjual ayam busuk paling berbahaya, sehingga diberi minus lima. Cek manual selalu aman tetapi membutuhkan biaya. Agen memakai epsilon-greedy: 90 persen memilih aksi terbaik, 10 persen mencoba aksi acak, dan nilai setiap aksi diperbarui dengan rata-rata reward, sesuai materi multi-armed bandit. Hasilnya, agen menemukan kebijakan sendiri: prediksi Fresh dijual, Semi-fresh dicek manual, Spoiled yang yakin dibuang, dan Spoiled yang ragu dicek manual.`);
}

// ================================================================== 9. iterations + data
{
  const s = content("Penambahan Iterasi dan Data", S3, "Rijal");
  const cw = (W - 2 * M - 0.33) / 2;
  img(s, "08_rl_kurva", M, 1.3, cw, 3.75);
  img(s, "09_tambah_sesi", M + cw + 0.33, 1.3, cw, 3.75);
  [[M, [
    { text: "Penambahan iterasi: ", options: { bold: true, color: C.accent2 } },
    { text: `reward agen ${fmt(rl.reward_start, 2)} → ${fmt(rl.reward_final, 2)}. Lebih baik dari "selalu ikuti prediksi model" (${fmt(rl.trust_model, 2)}) dan "selalu cek manual" (0,50).` },
  ]], [M + cw + 0.33, [
    { text: "Penambahan data: ", options: { bold: true, color: C.accent2 } },
    { text: `F1-macro ${fmt(sc[0].mean, 2)} (1 sesi) → ${fmt(sc[sc.length - 1].mean, 2)} (14 sesi). Data asli lebih efektif daripada augmentasi sintetis.` },
  ]]].forEach(([x, runs]) => {
    card(s, x, 5.2, cw, 1.4, C.text2);
    s.addText(runs, { isTextBox: true, x: x + 0.3, y: 5.2, w: cw - 0.6, h: 1.4, fontSize: 14, color: C.background1,
      valign: "middle", margin: 0 });
  });
  s.addNotes(`[Rijal] Rubrik reinforcement menanyakan bagaimana penambahan iterasi dan data meningkatkan performa. Grafik kiri: penambahan iterasi. Di awal agen belum memiliki pengetahuan, reward-nya ${fmt(rl.reward_start, 2)}. Setelah 100 iterasi menjadi ${fmt(rl.reward_after_100, 2)}, lalu stabil di ${fmt(rl.reward_final, 2)}. Ini lebih baik dari selalu mengikuti prediksi model (${fmt(rl.trust_model, 2)}) maupun selalu cek manual (0,50). Agen dievaluasi pada sesi yang tidak dipakai untuk belajar. Grafik kanan: penambahan data. Jika Random Forest dilatih dengan 1 sesi, F1-macro-nya ${fmt(sc[0].mean, 2)}. Dengan 14 sesi naik menjadi ${fmt(sc[sc.length - 1].mean, 2)}. Jadi penambahan sesi asli meningkatkan performa, berbeda dengan data sintetis pada bagian augmentasi.`);
}

// ================================================================== 10. discussion
{
  const s = content("Hasil dan Analisis", S3, "Rijal");
  const recall = R.recall_rf;
  const g = [
    ["Batasan", [`Hanya ${R.n_sessions} sesi; performa antar sesi bervariasi`, "Label berdasarkan lama penyimpanan, bukan uji laboratorium", "Nilai reward ditetapkan berdasarkan asumsi"]],
    ["Trade-off", ["Cek manual lebih aman, tetapi membutuhkan biaya", "Ambang keyakinan lebih tinggi → lebih banyak cek manual", "Ensemble lebih akurat, tetapi lebih sulit diinterpretasikan"]],
    ["Sumber galat", [`Semi-fresh paling sulit (recall ${Math.round(recall["Semi-fresh"] * 100)}%), di antara dua kelas`, `Level gas antar sesi sangat berbeda (puncak MQ-135 ${R.peak_min.toFixed(0)}–${R.peak_max.toFixed(0)})`, `Akurasi sesi terburuk ${pct(model["Random Forest"].akurasi_sesi_terburuk)}`]],
    ["Celah pengembangan", ["Penerapan pada 16 batch data skripsi (MQ-135 + DHT22)", "Label dari uji laboratorium (TVB-N, Total Plate Count)", `Sensor paling berpengaruh: ${sensorRank[0]}, ${sensorRank[1]}; MQ-135 peringkat ke-${mq135Rank}`]],
  ];
  const cw = (W - 2 * M - 0.3) / 2, ch = 2.55;
  g.forEach((b, i) => {
    const x = M + (i % 2) * (cw + 0.3), y = 1.35 + Math.floor(i / 2) * (ch + 0.25);
    card(s, x, y, cw, ch);
    s.addText(b[0], { isTextBox: true, x: x + 0.3, y: y + 0.18, w: cw - 0.6, h: 0.4, fontSize: 18, bold: true,
      color: C.accent1, margin: 0 });
    s.addText(bullets(b[1]), { isTextBox: true, x: x + 0.3, y: y + 0.68, w: cw - 0.6, h: ch - 0.85, fontSize: 15,
      color: C.text1, margin: 0, paraSpaceAfter: 5, valign: "top" });
  });
  s.addNotes(`[Rijal] Hasil dan analisis. Batasan: hanya ${R.n_sessions} sesi, label berdasarkan lama penyimpanan bukan uji laboratorium, dan nilai reward merupakan asumsi. Trade-off: cek manual lebih aman tetapi membutuhkan biaya; ambang keyakinan yang lebih tinggi membuat lebih banyak kasus dicek manual; ensemble lebih akurat tetapi lebih sulit diinterpretasikan. Sumber galat: kelas Semi-fresh paling sulit karena berada di antara dua kelas lain, level gas antar sesi sangat berbeda, dan akurasi sesi terburuk hanya ${pct(model["Random Forest"].akurasi_sesi_terburuk)}. Celah pengembangan: pipeline ini dapat diterapkan pada 16 batch data skripsi Erlangga, label sebaiknya dari uji laboratorium, dan sensor ${sensorRank[0]} dan ${sensorRank[1]} paling berpengaruh, sedangkan MQ-135 peringkat ke-${mq135Rank}.`);
}

// ================================================================== 11. conclusion
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: S3 });
  s.addText("Simpulan", { isTextBox: true, x: M, y: 0.6, w: 8, h: 0.9, fontFace: "Cambria", fontSize: 40, bold: true,
    color: C.background1, margin: 0 });
  const k = [
    ["Skema validasi menentukan keandalan hasil", `random split ${pct(leak.acak.akurasi)} (data leakage) vs LOSO ${pct(leak.loso.akurasi)} (realistis)`],
    ["Data wrangling memberi peningkatan terbesar", `fitur relatif terhadap baseline: F1-macro ${fmt(wr[0])} → ${fmt(wr[2])}`],
    ["Ensemble learning mengungguli model tunggal", `${bestEns.model} ${fmt(bestEns.macro_f1)} vs ${bestSingle.model} ${fmt(bestSingle.macro_f1)}`],
    ["Reinforcement learning: keputusan lebih aman", `reward ${fmt(rl.reward_final, 2)} vs ${fmt(rl.trust_model, 2)}; ayam busuk terjual ${rl.impact.busuk_terjual_ikut_model} → ${rl.impact.busuk_terjual_agen}`],
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
    { text: "Notebook (11 bagian):", options: { breakLine: true, color: LIGHT } },
    { text: "ETS_ML2_Kesegaran_Ayam.ipynb", options: { bold: true } },
  ], { isTextBox: true, x: bx + 0.3, y: 2.1, w: bw2 - 0.6, h: 3.8, fontSize: 13, color: C.background1, margin: 0,
    paraSpaceAfter: 4, valign: "top" });
  s.addNotes(`[Rijal] Simpulan kami ada empat. Pertama, skema validasi menentukan keandalan hasil: random split ${pct(leak.acak.akurasi)} terdampak data leakage, sedangkan LOSO ${pct(leak.loso.akurasi)} memberi estimasi yang realistis. Kedua, data wrangling memberi peningkatan terbesar: fitur relatif terhadap baseline menaikkan F1-macro Random Forest dari ${fmt(wr[0])} menjadi ${fmt(wr[2])}. Ketiga, ensemble learning mengungguli model tunggal. Keempat, agen reinforcement learning menghasilkan keputusan yang lebih aman dibanding sekadar mengikuti prediksi model, dan penambahan sesi asli lebih efektif daripada data sintetis. Seluruh kode dan notebook tersedia di GitHub. Terima kasih, kami persilakan untuk bertanya.`);
}

// ================================================================== 12. references (backup)
{
  const s = content("Daftar Pustaka", S3);
  const refs = [
    "Anwar, H., & Anwar, T. (2025). Quality assessment of chicken using machine learning and electronic nose. Sensing and Bio-Sensing Research, 47, 100739. https://doi.org/10.1016/j.sbsr.2025.100739",
    "Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of KDD.",
    "Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.",
    "Materi kuliah Machine Learning II: Ensemble_Learning.pptx, reinforcement.pptx (github.com/jendralhxr/machinelearning).",
    "Data dan kode penulis: github.com/talhaanwarch/chicken_assessment_e-nose (commit e7e3ca6).",
  ];
  s.addText(refs.map((t, i) => ({ text: t, options: { breakLine: i < refs.length - 1 } })),
    { isTextBox: true, x: M, y: 1.35, w: W - 2 * M, h: 5.4, fontSize: 14, color: C.text1, margin: 0, paraSpaceAfter: 8,
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

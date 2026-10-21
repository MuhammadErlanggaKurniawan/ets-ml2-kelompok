// Build the ETS deck from hasil/ringkasan.json and the notebook figures:  node deck/build.js
// 12 slides for a 15-minute talk, ordered by the lecturer's rubric. Each content slide names its speaker.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.join(__dirname, "..");
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "hasil", "ringkasan.json"), "utf8"));
const OUT = path.join(__dirname, "ETS_ML2_Kesegaran_Ayam.pptx");
const TITLE = "Deteksi Kesegaran Daging Ayam dengan Sensor Gas dan Machine Learning";

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
    { text: { text: "ETS Machine Learning II · Deteksi Kesegaran Daging Ayam",
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

const S1 = "Erlangga: latar belakang, data, pengolahan";
const S2 = "Michael: uji jujur, ensemble, augmentasi";
const S3 = "Rijal: reinforcement, diskusi, kesimpulan";

// ================================================================== 1. title
pres.addSection({ title: S1 });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: S1 });
  label(s, "Tugas ETS · Machine Learning II", M, 0.95, 8, C.accent2);
  s.addText([
    { text: "Deteksi Kesegaran Daging Ayam", options: { breakLine: true } },
    { text: "dengan Sensor Gas", options: { breakLine: true } },
    { text: "dan Machine Learning" },
  ], { isTextBox: true, x: M, y: 1.35, w: 11.6, h: 2.3, fontFace: "Cambria", fontSize: 40, bold: true,
    color: C.background1, margin: 0, valign: "top" });
  s.addText("Satu dataset e-nose, tiga teknik ML II: augmentasi data, ensemble learning, dan reinforcement learning",
    { isTextBox: true, x: M, y: 3.8, w: 10.5, h: 0.8, fontSize: 19, color: LIGHT, margin: 0 });
  s.addText([
    { text: "Muhammad Erlangga Kurniawan · 23083010045", options: { breakLine: true } },
    { text: "Michael Valencio Erasmo Bagas Ganendra · 23083010042", options: { breakLine: true } },
    { text: "Rijal Apuila Khufi · 23083010033" },
  ], { isTextBox: true, x: M, y: 4.95, w: 8, h: 1.2, fontSize: 15, color: C.background1, margin: 0,
    paraSpaceAfter: 4 });
  s.addText("Program Studi Sains Data · UPN \"Veteran\" Jawa Timur · Oktober 2026",
    { isTextBox: true, x: M, y: 6.55, w: 11, h: 0.4, fontSize: 12, color: "9FB8B3", margin: 0 });
  s.addNotes(`[Erlangga] Selamat pagi. Kami kelompok Erlangga, Michael, dan Rijal. Judul proyek kami: ${TITLE}. Saya membuka dengan latar belakang, data, dan pengolahan data. Michael menjelaskan cara menguji yang jujur, ensemble, dan augmentasi. Rijal menjelaskan reinforcement learning, diskusi, dan kesimpulan.`);
}

// ================================================================== 2. background
{
  const s = content("Kenapa topik ini?", S1, "Erlangga");
  const lw = 6.3;
  card(s, M, 1.4, lw, 4.15, C.text2);
  label(s, "Masalahnya", M + 0.35, 1.65, lw - 0.7, C.accent2);
  const flow = [
    ["Ayam yang mulai busuk mengeluarkan gas", "misalnya amonia dan senyawa organik yang mudah menguap"],
    ["Sensor gas murah (seri MQ) bisa \"mencium\" gas itu", "rangkaian beberapa sensor = hidung elektronik (e-nose)"],
    ["Bisakah komputer menebak kondisi ayam dari sensor saja?", "Fresh · Semi-fresh · Spoiled"],
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
  label(s, "Irisan skripsi kami bertiga", x, 1.45, w);
  const theses = [
    ["Erlangga", "Kesegaran makanan (termasuk ayam) dengan sensor MQ-135. Sensor yang sama ada di dataset ini."],
    ["Rijal", "Mutu produk unggas: kuning telur puyuh."],
    ["Michael", "Model gradient boosting (XGBoost), dipakai di bagian ensemble."],
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
    { text: "cek kesegaran yang cepat, murah, dan tanpa merusak sampel, misalnya di toko atau kulkas." },
  ], 5.85, 0.8);
  s.addNotes("[Erlangga] Daging ayam yang mulai busuk mengeluarkan gas. Sensor gas murah seri MQ bisa mencium gas itu. Kalau beberapa sensor digabung, namanya hidung elektronik atau e-nose. Pertanyaan kami: dari bacaan sensor saja, bisakah komputer menebak ayam masih Fresh, Semi-fresh, atau sudah Spoiled? Topik ini irisan skripsi kami: saya meneliti kesegaran makanan termasuk ayam dengan sensor MQ-135, Rijal meneliti mutu produk unggas yaitu telur puyuh, dan Michael memakai gradient boosting yang kami pakai di bagian ensemble. Manfaatnya: cek kesegaran yang cepat, murah, dan tanpa merusak sampel.");
}

// ================================================================== 3. data + pipeline
{
  const s = content("Data dan alur kerja", S1, "Erlangga");
  const steps = [
    ["Akuisisi", "Unduh data, cek keaslian file (SHA-256)", null],
    ["Pengolahan", "Rata-rata 5 menit, kalibrasi, 19 fitur", "Augmentasi"],
    ["Pemodelan", "Model tunggal vs gabungan", "Ensemble"],
    ["Uji jujur", "Diuji di sesi ayam yang belum pernah dilihat", null],
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
    `${R.n_sessions} sesi rekaman, ${num(R.n_readings)} bacaan sensor`,
    "7 sensor: MQ-135, MQ-2, MQ-3, MQ-6, MQ-7, MQ-9, kelembapan",
    "Label dari lama penyimpanan: Fresh · Semi-fresh · Spoiled",
  ]), { isTextBox: true, x: M + 0.3, y: dy + 1.05, w: 6.8, h: 1.4, fontSize: 14, color: C.text1, margin: 0,
    paraSpaceAfter: 4, valign: "top" });
  const x2 = M + 7.7, w2 = W - M - x2;
  card(s, x2, dy, w2, dh);
  label(s, "Jumlah data per kelas", x2 + 0.3, dy + 0.2, w2 - 0.6);
  [["Fresh", cc.Fresh, "13866B"], ["Semi-fresh", cc["Semi-fresh"], H.accent2], ["Spoiled", cc.Spoiled, H.accent4]]
    .forEach((c, i) => {
      const yy = dy + 0.65 + i * 0.45;
      s.addText(c[0], { isTextBox: true, x: x2 + 0.3, y: yy, w: 1.6, h: 0.38, fontSize: 14, color: C.text1, margin: 0,
        valign: "middle" });
      s.addText(num(c[1]), { isTextBox: true, x: x2 + 1.9, y: yy, w: 1.2, h: 0.38, fontSize: 15, bold: true,
        color: c[2], margin: 0, valign: "middle" });
    });
  s.addText(`Cukup seimbang (rasio ${fmt(R.class_ratio, 2)}), jadi tidak perlu SMOTE.`, { isTextBox: true,
    x: x2 + 0.3, y: dy + 2.0, w: w2 - 0.6, h: 0.45, fontSize: 12.5, color: GREY, margin: 0 });
  s.addNotes(`[Erlangga] Alur kerja kami ada lima tahap. Akuisisi: data diunduh dari repositori penulis dan dicek keasliannya. Pengolahan: data dirangkum dan diubah menjadi fitur. Pemodelan: kami membandingkan model tunggal dengan ensemble. Uji jujur: model selalu diuji di sesi ayam yang belum pernah ia lihat. Terakhir, agen reinforcement learning memilih keputusan. Datanya satu dataset: e-nose daging ayam dari Anwar dan Anwar 2025, ${R.n_sessions} sesi rekaman dengan ${num(R.n_readings)} bacaan dari tujuh sensor. Labelnya dibuat dari lama penyimpanan. Jumlah tiap kelas cukup seimbang, jadi kami tidak memakai SMOTE.`);
}

// ================================================================== 4. wrangling
{
  const s = content("Pengolahan data (wrangling)", S1, "Erlangga");
  const lw = 6.2;
  const steps = [
    ["Rata-rata per 5 menit", `Sensor membaca ±12 kali per menit. ${num(R.n_readings)} bacaan → ${num(R.n_windows_all)} jendela 5 menit.`],
    ["2 jam pertama = masa kalibrasi", `Sensor butuh waktu stabil. Bacaan jam ke-1 jadi titik pembanding; sisa ${num(R.n_windows)} jendela.`],
    [`${R.n_features} fitur, 3 cara menggambarkan gas`, "Level (seberapa banyak) · porsi tiap sensor (polanya) · kenaikan dibanding jam ke-1 (mirip Rs/R0) · kelembapan."],
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
    { text: "Kenapa waktu tidak jadi fitur? ", options: { bold: true, color: C.accent2 } },
    { text: "Label dibuat dari waktu. Kalau waktu dipakai, model cuma \"membaca jam\", bukan mencium ayam." },
  ], 6.0, 0.75);
  s.addNotes(`[Erlangga] Ada tiga langkah pengolahan. Pertama, sensor membaca sekitar 12 kali per menit, jadi kami rata-ratakan per 5 menit: ${num(R.n_readings)} bacaan menjadi ${num(R.n_windows_all)} jendela. Kedua, dua jam pertama dianggap masa kalibrasi karena sensor perlu stabil, dan bacaan jam pertama dipakai sebagai titik pembanding. Ketiga, kami membuat ${R.n_features} fitur yang menggambarkan gas dengan tiga cara: seberapa banyak, bagaimana porsinya antar sensor, dan seberapa jauh naik dibanding jam pertama. Yang terakhir mirip Rs per R0 di skripsi saya. Grafik di kanan menunjukkan efeknya: Random Forest naik dari ${fmt(wr[0])} ke ${fmt(wr[2])}. Satu hal penting: waktu tidak kami pakai sebagai fitur, karena label dibuat dari waktu. Selanjutnya Michael.`);
}

// ================================================================== 5. honest testing (Michael)
pres.addSection({ title: S2 });
{
  const s = content("Jebakan akurasi 97%", S2, "Michael");
  card(s, M, 1.45, 6.4, 3.4);
  label(s, "Random Forest · akurasi", M + 0.35, 1.65, 5.8);
  stat(s, M + 0.35, 2.1, 2.9, pct(leak.acak.akurasi), "data dibagi acak", C.accent4);
  stat(s, M + 3.3, 2.1, 2.9, pct(leak.loso.akurasi), "diuji di sesi ayam baru", C.accent1);
  s.addText("terlalu bagus", { isTextBox: true, x: M + 0.35, y: 3.8, w: 2.6, h: 0.35, fontSize: 13, italic: true,
    color: C.accent4, margin: 0 });
  s.addText("angka jujur", { isTextBox: true, x: M + 3.3, y: 3.8, w: 2.6, h: 0.35, fontSize: 13, italic: true,
    color: C.accent1, margin: 0 });
  const x = M + 6.7, w = W - M - x;
  card(s, x, 1.45, w, 3.4, C.text2);
  label(s, "Uji jujur: Leave-One-Session-Out", x + 0.3, 1.65, w - 0.6, C.accent2);
  s.addText(bullets(["Sembunyikan 1 sesi sebagai ujian", "Latih model dengan 14 sesi lain",
    "Ulangi 15 kali sampai semua sesi pernah diuji"]), { isTextBox: true, x: x + 0.3, y: 2.1, w: w - 0.6, h: 1.4,
    fontSize: 15, color: C.background1, margin: 0, paraSpaceAfter: 6, valign: "top" });
  s.addText("Meniru kondisi nyata: alat dipakai untuk ayam baru.", { isTextBox: true, x: x + 0.3, y: 3.75,
    w: w - 0.6, h: 0.8, fontSize: 13, color: LIGHT, margin: 0 });
  text(s, [
    { text: "Kenapa? ", options: { bold: true } },
    { text: "Jendela 5 menit yang berurutan isinya hampir sama. Kalau dibagi acak, model cukup \"menghafal\" sesi itu, seperti ujian yang soalnya sudah bocor." },
  ], { x: M, y: 5.05, w: W - 2 * M, h: 0.75, fontSize: 15 });
  bar(s, [
    { text: "Paper aslinya menemukan hal yang sama: ", options: { bold: true, color: C.accent2 } },
    { text: "Random Forest 100% dengan data acak, tetapi 69% tanpa pembagian acak." },
  ], 6.0, 0.75);
  s.addNotes(`[Michael] Terima kasih Erlangga. Sebelum membandingkan model, kami harus memastikan cara mengujinya jujur. Kalau data dibagi acak, Random Forest mendapat akurasi ${pct(leak.acak.akurasi)}, hampir sempurna. Tapi kalau diuji di sesi ayam yang belum pernah dilihat, akurasinya ${pct(leak.loso.akurasi)}. Penyebabnya: jendela 5 menit yang berurutan isinya hampir sama, jadi dengan pembagian acak model cukup menghafal sesi itu, seperti ujian yang soalnya bocor. Paper aslinya menemukan hal yang sama: 100 persen dengan data acak, 69 persen tanpa pembagian acak. Karena itu kami memakai Leave-One-Session-Out: satu sesi disembunyikan sebagai ujian, model dilatih dengan 14 sesi lain, diulang 15 kali.`);
}

// ================================================================== 6. ensemble
{
  const s = content("Ensemble: gabungan model menang", S2, "Michael");
  img(s, "04_ensemble", M, 1.3, 7.6, 4.5);
  const x = M + 7.85, w = W - M - x;
  card(s, x, 1.35, w, 2.55);
  label(s, "Jenis ensemble", x + 0.3, 1.5, w - 0.6);
  s.addText([
    { text: "Random Forest (bagging): ", options: { bold: true } },
    { text: "300 pohon, lalu voting", options: { breakLine: true } },
    { text: "XGBoost (boosting): ", options: { bold: true } },
    { text: "pohon dibuat berurutan, tiap pohon memperbaiki kesalahan pohon sebelumnya", options: { breakLine: true } },
    { text: "Voting: ", options: { bold: true } },
    { text: "rata-rata peluang dari 3 model berbeda" },
  ], { isTextBox: true, x: x + 0.3, y: 1.9, w: w - 0.6, h: 1.95, fontSize: 13, color: C.text1, margin: 0,
    paraSpaceAfter: 5, valign: "top" });
  card(s, x, 4.05, w, 1.75, C.text2);
  s.addText([
    { text: `${fmt(model["Random Forest"].macro_f1)} vs ${fmt(model["Decision Tree"].macro_f1)}`,
      options: { bold: true, color: C.accent2, fontSize: 24, fontFace: "Cambria", breakLine: true } },
    { text: "Random Forest vs Decision Tree. Kedalaman pohonnya sama (6), bedanya cuma jumlah pohon." },
  ], { isTextBox: true, x: x + 0.3, y: 4.15, w: w - 0.6, h: 1.55, fontSize: 13, color: C.background1, margin: 0,
    valign: "middle" });
  bar(s, [
    { text: "Rata-rata macro-F1: ", options: { bold: true, color: C.accent2 } },
    { text: `ensemble ${fmt(mean(ensemble.map((m) => m.macro_f1)))} vs model tunggal ${fmt(mean(single.map((m) => m.macro_f1)))}. Terbaik: ${bestEns.model} ${fmt(bestEns.macro_f1)} (akurasi ${pct(bestEns.akurasi)}).` },
  ], 6.0, 0.75);
  s.addNotes(`[Michael] Ini bagian ensemble. Idenya seperti minta pendapat banyak orang sebelum memutuskan. Kami membandingkan tiga model tunggal, yaitu Logistic Regression, KNN, dan Decision Tree, dengan tiga ensemble. Random Forest menggabungkan 300 pohon lalu voting. XGBoost membuat pohon berurutan, tiap pohon memperbaiki kesalahan sebelumnya. Voting merata-ratakan peluang dari tiga model berbeda. Hasilnya, ketiga ensemble di atas semua model tunggal. Terbaik Random Forest dengan macro-F1 ${fmt(bestEns.macro_f1)}, sedangkan model tunggal terbaik ${bestSingle.model} ${fmt(bestSingle.macro_f1)}. Yang paling adil dibandingkan: Random Forest ${fmt(model["Random Forest"].macro_f1)} lawan Decision Tree ${fmt(model["Decision Tree"].macro_f1)}, dengan kedalaman pohon yang sama. Bedanya cuma satu pohon versus 300 pohon. Pengaturan semua model ditetapkan dari awal, tidak diutak-atik pakai data uji.`);
}

// ================================================================== 7. augmentation
{
  const s = content("Augmentasi data", S2, "Michael");
  const lw = 5.6;
  card(s, M, 1.35, lw, 4.45);
  const blocks = [
    ["Kenapa perlu?", ["Hanya 15 sesi. Alat sungguhan bisa punya sensor yang lebih/kurang peka dan lebih berisik."]],
    ["Bagaimana?", ["Jitter: tambah derau acak kecil (5%)", "Variasi kepekaan: kali bacaan tiap sensor ±30%, seolah alat lain",
      "2 salinan buatan per sesi, hanya di data latih"]],
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
    { text: `skor tidak naik (Random Forest: jitter ${d("Random Forest", "jitter")}, kepekaan ${d("Random Forest", "kepekaan")}). Data buatan tidak sama dengan sesi ayam asli.` },
  ], 6.0, 0.75);
  s.addNotes(`[Michael] Terakhir dari saya: augmentasi. Kenapa perlu? Kami hanya punya 15 sesi, padahal alat sungguhan bisa punya sensor yang lebih atau kurang peka dan lebih berisik. Jadi kami membuat data latih buatan dengan dua cara. Jitter menambah derau acak kecil. Variasi kepekaan mengalikan bacaan tiap sensor sekitar plus minus 30 persen, seolah-olah dari alat lain. Ini hanya untuk data latih, data uji selalu asli. Hasilnya jujur saja: skor tidak naik. Random Forest malah turun sedikit, Decision Tree hanya naik tipis dengan variasi kepekaan. Pelajarannya: variasi buatan tidak sama dengan perbedaan nyata antar sesi ayam. Rijal akan menunjukkan bahwa sesi asli tambahan jauh lebih berguna. Selanjutnya Rijal.`);
}

// ================================================================== 8. RL agent (Rijal)
pres.addSection({ title: S3 });
{
  const s = content("Reinforcement learning: agen keputusan", S3, "Rijal");
  const lw = 6.3;
  card(s, M, 1.35, lw, 4.45);
  label(s, "Unsur RL", M + 0.3, 1.55, lw - 0.6);
  const rows = [
    ["State", "tebakan Random Forest × model yakin / ragu (6 kemungkinan)"],
    ["Aksi", "Jual · Cek manual · Buang"],
    ["Reward", "poin dari tabel di kanan (asumsi kami)"],
    ["Belajar", "ε-greedy: 90% pilih aksi terbaik sejauh ini, 10% coba acak; nilai aksi = rata-rata reward"],
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
  label(s, "Yang dipelajari agen", x, 3.75, w);
  const pol = rl.policy;
  s.addText([
    { text: "Tebakan Fresh → ", options: {} }, { text: pol["Fresh (yakin)"], options: { bold: true, color: C.accent1, breakLine: true } },
    { text: "Tebakan Semi-fresh → ", options: {} }, { text: pol["Semi-fresh (yakin)"], options: { bold: true, color: C.accent2, breakLine: true } },
    { text: "Spoiled & model yakin → ", options: {} }, { text: pol["Spoiled (yakin)"], options: { bold: true, color: C.accent4, breakLine: true } },
    { text: "Spoiled & model ragu → ", options: {} }, { text: pol["Spoiled (ragu)"], options: { bold: true, color: C.accent2 } },
  ], { isTextBox: true, x, y: 4.15, w, h: 1.6, fontSize: 14, color: C.text1, margin: 0, paraSpaceAfter: 3,
    valign: "top" });
  bar(s, [
    { text: "Tujuan sistem: ", options: { bold: true, color: C.accent2 } },
    { text: "bukan cuma menebak, tapi mengambil keputusan yang aman dan tidak boros." },
  ], 6.0, 0.75);
  s.addNotes(`[Rijal] Terima kasih Michael. Bagian saya reinforcement learning. Tujuan sistem kami bukan cuma menebak kelas, tapi mengambil keputusan: jual, cek manual, atau buang. Agen belajar lewat trial and error. State-nya adalah tebakan Random Forest ditambah apakah model yakin atau ragu, jadi ada enam kemungkinan. Aksinya tiga. Reward-nya dari tabel ini, angkanya asumsi kami: menjual ayam busuk paling berbahaya, jadi minus lima. Cek manual selalu aman tapi ada biayanya. Agen memakai ε-greedy: 90 persen memilih aksi terbaik sejauh ini, 10 persen mencoba acak, dan nilai tiap aksi diperbarui dengan rata-rata reward, seperti materi multi-armed bandit. Hasilnya agen belajar sendiri: tebakan Fresh dijual, Semi-fresh dicek manual, Spoiled yang yakin dibuang, dan Spoiled yang ragu dicek manual.`);
}

// ================================================================== 9. iterations + data
{
  const s = content("Tambah iterasi dan data, performa naik", S3, "Rijal");
  const cw = (W - 2 * M - 0.33) / 2;
  img(s, "08_rl_kurva", M, 1.3, cw, 3.75);
  img(s, "09_tambah_sesi", M + cw + 0.33, 1.3, cw, 3.75);
  [[M, [
    { text: "Tambahan iterasi: ", options: { bold: true, color: C.accent2 } },
    { text: `reward agen ${fmt(rl.reward_start, 2)} → ${fmt(rl.reward_final, 2)}. Lebih baik dari "selalu ikut model" (${fmt(rl.trust_model, 2)}) dan "selalu cek manual" (0,50).` },
  ]], [M + cw + 0.33, [
    { text: "Tambahan data: ", options: { bold: true, color: C.accent2 } },
    { text: `macro-F1 ${fmt(sc[0].mean, 2)} (1 sesi) → ${fmt(sc[sc.length - 1].mean, 2)} (14 sesi). Sesi asli tambahan jauh lebih berguna daripada data buatan.` },
  ]]].forEach(([x, runs]) => {
    card(s, x, 5.2, cw, 1.4, C.text2);
    s.addText(runs, { isTextBox: true, x: x + 0.3, y: 5.2, w: cw - 0.6, h: 1.4, fontSize: 14, color: C.background1,
      valign: "middle", margin: 0 });
  });
  s.addNotes(`[Rijal] Rubrik reinforcement bertanya bagaimana tambahan iterasi dan data meningkatkan performa. Grafik kiri: tambahan iterasi. Di awal agen belum tahu apa-apa, reward-nya ${fmt(rl.reward_start, 2)}. Setelah 100 iterasi sudah ${fmt(rl.reward_after_100, 2)}, lalu stabil di ${fmt(rl.reward_final, 2)}. Ini lebih baik dari selalu ikut tebakan model (${fmt(rl.trust_model, 2)}) maupun selalu cek manual (0,50). Agen dinilai di sesi yang tidak dipakai untuk belajar. Grafik kanan: tambahan data. Kalau Random Forest dilatih dengan 1 sesi, macro-F1-nya ${fmt(sc[0].mean, 2)}. Dengan 14 sesi naik ke ${fmt(sc[sc.length - 1].mean, 2)}. Jadi menambah sesi ayam asli jelas menaikkan performa, berbeda dengan data buatan di bagian augmentasi.`);
}

// ================================================================== 10. discussion
{
  const s = content("Analisis dan diskusi", S3, "Rijal");
  const recall = R.recall_rf;
  const g = [
    ["Batasan", [`Hanya ${R.n_sessions} sesi, hasil antar sesi masih naik-turun`, "Label dari lama penyimpanan, bukan uji lab per sampel", "Reward agen RL adalah asumsi kami"]],
    ["Trade-off", ["Cek manual lebih aman, tapi ada biaya", "Ambang \"yakin\" 0,7: dinaikkan berarti lebih aman, tapi lebih sering cek", "Ensemble lebih akurat, tapi lebih sulit dijelaskan dari 1 pohon"]],
    ["Sumber galat", [`Semi-fresh paling sulit (recall ${Math.round(recall["Semi-fresh"] * 100)}%), letaknya di tengah`, `Level gas tiap sesi beda jauh (puncak MQ-135 ${R.peak_min.toFixed(0)}–${R.peak_max.toFixed(0)})`, `Akurasi sesi terburuk ${pct(model["Random Forest"].akurasi_sesi_terburuk)}`]],
    ["Pengembangan", ["Jalankan di 16 batch data skripsi Erlangga (MQ-135 + DHT22)", "Label dari panel sensori / uji lab", `Sensor paling berpengaruh: ${sensorRank[0]}, ${sensorRank[1]}; MQ-135 urutan ke-${mq135Rank}`]],
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
  s.addNotes(`[Rijal] Analisis dan diskusi. Batasan: hanya ${R.n_sessions} sesi, labelnya dari lama penyimpanan bukan uji lab, dan reward agen adalah asumsi kami. Trade-off: cek manual lebih aman tapi ada biaya; kalau ambang yakin dinaikkan, keputusan lebih aman tapi lebih sering cek manual; ensemble lebih akurat tapi lebih sulit dijelaskan. Sumber galat: kelas Semi-fresh paling sulit karena letaknya di tengah, level gas tiap sesi berbeda jauh, dan sesi terburuk akurasinya hanya ${pct(model["Random Forest"].akurasi_sesi_terburuk)}. Pengembangan: pipeline ini siap dipakai untuk 16 batch data skripsi Erlangga, label sebaiknya dari panel sensori atau uji lab, dan sensor ${sensorRank[0]} dan ${sensorRank[1]} paling berpengaruh, sedangkan MQ-135 urutan ke-${mq135Rank}. Ini masukan untuk pemilihan sensor.`);
}

// ================================================================== 11. conclusion
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: S3 });
  s.addText("Kesimpulan", { isTextBox: true, x: M, y: 0.6, w: 8, h: 0.9, fontFace: "Cambria", fontSize: 40, bold: true,
    color: C.background1, margin: 0 });
  const k = [
    ["Uji yang jujur menentukan segalanya", `${pct(leak.acak.akurasi)} (dibagi acak) itu palsu; ${pct(leak.loso.akurasi)} (per sesi) yang jujur`],
    ["Pengolahan data paling berpengaruh", `kalibrasi per sesi: Random Forest ${fmt(wr[0])} → ${fmt(wr[2])}`],
    ["Ensemble lebih baik dari model tunggal", `${bestEns.model} ${fmt(bestEns.macro_f1)} vs ${bestSingle.model} ${fmt(bestSingle.macro_f1)}`],
    ["Agen RL mengambil keputusan lebih aman", `reward ${fmt(rl.reward_final, 2)} vs ${fmt(rl.trust_model, 2)}; data asli > data buatan`],
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
    { text: "Notebook Colab (11 bagian):", options: { breakLine: true, color: LIGHT } },
    { text: "ETS_ML2_Kesegaran_Ayam.ipynb", options: { bold: true } },
  ], { isTextBox: true, x: bx + 0.3, y: 2.1, w: bw2 - 0.6, h: 3.8, fontSize: 13, color: C.background1, margin: 0,
    paraSpaceAfter: 4, valign: "top" });
  s.addNotes(`[Rijal] Kesimpulan kami ada empat. Pertama, cara menguji menentukan segalanya: ${pct(leak.acak.akurasi)} dengan data acak itu palsu, ${pct(leak.loso.akurasi)} per sesi yang jujur. Kedua, pengolahan data paling berpengaruh: kalibrasi per sesi menaikkan Random Forest dari ${fmt(wr[0])} ke ${fmt(wr[2])}. Ketiga, ensemble lebih baik dari model tunggal. Keempat, agen reinforcement learning mengambil keputusan yang lebih aman daripada sekadar mengikuti tebakan model, dan menambah sesi ayam asli lebih berguna daripada data buatan. Semua kode dan notebook ada di GitHub. Terima kasih, kami siap menerima pertanyaan.`);
}

// ================================================================== 12. references (backup)
{
  const s = content("Lampiran: referensi", S3);
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

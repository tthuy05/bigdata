// Các hàm tiện ích dựng tài liệu Word dạng bài báo khoa học 2 cột (docx-js)
const fs = require("fs");
const path = require("path");
const d = require("docx");
const { Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, ShadingType, HeadingLevel, Math: OMath, MathRun, MathFraction, MathSuperScript,
  MathSubScript, MathSum, MathSubSuperScript, VerticalAlign } = d;

const FONT = "Times New Roman";
const BODY = 21;          // 10,5 pt
const SMALL = 17;         // 8,5 pt
const COL_W_IN = 3.24;    // bề rộng một cột (inch)
const COL_W = 4650;       // bề rộng bảng trong một cột (DXA)

let figNo = 0, tabNo = 0, eqNo = 0;

// ---- định dạng số kiểu Việt Nam: 6.362.620 ; 0,9542
const fmtInt = (n) => Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
const fmtDec = (x, k = 4) => {
  const [i, f] = Number(x).toFixed(k).split(".");
  return fmtInt(Number(i)) + (f ? "," + f : "");
};
const fmtPct = (x, k = 2) => fmtDec(x, k) + "%";

// ---- văn bản có đánh dấu **đậm**, *nghiêng*, `mã`
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|_\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: BODY, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, size: BODY, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, size: BODY, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: BODY - 3, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, size: BODY, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: BODY, ...base }));
  return out;
}

const P = (text, opt = {}) => new Paragraph({
  alignment: opt.align || AlignmentType.JUSTIFIED,
  indent: opt.noIndent ? undefined : { firstLine: 245 },
  spacing: { after: opt.after ?? 50, line: 245 },
  children: runs(text, opt.run || {}),
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1, keepNext: true,
  spacing: { before: 200, after: 100 },
  children: [new TextRun({ text, bold: true, font: FONT, size: 23 })],
});
const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, keepNext: true,
  spacing: { before: 120, after: 60 },
  children: [new TextRun({ text, bold: true, italics: true, font: FONT, size: BODY })],
});

const bullet = (text) => new Paragraph({
  numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 30, line: 245 }, children: runs(text),
});
const numbered = (text, ref = "steps") => new Paragraph({
  numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 30, line: 245 }, children: runs(text),
});

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// ---- hình (căn giữa, rộng bằng cột) + chú thích
function figure(file, caption, widthIn = COL_W_IN) {
  if (!fs.existsSync(file)) return [P(`[Thiếu hình: ${path.basename(file)}]`)];
  figNo += 1;
  const { w, h } = pngSize(file);
  const wpx = widthIn * 96, hpx = wpx * h / w;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 80, after: 30 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
        transformation: { width: Math.round(wpx), height: Math.round(hpx) },
        altText: { title: caption, description: caption, name: path.basename(file) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
      children: [new TextRun({ text: `Hình ${figNo}. `, bold: true, font: FONT, size: SMALL + 1 }),
        new TextRun({ text: caption, italics: true, font: FONT, size: SMALL + 1 })] }),
  ];
}
const nextFig = () => figNo + 1;
const nextTab = () => tabNo + 1;
const nextEq = () => eqNo + 1;

// ---- bảng: chú thích phía trên
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(caption, headers, rows, ratios, opt = {}) {
  tabNo += 1;
  const total = opt.width || COL_W;
  const sum = ratios.reduce((a, b) => a + b, 0);
  const widths = ratios.map((r) => Math.floor(total * r / sum));
  widths[widths.length - 1] += total - widths.reduce((a, b) => a + b, 0);
  const cell = (txt, i, head, shade) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: head ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" }
      : shade ? { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    children: [new Paragraph({
      alignment: (i === 0 && !opt.centerFirst) ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(txt), { size: opt.size || SMALL, bold: head || undefined }),
    })],
  });
  const tbl = new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    alignment: AlignmentType.CENTER,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, k) => new TableRow({ cantSplit: true,
        children: r.map((c, i) => cell(c, i, false, opt.highlight && opt.highlight(k))) }))],
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 100, after: 50 },
      children: [new TextRun({ text: `Bảng ${tabNo}. `, bold: true, font: FONT, size: SMALL + 1 }),
        new TextRun({ text: caption, italics: true, font: FONT, size: SMALL + 1 })] }),
    tbl,
    new Paragraph({ spacing: { after: 80 }, children: [] }),
  ];
}

// ---- đoạn mã nguồn
function code(lines) {
  return lines.map((l, i) => new Paragraph({
    shading: { fill: "F3F3F3", type: ShadingType.CLEAR, color: "auto" },
    spacing: { before: i === 0 ? 60 : 0, after: i === lines.length - 1 ? 100 : 0, line: 220 },
    keepLines: true, keepNext: i < lines.length - 1,
    indent: { left: 60, right: 60 },
    children: [new TextRun({ text: l.length ? l : " ", font: "Consolas", size: 14 })],
  }));
}

// ---- công thức toán (OMML) có đánh số
const mr = (t) => new MathRun(t);
const frac = (n, dd) => new MathFraction({ numerator: n, denominator: dd });
const sup = (b, s) => new MathSuperScript({ children: b, superScript: s });
const sub = (b, s) => new MathSubScript({ children: b, subScript: s });
const subsup = (b, lo, hi) => new MathSubSuperScript({ children: b, subScript: lo, superScript: hi });
const sum = (lo, hi, body) => new MathSum({ children: body, subScript: lo, superScript: hi });
function eq(children) {
  eqNo += 1;
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 60, after: 80 },
    children: [new OMath({ children }), new TextRun({ text: `     (${eqNo})`, font: FONT, size: BODY })],
  });
}

module.exports = { d, FONT, BODY, SMALL, COL_W, COL_W_IN, fmtInt, fmtDec, fmtPct, runs, P, H1, H2,
  bullet, numbered, figure, table, code, eq, mr, frac, sup, sub, subsup, sum, nextFig, nextTab, nextEq };

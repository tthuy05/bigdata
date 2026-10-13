/*
 * Dựng báo cáo Word 2 cột (phong cách bài báo khoa học) từ kết quả thực nghiệm.
 * Mọi con số được đọc từ outputs/metrics/*.json, hình từ outputs/figures/.
  * Thường được gọi tự động ở bước 13 của main.py; chạy tay: cd report && node build_report.js
 * Sửa danh sách thành viên trong report/team.json.
 */
const fs = require("fs");
const { d, FONT } = require("./lib");
const { Document, Packer, Paragraph, TextRun, AlignmentType, SectionType, LevelFormat, Footer, PageNumber } = d;
const p1 = require("./content_part1");
const { methods } = require("./content_part2");
const { results, conclusion, references } = require("./content_part3");

const page = { size: { width: 11906, height: 16838 },
  margin: { top: 1077, bottom: 1077, left: 964, right: 964, header: 567, footer: 567 } };
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] });

const doc = new Document({
  creator: "Nhóm 8", title: "Ứng dụng học máy trong phát hiện gian lận giao dịch tài chính trên nền tảng Apache Spark",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 23, bold: true, color: "000000" }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 21, bold: true, italics: true, color: "000000" }, paragraph: { outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 280, hanging: 200 } } } }] },
    { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 300, hanging: 240 } } } }] },
  ] },
  sections: [
    { properties: { page }, footers: { default: footer }, children: p1.titleBlock },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 397, equalWidth: true } },
      footers: { default: footer },
      children: [...p1.abstract, ...p1.intro, ...p1.theory, ...methods, ...results, ...conclusion, ...references] },
  ],
});

const out = p1.ctx.OUT;
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("Đã tạo", out); });

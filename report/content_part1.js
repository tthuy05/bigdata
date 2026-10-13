/*
 * Dựng báo cáo Word 2 cột (phong cách bài báo khoa học) từ kết quả thực nghiệm.
 * Mọi con số được đọc trực tiếp từ outputs/metrics/*.json và hình từ outputs/figures/.
 * (Phần 1: trang đầu, tóm tắt, mục I, II) – được build_report.js sử dụng
 */
const fs = require("fs");
const path = require("path");
const L = require("./lib");
const { d, FONT, P, H1, H2, bullet, numbered, figure, table, code, eq, mr, frac, sup, sub, sum,
  fmtInt, fmtDec, fmtPct, nextFig, nextTab } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, SectionType, LevelFormat, Footer,
  PageNumber, BorderStyle } = d;

const ROOT = path.join(__dirname, "..");
const MET = path.join(ROOT, "outputs", "metrics");
const FIG = path.join(ROOT, "outputs", "figures");
const OUT = process.argv[2] || path.join(ROOT, "outputs", "BaoCao_BigData_Nhom8.docx");
const J = (f) => JSON.parse(fs.readFileSync(path.join(MET, f), "utf8"));
const q = J("quality.json"), e = J("eda_stats.json"), m = J("model_results.json");
const env = J("env.json");
if (m.schema_version !== 2 || m.status !== "VERIFIED" || m.selection.dataset !== "validation") {
  throw new Error("Chỉ dựng báo cáo từ kết quả schema v2 đã VERIFIED và chọn mô hình trên Validation.");
}
const team = JSON.parse(fs.readFileSync(path.join(__dirname, "team.json"), "utf8"));
const fig = (f) => path.join(FIG, f);

// ======================= số liệu dẫn xuất =======================
const lab = Object.fromEntries(e.label.map((r) => [r.isFraud, r]));
const N = q.rows, N0 = lab[0].n, N1 = lab[1].n, pct1 = lab[1].pct;
const byType = e.by_type;
const tOf = (t) => byType.find((r) => r.type === t) || { n: 0, fraud: 0, fraud_rate_pct: 0 };
const amt = Object.fromEntries(e.amount_by_label.map((r) => [r.isFraud, r]));
const bp = Object.fromEntries(e.balance_patterns.map((r) => [r.isFraud, r]));
const flaggedTotal = e.flagged.filter((r) => r.isFlaggedFraud === 1).reduce((a, r) => a + r.n, 0);
const flaggedTrue = e.flagged.filter((r) => r.isFlaggedFraud === 1 && r.isFraud === 1).reduce((a, r) => a + r.n, 0);
const hours = [...e.by_hour].sort((a, b) => b.fraud_rate_pct - a.fraud_rate_pct);
const topHours = hours.slice(0, 3).map((h) => `${h.hour} giờ`).join(", ");
const busyHours = [...e.by_hour].sort((a, b) => b.n - a.n).slice(0, 2).map((h) => h.hour);
const R = m.results, names = Object.keys(R);
const rank = [...names].sort((a, b) => R[b].f1 - R[a].f1);
const best = m.best_model, B = R[best];
const worst = rank[rank.length - 1];
const byAucpr = [...names].sort((a, b) => R[b].auc_pr - R[a].auc_pr);
const fastest = [...names].sort((a, b) => R[a].train_time_s - R[b].train_time_s)[0];
const slowest = [...names].sort((a, b) => R[b].train_time_s - R[a].train_time_s)[0];
const rb = m.rule_baseline, S = m.split;
const imp = Object.entries(m.feature_importance).sort((a, b) => b[1] - a[1]);
const coef = Object.entries(m.lr_coefficients).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
const VN = { "Logistic Regression": "Hồi quy Logistic", "Decision Tree": "Cây quyết định",
  "Random Forest": "Rừng ngẫu nhiên", "GBT": "Gradient-Boosted Trees" };
const FEAT_VN = {
  type_CASH_OUT: "loại giao dịch (CASH_OUT)", log_amount: "số tiền giao dịch",
  log_oldbalanceOrg: "số dư nguồn trước giao dịch", log_newbalanceOrig: "số dư nguồn sau giao dịch",
  log_oldbalanceDest: "số dư đích trước giao dịch", log_newbalanceDest: "số dư đích sau giao dịch",
  slog_errorBalanceOrig: "sai lệch số dư nguồn", slog_errorBalanceDest: "sai lệch số dư đích",
  origZeroBalance: "cờ số dư nguồn bằng 0", destZeroBalance: "cờ số dư đích bằng 0",
  origEmptied: "cờ rút sạch tài khoản nguồn", hour: "giờ trong ngày" };
const fv = (k) => FEAT_VN[k] || k;
const relShare = 100 * e.relevant_rows / N;
const totalMin = env.total_time_s ? fmtDec(env.total_time_s / 60, 1) : null;
const allFraudTypes = byType.filter((r) => r.fraud > 0).map((r) => r.type);

// ======================= trang đầu (1 cột) =======================
const titleBlock = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP. HỒ CHÍ MINH – KHOA CÔNG NGHỆ THÔNG TIN", font: FONT, size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "Đồ án môn Nhập môn Big Data – " + team.group, font: FONT, size: 18, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "ỨNG DỤNG HỌC MÁY TRONG PHÁT HIỆN GIAN LẬN GIAO DỊCH TÀI CHÍNH TRÊN NỀN TẢNG APACHE SPARK", font: "Arial", size: 30, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
    children: [new TextRun({ text: team.members.map((x) => x.name).join(", "), font: FONT, size: 22, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
    children: [new TextRun({ text: "MSSV: " + team.members.map((x) => x.id).join(", "), font: FONT, size: 19 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
    children: [new TextRun({ text: "Khoa Công nghệ Thông tin, Trường Đại học Công Thương TP. Hồ Chí Minh", font: FONT, size: 19, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 280 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "808080", space: 8 } },
    children: [new TextRun({ text: "Giảng viên hướng dẫn: " + team.advisor, font: FONT, size: 19 })] }),
];

// ======================= ABSTRACT =======================
const abstract = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: "Tóm tắt (Abstract)", bold: true, font: FONT, size: 23 })] }),
  P(`Đề tài xây dựng quy trình phát hiện gian lận giao dịch bằng Apache Spark SQL, DataFrame và MLlib trên CSV PaySim gồm ${fmtInt(N)} giao dịch. Sau làm sạch còn ${fmtInt(q.clean_rows)} giao dịch, gồm ${fmtInt(N1)} gian lận (${fmtDec(pct1, 3)}%). Bốn thuật toán Logistic Regression, Decision Tree, Random Forest và Gradient-Boosted Trees được so sánh trong hai điều kiện đặc trưng: A dùng các số dư và đặc trưng sau giao dịch, B chỉ dùng loại giao dịch, số tiền và giờ mô phỏng. Mỗi điều kiện được chạy với chia phân tầng và chia theo thời gian; dữ liệu dùng mô hình là mẫu đồng đều ${fmtDec(100*m.data.sample_fraction, 1)}% từ CSV thật. Thống kê dữ liệu gốc sử dụng toàn bộ CSV. Mọi tiền xử lý học tham số được fit trên Train; ngưỡng và thuật toán được chọn trên Validation trước khi đánh giá Test. Với thí nghiệm chính ${m.experiment}, mô hình chọn từ Validation là ${best}; trên Test đạt Precision ${fmtDec(B.precision)}, Recall ${fmtDec(B.recall)}, F1 ${fmtDec(B.f1)} và AUC-PR ${fmtDec(B.auc_pr)}. So sánh A/B đánh giá ảnh hưởng của thông tin sau giao dịch và dấu vết mô phỏng; kết quả trên PaySim không chứng minh hiệu quả trong ngân hàng thực tế. Ứng dụng Streamlit phân biệt rõ chế độ trước và sau giao dịch.`, { run: { italics: true } }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
    children: [new TextRun({ text: "Từ khóa: ", bold: true, italics: true, font: FONT, size: L.BODY }),
      new TextRun({ text: "Apache Spark, Spark MLlib, phát hiện gian lận, dữ liệu mất cân bằng, PaySim, Random Forest, Big Data.", italics: true, font: FONT, size: L.BODY })] }),
];

// ======================= I. GIỚI THIỆU =======================
const intro = [
  H1("I. GIỚI THIỆU (INTRODUCTION)"),
  H2("1.1. Bối cảnh đề tài"),
  P("Chuyển đổi số trong lĩnh vực tài chính đã đưa ngân hàng số, ví điện tử, thanh toán bằng mã QR và các dịch vụ tiền di động (mobile money) trở thành kênh giao dịch phổ biến. Người dùng có thể chuyển tiền, rút tiền hoặc thanh toán chỉ trong vài giây, mọi lúc và mọi nơi. Mặt trái của sự tiện lợi này là bề mặt tấn công mở rộng: kẻ gian có thể chiếm đoạt tài khoản qua lừa đảo trực tuyến (phishing), đánh cắp thông tin đăng nhập, sau đó chuyển toàn bộ số dư sang các tài khoản trung gian và rút tiền mặt trước khi nạn nhân kịp phát hiện."),
  P("Mỗi giao dịch tạo ra một bản ghi gồm thời điểm, loại giao dịch, số tiền, tài khoản nguồn, tài khoản đích và số dư trước, sau khi thực hiện. Với hàng triệu giao dịch mỗi ngày, dữ liệu giao dịch mang đầy đủ đặc điểm của dữ liệu lớn theo mô hình 3V: khối lượng (Volume) lớn, tốc độ phát sinh (Velocity) cao và đa dạng (Variety) về kiểu dữ liệu [24]. Việc rà soát thủ công không còn khả thi, còn các công cụ xử lý trên một máy đơn lẻ nhanh chóng chạm giới hạn bộ nhớ và thời gian tính toán."),
  P("Một đặc điểm quan trọng khác là dữ liệu gian lận *mất cân bằng nghiêm trọng*: giao dịch gian lận thường chỉ chiếm dưới 1% tổng số giao dịch [15], [17]. Trong bối cảnh này, một mô hình dự đoán \"tất cả đều bình thường\" vẫn có độ chính xác (Accuracy) trên 99% nhưng hoàn toàn vô dụng. Do đó, bài toán đòi hỏi đồng thời một nền tảng xử lý dữ liệu có khả năng mở rộng và các phương pháp học máy, thang đo đánh giá phù hợp với dữ liệu mất cân bằng."),
  H2("1.2. Lý do chọn đề tài"),
  P("Các hệ thống phát hiện gian lận truyền thống chủ yếu dựa vào luật cố định do chuyên gia thiết lập, ví dụ cảnh báo khi số tiền vượt một ngưỡng hoặc khi có nhiều giao dịch trong thời gian ngắn. Cách tiếp cận này dễ triển khai nhưng có ba hạn chế: (i) khó thích nghi khi hành vi gian lận thay đổi; (ii) luật quá chặt tạo nhiều cảnh báo sai, luật quá rộng bỏ sót gian lận; (iii) chỉ khai thác một vài thuộc tính đơn lẻ, không tận dụng được tương tác giữa nhiều thuộc tính. Ngay trong bộ dữ liệu PaySim, luật cảnh báo có sẵn `isFlaggedFraud` chỉ gắn cờ " + fmtInt(flaggedTotal) + " giao dịch trên tổng số " + fmtInt(N1) + " giao dịch gian lận."),
  P("Học máy có giám sát cho phép mô hình tự học quy luật từ dữ liệu lịch sử đã gán nhãn, kết hợp nhiều thuộc tính để đưa ra xác suất gian lận cho từng giao dịch. Apache Spark là nền tảng phù hợp để hiện thực hóa cách tiếp cận này ở quy mô lớn vì cung cấp cơ chế xử lý phân tán trên bộ nhớ, API DataFrame và Spark SQL cho dữ liệu có cấu trúc, cùng thư viện MLlib tích hợp sẵn các thuật toán học máy [4], [5], [9]. Nhờ đó, toàn bộ các bước đọc dữ liệu, làm sạch, phân tích, tạo đặc trưng, huấn luyện và đánh giá được thực hiện trong một quy trình thống nhất, có thể chạy trên máy cá nhân và mở rộng lên cụm máy mà không phải viết lại mã nguồn. Đây là lý do nhóm lựa chọn đề tài nhằm vận dụng kiến thức môn Big Data vào một bài toán có giá trị thực tiễn cao."),
  H2("1.3. Mục tiêu đề tài"),
  P("Mục tiêu tổng quát của đề tài là ứng dụng Apache Spark để xử lý dữ liệu giao dịch tài chính quy mô lớn và xây dựng mô hình học máy hỗ trợ phát hiện giao dịch có dấu hiệu gian lận. Các mục tiêu cụ thể gồm:"),
  bullet("Tìm hiểu nền tảng dữ liệu lớn (Hadoop, MapReduce, Spark) và bài toán phát hiện gian lận trên dữ liệu mất cân bằng."),
  bullet("Sử dụng PySpark, DataFrame và Spark SQL để đọc, kiểm tra chất lượng, làm sạch, thống kê và trực quan hóa hơn sáu triệu giao dịch."),
  bullet("Thiết kế đặc trưng phản ánh hành vi gian lận; huấn luyện và so sánh bốn thuật toán MLlib: Hồi quy Logistic, Cây quyết định, Rừng ngẫu nhiên và Gradient-Boosted Trees."),
  bullet("Đánh giá mô hình bằng ma trận nhầm lẫn, Precision, Recall, F1-score, AUC-ROC, AUC-PR; lựa chọn mô hình tốt nhất và xây dựng ứng dụng demo."),
  H2("1.4. Đối tượng và phạm vi nghiên cứu"),
  ...table("Đối tượng và phạm vi nghiên cứu", ["Nội dung", "Phạm vi thực hiện"], [
    ["Đối tượng", "Dữ liệu giao dịch tài chính có cấu trúc, mỗi bản ghi có nhãn bình thường/gian lận."],
    ["Dữ liệu", "Bộ dữ liệu công khai PaySim (Kaggle) phục vụ học tập, không dùng dữ liệu cá nhân thật."],
    ["Xử lý", "Theo lô (batch) trên Spark chế độ local; chưa triển khai thời gian thực."],
    ["Bài toán", "Phân loại nhị phân: 0 – bình thường, 1 – gian lận."],
    ["Thuật toán", "Logistic Regression (mô hình cơ sở), Decision Tree, Random Forest, GBT."],
  ], [1, 3]),
  H2("1.5. Đóng góp và cấu trúc bài báo"),
  P("Đóng góp của đề tài gồm: (1) một quy trình Spark hoàn chỉnh, tái lập được từ dữ liệu thô đến mô hình đã lưu; (2) phân tích bằng Spark SQL làm rõ các dấu hiệu gian lận trong PaySim và dùng chúng để thiết kế đặc trưng; (3) so sánh định lượng bốn thuật toán MLlib cùng luật cơ sở trên dữ liệu mất cân bằng, có lựa chọn ngưỡng quyết định tránh rò rỉ thông tin từ tập kiểm thử; (4) ứng dụng web minh họa khả năng triển khai mô hình. Phần còn lại của bài báo được tổ chức như sau: Mục II trình bày cơ sở lý thuyết và các nghiên cứu liên quan; Mục III mô tả chi tiết phương pháp và quá trình xây dựng hệ thống; Mục IV trình bày kết quả thực nghiệm, so sánh mô hình và giao diện ứng dụng; Mục V kết luận và nêu hướng phát triển."),
];

// ======================= II. CƠ SỞ LÝ THUYẾT =======================
const theory = [
  H1("II. CƠ SỞ LÝ THUYẾT VÀ CÁC NGHIÊN CỨU LIÊN QUAN (RELATED WORK)"),
  H2("2.1. Dữ liệu lớn và hệ sinh thái Hadoop"),
  P("Dữ liệu lớn (Big Data) là tập dữ liệu có quy mô, tốc độ phát sinh hoặc mức độ đa dạng vượt quá khả năng xử lý hiệu quả của các công cụ truyền thống trong một khoảng thời gian chấp nhận được [24]. Giá trị của dữ liệu lớn không chỉ nằm ở số lượng bản ghi mà ở khả năng lưu trữ, xử lý và khai thác thông tin để hỗ trợ ra quyết định. Một hệ thống xử lý dữ liệu lớn cần mở rộng được khi dữ liệu tăng, phân phối công việc trên nhiều máy và tự phục hồi khi có lỗi."),
  P("Apache Hadoop là nền tảng mã nguồn mở đặt nền móng cho xử lý dữ liệu phân tán. Hệ thống tệp phân tán HDFS chia tệp thành các khối, lưu nhân bản trên nhiều DataNode, trong khi NameNode quản lý siêu dữ liệu và vị trí khối [1]. YARN quản lý tài nguyên cụm thông qua ResourceManager, NodeManager và ApplicationMaster [2]. Mô hình lập trình MapReduce chia công việc thành giai đoạn Map sinh các cặp khóa–giá trị trung gian và giai đoạn Reduce tổng hợp kết quả theo khóa; nền tảng tự lo việc chia dữ liệu, lập lịch và xử lý lỗi [3]. Tuy nhiên, MapReduce ghi kết quả trung gian xuống đĩa sau mỗi giai đoạn, khiến các thuật toán lặp như huấn luyện học máy trở nên chậm."),
  H2("2.2. Apache Spark"),
  P("Apache Spark là công cụ phân tích dữ liệu phân tán hợp nhất, hỗ trợ xử lý theo lô, truy vấn SQL, xử lý luồng và học máy trên cùng một nền tảng [9]. Khái niệm cốt lõi của Spark là RDD (Resilient Distributed Dataset) – tập dữ liệu phân tán, bất biến, có khả năng tự phục hồi nhờ lưu lại chuỗi phép biến đổi (lineage) thay vì nhân bản dữ liệu [8]. Spark có thể lưu dữ liệu trung gian trong bộ nhớ (cache), nhờ đó các thuật toán lặp nhanh hơn đáng kể so với MapReduce."),
  P("Một ứng dụng Spark gồm chương trình Driver và nhiều Executor (Hình " + nextFig() + "). Driver khởi tạo SparkSession, chuyển chương trình thành đồ thị có hướng không chu trình (DAG) các giai đoạn và phân phối tác vụ; Executor thực thi tác vụ trên từng phân vùng dữ liệu song song. Spark áp dụng cơ chế đánh giá lười (lazy evaluation): các phép biến đổi như `select`, `filter`, `groupBy` chỉ xây dựng kế hoạch, việc tính toán chỉ diễn ra khi gặp hành động (action) như `count`, `collect` hay `write`, cho phép Spark tối ưu toàn bộ kế hoạch trước khi chạy."),
  ...figure(fig("fig_spark_architecture.png"), "Kiến trúc thực thi của một ứng dụng Apache Spark"),
  H2("2.3. Spark SQL và DataFrame"),
  P("Spark SQL là mô-đun xử lý dữ liệu có cấu trúc, cho phép thao tác bằng câu lệnh SQL hoặc API DataFrame [4]. DataFrame là tập dữ liệu phân tán tổ chức thành các cột có tên và kiểu, tương tự bảng trong cơ sở dữ liệu quan hệ. Bộ tối ưu Catalyst phân tích, tối ưu kế hoạch logic (đẩy bộ lọc xuống sớm, cắt bỏ cột thừa) rồi sinh kế hoạch vật lý hiệu quả; công cụ Tungsten quản lý bộ nhớ và sinh mã để tăng tốc thực thi. Trong đề tài, DataFrame được dùng xuyên suốt từ đọc dữ liệu đến chuẩn bị đầu vào MLlib, còn Spark SQL được dùng cho các truy vấn thống kê sau khi đăng ký DataFrame thành bảng tạm."),
  P("Để lưu trữ dữ liệu phục vụ phân tích, hệ sinh thái Spark thường dùng định dạng Apache Parquet. Khác với CSV lưu theo dòng dưới dạng văn bản, Parquet lưu dữ liệu nhị phân theo cột, nén riêng từng cột và ghi kèm lược đồ cùng thống kê (giá trị nhỏ nhất, lớn nhất) của từng khối. Nhờ đó, khi truy vấn chỉ cần một vài cột, Spark chỉ đọc đúng các cột đó (column pruning) và có thể bỏ qua các khối không thỏa điều kiện lọc (predicate pushdown), giảm đáng kể lượng dữ liệu đọc từ đĩa so với CSV."),
  H2("2.4. Spark MLlib và cơ chế Pipeline"),
  P("MLlib là thư viện học máy của Spark, cung cấp các thuật toán phân loại, hồi quy, phân cụm, công cụ biến đổi đặc trưng, Pipeline và công cụ đánh giá, tinh chỉnh tham số [5], [7]. API `spark.ml` làm việc trực tiếp trên DataFrame với hai khái niệm chính. *Transformer* biến đổi một DataFrame thành DataFrame khác, chẳng hạn một mô hình đã huấn luyện sẽ thêm cột dự đoán. *Estimator* học từ dữ liệu để sinh ra một Transformer, chẳng hạn phương thức `fit()` của thuật toán Hồi quy Logistic trả về một mô hình đã huấn luyện."),
  P("Pipeline nối nhiều giai đoạn Transformer và Estimator thành một quy trình duy nhất [6]. Sau khi huấn luyện, đối tượng `PipelineModel` áp dụng chính xác cùng các phép biến đổi cho dữ liệu kiểm thử và dữ liệu mới, tránh sai lệch giữa giai đoạn huấn luyện và giai đoạn triển khai. Pipeline cũng có thể lưu xuống đĩa và nạp lại ở một chương trình khác, rất thuận tiện khi tích hợp vào ứng dụng."),
  H2("2.5. Phân loại trên dữ liệu mất cân bằng"),
  P("Trong học có giám sát, mô hình học ánh xạ từ véc-tơ đặc trưng x sang nhãn y. Khi lớp thiểu số (gian lận) chỉ chiếm tỷ lệ rất nhỏ, thuật toán tối thiểu hóa sai số trung bình sẽ thiên về lớp đa số [17]. Các kỹ thuật xử lý phổ biến gồm: lấy mẫu lại (giảm mẫu lớp đa số hoặc tăng mẫu lớp thiểu số, ví dụ SMOTE); gán trọng số cho lớp (cost-sensitive learning); và điều chỉnh ngưỡng quyết định [17], [18]. Đề tài chọn trọng số lớp vì không làm mất dữ liệu, không sinh dữ liệu giả và được hỗ trợ trực tiếp qua tham số `weightCol` của MLlib. Trọng số lớp c được tính theo công thức cân bằng:"),
  eq([sub([mr("w")], [mr("c")]), mr(" = "), frac([mr("N")], [mr("2 · "), sub([mr("N")], [mr("c")])])]),
  P("trong đó N là tổng số mẫu huấn luyện và N_{c} là số mẫu của lớp c. Như vậy tổng trọng số của hai lớp bằng nhau, buộc mô hình chú ý đến lớp gian lận.", { noIndent: true }),
  H2("2.6. Các thuật toán học máy"),
  P("Đề tài sử dụng bốn thuật toán phân loại của MLlib, trong bảng biểu được viết tắt lần lượt là LR, DT, RF và GBT.", { noIndent: true }),
  P("**Hồi quy Logistic** (Logistic Regression) là mô hình tuyến tính xác suất [22]. Mô hình tính tổ hợp tuyến tính z của đặc trưng rồi đưa qua hàm sigmoid để được xác suất thuộc lớp gian lận:"),
  eq([mr("P(y = 1 ∣ x) = "), frac([mr("1")], [mr("1 + "), sup([mr("e")], [mr("−z")])]), mr(",   z = "), sup([mr("w")], [mr("T")]), mr("x + b")]),
  P("Tham số w, b được học bằng cách cực tiểu hóa hàm mất mát log-loss có trọng số, kèm điều chuẩn L2 với hệ số λ:", { noIndent: true }),
  eq([mr("J(w) = −"), frac([mr("1")], [mr("N")]), sum([mr("i=1")], [mr("N")], [sub([mr("w")], [mr("i")]), sub([mr("ℓ")], [mr("i")])]), mr(" + "), frac([mr("λ")], [mr("2")]), sup([mr("‖w‖")], [mr("2")])]),
  eq([sub([mr("ℓ")], [mr("i")]), mr(" = "), sub([mr("y")], [mr("i")]), mr("log "), sub([mr("p")], [mr("i")]), mr(" + (1 − "), sub([mr("y")], [mr("i")]), mr(")log(1 − "), sub([mr("p")], [mr("i")]), mr(")")]),
  P("trong đó w_{i} là trọng số lớp của mẫu i, p_{i} là xác suất dự đoán. Mô hình huấn luyện nhanh, dễ giải thích qua dấu và độ lớn của hệ số, nhưng ranh giới quyết định tuyến tính nên khó nắm bắt tương tác phức tạp giữa các đặc trưng. Đề tài dùng Hồi quy Logistic làm mô hình cơ sở (baseline).", { noIndent: true }),
  P("**Cây quyết định** (Decision Tree) phân chia không gian đặc trưng thành các vùng bằng chuỗi điều kiện dạng \"đặc trưng ≤ ngưỡng\" [14]. Tại mỗi nút, MLlib chọn phép chia làm giảm nhiều nhất độ vẩn đục, mặc định là chỉ số Gini, với p_{k} là tỷ lệ mẫu lớp k tại nút:"),
  eq([mr("Gini = 1 − "), sum([mr("k=1")], [mr("K")], [sup([sub([mr("p")], [mr("k")])], [mr("2")])])]),
  P("Cây dễ diễn giải và không cần chuẩn hóa dữ liệu, nhưng một cây đơn lẻ dễ quá khớp và kém ổn định.", { noIndent: true }),
  P("**Rừng ngẫu nhiên** (Random Forest) là mô hình tổ hợp gồm T cây, mỗi cây được huấn luyện trên một mẫu bootstrap và chỉ xét một tập con ngẫu nhiên đặc trưng tại mỗi nút [12]. Xác suất dự đoán là trung bình xác suất của các cây:"),
  eq([mr("p(x) = "), frac([mr("1")], [mr("T")]), sum([mr("t=1")], [mr("T")], [sub([mr("p")], [mr("t")]), mr("(x)")])]),
  P("Sự ngẫu nhiên kép giúp giảm phương sai, chống quá khớp và cho phép ước lượng độ quan trọng của đặc trưng. MLlib huấn luyện các cây song song trên các Executor nên phù hợp với dữ liệu lớn.", { noIndent: true }),
  P("**Gradient-Boosted Trees** (GBT) xây dựng tuần tự M cây nông, mỗi cây h_{m} học theo hướng gradient âm của hàm mất mát để sửa lỗi của tổ hợp trước đó, với tốc độ học η [13]:"),
  eq([sub([mr("F")], [mr("m")]), mr("(x) = "), sub([mr("F")], [mr("m−1")]), mr("(x) + η · "), sub([mr("h")], [mr("m")]), mr("(x)")]),
  P("GBT thường có độ chính xác cao nhưng phải huấn luyện tuần tự giữa các vòng nên tốn thời gian hơn Rừng ngẫu nhiên.", { noIndent: true }),
  ...table("So sánh đặc điểm các thuật toán sử dụng", ["Tiêu chí", "LR", "DT", "RF", "GBT"], [
    ["Quan hệ học được", "Tuyến tính", "Phi tuyến", "Phi tuyến", "Phi tuyến"],
    ["Khả năng giải thích", "Cao", "Cao", "Trung bình", "Thấp"],
    ["Nguy cơ quá khớp", "Thấp", "Cao", "Thấp", "Trung bình"],
    ["Cần chuẩn hóa", "Có", "Không", "Không", "Không"],
    ["Song song hóa", "Theo dữ liệu", "Theo dữ liệu", "Theo cây", "Trong mỗi vòng"],
  ], [1.6, 1, 1, 1, 1.1]),
  H2("2.7. Các chỉ số đánh giá"),
  P("Kết quả phân loại được tổng hợp trong ma trận nhầm lẫn (Bảng " + nextTab() + "). TP là giao dịch gian lận được phát hiện đúng; TN là giao dịch bình thường được dự đoán đúng; FP là giao dịch bình thường bị cảnh báo nhầm, làm tăng chi phí kiểm tra thủ công và gây phiền hà cho khách hàng; FN là giao dịch gian lận bị bỏ sót, gây thiệt hại tài chính trực tiếp."),
  ...table("Ma trận nhầm lẫn cho bài toán phát hiện gian lận", ["Thực tế \\ Dự đoán", "Bình thường (0)", "Gian lận (1)"],
    [["Bình thường (0)", "TN", "FP"], ["Gian lận (1)", "FN", "TP"]], [1.4, 1, 1]),
  P("Từ ma trận nhầm lẫn, các chỉ số được tính như sau:", { noIndent: true }),
  eq([mr("Precision = "), frac([mr("TP")], [mr("TP + FP")]), mr(",   Recall = "), frac([mr("TP")], [mr("TP + FN")])]),
  eq([mr("F1 = "), frac([mr("2 · Precision · Recall")], [mr("Precision + Recall")])]),
  P("Precision đo tỷ lệ cảnh báo đúng trong các cảnh báo; Recall đo tỷ lệ gian lận được phát hiện; F1-score là trung bình điều hòa, chỉ cao khi cả hai cùng cao. AUC-ROC là diện tích dưới đường cong TPR–FPR, còn AUC-PR là diện tích dưới đường cong Precision–Recall. Với dữ liệu mất cân bằng, AUC-ROC có thể cho giá trị rất cao một cách lạc quan do số lượng TN khổng lồ, trong khi AUC-PR phản ánh trung thực hơn chất lượng nhận diện lớp thiểu số [20]. Vì vậy, đề tài dùng AUC-PR làm tiêu chí chính để chọn mô hình vì chỉ số này đánh giá chất lượng xếp hạng rủi ro trên mọi ngưỡng và không phụ thuộc vào ngưỡng được chọn; F1-score tại ngưỡng vận hành được dùng làm tiêu chí phụ.", { noIndent: true }),
  H2("2.8. Các nghiên cứu liên quan"),
  P("Bolton và Hand [15] tổng quan các phương pháp thống kê trong phát hiện gian lận, nhấn mạnh hai khó khăn cố hữu là dữ liệu mất cân bằng và hành vi gian lận thay đổi theo thời gian. Phua và cộng sự [23] cùng Abdallah và cộng sự [16] khảo sát các kỹ thuật khai phá dữ liệu cho chống gian lận, trong đó các mô hình học có giám sát dạng cây và mô hình tổ hợp thường cho kết quả tốt trên dữ liệu giao dịch có cấu trúc. Dal Pozzolo và cộng sự [18] phân tích bài toán gian lận thẻ tín dụng trong điều kiện thực tế, chỉ ra tầm quan trọng của việc chọn thang đo và xử lý mất cân bằng phù hợp."),
  P("Về khía cạnh dữ liệu, dữ liệu giao dịch tài chính thật hiếm khi được công bố vì lý do bảo mật. Lopez-Rojas và cộng sự [10] đề xuất PaySim – bộ mô phỏng giao dịch tiền di động dựa trên tác tử (agent-based), được hiệu chỉnh từ nhật ký giao dịch thật của một dịch vụ tiền di động ở châu Phi, sau đó chèn hành vi gian lận. Bộ dữ liệu sinh ra được công bố trên Kaggle [11] và trở thành chuẩn tham chiếu phổ biến cho các nghiên cứu phát hiện gian lận. Về khía cạnh hệ thống, Carcillo và cộng sự [19] xây dựng SCARFF – khung phát hiện gian lận thẻ tín dụng trên luồng dữ liệu, tích hợp Kafka, Spark và Cassandra, cho thấy Spark có thể đáp ứng yêu cầu khả năng mở rộng của hệ thống chống gian lận thực tế. Kế thừa các nghiên cứu này, đề tài tập trung vào một quy trình batch hoàn chỉnh trên Spark MLlib, so sánh các thuật toán sẵn có và phân tích ý nghĩa của từng đặc trưng."),
];

module.exports = { titleBlock, abstract, intro, theory, ctx: { q, e, m, env, team, N, N0, N1, pct1, byType, tOf, amt, bp,
  flaggedTotal, flaggedTrue, hours, topHours, busyHours, R, names, rank, best, B, worst, byAucpr, fastest, slowest,
  rb, S, imp, coef, VN, fv, relShare, totalMin, allFraudTypes, fig, OUT, FIG } };

// Phần 3: Mục IV – Kết quả, Mục V – Kết luận, Tài liệu tham khảo
const fs = require("fs");
const L = require("./lib");
const { d, FONT, P, H1, H2, bullet, figure, table, fmtInt, fmtDec, nextFig, nextTab } = L;
const { Paragraph, TextRun, AlignmentType } = d;
const { ctx } = require("./content_part1");
const { R, names, rank, best, B, worst, byAucpr, fastest, slowest, rb, S, imp, coef, VN, fv, totalMin, fig, N1, env } = ctx;

const f4 = (x) => fmtDec(x, 4);
const AB = { "Logistic Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF", "GBT": "GBT" };
const ab = (n) => AB[n] || n;
const TINY = 15;
const mm = /local\[(\d+|\*)\]/.exec(env.master || "");
const masterVN = mm ? `Spark chế độ local, ${mm[1] === "*" ? "dùng mọi nhân CPU" : mm[1] + " luồng xử lý"}` : `Spark master ${env.master}`;
const second = rank[1];
const lr = R["Logistic Regression"];
const bestTree = ctx.m.feature_importance_model;
const top3 = imp.slice(0, 3).map(([k, v]) => `${fv(k)} (${fmtDec(v, 3)})`).join(", ");
const posCoef = coef.filter(([, v]) => v > 0).slice(0, 2).map(([k]) => fv(k));
const negCoef = coef.filter(([, v]) => v < 0).slice(0, 2).map(([k]) => fv(k));
const fpRed = (n) => R[n].at_default_0_5.FP - R[n].FP;
const bestReduce = 100 * fpRed(best) / Math.max(R[best].at_default_0_5.FP, 1);
const sumTrain = names.reduce((a, n) => a + R[n].train_time_s, 0);
const appShots = ["fig_app_single.png", "fig_app_batch.png", "fig_app_dashboard.png"].filter((f) => fs.existsSync(fig(f)));

const results = [
  H1("IV. KẾT QUẢ THỰC NGHIỆM"),
  H2("4.1. Kết quả tổng hợp và so sánh các mô hình"),
  P(`Bảng ${nextTab()} tổng hợp kết quả của bốn mô hình trên tập kiểm thử gồm ${fmtInt(S.test)} giao dịch (${fmtInt(S.test_fraud)} gian lận), tại ngưỡng t∗ đã chọn trên tập kiểm định. Dòng cuối là luật cảnh báo có sẵn của hệ thống (\`isFlaggedFraud\`) trên cùng tập kiểm thử, dùng làm mốc so sánh; hàng tô đậm là mô hình được chọn.`),
  ...table("Kết quả các mô hình trên tập kiểm thử", ["Mô hình", "t∗", "Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"],
    [...names.map((n) => [n === best ? `**${ab(n)}**` : ab(n), fmtDec(R[n].threshold, 3), f4(R[n].precision), f4(R[n].recall),
      f4(R[n].f1), f4(R[n].auc_roc), f4(R[n].auc_pr)]),
    ["Luật hệ thống", "–", f4(rb.precision), f4(rb.recall), f4(rb.f1), "–", "–"]],
    [1.25, 0.6, 0.8, 0.75, 0.75, 0.8, 0.8], { highlight: (k) => names[k] === best, size: TINY }),
  P(`Xét theo tiêu chí chính AUC-PR, thứ tự các mô hình là ${byAucpr.join(" > ")}; ${VN[best]} (${best}) dẫn đầu với AUC-PR ${f4(B.auc_pr)}. Xét theo F1-score tại ngưỡng t∗, mô hình cao nhất là ${rank[0]} (${f4(R[rank[0]].f1)}) và thấp nhất là ${worst} (${f4(R[worst].f1)}); ${best} đạt F1 ${f4(B.f1)}. Chênh lệch F1 giữa các mô hình dạng cây là nhỏ, trong khi AUC-PR cho thấy khác biệt rõ hơn về khả năng xếp hạng rủi ro trên toàn dải ngưỡng. Tất cả các mô hình đều có AUC-ROC rất gần 1, minh chứng cho nhận định ở Mục 2.7 rằng AUC-ROC kém phân biệt trên dữ liệu mất cân bằng.`),
  P(`So với luật \`isFlaggedFraud\` chỉ phát hiện ${fmtInt(rb.TP)}/${fmtInt(rb.TP + rb.FN)} giao dịch gian lận trong tập kiểm thử (Recall ${f4(rb.recall)}), mô hình tốt nhất phát hiện ${fmtInt(B.TP)} giao dịch (Recall ${f4(B.recall)}) với Precision ${f4(B.precision)}. Nói cách khác, cứ 100 cảnh báo của mô hình thì khoảng ${Math.round(100 * B.precision)} cảnh báo là gian lận thật, và mô hình phát hiện được khoảng ${Math.round(100 * B.recall)}% tổng số vụ gian lận. Đây là mức cải thiện rất lớn so với cách tiếp cận dựa trên luật cố định.`),
  P(B.f1 - lr.f1 < 0.005
    ? `Đáng chú ý, Hồi quy Logistic – dù chỉ học ranh giới tuyến tính – vẫn đạt F1 ${f4(lr.f1)}, chỉ thấp hơn mô hình tốt nhất ${fmtDec(100 * (B.f1 - lr.f1), 2)} điểm phần trăm. Điều này cho thấy các đặc trưng phái sinh về số dư đã mã hóa gần như toàn bộ tín hiệu gian lận; ưu thế của các mô hình cây chủ yếu thể hiện ở AUC-PR và Precision, tức khả năng giảm cảnh báo sai. Hình ${nextFig()} trực quan hóa so sánh các chỉ số chính.`
    : `Hồi quy Logistic, dù chỉ học ranh giới tuyến tính, vẫn đạt F1 ${f4(lr.f1)} nhờ các đặc trưng phái sinh đã mã hóa sẵn phần lớn tín hiệu gian lận. Khoảng cách giữa Hồi quy Logistic và các mô hình cây phản ánh giá trị của khả năng học quan hệ phi tuyến và tương tác giữa đặc trưng, ví dụ tổ hợp "rút sạch tài khoản nguồn" và "số dư đích bằng 0". Hình ${nextFig()} trực quan hóa so sánh các chỉ số chính.`),
  ...figure(fig("fig_metrics_comparison.png"), "So sánh Precision, Recall, F1 và AUC-PR của bốn mô hình"),
  H2("4.2. Ma trận nhầm lẫn"),
  P(`Hình ${nextFig()} trình bày ma trận nhầm lẫn của bốn mô hình trên tập kiểm thử. Với mô hình tốt nhất, số giao dịch bình thường bị cảnh báo nhầm (FP) là ${fmtInt(B.FP)} trên tổng số ${fmtInt(B.TN + B.FP)} giao dịch bình thường, tương ứng tỷ lệ báo động giả ${fmtDec(100 * B.FP / (B.TN + B.FP), 4)}%; số giao dịch gian lận bị bỏ sót (FN) là ${fmtInt(B.FN)}. Trong vận hành thực tế, FP quyết định khối lượng công việc của bộ phận thẩm định, còn FN quyết định thiệt hại tài chính; ma trận nhầm lẫn giúp nhà quản lý định lượng cả hai loại chi phí này.`),
  ...figure(fig("fig_confusion_matrices.png"), "Ma trận nhầm lẫn của các mô hình trên tập kiểm thử (màu theo thang log)"),
  P(`Bảng ${nextTab()} liệt kê chi tiết các thành phần của ma trận nhầm lẫn cùng độ đặc hiệu (Specificity = TN/(TN+FP)) và độ chính xác tổng thể (Accuracy). Accuracy của mọi mô hình đều xấp xỉ 1 do lớp bình thường chiếm áp đảo, vì vậy chỉ số này không được dùng để so sánh.`),
  ...table("Chi tiết ma trận nhầm lẫn trên tập kiểm thử", ["Mô hình", "TP", "FP", "FN", "TN", "Specificity", "Accuracy"],
    names.map((n) => [ab(n), fmtInt(R[n].TP), fmtInt(R[n].FP), fmtInt(R[n].FN), fmtInt(R[n].TN), fmtDec(R[n].specificity, 5), fmtDec(R[n].accuracy, 5)]),
    [0.6, 0.6, 0.6, 0.55, 1.05, 0.95, 0.95], { size: TINY }),
  H2("4.3. Đường cong ROC và Precision–Recall"),
  P(`Hình ${nextFig()} và Hình ${nextFig() + 1} biểu diễn đường cong ROC (trục hoành thang log để làm rõ vùng FPR thấp) và đường cong Precision–Recall. Trên đường cong ROC, các mô hình gần như trùng nhau ở vùng FPR lớn và chỉ tách biệt ở vùng FPR rất nhỏ – vùng có ý nghĩa thực tiễn vì một ngân hàng chỉ chấp nhận tỷ lệ báo động giả rất thấp. Đường cong Precision–Recall phân biệt rõ hơn: mô hình có đường cong nằm phía trên – bên phải duy trì Precision cao ngay cả khi Recall tăng, nghĩa là có thể phát hiện thêm gian lận mà không phải trả giá bằng nhiều cảnh báo sai.`),
  ...figure(fig("fig_roc_curves.png"), "Đường cong ROC của các mô hình (trục FPR thang log)"),
  ...figure(fig("fig_pr_curves.png"), "Đường cong Precision–Recall của các mô hình"),
  H2("4.4. Ảnh hưởng của ngưỡng quyết định"),
  P(`Bảng ${nextTab()} so sánh kết quả tại ngưỡng mặc định 0,5 và tại ngưỡng t∗ chọn trên tập kiểm định. Do huấn luyện với trọng số lớp gian lận ≈ ${fmtDec(S.class_weights["1"], 0)}, tại ngưỡng 0,5 các mô hình thiên mạnh về phía cảnh báo: Recall cao nhưng FP lớn và Precision thấp. Việc chọn ngưỡng trên tập kiểm định giúp ${best} giảm ${fmtInt(fpRed(best))} cảnh báo sai (khoảng ${fmtDec(bestReduce, 1)}%) so với ngưỡng 0,5. Kết quả khẳng định trọng số lớp và điều chỉnh ngưỡng là hai kỹ thuật bổ sung cho nhau: trọng số giúp mô hình học được lớp thiểu số, còn ngưỡng giúp hiệu chỉnh điểm vận hành theo mục tiêu nghiệp vụ.`),
  ...table("So sánh kết quả tại ngưỡng 0,5 và ngưỡng t∗", ["Mô hình", "Ngưỡng", "FP", "FN", "Precision", "Recall", "F1"],
    names.flatMap((n) => [[ab(n), "0,5", fmtInt(R[n].at_default_0_5.FP), fmtInt(R[n].at_default_0_5.FN), f4(R[n].at_default_0_5.precision), f4(R[n].at_default_0_5.recall), f4(R[n].at_default_0_5.f1)],
      ["", fmtDec(R[n].threshold, 3), fmtInt(R[n].FP), fmtInt(R[n].FN), f4(R[n].precision), f4(R[n].recall), f4(R[n].f1)]]),
    [0.6, 0.75, 0.65, 0.55, 0.8, 0.75, 0.75], { highlight: (k) => k % 2 === 1, size: TINY }),
  H2("4.5. Độ quan trọng của đặc trưng"),
  P(`Hình ${nextFig()} trình bày độ quan trọng đặc trưng (feature importance) do mô hình ${bestTree} tính dựa trên mức giảm độ vẩn đục Gini tích lũy. Ba đặc trưng quan trọng nhất là ${top3}. Kết quả này thống nhất với phân tích ở Mục 3.7: các dấu hiệu về số dư tài khoản mang nhiều thông tin hơn hẳn số tiền giao dịch đơn thuần, điều này cũng giải thích vì sao luật dựa trên ngưỡng số tiền như \`isFlaggedFraud\` hoạt động kém.${imp[0][1] > 0.5 ? ` Đáng chú ý, riêng ${fv(imp[0][0])} chiếm ${fmtDec(100 * imp[0][1], 1)}% tổng độ quan trọng: mô hình phụ thuộc mạnh vào một dấu hiệu duy nhất. Trên dữ liệu thực tế, nơi kẻ gian có thể chủ động không rút sạch tài khoản để né tránh, cần bổ sung các đặc trưng hành vi khác để mô hình bền vững hơn.` : ""}`),
  ...figure(fig("fig_feature_importance.png"), `Độ quan trọng đặc trưng của mô hình ${bestTree}`),
  P(`Ở mô hình Hồi quy Logistic, các đặc trưng đã được chuẩn hóa nên có thể so sánh trực tiếp hệ số. Các đặc trưng có hệ số dương lớn nhất (làm tăng xác suất gian lận) là ${posCoef.join(" và ") || "–"}; các đặc trưng có hệ số âm lớn nhất (làm giảm xác suất gian lận) là ${negCoef.join(" và ") || "–"}. Khả năng diễn giải này là ưu điểm của mô hình tuyến tính khi cần giải trình quyết định với khách hàng hoặc cơ quan quản lý.`),
  H2("4.6. Hiệu năng xử lý trên Spark"),
  P(`Bảng ${nextTab()} trình bày thời gian huấn luyện và thời gian chấm điểm toàn bộ tập kiểm thử của từng mô hình. ${fastest} huấn luyện nhanh nhất, ${slowest} chậm nhất. Rừng ngẫu nhiên có thể huấn luyện các cây song song, trong khi GBT phải xây dựng tuần tự từng cây nên chi phí tăng tuyến tính theo số vòng lặp. Tổng thời gian huấn luyện bốn mô hình trên ${fmtInt(S.train)} giao dịch là ${fmtDec(sumTrain / 60, 1)} phút${totalMin ? `; toàn bộ quy trình từ đọc CSV đến lưu mô hình mất khoảng ${totalMin} phút` : ""} với ${masterVN}.`),
  ...table("Thời gian huấn luyện và dự đoán", ["Mô hình", "Huấn luyện (giây)", "Dự đoán tập kiểm thử (giây)"],
    names.map((n) => [n, fmtDec(R[n].train_time_s, 1), fmtDec(R[n].predict_time_s, 1)]), [1.5, 1.1, 1.4]),
  P("Do Spark chia dữ liệu thành các phân vùng độc lập, khi triển khai trên cụm nhiều máy hoặc máy nhiều nhân hơn, các giai đoạn đọc dữ liệu, tổng hợp và huấn luyện được song song hóa mà không phải thay đổi mã nguồn. Việc lưu dữ liệu sạch dưới dạng Parquet và lưu đệm DataFrame trong bộ nhớ cũng giúp các lần chạy phân tích, huấn luyện tiếp theo nhanh hơn đáng kể so với đọc lại tệp CSV."),
  H2("4.7. Lựa chọn mô hình tốt nhất"),
  P(`Tổng hợp các tiêu chí – AUC-PR là tiêu chí chính, F1-score là tiêu chí phụ, kết hợp chi phí tính toán và khả năng triển khai – nhóm lựa chọn **${VN[best]} (${best})** làm mô hình chính thức của hệ thống. Mô hình xếp hạng rủi ro tốt nhất và tại ngưỡng t∗ đạt sự cân bằng tốt giữa phát hiện gian lận (Recall ${f4(B.recall)}) và hạn chế cảnh báo sai (Precision ${f4(B.precision)}). Hồi quy Logistic được giữ làm mô hình đối chứng nhờ tốc độ cao và khả năng diễn giải. Mô hình tốt nhất được lưu dưới dạng \`PipelineModel\` – bao gồm cả các bước tiền xử lý – cùng ngưỡng t∗ = ${fmtDec(B.threshold, 3)}, sẵn sàng tích hợp vào ứng dụng.`),
  H2("4.8. Giao diện ứng dụng"),
  P(`Ứng dụng demo gồm ba chức năng tương ứng ba thẻ (tab). *Kiểm tra một giao dịch* cho phép nhập loại giao dịch, số tiền và số dư hai phía; hệ thống hiển thị xác suất gian lận, hai đại lượng sai lệch số dư và kết luận cảnh báo theo ngưỡng (Hình ${nextFig()}). *Kiểm tra theo lô* nhận tệp CSV có cùng cấu trúc PaySim, dùng Spark chấm điểm toàn bộ giao dịch, hiển thị số cảnh báo, bảng kết quả sắp xếp theo mức rủi ro và cho phép tải kết quả về${appShots.length > 1 ? ` (Hình ${nextFig() + 1})` : ""}. *Thống kê và đánh giá* hiển thị bảng so sánh mô hình và các biểu đồ phân tích${appShots.length > 2 ? ` (Hình ${nextFig() + 2})` : ""}. Thanh bên cho phép điều chỉnh ngưỡng cảnh báo, giúp người dùng quan sát trực tiếp sự đánh đổi giữa Precision và Recall.`),
  ...(appShots.includes("fig_app_single.png") ? figure(fig("fig_app_single.png"), "Giao diện kiểm tra một giao dịch: giao dịch rút sạch tài khoản bị cảnh báo gian lận") : []),
  ...(appShots.includes("fig_app_batch.png") ? figure(fig("fig_app_batch.png"), "Giao diện kiểm tra theo lô từ tệp CSV") : []),
  ...(appShots.includes("fig_app_dashboard.png") ? figure(fig("fig_app_dashboard.png"), "Giao diện thống kê và đánh giá mô hình") : []),
  H2("4.9. Thảo luận"),
  P("Kết quả thực nghiệm cho thấy chất lượng phát hiện gian lận phụ thuộc mạnh vào chất lượng đặc trưng. Các đặc trưng phái sinh về số dư được thiết kế từ phân tích Spark SQL đóng vai trò quyết định, còn lựa chọn thuật toán tạo ra khác biệt ở mức tinh chỉnh. Đây là bài học quan trọng: phân tích khám phá dữ liệu không chỉ là bước mô tả mà trực tiếp định hướng thiết kế mô hình."),
  P("Cần lưu ý rằng PaySim là dữ liệu mô phỏng nên các quy luật gian lận khá nhất quán, giúp mô hình đạt kết quả cao hơn so với dữ liệu thực tế, nơi hành vi gian lận đa dạng và thay đổi theo thời gian [15], [27]. Ngoài ra, các đặc trưng số dư sau giao dịch chỉ có được khi giao dịch đã được hệ thống ghi nhận; trong kịch bản chặn giao dịch trước khi thực hiện, cần thay bằng đặc trưng tính từ số dư hiện tại và lịch sử giao dịch của tài khoản."),
];

const conclusion = [
  H1("V. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN"),
  H2("5.1. Kết quả đạt được"),
  P(`Đề tài đã xây dựng thành công một hệ thống phát hiện gian lận giao dịch tài chính hoàn chỉnh trên nền tảng Apache Spark. Về dữ liệu, nhóm đã nạp, kiểm tra chất lượng, làm sạch và phân tích hơn sáu triệu giao dịch bằng DataFrame và Spark SQL, làm rõ các đặc điểm của gian lận: chỉ xuất hiện ở TRANSFER và CASH_OUT, thường rút sạch tài khoản nguồn, có các mẫu số dư bất thường và tỷ lệ cao vào ban đêm. Về mô hình, nhóm đã thiết kế 12 đặc trưng, xử lý mất cân bằng bằng trọng số lớp và chọn ngưỡng trên tập kiểm định, huấn luyện và so sánh bốn thuật toán MLlib. ${VN[best]} đạt kết quả tốt nhất với F1-score ${f4(B.f1)} và AUC-PR ${f4(B.auc_pr)}, vượt xa luật cảnh báo có sẵn. Về ứng dụng, mô hình được đóng gói thành \`PipelineModel\` và tích hợp vào ứng dụng web hỗ trợ kiểm tra từng giao dịch và chấm điểm theo lô.`),
  H2("5.2. Hạn chế"),
  bullet("Dữ liệu PaySim là dữ liệu mô phỏng; kết quả cần được kiểm chứng thêm trên dữ liệu giao dịch thực tế."),
  bullet("Hệ thống mới xử lý theo lô, chưa đáp ứng yêu cầu chặn gian lận theo thời gian thực."),
  bullet("Một số đặc trưng dùng số dư sau giao dịch, chỉ khả dụng sau khi giao dịch được ghi nhận."),
  bullet("Siêu tham số được chọn thủ công; thực nghiệm chạy ở chế độ local, chưa đo hiệu năng trên cụm nhiều máy."),
  bullet("Chưa khai thác quan hệ giữa các tài khoản (mạng lưới tài khoản trung gian) và lịch sử hành vi của từng khách hàng."),
  H2("5.3. Hướng phát triển và cải tiến"),
  P("Trên cơ sở các hạn chế trên, nhóm đề xuất các hướng phát triển sau, kèm công nghệ cụ thể có thể sử dụng:", { noIndent: true }),
  bullet("**Phát hiện thời gian thực:** dùng Apache Kafka tiếp nhận luồng giao dịch và Spark Structured Streaming [25] áp dụng trực tiếp `PipelineModel` đã huấn luyện để chấm điểm từng giao dịch với độ trễ thấp, theo hướng kiến trúc SCARFF [19]."),
  bullet("**Tinh chỉnh siêu tham số:** dùng `CrossValidator` hoặc `TrainValidationSplit` kết hợp `ParamGridBuilder` của MLlib để tìm cấu hình tối ưu một cách có hệ thống."),
  bullet("**Thuật toán mạnh hơn:** thử nghiệm XGBoost [21] hoặc LightGBM phiên bản phân tán trên Spark (XGBoost4J-Spark, SynapseML), và các mô hình học sâu phát hiện bất thường."),
  bullet("**Đặc trưng hành vi và đồ thị:** tính đặc trưng theo cửa sổ thời gian (số giao dịch, tổng tiền của tài khoản trong 1 giờ/24 giờ) bằng Window function; dùng GraphFrames phân tích mạng lưới tài khoản để phát hiện các tài khoản trung gian (mule)."),
  bullet("**Giải thích mô hình:** áp dụng SHAP [26] để giải thích từng cảnh báo cho nhân viên thẩm định và khách hàng."),
  bullet("**Vận hành mô hình (MLOps):** theo dõi hiện tượng trôi dạt khái niệm (concept drift) [27], huấn luyện lại định kỳ, quản lý phiên bản mô hình bằng MLflow và triển khai trên cụm Spark (YARN/Kubernetes) với dữ liệu lưu trên HDFS hoặc kho đối tượng."),
];

const REFS = [
  "K. Shvachko, H. Kuang, S. Radia, and R. Chansler, \"The Hadoop Distributed File System,\" in Proc. IEEE 26th Symp. Mass Storage Systems and Technologies (MSST), 2010, pp. 1–10.",
  "V. K. Vavilapalli et al., \"Apache Hadoop YARN: Yet Another Resource Negotiator,\" in Proc. 4th ACM Symp. Cloud Computing (SoCC), 2013.",
  "J. Dean and S. Ghemawat, \"MapReduce: Simplified Data Processing on Large Clusters,\" Communications of the ACM, vol. 51, no. 1, pp. 107–113, 2008.",
  "M. Armbrust et al., \"Spark SQL: Relational Data Processing in Spark,\" in Proc. ACM SIGMOD Int. Conf. Management of Data, 2015, pp. 1383–1394.",
  "X. Meng et al., \"MLlib: Machine Learning in Apache Spark,\" Journal of Machine Learning Research, vol. 17, no. 34, pp. 1–7, 2016.",
  "Apache Software Foundation, \"ML Pipelines – Spark 3.5 Documentation.\" [Online]. Available: https://spark.apache.org/docs/latest/ml-pipeline.html",
  "Apache Software Foundation, \"Classification and Regression – Spark MLlib Documentation.\" [Online]. Available: https://spark.apache.org/docs/latest/ml-classification-regression.html",
  "M. Zaharia et al., \"Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing,\" in Proc. 9th USENIX Symp. Networked Systems Design and Implementation (NSDI), 2012.",
  "M. Zaharia et al., \"Apache Spark: A Unified Engine for Big Data Processing,\" Communications of the ACM, vol. 59, no. 11, pp. 56–65, 2016.",
  "E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, \"PaySim: A Financial Mobile Money Simulator for Fraud Detection,\" in Proc. 28th European Modeling and Simulation Symposium (EMSS), Larnaca, Cyprus, 2016.",
  "E. Lopez-Rojas, \"Synthetic Financial Datasets For Fraud Detection,\" Kaggle. [Online]. Available: https://www.kaggle.com/datasets/ealaxi/paysim1",
  "L. Breiman, \"Random Forests,\" Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
  "J. H. Friedman, \"Greedy Function Approximation: A Gradient Boosting Machine,\" The Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.",
  "L. Breiman, J. Friedman, R. Olshen, and C. Stone, Classification and Regression Trees. Wadsworth, 1984.",
  "R. J. Bolton and D. J. Hand, \"Statistical Fraud Detection: A Review,\" Statistical Science, vol. 17, no. 3, pp. 235–255, 2002.",
  "A. Abdallah, M. A. Maarof, and A. Zainal, \"Fraud Detection System: A Survey,\" Journal of Network and Computer Applications, vol. 68, pp. 90–113, 2016.",
  "H. He and E. A. Garcia, \"Learning from Imbalanced Data,\" IEEE Trans. Knowledge and Data Engineering, vol. 21, no. 9, pp. 1263–1284, 2009.",
  "A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, \"Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy,\" IEEE Trans. Neural Networks and Learning Systems, vol. 29, no. 8, pp. 3784–3797, 2018.",
  "F. Carcillo, A. Dal Pozzolo, Y.-A. Le Borgne, O. Caelen, Y. Mazzer, and G. Bontempi, \"SCARFF: A Scalable Framework for Streaming Credit Card Fraud Detection with Spark,\" Information Fusion, vol. 41, pp. 182–194, 2018.",
  "T. Saito and M. Rehmsmeier, \"The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,\" PLoS ONE, vol. 10, no. 3, e0118432, 2015.",
  "T. Chen and C. Guestrin, \"XGBoost: A Scalable Tree Boosting System,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785–794.",
  "D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, Applied Logistic Regression, 3rd ed. Wiley, 2013.",
  "C. Phua, V. Lee, K. Smith, and R. Gayler, \"A Comprehensive Survey of Data Mining-based Fraud Detection Research,\" arXiv:1009.6119, 2010.",
  "D. Laney, \"3D Data Management: Controlling Data Volume, Velocity, and Variety,\" META Group Research Note, 2001.",
  "M. Armbrust et al., \"Structured Streaming: A Declarative API for Real-Time Applications in Apache Spark,\" in Proc. ACM SIGMOD Int. Conf. Management of Data, 2018.",
  "S. M. Lundberg and S.-I. Lee, \"A Unified Approach to Interpreting Model Predictions,\" in Advances in Neural Information Processing Systems 30 (NeurIPS), 2017.",
  "J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia, \"A Survey on Concept Drift Adaptation,\" ACM Computing Surveys, vol. 46, no. 4, 2014.",
];
const references = [
  new Paragraph({ spacing: { before: 200, after: 100 }, keepNext: true,
    children: [new TextRun({ text: "TÀI LIỆU THAM KHẢO", bold: true, font: FONT, size: 23 })] }),
  ...REFS.map((r, i) => new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 40, line: 230 },
    indent: { left: 400, hanging: 400 },
    children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 17 }), new TextRun({ text: r, font: FONT, size: 17 })],
  })),
];

module.exports = { results, conclusion, references };

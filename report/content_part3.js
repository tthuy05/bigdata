// Results are derived from verified run artifacts; preserve the existing sections/tables/figures.
const fs = require("fs");
const path = require("path");
const { P, H1, H2, bullet, table, figure, fmtInt, fmtDec } = require("./lib");
const { ctx } = require("./content_part1");
const { references } = require("./references");
const { m, q, R, names, best, B, rb, S, imp, coef, fig } = ctx;
const f4 = x => fmtDec(x, 4);
const ab = n => ({"Logistic Regression":"LR", "Decision Tree":"DT", "Random Forest":"RF", "GBT":"GBT"}[n]);
const root = path.join(__dirname, "..");
const manifest = JSON.parse(fs.readFileSync(path.join(root, "outputs/metrics/experiments.json"), "utf8"));
if (manifest.status !== "VERIFIED" || manifest.run_id !== m.run_id) throw new Error("Manifest và kết quả không khớp.");
const experiments = Object.fromEntries(Object.entries(manifest.experiments).map(([key, value]) =>
  [key, JSON.parse(fs.readFileSync(path.join(root, value.results_path), "utf8"))]));
const historicalPath = path.join(root, "outputs/baseline/model_results.json");
const historicalSplit = fs.existsSync(historicalPath) ? JSON.parse(fs.readFileSync(historicalPath, "utf8")).split : null;
const allRows = Object.entries(experiments).flatMap(([key, e]) => Object.entries(e.results).map(([name, r]) =>
  [key, ab(name), f4(r.precision), f4(r.recall), f4(r.f1), f4(r.auc_roc), f4(r.auc_pr)]));

const results = [
  H1("IV. KẾT QUẢ THỰC NGHIỆM"),
  H2("4.1. Kết quả tổng hợp và so sánh các mô hình"),
  P(`Thí nghiệm chính được xác định trước là ${m.experiment}, với bộ đặc trưng ${m.feature_profile}. Mẫu đồng đều ${fmtDec(100*m.data.sample_fraction,1)}% từ CSV thật có ${fmtInt(m.data.modeling_rows)} dòng thuộc phạm vi TRANSFER/CASH_OUT. Test có ${fmtInt(S.test)} giao dịch, ${fmtInt(S.test_fraud)} gian lận. Mô hình ${best} được chọn trên Validation, không dựa vào thứ hạng trên Test. Các bảng sau ghi nhận kết quả Test tại ngưỡng đã cố định trước khi đánh giá.`),
  ...table("Kết quả các mô hình trên tập kiểm thử", ["Mô hình","t*","Precision","Recall","F1","AUC-ROC","AUC-PR"],
    [...names.map(n => [ab(n),fmtDec(R[n].threshold,3),f4(R[n].precision),f4(R[n].recall),f4(R[n].f1),f4(R[n].auc_roc),f4(R[n].auc_pr)]),
     ["Luật hệ thống","–",f4(rb.precision),f4(rb.recall),f4(rb.f1),"–","–"]],
    [1,.6,.9,.85,.75,.9,.9], {size:15, highlight:k=>names[k]===best}),
  P(`Mô hình chọn từ Validation đạt AUC-PR ${f4(B.auc_pr)}, Precision ${f4(B.precision)}, Recall ${f4(B.recall)} và F1 ${f4(B.f1)} trên Test. AUC-PR, Precision và Recall được ưu tiên vì tỷ lệ gian lận thấp; Accuracy chỉ được ghi để tham khảo. Thứ hạng Test là mô tả kết quả, không được dùng để thay thuật toán hoặc ngưỡng đã chọn.`),
  ...figure(fig("fig_metrics_comparison.png"), `So sánh chỉ số Test của ${m.experiment}`),
  H2("4.2. Ma trận nhầm lẫn"),
  P(`Với ${best}, Test ghi nhận TP=${fmtInt(B.TP)}, TN=${fmtInt(B.TN)}, FP=${fmtInt(B.FP)}, FN=${fmtInt(B.FN)}. FP phản ánh tải thẩm định cảnh báo nhầm; FN phản ánh các trường hợp gian lận bị bỏ sót. Các số đếm được kiểm tra khớp quy mô và số nhãn của Test.`),
  ...figure(fig("fig_confusion_matrices.png"), "Ma trận nhầm lẫn tại ngưỡng Validation đã lưu"),
  ...table("Chi tiết ma trận nhầm lẫn trên Test", ["Mô hình","TP","FP","FN","TN","Specificity","Accuracy"],
    names.map(n=>[ab(n),fmtInt(R[n].TP),fmtInt(R[n].FP),fmtInt(R[n].FN),fmtInt(R[n].TN),fmtDec(R[n].specificity,5),fmtDec(R[n].accuracy,5)]),
    [.7,.75,.7,.7,1,.95,.95],{size:15}),
  H2("4.3. Đường cong ROC và Precision–Recall"),
  P("Các đường cong được vẽ từ probability[1] và nhãn thật của toàn bộ Test trong phạm vi thí nghiệm. AUC trong bảng được tính bằng BinaryClassificationEvaluator của Spark với cột điểm cùng loại và numBins=0, không lấy mẫu bin. AUC-PR là diện tích theo quy ước Spark, không phải Average Precision của scikit-learn. Đường cong là mô tả kết quả cuối, không dùng tìm ngưỡng trên Test."),
  ...figure(fig("fig_roc_curves.png"), "Đường cong ROC trên Test, trục FPR theo thang log"),
  ...figure(fig("fig_pr_curves.png"), "Đường cong Precision–Recall trên Test"),
  H2("4.4. Ảnh hưởng của ngưỡng quyết định"),
  P("Ngưỡng t* tối đa F1 trên Validation của từng mô hình, dùng quy tắc điểm >= ngưỡng. Ngưỡng 1,0 vẫn hợp lệ. Bảng dưới đối chiếu ngưỡng mặc định với ngưỡng lưu; khác biệt trên Test không được dùng để tối ưu lại. Class weighting thay đổi phân bố điểm nên probability[1] cần được xem là điểm rủi ro chưa hiệu chuẩn."),
  ...table("So sánh ngưỡng 0,5 và t*", ["Mô hình","Ngưỡng","FP","FN","Precision","Recall","F1"],
    names.flatMap(n=>[[ab(n),"0,5",fmtInt(R[n].at_default_0_5.FP),fmtInt(R[n].at_default_0_5.FN),f4(R[n].at_default_0_5.precision),f4(R[n].at_default_0_5.recall),f4(R[n].at_default_0_5.f1)],
      ["",fmtDec(R[n].threshold,3),fmtInt(R[n].FP),fmtInt(R[n].FN),f4(R[n].precision),f4(R[n].recall),f4(R[n].f1)]]),[.7,.8,.7,.7,.9,.85,.8],{size:15}),
  H2("4.5. Độ quan trọng của đặc trưng"),
  P(`Độ quan trọng được lấy trực tiếp từ ${m.feature_importance_model}. Các đặc trưng có trọng số lớn nhất: ${imp.slice(0,3).map(([k,v])=>`${k} (${fmtDec(v,3)})`).join(", ")}. Đây là mức giảm độ vẩn đục tích lũy, không phải bằng chứng quan hệ nhân quả. Nếu thuật toán chọn là LR, hình dùng RF đối chiếu và ghi rõ tên.`),
  ...figure(fig("fig_feature_importance.png"), `Độ quan trọng đặc trưng của ${m.feature_importance_model}`),
  P(`Hệ số LR sau chuẩn hóa: ${coef.map(([k,v])=>`${k}=${fmtDec(v,4)}`).join("; ")}. Dấu của hệ số mô tả quan hệ trong mô hình; các dấu vết mô phỏng và sự phụ thuộc giữa biến giới hạn khả năng diễn giải sang môi trường thực tế.`),
  H2("4.6. Hiệu năng xử lý trên Spark"),
  P("Thời gian Train đo quanh Pipeline.fit trên Train; thời gian dự đoán đo quanh transform và materialize cache của Test, không gồm tính chỉ số hoặc thu thập để vẽ hình. Đây là một lần đo trên máy local, chưa có khoảng tin cậy hay so sánh với cụm nhiều máy."),
  ...table("Thời gian huấn luyện và dự đoán",["Mô hình","Train (giây)","Dự đoán Test (giây)"],
    names.map(n=>[n,fmtDec(R[n].train_time_s,1),fmtDec(R[n].predict_time_s,3)]),[1.5,1,1.2]),
  H2("4.7. Lựa chọn mô hình tốt nhất"),
  P(`Tiêu chí cố định: AUC-PR Validation cao nhất, sau đó F1 Validation, Precision Validation; không làm tròn khi lựa chọn. Mô hình được chọn là ${best}, với AUC-PR Validation ${f4(B.validation.auc_pr)}, F1 Validation ${f4(B.validation.f1)}, ngưỡng ${fmtDec(B.threshold,12)}. Quyết định này được cố định trước khi bất kỳ mô hình nào chấm điểm Test. Thí nghiệm chính B_temporal được xác định theo kịch bản trước giao dịch và đánh giá theo thời gian, không dựa trên thí nghiệm có điểm Test cao nhất.`),
  H2("4.8. Giao diện ứng dụng"),
  P("Ứng dụng giữ ba tab: một giao dịch, CSV hàng loạt, dashboard. B chỉ yêu cầu step, type và amount. A ghi rõ phân tích sau giao dịch và yêu cầu thêm số dư trước/sau. Bộ đặc trưng trong metadata phải khớp VectorAssembler đã lưu. CSV có dòng lỗi vẫn được trả về với lý do lỗi, giữ thứ tự và nhãn thực tế; ba loại chưa được huấn luyện trả về chưa đánh giá. Dashboard sử dụng đúng artifact của thí nghiệm được chọn. Ngưỡng lưu được giữ nguyên độ chính xác; đổi ngưỡng trên giao diện không thay bảng chỉ số đã lưu."),
  ...["fig_app_single.png","fig_app_batch.png","fig_app_dashboard.png"].filter(f=>fs.existsSync(fig(f))).flatMap(f=>figure(fig(f),"Giao diện ứng dụng đã kiểm tra")),
  H2("4.9. Thảo luận và so sánh thí nghiệm A/B"),
  P("A giữ 12 đặc trưng hiện tại để phân tích sau giao dịch; B dùng 3 đặc trưng biết trước khi thực hiện và loại toàn bộ số dư PaySim. Trong cùng cách chia, A/B có cùng các bản ghi Train/Validation/Test, class weights và siêu tham số. So sánh này đo tác động tổng thể của bộ số dư và các đặc trưng phái sinh; không thể quy toàn bộ chênh lệch cho riêng số dư sau, vì B đồng thời loại số dư trước để tránh dấu vết mô phỏng. Chia thời gian kiểm tra sự thay đổi phân bố về sau, còn chia phân tầng là đối chứng."),
  ...table("Bốn thuật toán theo từng thí nghiệm",["Thí nghiệm","MH","Precision","Recall","F1","AUC-ROC","AUC-PR"],allRows,
    [1.5,.55,.8,.75,.65,.85,.85],{size:15}),
  P(Object.entries(experiments).filter(([k])=>k.startsWith("B_")).map(([key,b])=>{
    const a=experiments[key.replace("B_","A_")];
    return `${key}: chênh lệch AUC-PR (A trừ B) theo LR/DT/RF/GBT là ${names.map(n=>fmtDec(a.results[n].auc_pr-b.results[n].auc_pr,4)).join("; ")}.`;
  }).join(" ")),
  H2("4.10. Khả dụng của đặc trưng và nguy cơ rò rỉ"),
  ...table("Thời điểm khả dụng của các đặc trưng",["Đặc trưng","Khả dụng / quyết định"],[
    ["newbalanceOrig, newbalanceDest","Chỉ sau ghi nhận giao dịch; loại khỏi B."],
    ["errorBalanceOrig, errorBalanceDest","Dùng số dư sau; rò rỉ thời điểm nếu dùng chặn trước giao dịch."],
    ["destZeroBalance, origZeroBalance","Kiểm tra cả số dư trước và sau; loại khỏi B."],
    ["oldbalanceOrg, oldbalanceDest","Về nghĩa là số dư trước; vẫn loại khỏi B vì dấu vết mô phỏng/hủy giao dịch và hạn chế số dư đích."],
    ["origEmptied","Chỉ dùng amount và oldbalanceOrg trong mã; khả dụng trước nếu số dư nguồn đáng tin cậy, nhưng B loại theo chính sách số dư."],
    ["type, amount, step % 24","Biết trước theo kịch bản dữ liệu; B giữ. Giờ là pha mô phỏng, không có timezone ngân hàng."],
    ["isFraud, isFlaggedFraud","Nhãn và luật đối chứng; không đưa vào véc-tơ đặc trưng."]],[1.4,2.2]),
  H2("4.11. Kiểm toán dữ liệu và tỷ lệ lớp"),
  P(`CSV gốc: ${fmtInt(q.rows)} dòng; gian lận ${fmtInt(q.raw_label_counts["1"])}, bình thường ${fmtInt(q.raw_label_counts["0"])}. Sau làm sạch: ${fmtInt(q.clean_rows)} dòng; gian lận ${fmtInt(q.clean_label_counts["1"])}. Tỷ lệ gian lận trước/sau: ${fmtDec(100*q.raw_fraud_rate,6)}% / ${fmtDec(100*q.clean_fraud_rate,6)}%. Nguyên nhân loại từng nhóm: ${q.removal_groups.map(g=>`${g.reason}: ${fmtInt(g.removed)}`).join("; ")}. ${historicalSplit ? `Trọng số lịch sử ${fmtDec(historicalSplit.class_weights["1"],12)} đúng theo ${fmtInt(historicalSplit.train)}/(2×${fmtInt(historicalSplit.train_fraud)}).` : "Không có artifact trọng số lịch sử để đối chiếu."} Việc loại ${fmtInt(q.zero_amount_label_counts["1"] || 0)} gian lận trước huấn luyện là vấn đề trong thống kê cũ.`),
  ...table("Quy mô và khoảng thời gian theo thí nghiệm",["Thí nghiệm","Tập","GD / gian lận","Khoảng step"],
    Object.entries(experiments).flatMap(([key,e])=>["train","valid","test"].map(k=>[key,k,
      `${fmtInt(e.split[k])} / ${fmtInt(e.split[k+"_fraud"])}`,e.split[k+"_step_range"].join("–")])),[1.3,.7,1.4,1],{size:16}),
];

const conclusion = [
  H1("V. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN"),
  H2("5.1. Kết quả đạt được"),
  P(`Quy trình đã chạy bốn thuật toán Spark MLlib trong A/B và hai cách chia dữ liệu, giữ được kết quả cũ để đối chiếu. Việc loại amount=0 trước đây làm mất 16 gian lận và đã được sửa. Mô hình ${best} được chọn cho ${m.experiment} bằng Validation; Test ghi nhận F1 ${f4(B.f1)} và AUC-PR ${f4(B.auc_pr)}. Chất lượng phải được đánh giá cùng phạm vi mẫu và thời điểm khả dụng của đầu vào. Đầu ra của A chỉ có ý nghĩa phân tích sau giao dịch; B phục vụ minh họa chấm điểm trước giao dịch với bộ đầu vào thận trọng.`),
  H2("5.2. Hạn chế"),
  bullet("PaySim là dữ liệu mô phỏng từ một kịch bản chiếm tài khoản; các quy luật về loại, số dư, giờ và hủy giao dịch có thể làm kết quả quá lạc quan."),
  bullet(`Thực nghiệm hiện tại dùng mẫu đồng đều ${fmtDec(100*m.data.sample_fraction,1)}% CSV thật, một seed và một lần chia; chưa có độ bất định nhiều seed. CSV được ghi SHA-256 nhưng chưa xác thực byte với nguồn phát hành.`),
  bullet("A/B đồng thời loại số dư trước và sau cùng đặc trưng phái sinh; không tách riêng mức tác động nhân quả của từng biến."),
  bullet("Không có dữ liệu ngân hàng thật, chi phí FP/FN hoặc lịch sử hành vi khách hàng. Số dư đích trước giao dịch cũng có thể không sẵn có giữa các tổ chức."),
  bullet("Điểm từ mô hình có trọng số lớp chưa được hiệu chuẩn; tỷ lệ gian lận mô phỏng không phải tỷ lệ gian lận triển khai."),
  bullet("Thực nghiệm ở chế độ local; chưa đo thông lượng, độ trễ, khả năng mở rộng trên cụm hoặc biến động theo nhiều lần đo."),
  H2("5.3. Khả năng ứng dụng thực tế"),
  P("Có thể tái sử dụng quy trình Spark, cơ chế lưu PipelineModel và kiểm tra hợp đồng đầu vào cho một nghiên cứu trên dữ liệu ngân hàng được phép sử dụng. Trước triển khai, cần xác nhận đầu vào thực sự có sẵn tại thời điểm chấm điểm, tạo lịch sử chỉ từ quá khứ, đánh giá theo thời gian và theo tài khoản khi phù hợp, hiệu chuẩn điểm trên tập độc lập, chọn tiêu chí theo chi phí nghiệp vụ và kiểm tra thay đổi phân bố. Chưa có bằng chứng cho phép kết luận độ chính xác 100% hoặc hiệu quả tương đương trong ngân hàng thực tế."),
];
module.exports = { results, conclusion, references };

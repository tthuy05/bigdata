"""
ĐỐI CHIẾU TỪNG BẢNG TRONG BÁO CÁO WORD VỚI NGUỒN SỐ LIỆU
---------------------------------------------------------
Với mỗi bảng trong báo cáo: ghi rõ số lấy từ tệp nào, khóa (key) nào, do đoạn code nào tính ra,
dựng lại bảng đúng định dạng như trong Word, và (nếu chỉ định --docx) tô màu từng ô:
    xanh = giá trị này CÓ trong file Word,  đỏ = KHÔNG tìm thấy trong Word.
Kết quả: outputs/doi_chieu_bang_word.html  (mở bằng trình duyệt)

Cách dùng (tại thư mục fraud_detection_spark):
    python tools/doi_chieu_bang_word.py
    python tools/doi_chieu_bang_word.py --docx "D:/duong/dan/BaoCao_BigData_Nhom8.docx"
Không cần chạy Spark – chỉ đọc outputs/metrics/*.json.
"""
import argparse
import html
import json
import os
import re
import sys
import webbrowser
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ định dạng giống báo cáo
def fi(n):
    return f"{int(round(n)):,}".replace(",", ".")


def fd(x, k=4):
    i, f = f"{float(x):.{k}f}".split(".")
    return fi(int(i)) + "," + f if k else fi(int(i))


def load(metric_dir, name):
    p = os.path.join(metric_dir, name)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        raise SystemExit(f"Thiếu hoặc rỗng: {p}. Hãy chạy python main.py trước.")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ đọc file Word
def docx_texts(path):
    """Trả về (tập văn bản từng ô/đoạn, toàn văn) của file .docx."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    paras = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, flags=re.S))
        paras.append(html.unescape(t).strip())
    return set(paras), "\n".join(paras)


# ------------------------------------------------------------------ định nghĩa các bảng
def build_tables(q, e, m, env):
    R, S = m["results"], m["split"]
    ab = {"Logistic Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF", "GBT": "GBT"}
    names = list(R)
    T = []

    def add(no, title, source, code, headers, rows, note=""):
        T.append(dict(no=no, title=title, source=source, code=code, headers=headers,
                      rows=rows, note=note, data=bool(rows)))

    add(1, "Đối tượng và phạm vi nghiên cứu", "Nội dung mô tả (viết tay)", "–", [], [])
    add(2, "So sánh đặc điểm các thuật toán sử dụng", "Lý thuyết (viết tay)", "–", [], [])
    add(3, "Ma trận nhầm lẫn cho bài toán phát hiện gian lận", "Lý thuyết (viết tay)", "–", [], [])

    java = str(env.get("java", "")).replace('"', "")
    add(4, "Môi trường và công cụ thực nghiệm", "outputs/metrics/env.json",
        "main.py bước 2 (env.json) + src/envcheck.py (java -version)",
        ["Thành phần", "Giá trị", "Khóa JSON"],
        [["Apache Spark (PySpark)", env.get("spark_version", ""), "spark_version"],
         ["Python", env.get("python", ""), "python"],
         ["Java", java, "java"],
         ["Chế độ chạy Spark", f"{env.get('master', '')}, bộ nhớ Driver {env.get('driver_memory', '')}",
          "master, driver_memory"],
         ["Số nhân CPU", str(env.get("cores", "")), "cores"]])

    add(5, "Các thuộc tính của bộ dữ liệu PaySim", "Lược đồ khai báo trong code",
        "src/data_loader.py – PAYSIM_SCHEMA", [], [])

    add(6, "Kết quả kiểm tra chất lượng dữ liệu", "outputs/metrics/quality.json",
        "src/data_loader.py – quality_report() và clean()",
        ["Tiêu chí", "Giá trị", "Khóa JSON / cách tính"],
        [["Số bản ghi ban đầu", fi(q["rows"]), "rows = df.count()"],
         ["Số thuộc tính", str(q["columns"]), "columns = len(df.columns)"],
         ["Giá trị thiếu", fi(sum(q["nulls"].values())), "tổng nulls (đếm isNull từng cột)"],
         ["Bản ghi trùng lặp", fi(q["duplicates"]), "duplicates = count - dropDuplicates().count()"],
         ["Giá trị âm ở cột tiền tệ", fi(sum(q["negatives"].values())), "tổng negatives"],
         ["Giao dịch có amount ≤ 0", fi(q["zero_amount"]), "zero_amount"],
         ["Số bản ghi sau làm sạch", fi(q["clean_rows"]), "clean_rows (đếm sau clean())"],
         ["Thời gian kiểm tra (giây)", fd(q.get("time_s", 0), 1), "time_s"]])

    add(7, "Thống kê giao dịch theo loại (Spark SQL)", "outputs/metrics/eda_stats.json → by_type",
        "src/stats.py – compute_stats(), mục 2:  SELECT type, COUNT(*), SUM(isFraud), "
        "ROUND(100*AVG(isFraud),4), AVG(amount) FROM transactions GROUP BY type",
        ["Loại", "Số giao dịch", "Gian lận", "Tỷ lệ GL (%)", "TB số tiền"],
        [[r["type"], fi(r["n"]), fi(r["fraud"]), fd(r["fraud_rate_pct"], 3), fi(r["avg_amount"])]
         for r in e["by_type"]])

    add(8, "Thống kê số tiền theo nhãn", "outputs/metrics/eda_stats.json → amount_by_label",
        "src/stats.py mục 3:  SELECT isFraud, AVG(amount), percentile_approx(amount,0.5), "
        "STDDEV(amount), MAX(amount) FROM transactions GROUP BY isFraud",
        ["Nhãn", "Trung bình", "Trung vị", "Độ lệch chuẩn", "Lớn nhất"],
        [["Gian lận" if r["isFraud"] else "Bình thường", fi(r["mean"]), fi(r["median"]),
          fi(r["std"]), fi(r["max"])] for r in e["amount_by_label"]])

    bp = {r["isFraud"]: r for r in e["balance_patterns"]}
    add(9, "Các mẫu số dư bất thường theo nhãn (TRANSFER, CASH_OUT)",
        "outputs/metrics/eda_stats.json → balance_patterns",
        "src/stats.py mục 5:  SELECT isFraud, 100*AVG(origZeroBalance), 100*AVG(destZeroBalance), "
        "100*AVG(origEmptied), ... FROM rel GROUP BY isFraud  (rel = TRANSFER + CASH_OUT đã tạo đặc trưng)",
        ["Mẫu (% giao dịch)", "Bình thường", "Gian lận", "Khóa JSON"],
        [[n, fd(bp[0][k], 2), fd(bp[1][k], 2), k] for n, k in [
            ("Số dư nguồn 0 → 0", "pct_orig_zero"), ("Số dư đích 0 → 0", "pct_dest_zero"),
            ("Rút sạch tài khoản nguồn", "pct_orig_emptied"),
            ("Sai lệch số dư nguồn ≠ 0", "pct_err_orig"), ("Sai lệch số dư đích ≠ 0", "pct_err_dest")]])

    add(10, "Danh sách đặc trưng đầu vào của mô hình", "Định nghĩa trong code",
        "src/features.py – add_features(), FEATURE_NAMES", [], [])

    add(11, "Quy mô các tập dữ liệu sau khi chia", "outputs/metrics/model_results.json → split",
        "src/modeling.py – stratified_split() (randomSplit seed=42 theo từng lớp) rồi .count()",
        ["Tập", "Số giao dịch", "Số gian lận", "Tỷ lệ (%)"],
        [[lbl, fi(S[k]), fi(S[k + "_fraud"]), fd(100 * S[k + "_fraud"] / S[k], 3)] for lbl, k in
         [("Huấn luyện (70%)", "train"), ("Kiểm định (10%)", "valid"), ("Kiểm thử (20%)", "test")]],
        note=f"Trọng số lớp (dùng trong đoạn văn mục 3.9): gian lận = {fd(S['class_weights']['1'], 2)}, "
             f"bình thường = {fd(S['class_weights']['0'], 4)}  (split.class_weights)")

    add(12, "Siêu tham số của các mô hình", "Khai báo trong code",
        "src/modeling.py – build_models()", [], [])

    rb = m["rule_baseline"]
    rows = [[ab[n], fd(R[n]["threshold"], 3), fd(R[n]["precision"]), fd(R[n]["recall"]),
             fd(R[n]["f1"]), fd(R[n]["auc_roc"]), fd(R[n]["auc_pr"])] for n in names]
    rows.append(["Luật hệ thống", "–", fd(rb["precision"]), fd(rb["recall"]), fd(rb["f1"]), "–", "–"])
    add(13, "Kết quả các mô hình trên tập kiểm thử", "outputs/metrics/model_results.json → results, rule_baseline",
        "src/modeling.py – tune_threshold() (t*, trên tập kiểm định), evaluate() (Precision/Recall/F1 từ "
        "ma trận nhầm lẫn; AUC-ROC, AUC-PR bằng BinaryClassificationEvaluator), rule_baseline()",
        ["Mô hình", "t*", "Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"], rows,
        note=f"Mô hình được chọn (best_model): {m['best_model']}  – tiêu chí AUC-PR, sau đó F1")

    add(14, "Chi tiết ma trận nhầm lẫn trên tập kiểm thử", "outputs/metrics/model_results.json → results",
        "src/modeling.py – evaluate(): đếm groupBy(isFraud, pred); Specificity = TN/(TN+FP); "
        "Accuracy = (TP+TN)/tổng",
        ["Mô hình", "TP", "FP", "FN", "TN", "Specificity", "Accuracy"],
        [[ab[n], fi(R[n]["TP"]), fi(R[n]["FP"]), fi(R[n]["FN"]), fi(R[n]["TN"]),
          fd(R[n]["specificity"], 5), fd(R[n]["accuracy"], 5)] for n in names])

    rows = []
    for n in names:
        d = R[n]["at_default_0_5"]
        rows.append([ab[n], "0,5", fi(d["FP"]), fi(d["FN"]), fd(d["precision"]), fd(d["recall"]), fd(d["f1"])])
        rows.append(["", fd(R[n]["threshold"], 3), fi(R[n]["FP"]), fi(R[n]["FN"]), fd(R[n]["precision"]),
                     fd(R[n]["recall"]), fd(R[n]["f1"])])
    add(15, "So sánh kết quả tại ngưỡng 0,5 và ngưỡng t*",
        "outputs/metrics/model_results.json → results[...].at_default_0_5 và results[...]",
        "src/modeling.py – evaluate(pred, 0.5) và evaluate(pred, t*)",
        ["Mô hình", "Ngưỡng", "FP", "FN", "Precision", "Recall", "F1"], rows)

    add(16, "Thời gian huấn luyện và dự đoán", "outputs/metrics/model_results.json → train_time_s, predict_time_s",
        "src/modeling.py – đo time.time() quanh pipe.fit(train) và evaluate(test)",
        ["Mô hình", "Huấn luyện (giây)", "Dự đoán tập kiểm thử (giây)"],
        [[n, fd(R[n]["train_time_s"], 1), fd(R[n]["predict_time_s"], 1)] for n in names])
    return T


def paragraph_numbers(q, e, m):
    """Các con số quan trọng xuất hiện trong ĐOẠN VĂN (không nằm trong bảng)."""
    R, best = m["results"], m["best_model"]
    lab = {r["isFraud"]: r for r in e["label"]}
    top = sorted(e["by_hour"], key=lambda r: -r["fraud_rate_pct"])[:3]
    flagged = sum(r["n"] for r in e["flagged"] if r["isFlaggedFraud"] == 1)
    imp = sorted(m["feature_importance"].items(), key=lambda x: -x[1])[:3]
    return [
        ("Tóm tắt, 3.6", "Số giao dịch gian lận", fi(lab[1]["n"]), "eda_stats.label"),
        ("Tóm tắt, 3.6", "Tỷ lệ gian lận (%)", fd(lab[1]["pct"], 3), "eda_stats.label.pct"),
        ("3.6", "Số giao dịch bình thường", fi(lab[0]["n"]), "eda_stats.label"),
        ("3.6", "Gian lận đúng hạn mức 10 triệu", fi(e["max_fraud_amount"][0]["n_10M"]), "eda_stats.max_fraud_amount"),
        ("3.7", "Số giao dịch TRANSFER + CASH_OUT", fi(e["relevant_rows"]), "eda_stats.relevant_rows"),
        ("3.7", "3 giờ tỷ lệ gian lận cao nhất", ", ".join(f"{r['hour']} giờ" for r in top), "eda_stats.by_hour"),
        ("1.2, 3.7", "Số giao dịch luật isFlaggedFraud gắn cờ", fi(flagged), "eda_stats.flagged"),
        ("3.3", "Số tài khoản nguồn khác nhau", fi(e["unique_accounts"][0]["orig"]), "eda_stats.unique_accounts"),
        ("3.9", "Trọng số lớp gian lận", fd(m["split"]["class_weights"]["1"], 2), "model_results.split.class_weights"),
        ("Tóm tắt, 4.1, 4.7, 5.1", f"F1 của {best}", fd(R[best]["f1"]), "model_results.results"),
        ("Tóm tắt, 4.1, 4.7", f"Precision của {best}", fd(R[best]["precision"]), "model_results.results"),
        ("Tóm tắt, 4.1, 4.7", f"Recall của {best}", fd(R[best]["recall"]), "model_results.results"),
        ("Tóm tắt, 4.1, 5.1", f"AUC-PR của {best}", fd(R[best]["auc_pr"]), "model_results.results"),
        ("4.7", f"Ngưỡng t* của {best}", fd(R[best]["threshold"], 3), "model_results.results.threshold"),
        ("Tóm tắt", "Recall của luật isFlaggedFraud", fd(m["rule_baseline"]["recall"]), "model_results.rule_baseline"),
        ("4.5", f"Độ quan trọng của đặc trưng hạng 1 ({imp[0][0]})", fd(imp[0][1], 3), "model_results.feature_importance"),
        ("4.5", f"Độ quan trọng của đặc trưng hạng 2 ({imp[1][0]})", fd(imp[1][1], 3), "model_results.feature_importance"),
        ("4.5", f"Độ quan trọng của đặc trưng hạng 3 ({imp[2][0]})", fd(imp[2][1], 3), "model_results.feature_importance"),
    ]


# ------------------------------------------------------------------ xuất HTML
CSS = """body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
h1{font-size:22px}h2{font-size:17px;margin:28px 0 6px;border-left:5px solid #4C72B0;padding-left:8px}
.meta{font-size:13px;margin:2px 0}.meta b{display:inline-block;width:150px}
code{background:#f1f3f5;padding:1px 4px;border-radius:3px;font-size:12px}
table{border-collapse:collapse;margin:8px 0;font-size:13px}th,td{border:1px solid #bbb;padding:4px 9px;text-align:right}
th{background:#D9E2F3}td:first-child,th:first-child{text-align:left}
.ok{background:#e3f6e5}.miss{background:#fde2e2}.note{font-size:13px;color:#444;margin:4px 0}
.sum{padding:10px 14px;border-radius:6px;background:#f5f7fa;font-size:14px}.nodata{color:#777;font-size:13px}
.legend span{padding:2px 8px;margin-right:8px;border:1px solid #bbb}"""


def render(tables, paras, docx_set, docx_full, docx_name, out_path):
    checked = docx_set is not None
    tot = {"ok": 0, "miss": 0}

    def cell(v, check=True):
        v = str(v)
        cls = ""
        if checked and check and v not in ("", "–"):
            hit = v in docx_set or v in docx_full
            cls = "ok" if hit else "miss"
            tot["ok" if hit else "miss"] += 1
        return f'<td class="{cls}">{html.escape(v)}</td>'

    parts = [f"<html><head><meta charset='utf-8'><title>Đối chiếu bảng Word</title><style>{CSS}</style></head><body>",
             "<h1>Đối chiếu số liệu từng bảng trong báo cáo Word</h1>"]
    if checked:
        parts.append(f"<p class='legend'>File Word: <code>{html.escape(os.path.basename(docx_name))}</code><br>"
                     "<span class='ok'>xanh</span> = giá trị có trong Word "
                     "<span class='miss'>đỏ</span> = KHÔNG tìm thấy trong Word (cần kiểm tra/sửa)</p>")
    else:
        parts.append("<p class='legend'>Chưa chỉ định file Word (--docx) nên không tô màu đối chiếu.</p>")
    parts.append("<div class='sum' id='sum'></div>")

    for t in tables:
        parts.append(f"<h2>Bảng {t['no']}. {html.escape(t['title'])}</h2>")
        parts.append(f"<p class='meta'><b>Nguồn số liệu:</b> <code>{html.escape(t['source'])}</code></p>")
        parts.append(f"<p class='meta'><b>Code tính ra:</b> <code>{html.escape(t['code'])}</code></p>")
        if not t["data"]:
            parts.append("<p class='nodata'>Bảng mô tả/lý thuyết – không có số liệu tính toán, không cần đối chiếu.</p>")
            continue
        parts.append("<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in t["headers"]) + "</tr>")
        last_is_key = t["headers"][-1].startswith("Khóa")
        for r in t["rows"]:
            tds = []
            for i, v in enumerate(r):
                is_label = i == 0
                is_key = last_is_key and i == len(r) - 1
                tds.append(cell(v, check=not (is_label or is_key)))
            parts.append("<tr>" + "".join(tds) + "</tr>")
        parts.append("</table>")
        if t["note"]:
            parts.append(f"<p class='note'>{html.escape(t['note'])}</p>")

    parts.append("<h2>Các con số quan trọng trong đoạn văn</h2><table><tr><th>Mục</th><th>Nội dung</th>"
                 "<th>Giá trị</th><th>Nguồn (JSON)</th></tr>")
    for sec, lab, val, src in paras:
        parts.append(f"<tr><td>{html.escape(sec)}</td><td>{html.escape(lab)}</td>{cell(val)}"
                     f"<td>{html.escape(src)}</td></tr>")
    parts.append("</table>")
    parts.append("<h2>Ghi chú</h2><ul class='note'>"
                 "<li>Mọi tệp JSON được sinh ra khi chạy <code>python main.py</code> (bước 5, 6, 10, 11).</li>"
                 "<li>Hình trong báo cáo được vẽ từ cùng các khóa JSON này (xem cột Nguồn).</li>"
                 "<li>Ô đỏ thường do: Word chưa cập nhật số mới, hoặc gõ định dạng khác "
                 "(ví dụ 0.9975 thay vì 0,9975).</li></ul>")
    summary = (f"Tổng số ô đã đối chiếu: <b>{tot['ok'] + tot['miss']}</b> — "
               f"<span style='color:green'>khớp với Word: <b>{tot['ok']}</b></span> — "
               f"<span style='color:#c00'>không tìm thấy: <b>{tot['miss']}</b></span>") if checked else \
        "Thêm tham số --docx để tô màu đối chiếu với file Word."
    page = "\n".join(parts).replace("<div class='sum' id='sum'></div>", f"<div class='sum'>{summary}</div>")
    page += "</body></html>"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    return tot


def main():
    print("Lưu ý: công cụ này chỉ tìm chuỗi số xuất hiện trong Word; dùng verify_artifacts.py để kiểm tra chính xác ô bảng.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=os.path.join(BASE, "outputs", "metrics"))
    ap.add_argument("--docx", default=None, help="đường dẫn file Word báo cáo để tô màu đối chiếu")
    ap.add_argument("--out", default=os.path.join(BASE, "outputs", "doi_chieu_bang_word.html"))
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    q, e, m = (load(a.metrics, n) for n in ("quality.json", "eda_stats.json", "model_results.json"))
    env_path = os.path.join(a.metrics, "env.json")
    env = load(a.metrics, "env.json") if os.path.exists(env_path) else {}
    tables, paras = build_tables(q, e, m, env), paragraph_numbers(q, e, m)
    dset = dfull = None
    if a.docx:
        dset, dfull = docx_texts(a.docx)
    tot = render(tables, paras, dset, dfull, a.docx or "", a.out)

    print(f"Đã tạo: {a.out}")
    for t in tables:
        print(f"  Bảng {t['no']:>2}. {t['title'][:55]:55s} <- {t['source']}")
    if a.docx:
        print(f"Đối chiếu với Word: {tot['ok']} ô khớp, {tot['miss']} ô KHÔNG tìm thấy (ô đỏ trong file HTML)")
    if not a.no_open:
        webbrowser.open("file:///" + os.path.abspath(a.out).replace("\\", "/"))


if __name__ == "__main__":
    main()

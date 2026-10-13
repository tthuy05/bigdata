"""Đối chiếu số liệu lần chạy mới (outputs/metrics) với số liệu gốc của báo cáo (outputs/baseline).
KHỚP = giống hệt;  LỆCH NHỎ = sai khác rất nhỏ ở số liệu phụ thuộc huấn luyện;  LỆCH = khác thật sự."""
from src import console as c


def _items(q, e, m):
    it = {}

    def add(group, ref, label, value, strict=True):
        it[f"{group}|{label}"] = (group, ref, label, value, strict)

    g, ref = "Chất lượng dữ liệu", "Bảng 6"
    add(g, ref, "Số dòng gốc", q["rows"]); add(g, ref, "Số cột", q["columns"])
    add(g, ref, "Tổng giá trị thiếu", sum(q["nulls"].values()))
    add(g, ref, "Số dòng trùng", q["duplicates"])
    add(g, ref, "Giá trị tiền âm", sum(q["negatives"].values()))
    add(g, ref, "amount <= 0", q["zero_amount"])
    add(g, ref, "Số dòng sau làm sạch", q["clean_rows"])
    g, ref = "Phân bố nhãn", "fig_label_distribution"
    for r in e["label"]:
        add(g, ref, f"isFraud={r['isFraud']} – số giao dịch", r["n"])
        add(g, ref, f"isFraud={r['isFraud']} – tỷ lệ %", float(r["pct"]))
    g, ref = "Theo loại giao dịch", "Bảng 7, fig_type_distribution"
    for r in e["by_type"]:
        add(g, ref, f"{r['type']} – số giao dịch", r["n"])
        add(g, ref, f"{r['type']} – số gian lận", r["fraud"])
        add(g, ref, f"{r['type']} – TB số tiền", round(float(r["avg_amount"])))
    g, ref = "Số tiền theo nhãn", "Bảng 8"
    for r in e["amount_by_label"]:
        for k in ("mean", "median", "max"):
            add(g, ref, f"isFraud={r['isFraud']} – {k}", round(float(r[k])))
    g, ref = "Theo giờ", "fig_fraud_by_hour"
    for r in e["by_hour"]:
        add(g, ref, f"giờ {r['hour']:02d} – tỷ lệ gian lận %", float(r["fraud_rate_pct"]))
    g, ref = "Mẫu số dư", "Bảng 9, fig_balance_patterns"
    for r in e["balance_patterns"]:
        for k in ("pct_orig_zero", "pct_dest_zero", "pct_orig_emptied", "pct_err_orig", "pct_err_dest"):
            add(g, ref, f"isFraud={r['isFraud']} – {k}", float(r[k]))
    g, ref = "Chia dữ liệu", "Bảng 11"
    for k in ("train", "train_fraud", "valid", "valid_fraud", "test", "test_fraud"):
        add(g, ref, k, m["split"][k], strict=False)
    g, ref = "Ma trận nhầm lẫn", "Bảng 14, fig_confusion_matrices"
    for name, r in m["results"].items():
        for k in ("TP", "FP", "FN", "TN"):
            add(g, ref, f"{name} – {k}", r[k], strict=False)
    g, ref = "Chỉ số mô hình", "Bảng 13, fig_metrics_comparison"
    for name, r in m["results"].items():
        for k in ("threshold", "precision", "recall", "f1", "auc_roc", "auc_pr"):
            add(g, ref, f"{name} – {k}", round(float(r[k]), 4), strict=False)
    g, ref = "Độ quan trọng đặc trưng", "fig_feature_importance"
    for k, v in m["feature_importance"].items():
        add(g, ref, k, round(float(v), 4), strict=False)
    add("Mô hình được chọn", "mục 4.7", "best_model", m["best_model"])
    return it


def _judge(old, new, strict):
    if isinstance(old, str) or isinstance(new, str):
        return "KHỚP" if old == new else "LỆCH"
    d = abs(old - new)
    if d <= 1e-9:
        return "KHỚP"
    if strict:
        return "LỆCH"
    small = (d <= 5 or d <= 0.01 * max(abs(old), 1)) if isinstance(old, int) else d <= 0.002
    return "LỆCH NHỎ" if small else "LỆCH"


def compare(base, new):
    """base, new: (quality, eda_stats, model_results). In bảng theo nhóm; trả về tổng số."""
    old_it, new_it = _items(*base), _items(*new)
    col = {"KHỚP": c.GREEN, "LỆCH NHỎ": c.YELLOW, "LỆCH": c.RED}
    mark = {"KHỚP": "✔", "LỆCH NHỎ": "≈", "LỆCH": "✘"}
    tot = {"KHỚP": 0, "LỆCH NHỎ": 0, "LỆCH": 0}
    groups = {}
    for key, (g, ref, label, old, strict) in old_it.items():
        newv = new_it[key][3] if key in new_it else "(không có)"
        v = _judge(old, newv, strict) if key in new_it else "LỆCH"
        tot[v] += 1
        groups.setdefault((g, ref), []).append((label, old, newv, v))
    for (g, ref), rows in groups.items():
        k = sum(r[3] == "KHỚP" for r in rows)
        hc = c.GREEN if k == len(rows) else (c.RED if any(r[3] == "LỆCH" for r in rows) else c.YELLOW)
        print(f"\n   {c.BOLD}{g}{c.RESET} ({ref}): {hc}{k}/{len(rows)} khớp{c.RESET}")
        shown = [r for r in rows if r[3] != "KHỚP"] or rows[:3]
        if len(shown) < len(rows) and all(r[3] == "KHỚP" for r in rows):
            print(f"      (hiển thị 3/{len(rows)} dòng – tất cả đều khớp)")
        w = max(len(r[0]) for r in shown)
        for label, old, newv, v in shown:
            print(f"      {col[v]}{mark[v]}{c.RESET} {label:<{w}}  báo cáo = {c._fmt(old):>12}  "
                  f"chạy lại = {c._fmt(newv):>12}  {col[v]}{v}{c.RESET}")
    return tot

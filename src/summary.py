"""Tạo bản tóm tắt số liệu (outputs/ket_qua.txt) để chép vào báo cáo Word."""


def build_text(q: dict, e: dict, m: dict, env: dict) -> str:
    L = []
    w = L.append
    line = "=" * 70
    w(line); w("KẾT QUẢ ĐỒ ÁN – PHÁT HIỆN GIAN LẬN GIAO DỊCH TRÊN APACHE SPARK"); w(line)
    if m.get("schema_version") == 2:
        w(f"Thí nghiệm: {m['experiment']} | run_id: {m['run_id']} | trạng thái: {m['status']}")
        w(f"Mẫu đồng đều dùng huấn luyện: {m['data']['sample_fraction']:.0%} CSV thật; thống kê gốc dùng toàn bộ CSV.")
        w("Chọn mô hình trên VALIDATION (AUC-PR, F1, Precision); Test chỉ đánh giá cuối.")
    else:
        w("KẾT QUẢ LỊCH SỬ: chưa tái kiểm chứng; có đặc trưng sau giao dịch và lựa chọn mô hình dựa trên Test.")

    w("\n1. MÔI TRƯỜNG")
    for k, v in env.items():
        w(f"   {k:16s}: {v}")

    w("\n2. CHẤT LƯỢNG DỮ LIỆU")
    w(f"   Số dòng ban đầu      : {q['rows']:,}")
    w(f"   Số cột               : {q['columns']}")
    w(f"   Giá trị thiếu        : {sum(q['nulls'].values())}")
    w(f"   Bản ghi trùng lặp    : {q['duplicates']}")
    w(f"   Giá trị âm           : {sum(q['negatives'].values())}")
    w(f"   amount <= 0          : {q['zero_amount']}")
    w(f"   Số dòng sau làm sạch : {q['clean_rows']:,}")
    if "raw_label_counts" in q:
        w(f"   Nhãn dữ liệu gốc: {q['raw_label_counts']}; sau làm sạch: {q['clean_label_counts']}")
        for group in q["removal_groups"]:
            w(f"   Loại bỏ {group['reason']}: {group['removed']} ({group['labels']})")

    w("\n3. PHÂN BỐ NHÃN (Hình fig_label_distribution)")
    for r in e["label"]:
        w(f"   isFraud={r['isFraud']}: {r['n']:>10,} ({r['pct']}%)")

    w("\n4. THỐNG KÊ THEO LOẠI GIAO DỊCH (Hình fig_type_distribution)")
    w(f"   {'Loại':10s}{'Số GD':>12s}{'Gian lận':>10s}{'Tỷ lệ %':>10s}{'TB số tiền':>14s}")
    for r in e["by_type"]:
        w(f"   {r['type']:10s}{r['n']:>12,}{r['fraud']:>10,}{r['fraud_rate_pct']:>10}{r['avg_amount']:>14,.0f}")

    w("\n5. SỐ TIỀN THEO NHÃN (Hình fig_amount_distribution)")
    for r in e["amount_by_label"]:
        w(f"   isFraud={r['isFraud']}: TB={r['mean']:,.0f}  trung vị={r['median']:,.0f}  "
          f"độ lệch chuẩn={r['std']:,.0f}  max={r['max']:,.0f}")
    w(f"   Số gian lận đúng hạn mức 10.000.000: {e['max_fraud_amount'][0]['n_10M']}")

    w("\n6. THEO GIỜ (Hình fig_fraud_by_hour) – 5 giờ có tỷ lệ gian lận cao nhất")
    for r in sorted(e["by_hour"], key=lambda r: -r["fraud_rate_pct"])[:5]:
        w(f"   {r['hour']:2d}h: {r['n']:>9,} giao dịch, {r['fraud']:>5,} gian lận, tỷ lệ {r['fraud_rate_pct']}%")

    w("\n7. MẪU SỐ DƯ – TRANSFER + CASH_OUT (Hình fig_balance_patterns), % giao dịch")
    for r in e["balance_patterns"]:
        w(f"   isFraud={r['isFraud']}: nguồn 0→0={r['pct_orig_zero']}  đích 0→0={r['pct_dest_zero']}  "
          f"rút sạch={r['pct_orig_emptied']}  lệch nguồn={r['pct_err_orig']}  lệch đích={r['pct_err_dest']}")
    fl = sum(r["n"] for r in e["flagged"] if r["isFlaggedFraud"] == 1)
    w(f"   isFlaggedFraud gắn cờ: {fl} giao dịch")

    s = m["split"]
    w("\n8. CHIA DỮ LIỆU")
    for k in ("train", "valid", "test"):
        w(f"   {k:6s}: {s[k]:>10,} giao dịch, {s[k + '_fraud']:,} gian lận")
    w(f"   Trọng số lớp: gian lận={s['class_weights']['1']:.2f}, bình thường={s['class_weights']['0']:.4f}")

    w("\n9. KẾT QUẢ MÔ HÌNH TRÊN TẬP KIỂM THỬ (Hình fig_metrics_comparison, fig_confusion_matrices)")
    w(f"   {'Mô hình':20s}{'t*':>7s}{'Prec':>8s}{'Recall':>8s}{'F1':>8s}{'AUC-ROC':>9s}{'AUC-PR':>8s}"
      f"{'TP':>6s}{'FP':>5s}{'FN':>5s}{'Train(s)':>9s}")
    for n, r in m["results"].items():
        w(f"   {n:20s}{r['threshold']:>7.3f}{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}"
          f"{r['auc_roc']:>9.4f}{r['auc_pr']:>8.4f}{r['TP']:>6}{r['FP']:>5}{r['FN']:>5}{r['train_time_s']:>9.1f}")
    rb = m["rule_baseline"]
    w(f"   {'Luật isFlaggedFraud':20s}{'-':>7s}{rb['precision']:>8.4f}{rb['recall']:>8.4f}{rb['f1']:>8.4f}")
    w("\n   Tại ngưỡng mặc định 0,5:")
    for n, r in m["results"].items():
        d = r["at_default_0_5"]
        w(f"   {n:20s} P={d['precision']:.4f} R={d['recall']:.4f} F1={d['f1']:.4f} FP={d['FP']} FN={d['FN']}")
    w(f"\n   Mô hình được chọn: {m['best_model']}; nguồn lựa chọn: {m.get('selection', {}).get('dataset', 'Test, lịch sử')}")

    w(f"\n10. ĐỘ QUAN TRỌNG ĐẶC TRƯNG – {m['feature_importance_model']} (Hình fig_feature_importance)")
    for i, (k, v) in enumerate(sorted(m["feature_importance"].items(), key=lambda x: -x[1]), 1):
        w(f"   {i:2d}. {k:24s}{v:.4f}")
    w("\n   Hệ số Logistic Regression (đã chuẩn hóa; dương = tăng khả năng gian lận):")
    for k, v in sorted(m["lr_coefficients"].items(), key=lambda x: -abs(x[1])):
        w(f"       {k:24s}{v:+.4f}")
    w(line)
    return "\n".join(L)

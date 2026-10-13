"""Vẽ 11 hình từ số liệu ĐÃ TÍNH và ĐÃ IN ở các bước trước (không tính lại gì).
Mỗi hình in ra: số liệu lấy từ đâu và các giá trị thể hiện trên hình để đối chiếu nhanh."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import config
from src import console as c
from src.features import FEATURE_NAMES

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                     "font.family": "DejaVu Sans"})
FIG_SIZE = (3.4, 2.3)
C_NORMAL, C_FRAUD = "#4C72B0", "#C44E52"
COLORS = {"Logistic Regression": "#4C72B0", "Decision Tree": "#55A868",
          "Random Forest": "#C44E52", "GBT": "#8172B2"}
vn = lambda v: f"{int(v):,}".replace(",", ".")


def spark_pr_curve(labels, scores):
    """Spark prepends (0, precision at highest score), rather than sklearn's (0,1)."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    precision[-1] = precision[-2]
    return precision, recall


def _save(fig, name, source, values=""):
    path = os.path.join(config.FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    c.figure_saved(path, source, values)
    return path


def hour_figure(bh):
    """step mod 24 is a simulation phase, without a known banking timezone."""
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    ax1.bar([r["hour"] for r in bh], [r["n"] / 1e3 for r in bh], color=C_NORMAL, alpha=.6)
    ax1.set_xlabel("Pha 24 giờ (step mod 24)")
    ax1.set_ylabel("Nghìn giao dịch")
    ax2 = ax1.twinx()
    ax2.plot([r["hour"] for r in bh], [r["fraud_rate_pct"] for r in bh], color=C_FRAUD,
             marker="o", ms=2.5, lw=1)
    ax2.set_ylabel("Tỷ lệ gian lận (%)", color=C_FRAUD)
    ax1.set_title("Giao dịch và gian lận theo pha mô phỏng")
    return _save(fig, "fig_fraud_by_hour.png", "eda_stats.json → by_hour (toàn bộ CSV sạch)")


# ============================================================================ 6 HÌNH EDA
def eda_figures(s, sample):
    figs = []
    c.sub("Hình phân tích dữ liệu (EDA)")
    # 1. Phân bố nhãn
    lab = {r["isFraud"]: r["n"] for r in s["label"]}
    n0, n1 = lab.get(0, 0), lab.get(1, 0)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.bar(["Bình thường (0)", "Gian lận (1)"], [n0, n1], color=[C_NORMAL, C_FRAUD])
    ax.set_yscale("log"); ax.set_ylabel("Số giao dịch (thang log)"); ax.set_ylim(1e3, 3e7)
    ax.set_title("Phân bố nhãn isFraud")
    for b, v in zip(bars, [n0, n1]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, vn(v), ha="center", fontsize=7)
    figs.append(_save(fig, "fig_label_distribution.png", "eda_stats.json → label (bước 6, bảng 1)",
                      f"bình thường = {vn(n0)}, gian lận = {vn(n1)}"))

    # 2. Theo loại
    rows = s["by_type"]
    types = [r["type"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.2))
    axes[0].barh(types, [r["n"] / 1e6 for r in rows], color=C_NORMAL)
    axes[0].set_xlabel("Triệu giao dịch"); axes[0].set_title("Theo loại"); axes[0].invert_yaxis()
    axes[1].barh(types, [r["fraud"] for r in rows], color=C_FRAUD)
    axes[1].set_xlabel("Số gian lận"); axes[1].set_title("Gian lận theo loại")
    axes[1].invert_yaxis(); axes[1].set_yticklabels([])
    figs.append(_save(fig, "fig_type_distribution.png", "eda_stats.json → by_type (bước 6, bảng 2)",
                      "; ".join(f"{r['type']}: {vn(r['n'])} GD / {vn(r['fraud'])} GL" for r in rows)))

    # 3. Phân bố số tiền
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bins = np.linspace(0, 7.2, 60)
    for lbl, col, name in [(0, C_NORMAL, "Bình thường"), (1, C_FRAUD, "Gian lận")]:
        ax.hist(np.log10(sample.loc[sample.isFraud == lbl, "amount"] + 1), bins=bins, density=True,
                alpha=.6, color=col, label=name)
    ax.set_xlabel("log10(amount + 1)"); ax.set_ylabel("Mật độ")
    ax.set_title("Phân bố số tiền (TRANSFER, CASH_OUT)"); ax.legend(frameon=False)
    figs.append(_save(fig, "fig_amount_distribution.png", "mẫu ở bước 6 mục 8 (toàn bộ gian lận + 1% bình thường)",
                      f"{vn(len(sample))} giao dịch mẫu; đỉnh đỏ ở 7 = gian lận 10.000.000"))

    # 4. Theo giờ
    bh = s["by_hour"]
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    ax1.bar([r["hour"] for r in bh], [r["n"] / 1e3 for r in bh], color=C_NORMAL, alpha=.6)
    ax1.set_xlabel("Pha 24 giờ (step mod 24)"); ax1.set_ylabel("Nghìn giao dịch")
    ax2 = ax1.twinx()
    ax2.plot([r["hour"] for r in bh], [r["fraud_rate_pct"] for r in bh], color=C_FRAUD, marker="o",
             ms=2.5, lw=1)
    ax2.set_ylabel("Tỷ lệ gian lận (%)", color=C_FRAUD)
    ax1.set_title("Giao dịch và gian lận theo pha mô phỏng")
    top = sorted(bh, key=lambda r: -r["fraud_rate_pct"])[:3]
    figs.append(_save(fig, "fig_fraud_by_hour.png", "eda_stats.json → by_hour (bước 6, bảng 4)",
                      "đỉnh đường đỏ: " + ", ".join(f"{r['hour']}h = {r['fraud_rate_pct']}%" for r in top)))

    # 5. Mẫu số dư
    bp = {r["isFraud"]: r for r in s["balance_patterns"]}
    keys = [("pct_orig_zero", "Nguồn 0→0"), ("pct_dest_zero", "Đích 0→0"),
            ("pct_orig_emptied", "Rút sạch\nnguồn"), ("pct_err_dest", "Lệch số dư\nđích")]
    x = np.arange(len(keys)); w = .38
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(x - w / 2, [bp[0][k] for k, _ in keys], w, color=C_NORMAL, label="Bình thường")
    ax.bar(x + w / 2, [bp[1][k] for k, _ in keys], w, color=C_FRAUD, label="Gian lận")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in keys]); ax.set_ylabel("% giao dịch")
    ax.set_title("Đặc điểm số dư theo nhãn"); ax.legend(frameon=False)
    figs.append(_save(fig, "fig_balance_patterns.png", "eda_stats.json → balance_patterns (bước 6, bảng 5)",
                      "; ".join(f"{n.replace(chr(10), ' ')}: {bp[0][k]}% / {bp[1][k]}%" for k, n in keys)))

    # 6. Tương quan
    cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
            "errorBalanceOrig", "errorBalanceDest", "origEmptied", "destZeroBalance", "isFraud"]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    sns.heatmap(sample[cols].corr(), cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, annot=True, fmt=".1f",
                annot_kws={"size": 5}, cbar_kws={"shrink": .7})
    ax.tick_params(labelsize=5.5); ax.set_title("Ma trận tương quan (mẫu)")
    figs.append(_save(fig, "fig_correlation.png", "mẫu ở bước 6 mục 8",
                      "hàng cuối = bảng corr(isFraud) ở bước 6"))
    return figs


# ============================================================================ 5 HÌNH KẾT QUẢ
def result_figures(results, curves, importances, tree_name):
    figs = []
    c.sub("Hình kết quả mô hình")
    # 1. So sánh chỉ số
    keys, labels = ["precision", "recall", "f1", "auc_pr"], ["Precision", "Recall", "F1", "AUC-PR"]
    x = np.arange(len(keys)); w = 0.2
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for i, (name, r) in enumerate(results.items()):
        ax.bar(x + (i - 1.5) * w, [r[k] for k in keys], w, color=COLORS[name], label=name)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.32)
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.legend(frameon=False, fontsize=5.5, ncol=4, loc="upper center", handlelength=1, columnspacing=.8)
    ax.set_title("So sánh các chỉ số trên tập kiểm thử")
    figs.append(_save(fig, "fig_metrics_comparison.png", "model_results.json → results (bảng so sánh ở bước 10)",
                      "chiều cao cột = Precision, Recall, F1, AUC-PR của từng mô hình"))

    # 2. Ma trận nhầm lẫn
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.2))
    for ax, (name, r) in zip(axes.ravel(), results.items()):
        mat = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        ax.imshow(np.log10(mat + 1), cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, vn(mat[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if np.log10(mat[i, j] + 1) > 3.5 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Dự đoán 0", "Dự đoán 1"], fontsize=6)
        ax.set_yticklabels(["Thực tế 0", "Thực tế 1"], fontsize=6)
        ax.set_title(name, fontsize=7)
    figs.append(_save(fig, "fig_confusion_matrices.png", "model_results.json → TP/FP/FN/TN (bảng ở bước 10)",
                      "; ".join(f"{n}: FP={r['FP']} FN={r['FN']}" for n, r in results.items())))

    # 3-4. ROC và PR
    vals = []
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        for name, (y, s) in curves.items():
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y, s)
                ax.plot(fpr, tpr, lw=1, color=COLORS[name], label=name)
                vals.append(f"{name}: {auc(fpr, tpr):.4f}")
            else:
                p, r = spark_pr_curve(y, s)
                ax.plot(r, p, lw=1, color=COLORS[name], label=name)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=.6); ax.set_xscale("log"); ax.set_xlim(1e-5, 1)
            ax.set_xlabel("False Positive Rate (thang log)"); ax.set_ylabel("True Positive Rate")
            ax.set_title("Đường cong ROC")
            note = "diện tích ROC: " + ", ".join(vals)
        else:
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
            ax.set_title("Đường cong Precision–Recall")
            note = "vẽ từ (xác suất dự đoán, nhãn thật) của toàn bộ tập kiểm thử"
        high_quality = min(r["auc_pr" if kind == "pr" else "auc_roc"] for r in results.values()) > .9
        location = ("lower right" if high_quality else "upper left") if kind == "roc" else ("lower left" if high_quality else "upper right")
        ax.legend(frameon=False, fontsize=6, loc=location)
        figs.append(_save(fig, f"fig_{kind}_curves.png", "xác suất dự đoán của 4 mô hình trên tập kiểm thử (bước 9)", note))

    # 5. Độ quan trọng đặc trưng
    names = np.array(list(importances))
    imp = np.array(list(importances.values()))
    o = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.barh(names[o], imp[o], color=COLORS.get(tree_name, "#C44E52"))
    ax.set_xlabel("Mức độ quan trọng"); ax.set_title(f"Độ quan trọng đặc trưng – {tree_name}")
    ax.tick_params(axis="y", labelsize=6)
    top = sorted(importances.items(), key=lambda x: -x[1])[:3]
    figs.append(_save(fig, "fig_feature_importance.png", "model_results.json → feature_importance (bước 10)",
                      "top 3: " + ", ".join(f"{k} = {v:.3f}" for k, v in top)))
    return figs

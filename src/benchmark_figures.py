"""Tự động sinh các biểu đồ so sánh trực quan giữa Python thuần (Scikit-Learn) và Apache Spark.
Bao gồm:
1. So sánh thời gian chạy (Huấn luyện & Dự đoán tính bằng giây `s`).
2. So sánh độ chính xác (Accuracy, F1-Score, AUC-PR, Precision, Recall).
3. Đối chứng giải thuật Baseline (Logistic Regression) vs các giải thuật đề tài chọn.
4. Đường cong ROC và Precision-Recall lồng ghép đối chứng.
5. Sơ đồ tỷ lệ chênh lệch thời gian (Speedup / Runtime Ratio).
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import config
from src import console as c

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "font.family": "DejaVu Sans"
})

COLOR_PY = "#1F77B4"    # Xanh dương - Python thuần
COLOR_SP = "#FF7F0E"    # Cam - Apache Spark
COLOR_BASE = "#D62728"  # Đỏ - Baseline Logistic Regression
COLOR_SEL = "#2CA02C"   # Xanh lá - Giải thuật đề tài chọn


def save_fig(fig, filepath, title_log=""):
    fig.tight_layout()
    fig.savefig(filepath, dpi=250, bbox_inches="tight")
    plt.close(fig)
    c.info(f"Đã lưu biểu đồ: {os.path.basename(filepath)} {title_log}")
    return filepath


def plot_execution_time(py_results: dict, sp_results: dict, out_dir: Path):
    """1. Biểu đồ so sánh thời gian chạy (giây): Huấn luyện và Dự đoán giữa Python vs Spark."""
    models = list(py_results.keys())
    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # --- Đồ thị 1: Thời gian huấn luyện (Train Time) ---
    py_train = [py_results[m]["train_time_s"] for m in models]
    sp_train = [sp_results[m]["train_time_s"] for m in models]

    b1 = ax1.bar(x - width/2, py_train, width, label="Python thuần (s)", color=COLOR_PY, alpha=0.9)
    b2 = ax1.bar(x + width/2, sp_train, width, label="Apache Spark (s)", color=COLOR_SP, alpha=0.9)

    ax1.set_title("Thời gian Huấn luyện (Training Time - s)\n[Càng thấp càng tốt]", fontweight="bold")
    ax1.set_ylabel("Thời gian (giây)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#ccc")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Hiển thị số giây cụ thể trên đầu cột
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.05 * max(py_train + sp_train),
                 f"{h:.1f}s", ha="center", va="bottom", fontsize=7)
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.05 * max(py_train + sp_train),
                 f"{h:.1f}s", ha="center", va="bottom", fontsize=7)
    ax1.set_ylim(0, max(py_train + sp_train) * 1.25)

    # --- Đồ thị 2: Thời gian dự đoán trên Test (Predict Time) ---
    py_pred = [py_results[m]["predict_time_s"] for m in models]
    sp_pred = [sp_results[m]["predict_time_s"] for m in models]

    b3 = ax2.bar(x - width/2, py_pred, width, label="Python thuần (s)", color=COLOR_PY, alpha=0.9)
    b4 = ax2.bar(x + width/2, sp_pred, width, label="Apache Spark (s)", color=COLOR_SP, alpha=0.9)

    ax2.set_title("Thời gian Dự đoán Test (Predict Time - s)\n[Càng thấp càng tốt]", fontweight="bold")
    ax2.set_ylabel("Thời gian (giây)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#ccc")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in b3:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.05 * max(py_pred + sp_pred),
                 f"{h:.2f}s", ha="center", va="bottom", fontsize=7)
    for bar in b4:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.05 * max(py_pred + sp_pred),
                 f"{h:.2f}s", ha="center", va="bottom", fontsize=7)
    ax2.set_ylim(0, max(py_pred + sp_pred) * 1.25)

    return save_fig(fig, out_dir / "fig_time_execution_comparison.png", "(Thời gian chạy)")


def plot_accuracy_comparison(py_results: dict, sp_results: dict, out_dir: Path):
    """2. Biểu đồ so sánh các chỉ số độ chính xác (F1, AUC-PR, Precision, Recall, Accuracy)."""
    metrics = ["f1", "auc_pr", "precision", "recall", "accuracy"]
    metric_labels = ["F1-Score", "AUC-PR", "Precision", "Recall", "Accuracy"]
    models = list(py_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)

    x = np.arange(len(metric_labels))
    width = 0.2

    # Panel 1: Python thuần
    ax_py = axes[0]
    colors_models = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, (m, col) in enumerate(zip(models, colors_models)):
        vals = [py_results[m][k] for k in metrics]
        ax_py.bar(x + (i - 1.5) * width, vals, width, label=m, color=col, alpha=0.85)

    ax_py.set_title("Python thuần (Scikit-Learn): Các chỉ số đánh giá", fontweight="bold")
    ax_py.set_xticks(x)
    ax_py.set_xticklabels(metric_labels, rotation=15)
    ax_py.set_ylabel("Điểm số (0.0 - 1.0)")
    ax_py.set_ylim(0, 1.15)
    ax_py.grid(axis="y", linestyle="--", alpha=0.5)
    ax_py.legend(frameon=True, fontsize=7, loc="upper right")

    # Panel 2: Apache Spark
    ax_sp = axes[1]
    for i, (m, col) in enumerate(zip(models, colors_models)):
        vals = [sp_results[m][k] for k in metrics]
        bars = ax_sp.bar(x + (i - 1.5) * width, vals, width, label=m, color=col, alpha=0.85)

    ax_sp.set_title("Apache Spark (PySpark MLlib): Các chỉ số đánh giá", fontweight="bold")
    ax_sp.set_xticks(x)
    ax_sp.set_xticklabels(metric_labels, rotation=15)
    ax_sp.set_ylim(0, 1.15)
    ax_sp.grid(axis="y", linestyle="--", alpha=0.5)
    ax_sp.legend(frameon=True, fontsize=7, loc="upper right")

    return save_fig(fig, out_dir / "fig_accuracy_metrics_comparison.png", "(Chỉ số độ chính xác)")


def plot_baseline_vs_selected(py_results: dict, sp_results: dict, out_dir: Path):
    """3. Sơ đồ đối chứng trực diện giữa Baseline (Logistic Regression) và các giải thuật đề tài chọn."""
    models = list(py_results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # Đối chứng F1-score
    py_f1 = [py_results[m]["f1"] for m in models]
    sp_f1 = [sp_results[m]["f1"] for m in models]
    x = np.arange(len(models))
    w = 0.35

    bar1 = ax1.bar(x - w/2, py_f1, w, label="Python thuần", color=COLOR_PY)
    bar2 = ax1.bar(x + w/2, sp_f1, w, label="Apache Spark", color=COLOR_SP)

    # Tô viền đỏ nổi bật cho Baseline Logistic Regression
    bar1[0].set_edgecolor("red"); bar1[0].set_linewidth(2.0)
    bar2[0].set_edgecolor("red"); bar2[0].set_linewidth(2.0)

    ax1.set_title("F1-Score: Baseline vs Giải thuật đề tài", fontweight="bold")
    ax1.set_ylabel("F1-Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 1.2)
    ax1.legend(frameon=True, fontsize=7.5)

    for b in list(bar1) + list(bar2):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    # Đối chứng AUC-PR
    py_auc_pr = [py_results[m]["auc_pr"] for m in models]
    sp_auc_pr = [sp_results[m]["auc_pr"] for m in models]

    bar3 = ax2.bar(x - w/2, py_auc_pr, w, label="Python thuần", color=COLOR_PY)
    bar4 = ax2.bar(x + w/2, sp_auc_pr, w, label="Apache Spark", color=COLOR_SP)

    bar3[0].set_edgecolor("red"); bar3[0].set_linewidth(2.0)
    bar4[0].set_edgecolor("red"); bar4[0].set_linewidth(2.0)

    ax2.set_title("AUC-PR: Baseline vs Giải thuật đề tài", fontweight="bold")
    ax2.set_ylabel("AUC-PR")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_ylim(0, 1.2)
    ax2.legend(frameon=True, fontsize=7.5)

    for b in list(bar3) + list(bar4):
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    return save_fig(fig, out_dir / "fig_baseline_vs_selected.png", "(Baseline vs Đề tài)")


def plot_curves_comparison(py_curves: dict, sp_curves: dict, out_dir: Path):
    """4. Biểu đồ đường cong ROC và Precision-Recall đối chiếu giữa Python và Spark."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    models = list(py_curves.keys())
    colors = {"Logistic Regression": "#D62728", "Decision Tree": "#2CA02C",
              "Random Forest": "#1F77B4", "GBT": "#9467BD"}

    # ROC Curves
    for m in models:
        # Python curve
        y_py, s_py = np.array(py_curves[m]["labels"]), np.array(py_curves[m]["probs"])
        fpr_py, tpr_py, _ = roc_curve(y_py, s_py)
        auc_py = auc(fpr_py, tpr_py)
        ax_roc.plot(fpr_py, tpr_py, linestyle="--", color=colors[m], label=f"{m} (Py: {auc_py:.3f})", lw=1.2)

        # Spark curve
        y_sp, s_sp = np.array(sp_curves[m]["labels"]), np.array(sp_curves[m]["probs"])
        fpr_sp, tpr_sp, _ = roc_curve(y_sp, s_sp)
        auc_sp = auc(fpr_sp, tpr_sp)
        ax_roc.plot(fpr_sp, tpr_sp, linestyle="-", color=colors[m], label=f"{m} (Spark: {auc_sp:.3f})", lw=1.4)

    ax_roc.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax_roc.set_title("Đường cong ROC (Nét đứt: Py, Nét liền: Spark)", fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR)")
    ax_roc.grid(True, linestyle="--", alpha=0.5)
    ax_roc.legend(fontsize=6.5, loc="lower right")

    # PR Curves
    for m in models:
        y_py, s_py = np.array(py_curves[m]["labels"]), np.array(py_curves[m]["probs"])
        p_py, r_py, _ = precision_recall_curve(y_py, s_py)
        ax_pr.plot(r_py, p_py, linestyle="--", color=colors[m], label=f"{m} (Py)", lw=1.2)

        y_sp, s_sp = np.array(sp_curves[m]["labels"]), np.array(sp_curves[m]["probs"])
        p_sp, r_sp, _ = precision_recall_curve(y_sp, s_sp)
        ax_pr.plot(r_sp, p_sp, linestyle="-", color=colors[m], label=f"{m} (Spark)", lw=1.4)

    ax_pr.set_title("Đường cong Precision-Recall", fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(True, linestyle="--", alpha=0.5)
    ax_pr.legend(fontsize=6.5, loc="lower left")

    return save_fig(fig, out_dir / "fig_roc_and_pr_curves_comparison.png", "(ROC & PR Curves)")


def generate_all_benchmark_figures(py_res: dict, sp_res: dict, py_cur: dict, sp_cur: dict):
    """Sinh tất cả các sơ đồ và lưu vào outputs/figures/benchmark/."""
    out_dir = Path(config.OUTPUT_DIR) / "figures" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    c.sub("SINH SƠ ĐỒ / BIỂU ĐỒ SO SÁNH (PYTHON THUẦN VS APACHE SPARK)")
    figs = [
        plot_execution_time(py_res["results"], sp_res["results"], out_dir),
        plot_accuracy_comparison(py_res["results"], sp_res["results"], out_dir),
        plot_baseline_vs_selected(py_res["results"], sp_res["results"], out_dir),
        plot_curves_comparison(py_cur, sp_cur, out_dir)
    ]
    c.ok(f"Đã tạo thành công {len(figs)} biểu đồ so sánh chất lượng cao tại: {out_dir}")
    return figs

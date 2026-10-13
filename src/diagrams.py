"""Vẽ 3 sơ đồ minh họa: quy trình hệ thống, kiến trúc Spark, kiến trúc ứng dụng."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans"})
K = 1.0


def box(ax, x, y, w, h, title, sub, fc, ec):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.01,rounding_size=0.025",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x, y + ((0.012 * K + (0.006 * K * sub.count("\n"))) if sub else 0), title, ha="center", va="center", fontsize=7.2, weight="bold", color="#1b1b1b")
    if sub:
        ax.text(x, y - 0.021 * K, sub, ha="center", va="center", fontsize=5.6, color="#333333")

def arrow(ax, x1, y1, x2, y2, txt=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#444"))
    if txt:
        ax.text((x1+x2)/2 + 0.02, (y1+y2)/2, txt, fontsize=5.3, va="center", color="#555", style="italic")

def draw_all(out_dir):
    """Vẽ và lưu 3 sơ đồ vào out_dir; trả về danh sách đường dẫn."""
    global K
    K = 1.0
    os.makedirs(out_dir, exist_ok=True)
    # ---------- Hình: quy trình xây dựng hệ thống
    fig, ax = plt.subplots(figsize=(3.4, 5.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [
        ("1. Thu thập dữ liệu", "PaySim (Kaggle) – CSV 6,36 triệu dòng, 11 cột", "#E8F0FA", "#4C72B0"),
        ("2. Nạp dữ liệu vào Spark", "SparkSession · lược đồ tường minh · DataFrame", "#E8F0FA", "#4C72B0"),
        ("3. Kiểm tra & làm sạch", "null · trùng lặp · số âm · giữ amount = 0", "#E8F0FA", "#4C72B0"),
        ("4. Phân tích khám phá (EDA)", "Spark SQL · thống kê theo loại, giờ, số dư · trực quan hóa", "#EAF5EC", "#55A868"),
        ("5. Lọc & tạo đặc trưng A/B", "TRANSFER + CASH_OUT · A sau / B trước giao dịch", "#EAF5EC", "#55A868"),
        ("6. Chia phân tầng / theo thời gian", "Train: fit · Validation: chọn mô hình/ngưỡng · Test", "#EAF5EC", "#55A868"),
        ("7. Pipeline MLlib", "StringIndexer → OneHot → VectorAssembler (→ Scaler)", "#FBEDED", "#C44E52"),
        ("8. Huấn luyện 4 mô hình", "Logistic Regression · Decision Tree · Random Forest · GBT", "#FBEDED", "#C44E52"),
        ("9. Đánh giá & so sánh", "Confusion matrix · Precision · Recall · F1 · AUC-PR", "#F1EEF8", "#8172B2"),
        ("10. Lưu mô hình & triển khai", "PipelineModel → ứng dụng Streamlit dự đoán", "#F1EEF8", "#8172B2"),
    ]
    n = len(steps); top, bot = 0.955, 0.045; h = 0.068
    ys = [top - i * (top - bot) / (n - 1) for i in range(n)]
    for (t, s, fc, ec), y in zip(steps, ys):
        box(ax, 0.5, y, 0.92, h, t, s, fc, ec)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, 0.5, y1 - h/2, 0.5, y2 + h/2)
    fig.savefig(os.path.join(out_dir, "fig_pipeline_flow.png"), dpi=250, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # ---------- Hình: kiến trúc thực thi Spark
    K = 2.2
    fig, ax = plt.subplots(figsize=(3.4, 2.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.5, 0.86, 0.62, 0.17, "Driver Program", "SparkSession · DAG Scheduler · Catalyst", "#E8F0FA", "#4C72B0")
    box(ax, 0.5, 0.56, 0.62, 0.13, "Cluster Manager", "local[*] / Standalone / YARN", "#EAF5EC", "#55A868")
    for i, x in enumerate([0.17, 0.5, 0.83]):
        box(ax, x, 0.2, 0.28, 0.22, f"Executor {i+1}", "Task · Cache", "#FBEDED", "#C44E52")
        arrow(ax, 0.5, 0.495, x, 0.315)
    arrow(ax, 0.5, 0.775, 0.5, 0.625)
    ax.text(0.03, 0.02, "Dữ liệu được chia thành các phân vùng (partition) và xử lý song song",
            fontsize=5.5, color="#555")
    fig.savefig(os.path.join(out_dir, "fig_spark_architecture.png"), dpi=250, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # ---------- Hình: kiến trúc ứng dụng demo
    K = 2.0
    fig, ax = plt.subplots(figsize=(3.4, 2.5)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.5, 0.92, 0.9, 0.12, "Người dùng (trình duyệt)", "Nhập 1 giao dịch  ·  Tải tệp CSV  ·  Chỉnh ngưỡng", "#F1EEF8", "#8172B2")
    box(ax, 0.5, 0.70, 0.9, 0.12, "Giao diện Streamlit (app.py)", "3 tab: kiểm tra đơn · theo lô · thống kê", "#E8F0FA", "#4C72B0")
    box(ax, 0.27, 0.45, 0.44, 0.17, "SparkSession", "add_features()", "#EAF5EC", "#55A868")
    box(ax, 0.75, 0.45, 0.42, 0.17, "PipelineModel", "best_model · ngưỡng t*", "#FBEDED", "#C44E52")
    box(ax, 0.5, 0.15, 0.9, 0.13, "Kết quả", "xác suất gian lận · nhãn cảnh báo · tải CSV", "#F1EEF8", "#8172B2")
    arrow(ax, 0.5, 0.86, 0.5, 0.76); arrow(ax, 0.35, 0.64, 0.3, 0.525); arrow(ax, 0.49, 0.45, 0.54, 0.45)
    arrow(ax, 0.75, 0.375, 0.6, 0.215); arrow(ax, 0.1, 0.215, 0.1, 0.64)
    ax.text(0.115, 0.43, "", fontsize=5)
    fig.savefig(os.path.join(out_dir, "fig_app_architecture.png"), dpi=250, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    return [os.path.join(out_dir, f) for f in
            ("fig_pipeline_flow.png", "fig_spark_architecture.png", "fig_app_architecture.png")]


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in draw_all(os.path.join(base, "outputs", "figures")):
        print("Đã vẽ", p)

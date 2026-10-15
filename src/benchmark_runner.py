"""Bộ điều phối toàn diện cho bài toán so sánh Baseline vs Đề tài (Python vs Spark).
Quy trình thực hiện:
1. Chạy tuần tự Python thuần trước (4 mô hình).
2. Chạy Spark MLlib sau (4 mô hình).
3. Tự động vẽ các biểu đồ so sánh (thời gian chạy giây s, độ chính xác F1/AUC/Accuracy).
4. Tự động xuất bản báo cáo Markdown chuyên sâu.
"""
import json
import time
from pathlib import Path

import config
from src import console as c
from src.pure_python_bench import run_pure_python_benchmark
from src.spark_bench import run_spark_benchmark
from src.benchmark_figures import generate_all_benchmark_figures
from src.benchmark_report import generate_markdown_report


def run_full_benchmark(sample_fraction: float = 0.1, skip_python: bool = False, skip_spark: bool = False):
    """Điều phối toàn bộ quy trình so sánh đối chứng tuần tự."""
    c.sub("QUY TRÌNH SO SÁNH GIẢI THUẬT BASELINE VS ĐỀ TÀI: PYTHON THUẦN & APACHE SPARK")
    c.info(f"Tỷ lệ phân chia dữ liệu: Train 70%, Validation 10%, Test 20%")
    c.info(f"Tỷ lệ mẫu thực nghiệm: {sample_fraction * 100:.1f}%")

    out_dir = Path(config.OUTPUT_DIR) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    py_res_file = out_dir / "python_results.json"
    sp_res_file = out_dir / "spark_results.json"
    py_cur_file = out_dir / "python_curves.json"
    sp_cur_file = out_dir / "spark_curves.json"

    # Bước 1: Chạy Python thuần trước
    if not skip_python:
        t0 = time.perf_counter()
        py_res = run_pure_python_benchmark(sample_fraction=sample_fraction)
        c.ok(f"Phiên 1 (Python thuần) hoàn tất trong {time.perf_counter() - t0:.2f}s\n")
    else:
        c.warn("Bỏ qua huấn luyện Python thuần, nạp kết quả đã lưu...")
        if not py_res_file.exists():
            raise FileNotFoundError("Chưa có kết quả Python thuần để nạp.")
        with open(py_res_file, "r", encoding="utf-8") as f:
            py_res = json.load(f)

    # Bước 2: Chạy Spark sau
    if not skip_spark:
        t0 = time.perf_counter()
        sp_res = run_spark_benchmark(sample_fraction=sample_fraction)
        c.ok(f"Phiên 2 (Apache Spark) hoàn tất trong {time.perf_counter() - t0:.2f}s\n")
    else:
        c.warn("Bỏ qua huấn luyện Spark, nạp kết quả đã lưu...")
        if not sp_res_file.exists():
            raise FileNotFoundError("Chưa có kết quả Spark để nạp.")
        with open(sp_res_file, "r", encoding="utf-8") as f:
            sp_res = json.load(f)

    # Nạp dữ liệu đường cong ROC/PR
    with open(py_cur_file, "r", encoding="utf-8") as f:
        py_cur = json.load(f)
    with open(sp_cur_file, "r", encoding="utf-8") as f:
        sp_cur = json.load(f)

    # Bước 3: Tự động vẽ các biểu đồ so sánh
    fig_paths = generate_all_benchmark_figures(py_res, sp_res, py_cur, sp_cur)

    # Bước 4: Tự động xuất bản báo cáo Markdown
    c.sub("SINH BẢN BÁO CÁO CHUYÊN SÂU (MARKDOWN)")
    report_path = generate_markdown_report(py_res, sp_res, fig_paths)
    c.ok(f"Đã xuất bản báo cáo Markdown tại: {report_path}")

    # In tóm tắt kết quả lên console
    c.sub("BẢNG TỔNG KẾT SO SÁNH THỜI GIAN & F1-SCORE TRÊN TẬP TEST")
    models = list(py_res["results"].keys())
    summary_rows = []
    for m in models:
        py_m = py_res["results"][m]
        sp_m = sp_res["results"][m]
        summary_rows.append([
            m,
            f"{py_m['train_time_s']}s",
            f"{sp_m['train_time_s']}s",
            f"{py_m['predict_time_s']}s",
            f"{sp_m['predict_time_s']}s",
            f"{py_m['f1']:.4f}",
            f"{sp_m['f1']:.4f}",
            f"{py_m['auc_pr']:.4f}",
            f"{sp_m['auc_pr']:.4f}"
        ])
    c.table(
        ["Mô hình", "Py Fit(s)", "Spark Fit(s)", "Py Pred(s)", "Spark Pred(s)", "Py F1", "Spark F1", "Py AUC-PR", "Spark AUC-PR"],
        summary_rows
    )

    return {
        "python_results": py_res,
        "spark_results": sp_res,
        "figures": fig_paths,
        "report": report_path
    }

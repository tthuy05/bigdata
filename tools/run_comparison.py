"""CLI Tool: So sánh giải thuật Baseline (Logistic Regression) với giải thuật đề tài chọn (Python thuần vs Apache Spark).
Sử dụng:
    python tools/run_comparison.py --sample-fraction 0.1
    python tools/run_comparison.py --sample-fraction 1.0
"""
import argparse
import sys
import os
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import console as c
from src.benchmark_runner import run_full_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="So sánh giải thuật Baseline vs Đề tài chọn (Python thuần vs Apache Spark)"
    )
    parser.add_argument(
        "--sample-fraction", type=float, default=0.1,
        help="Tỷ lệ mẫu dữ liệu PaySim lấy từ CSV (0.01 đến 1.0; mặc định: 0.1 = 10%)"
    )
    parser.add_argument(
        "--skip-python", action="store_true",
        help="Bỏ qua huấn luyện Python thuần (sử dụng kết quả đã lưu trong outputs/benchmarks/)"
    )
    parser.add_argument(
        "--skip-spark", action="store_true",
        help="Bỏ qua huấn luyện Spark (sử dụng kết quả đã lưu trong outputs/benchmarks/)"
    )

    args = parser.parse_args()

    if not 0.0 < args.sample_fraction <= 1.0:
        c.err("sample-fraction phải nằm trong khoảng (0, 1.0].")
        sys.exit(1)

    try:
        run_full_benchmark(
            sample_fraction=args.sample_fraction,
            skip_python=args.skip_python,
            skip_spark=args.skip_spark
        )
    except KeyboardInterrupt:
        c.warn("\nĐã hủy tiến trình thực nghiệm bởi người dùng.")
        sys.exit(1)
    except Exception as e:
        c.err(f"Lỗi trong quá trình chạy: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

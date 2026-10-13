"""
ỨNG DỤNG HỌC MÁY TRONG PHÁT HIỆN GIAN LẬN GIAO DỊCH TÀI CHÍNH TRÊN NỀN TẢNG APACHE SPARK
=========================================================================================
Thứ tự:  IN SỐ LIỆU từng bước  ->  VẼ HÌNH  ->  ĐỐI CHIẾU với báo cáo  ->  DỰNG WORD  ->  MỞ WEB APP

    python main.py                       # chạy đầy đủ 14 bước, dừng sau mỗi bước (nhấn Enter)
    python main.py --auto                # chạy liền một mạch
    python main.py --use-saved-models    # không huấn luyện lại – nạp 4 mô hình đã lưu (nhanh)
    python main.py --app-only            # chỉ mở web app
    python main.py --word-only           # chỉ dựng lại Word từ số liệu + hình đã có
Xem README.md để biết thêm.
"""
import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import webbrowser

import config
from src import console as c

TOTAL = 14


def save_json(folder, name, obj):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(folder, name):
    p = os.path.join(folder, name)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return json.load(open(p, encoding="utf-8"))
    return None


# ------------------------------------------------------------------ web app
def _port_open(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def launch_app(port):
    if not os.path.exists(config.BEST_MODEL_PATH):
        raise SystemExit("Chưa có models/best_model – hãy chạy  python main.py  (đầy đủ) trước.")
    url = f"http://localhost:{port}"
    proc = None
    if _port_open(port):
        c.info(f"Cổng {port} đang có ứng dụng chạy sẵn.")
    else:
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run",
                                 os.path.join(config.BASE_DIR, "app", "app.py"),
                                 "--server.port", str(port), "--server.headless", "true",
                                 "--browser.gatherUsageStats", "false"], cwd=config.BASE_DIR)
        for _ in range(120):
            if _port_open(port):
                break
            if proc.poll() is not None:
                raise SystemExit("Streamlit thoát sớm – xem thông báo lỗi phía trên.")
            time.sleep(0.5)
    print(f"\n   {c.BOLD}{c.GREEN}{'#' * 60}")
    print(f"   #   ỨNG DỤNG WEB ĐANG CHẠY TẠI:  {url:<25s}#")
    print(f"   #   Ctrl + Click vào link để mở  |  Ctrl + C để tắt      #")
    print(f"   {'#' * 60}{c.RESET}")
    c.info("Mẹo: bấm 'Phân tích giao dịch' một lần trước khi demo – lần đầu Spark cần 20–60 giây khởi động.")
    webbrowser.open(url)
    return proc


def wait_app(proc):
    if proc is None:
        return
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n   Đã tắt ứng dụng web.")


# ------------------------------------------------------------------ chương trình chính
def main():
    ap = argparse.ArgumentParser(description="Phát hiện gian lận giao dịch trên Apache Spark")
    ap.add_argument("--data", default=config.DATA_PATH, help="đường dẫn tệp CSV PaySim")
    ap.add_argument("--auto", action="store_true", help="không dừng giữa các bước")
    ap.add_argument("--use-saved-models", action="store_true", help="nạp 4 mô hình đã lưu, không huấn luyện lại")
    ap.add_argument("--open-figs", action="store_true", help="mở mỗi hình ngay khi vẽ xong (Windows)")
    ap.add_argument("--no-word", action="store_true", help="không dựng báo cáo Word")
    ap.add_argument("--no-app", action="store_true", help="không mở web app ở cuối")
    ap.add_argument("--app-only", action="store_true", help="chỉ mở web app")
    ap.add_argument("--word-only", action="store_true", help="chỉ dựng lại Word từ outputs/ hiện có")
    ap.add_argument("--save-baseline", action="store_true",
                    help="lưu kết quả lần này làm số liệu gốc của báo cáo (sau khi đã cập nhật Word)")
    ap.add_argument("--port", type=int, default=8501)
    a = ap.parse_args()

    # ---------------- chế độ rút gọn
    if a.app_only:
        c.setup(True, False, 1)
        with c.step(1, "KHỞI ĐỘNG ỨNG DỤNG WEB", pause_after=False):
            proc = launch_app(a.port)
        wait_app(proc)
        return
    if a.word_only:
        from src import word
        c.setup(True, False, 2)
        with c.step(1, "VẼ LẠI 3 SƠ ĐỒ", pause_after=False):
            from src.diagrams import draw_all
            for p in draw_all(config.FIG_DIR):
                c.figure_saved(p, "sơ đồ minh họa (src/diagrams.py)")
        with c.step(2, "DỰNG BÁO CÁO WORD TỪ outputs/metrics + outputs/figures", pause_after=False):
            if not load_json(config.METRIC_DIR, "model_results.json"):
                raise SystemExit("Chưa có outputs/metrics – hãy chạy  python main.py  trước.")
            word.build()
        c.checklist()
        return

    c.setup(a.auto, a.open_figs, TOTAL)
    print(f"{c.BOLD}{__doc__.splitlines()[1]}{c.RESET}")

    # =================================================================== 1
    with c.step(1, "KIỂM TRA MÔI TRƯỜNG"):
        from src.envcheck import check
        envst = check(a.data, need_word=not a.no_word)

    # =================================================================== 2
    with c.step(2, "KHỞI ĐỘNG APACHE SPARK"):
        from src.data_loader import get_spark, load_raw, quality_report, clean
        t0 = time.time()
        spark = get_spark()
        sc = spark.sparkContext
        env = {"spark_version": spark.version, "python": platform.python_version(),
               "java": envst["java"], "master": sc.master, "cores": os.cpu_count(),
               "parallelism": sc.defaultParallelism, "driver_memory": config.DRIVER_MEMORY,
               "os": platform.platform()}
        c.kv("Phiên bản Spark", env["spark_version"])
        c.kv("Chế độ chạy (master)", env["master"])
        c.kv("Số luồng xử lý song song", env["parallelism"])
        c.kv("Bộ nhớ Driver", env["driver_memory"])
        c.kv("Spark UI (xem job Spark đang chạy)", sc.uiWebUrl)
        c.kv("Thời gian khởi động (giây)", round(time.time() - t0, 1))

    # =================================================================== 3
    with c.step(3, "NẠP DỮ LIỆU CSV VÀO SPARK DATAFRAME"):
        c.kv("Tệp dữ liệu", os.path.relpath(a.data))
        raw = load_raw(spark, a.data)
        c.info("Lược đồ (khai báo tường minh trong src/data_loader.py):")
        raw.printSchema()
        c.info("5 giao dịch đầu tiên:")
        raw.show(5, truncate=False)
        c.info("5 giao dịch gian lận đầu tiên:")
        raw.filter("isFraud = 1").show(5, truncate=False)
        c.kv("Số phân vùng (partition)", raw.rdd.getNumPartitions())

    # =================================================================== 4
    with c.step(4, "KIỂM TRA CHẤT LƯỢNG DỮ LIỆU  (→ Bảng 6)"):
        t0 = time.time()
        q = quality_report(raw)
        q["time_s"] = round(time.time() - t0, 1)
        c.kv("Số dòng", q["rows"]); c.kv("Số cột", q["columns"])
        c.info("Giá trị thiếu (null) và giá trị âm theo từng cột:")
        c.table(["Cột", "Null", "Giá trị âm"], [[k, v, q["negatives"].get(k, "-")] for k, v in q["nulls"].items()])
        c.kv("Bản ghi trùng lặp", q["duplicates"])
        c.kv("Giao dịch có amount <= 0", q["zero_amount"])
        c.kv("Thời gian kiểm tra (giây)", q["time_s"])

    # =================================================================== 5
    with c.step(5, "LÀM SẠCH DỮ LIỆU VÀ LƯU PARQUET"):
        c.info("Quy tắc: bỏ trùng lặp, bỏ null, bỏ amount <= 0, chỉ giữ 5 loại giao dịch hợp lệ")
        t0 = time.time()
        if a.use_saved_models and os.path.exists(config.PARQUET_PATH):
            c.info("Dùng lại Parquet đã lưu để tập kiểm thử trùng với lúc huấn luyện mô hình")
        else:
            clean(raw).repartition(8).write.mode("overwrite").parquet(config.PARQUET_PATH)
        df = spark.read.parquet(config.PARQUET_PATH).cache()
        q["clean_rows"] = df.count()
        q["clean_parquet_time_s"] = round(time.time() - t0, 1)
        c.table(["", "Số dòng"], [["Trước làm sạch", q["rows"]], ["Sau làm sạch", q["clean_rows"]],
                                  ["Đã loại bỏ", q["rows"] - q["clean_rows"]]])
        c.kv("Lưu tại", os.path.relpath(config.PARQUET_PATH))
        save_json(config.METRIC_DIR, "quality.json", q)

    # =================================================================== 6
    with c.step(6, "THỐNG KÊ BẰNG SPARK SQL  (→ Bảng 7, 8, 9 và 6 hình EDA)"):
        from src.stats import compute_stats
        stats, sample = compute_stats(spark, df)
        save_json(config.METRIC_DIR, "eda_stats.json", stats)

    # =================================================================== 7
    with c.step(7, "LỌC TRANSFER/CASH_OUT VÀ TẠO ĐẶC TRƯNG  (→ Bảng 10)"):
        from src.features import add_features, filter_relevant, FEATURE_NAMES
        data = add_features(filter_relevant(df))
        c.kv("Số giao dịch sau khi lọc", stats["relevant_rows"])
        cols = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                "errorBalanceOrig", "errorBalanceDest", "origEmptied", "destZeroBalance", "hour", "isFraud"]
        c.info("Giao dịch GIAN LẬN sau khi tạo đặc trưng:")
        data.filter("isFraud = 1").select(cols).show(5)
        c.info("Giao dịch BÌNH THƯỜNG sau khi tạo đặc trưng:")
        data.filter("isFraud = 0").select(cols).show(5)
        c.table(["#", "Đặc trưng đưa vào mô hình"], [[i + 1, f] for i, f in enumerate(FEATURE_NAMES)])

    # =================================================================== 8
    with c.step(8, "CHIA DỮ LIỆU 70/10/20 VÀ TRỌNG SỐ LỚP  (→ Bảng 11)"):
        from src import modeling as M
        train, valid, test, split = M.split_data(data)

    # =================================================================== 9
    title = "NẠP 4 MÔ HÌNH ĐÃ LƯU" if a.use_saved_models else "HUẤN LUYỆN 4 MÔ HÌNH SPARK MLlib"
    with c.step(9, f"{title}  (→ Bảng 12)"):
        prev = (load_json(config.METRIC_DIR, "model_results.json") or {}).get("results", {})
        results, curves, fitted = M.train_all(train, valid, test, a.use_saved_models, prev)

    # =================================================================== 10
    with c.step(10, "SO SÁNH MÔ HÌNH VÀ CHỌN MÔ HÌNH TỐT NHẤT  (→ Bảng 13–16)"):
        extra = M.summarize(results, fitted, test)
        best = extra["best_model"]
        M.save_best(fitted, best)
        m = {"split": split, "results": results, **extra, "features": FEATURE_NAMES}
        with open(os.path.join(config.MODEL_DIR, "best_model_info.json"), "w", encoding="utf-8") as f:
            json.dump({"name": best, "threshold": results[best]["threshold"], "metrics": results[best]},
                      f, ensure_ascii=False, indent=2)
        for d in (train, valid, test, df):
            d.unpersist()
        spark.stop()
        c.info("Đã dừng Spark (các bước sau chỉ dùng số liệu đã tính).")

    # =================================================================== 11
    with c.step(11, "VẼ HÌNH TỪ SỐ LIỆU ĐÃ IN Ở TRÊN"):
        from src import figures
        from src.diagrams import draw_all
        figs = figures.eda_figures(stats, sample)
        figs += figures.result_figures(results, curves, extra["feature_importance"],
                                       extra["feature_importance_model"])
        c.sub("Sơ đồ minh họa")
        for p in draw_all(config.FIG_DIR):
            c.figure_saved(p, "src/diagrams.py (sơ đồ, không dùng số liệu)")
        m["figures"] = figs
        env["total_time_s"] = round(c.elapsed(), 1)
        save_json(config.METRIC_DIR, "model_results.json", m)
        save_json(config.METRIC_DIR, "env.json", env)
        from src.summary import build_text
        with open(os.path.join(config.OUTPUT_DIR, "ket_qua.txt"), "w", encoding="utf-8") as f:
            f.write(build_text(q, stats, m, env))
        c.ok("Đã ghi toàn bộ số liệu: outputs/metrics/*.json và outputs/ket_qua.txt")
        if os.name == "nt":
            try:
                os.startfile(config.FIG_DIR)
            except OSError:
                pass

    # =================================================================== 12
    with c.step(12, "ĐỐI CHIẾU VỚI SỐ LIỆU GỐC CỦA BÁO CÁO (outputs/baseline)"):
        from src.compare import compare
        base = tuple(load_json(config.BASELINE_DIR, n) for n in ("quality.json", "eda_stats.json", "model_results.json"))
        if any(b is None for b in base):
            for n in ("quality.json", "eda_stats.json", "model_results.json", "env.json"):
                shutil.copy2(os.path.join(config.METRIC_DIR, n), os.path.join(config.BASELINE_DIR, n))
            raise c.SkipStep("chưa có số liệu gốc – đã lưu kết quả lần này làm gốc (outputs/baseline)")
        tot = compare(base, (q, stats, m))
        n = sum(tot.values())
        print(f"\n   {c.BOLD}KẾT QUẢ: {c.GREEN}{tot['KHỚP']} khớp{c.RESET}{c.BOLD} · {c.YELLOW}{tot['LỆCH NHỎ']} lệch nhỏ"
              f"{c.RESET}{c.BOLD} · {c.RED}{tot['LỆCH']} lệch{c.RESET}{c.BOLD} / {n} giá trị{c.RESET}")
        if tot["LỆCH"]:
            c.warn("Có số liệu khác báo cáo. Nếu đây là kết quả mới bạn muốn dùng: cập nhật Word, "
                   "rồi chạy  python main.py --use-saved-models --save-baseline")
        else:
            c.ok("SỐ LIỆU TRONG BÁO CÁO KHỚP VỚI KẾT QUẢ CHẠY LẠI")
        if a.save_baseline:
            for n_ in ("quality.json", "eda_stats.json", "model_results.json", "env.json"):
                shutil.copy2(os.path.join(config.METRIC_DIR, n_), os.path.join(config.BASELINE_DIR, n_))
            c.ok("Đã lưu kết quả lần này làm số liệu gốc mới (--save-baseline)")

    # =================================================================== 13
    with c.step(13, "DỰNG BÁO CÁO WORD"):
        if a.no_word:
            raise c.SkipStep("dùng --no-word")
        if not envst["word_ready"]:
            raise c.SkipStep("chưa có Node.js/thư viện docx (README mục 2.4)")
        from src import word
        word.build()

    # =================================================================== 14
    proc = None
    with c.step(14, "KHỞI ĐỘNG ỨNG DỤNG WEB", pause_after=False):
        if a.no_app:
            raise c.SkipStep("dùng --no-app")
        proc = launch_app(a.port)
    c.checklist()
    wait_app(proc)


if __name__ == "__main__":
    main()

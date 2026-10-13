"""Apache Spark PaySim: experiments, saved-model demo, Streamlit and Word."""
import argparse
import importlib.util
import json
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

import config
from src import console as c


def load_json(folder, name):
    path = Path(folder) / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except (OSError, ValueError):
        return False


def launch_app(port, open_browser=True):
    info = load_json(config.MODEL_DIR, "best_model_info.json")
    if info is None:
        raise FileNotFoundError("Thiếu models/best_model_info.json. Hãy chạy thực nghiệm trước.")
    model_path = Path(info.get("model_path", config.BEST_MODEL_PATH))
    if not model_path.is_absolute():
        model_path = Path(config.BASE_DIR) / model_path
    if not (model_path / "metadata").is_dir():
        raise FileNotFoundError(f"Thiếu mô hình đã lưu: {model_path}")
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError("Thiếu Streamlit. Chạy python -m pip install -r requirements.txt.")
    proc = None
    if _port_open(port):
        if not _healthy(port):
            raise RuntimeError(f"Cổng {port} đang bị ứng dụng khác chiếm; chọn --port khác.")
        c.info(f"Streamlit đã chạy trên cổng {port}.")
    else:
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run",
                                 str(Path(config.BASE_DIR) / "app" / "app.py"),
                                 "--server.port", str(port), "--server.address", "127.0.0.1",
                                 "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                                cwd=config.BASE_DIR)
        deadline = time.monotonic() + 60
        while not _healthy(port):
            if proc.poll() is not None:
                raise RuntimeError("Streamlit thoát sớm; xem lỗi phía trên.")
            if time.monotonic() >= deadline:
                proc.terminate()
                raise TimeoutError("Streamlit chưa sẵn sàng sau 60 giây.")
            time.sleep(0.5)
    url = f"http://localhost:{port}"
    c.ok(f"Ứng dụng web: {url} (Ctrl+C để tắt)")
    if open_browser:
        webbrowser.open(url)
    return proc


def wait_app(proc):
    if proc is not None:
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait(timeout=15)
            return
        if proc.returncode:
            raise RuntimeError(f"Streamlit đã thoát với mã lỗi {proc.returncode}.")


def saved_demo():
    """Read-only: no fit, retuning, split regeneration or artifact replacement."""
    from src.model_io import load_pipeline
    from src.data_loader import get_spark
    from src.features import add_features
    from src.modeling import SLUG
    from src.prediction import load_bundle
    m = load_json(config.METRIC_DIR, "model_results.json")
    if not m:
        raise FileNotFoundError("Thiếu outputs/metrics/model_results.json.")
    info = load_json(config.MODEL_DIR, "best_model_info.json")
    if not info:
        raise FileNotFoundError("Thiếu metadata mô hình tốt nhất.")
    if "model_path" in info:
        selected_path = Path(info["model_path"])
        model_dir = (selected_path if selected_path.is_absolute() else Path(config.BASE_DIR) / selected_path).parent
    else:
        model_dir = Path(config.MODEL_DIR)
    paths = {name: model_dir / slug for name, slug in SLUG.items()}
    missing = [str(p) for p in paths.values() if not (p / "metadata").is_dir()]
    if missing:
        raise FileNotFoundError("Thiếu mô hình; không tự huấn luyện: " + ", ".join(missing))
    if m.get("schema_version") == 2:
        from src.artifacts import validate_result
        validate_result(m)
    else:
        c.warn("Kết quả lịch sử: đặc trưng sau giao dịch và lựa chọn dựa trên Test; chất lượng chưa tái kiểm chứng.")
    spark = get_spark("FraudDetection-SavedDemo")
    try:
        _, bundle = load_bundle(spark)
        # Logic fixture verifies loading/transform only; never an experiment metric.
        fixture = spark.sql("""SELECT 1 AS step, 'TRANSFER' AS type, 1000D AS amount,
                               1000D AS oldbalanceOrg, 0D AS newbalanceOrig,
                               0D AS oldbalanceDest, 0D AS newbalanceDest""")
        for name, path in paths.items():
            model = load_pipeline(str(path))
            prediction = model.transform(add_features(fixture, bundle["feature_profile"])) \
                              .select("probability").first()[0]
            if len(prediction) != 2:
                raise RuntimeError(f"Mô hình {name} không trả về xác suất nhị phân.")
            c.ok(f"Nạp và transform thành công: {name}")
        c.info(f"Ngưỡng lưu sẵn: {bundle['threshold']:.12g}; không tối ưu lại ngưỡng.")
    finally:
        spark.stop()
    from src.summary import build_text
    print(build_text(load_json(config.METRIC_DIR, "quality.json"),
                     load_json(config.METRIC_DIR, "eda_stats.json"), m,
                     load_json(config.METRIC_DIR, "env.json") or {}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=config.DATA_PATH)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--use-saved-models", action="store_true")
    ap.add_argument("--app-only", action="store_true")
    ap.add_argument("--word-only", action="store_true")
    ap.add_argument("--no-word", action="store_true")
    ap.add_argument("--no-app", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--open-figs", action="store_true")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--sample-fraction", type=float, default=1.0,
                    help="mẫu đồng đều từ CSV thật; mặc định toàn bộ")
    ap.add_argument("--port", type=int, default=8501)
    a = ap.parse_args(argv)
    if sum((a.app_only, a.word_only, a.use_saved_models)) > 1:
        ap.error("Chọn một chế độ: app-only, word-only hoặc use-saved-models.")
    if a.save_baseline:
        ap.error("Baseline lịch sử được giữ nguyên. Kết quả mới có run_id riêng.")
    if not 1 <= a.port <= 65535 or not 0 < a.sample_fraction <= 1:
        ap.error("port phải thuộc 1..65535 và sample-fraction thuộc (0,1].")
    c.setup(a.auto or a.app_only or a.word_only, a.open_figs, 3)
    if a.app_only:
        with c.step(1, "KHỞI ĐỘNG STREAMLIT", pause_after=False):
            proc = launch_app(a.port, not a.no_browser)
        wait_app(proc)
        return
    if a.word_only:
        from src.word import build
        with c.step(1, "DỰNG WORD TỪ KẾT QUẢ ĐÃ KIỂM CHỨNG", pause_after=False):
            build()
        return
    with c.step(1, "NẠP MÔ HÌNH" if a.use_saved_models else "THỰC NGHIỆM A/B × PHÂN TẦNG/THỜI GIAN"):
        if a.use_saved_models:
            saved_demo()
        else:
            if not Path(a.data).is_file():
                raise FileNotFoundError(f"Thiếu CSV PaySim: {a.data}")
            from src.experiments import run, promote
            folder = run(a.data, a.sample_fraction)
            promote(folder)
            c.ok(f"Đã kiểm chứng và công bố lần chạy {folder.name}")
    with c.step(2, "BÁO CÁO WORD"):
        if a.no_word or a.use_saved_models:
            raise c.SkipStep("demo giữ nguyên artifacts; dùng --word-only khi cần dựng báo cáo")
        from src.word import build
        build()
    proc = None
    with c.step(3, "ỨNG DỤNG WEB", pause_after=False):
        if a.no_app:
            raise c.SkipStep("--no-app")
        proc = launch_app(a.port, not a.no_browser)
    c.checklist()
    wait_app(proc)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, FileNotFoundError, ValueError, RuntimeError, TimeoutError) as exc:
        c.err(str(exc))
        if isinstance(exc, ImportError):
            c.info("Cài dependencies: python -m pip install -r requirements.txt")
        sys.exit(1)

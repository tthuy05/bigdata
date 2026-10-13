"""Kiểm tra môi trường trước khi chạy: Python, thư viện, Java, Hadoop (Windows), dữ liệu, Node.js."""
import importlib
import os
import shutil
import subprocess
import sys

import config
from src import console as c


def _cmd(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return (r.stdout + r.stderr).strip().splitlines()[0] if (r.stdout + r.stderr).strip() else ""
    except Exception:
        return None


def check(data_path, need_word=True):
    """Trả về dict trạng thái. Lỗi nghiêm trọng -> SystemExit với hướng dẫn sửa."""
    st, fatal = {}, []
    c.kv("Python", sys.version.split()[0])
    for mod in ("pyspark", "pandas", "numpy", "matplotlib", "seaborn", "sklearn", "streamlit"):
        try:
            v = getattr(importlib.import_module(mod), "__version__", "?")
            c.ok(f"Thư viện {mod:<11s} {v}")
        except ImportError:
            c.err(f"Thiếu thư viện {mod}")
            fatal.append("Chạy:  pip install -r requirements.txt")
    java = _cmd(["java", "-version"])
    if java:
        c.ok(f"Java: {java}")
    else:
        c.err("Không tìm thấy Java")
        fatal.append("Cài Java 17 (hoặc 8/11) và thêm vào PATH (xem README mục 2)")
    st["java"] = java or "N/A"
    if os.name == "nt":
        hh = os.environ.get("HADOOP_HOME", os.path.join(config.BASE_DIR, ".runtime", "hadoop"))
        wu = os.path.join(hh, "bin", "winutils.exe") if hh else ""
        if hh and os.path.exists(wu):
            c.ok(f"HADOOP_HOME = {hh} (có winutils.exe)")
        else:
            c.err("Chưa có HADOOP_HOME/winutils.exe – Spark sẽ lỗi khi ghi Parquet/mô hình")
            fatal.append("Cài winutils theo README mục 2.3")
    if os.path.exists(data_path):
        c.ok(f"Dữ liệu: {os.path.relpath(data_path)} ({os.path.getsize(data_path) / 2**20:.1f} MB)")
    else:
        c.err(f"Không tìm thấy dữ liệu: {data_path}")
        fatal.append("Tải PaySim từ Kaggle và đặt vào thư mục data/ (README mục 3)")
    node = os.environ.get("REPORT_NODE_BINARY") or shutil.which("node")
    if not node:
        for candidate in [
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "nodejs", "node.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "nodejs", "node.exe"),
        ]:
            if os.path.isfile(candidate):
                node = candidate
                break
    st["word_ready"] = False
    if need_word:
        if not node:
            c.warn("Không có Node.js – sẽ BỎ QUA bước dựng Word (cài Node.js LTS nếu cần, README mục 2.4)")
        elif not os.path.isdir(os.path.join(config.REPORT_DIR, "node_modules", "docx")):
            c.warn("Chưa cài thư viện docx cho Node – chạy 1 lần:  cd report  &&  npm install")
        else:
            c.ok(f"Node.js {_cmd([node, '--version'])} + thư viện docx (dựng được Word)")
            st["word_ready"] = True
    import json
    tj = os.path.join(config.REPORT_DIR, "team.json")
    if os.path.exists(tj):
        ids = [m["id"] for m in json.load(open(tj, encoding="utf-8"))["members"]]
        if len(ids) != len(set(ids)):
            c.warn("report/team.json có MSSV bị trùng – hãy sửa họ tên, MSSV từng thành viên")
    base_ok = all(os.path.exists(os.path.join(config.BASELINE_DIR, f))
                  for f in ("quality.json", "eda_stats.json", "model_results.json"))
    (c.ok if base_ok else c.warn)("Số liệu gốc của báo cáo (outputs/baseline): " +
          ("có – giữ nguyên để đối chiếu lịch sử" if base_ok else "chưa có; không tự tạo baseline"))
    st["baseline"] = base_ok
    if fatal:
        c.err("CHƯA ĐỦ ĐIỀU KIỆN ĐỂ CHẠY:")
        for f in dict.fromkeys(fatal):
            c.info("  -> " + f)
        raise SystemExit(1)
    return st

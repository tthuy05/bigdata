"""Dựng báo cáo Word bằng report/build_report.js (Node.js). Tự sao lưu bản cũ trước khi ghi đè,
nếu file đang mở trong Word thì ghi ra file mới có gắn thời gian."""
import os
import shutil
import subprocess
import time

import config
from src import console as c


def _locked(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "a+b"):
            return False
    except (PermissionError, OSError):
        return True


def build():
    from pathlib import Path
    from src.artifacts import validate_result
    import json
    result_path = Path(config.METRIC_DIR) / "model_results.json"
    if not result_path.is_file():
        raise FileNotFoundError("Thiếu outputs/metrics/model_results.json.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != 2 or result.get("status") != "VERIFIED":
        raise ValueError("Kết quả lịch sử chưa được kiểm chứng; hãy chạy A/B trước khi dựng Word mới.")
    validate_result(result)
    out = config.REPORT_PATH
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if _locked(out):
        c.warn(f"{os.path.relpath(out)} đang mở trong Word – ghi ra file mới thay vì ghi đè")
        out = out.replace(".docx", f"_{stamp}.docx")
    elif os.path.exists(out):
        os.makedirs(config.BACKUP_DIR, exist_ok=True)
        bk = os.path.join(config.BACKUP_DIR, f"BaoCao_BigData_Nhom8_{stamp}.docx")
        shutil.copy2(out, bk)
        c.info(f"Đã sao lưu bản Word cũ: {os.path.relpath(bk)}")
    node = os.environ.get("REPORT_NODE_BINARY") or shutil.which("node")
    if not node:
        for candidate in [
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "nodejs", "node.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "nodejs", "node.exe"),
        ]:
            if os.path.isfile(candidate):
                node = candidate
                break
    if not node:
        raise RuntimeError("Thiếu Node.js để dựng Word.")
    r = subprocess.run([node, "build_report.js", out], cwd=config.REPORT_DIR,
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        c.err("Dựng Word thất bại:")
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise RuntimeError("build_report.js lỗi")
    c.ok(f"Đã tạo báo cáo Word: {os.path.relpath(out)}")
    return out

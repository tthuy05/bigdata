"""Bộ quản lý Cụm phân tán Apache Spark Standalone (1 Master + 2 Workers).
Dựng cụm phân tán đa nút ngay trên máy đơn để mô phỏng môi trường phân tán thực thụ.

Cách dùng:
    python tools/cluster_manager.py run      (Khởi động và giữ cụm hoạt động, nhấn Ctrl+C để tắt)
    python tools/cluster_manager.py status   (Xem trạng thái cụm)
    python tools/cluster_manager.py stop     (Dừng cụm)
"""
import os
import sys
import time
import json
import socket
import urllib.request
import subprocess
from pathlib import Path

# Đảm bảo đường dẫn gốc
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src import console as c

RUNTIME_DIR = BASE_DIR / ".runtime"
CLUSTER_STATE_FILE = RUNTIME_DIR / "cluster_state.json"
PYSPARK_HOME = BASE_DIR / ".venv" / "Lib" / "site-packages" / "pyspark"
SPARK_CLASS_CMD = PYSPARK_HOME / "bin" / "spark-class.cmd"
JDK_PATH = r"C:\Program Files\Java\jdk-17"


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def is_pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        if os.name == "nt":
            res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
            return str(pid) in res.stdout
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def get_cluster_env():
    env = os.environ.copy()
    env["SPARK_HOME"] = str(PYSPARK_HOME)
    if os.path.isdir(JDK_PATH):
        env["JAVA_HOME"] = JDK_PATH
        env["PATH"] = os.path.join(JDK_PATH, "bin") + os.pathsep + env.get("PATH", "")
    local_hadoop = str(RUNTIME_DIR / "hadoop")
    if os.path.isdir(local_hadoop):
        env["HADOOP_HOME"] = local_hadoop
        env["PATH"] = os.path.join(local_hadoop, "bin") + os.pathsep + env.get("PATH", "")
    env["PYSPARK_PYTHON"] = sys.executable
    env["PYSPARK_DRIVER_PYTHON"] = sys.executable
    return env


def start_cluster(worker_cores: int = 2, worker_memory: str = "2G"):
    c.sub("KHỞI ĐỘNG CỤM PHÂN TÁN APACHE SPARK (1 MASTER + 2 WORKERS)")
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    if CLUSTER_STATE_FILE.exists():
        try:
            with open(CLUSTER_STATE_FILE, "r", encoding="utf-8") as f:
                st = json.load(f)
            if is_pid_alive(st.get("master_pid")):
                c.warn("Cụm đã đang chạy:")
                status_cluster()
                return []
        except Exception:
            pass
        CLUSTER_STATE_FILE.unlink(missing_ok=True)

    env = get_cluster_env()
    log_dir = RUNTIME_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Khởi động Master
    c.info("[1/3] Đang khởi chạy Spark Master (Port: 7077, Web UI: 8080)...")
    master_log = open(log_dir / "spark_master.log", "w", encoding="utf-8")
    master_cmd = [
        str(SPARK_CLASS_CMD),
        "org.apache.spark.deploy.master.Master",
        "--host", "127.0.0.1",
        "--port", "7077",
        "--webui-port", "8080"
    ]
    master_proc = subprocess.Popen(
        master_cmd, env=env, stdout=master_log, stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR)
    )

    # Chờ Master sẵn sàng
    deadline = time.time() + 25
    master_ready = False
    while time.time() < deadline:
        if is_port_open(7077) and is_port_open(8080):
            master_ready = True
            break
        time.sleep(0.5)

    if not master_ready:
        master_proc.terminate()
        c.err("Không thể khởi động Spark Master sau 25s. Kiểm tra logs trong .runtime/logs/spark_master.log")
        return []

    c.ok("Spark Master đã sẵn sàng tại spark://127.0.0.1:7077")
    c.info("Giao diện Master Web UI: http://localhost:8080")

    # 2. Khởi động Worker 1
    c.info(f"[2/3] Đang khởi chạy Worker 1 ({worker_cores} Cores, {worker_memory} RAM, Web UI: 8081)...")
    w1_log = open(log_dir / "spark_worker_1.log", "w", encoding="utf-8")
    w1_cmd = [
        str(SPARK_CLASS_CMD),
        "org.apache.spark.deploy.worker.Worker",
        "spark://127.0.0.1:7077",
        "--webui-port", "8081",
        "-c", str(worker_cores),
        "-m", worker_memory
    ]
    w1_proc = subprocess.Popen(
        w1_cmd, env=env, stdout=w1_log, stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR)
    )

    # 3. Khởi động Worker 2
    c.info(f"[3/3] Đang khởi chạy Worker 2 ({worker_cores} Cores, {worker_memory} RAM, Web UI: 8082)...")
    w2_log = open(log_dir / "spark_worker_2.log", "w", encoding="utf-8")
    w2_cmd = [
        str(SPARK_CLASS_CMD),
        "org.apache.spark.deploy.worker.Worker",
        "spark://127.0.0.1:7077",
        "--webui-port", "8082",
        "-c", str(worker_cores),
        "-m", worker_memory
    ]
    w2_proc = subprocess.Popen(
        w2_cmd, env=env, stdout=w2_log, stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR)
    )

    # Lưu thông tin trạng thái
    state = {
        "master_pid": master_proc.pid,
        "worker_pids": [w1_proc.pid, w2_proc.pid],
        "master_url": "spark://127.0.0.1:7077",
        "web_ui": {
            "master": "http://localhost:8080",
            "worker_1": "http://localhost:8081",
            "worker_2": "http://localhost:8082"
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(CLUSTER_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    c.info("Đang chờ 2 Workers kết nối và đăng ký vào Master...")
    time.sleep(3)
    status_cluster()
    return [master_proc, w1_proc, w2_proc]


def status_cluster():
    if not CLUSTER_STATE_FILE.exists():
        c.warn("Cụm phân tán chưa được khởi động.")
        return False

    with open(CLUSTER_STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    c.sub("TRẠNG THÁI CỤM PHÂN TÁN APACHE SPARK")
    c.kv("Địa chỉ Master (RPC)", state["master_url"])
    c.kv("Master Web UI", state["web_ui"]["master"])
    c.kv("Worker 1 Web UI", state["web_ui"]["worker_1"])
    c.kv("Worker 2 Web UI", state["web_ui"]["worker_2"])

    try:
        req = urllib.request.urlopen("http://127.0.0.1:8080/json/", timeout=3)
        info = json.loads(req.read().decode("utf-8"))
        alive_workers = [w for w in info.get("workers", []) if w.get("state") == "ALIVE"]
        c.ok(f"Trạng thái Master: {info.get('status')} | Số Workers ALIVE: {len(alive_workers)}")
        for i, w in enumerate(alive_workers, 1):
            c.info(f"   -> Worker {i}: ID={w.get('id')}, Cores={w.get('cores')}, Memory={w.get('memory')}MB")
        return len(alive_workers) >= 2
    except Exception as e:
        c.warn(f"Chưa lấy được JSON từ Master UI: {e}")
        return False


def stop_cluster():
    c.sub("DỪNG CỤM PHÂN TÁN APACHE SPARK")
    if CLUSTER_STATE_FILE.exists():
        try:
            with open(CLUSTER_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            pids = [state["master_pid"]] + state["worker_pids"]
            for pid in pids:
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
                    c.info(f"Đã dừng tiến trình PID: {pid}")
                except Exception:
                    pass
        except Exception:
            pass
        CLUSTER_STATE_FILE.unlink(missing_ok=True)

    # Dọn dẹp bất kỳ tiến trình Spark Master / Worker nào còn sót
    subprocess.run(
        ["powershell", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*org.apache.spark.deploy.master.Master*' -or $_.CommandLine -like '*org.apache.spark.deploy.worker.Worker*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
        capture_output=True
    )
    c.ok("Đã tắt hoàn toàn cụm phân tán Spark.")


def run_cluster(worker_cores: int = 2, worker_memory: str = "2G"):
    procs = start_cluster(worker_cores, worker_memory)
    if not procs:
        return
    print("\n   " + "="*65)
    c.ok("CỤM SPARK PHÂN TÁN ĐANG HOẠT ĐỘNG (Nhấn Ctrl+C để dừng cụm)")
    c.info("Master Web UI   : http://localhost:8080")
    c.info("Worker 1 Web UI : http://localhost:8081")
    c.info("Worker 2 Web UI : http://localhost:8082")
    print("   " + "="*65 + "\n")
    try:
        while True:
            time.sleep(2)
            for p in procs:
                if p.poll() is not None:
                    raise RuntimeError(f"Tiến trình Spark PID {p.pid} đã thoát sớm với mã {p.returncode}")
    except KeyboardInterrupt:
        c.info("\nĐang nhận tín hiệu dừng từ người dùng...")
    finally:
        stop_cluster()


if __name__ == "__main__":
    action = sys.argv[1].lower() if len(sys.argv) > 1 else "run"
    if action in ("start", "run"):
        run_cluster()
    elif action == "stop":
        stop_cluster()
    elif action == "status":
        status_cluster()
    else:
        print("Sử dụng: python tools/cluster_manager.py [run|status|stop]")

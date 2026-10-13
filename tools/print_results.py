"""In lại bản tóm tắt số liệu từ outputs/metrics/*.json (không cần chạy Spark).
Cách dùng:  python tools/print_results.py"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from src.summary import build_text  # noqa: E402

MET = os.path.join(BASE, "outputs", "metrics")
load = lambda n: json.load(open(os.path.join(MET, n), encoding="utf-8"))
env = load("env.json") if os.path.exists(os.path.join(MET, "env.json")) else {}
print(build_text(load("quality.json"), load("eda_stats.json"), load("model_results.json"), env))

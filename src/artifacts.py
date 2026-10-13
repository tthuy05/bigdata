"""Experiment provenance and consistency checks, shared by report/demo/tests."""
import hashlib
import json
import math
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_result(m):
    from src.modeling import select_best, SLUG
    if set(m["results"]) != set(SLUG):
        raise ValueError("Kết quả phải có đúng 4 thuật toán.")
    if m.get("selection", {}).get("dataset") != "validation" or m["best_model"] != select_best(m["results"]):
        raise ValueError("Mô hình tốt nhất không khớp tiêu chí Validation.")
    for r in m["results"].values():
        for metrics, n, positives in ((r, m["split"]["test"], m["split"]["test_fraud"]),
                                      (r["validation"], m["split"]["valid"], m["split"]["valid_fraud"])):
            if sum(metrics[k] for k in ("TP", "TN", "FP", "FN")) != n or metrics["TP"] + metrics["FN"] != positives:
                raise ValueError("Confusion matrix không khớp quy mô tập dữ liệu.")
            p = metrics["TP"] / max(metrics["TP"] + metrics["FP"], 1)
            recall = metrics["TP"] / max(positives, 1)
            f1 = 2 * p * recall / (p + recall) if p + recall else 0
            for key, value in (("precision", p), ("recall", recall), ("f1", f1)):
                if not math.isclose(metrics[key], value, abs_tol=1e-12):
                    raise ValueError(f"Chỉ số {key} không khớp ma trận nhầm lẫn.")
            for key in ("auc_pr", "auc_roc"):
                if not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
                    raise ValueError(f"Chỉ số {key} không hợp lệ.")
    n, pos = m["split"]["train"], m["split"]["train_fraud"]
    for label, count in (("1", pos), ("0", n - pos)):
        if not math.isclose(m["split"]["class_weights"][label], n / (2 * count), abs_tol=1e-10):
            raise ValueError("Trọng số lớp không khớp số mẫu Train.")
    return True

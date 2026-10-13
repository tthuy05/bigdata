"""Verify exact experiment identities, confusion counts, weights and Word table cells."""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import config
from src.artifacts import validate_result, save_json


def fmt(value, decimals=4):
    return f"{value:.{decimals}f}".replace(".", ",")


def verify(docx=None):
    manifest = json.loads(Path(config.EXPERIMENT_MANIFEST).read_text(encoding="utf-8"))
    if manifest["status"] != "VERIFIED":
        raise ValueError("Manifest chưa VERIFIED.")
    experiments = {}
    for key, entry in manifest["experiments"].items():
        m = json.loads((BASE / entry["results_path"]).read_text(encoding="utf-8"))
        validate_result(m)
        if m["run_id"] != manifest["run_id"] or m["data"]["sha256"] != manifest["data"]["sha256"]:
            raise ValueError("Nguồn dữ liệu hoặc run_id không khớp.")
        experiments[key] = m
    for split in ("stratified", "temporal"):
        if experiments[f"A_{split}"]["split"] != experiments[f"B_{split}"]["split"]:
            raise ValueError("A/B không dùng cùng các tập dữ liệu và trọng số.")
    primary = json.loads(Path(config.METRIC_DIR, "model_results.json").read_text(encoding="utf-8"))
    if primary != experiments[manifest["primary_experiment"]]:
        raise ValueError("Bản công bố không khớp thí nghiệm chính.")
    q = json.loads(Path(config.METRIC_DIR, "quality.json").read_text(encoding="utf-8"))
    eda = json.loads(Path(config.METRIC_DIR, "eda_stats.json").read_text(encoding="utf-8"))
    if q["clean_label_counts"] != {str(row["isFraud"]): row["n"] for row in eda["label"]}:
        raise ValueError("Thống kê EDA không khớp chất lượng dữ liệu.")
    if q["rows"] - q["clean_rows"] != sum(g["removed"] for g in q["removal_groups"]):
        raise ValueError("Số loại bỏ không khớp các nhóm lý do.")
    report = {"status": "VERIFIED", "run_id": manifest["run_id"], "experiments": len(experiments),
              "algorithms": sum(len(m["results"]) for m in experiments.values()),
              "checks": ["counts/metrics", "train weights", "validation selection", "A/B equal splits",
                         "run_id/source hash", "primary artifact identity", "EDA/quality counts"]}
    if docx:
        from docx import Document
        doc = Document(docx)
        abbreviation = {"Logistic Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF", "GBT": "GBT"}
        tables = [[[c.text for c in row.cells] for row in table.rows] for table in doc.tables]
        primary_table = next(t for t in tables if t[0] == ["Mô hình", "t*", "Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"])
        expected = [[abbreviation[n], fmt(r["threshold"], 3)] + [fmt(r[k]) for k in
                    ("precision", "recall", "f1", "auc_roc", "auc_pr")] for n, r in primary["results"].items()]
        if primary_table[1:5] != expected:
            raise ValueError("Các ô kết quả chính trong Word không khớp JSON.")
        compare_table = next(t for t in tables if t[0] == ["Thí nghiệm", "MH", "Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"])
        expected_all = [[key, abbreviation[n]] + [fmt(r[k]) for k in ("precision", "recall", "f1", "auc_roc", "auc_pr")]
                        for key, m in experiments.items() for n, r in m["results"].items()]
        if compare_table[1:] != expected_all:
            raise ValueError("Các ô so sánh A/B trong Word không khớp JSON.")
        report.update(docx_tables=len(tables), docx_images=len(doc.inline_shapes),
                      docx_cells_verified=len(expected)*7 + len(expected_all)*7,
                      visual_layout_status="NOT VERIFIED until render and all-page PNG inspection")
        report["checks"].append("exact Word table cells")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx")
    ap.add_argument("--out", default=str(BASE / "outputs/qa/artifact_consistency.json"))
    args = ap.parse_args()
    result = verify(args.docx)
    save_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

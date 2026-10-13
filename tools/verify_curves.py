"""Replay saved models on the exact holdout, verify real scores, persist/rebuild curves.

No training or threshold tuning. Cleaning is re-applied; the full CSV fingerprint
and split counts must match the recorded run before predictions are accepted.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve
from pyspark import StorageLevel
from pyspark.sql import functions as F

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import config
from src.artifacts import file_sha256, save_json
from src.data_loader import get_spark, load_raw, clean
from src.features import filter_relevant, add_features
from src.modeling import assign_splits, proba, confusion, SLUG
from src.model_io import load_pipeline
from src import figures


def verify(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "VERIFIED" or file_sha256(config.DATA_PATH) != manifest["data"]["sha256"]:
        raise ValueError("Lần chạy hoặc nguồn CSV không khớp.")
    output = {"status": "VERIFIED", "run_id": manifest["run_id"], "experiments": {}}
    spark = get_spark("FraudDetection-CurveVerification")
    try:
        raw = load_raw(spark)
        # Fingerprint columns/order match the training runner exactly.
        df = clean(raw)
        base = filter_relevant(df).withColumn("_row_id", F.sha2(F.to_json(F.struct(*sorted(df.columns))), 256))
        fraction = manifest["data"]["sample_fraction"]
        if fraction < 1:
            base = base.filter(F.pmod(F.xxhash64("_row_id", F.lit(config.SEED)), F.lit(1_000_000))
                               < int(fraction * 1_000_000))
        base = base.persist(StorageLevel.DISK_ONLY)
        if base.count() != manifest["data"]["modeling_rows"]:
            raise ValueError("Mẫu dùng kiểm tra không khớp mẫu thực nghiệm.")
        for method in ("stratified", "temporal"):
            assigned, _ = assign_splits(base, method)
            assigned = assigned.persist(StorageLevel.DISK_ONLY)
            assigned.count()
            for prefix in ("A", "B"):
                key = prefix + "_" + method
                m = json.loads((folder / key / "model_results.json").read_text(encoding="utf-8"))
                test = add_features(assigned.filter("_split = 'test'"), m["feature_profile"]).cache()
                if test.count() != m["split"]["test"]:
                    raise ValueError("Số dòng Test không khớp.")
                curves, checked = {}, {}
                for name, expected in m["results"].items():
                    model = load_pipeline(folder / key / "models" / SLUG[name])
                    pred = proba(model.transform(test)).cache()
                    actual = confusion(pred, expected["threshold"])
                    for metric in ("TP", "TN", "FP", "FN", "precision", "recall", "f1"):
                        if actual[metric] != expected[metric]:
                            raise ValueError(f"{key}/{name}: dự đoán nạp lại không khớp {metric}.")
                    pdf = pred.select(config.LABEL, "prob").toPandas()
                    y, scores = pdf[config.LABEL].values, pdf["prob"].values
                    p, r = figures.spark_pr_curve(y, scores)
                    x, tpr, _ = roc_curve(y, scores)
                    auc_pr, auc_roc = float(auc(r, p)), float(auc(x, tpr))
                    if not (np.isclose(auc_pr, expected["auc_pr"], atol=1e-12, rtol=1e-10) and
                            np.isclose(auc_roc, expected["auc_roc"], atol=1e-12, rtol=1e-10)):
                        raise ValueError(f"{key}/{name}: diện tích đường cong không khớp Spark.")
                    score_dir = folder / key / "test_scores"
                    score_dir.mkdir(exist_ok=True)
                    np.savez_compressed(score_dir / (SLUG[name] + ".npz"), labels=y, scores=scores)
                    curves[name] = (y, scores)
                    checked[name] = {"auc_pr": auc_pr, "auc_roc": auc_roc, "loaded_prediction_counts_match": True}
                    pred.unpersist()
                    print(key, name, "SCORE/CURVE_VERIFIED", flush=True)
                original_dir = config.FIG_DIR
                config.FIG_DIR = str(folder / key / "figures")
                try:
                    figures.result_figures(m["results"], curves, m["feature_importance"], m["feature_importance_model"])
                finally:
                    config.FIG_DIR = original_dir
                output["experiments"][key] = checked
                test.unpersist()
            assigned.unpersist()
        save_json(BASE / "outputs/qa/curve_consistency.json", output)
        return output
    finally:
        spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    args = ap.parse_args()
    verify(args.folder)

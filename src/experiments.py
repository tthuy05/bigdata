"""Fair A/B × stratified/time experiments; immutable run folders, explicit promotion."""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import functions as F

import config
from src import console as c
from src.artifacts import save_json, file_sha256, validate_result
from src.data_loader import get_spark, load_raw, quality_report, clean
from src.features import add_features, filter_relevant, feature_names
from src import modeling as M


def run(data_path, sample_fraction=1.0, methods=("stratified", "temporal")):
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample-fraction phải thuộc (0, 1].")
    sidecar = Path(str(data_path) + ".metadata.json")
    if sidecar.exists() and json.loads(sidecar.read_text(encoding="utf-8")).get("kind") == "synthetic":
        raise ValueError("Dữ liệu tự tạo chỉ dành cho unit test, không được xuất thành kết quả thực nghiệm PaySim.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    folder = Path(config.EXPERIMENT_DIR) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest = {"schema_version": 2, "run_id": run_id, "status": "RUNNING",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "data": {"path": str(Path(data_path).resolve()), "sha256": file_sha256(data_path),
                         "sample_fraction": sample_fraction,
                         "sampling": "uniform stable row-hash sample, same for all classes; no oversampling",
                         "origin": "user-provided PaySim CSV; not externally authenticated"},
                "experiments": {}, "primary_experiment": "B_temporal" if "temporal" in methods else "B_stratified",
                "selection_policy": "predefined B with temporal holdout; algorithm selected on Validation AUC-PR/F1/Precision"}
    save_json(folder / "manifest.json", manifest)
    spark = None
    try:
        spark = get_spark()
        raw = load_raw(spark, data_path)
        quality_start = time.perf_counter()
        q = quality_report(raw)
        q["time_s"] = round(time.perf_counter() - quality_start, 3)
        cleaned, audit = clean(raw, return_audit=True)
        q.update(audit)
        cleaned = cleaned.persist(StorageLevel.DISK_ONLY)
        cleaned.count()
        save_json(folder / "quality.json", q)
        from src.stats import compute_stats
        stats, eda_sample = compute_stats(spark, cleaned)
        save_json(folder / "eda_stats.json", stats)
        base = filter_relevant(cleaned).withColumn("_row_id", F.sha2(F.to_json(F.struct(
            *sorted(cleaned.columns))), 256))
        if sample_fraction < 1:
            # This filter depends only on row identity, not on class or model scores.
            base = base.filter(F.pmod(F.xxhash64("_row_id", F.lit(config.SEED)), F.lit(1_000_000))
                               < int(sample_fraction * 1_000_000))
        base = base.persist(StorageLevel.DISK_ONLY)
        manifest["data"]["modeling_rows"] = base.count()
        manifest["data"]["clean_rows"] = q["clean_rows"]
        env = {"spark_version": spark.version, "python": platform.python_version(),
               "java": subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0],
               "master": spark.sparkContext.master, "cores": os.cpu_count(),
               "driver_memory": config.DRIVER_MEMORY, "os": platform.platform(), "seed": config.SEED}
        from src import figures
        from src.diagrams import draw_all
        old_fig = config.FIG_DIR
        config.FIG_DIR = str(folder / "figures")
        Path(config.FIG_DIR).mkdir()
        try:
            figures.eda_figures(stats, eda_sample)
            draw_all(config.FIG_DIR)
            del eda_sample
            for method in methods:
                assigned, split_meta = M.assign_splits(base, method)
                assigned = assigned.persist(StorageLevel.DISK_ONLY)
                assigned.count()
                # Row identity disjointness is measured once and shared by A/B.
                overlap = assigned.groupBy("_row_id").agg(F.countDistinct("_split").alias("n")) \
                                  .filter("n > 1").count()
                if overlap:
                    raise ValueError(f"Có {overlap} bản ghi chồng lấn giữa các tập.")
                for letter, profile in (("A", "post_transaction"), ("B", "pre_transaction")):
                    key = f"{letter}_{method}"
                    c.sub(f"THÍ NGHIỆM {key}: {profile}")
                    experiment_folder = folder / key
                    experiment_folder.mkdir()
                    data = add_features(assigned, profile)
                    train, valid, test, split = M.split_data(data, method, assigned=True)
                    split.update(split_meta, overlapping_rows=overlap)
                    try:
                        results, curves, fitted = M.train_all(train, valid, test, False, {}, profile,
                                                             str(experiment_folder / "models"))
                        extra = M.summarize(results, fitted, test, profile)
                        m = {"schema_version": 2, "experiment": key, "feature_profile": profile,
                             "features": feature_names(profile), "split": split, "results": results,
                             "status": "VERIFIED", "run_id": run_id, "data": manifest["data"], **extra}
                        validate_result(m)
                        config.FIG_DIR = str(experiment_folder / "figures")
                        Path(config.FIG_DIR).mkdir()
                        m["figures"] = [os.path.basename(p) for p in figures.result_figures(
                            results, curves, extra["feature_importance"], extra["feature_importance_model"])]
                        save_json(experiment_folder / "model_results.json", m)
                        best = extra["best_model"]
                        bundle = {"name": best, "threshold": results[best]["threshold"],
                                  "metrics": results[best], "feature_profile": profile,
                                  "features": feature_names(profile), "run_id": run_id, "experiment": key,
                                  "model_path": str((experiment_folder / "models" / M.SLUG[best]).relative_to(config.BASE_DIR)),
                                  "selection": extra["selection"], "status": "VERIFIED"}
                        save_json(experiment_folder / "best_model_info.json", bundle)
                        manifest["experiments"][key] = {"results_path": str((experiment_folder / "model_results.json")
                                                                                    .relative_to(config.BASE_DIR)),
                                                            "bundle_path": str((experiment_folder / "best_model_info.json")
                                                                                    .relative_to(config.BASE_DIR)),
                                                            "best_model": best, "status": "VERIFIED"}
                        save_json(folder / "manifest.json", manifest)
                        del curves, fitted
                    finally:
                        for frame in (train, valid, test):
                            frame.unpersist()
                assigned.unpersist()
            env["total_time_s"] = round(time.perf_counter() - start, 2)
            save_json(folder / "env.json", env)
            manifest["status"] = "VERIFIED"
            save_json(folder / "manifest.json", manifest)
        finally:
            config.FIG_DIR = old_fig
        return folder
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        save_json(folder / "manifest.json", manifest)
        raise
    finally:
        if spark is not None:
            spark.stop()


def promote(folder):
    """Publish only a complete, validated run; preserve every previous artifact."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "VERIFIED":
        raise ValueError("Không công bố lần chạy chưa hoàn tất kiểm chứng.")
    expected = {f"{a}_{s}" for a in ("A", "B") for s in ("stratified", "temporal")}
    if set(manifest["experiments"]) != expected:
        raise ValueError("Cần đủ 4 thí nghiệm A/B × phân tầng/thời gian trước khi công bố.")
    for entry in manifest["experiments"].values():
        validate_result(json.loads((Path(config.BASE_DIR) / entry["results_path"]).read_text(encoding="utf-8")))
    primary = manifest["primary_experiment"]
    chosen = folder / primary
    stamp = manifest["run_id"]
    backup = Path(config.BACKUP_DIR) / ("before_" + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    for path in (Path(config.METRIC_DIR), Path(config.FIG_DIR), Path(config.OUTPUT_DIR) / "ket_qua.txt",
                 Path(config.MODEL_DIR) / "best_model_info.json"):
        if path.exists():
            if path.is_dir():
                shutil.copytree(path, backup / path.name)
            else:
                shutil.copy2(path, backup / path.name)
    for name in ("quality.json", "eda_stats.json", "env.json"):
        shutil.copy2(folder / name, Path(config.METRIC_DIR) / name)
    shutil.copy2(chosen / "model_results.json", Path(config.METRIC_DIR) / "model_results.json")
    shutil.copy2(chosen / "best_model_info.json", Path(config.MODEL_DIR) / "best_model_info.json")
    for source in (folder / "figures", chosen / "figures"):
        shutil.copytree(source, config.FIG_DIR, dirs_exist_ok=True)
    # Old application screenshots describe the old model; they remain in the backup.
    for path in Path(config.FIG_DIR).glob("fig_app_*.png"):
        if path.name != "fig_app_architecture.png":
            path.unlink()
    save_json(config.EXPERIMENT_MANIFEST, manifest)
    from src.summary import build_text
    load = lambda name: json.loads((Path(config.METRIC_DIR) / name).read_text(encoding="utf-8"))
    Path(config.OUTPUT_DIR, "ket_qua.txt").write_text(build_text(
        load("quality.json"), load("eda_stats.json"), load("model_results.json"), load("env.json")), encoding="utf-8")

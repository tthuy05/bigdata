"""Thực nghiệm đối chứng trên Apache Spark (PySpark MLlib).
Chạy tuần tự 4 mô hình:
1. Logistic Regression (Baseline)
2. Decision Tree
3. Random Forest
4. GBT (Gradient Boosted Trees)
Đo lường thời gian huấn luyện (s), thời gian dự đoán (s) và các chỉ số độ chính xác.
Tỷ lệ chia dữ liệu: 70% Train, 10% Validation, 20% Test (Stratified, seed=42).
"""
import os
import time
import json
from pathlib import Path
import numpy as np

from pyspark.sql import functions as F, DataFrame, SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier,
    RandomForestClassifier, GBTClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

import config
from src import console as c
from src.data_loader import get_spark, load_raw, clean
from src.features import add_features, filter_relevant, preprocessing_stages, add_class_weight
from src import modeling as M

from pyspark import StorageLevel

SLUG = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "GBT": "gbt"
}


def build_spark_models():
    """Khởi tạo 4 mô hình PySpark MLlib khớp siêu tham số với Python thuần."""
    return {
        "Logistic Regression": (
            LogisticRegression(
                maxIter=25, tol=1e-3, regParam=0.01, elasticNetParam=0.0,
                featuresCol="features", labelCol=config.LABEL, weightCol="classWeight"
            ),
            True, True  # (model, scale_flag, is_baseline)
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                maxDepth=10, minInstancesPerNode=5, maxMemoryInMB=1024, seed=config.SEED,
                featuresCol="raw_features", labelCol=config.LABEL, weightCol="classWeight"
            ),
            False, False
        ),
        "Random Forest": (
            RandomForestClassifier(
                numTrees=30, maxDepth=10, maxMemoryInMB=1024,
                featureSubsetStrategy="sqrt", subsamplingRate=0.8, seed=config.SEED,
                featuresCol="raw_features", labelCol=config.LABEL, weightCol="classWeight"
            ),
            False, False
        ),
        "GBT": (
            GBTClassifier(
                maxIter=20, maxDepth=5, stepSize=0.1, maxMemoryInMB=1024, seed=config.SEED,
                featuresCol="raw_features", labelCol=config.LABEL, weightCol="classWeight"
            ),
            False, False
        )
    }


def run_spark_benchmark(spark: SparkSession = None, sample_fraction: float = 1.0) -> dict:
    """Thực thi đầy đủ phiên đo kiểm Apache Spark MLlib."""
    c.sub("PHIÊN 2: CHẠY BENCHMARK APACHE SPARK (PYSPARK MLLIB)")

    stop_spark_later = False
    if spark is None:
        spark = get_spark("Benchmark-Spark-PaySim")
        stop_spark_later = True

    try:
        t_prep_start = time.perf_counter()
        raw = load_raw(spark, config.DATA_PATH)
        cleaned = clean(raw)
        base = filter_relevant(cleaned).withColumn(
            "_row_id", F.sha2(F.to_json(F.struct(*sorted(cleaned.columns))), 256)
        )

        if sample_fraction < 1.0:
            c.info(f"[Spark] Lấy mẫu ngẫu nhiên {sample_fraction * 100:.1f}%...")
            base = base.filter(
                F.pmod(F.xxhash64("_row_id", F.lit(config.SEED)), F.lit(1_000_000))
                < int(sample_fraction * 1_000_000)
            )

        data = add_features(base, profile="post_transaction")
        prep_time_s = round(time.perf_counter() - t_prep_start, 3)

        c.info(f"[Spark] Chia dữ liệu 70% Train, 10% Valid, 20% Test (Stratified)...")
        train, valid, test, split_meta = M.split_data(data, method="stratified")

        # Tiền xử lý đặc trưng 1 lần duy nhất và nạp RAM (Công bằng 100% với Python thuần)
        c.info("[Spark] Đang tiền xử lý Pipeline đặc trưng và nạp bộ nhớ đệm RAM...")
        prep_stages = preprocessing_stages(scale=True, profile="post_transaction")
        prep_model = Pipeline(stages=prep_stages).fit(train)

        train_prep = prep_model.transform(train).select(
            "features", "raw_features", config.LABEL, "classWeight"
        ).persist(StorageLevel.MEMORY_AND_DISK)
        train_prep.count()  # Kích hoạt nạp RAM

        valid_prep = prep_model.transform(valid).select(
            "features", "raw_features", config.LABEL
        ).persist(StorageLevel.MEMORY_AND_DISK)
        valid_prep.count()

        test_prep = prep_model.transform(test).select(
            "features", "raw_features", config.LABEL
        ).persist(StorageLevel.MEMORY_AND_DISK)
        test_prep.count()

        ev = BinaryClassificationEvaluator(labelCol=config.LABEL, rawPredictionCol="prob", numBins=0)
        results = {}
        curves_data = {}
        models_dict = build_spark_models()

        for name, (clf, scale, is_baseline) in models_dict.items():
            c.sub(f"[Spark] Huấn luyện & Đánh giá: {name}")

            # Đo thời gian huấn luyện (Fit)
            t0_train = time.perf_counter()
            fitted_model = clf.fit(train_prep)
            t_train = round(time.perf_counter() - t0_train, 3)
            c.info(f"Thời gian huấn luyện (Train time): {t_train:.3f}s")

            # Tối ưu ngưỡng trên Validation
            valid_pred = M.proba(fitted_model.transform(valid_prep)).cache()
            valid_pdf = valid_pred.select(config.LABEL, "prob").toPandas()
            thr = M.threshold_from_scores(valid_pdf[config.LABEL].values, valid_pdf["prob"].values)
            valid_pred.unpersist()
            c.info(f"Ngưỡng tối ưu F1 trên Validation: t* = {thr:.4f}")

            # Đo thời gian dự đoán trên Test (Predict)
            t0_pred = time.perf_counter()
            test_pred = M.proba(fitted_model.transform(test_prep)).cache()
            n_count = test_pred.count()  # Kích hoạt thực thi Spark DataFrame
            t_pred = round(time.perf_counter() - t0_pred, 3)
            c.info(f"Thời gian dự đoán Test ({n_count:,} dòng): {t_pred:.3f}s")

            # Đánh giá chỉ số trên Test
            met = M.confusion(test_pred, thr)
            met["accuracy"] = (met["TP"] + met["TN"]) / max(n_count, 1)
            met["auc_roc"] = float(ev.evaluate(test_pred, {ev.metricName: "areaUnderROC"}))
            met["auc_pr"] = float(ev.evaluate(test_pred, {ev.metricName: "areaUnderPR"}))
            met["at_default_0_5"] = M.confusion(test_pred, 0.5)

            met["etl_prep_time_s"] = prep_time_s
            met["train_time_s"] = t_train
            met["predict_time_s"] = t_pred
            met["total_time_s"] = round(t_train + t_pred, 3)
            met["total_pipeline_time_s"] = round(prep_time_s + t_train + t_pred, 3)
            met["threshold"] = float(thr)
            met["is_baseline"] = is_baseline
            met["platform"] = "Apache Spark (PySpark MLlib)"

            results[name] = met

            # Trích xuất dữ liệu xác suất để vẽ ROC/PR
            pdf = test_pred.select(config.LABEL, "prob").toPandas()
            test_pred.unpersist()

            curves_data[name] = {
                "labels": pdf[config.LABEL].values.tolist(),
                "probs": pdf["prob"].values.tolist()
            }

            c.table(["Chỉ số", "Giá trị"], [
                ["Ngưỡng t*", f"{met['threshold']:.4f}"],
                ["Accuracy", f"{met['accuracy']:.6f}"],
                ["Precision", f"{met['precision']:.4f}"],
                ["Recall", f"{met['recall']:.4f}"],
                ["F1-Score", f"{met['f1']:.4f}"],
                ["AUC-ROC", f"{met['auc_roc']:.4f}"],
                ["AUC-PR", f"{met['auc_pr']:.4f}"],
                ["Thời gian huấn luyện", f"{met['train_time_s']}s"],
                ["Thời gian dự đoán", f"{met['predict_time_s']}s"]
            ])

        output_data = {
            "platform": "Apache Spark (PySpark MLlib)",
            "prep_time_s": prep_time_s,
            "sample_fraction": sample_fraction,
            "split_ratio": {"train": 0.7, "validation": 0.1, "test": 0.2},
            "counts": {
                "train": split_meta["train"],
                "valid": split_meta["valid"],
                "test": split_meta["test"],
                "fraud_test": split_meta["test_fraud"]
            },
            "results": results
        }

        out_dir = Path(config.OUTPUT_DIR) / "benchmarks"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "spark_results.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        with open(out_dir / "spark_curves.json", "w", encoding="utf-8") as f:
            json.dump(curves_data, f)

        c.ok(f"[Spark] Hoàn tất và lưu kết quả vào {out_dir / 'spark_results.json'}")
        return output_data

    finally:
        if stop_spark_later and spark is not None:
            spark.stop()

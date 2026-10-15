"""Chia dữ liệu, huấn luyện (hoặc nạp lại) 4 mô hình Spark MLlib, chọn ngưỡng, đánh giá.
Chỉ tính và in số liệu – hình được vẽ ở src/figures.py."""
import os
import shutil
import time

import numpy as np
from sklearn.metrics import precision_recall_curve
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier,
                                       RandomForestClassifier, GBTClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F

import config
from src import console as c
from src.features import add_class_weight, preprocessing_stages, FEATURE_NAMES, feature_names
from src.model_io import load_pipeline, save_pipeline

COLORS = {"Logistic Regression": "#4C72B0", "Decision Tree": "#55A868",
          "Random Forest": "#C44E52", "GBT": "#8172B2"}
SLUG = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
        "Random Forest": "random_forest", "GBT": "gbt"}


def build_models():
    common = dict(featuresCol="features", labelCol=config.LABEL, weightCol="classWeight")
    return {
        "Logistic Regression": (LogisticRegression(maxIter=100, regParam=0.001,
                                                   elasticNetParam=0.0, **common), True),
        "Decision Tree": (DecisionTreeClassifier(maxDepth=10, minInstancesPerNode=5,
                                                 seed=config.SEED, **common), False),
        "Random Forest": (RandomForestClassifier(numTrees=50, maxDepth=12,
                                                 featureSubsetStrategy="sqrt",
                                                 subsamplingRate=0.8, seed=config.SEED,
                                                 **common), False),
        "GBT": (GBTClassifier(maxIter=50, maxDepth=6, stepSize=0.1, seed=config.SEED,
                              **common), False),
    }


# --------------------------------------------------------------------------- chia dữ liệu
def stratified_split(df: DataFrame, ratios):
    """Chia theo từng lớp bằng randomSplit (seed = 42) để giữ nguyên tỷ lệ gian lận ở mọi tập.
    Giống hệt bản v1.1 -> cùng dữ liệu Parquet và cùng máy thì ra đúng các tập như báo cáo."""
    pos = df.filter(F.col(config.LABEL) == 1).randomSplit(ratios, seed=config.SEED)
    neg = df.filter(F.col(config.LABEL) == 0).randomSplit(ratios, seed=config.SEED)
    return [p.unionByName(n) for p, n in zip(pos, neg)]


def assign_splits(data, method="stratified"):
    """Assign each row once, before A/B feature engineering; whole steps stay together."""
    if "_row_id" not in data.columns:
        data = data.withColumn("_row_id", F.sha2(F.to_json(F.struct(*sorted(data.columns))), 256))
    if method == "stratified":
        # Keep the original class-wise randomSplit, with stable physical ordering.
        ordered = data.repartition(config.SHUFFLE_PARTITIONS, "_row_id").sortWithinPartitions("_row_id")
        parts = stratified_split(ordered, [0.7, 0.1, 0.2])
        return parts[0].withColumn("_split", F.lit("train")).unionByName(
            parts[1].withColumn("_split", F.lit("valid"))).unionByName(
            parts[2].withColumn("_split", F.lit("test"))), {"method": method, "seed": config.SEED}
    if method != "temporal":
        raise ValueError(f"Phương pháp chia không hợp lệ: {method}")
    counts = data.groupBy("step").count().orderBy("step").collect()
    total = sum(r["count"] for r in counts)
    if len(counts) < 3:
        raise ValueError("Cần ít nhất 3 bước thời gian khác nhau để chia theo thời gian.")
    cumulative, cutoffs = 0, []
    for row in counts:
        cumulative += row["count"]
        cutoffs.append((row["step"], cumulative / total))
    # Only row counts determine the boundaries, never labels/model performance.
    train_cut = min(cutoffs[:-2], key=lambda x: abs(x[1] - 0.7))[0]
    candidates = [x for x in cutoffs[:-1] if x[0] > train_cut]
    valid_cut = min(candidates, key=lambda x: abs(x[1] - 0.8))[0]
    assigned = data.withColumn("_split", F.when(F.col("step") <= train_cut, "train")
                              .when(F.col("step") <= valid_cut, "valid").otherwise("test"))
    return assigned, {"method": method, "train_end_step": train_cut, "valid_end_step": valid_cut,
                      "boundary_policy": "nearest cumulative row fractions 70/80%; entire steps"}


def split_data(data: DataFrame, method="stratified", assigned=False):
    data, metadata = (data, {"method": method}) if assigned else assign_splits(data, method)
    train, valid, test = [data.filter(F.col("_split") == name) for name in ("train", "valid", "test")]
    train = add_class_weight(train).cache()
    valid, test = valid.cache(), test.cache()
    split = dict(metadata)
    rows = []
    for name, d in (("train", train), ("valid", valid), ("test", test)):
        summary = d.agg(F.count("*").alias("n"), F.sum(config.LABEL).alias("k"),
                        F.min("step").alias("start"), F.max("step").alias("end")).first()
        n, k = summary["n"], summary["k"] or 0
        if not n or not 0 < k < n:
            raise ValueError(f"Tập {name} cần đủ cả hai lớp; thực tế {n} giao dịch, {k} gian lận.")
        split[name], split[name + "_fraud"] = n, k
        split[name + "_normal"] = n - k
        split[name + "_fraud_rate"] = k / n
        split[name + "_step_range"] = [summary["start"], summary["end"]]
        rows.append([{"train": "Huấn luyện (70%)", "valid": "Kiểm định (10%)",
                      "test": "Kiểm thử (20%)"}[name], n, k, round(100 * k / n, 4)])
    c.table(["Tập", "Số giao dịch", "Số gian lận", "Tỷ lệ %"], rows)
    w = {str(r["isFraud"]): r["classWeight"]
         for r in train.select("isFraud", "classWeight").distinct().collect()}
    split["class_weights"] = w
    c.info(f"Trọng số lớp (w_c = N / (2·N_c)): gian lận = {w['1']:.2f}, bình thường = {w['0']:.4f}")
    if method == "temporal" and not (split["train_step_range"][1] < split["valid_step_range"][0]
                                    <= split["valid_step_range"][1] < split["test_step_range"][0]):
        raise ValueError("Tập thời gian bị chồng lấn.")
    return train, valid, test, split


# --------------------------------------------------------------------------- đánh giá
def proba(pred: DataFrame) -> DataFrame:
    return pred.withColumn("prob", vector_to_array("probability")[1])


def tune_threshold(model, valid: DataFrame) -> float:
    """Ngưỡng làm F1 lớn nhất trên tập KIỂM ĐỊNH (không dùng tập kiểm thử)."""
    pdf = proba(model.transform(valid)).select(config.LABEL, "prob").toPandas()
    return threshold_from_scores(pdf[config.LABEL].values, pdf["prob"].values)


def threshold_from_scores(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if not len(labels) or set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Validation phải có đủ hai lớp và điểm dự đoán hữu hạn.")
    p, r, t = precision_recall_curve(labels, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def confusion(pred: DataFrame, thr: float) -> dict:
    pred = pred.withColumn("pred", (F.col("prob") >= thr).cast("int"))
    m = {(r[config.LABEL], r["pred"]): r["count"]
         for r in pred.groupBy(config.LABEL, "pred").count().collect()}
    tp, tn, fp, fn = m.get((1, 1), 0), m.get((0, 0), 0), m.get((0, 1), 0), m.get((1, 0), 0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "specificity": tn / (tn + fp) if tn + fp else 0.0}


def rule_baseline(test: DataFrame) -> dict:
    m = {(r[config.LABEL], r["isFlaggedFraud"]): r["count"]
         for r in test.groupBy(config.LABEL, "isFlaggedFraud").count().collect()}
    tp, tn, fp, fn = m.get((1, 1), 0), m.get((0, 0), 0), m.get((0, 1), 0), m.get((1, 0), 0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def print_confusion(name, met):
    c.info(f"Ma trận nhầm lẫn của {name} trên tập kiểm thử:")
    c.table(["Thực tế \\ Dự đoán", "Dự đoán 0", "Dự đoán 1"],
            [["Thực tế 0 (bình thường)", met["TN"], met["FP"]],
             ["Thực tế 1 (gian lận)", met["FN"], met["TP"]]])


def _safe_relpath(p):
    try:
        return os.path.relpath(p)
    except (ValueError, OSError):
        return str(p)


# --------------------------------------------------------------------------- huấn luyện
def train_all(train, valid, test, use_saved: bool, prev: dict, profile="post_transaction", model_dir=None):
    results, curves, fitted = {}, {}, {}
    ev = BinaryClassificationEvaluator(labelCol=config.LABEL, rawPredictionCol="prob", numBins=0)
    model_dir = model_dir or config.MODEL_DIR
    if use_saved:
        missing = [n for n in SLUG if not os.path.isdir(os.path.join(model_dir, SLUG[n], "metadata"))]
        if missing:
            raise FileNotFoundError("Thiếu mô hình đã lưu; không huấn luyện thay thế: " + ", ".join(missing))
    for i, (name, (clf, scale)) in enumerate(build_models().items(), 1):
        c.sub(f"Mô hình {i}/4: {name}")
        path = os.path.join(model_dir, SLUG[name])
        t0 = time.time()
        if use_saved and os.path.exists(path):
            model = load_pipeline(path)
            t_train = prev.get(name, {}).get("train_time_s")
            c.info(f"Nạp mô hình đã lưu: {_safe_relpath(path)} ({time.time() - t0:.1f}s)")
        else:
            stages = preprocessing_stages(scale, profile) + [clf]
            c.info("Pipeline: " + " -> ".join(type(s).__name__ for s in stages))
            model = Pipeline(stages=stages).fit(train)
            t_train = round(time.time() - t0, 1)
            save_pipeline(model, path)
            c.info(f"Huấn luyện xong trong {t_train:.1f}s, đã lưu: {_safe_relpath(path)}")
        valid_pred = proba(model.transform(valid)).cache()
        valid_pdf = valid_pred.select(config.LABEL, "prob").toPandas()
        thr = threshold_from_scores(valid_pdf[config.LABEL].values, valid_pdf["prob"].values)
        validation = confusion(valid_pred, thr)
        validation["auc_roc"] = ev.evaluate(valid_pred, {ev.metricName: "areaUnderROC"})
        validation["auc_pr"] = ev.evaluate(valid_pred, {ev.metricName: "areaUnderPR"})
        valid_pred.unpersist()
        c.info(f"Ngưỡng tối ưu F1 trên tập kiểm định: t* = {thr:.4f}")
        results[name] = {"validation": validation, "threshold": thr, "train_time_s": t_train}
        fitted[name] = model
    # Freeze selection before the first Test prediction is requested.
    selected_before_test = select_best(results)
    c.info(f"Đã cố định mô hình từ Validation trước đánh giá Test: {selected_before_test}")
    for name, model in fitted.items():
        thr = results[name]["threshold"]
        validation = results[name]["validation"]
        t_train = results[name]["train_time_s"]
        t0 = time.time()
        pred = proba(model.transform(test)).cache()
        n_pred = pred.count()  # Force scoring separately from evaluation/collection.
        t_predict = time.time() - t0
        met = confusion(pred, thr)
        if sum(met[k] for k in ("TP", "TN", "FP", "FN")) != n_pred:
            raise RuntimeError("Confusion matrix không khớp số dự đoán.")
        met["auc_roc"] = ev.evaluate(pred, {ev.metricName: "areaUnderROC"})
        met["auc_pr"] = ev.evaluate(pred, {ev.metricName: "areaUnderPR"})
        met["at_default_0_5"] = {k: v for k, v in confusion(pred, 0.5).items()
                                 if k in ("TP", "FP", "FN", "precision", "recall", "f1")}
        met.update(threshold=thr, train_time_s=t_train, predict_time_s=round(t_predict, 3),
                   validation=validation, evaluation_scope="test", score_column="probability[1]",
                   auc_method="Spark BinaryClassificationEvaluator numBins=0",
                   timing_scope="fit only; transform+cache materialization only; excludes metrics")
        pdf = pred.select(config.LABEL, "prob").toPandas()
        curves[name] = (pdf[config.LABEL].values, pdf["prob"].values)
        score_dir = os.path.join(os.path.dirname(model_dir), "test_scores")
        os.makedirs(score_dir, exist_ok=True)
        np.savez_compressed(os.path.join(score_dir, SLUG[name] + ".npz"),
                            labels=pdf[config.LABEL].values, scores=pdf["prob"].values)
        pred.unpersist()
        print_confusion(name, met)
        c.table(["Chỉ số", "Giá trị"],
                [["Precision", met["precision"]], ["Recall", met["recall"]], ["F1-score", met["f1"]],
                 ["AUC-ROC", met["auc_roc"]], ["AUC-PR", met["auc_pr"]]])
        met["selected_before_test"] = name == selected_before_test
        results[name] = met
    return results, curves, fitted


def save_best(fitted, best):
    save_pipeline(fitted[best], config.BEST_MODEL_PATH, overwrite=True)


def select_best(results):
    """No Test metric is read here. Exact AUC-PR, then F1, then Precision; stable tie."""
    if not results or any("validation" not in r for r in results.values()):
        raise ValueError("Cần chỉ số Validation của tất cả mô hình để lựa chọn.")
    return max(sorted(results), key=lambda n: tuple(results[n]["validation"][k]
                                                   for k in ("auc_pr", "f1", "precision")))


def summarize(results, fitted, test, profile="post_transaction"):
    """Bảng so sánh, chọn mô hình tốt nhất, độ quan trọng đặc trưng, hệ số LR, luật cơ sở."""
    rb = rule_baseline(test)
    c.sub("Bảng so sánh 4 mô hình trên tập kiểm thử (dữ liệu của Bảng 13)")
    rows = [[n, round(r["threshold"], 3), r["precision"], r["recall"], r["f1"], r["auc_roc"],
             r["auc_pr"]] for n, r in results.items()]
    rows.append(["Luật isFlaggedFraud", "-", rb["precision"], rb["recall"], rb["f1"], "-", "-"])
    c.table(["Mô hình", "t*", "Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"], rows)

    c.sub("Ma trận nhầm lẫn 4 mô hình (dữ liệu của Bảng 14)")
    c.table(["Mô hình", "TP", "FP", "FN", "TN", "Specificity", "Accuracy"],
            [[n, r["TP"], r["FP"], r["FN"], r["TN"], round(r["specificity"], 5), round(r["accuracy"], 5)]
             for n, r in results.items()])

    c.sub("Ngưỡng 0,5 so với t* (dữ liệu của Bảng 15)")
    rows = []
    for n, r in results.items():
        d = r["at_default_0_5"]
        rows.append([n, "0.5", d["FP"], d["FN"], d["precision"], d["recall"], d["f1"]])
        rows.append(["", f"{r['threshold']:.3f}", r["FP"], r["FN"], r["precision"], r["recall"], r["f1"]])
    c.table(["Mô hình", "Ngưỡng", "FP", "FN", "Precision", "Recall", "F1"], rows)

    c.sub("Thời gian (dữ liệu của Bảng 16) – dao động giữa các lần chạy, KHÔNG đưa vào đối chiếu")
    c.table(["Mô hình", "Huấn luyện (s)", "Dự đoán test (s)"],
            [[n, r["train_time_s"], r["predict_time_s"]] for n, r in results.items()])

    best = select_best(results)
    tree = best if best != "Logistic Regression" else "Random Forest"
    imp = fitted[tree].stages[-1].featureImportances.toArray()
    names = feature_names(profile)
    if len(imp) != len(names):
        raise ValueError("Kích thước vector không khớp danh sách đặc trưng.")
    importances = {names[k]: float(imp[k]) for k in range(len(imp))}
    c.sub(f"Độ quan trọng đặc trưng – {tree}")
    c.table(["Hạng", "Đặc trưng", "Độ quan trọng"],
            [[i + 1, k, v] for i, (k, v) in enumerate(sorted(importances.items(), key=lambda x: -x[1]))])
    lr_coef = dict(zip(names, map(float,
                   fitted["Logistic Regression"].stages[-1].coefficients.toArray())))
    c.sub("Hệ số Logistic Regression (đã chuẩn hóa; dương = làm tăng khả năng gian lận)")
    c.table(["Đặc trưng", "Hệ số"], [[k, v] for k, v in sorted(lr_coef.items(), key=lambda x: -abs(x[1]))])

    c.sub("Chọn mô hình tốt nhất")
    c.info("Tiêu chí trên VALIDATION: AUC-PR cao nhất, sau đó F1, Precision; không làm tròn khi chọn")
    c.ok(f"MÔ HÌNH ĐƯỢC CHỌN: {best}; ngưỡng t* = {results[best]['threshold']:.6f}")
    return {"rule_baseline": rb, "best_model": best, "feature_importance_model": tree,
            "feature_importance": importances, "lr_coefficients": lr_coef,
            "selection": {"dataset": "validation", "criteria": ["auc_pr", "f1", "precision"]}}

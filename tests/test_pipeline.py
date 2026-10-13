import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.sql import functions as F

import config
from src.artifacts import validate_result
from src.data_loader import clean, load_raw, PAYSIM_SCHEMA
from src.features import add_class_weight, add_features, feature_names, preprocessing_stages
from src.modeling import assign_splits, split_data, threshold_from_scores, select_best, train_all, SLUG
from src.prediction import load_bundle, predict_validated, validate_csv


def test_cleaning_keeps_zero_fraud_and_accounts_each_removal(spark, transactions):
    good = transactions.limit(1)
    zero = good.withColumn("amount", F.lit(0.0)).withColumn("nameOrig", F.lit("zero"))
    negative = good.withColumn("amount", F.lit(-1.0)).withColumn("nameOrig", F.lit("negative"))
    null = good.withColumn("nameOrig", F.lit(None).cast("string"))
    bad_type = good.withColumn("type", F.lit("UNKNOWN"))
    infinite = good.withColumn("amount", F.lit(float("inf")))
    data = good.unionByName(good).unionByName(zero).unionByName(negative).unionByName(null).unionByName(bad_type).unionByName(infinite)
    output, audit = clean(data, True)
    assert output.count() == 2
    assert output.filter("amount = 0 AND isFraud = 1").count() == 1
    assert sum(g["removed"] for g in audit["removal_groups"]) == 5
    assert audit["clean_label_counts"] == {"1": 2}


def test_malformed_csv_fails_instead_of_silently_discarding(spark, tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(",".join(PAYSIM_SCHEMA.fieldNames()) + "\nnot_an_int,TRANSFER,1,C1,0,0,C2,0,0,1,0\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_raw(spark, str(path)).collect()
    path.write_text("type,step,amount\nTRANSFER,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="11 cột"):
        load_raw(spark, str(path))


def test_feature_values_and_pre_profile_never_reads_balances(spark):
    df = spark.sql("""SELECT 25 AS step, 'TRANSFER' AS type, 100D AS amount, 100D AS oldbalanceOrg,
                    0D AS newbalanceOrig, 20D AS oldbalanceDest, 80D AS newbalanceDest""")
    row = add_features(df).first()
    assert row.errorBalanceOrig == 0
    assert row.errorBalanceDest == 40
    assert row.origEmptied == 1 and row.hour == 1
    assert row.log_amount == pytest.approx(np.log1p(100))
    pre = add_features(df.select("step", "type", "amount"), "pre_transaction")
    assert set(pre.columns) == {"step", "type", "amount", "log_amount", "hour"}
    assert feature_names("pre_transaction") == ["type_CASH_OUT", "log_amount", "hour"]


@pytest.mark.parametrize("method", ["stratified", "temporal"])
def test_splits_disjoint_complete_and_reproducible(transactions, method):
    assigned, metadata = assign_splits(transactions, method)
    rows = assigned.select("_row_id", "_split", "step").collect()
    assert len(rows) == 1000 and len({r._row_id for r in rows}) == 1000
    assert {r._split for r in rows} == {"train", "valid", "test"}
    again, _ = assign_splits(transactions.repartition(3), method)
    assert {r._row_id: r._split for r in rows} == {r._row_id: r._split for r in again.select("_row_id", "_split").collect()}
    if method == "temporal":
        assert max(r.step for r in rows if r._split == "train") < min(r.step for r in rows if r._split == "valid")
        assert max(r.step for r in rows if r._split == "valid") < min(r.step for r in rows if r._split == "test")


def test_weights_use_train_counts_and_reject_missing_class(transactions):
    train = transactions.filter("step <= 14")
    n = train.count()
    counts = {r.isFraud: r['count'] for r in train.groupBy("isFraud").count().collect()}
    weighted = add_class_weight(train)
    weights = {r.isFraud: r.classWeight for r in weighted.select("isFraud", "classWeight").distinct().collect()}
    assert weights[1] == pytest.approx(n / (2 * counts[1]))
    assert weights[0] == pytest.approx(n / (2 * counts[0]))
    with pytest.raises(ValueError):
        add_class_weight(train.filter("isFraud = 0"))


def test_scaler_fits_train_only(transactions):
    train = add_features(transactions.filter("step <= 14"), "pre_transaction")
    model = Pipeline(stages=preprocessing_stages(True, "pre_transaction")).fit(train)
    expected = train.agg(F.avg("log_amount"), F.avg("hour")).first()
    assert model.stages[-1].mean[1] == pytest.approx(expected[0])
    assert model.stages[-1].mean[2] == pytest.approx(expected[1])
    before = model.stages[-1].mean.toArray().copy()
    shifted = train.withColumn("log_amount", F.col("log_amount") + 100)
    model.transform(shifted).count()
    assert np.array_equal(before, model.stages[-1].mean.toArray())


def test_threshold_exact_and_selection_ignores_test():
    assert threshold_from_scores([0, 0, 1, 1], [0.1, 0.7, 0.8, 1.0]) == 0.8
    assert threshold_from_scores([0, 1], [0.0, 1.0]) == 1.0
    for y, p in [([], []), ([0, 0], [0, 0]), ([0, 1], [np.nan, 1])]:
        with pytest.raises(ValueError):
            threshold_from_scores(y, p)
    results = {"A": {"auc_pr": 1, "validation": {"auc_pr": .4, "f1": .9, "precision": 1}},
               "B": {"auc_pr": 0, "validation": {"auc_pr": .5, "f1": .1, "precision": .1}}}
    assert select_best(results) == "B"
    results["A"]["auc_pr"], results["B"]["auc_pr"] = 0, 1
    assert select_best(results) == "B"


def test_curve_boundary_matches_spark_auc(spark):
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from sklearn.metrics import auc
    from src.figures import spark_pr_curve
    df = spark.sql("SELECT * FROM VALUES (0,1D),(1,1D),(0,0.2D),(1,0.1D) AS t(isFraud,prob)")
    evaluator = BinaryClassificationEvaluator(labelCol="isFraud", rawPredictionCol="prob", numBins=0,
                                               metricName="areaUnderPR")
    p, r = spark_pr_curve([0,1,0,1], [1,1,.2,.1])
    assert p[-1] == 0.5
    assert auc(r,p) == pytest.approx(evaluator.evaluate(df), abs=1e-12)


def test_csv_validation_invalid_money_step_type_and_empty():
    pdf = pd.DataFrame({"step": [1, 1.5, 2, 3], "type": [" transfer ", "CASH_OUT", "DEBIT", "UNKNOWN"],
                        "amount": [0, 10, np.inf, -1]})
    good, bad = validate_csv(pdf, "pre_transaction")
    assert len(good) == 1 and good.iloc[0].type == "TRANSFER"
    assert len(bad) == 3 and bad.validation_error.str.len().min() > 0
    for df in (pdf.drop(columns="amount"), pdf.iloc[:0]):
        with pytest.raises(ValueError):
            validate_csv(df, "pre_transaction")


def test_saved_missing_never_calls_fit(spark, transactions, tmp_path, monkeypatch):
    def forbidden_fit(*args, **kwargs):
        pytest.fail("Saved mode must never fit")
    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    with pytest.raises(FileNotFoundError):
        train_all(transactions, transactions, transactions, True, {}, model_dir=str(tmp_path))


def test_legacy_bundle_load_prediction_order_invalid_and_scope(spark, tmp_path):
    info_path = tmp_path / "info.json"
    info_path.write_text(json.dumps({"name": "Random Forest", "threshold": 1.0,
                                     "model_path": str(Path(config.MODEL_DIR) / "random_forest")}), encoding="utf-8")
    model, info = load_bundle(spark, info_path=str(info_path))
    assert info["threshold"] == 1.0 and info["feature_profile"] == "post_transaction"
    row = {"step": 1, "type": "TRANSFER", "amount": 100., "oldbalanceOrg": 100.,
           "newbalanceOrig": 0., "oldbalanceDest": 0., "newbalanceDest": 0.}
    pdf = pd.DataFrame([{**row, "nameOrig": "A", "isFraud": 1},
                        {**row, "nameOrig": "B", "type": "PAYMENT", "isFraud": 0},
                        {**row, "nameOrig": "C", "amount": -1., "isFraud": 1}])
    output = predict_validated(spark, model, info, pdf)
    assert list(output.nameOrig) == ["A", "B", "C"] and list(output.isFraud) == [1, 0, 1]
    assert output.iloc[0].scoring_status == "Đã chấm điểm"
    assert pd.isna(output.iloc[1].fraud_probability) and output.iloc[1].scoring_status == "Ngoài phạm vi mô hình"
    assert output.iloc[2].validation_error
    info_path.write_text(json.dumps({**info, "feature_profile": "pre_transaction"}), encoding="utf-8")
    with pytest.raises(ValueError, match="không khớp"):
        load_bundle(spark, info_path=str(info_path))


def test_four_spark_algorithms_fit_save_and_evaluate(spark, transactions, tmp_path, monkeypatch):
    import src.modeling as modeling
    common = dict(labelCol="isFraud", weightCol="classWeight")
    monkeypatch.setattr(modeling, "build_models", lambda: {
        "Logistic Regression": (LogisticRegression(maxIter=4, **common), True),
        "Decision Tree": (DecisionTreeClassifier(maxDepth=2, **common), False),
        "Random Forest": (RandomForestClassifier(numTrees=2, maxDepth=2, seed=42, **common), False),
        "GBT": (GBTClassifier(maxIter=2, maxDepth=2, seed=42, **common), False)})
    assigned, _ = assign_splits(transactions, "temporal")
    data = add_features(assigned, "pre_transaction")
    train, valid, test, split = split_data(data, "temporal", assigned=True)
    try:
        results, _, fitted = train_all(train, valid, test, False, {}, "pre_transaction", str(tmp_path))
        m = {"results": results, "best_model": select_best(results), "split": split,
             "selection": {"dataset": "validation"}}
        assert validate_result(m)
        assert set(fitted) == set(SLUG)
        assert sum(r["selected_before_test"] for r in results.values()) == 1
        for slug in SLUG.values():
            assert (tmp_path / slug / "metadata").exists()
        damaged = copy.deepcopy(m)
        damaged["results"]["GBT"]["TP"] += 1
        with pytest.raises(ValueError):
            validate_result(damaged)
    finally:
        for df in (train, valid, test):
            df.unpersist()

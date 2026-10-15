"""Kiểm thử tính đúng đắn cho mô-đun so sánh Python thuần vs Spark."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.pure_python_bench import (
    engineer_features, split_data_70_10_20,
    compute_class_weights, tune_threshold_f1, evaluate_predictions,
    build_models
)
from src.benchmark_report import generate_markdown_report


@pytest.fixture
def mock_dataset():
    """Tạo mẫu dữ liệu nhỏ giả lập để kiểm thử logic."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "step": np.random.randint(1, 100, n),
        "type": np.random.choice(["TRANSFER", "CASH_OUT"], n),
        "amount": np.random.uniform(100, 50000, n),
        "oldbalanceOrg": np.random.uniform(0, 100000, n),
        "newbalanceOrig": np.random.uniform(0, 100000, n),
        "oldbalanceDest": np.random.uniform(0, 100000, n),
        "newbalanceDest": np.random.uniform(0, 100000, n),
        "isFraud": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "isFlaggedFraud": np.zeros(n, dtype=int)
    })
    return df


def test_split_data_70_10_20(mock_dataset):
    y = mock_dataset["isFraud"]
    X = engineer_features(mock_dataset)
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_data_70_10_20(X, y)

    total = len(mock_dataset)
    # Tỷ lệ 70% Train, 10% Valid, 20% Test
    assert len(X_tr) == int(total * 0.7)
    assert len(X_val) == int(total * 0.1)
    assert len(X_te) == int(total * 0.2)
    # Cả 3 tập đều có nhãn
    assert y_tr.sum() > 0
    assert y_te.sum() > 0


def test_feature_engineering_columns(mock_dataset):
    feats = engineer_features(mock_dataset)
    assert len(feats.columns) == 12
    assert "type_CASH_OUT" in feats.columns
    assert "hour" in feats.columns
    assert "slog_errorBalanceOrig" in feats.columns
    assert not feats.isna().any().any()


def test_threshold_tuning_and_metrics():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.85, 0.9])
    thr = tune_threshold_f1(y_true, y_prob)
    assert 0.0 <= thr <= 1.0

    met = evaluate_predictions(y_true, y_prob, thr)
    assert "accuracy" in met
    assert "f1" in met
    assert "auc_roc" in met
    assert "auc_pr" in met
    assert met["f1"] > 0.8


def test_generate_markdown_report(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))

    py_res = {
        "results": {
            "Logistic Regression": {
                "train_time_s": 0.5, "predict_time_s": 0.05, "total_time_s": 0.55,
                "threshold": 0.6, "accuracy": 0.95, "precision": 0.8, "recall": 0.75,
                "f1": 0.774, "auc_roc": 0.92, "auc_pr": 0.81, "specificity": 0.98,
                "TP": 15, "FP": 4, "FN": 5, "TN": 176
            },
            "Decision Tree": {
                "train_time_s": 0.3, "predict_time_s": 0.02, "total_time_s": 0.32,
                "threshold": 0.7, "accuracy": 0.98, "precision": 0.9, "recall": 0.9,
                "f1": 0.9, "auc_roc": 0.95, "auc_pr": 0.89, "specificity": 0.99,
                "TP": 18, "FP": 2, "FN": 2, "TN": 178
            }
        },
        "counts": {"train": 140, "valid": 20, "test": 40}
    }
    sp_res = {
        "results": {
            "Logistic Regression": {
                "train_time_s": 2.5, "predict_time_s": 0.5, "total_time_s": 3.0,
                "threshold": 0.6, "accuracy": 0.95, "precision": 0.8, "recall": 0.75,
                "f1": 0.774, "auc_roc": 0.92, "auc_pr": 0.81, "specificity": 0.98,
                "TP": 15, "FP": 4, "FN": 5, "TN": 176
            },
            "Decision Tree": {
                "train_time_s": 2.1, "predict_time_s": 0.4, "total_time_s": 2.5,
                "threshold": 0.7, "accuracy": 0.98, "precision": 0.9, "recall": 0.9,
                "f1": 0.9, "auc_roc": 0.95, "auc_pr": 0.89, "specificity": 0.99,
                "TP": 18, "FP": 2, "FN": 2, "TN": 178
            }
        }
    }

    report_path = generate_markdown_report(py_res, sp_res)
    assert Path(report_path).is_file()
    text = Path(report_path).read_text(encoding="utf-8")
    assert "# BÁO CÁO THỰC NGHIỆM ĐỐI CHỨNG CHUYÊN SÂU" in text
    assert "Logistic Regression" in text
    assert "Decision Tree" in text

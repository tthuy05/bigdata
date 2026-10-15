"""Thực nghiệm đối chứng trên Python thuần (Single-node / Scikit-Learn).
Chạy tuần tự 4 mô hình:
1. Logistic Regression (Baseline)
2. Decision Tree
3. Random Forest
4. Gradient Boosting (GBT)
Đo lường thời gian huấn luyện (s), thời gian dự đoán (s) và các chỉ số độ chính xác.
Tỷ lệ chia dữ liệu: 70% Train, 10% Validation, 20% Test (Stratified, seed=42).
"""
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)

import config
from src import console as c

MONEY_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
FEATURE_NAMES = [
    "type_CASH_OUT",
    "log_amount", "log_oldbalanceOrg", "log_newbalanceOrig", "log_oldbalanceDest", "log_newbalanceDest",
    "slog_errorBalanceOrig", "slog_errorBalanceDest",
    "origZeroBalance", "destZeroBalance", "origEmptied", "hour"
]


def load_clean_pandas(csv_path: str = config.DATA_PATH, sample_fraction: float = 1.0) -> pd.DataFrame:
    """Đọc CSV PaySim bằng Pandas, làm sạch và lọc đúng tập gian lận TRANSFER & CASH_OUT."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    c.info(f"[Python Thuần] Đang nạp dữ liệu từ {csv_path}...")
    t0 = time.perf_counter()
    use_cols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
                "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]
    df = pd.read_csv(csv_path, usecols=use_cols)
    c.info(f"[Python Thuần] Đã đọc {len(df):,} dòng trong {time.perf_counter() - t0:.2f}s")

    # Quy tắc làm sạch đồng nhất với Spark
    df = df.dropna(subset=use_cols)
    for col in MONEY_COLS:
        df = df[df[col] >= 0]
    df = df[df["type"].isin(config.VALID_TYPES)]
    df = df[(df["step"] >= 1) & (df["isFraud"].isin([0, 1])) & (df["isFlaggedFraud"].isin([0, 1]))]
    df = df[(df["nameOrig"].str.strip() != "") & (df["nameDest"].str.strip() != "")]
    df = df.drop_duplicates()

    # Lọc TRANSFER và CASH_OUT
    df = df[df["type"].isin(config.FRAUD_TYPES)].copy()
    c.info(f"[Python Thuần] Số dòng TRANSFER & CASH_OUT sau làm sạch: {len(df):,}")

    if sample_fraction < 1.0:
        c.info(f"[Python Thuần] Lấy mẫu ngẫu nhiên {sample_fraction * 100:.1f}% (seed={config.SEED})...")
        df = df.sample(frac=sample_fraction, random_state=config.SEED).copy()
        c.info(f"[Python Thuần] Kích thước mẫu thực nghiệm: {len(df):,} dòng")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng phái sinh khớp hoàn toàn với src/features.py."""
    data = pd.DataFrame(index=df.index)
    data["type_CASH_OUT"] = (df["type"] == "CASH_OUT").astype(float)
    data["hour"] = (df["step"] % 24).astype(float)

    for c_name in MONEY_COLS:
        data[f"log_{c_name}"] = np.log1p(np.maximum(0, df[c_name].values))

    err_orig = df["newbalanceOrig"].values + df["amount"].values - df["oldbalanceOrg"].values
    err_dest = df["oldbalanceDest"].values + df["amount"].values - df["newbalanceDest"].values
    data["slog_errorBalanceOrig"] = np.sign(err_orig) * np.log1p(np.abs(err_orig))
    data["slog_errorBalanceDest"] = np.sign(err_dest) * np.log1p(np.abs(err_dest))

    data["origZeroBalance"] = ((df["oldbalanceOrg"] == 0) & (df["newbalanceOrig"] == 0)).astype(float)
    data["destZeroBalance"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype(float)
    data["origEmptied"] = ((df["oldbalanceOrg"] > 0) & (np.abs(df["oldbalanceOrg"] - df["amount"]) < 0.01)).astype(float)

    return data[FEATURE_NAMES]


def split_data_70_10_20(X: pd.DataFrame, y: pd.Series):
    """Chia dữ liệu phân tầng Stratified đúng tỷ lệ 70% Train, 10% Valid, 20% Test (seed=42)."""
    # Bước 1: Tách 20% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=config.SEED
    )
    # Bước 2: Tách 10% Validation từ 80% còn lại (10/80 = 0.125)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, stratify=y_train_val, random_state=config.SEED
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_class_weights(y_train: pd.Series) -> dict:
    """Trọng số lớp cân bằng tương tự Spark: w_c = N / (2 * N_c)."""
    n_total = len(y_train)
    n_pos = int(y_train.sum())
    n_neg = n_total - n_pos
    w_pos = n_total / (2.0 * n_pos) if n_pos else 1.0
    w_neg = n_total / (2.0 * n_neg) if n_neg else 1.0
    return {0: w_neg, 1: w_pos}


def tune_threshold_f1(y_val: np.ndarray, y_prob: np.ndarray) -> float:
    """Tìm ngưỡng t* làm cực đại F1 trên tập Validation."""
    prec, rec, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Tính toàn bộ chỉ số đánh giá theo ngưỡng đã cho."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc_val = float(auc(fpr, tpr))

    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    auc_pr_val = float(auc(r_curve, p_curve))

    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(spec),
        "auc_roc": auc_roc_val,
        "auc_pr": auc_pr_val,
        "threshold": float(threshold)
    }


def build_models(class_weights: dict):
    """Khởi tạo 4 mô hình Scikit-Learn với siêu tham số tương ứng PySpark MLlib."""
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=50, C=100.0, penalty="l2", solver="lbfgs",
                class_weight=class_weights, random_state=config.SEED
            ),
            "scale": True,
            "is_baseline": True
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                max_depth=10, min_samples_split=5,
                class_weight=class_weights, random_state=config.SEED
            ),
            "scale": False,
            "is_baseline": False
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=30, max_depth=10, max_features="sqrt",
                class_weight=class_weights, random_state=config.SEED, n_jobs=-1
            ),
            "scale": False,
            "is_baseline": False
        },
        "GBT": {
            # Dùng GradientBoostingClassifier chuẩn mực của Scikit-Learn tương ứng GBTClassifier của Spark
            "model": GradientBoostingClassifier(
                n_estimators=20, max_depth=5, learning_rate=0.1,
                random_state=config.SEED
            ),
            "scale": False,
            "is_baseline": False
        }
    }
    return models


def run_pure_python_benchmark(df_clean: pd.DataFrame = None, sample_fraction: float = 1.0) -> dict:
    """Thực thi đầy đủ phiên đo kiểm Python thuần."""
    c.sub("PHIÊN 1: CHẠY BENCHMARK PYTHON THUẦN (SCIKIT-LEARN)")

    t_prep_start = time.perf_counter()
    if df_clean is None:
        df_clean = load_clean_pandas(sample_fraction=sample_fraction)

    y = df_clean[config.LABEL].astype(int)
    X = engineer_features(df_clean)
    prep_time_s = round(time.perf_counter() - t_prep_start, 3)

    c.info(f"[Python Thuần] Chia dữ liệu 70% Train, 10% Valid, 20% Test...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_70_10_20(X, y)
    c.table(["Tập", "Số giao dịch", "Số gian lận", "Tỷ lệ %"], [
        ["Train (70%)", len(y_train), int(y_train.sum()), round(100 * y_train.sum() / len(y_train), 4)],
        ["Validation (10%)", len(y_val), int(y_val.sum()), round(100 * y_val.sum() / len(y_val), 4)],
        ["Test (20%)", len(y_test), int(y_test.sum()), round(100 * y_test.sum() / len(y_test), 4)],
    ])

    class_weights = compute_class_weights(y_train)
    sw_train = y_train.map(class_weights).values
    c.info(f"[Python Thuần] Trọng số lớp: normal={class_weights[0]:.4f}, fraud={class_weights[1]:.2f}")

    # Chuẩn bị scaler cho Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model_configs = build_models(class_weights)
    results = {}
    curves_data = {}

    for name, cfg in model_configs.items():
        c.sub(f"[Python Thuần] Huấn luyện & Đánh giá: {name}")
        clf = cfg["model"]
        is_scaled = cfg["scale"]

        X_tr = X_train_scaled if is_scaled else X_train.values
        X_va = X_val_scaled if is_scaled else X_val.values
        X_te = X_test_scaled if is_scaled else X_test.values

        # Đo thời gian huấn luyện (Fit)
        t0_train = time.perf_counter()
        if isinstance(clf, GradientBoostingClassifier):
            clf.fit(X_tr, y_train.values, sample_weight=sw_train)
        else:
            clf.fit(X_tr, y_train.values)
        t_train = round(time.perf_counter() - t0_train, 3)
        c.info(f"Thời gian huấn luyện (Train time): {t_train:.3f}s")

        # Dự đoán trên Validation để tối ưu ngưỡng
        val_probs = clf.predict_proba(X_va)[:, 1]
        best_threshold = tune_threshold_f1(y_val.values, val_probs)
        c.info(f"Ngưỡng tối ưu F1 trên Validation: t* = {best_threshold:.4f}")

        # Đo thời gian dự đoán trên Test (Predict)
        t0_pred = time.perf_counter()
        test_probs = clf.predict_proba(X_te)[:, 1]
        t_pred = round(time.perf_counter() - t0_pred, 3)
        c.info(f"Thời gian dự đoán Test (Predict time): {t_pred:.3f}s")

        # Đánh giá chỉ số trên Test
        met = evaluate_predictions(y_test.values, test_probs, best_threshold)
        met["at_default_0_5"] = evaluate_predictions(y_test.values, test_probs, 0.5)
        met["etl_prep_time_s"] = prep_time_s
        met["train_time_s"] = t_train
        met["predict_time_s"] = t_pred
        met["total_time_s"] = round(t_train + t_pred, 3)
        met["total_pipeline_time_s"] = round(prep_time_s + t_train + t_pred, 3)
        met["is_baseline"] = cfg["is_baseline"]
        met["platform"] = "Python thuần (Scikit-Learn)"

        results[name] = met
        curves_data[name] = {
            "labels": y_test.values.tolist(),
            "probs": test_probs.tolist()
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
        "platform": "Python thuần (Scikit-Learn)",
        "prep_time_s": prep_time_s,
        "sample_fraction": sample_fraction,
        "split_ratio": {"train": 0.7, "validation": 0.1, "test": 0.2},
        "counts": {
            "train": len(y_train), "valid": len(y_val), "test": len(y_test),
            "fraud_test": int(y_test.sum())
        },
        "results": results
    }

    out_dir = Path(config.OUTPUT_DIR) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "python_results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    with open(out_dir / "python_curves.json", "w", encoding="utf-8") as f:
        json.dump(curves_data, f)

    c.ok(f"[Python Thuần] Hoàn tất và lưu kết quả vào {out_dir / 'python_results.json'}")
    return output_data

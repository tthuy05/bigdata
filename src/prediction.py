"""Shared, validated prediction contract for Streamlit and QA."""
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.features import RAW_INPUTS, PROFILES, add_features


def required_columns(profile):
    return RAW_INPUTS[profile]


def validate_csv(pdf, profile):
    """Return valid rows and per-row errors; never silently drop bad rows."""
    missing = [k for k in required_columns(profile) if k not in pdf.columns]
    if missing:
        raise ValueError("CSV thiếu cột bắt buộc: " + ", ".join(missing))
    if pdf.empty:
        raise ValueError("CSV không có giao dịch.")
    reserved = {"_row_id", "fraud_probability", "prediction", "validation_error", "scoring_status"}
    if reserved.intersection(pdf.columns):
        raise ValueError("CSV chứa tên cột dành cho kết quả: " + ", ".join(sorted(reserved.intersection(pdf.columns))))
    out = pdf.copy().reset_index(drop=True)
    out["_row_id"] = np.arange(len(out), dtype=np.int64)
    errors = pd.Series("", index=out.index, dtype=object)

    def flag(mask, msg):
        errors.loc[mask] = errors.loc[mask].map(lambda old: old + ("; " if old else "") + msg)

    numeric = [k for k in required_columns(profile) if k != "type"]
    for k in numeric:
        value = pd.to_numeric(out[k], errors="coerce")
        invalid = ~np.isfinite(value.to_numpy(dtype=float)) | (value < (1 if k == "step" else 0))
        if k == "step":
            invalid |= (value % 1 != 0) | (value > 2_147_483_647)
        flag(invalid, f"{k}: phải là số hữu hạn {'nguyên >= 1' if k == 'step' else '>= 0'}")
        out[k] = value
    out["type"] = out["type"].astype("string").str.strip().str.upper()
    flag(~out["type"].isin(config.VALID_TYPES), "Loại giao dịch không hợp lệ")
    out["validation_error"] = errors
    return out.loc[errors == ""].copy(), out.loc[errors != ""].copy()


def load_bundle(spark, model_path=None, info_path=None):
    from src.model_io import load_pipeline
    from pyspark.ml.feature import VectorAssembler
    info_path = Path(info_path or Path(config.MODEL_DIR) / "best_model_info.json")
    if not info_path.is_file():
        raise FileNotFoundError(f"Thiếu thông tin mô hình: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    profile = info.get("feature_profile", "post_transaction")
    if profile not in PROFILES:
        raise ValueError("Bộ đặc trưng trong metadata không được hỗ trợ.")
    threshold = float(info["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Ngưỡng đã lưu phải thuộc [0, 1].")
    selected = Path(model_path or info.get("model_path") or config.BEST_MODEL_PATH)
    if not selected.is_absolute():
        selected = Path(config.BASE_DIR) / selected
    if not (selected / "metadata").is_dir():
        raise FileNotFoundError(f"Thiếu PipelineModel: {selected}")
    model = load_pipeline(str(selected), spark=spark)
    expected_class = {"Logistic Regression": "LogisticRegressionModel",
                      "Decision Tree": "DecisionTreeClassificationModel",
                      "Random Forest": "RandomForestClassificationModel",
                      "GBT": "GBTClassificationModel"}.get(info.get("name"))
    if expected_class is None or type(model.stages[-1]).__name__ != expected_class:
        raise ValueError("Thuật toán trong PipelineModel không khớp tên mô hình đã lưu.")
    assembler = next((s for s in model.stages if isinstance(s, VectorAssembler)), None)
    if assembler is None or assembler.getInputCols() != ["type_vec"] + PROFILES[profile]:
        raise ValueError("PipelineModel không khớp bộ đặc trưng đã lưu; cần kiểm tra bundle.")
    if model.stages[-1].numFeatures != len(PROFILES[profile]) + 1:
        raise ValueError("Kích thước đặc trưng mô hình không khớp profile.")
    return model, {**info, "feature_profile": profile, "model_path": str(selected),
                   "threshold": threshold}


def predict_validated(spark, model, info, pdf, threshold=None):
    """Keep upload order/labels. Unsupported transaction types remain unscored."""
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType
    profile = info["feature_profile"]
    threshold = info["threshold"] if threshold is None else float(threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Ngưỡng quyết định phải thuộc [0, 1].")
    valid, invalid = validate_csv(pdf, profile)
    scored = valid[valid.type.isin(config.FRAUD_TYPES)]
    scores = {}
    if not scored.empty:
        cols = ["_row_id"] + required_columns(profile)
        schema = StructType([StructField(k, LongType() if k in ("_row_id", "step") else
                                         StringType() if k == "type" else DoubleType(), False) for k in cols])
        # JSON is read by the JVM; no Python worker is required for user input.
        with tempfile.TemporaryDirectory(prefix="fraud_predict_") as temporary:
            path = Path(temporary) / "transactions.json"
            with path.open("w", encoding="utf-8") as handle:
                for row in scored.to_dict("records"):
                    record = {k: int(row[k]) if k in ("_row_id", "step") else str(row[k]) if k == "type"
                              else float(row[k]) for k in cols}
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
            sdf = spark.read.schema(schema).json(str(path))
            pred = (model.transform(add_features(sdf, profile))
                    .select("_row_id", vector_to_array("probability")[1].alias("prob")))
            scores = {r["_row_id"]: r["prob"] for r in pred.collect()}
        if len(scores) != len(scored):
            raise RuntimeError("Mô hình không trả về đủ số giao dịch đầu vào.")
    output = pd.concat([valid, invalid]).sort_values("_row_id").reset_index(drop=True)
    output["fraud_probability"] = output["_row_id"].map(scores)
    output["scoring_status"] = np.where(output.validation_error != "", "Dữ liệu không hợp lệ",
                                        np.where(output.type.isin(config.FRAUD_TYPES), "Đã chấm điểm",
                                                 "Ngoài phạm vi mô hình"))
    output["prediction"] = np.where(output.fraud_probability.isna(), "Chưa đánh giá",
                                     np.where(output.fraud_probability >= threshold,
                                              "Nghi ngờ gian lận", "Bình thường"))
    output["decision_threshold"] = threshold
    return output.drop(columns="_row_id")

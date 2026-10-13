"""Tạo đặc trưng (feature engineering) và các giai đoạn tiền xử lý của Pipeline."""
from pyspark.sql import DataFrame, functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

import config

MONEY_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
ENGINEERED_COLS = ["errorBalanceOrig", "errorBalanceDest", "origZeroBalance",
                   "destZeroBalance", "origEmptied", "hour"]
NUMERIC_FEATURES = [f"log_{c}" for c in MONEY_COLS] + \
                   ["slog_errorBalanceOrig", "slog_errorBalanceDest",
                    "origZeroBalance", "destZeroBalance", "origEmptied", "hour"]
FEATURE_NAMES = ["type_CASH_OUT"] + NUMERIC_FEATURES
PRE_NUMERIC_FEATURES = ["log_amount", "hour"]
PROFILES = {"post_transaction": NUMERIC_FEATURES, "pre_transaction": PRE_NUMERIC_FEATURES}
RAW_INPUTS = {
    "post_transaction": ["step", "type"] + MONEY_COLS,
    # Conservative PaySim experiment: no recorded balances or descendants.
    "pre_transaction": ["step", "type", "amount"],
}


def feature_names(profile="post_transaction"):
    return ["type_CASH_OUT"] + PROFILES[profile]


def _slog(c):
    """Logarit có dấu: sign(x) * ln(1 + |x|) -> nén thang đo nhưng giữ dấu."""
    return F.signum(F.col(c)) * F.log1p(F.abs(F.col(c)))


def add_features(df: DataFrame, profile="post_transaction") -> DataFrame:
    """Áp dụng cho cả dữ liệu huấn luyện lẫn giao dịch mới trong ứng dụng."""
    if profile not in PROFILES:
        raise ValueError(f"Bộ đặc trưng không hợp lệ: {profile}")
    if profile == "pre_transaction":
        return (df.withColumn("log_amount", F.log1p(F.col("amount")))
                .withColumn("hour", (F.col("step") % 24).cast("double")))
    df = (df
          .withColumn("errorBalanceOrig",
                      F.col("newbalanceOrig") + F.col("amount") - F.col("oldbalanceOrg"))
          .withColumn("errorBalanceDest",
                      F.col("oldbalanceDest") + F.col("amount") - F.col("newbalanceDest"))
          .withColumn("origZeroBalance",
                      ((F.col("oldbalanceOrg") == 0) & (F.col("newbalanceOrig") == 0)).cast("double"))
          .withColumn("destZeroBalance",
                      ((F.col("oldbalanceDest") == 0) & (F.col("newbalanceDest") == 0)).cast("double"))
          .withColumn("origEmptied",
                      ((F.col("oldbalanceOrg") > 0) &
                       (F.abs(F.col("oldbalanceOrg") - F.col("amount")) < 0.01)).cast("double"))
          .withColumn("hour", (F.col("step") % 24).cast("double")))
    for c in MONEY_COLS:
        df = df.withColumn(f"log_{c}", F.log1p(F.col(c)))
    df = (df.withColumn("slog_errorBalanceOrig", _slog("errorBalanceOrig"))
            .withColumn("slog_errorBalanceDest", _slog("errorBalanceDest")))
    return df


def filter_relevant(df: DataFrame) -> DataFrame:
    """Chỉ giữ TRANSFER và CASH_OUT – hai loại duy nhất có gian lận (kết quả EDA)."""
    return df.filter(F.col("type").isin(*config.FRAUD_TYPES))


def add_class_weight(df: DataFrame) -> DataFrame:
    """Trọng số lớp cân bằng: w_c = N / (2 * N_c) để xử lý mất cân bằng dữ liệu."""
    counts = {r[config.LABEL]: r["count"] for r in df.groupBy(config.LABEL).count().collect()}
    n = sum(counts.values())
    if set(counts) != {0, 1} or not all(counts.values()):
        raise ValueError("Train phải có đủ hai lớp 0 và 1 để tính trọng số lớp.")
    w_pos, w_neg = n / (2.0 * counts[1]), n / (2.0 * counts[0])
    return df.withColumn("classWeight",
                         F.when(F.col(config.LABEL) == 1, F.lit(w_pos)).otherwise(F.lit(w_neg)))


def preprocessing_stages(scale: bool, profile="post_transaction"):
    """StringIndexer -> OneHotEncoder -> VectorAssembler (-> StandardScaler cho LR)."""
    indexer = StringIndexer(inputCol="type", outputCol="type_idx",
                            stringOrderType="alphabetAsc", handleInvalid="error")
    # alphabetAsc: CASH_OUT=0, TRANSFER=1; dropLast=True -> 1 cột nhị phân (type_CASH_OUT)
    encoder = OneHotEncoder(inputCols=["type_idx"], outputCols=["type_vec"], dropLast=True)
    assembler = VectorAssembler(inputCols=["type_vec"] + PROFILES[profile],
                                outputCol="raw_features" if scale else "features",
                                handleInvalid="error")
    stages = [indexer, encoder, assembler]
    if scale:
        stages.append(StandardScaler(inputCol="raw_features", outputCol="features",
                                     withMean=True, withStd=True))
    return stages

"""Khởi tạo SparkSession, đọc dữ liệu PaySim và làm sạch."""
import csv
import os
import sys
from functools import reduce
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import (StructType, StructField, IntegerType, StringType, DoubleType)

import config

# Khai báo lược đồ tường minh: tránh inferSchema (phải quét dữ liệu 2 lần) và sai kiểu
PAYSIM_SCHEMA = StructType([
    StructField("step", IntegerType(), False),
    StructField("type", StringType(), False),
    StructField("amount", DoubleType(), False),
    StructField("nameOrig", StringType(), False),
    StructField("oldbalanceOrg", DoubleType(), False),
    StructField("newbalanceOrig", DoubleType(), False),
    StructField("nameDest", StringType(), False),
    StructField("oldbalanceDest", DoubleType(), False),
    StructField("newbalanceDest", DoubleType(), False),
    StructField("isFraud", IntegerType(), False),
    StructField("isFlaggedFraud", IntegerType(), False),
])


def get_spark(app_name: str = config.APP_NAME) -> SparkSession:
    # Pin workers to this interpreter, avoiding Windows' Microsoft Store alias.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    local_hadoop = os.path.join(config.BASE_DIR, ".runtime", "hadoop")
    if os.name == "nt" and not os.environ.get("HADOOP_HOME") and os.path.isfile(
            os.path.join(local_hadoop, "bin", "winutils.exe")):
        os.environ["HADOOP_HOME"] = local_hadoop
        os.environ["PATH"] = os.path.join(local_hadoop, "bin") + os.pathsep + os.environ["PATH"]
    spark = (SparkSession.builder.appName(app_name)
             .master(config.SPARK_MASTER)
             .config("spark.driver.memory", config.DRIVER_MEMORY)
             .config("spark.sql.shuffle.partitions", config.SHUFFLE_PARTITIONS)
             .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.sql.csv.parser.columnPruning.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_raw(spark: SparkSession, path: str = config.DATA_PATH) -> DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy CSV: {path}")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != PAYSIM_SCHEMA.fieldNames():
        raise ValueError("CSV PaySim phải có đúng 11 cột theo thứ tự lược đồ; không tự đoán vị trí cột.")
    return (spark.read.option("header", True).schema(PAYSIM_SCHEMA)
            .option("mode", "FAILFAST").option("enforceSchema", "false").csv(path))


def quality_report(df: DataFrame) -> dict:
    """Kiểm tra chất lượng: số dòng, giá trị thiếu, trùng lặp, số âm."""
    total = df.count()
    nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]) \
              .first().asDict()
    dup = total - df.dropDuplicates().count()
    money = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    negatives = df.select([F.sum((F.col(c) < 0).cast("int")).alias(c) for c in money]) \
                  .first().asDict()
    zero_amount = df.filter(F.col("amount") <= 0).count()
    labels = {str(r[config.LABEL]): r["count"] for r in df.groupBy(config.LABEL).count().collect()}
    zero_labels = {str(r[config.LABEL]): r["count"] for r in
                   df.filter(F.col("amount") == 0).groupBy(config.LABEL).count().collect()}
    return {"rows": total, "columns": len(df.columns), "nulls": nulls,
            "duplicates": dup, "negatives": negatives, "zero_amount": zero_amount,
            "raw_label_counts": labels, "zero_amount_label_counts": zero_labels,
            "raw_fraud_rate": labels.get("1", 0) / max(total, 1)}


def cleaning_rules(df):
    """Disjoint reasons applied in this order; zero amounts are retained."""
    required = PAYSIM_SCHEMA.fieldNames()
    missing = reduce(lambda a, b: a | b, [F.col(k).isNull() for k in required])
    money = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    bad_money = reduce(lambda a, b: a | b, [
        F.isnan(k) | (F.abs(F.col(k)) == float("inf")) | (F.col(k) < 0) for k in money])
    return [("missing_values", missing), ("invalid_money", bad_money),
            ("invalid_type", ~F.col("type").isin(*config.VALID_TYPES)),
            ("invalid_step_or_label", (F.col("step") < 1) | ~F.col(config.LABEL).isin(0, 1)
             | ~F.col("isFlaggedFraud").isin(0, 1)),
            ("empty_account", (F.trim(F.col("nameOrig")) == "") | (F.trim(F.col("nameDest")) == ""))]


def clean(df: DataFrame, return_audit=False):
    audit = []
    for reason, bad in cleaning_rules(df):
        if return_audit:
            removed = {str(r[config.LABEL]): r["count"] for r in
                       df.filter(bad).groupBy(config.LABEL).count().collect()}
            audit.append({"reason": reason, "removed": sum(removed.values()), "labels": removed})
        df = df.filter(~bad)
    if return_audit:
        before = {str(r[config.LABEL]): r["count"] for r in df.groupBy(config.LABEL).count().collect()}
    deduped = df.dropDuplicates()
    if return_audit:
        after = {str(r[config.LABEL]): r["count"] for r in deduped.groupBy(config.LABEL).count().collect()}
        labels = {k: before.get(k, 0) - after.get(k, 0) for k in before}
        audit.append({"reason": "duplicate_rows", "removed": sum(labels.values()), "labels": labels})
        return deduped, {"removal_groups": audit, "clean_label_counts": after,
                         "clean_rows": sum(after.values()),
                         "clean_fraud_rate": after.get("1", 0) / max(sum(after.values()), 1),
                         "zero_amount_policy": "retained; not a reason to discard fraud"}
    return deduped

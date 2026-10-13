"""Spark JVM PipelineModel persistence (pinned Spark 3.5.3, Java-only stages).

Python's generic Pipeline reader/writer starts RDD Python workers for metadata.
On Windows/Python 3.12 these workers can crash. The equivalent JVM persistence
keeps the standard Spark PipelineModel format and never starts such a worker.
"""
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


def load_pipeline(path, spark=None):
    if not (Path(path) / "metadata").is_dir():
        raise FileNotFoundError(f"Thiếu PipelineModel: {path}")
    # Streamlit reruns in a new thread, where the cached session may not be active.
    spark = spark or SparkSession.getActiveSession() or SparkSession._instantiatedSession
    if spark is None:
        raise RuntimeError("Cần SparkSession trước khi nạp mô hình.")
    if spark.sparkContext._jsc is None or spark.sparkContext._jsc.sc().isStopped():
        raise RuntimeError("SparkSession đã dừng; cần khởi động lại ứng dụng.")
    java_model = spark._jvm.org.apache.spark.ml.PipelineModel.load(str(Path(path).resolve()))
    return PipelineModel._from_java(java_model)


def save_pipeline(model, path, overwrite=False):
    writer = model._to_java().write()
    if overwrite:
        writer = writer.overwrite()
    writer.save(str(Path(path).resolve()))

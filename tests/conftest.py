import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SPARK_MASTER", "local[2]")
os.environ.setdefault("SPARK_DRIVER_MEMORY", "1g")


@pytest.fixture(scope="session")
def spark():
    from src.data_loader import get_spark
    session = get_spark("FraudDetection-Tests")
    yield session
    session.stop()


@pytest.fixture
def transactions(spark):
    return spark.sql("""SELECT CAST(id % 20 + 1 AS INT) AS step,
        CASE WHEN id % 2 = 0 THEN 'TRANSFER' ELSE 'CASH_OUT' END AS type,
        CAST(id + 1 AS DOUBLE) AS amount, concat('C', id) AS nameOrig,
        CAST(id + 100 AS DOUBLE) AS oldbalanceOrg, 99D AS newbalanceOrig,
        concat('D', id) AS nameDest, 50D AS oldbalanceDest,
        CAST(id + 51 AS DOUBLE) AS newbalanceDest,
        CAST(CASE WHEN id % 11 = 0 THEN 1 ELSE 0 END AS INT) AS isFraud,
        0 AS isFlaggedFraud FROM range(1000)""")

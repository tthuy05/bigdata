"""Thống kê khám phá dữ liệu bằng Spark SQL – CHỈ tính và in số liệu, chưa vẽ hình.
Các hình ở bước sau được vẽ từ đúng các bảng này (lưu trong outputs/metrics/eda_stats.json)."""
from pyspark.sql import DataFrame, SparkSession

import config
from src import console as c
from src.features import add_features, filter_relevant


def _run(spark, title, sql, n=30, note=""):
    c.sub(title)
    c.show_sql(sql)
    res = spark.sql(sql)
    res.show(n, truncate=False)
    if note:
        c.info(note)
    return [r.asDict() for r in res.collect()]


def compute_stats(spark: SparkSession, df: DataFrame):
    """df: dữ liệu đã làm sạch. Trả về (stats, sample_pdf)."""
    df.createOrReplaceTempView("transactions")
    rel = add_features(filter_relevant(df))
    rel.createOrReplaceTempView("rel")
    s = {}

    s["label"] = _run(spark, "1) Phân bố nhãn isFraud  → Hình fig_label_distribution", """
        SELECT isFraud, COUNT(*) AS n,
               ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS pct
        FROM transactions GROUP BY isFraud ORDER BY isFraud""")
    lab = {r["isFraud"]: r["n"] for r in s["label"]}
    c.info(f"=> Cứ 1 giao dịch gian lận có khoảng {lab.get(0, 0) // max(lab.get(1, 1), 1):,} "
           "giao dịch bình thường (mất cân bằng nặng)".replace(",", "."))

    s["by_type"] = _run(spark, "2) Thống kê theo loại giao dịch  → Bảng 7, Hình fig_type_distribution", """
        SELECT type, COUNT(*) AS n, SUM(isFraud) AS fraud,
               ROUND(100 * AVG(isFraud), 4) AS fraud_rate_pct,
               ROUND(SUM(amount) / 1e9, 2) AS total_amount_billion,
               ROUND(AVG(amount), 2) AS avg_amount
        FROM transactions GROUP BY type ORDER BY n DESC""")
    c.info("=> Gian lận CHỈ xuất hiện ở: " + ", ".join(r["type"] for r in s["by_type"] if r["fraud"] > 0))

    s["amount_by_label"] = _run(spark, "3) Số tiền theo nhãn  → Bảng 8, Hình fig_amount_distribution", """
        SELECT isFraud, ROUND(AVG(amount), 2) AS mean,
               ROUND(percentile_approx(amount, 0.5), 2) AS median,
               ROUND(MIN(amount), 2) AS min, ROUND(MAX(amount), 2) AS max,
               ROUND(STDDEV(amount), 2) AS std
        FROM transactions GROUP BY isFraud ORDER BY isFraud""")
    s["max_fraud_amount"] = _run(spark, "   Số gian lận đúng hạn mức 10 triệu", """
        SELECT COUNT(*) AS n_10M FROM transactions WHERE isFraud = 1 AND amount = 10000000""")

    s["by_hour"] = _run(spark, "4) Theo giờ trong ngày  → Hình fig_fraud_by_hour", """
        SELECT step % 24 AS hour, COUNT(*) AS n, SUM(isFraud) AS fraud,
               ROUND(100 * AVG(isFraud), 4) AS fraud_rate_pct
        FROM transactions GROUP BY step % 24 ORDER BY hour""", n=24)
    top = sorted(s["by_hour"], key=lambda r: -r["fraud_rate_pct"])[:3]
    c.info("=> Tỷ lệ gian lận cao nhất: " + ", ".join(f"{r['hour']}h ({r['fraud_rate_pct']}%)" for r in top))

    s["balance_patterns"] = _run(spark, "5) Mẫu số dư (TRANSFER, CASH_OUT)  → Bảng 9, Hình fig_balance_patterns", """
        SELECT isFraud, COUNT(*) AS n,
          ROUND(100*AVG(origZeroBalance),2) AS pct_orig_zero,
          ROUND(100*AVG(destZeroBalance),2) AS pct_dest_zero,
          ROUND(100*AVG(origEmptied),2) AS pct_orig_emptied,
          ROUND(100*AVG(CASE WHEN abs(errorBalanceOrig) > 0.01 THEN 1 ELSE 0 END),2) AS pct_err_orig,
          ROUND(100*AVG(CASE WHEN abs(errorBalanceDest) > 0.01 THEN 1 ELSE 0 END),2) AS pct_err_dest
        FROM rel GROUP BY isFraud ORDER BY isFraud""")

    s["flagged"] = _run(spark, "6) Luật có sẵn isFlaggedFraud", """
        SELECT isFraud, isFlaggedFraud, COUNT(*) AS n FROM transactions
        GROUP BY isFraud, isFlaggedFraud ORDER BY isFraud, isFlaggedFraud""")
    c.info(f"=> Luật chỉ gắn cờ {sum(r['n'] for r in s['flagged'] if r['isFlaggedFraud'] == 1)} giao dịch")

    u = _run(spark, "7) Thông tin chung", """
        SELECT COUNT(DISTINCT nameOrig) AS orig, COUNT(DISTINCT nameDest) AS dest,
               MIN(step) AS min_step, MAX(step) AS max_step FROM transactions""")[0]
    s["unique_accounts"] = [{"orig": u["orig"], "dest": u["dest"]}]
    s["n_steps"] = [{"min_step": u["min_step"], "max_step": u["max_step"]}]
    s["describe"] = {r["summary"]: {k: v for k, v in r.asDict().items() if k != "summary"}
                     for r in df.select("step", "amount", "oldbalanceOrg", "newbalanceOrig",
                                        "oldbalanceDest", "newbalanceDest").describe().collect()}
    s["relevant_rows"] = rel.count()
    c.kv("Số giao dịch TRANSFER + CASH_OUT", s["relevant_rows"])

    c.sub("8) Lấy mẫu cho histogram và ma trận tương quan  → Hình fig_amount_distribution, fig_correlation")
    c.info("Mẫu = TOÀN BỘ giao dịch gian lận + 1% giao dịch bình thường (TRANSFER, CASH_OUT), seed = 42")
    sample = (rel.sampleBy("isFraud", fractions={0: 0.01, 1: 1.0}, seed=config.SEED)
                 .select("amount", "isFraud", "hour", "errorBalanceOrig", "errorBalanceDest",
                         "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                         "origZeroBalance", "destZeroBalance", "origEmptied").toPandas())
    c.kv("Kích thước mẫu", len(sample))
    cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
            "errorBalanceOrig", "errorBalanceDest", "origEmptied", "destZeroBalance", "isFraud"]
    corr = sample[cols].corr()["isFraud"].drop("isFraud")
    c.info("Tương quan Pearson của từng biến với nhãn isFraud (trên mẫu):")
    c.table(["Biến", "corr(isFraud)"],
            [[k, float(v)] for k, v in corr.sort_values(key=abs, ascending=False).items()])
    s["corr_with_label"] = {k: float(v) for k, v in corr.items()}
    return s, sample

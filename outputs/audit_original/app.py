"""
Ứng dụng demo phát hiện giao dịch gian lận (Streamlit + Spark MLlib).
Chạy:  streamlit run app/app.py
Yêu cầu: đã chạy main.py để có models/best_model và outputs/.
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

st.set_page_config(page_title="Phát hiện gian lận giao dịch – Spark MLlib", page_icon="🛡️",
                   layout="wide")

COLS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest"]


@st.cache_resource(show_spinner="Đang khởi động Spark và nạp mô hình...")
def load_model():
    from pyspark.ml import PipelineModel
    from src.data_loader import get_spark
    spark = get_spark("FraudDetection-App")
    model = PipelineModel.load(config.BEST_MODEL_PATH)
    info_path = os.path.join(config.MODEL_DIR, "best_model_info.json")
    info = json.load(open(info_path, encoding="utf-8")) if os.path.exists(info_path) else {}
    return spark, model, info


def predict(pdf: pd.DataFrame, threshold: float) -> pd.DataFrame:
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    from src.features import add_features
    spark, model, _ = load_model()
    pdf = pdf.copy()
    for c in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        pdf[c] = pdf[c].astype(float)
    pdf["step"] = pdf["step"].astype(int)
    sdf = spark.createDataFrame(pdf)
    relevant = sdf.filter(F.col("type").isin(*config.FRAUD_TYPES))
    out = (model.transform(add_features(relevant))
           .withColumn("fraud_probability", vector_to_array("probability")[1])
           .select(*COLS, "fraud_probability").toPandas())
    out["prediction"] = (out["fraud_probability"] >= threshold).map(
        {True: "GIAN LẬN", False: "Bình thường"})
    others = pdf[~pdf["type"].isin(config.FRAUD_TYPES)][COLS].copy()
    if len(others):  # PAYMENT, CASH_IN, DEBIT không có gian lận trong dữ liệu huấn luyện
        others["fraud_probability"] = 0.0
        others["prediction"] = "Bình thường (loại không rủi ro)"
    return pd.concat([out, others], ignore_index=True)


# ------------------------------------------------------------------ giao diện
st.title("🛡️ Hệ thống phát hiện gian lận giao dịch tài chính")
st.caption("Apache Spark MLlib · Dữ liệu PaySim · Nhóm 8 – Nhập môn Big Data")

if not os.path.exists(config.BEST_MODEL_PATH):
    st.error("Chưa có mô hình. Hãy chạy `python main.py` trước.")
    st.stop()

with st.sidebar:
    st.header("Cấu hình")
    _, _, info = load_model()
    default_thr = round(float(info.get("threshold", config.THRESHOLD)), 2)
    threshold = st.slider("Ngưỡng cảnh báo (xác suất gian lận)", 0.01, 0.99,
                          min(max(default_thr, 0.01), 0.99), 0.01,
                          help="Mặc định là ngưỡng tối ưu F1 trên tập kiểm định")
    if info:
        st.markdown(f"**Mô hình:** {info['name']}")
        m = info["metrics"]
        st.markdown(f"Precision: `{m['precision']:.4f}`  \nRecall: `{m['recall']:.4f}`  \n"
                    f"F1-score: `{m['f1']:.4f}`  \nAUC-PR: `{m['auc_pr']:.4f}`")

tab1, tab2, tab3 = st.tabs(["🔍 Kiểm tra một giao dịch", "📂 Kiểm tra theo lô (CSV)",
                            "📊 Thống kê & đánh giá"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        ttype = st.selectbox("Loại giao dịch", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Số tiền (amount)", min_value=0.0, value=181000.0, step=1000.0)
        step = st.number_input("Bước thời gian (step, giờ)", min_value=1, max_value=744, value=1)
    with c2:
        old_o = st.number_input("Số dư nguồn trước (oldbalanceOrg)", min_value=0.0, value=181000.0)
        new_o = st.number_input("Số dư nguồn sau (newbalanceOrig)", min_value=0.0, value=0.0)
        name_o = st.text_input("Tài khoản nguồn", "C1305486145")
    with c3:
        old_d = st.number_input("Số dư đích trước (oldbalanceDest)", min_value=0.0, value=0.0)
        new_d = st.number_input("Số dư đích sau (newbalanceDest)", min_value=0.0, value=0.0)
        name_d = st.text_input("Tài khoản đích", "C553264065")
    if st.button("Phân tích giao dịch", type="primary"):
        row = pd.DataFrame([dict(step=step, type=ttype, amount=amount, nameOrig=name_o,
                                 oldbalanceOrg=old_o, newbalanceOrig=new_o, nameDest=name_d,
                                 oldbalanceDest=old_d, newbalanceDest=new_d)])
        res = predict(row, threshold).iloc[0]
        p = float(res["fraud_probability"])
        k1, k2, k3 = st.columns(3)
        k1.metric("Xác suất gian lận", f"{p:.2%}")
        k2.metric("Sai lệch số dư nguồn", f"{new_o + amount - old_o:,.2f}")
        k3.metric("Sai lệch số dư đích", f"{old_d + amount - new_d:,.2f}")
        if p >= threshold:
            st.error(f"⚠️ CẢNH BÁO: giao dịch có dấu hiệu GIAN LẬN (ngưỡng {threshold:.2f})")
        else:
            st.success("✅ Giao dịch được đánh giá là bình thường")

with tab2:
    st.write("Tải lên tệp CSV có các cột: " + ", ".join(f"`{c}`" for c in COLS))
    up = st.file_uploader("Chọn tệp CSV", type=["csv"])
    if up is not None:
        pdf = pd.read_csv(up)
        missing = [c for c in COLS if c not in pdf.columns]
        if missing:
            st.error(f"Thiếu cột: {missing}")
        else:
            with st.spinner(f"Spark đang chấm điểm {len(pdf):,} giao dịch..."):
                res = predict(pdf[COLS], threshold)
            n_alert = int((res["prediction"] == "GIAN LẬN").sum())
            a, b, c = st.columns(3)
            a.metric("Tổng giao dịch", f"{len(res):,}")
            b.metric("Số cảnh báo gian lận", f"{n_alert:,}")
            c.metric("Tỷ lệ cảnh báo", f"{n_alert / max(len(res), 1):.3%}")
            if "isFraud" in pdf.columns:
                st.info("Tệp có cột isFraud – có thể đối chiếu kết quả với nhãn thực tế.")
            st.dataframe(res.sort_values("fraud_probability", ascending=False),
                         width="stretch", height=380)
            st.download_button("Tải kết quả (CSV)", res.to_csv(index=False).encode("utf-8"),
                               "ket_qua_du_doan.csv", "text/csv")

with tab3:
    mpath = os.path.join(config.METRIC_DIR, "model_results.json")
    if os.path.exists(mpath):
        mr = json.load(open(mpath, encoding="utf-8"))
        table = pd.DataFrame(mr["results"]).T[["precision", "recall", "f1", "auc_roc", "auc_pr",
                                               "TP", "FP", "FN", "train_time_s"]]
        st.subheader("So sánh mô hình trên tập kiểm thử")
        st.dataframe(table.style.format(precision=4), width="stretch")
    figs = ["fig_type_distribution.png", "fig_balance_patterns.png", "fig_confusion_matrices.png",
            "fig_pr_curves.png", "fig_feature_importance.png", "fig_fraud_by_hour.png"]
    cols = st.columns(3)
    for i, f in enumerate(figs):
        p = os.path.join(config.FIG_DIR, f)
        if os.path.exists(p):
            cols[i % 3].image(p, width="stretch")

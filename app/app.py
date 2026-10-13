"""Three-tab Vietnamese demo with an explicit pre/post transaction contract."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.prediction import load_bundle, predict_validated, required_columns

st.set_page_config(page_title="Phát hiện gian lận giao dịch – Spark MLlib", page_icon="🛡️", layout="wide")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Đang khởi động Spark và nạp mô hình...")
def load_model(info_path, metadata_signature):
    from src.data_loader import get_spark
    spark = get_spark("FraudDetection-App")
    model, info = load_bundle(spark, info_path=info_path)
    return spark, model, info


st.title("🛡️ Hệ thống phát hiện gian lận giao dịch tài chính")
st.caption("Apache Spark MLlib · Dữ liệu mô phỏng PaySim · Nhóm 8 – Nhập môn Big Data")
try:
    manifest = read_json(config.EXPERIMENT_MANIFEST) if Path(config.EXPERIMENT_MANIFEST).is_file() else None
    if manifest is not None and (manifest.get("status") != "VERIFIED" or
                                 manifest.get("primary_experiment") not in manifest.get("experiments", {})):
        raise ValueError("Manifest chưa hoàn tất hoặc thiếu thí nghiệm chính.")
except (OSError, ValueError) as exc:
    st.error("Không thể đọc cấu hình thí nghiệm đã lưu: " + str(exc))
    st.stop()
info_path = str(Path(config.MODEL_DIR) / "best_model_info.json")
selected = None
legacy = False
with st.sidebar:
    st.header("Cấu hình")
    if manifest:
        keys = list(manifest["experiments"])
        legacy_info = Path(config.OUTPUT_DIR) / "audit_original" / "best_model_info.json"
        if legacy_info.is_file():
            keys.append("legacy")
        selected = st.selectbox("Thí nghiệm", keys, index=keys.index(manifest["primary_experiment"]),
                                format_func=lambda k: "Lịch sử · Sau giao dịch (chưa tái kiểm chứng)" if k == "legacy" else k.replace("A_", "A · Sau giao dịch · ")
                                .replace("B_", "B · Trước giao dịch · ")
                                .replace("temporal", "Theo thời gian").replace("stratified", "Phân tầng"))
        legacy = selected == "legacy"
        info_path = str(legacy_info if legacy else Path(config.BASE_DIR) / manifest["experiments"][selected]["bundle_path"])
    try:
        signature = Path(info_path).read_bytes()
        spark, model, info = load_model(info_path, signature)
    except Exception as exc:
        st.error("Không thể nạp mô hình. Kiểm tra Java, PySpark và bundle đã lưu.")
        st.caption(str(exc))
        st.stop()
    profile = info["feature_profile"]
    st.markdown(f"**Mô hình:** {info['name']}")
    st.markdown("**Chế độ:** " + ("Dự đoán trước giao dịch" if profile == "pre_transaction" else "Phân tích sau giao dịch"))
    override = st.checkbox("Thử ngưỡng khác", value=False)
    threshold = (st.number_input("Ngưỡng thử nghiệm", min_value=0.0, max_value=1.0,
                                value=info["threshold"], step=0.01, format="%.12f")
                 if override else info["threshold"])
    st.caption(f"Ngưỡng đang dùng: {threshold:.12g}; ngưỡng lưu: {info['threshold']:.12g}")
    if override:
        st.warning("Bảng đánh giá dùng ngưỡng đã lưu; các chỉ số không được tính lại theo ngưỡng thử.")

if profile == "post_transaction":
    st.warning("Chế độ sau giao dịch dùng số dư sau và sai lệch số dư. Kết quả này không chứng minh khả năng chặn giao dịch trước khi thực hiện.")
else:
    st.info("Chế độ trước giao dịch chỉ dùng loại, số tiền và bước thời gian; không yêu cầu số dư sau giao dịch.")
if info.get("status") != "VERIFIED":
    st.warning("Bundle lịch sử: mô hình từng được chọn theo Test và dùng đặc trưng số dư. Số liệu chất lượng chưa được tái kiểm chứng theo quy trình mới.")
st.caption("Điểm rủi ro là đầu ra mô hình có trọng số lớp, chưa được hiệu chuẩn thành xác suất gian lận thực tế. Phạm vi mô hình: TRANSFER và CASH_OUT.")

tab1, tab2, tab3 = st.tabs(["🔍 Kiểm tra một giao dịch", "📂 Kiểm tra theo lô (CSV)", "📊 Thống kê & đánh giá"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        ttype = st.selectbox("Loại giao dịch", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Số tiền (amount)", min_value=0.0, value=181000.0, step=1000.0)
        step = st.number_input("Bước thời gian (step, giờ)", min_value=1, max_value=2_147_483_647, value=1)
    row = {"step": step, "type": ttype, "amount": amount}
    with c2:
        row["nameOrig"] = st.text_input("Tài khoản nguồn (truy vết)", "C1305486145")
        if profile == "post_transaction":
            row["oldbalanceOrg"] = st.number_input("Số dư nguồn trước (oldbalanceOrg)", min_value=0.0, value=181000.0)
            row["newbalanceOrig"] = st.number_input("Số dư nguồn sau (newbalanceOrig)", min_value=0.0, value=0.0)
    with c3:
        row["nameDest"] = st.text_input("Tài khoản đích (truy vết)", "C553264065")
        if profile == "post_transaction":
            row["oldbalanceDest"] = st.number_input("Số dư đích trước (oldbalanceDest)", min_value=0.0, value=0.0)
            row["newbalanceDest"] = st.number_input("Số dư đích sau (newbalanceDest)", min_value=0.0, value=0.0)
    if st.button("Phân tích giao dịch", type="primary"):
        try:
            res = predict_validated(spark, model, info, pd.DataFrame([row]), threshold).iloc[0]
            if res["scoring_status"] != "Đã chấm điểm":
                st.info(res["scoring_status"] + ": " + (res["validation_error"] or "Không có mô hình được huấn luyện cho loại này."))
            else:
                a, b = st.columns(2)
                a.metric("Điểm rủi ro", f"{res['fraud_probability']:.2%}")
                b.metric("Ngưỡng quyết định", f"{threshold:.12g}")
                if res["prediction"] == "Nghi ngờ gian lận":
                    st.error("⚠️ CẢNH BÁO: giao dịch nghi ngờ gian lận; cần thẩm định.")
                else:
                    st.success("Giao dịch được mô hình phân loại bình thường.")
        except Exception as exc:
            st.error("Không thể phân tích giao dịch. Hãy kiểm tra dữ liệu và môi trường Spark.")
            st.caption(str(exc))

with tab2:
    st.write("Các cột bắt buộc: " + ", ".join(f"`{k}`" for k in required_columns(profile)))
    st.caption("Giữ nguyên thứ tự và nhãn isFraud nếu có. Dòng lỗi và loại ngoài phạm vi được trả về với trạng thái chưa đánh giá.")
    up = st.file_uploader("Chọn tệp CSV", type=["csv"])
    if up is not None:
        try:
            pdf = pd.read_csv(up)
            with st.spinner(f"Spark đang xử lý {len(pdf):,} giao dịch..."):
                res = predict_validated(spark, model, info, pdf, threshold)
            cols = st.columns(4)
            for col, label, count in zip(cols, ["Tổng giao dịch", "Bình thường", "Nghi ngờ gian lận", "Chưa đánh giá"],
                                        [len(res), int((res.prediction == "Bình thường").sum()),
                                         int((res.prediction == "Nghi ngờ gian lận").sum()),
                                         int((res.prediction == "Chưa đánh giá").sum())]):
                col.metric(label, f"{count:,}")
            bad = res[res.validation_error != ""]
            if len(bad):
                st.warning(f"Có {len(bad)} dòng không hợp lệ; xem cột validation_error.")
            st.dataframe(res, width="stretch", height=380)
            st.download_button("Tải kết quả (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                               "ket_qua_du_doan.csv", "text/csv")
        except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
            st.error("CSV không hợp lệ: " + str(exc))
        except Exception as exc:
            st.error("Chấm điểm thất bại. Kiểm tra môi trường Spark và mô hình.")
            st.caption(str(exc))

with tab3:
    try:
        metric_dir = Path(config.OUTPUT_DIR) / "audit_original/metrics" if legacy else Path(config.METRIC_DIR)
        mpath = (Path(config.BASE_DIR) / manifest["experiments"][selected]["results_path"]
                 if manifest and not legacy else metric_dir / "model_results.json")
        mr = read_json(mpath)
        if mr.get("data", {}).get("sample_fraction", 1) < 1:
            st.warning(f"Thực nghiệm trên mẫu {mr['data']['sample_fraction']:.0%} đồng đều từ CSV thật; không phải kết quả huấn luyện toàn bộ dữ liệu.")
        st.subheader("So sánh 4 thuật toán trên Test tại ngưỡng lưu")
        results = mr["results"]
        table = pd.DataFrame(results).T[["precision", "recall", "f1", "auc_roc", "auc_pr", "TP", "TN", "FP", "FN", "train_time_s"]]
        st.dataframe(table.rename(columns={"precision": "Precision", "recall": "Recall", "f1": "F1",
                                          "auc_roc": "AUC-ROC", "auc_pr": "AUC-PR", "train_time_s": "Huấn luyện (giây)"}), width="stretch")
        st.bar_chart(table[["precision", "recall", "f1", "auc_pr"]])
        if mr.get("selection"):
            st.caption(f"Mô hình chọn: {mr['best_model']}; tiêu chí AUC-PR, F1, Precision trên Validation.")
            validation = pd.DataFrame({n: r["validation"] for n, r in results.items()}).T
            st.dataframe(validation[["auc_pr", "precision", "recall", "f1"]], width="stretch")
        q = read_json(metric_dir / "quality.json")
        st.write(f"CSV gốc: {q['rows']:,} giao dịch. Sau làm sạch: {q['clean_rows']:,}.")
        if "raw_label_counts" in q:
            st.write(f"Gian lận gốc: {q['raw_label_counts'].get('1', 0):,}; sau làm sạch: {q['clean_label_counts'].get('1', 0):,}.")
        st.dataframe(pd.DataFrame([{"Tập": k, "Giao dịch": mr["split"][k],
                                    "Gian lận": mr["split"][k + "_fraud"],
                                    "Tỷ lệ gian lận": mr["split"][k + "_fraud"] / mr["split"][k]}
                                   for k in ("train", "valid", "test")]), width="stretch")
    except Exception as exc:
        st.error("Không thể đọc số liệu đánh giá thực tế: " + str(exc))

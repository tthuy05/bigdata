"""Streamlit's own script harness; no synthetic metrics are written."""
from pathlib import Path
from io import BytesIO

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
from src.prediction import load_bundle, predict_validated


def test_streamlit_three_tabs_single_prediction_and_profile_contract():
    app = AppTest.from_file(str(Path(config.BASE_DIR) / "app/app.py"), default_timeout=120).run()
    assert not app.exception
    assert len(app.tabs) == 3
    assert [t.label for t in app.tabs] == ["🔍 Kiểm tra một giao dịch", "📂 Kiểm tra theo lô (CSV)", "📊 Thống kê & đánh giá"]
    labels = [n.label for n in app.number_input]
    if Path(config.EXPERIMENT_MANIFEST).is_file():
        assert not any("newbalance" in label or "oldbalance" in label for label in labels)
    button = next(b for b in app.button if b.label == "Phân tích giao dịch")
    button.click().run()
    assert not app.exception and any(m.label == "Điểm rủi ro" for m in app.metric)
    assert any(m.label == "Ngưỡng quyết định" for m in app.metric)
    assert len(app.dataframe) >= 2
    if Path(config.EXPERIMENT_MANIFEST).is_file():
        experiment_selector = next(s for s in app.selectbox if s.label == "Thí nghiệm")
        experiment_selector.select("A_stratified").run()
        assert not app.exception
        assert any("newbalanceOrig" in n.label for n in app.number_input)


def test_current_bundle_csv_predictions_and_bad_rows(spark):
    model, info = load_bundle(spark)
    sample = pd.read_csv(config.DATA_PATH, nrows=100)
    sample = pd.concat([sample.iloc[:5], sample.iloc[:1].assign(amount=-1)], ignore_index=True)
    result = predict_validated(spark, model, info, sample)
    assert len(result) == len(sample)
    assert list(result.isFraud) == list(sample.isFraud)
    assert result.iloc[-1].scoring_status == "Dữ liệu không hợp lệ"
    assert result.iloc[-1].prediction == "Chưa đánh giá"


def test_current_experiment_artifact_consistency():
    if not Path(config.EXPERIMENT_MANIFEST).is_file():
        pytest.skip("Chưa có manifest A/B đã công bố; không coi lịch sử là bằng chứng mới.")
    from tools.verify_artifacts import verify
    assert verify()["algorithms"] == 16


def test_streamlit_csv_branch_and_download_payload(monkeypatch):
    # Exercise the actual CSV branch with real-source rows; the file picker itself is mocked.
    sample = pd.read_csv(config.DATA_PATH, nrows=10)
    sample = pd.concat([sample, sample.iloc[:1].assign(amount=-1)], ignore_index=True)
    content = sample.to_csv(index=False).encode("utf-8")
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: BytesIO(content))
    captured = {}
    original_download = st.download_button
    def download(label, data, *args, **kwargs):
        captured["data"] = data
        return original_download(label, data, *args, **kwargs)
    monkeypatch.setattr(st, "download_button", download)
    app = AppTest.from_file(str(Path(config.BASE_DIR) / "app/app.py"), default_timeout=120).run()
    assert not app.exception
    result = next(d.value for d in app.dataframe if "validation_error" in d.value.columns)
    assert len(result) == 11 and result.iloc[-1].validation_error
    assert any(m.label == "Tổng giao dịch" and m.value == "11" for m in app.metric)
    assert len(app.get("download_button")) == 1
    downloaded = pd.read_csv(BytesIO(captured["data"]))
    assert len(downloaded) == 11 and list(downloaded.isFraud) == list(sample.isFraud)
    assert downloaded.iloc[-1].prediction == "Chưa đánh giá"


def test_streamlit_bad_csv_shows_vietnamese_error(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: BytesIO(b"step,type\n1,TRANSFER\n"))
    app = AppTest.from_file(str(Path(config.BASE_DIR) / "app/app.py"), default_timeout=120).run()
    assert not app.exception
    assert any("CSV không hợp lệ" in error.value and "amount" in error.value for error in app.error)

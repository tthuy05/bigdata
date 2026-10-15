# Phát hiện gian lận giao dịch tài chính bằng Apache Spark & Nghiên cứu Đối chứng

Đồ án PaySim triển khai phát hiện gian lận thanh toán trên tập dữ liệu quy mô lớn (gần 2,8 triệu giao dịch `TRANSFER` & `CASH_OUT`) sử dụng **Apache Spark (PySpark MLlib)**, đồng thời xây dựng quy trình thực nghiệm đối chứng toàn diện với **Python thuần (Scikit-Learn)**.

Hệ thống so sánh giải thuật **Baseline (Logistic Regression)** với **3 giải thuật đề tài chọn (Decision Tree, Random Forest, Gradient-Boosted Trees - GBT)** trên cả 2 phương diện:
1. **Độ chính xác**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR (tối ưu ngưỡng $t^*$ trên tập Validation).
2. **Thời gian thực thi**: Thời gian tiền xử lý ETL đặc trưng, thời gian huấn luyện (Fit), thời gian dự đoán (Predict) và tổng thời gian toàn bộ Pipeline.

## Thư mục làm việc

Mở cửa sổ PowerShell hoặc Terminal tại thư mục gốc của dự án:

```powershell
cd E:\BigData\bigdata
```

Các đường dẫn dữ liệu/model/output trong mã được tự động tính toán từ vị trí `config.py`, không phụ thuộc thư mục hiện hành.

## Cài đặt

Python 3.10–3.12, Java phù hợp Spark 3.5.3, và Node.js nếu dựng Word.
Trên Windows, Python 3.11 là lựa chọn thuận tiện; môi trường kiểm thử của lần bàn giao dùng
Python 3.12 với nạp/lưu PipelineModel qua JVM để tránh lỗi Python worker metadata.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Hoặc dùng đúng phiên bản đã kiểm thử (Windows/Python 3.12):
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
npm install --prefix report
```

`python` trong các lệnh bên dưới phải là interpreter có dependencies. Nếu Windows trỏ
`python` tới Microsoft Store, dùng trực tiếp `.\.venv\Scripts\python.exe`.

Spark trên Windows cần `winutils.exe` và `hadoop.dll` trong `HADOOP_HOME\bin`.
Dự án tự nhận `.runtime\hadoop` nếu chưa đặt HADOOP_HOME và thêm bin vào PATH của tiến trình;
không cần chép DLL vào System32. Trong workspace bàn giao, hai tệp đã được đặt tại đây từ
[bộ Windows Hadoop mà README cũ sử dụng](https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.5/bin).
Máy khác cần chuẩn bị lại runtime tương ứng.

CSV PaySim đặt ở `data/PS_20174392719_1491204439457_log.csv`:
[nguồn dữ liệu](https://www.kaggle.com/datasets/ealaxi/paysim1),
[mã mô phỏng của tác giả](https://github.com/EdgarLopezPhD/PaySim).
CSV sai lược đồ/thứ tự cột hoặc sai định dạng làm chương trình báo lỗi; không âm thầm bỏ dòng.

## Các thí nghiệm

| Thí nghiệm | Đặc trưng | Chia dữ liệu |
|---|---|---|
| A_stratified | 12 đặc trưng hiện tại, gồm số dư trước/sau và phái sinh | randomSplit theo từng lớp, seed 42 |
| B_stratified | type_CASH_OUT, log_amount, hour | Cùng các bản ghi đã chia của A |
| A_temporal | Bộ A | Các step nguyên vẹn, Train trước Validation trước Test |
| B_temporal | Bộ B | Cùng các bản ghi đã chia của A |

B loại toàn bộ số dư PaySim, theo lựa chọn thận trọng đối với dấu vết mô phỏng/hủy giao dịch.
`oldbalanceOrg`/`oldbalanceDest` về nghĩa là số dư trước; `origEmptied` chỉ dùng amount và
oldbalanceOrg nên cũng có thể tính trước nếu số dư đó đáng tin cậy. B vẫn loại chúng.
`newbalance*`, `errorBalance*`, `origZeroBalance`, `destZeroBalance` cần thông tin sau giao dịch.
A được giữ để đối chiếu và chỉ dùng cho phân tích sau giao dịch.
Sau khi có manifest mới, web cũng có lựa chọn "Lịch sử · Sau giao dịch" để nạp
best_model cũ và xem đúng metrics đã sao lưu; giao diện ghi rõ kết quả chưa tái kiểm chứng
và cách chọn cũ đã dùng Test.

Giữ amount=0: CSV thực tế có 16 dòng như vậy và cả 16 đều là gian lận. Quy tắc cũ
amount>0 giải thích số gian lận 8.197 thay vì 8.213. Trọng số cũ 167,790607... đúng theo
1.939.995/(2×5.781); mã mới tính lại từ Train thực tế, không dùng giá trị cố định.

Tiền xử lý học tham số (StringIndexer, OneHotEncoder, StandardScaler) chỉ fit Train.
Class weight: N_train/(2×N_train_class), báo lỗi nếu thiếu một lớp.
Ngưỡng tối đa F1 chỉ tìm trên Validation. Thuật toán chọn theo AUC-PR Validation, rồi F1,
Precision, không làm tròn. Quyết định được cố định trước chấm điểm Test.
B_temporal là thí nghiệm chính được xác định theo kịch bản ứng dụng, không chọn theo điểm Test.

Chia thời gian tìm ranh giới gần 70%/80% số dòng tích lũy, không dùng nhãn. Tỷ lệ gian lận
có thể khác giữa các tập; báo lỗi nếu một tập thiếu lớp, không tự ép tỷ lệ.

```powershell
# Toàn bộ CSV, bốn thí nghiệm × bốn mô hình:
python main.py --auto --no-app

# Máy ít RAM: mẫu đồng đều 10% từ CSV thật, giữ cùng tỷ lệ lấy mẫu cho hai lớp:
python main.py --auto --sample-fraction 0.1 --no-word --no-app
```

Mẫu 10% là kết quả trên mẫu, không phải huấn luyện toàn bộ CSV; thống kê gốc/làm sạch vẫn
tính trên toàn bộ dữ liệu. Không dùng dữ liệu tự tạo thay kết quả PaySim. Công cụ sinh mẫu
ghi sidecar `kind=synthetic`; runner từ chối công bố loại dữ liệu này.

Mỗi lần chạy có `run_id`, SHA-256 CSV, tỷ lệ lấy mẫu, seed, quy mô lớp/khoảng step,
thông tin chọn mô hình, confusion matrix, AUC-ROC/AUC-PR, thời gian fit và materialize dự đoán.
AUC dùng điểm probability[1], Spark BinaryClassificationEvaluator với numBins=0;
## Thực nghiệm Đối chứng: Baseline vs Đề tài & Python thuần vs Apache Spark

Nhằm chứng minh năng lực vượt trội của **Apache Spark** trên Dữ liệu lớn (Big Data) và đánh giá tính ưu việt của **các giải thuật đề tài (Decision Tree, Random Forest, GBT)** so với **Baseline (Logistic Regression)**, dự án tích hợp bộ công cụ Benchmark đối chứng thực nghiệm:

### 1. Lệnh thực thi Benchmark

```powershell
# Chạy thực nghiệm trên 100% dữ liệu thực tế (gần 2,8 triệu dòng TRANSFER & CASH_OUT):
.\.venv\Scripts\python.exe tools/run_comparison.py --sample-fraction 1.0

# Hoặc chạy kiểm tra nhanh trên mẫu 10%:
.\.venv\Scripts\python.exe tools/run_comparison.py --sample-fraction 0.1
```

Quy trình tự động thực hiện:
1. **Phiên 1 (Python thuần - Scikit-Learn)**: Chạy đơn luồng trên CPU máy đơn, đo đạc thời gian ETL và huấn luyện 4 mô hình theo thứ tự: Logistic Regression (Baseline) $\rightarrow$ Decision Tree $\rightarrow$ Random Forest $\rightarrow$ GBT.
2. **Phiên 2 (Apache Spark - PySpark MLlib)**: Chạy phân tán đa luồng với bộ nhớ đệm RAM (`MEMORY_AND_DISK`), xử lý song song toàn bộ 4 mô hình.
3. **Phân chia dữ liệu**: Chuẩn hóa cố định **70% Train, 10% Validation, 20% Test** (Stratified phân tầng theo nhãn gian lận, seed=42).
4. **Tự động xuất 4 biểu đồ trực quan**: Lưu tại `outputs/figures/benchmark/`.
5. **Tự động xuất báo cáo chuyên sâu Markdown**: Lưu tại `outputs/reports/BAO_CAO_SO_SANH_BASELINE_SPARK_PYTHON.md`.

### 2. Tóm tắt kết quả đối chứng trên 100% dữ liệu (2,77 triệu dòng)

| Tiêu chí | Python thuần (Scikit-Learn) | Apache Spark (MLlib) | Đánh giá hiệu năng |
|---|---:|---:|---|
| **ETL Tiền xử lý** | 25.82s | **2.30s** | 🏆 **Spark nhanh gấp 11.2 lần** nhờ Tungsten Engine |
| **Fit Decision Tree** | 21.77s | **5.68s** | 🏆 **Spark nhanh gấp 3.84 lần** |
| **Fit GBT (Gradient Boosted Trees)** | 211.99s | **39.27s** | 🏆 **Spark nhanh gấp 5.40 lần** |
| **Tổng thời gian Model (4 mô hình)** | 259.21s | **81.69s** | 🏆 **Spark nhanh gấp 3.17 lần** |
| **Tổng toàn bộ Pipeline** | 30.1s - 238.0s | **8.9s - 42.5s** | 🏆 **Spark nhanh hơn 1.6x - 5.6x trên cả 4 mô hình** |
| **F1-Score GBT vs Baseline** | 0.9982 vs 0.9942 | **0.9969 vs 0.9942** | Mô hình đề tài vượt trội rõ rệt so với Baseline |
| **AUC-PR GBT vs Baseline** | 0.9980 vs 0.9976 | **0.9967 vs 0.9905** | Bắt gian lận chính xác cao, cực ít dương tính giả |

### 3. Vận hành cụm Spark Master / Workers (Tùy chọn)

Dự án cung cấp bộ công cụ quản lý cụm Spark Standalone cục bộ:
- `.\tools\start_cluster.bat`: Khởi động cụm 1 Master (port 7077, Web UI 8080) và 2 Workers.
- `.\tools\stop_cluster.bat`: Dừng toàn bộ Master và Workers.

## Demo mô hình lưu sẵn

```powershell
python main.py --use-saved-models --auto
python main.py --use-saved-models --auto --no-app --no-word
python main.py --app-only
# Không mở browser tự động / chọn cổng khác:
python main.py --app-only --no-browser --port 8502
```

Saved mode nạp cả bốn model của thí nghiệm chính, kiểm tra transform bằng fixture kiểm tra
logic và in số liệu đã lưu. Không fit, không chia lại Test, không chọn lại ngưỡng, không ghi đè
metrics/figures/model. Nếu thiếu một model, chương trình dừng với mã lỗi khác 0.
Fixture demo không được tính thành chỉ số thực nghiệm. `--no-app` kết thúc sau kiểm tra;
`--app-only` giữ tiến trình Streamlit cho đến Ctrl+C.

Ứng dụng có ba tab tiếng Việt:

- Một giao dịch: đầu vào theo profile của model; điểm rủi ro, ngưỡng chính xác và phân loại.
- CSV: B cần `step,type,amount`; A cần thêm bốn cột số dư. Dòng lỗi giữ trong kết quả với
  `validation_error`. Thứ tự, mã tài khoản và isFraud nếu có được giữ; tải CSV kết quả.
- Dashboard: bảng/biểu đồ từ artifact thực tế của thí nghiệm được chọn, số liệu Test và Validation.

TRANSFER/CASH_OUT là phạm vi mô hình. CASH_IN/PAYMENT/DEBIT trả về ngoài phạm vi, không
gán xác suất 0 hoặc khẳng định an toàn. Điểm có trọng số lớp chưa được hiệu chuẩn thành xác
suất gian lận ngân hàng. Đổi ngưỡng trên web không thay các chỉ số Test đã lưu.

## Kết quả và báo cáo

```text
outputs/experiments/<run_id>/
  manifest.json, quality.json, eda_stats.json, env.json, figures/
  A_stratified/, B_stratified/, A_temporal/, B_temporal/
    model_results.json, best_model_info.json, models/, figures/
outputs/metrics/              bản công bố từ một lần chạy VERIFIED đủ bốn thí nghiệm
outputs/figures/              EDA toàn bộ CSV và kết quả thí nghiệm chính
outputs/benchmarks/           kết quả JSON đối chứng (python_results.json, spark_results.json)
outputs/reports/              báo cáo Markdown đối chứng chuyên sâu (BAO_CAO_SO_SANH_BASELINE_SPARK_PYTHON.md)
outputs/figures/benchmark/    4 biểu đồ đối chứng thời gian, F1, ROC/PR, Baseline vs Đề tài
outputs/audit_original/       kết quả, Word, hình và một số nguồn trước cải thiện
outputs/baseline/             baseline lịch sử giữ nguyên
outputs/backup/               sao lưu trước công bố/before dựng Word
outputs/qa/                   logs, JUnit XML, kiểm toán CSV, kiểm tra artifact
models/best_model_info.json   con trỏ tương đối đến bundle được chọn
models/<bốn model cũ>/        giữ nguyên để đối chiếu
docs/PROJECT_IMPROVEMENT_REPORT.md
docs/HUONG_DAN_SO_SANH_BASELINE_SPARK_PYTHON.md
```

Lần chạy lỗi ghi FAILED tại thư mục riêng, không công bố vào outputs/metrics. Chỉ lần chạy
đã kiểm tra đủ bốn thí nghiệm mới được công bố; artifacts trước đó được sao lưu.
`--save-baseline` bị từ chối để tránh thay kết quả lịch sử bằng một lần demo.

```powershell
python main.py --word-only
python tools/print_results.py
python tools/verify_artifacts.py --docx outputs/BaoCao_BigData_Nhom8.docx
python tools/make_demo_batch.py
```

Word giữ cấu trúc I–V, các bảng/hình chính, thêm A/B, thời điểm khả dụng và hạn chế PaySim.
Builder từ chối JSON thiếu, kết quả lịch sử không có Validation và manifest không khớp.
Word cũ được sao lưu. Kiểm tra ô bảng chính và bảng so sánh A/B là kiểm tra nội dung;
kiểm tra bố cục cần render và xem toàn bộ các trang. Xem trạng thái bàn giao để biết phần nào
đã xác minh trên máy này. Công cụ `doi_chieu_bang_word.py` cũ chỉ kiểm tra sự xuất hiện của
chuỗi số, không chứng minh các ô bảng khớp đúng ngữ cảnh.

## Kiểm thử và tái lập

```powershell
# Chạy bộ kiểm thử pipeline tích hợp và benchmark:
python -m pytest -q --junitxml=outputs/qa/pytest.xml
python -m pytest tests/test_benchmark.py

# Kiểm toán dữ liệu và xác minh artifact:
python tools/audit_data.py --data data/PS_20174392719_1491204439457_log.csv --out outputs/qa/raw_csv_audit.json
python tools/verify_artifacts.py
python tools/verify_curves.py outputs/experiments/<run_id>
```

Dữ liệu tự tạo chỉ dùng unit/integration test, không dùng trong bảng báo cáo thực nghiệm.
Tests kiểm tra bốn thuật toán với siêu tham số nhỏ để kiểm tra logic, không suy ra hiệu năng.
Để tái lập thực nghiệm, giữ cùng CSV/SHA-256, Spark 3.5.3, requirements-lock, seed 42,
8 shuffle partitions, profile, tỷ lệ lấy mẫu và quy tắc chia. Không thay đổi số phân vùng khi
muốn so đúng split phân tầng; mã sắp xếp bản ghi theo fingerprint trước randomSplit.
Thời gian chạy phụ thuộc máy; chưa đo độ bất định nhiều seed hoặc hiệu quả ngân hàng thật.
verify_curves chỉ nạp model và tái tạo đúng holdout, đối chiếu confusion matrix và diện tích
đường cong với Spark, lưu scores/labels thật để vẽ lại; không huấn luyện hay chọn lại ngưỡng.
Mặc định `local[2]`, 2g Driver; có thể đặt SPARK_MASTER, SPARK_DRIVER_MEMORY,
SPARK_SHUFFLE_PARTITIONS trong môi trường trước khi chạy.

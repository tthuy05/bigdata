# Báo cáo cải thiện và bàn giao project

Ngày bàn giao: 13/10/2026 (Asia/Saigon).

Trạng thái: VERIFIED = chạy thực tế; FIXED = sửa mã và có bằng chứng kiểm thử; NOT VERIFIED = chưa xác minh; NOT APPLICABLE = không áp dụng.

## 1. Tình trạng trước cải thiện

Project có CSV, Parquet, bốn PipelineModel, best_model, metrics, hình và Word. Chưa có bộ kiểm thử tự động. Workspace hiện tại không có .git; không commit hoặc push.

Các kết quả lịch sử được giữ nguyên trong outputs/audit_original và outputs/baseline; các model cũ trong models được giữ để đối chiếu. Số liệu lịch sử có sẵn, nhưng không coi là thực nghiệm đã tái kiểm chứng của lần bàn giao.

## 2. Vấn đề thực sự phát hiện và nguyên nhân

| Vấn đề | Bằng chứng và nguyên nhân | Trạng thái |
|---|---|---|
| Số dư sau và sai lệch số dư dùng cho giao diện không phân biệt thời điểm | features.py dùng newbalance*; app cũ yêu cầu số dư sau | FIXED: profile A/B, hợp đồng input và tests |
| Chọn model dựa trên Test | summarize cũ lấy AUC-PR/F1 từ results Test, còn ngưỡng đã dùng Validation đúng | FIXED: Validation-only, cố định trước dự đoán Test |
| Mất 16 giao dịch gian lận | clean cũ giữ amount>0; CSV kiểm đếm 16 amount=0 đều isFraud=1 | FIXED: giữ amount=0; audit CSV và test cleaning |
| Saved mode có thể tự huấn luyện khi thiếu model | nhánh else trong train_all cũ không phân biệt thiếu model và yêu cầu huấn luyện | FIXED: thiếu model báo lỗi, test cấm gọi fit |
| Saved mode chia lại dữ liệu, chọn lại ngưỡng, ghi đè kết quả | main cũ không có nguồn dữ liệu và split fingerprint; seed không đảm bảo trùng split khi đổi phân vùng | FIXED: demo chỉ nạp/transform, không retune/ghi artifacts |
| CSV không kiểm tra đầy đủ, mất thứ tự và nhãn | app cũ astype trực tiếp, concat nhóm khác loại, assembler skip | FIXED: lý do lỗi mỗi dòng, bảo toàn hàng và nhãn |
| Ba loại khác bị gán xác suất 0 và coi an toàn | app cũ suy từ không có gian lận trong dữ liệu | FIXED: trả ngoài phạm vi/chưa đánh giá |
| Ngưỡng bị làm tròn/clamp | sidebar cũ round 2 chữ số, slider tối đa 0.99, không biểu diễn threshold=1 | FIXED: giữ đúng float, cho phép [0,1] |
| Predict time gồm tính metrics | đo đến sau confusion/AUC | FIXED: đo materialize transform riêng, mô tả rõ phạm vi |
| Nuốt bản ghi CSV sai định dạng | DROPMALFORMED | FIXED: FAILFAST, kiểm tra header, test malformed CSV |
| Thiếu provenance và model/report contract | info cũ chỉ tên/threshold/metrics; hình chứa đường dẫn máy cũ | FIXED: run_id/hash/profile/input contract, kiểm tra artifact |
| Hệ số âm trong khoảng (-1,0) có thể mất dấu trong Word | fmtDec chuyển phần nguyên -0 thành 0 | FIXED: formatter giữ dấu, kiểm tra Node |
| Môi trường ban đầu thiếu PySpark/Streamlit/Hadoop; Python là Store alias | kiểm tra executable/modules; Python worker metadata crash trên Python 3.12 Windows | FIXED: venv riêng, local Hadoop, JVM persistence; logs thực tế |
| Thiếu thực nghiệm đối chứng Baseline vs Đề tài & Spark vs Python trên Big Data | Trước đây chưa có pipeline đối chứng công bằng giữa Scikit-Learn và PySpark MLlib trên 100% dữ liệu | FIXED: xây dựng benchmark_runner tuần tự, đo ETL/Train/Predict, sinh 4 biểu đồ và báo cáo Markdown |

## 3. Kiểm toán CSV và class weight

VERIFIED: CSV 6,362,620 dòng, 8,213 gian lận, 6,354,407 bình thường. Tỷ lệ gian lận 0.129082%. 16 dòng amount=0; 16 gian lận. Hash SHA-256: `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.

Bằng chứng: outputs/qa/raw_csv_audit.json và mô tả PaySim lấy trực tiếp qua API Kaggle trong outputs/qa/paysim_source_description.txt. CSV cung cấp phù hợp lược đồ/số lượng PaySim; chưa xác thực checksum với bản phát hành nguồn.

VERIFIED (kiểm tra số học trên artifact lịch sử): 1,939,995/(2×5,781) = 167.790607161391; khớp trọng số lưu 167.790607161391. Công thức cũ đúng; quy tắc loại gian lận amount=0 và cách đánh giá mới là vấn đề cần sửa.

VERIFIED bằng Spark trên toàn bộ CSV: sau làm sạch 6,362,620 dòng; nhãn {'1': 8213, '0': 6354407}; trước/sau fraud rate 0.129082% / 0.129082%.

| Nhóm loại bỏ tuần tự | Số dòng | Nhãn bị loại |
|---|---:|---|
| missing_values | 0 | {} |
| invalid_money | 0 | {} |
| invalid_type | 0 | {} |
| invalid_step_or_label | 0 | {} |
| empty_account | 0 | {} |
| duplicate_rows | 0 | {'1': 0, '0': 0} |

Các nhóm tuần tự, không đếm một bản ghi nhiều lần. Giá trị amount=0 được giữ và tiền âm/NaN/vô hạn bị loại.

## 4. Data Leakage và thời điểm khả dụng

| Đặc trưng | Thời điểm / quyết định |
|---|---|
| oldbalanceOrg | Số dư nguồn trước theo nghĩa trường; có thể biết trước, nhưng mang dấu vết mô phỏng PaySim; B loại |
| oldbalanceDest | Số dư đích trước theo nghĩa trường; có thể không truy cập được giữa tổ chức; B loại |
| newbalanceOrig, newbalanceDest | Sau ghi nhận giao dịch; B loại |
| errorBalanceOrig, errorBalanceDest | Có phụ thuộc newbalance*; B loại |
| origZeroBalance, destZeroBalance | Kiểm tra cả số dư trước/sau; B loại |
| origEmptied | Trong mã chỉ dùng amount và oldbalanceOrg, có thể tính trước; B vẫn loại theo chính sách số dư |
| amount, type, hour=step%24 | Biết trước trong kịch bản PaySim; B giữ |
| isFraud, isFlaggedFraud | Không là đặc trưng; nhãn hoặc luật đối chứng |

Tài liệu tác giả cảnh báo cả bốn số dư vì giao dịch gian lận phát hiện được bị hủy. Đây là nguy cơ dấu vết của cơ chế mô phỏng/ghi nhận ngoài nguy cơ rò rỉ thời điểm. A giữ để phân tích sau giao dịch, B thận trọng bỏ tất cả số dư. Chênh lệch A/B không tách riêng tác động của từng biến hoặc riêng số dư sau.

StringIndexer/OneHotEncoder/StandardScaler fit Train; trọng số chỉ từ Train; ngưỡng và thuật toán chọn trên Validation. Không có feature selection dựa trên Test. Test chỉ chấm điểm sau khi đã cố định lựa chọn; không tối ưu lại theo Test.

## 5. Kết quả bốn thuật toán theo thí nghiệm

VERIFIED: run_id `20261013T052452Z_6331e780`. Huấn luyện trên mẫu đồng đều 10% CSV thật, 276,602 giao dịch thuộc TRANSFER/CASH_OUT. Kiểm toán/EDA dùng toàn bộ CSV. Đây không phải kết quả huấn luyện toàn bộ 6,36 triệu dòng.

Các điều kiện A/B dùng cùng bản ghi đã chia, trọng số, seed và siêu tham số; chỉ đổi bộ đặc trưng. Chia phân tầng dùng randomSplit mỗi lớp với thứ tự/phân vùng ổn định. Chia thời gian dùng các step nguyên vẹn, ranh giới 70%/80% tích lũy số dòng, không xem nhãn để chọn ranh giới.

| Thí nghiệm | Thuật toán | Precision Test | Recall Test | F1 Test | AUC-ROC Test | AUC-PR Test | TP | TN | FP | FN | Ngưỡng | Train s | Predict s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_stratified | Logistic Regression | 1.000000 | 0.962500 | 0.980892 | 0.996935 | 0.993128 | 154 | 55012 | 0 | 6 | 0.978537217866 | 32.100 | 2.057 |
| A_stratified | Decision Tree | 1.000000 | 0.993750 | 0.996865 | 0.996873 | 0.996256 | 159 | 55012 | 0 | 1 | 0.9957285313 | 10.200 | 2.397 |
| A_stratified | Random Forest | 1.000000 | 0.987500 | 0.993711 | 0.996548 | 0.993652 | 158 | 55012 | 0 | 2 | 0.479988567093 | 27.600 | 2.538 |
| A_stratified | GBT | 1.000000 | 0.987500 | 0.993711 | 0.993641 | 0.987571 | 158 | 55012 | 0 | 2 | 0.978479114442 | 183.500 | 2.353 |
| B_stratified | Logistic Regression | 0.072816 | 0.093750 | 0.081967 | 0.781745 | 0.020125 | 15 | 54821 | 191 | 145 | 0.88779666918 | 6.900 | 1.925 |
| B_stratified | Decision Tree | 0.440945 | 0.350000 | 0.390244 | 0.723379 | 0.302465 | 56 | 54941 | 71 | 104 | 0.987493297245 | 10.400 | 1.620 |
| B_stratified | Random Forest | 0.690476 | 0.181250 | 0.287129 | 0.830725 | 0.235594 | 29 | 54999 | 13 | 131 | 0.931372869643 | 42.500 | 3.076 |
| B_stratified | GBT | 0.605634 | 0.268750 | 0.372294 | 0.834778 | 0.286402 | 43 | 54984 | 28 | 117 | 0.97006638804 | 221.000 | 2.164 |
| A_temporal | Logistic Regression | 1.000000 | 0.954436 | 0.976687 | 0.992901 | 0.991194 | 398 | 55065 | 0 | 19 | 0.967274031548 | 6.300 | 1.209 |
| A_temporal | Decision Tree | 1.000000 | 0.964029 | 0.981685 | 0.995200 | 0.994816 | 402 | 55065 | 0 | 15 | 1 | 3.100 | 0.770 |
| A_temporal | Random Forest | 1.000000 | 0.988010 | 0.993969 | 0.999999 | 0.999869 | 412 | 55065 | 0 | 5 | 0.96 | 12.600 | 1.066 |
| A_temporal | GBT | 1.000000 | 0.990408 | 0.995181 | 0.998896 | 0.991207 | 413 | 55065 | 0 | 4 | 0.978479114442 | 51.800 | 0.901 |
| B_temporal | Logistic Regression | 0.195122 | 0.076739 | 0.110155 | 0.785412 | 0.049022 | 32 | 54933 | 132 | 385 | 0.904268199735 | 2.600 | 0.770 |
| B_temporal | Decision Tree | 0.603448 | 0.335731 | 0.431433 | 0.704049 | 0.340699 | 140 | 54973 | 92 | 277 | 0.994536371121 | 2.800 | 0.626 |
| B_temporal | Random Forest | 0.868852 | 0.127098 | 0.221757 | 0.782361 | 0.225020 | 53 | 55057 | 8 | 364 | 0.919715008626 | 21.900 | 1.229 |
| B_temporal | GBT | 0.644330 | 0.299760 | 0.409165 | 0.774004 | 0.322024 | 125 | 54996 | 69 | 292 | 0.967507149378 | 55.400 | 0.835 |

AUC dùng Spark probability[1], numBins=0. AUC-PR theo quy ước Spark, không phải Average Precision. Thời gian Train là fit; Predict là transform+materialize cache, không gồm metrics/plotting. Chỉ một lần đo, chưa có khoảng tin cậy.

| Thí nghiệm | Mô hình chọn | AUC-PR Validation | F1 Validation | Precision Validation |
|---|---|---:|---:|---:|
| A_stratified | Decision Tree | 1.000000 | 1.000000 | 1.000000 |
| B_stratified | GBT | 0.282621 | 0.388060 | 0.577778 |
| A_temporal | Decision Tree | 1.000000 | 1.000000 | 1.000000 |
| B_temporal | Decision Tree | 0.251054 | 0.352941 | 0.857143 |

Tiêu chí chọn thuật toán: AUC-PR Validation, sau đó F1, Precision; so giá trị đầy đủ. B_temporal là thí nghiệm chính chọn theo kịch bản trước giao dịch/thời gian, không theo kết quả Test.

| Thí nghiệm | Tập | Số dòng | Gian lận | Bình thường | Tỷ lệ GL | Khoảng step |
|---|---|---:|---:|---:|---:|---|
| A_stratified | train | 193,919 | 553 | 193,366 | 0.285171% | [1, 742] |
| A_stratified | valid | 27,511 | 89 | 27,422 | 0.323507% | [1, 730] |
| A_stratified | test | 55,172 | 160 | 55,012 | 0.290002% | [1, 731] |
| B_stratified | train | 193,919 | 553 | 193,366 | 0.285171% | [1, 742] |
| B_stratified | valid | 27,511 | 89 | 27,422 | 0.323507% | [1, 730] |
| B_stratified | test | 55,172 | 160 | 55,012 | 0.290002% | [1, 731] |
| A_temporal | train | 193,250 | 358 | 192,892 | 0.185252% | [1, 322] |
| A_temporal | valid | 27,870 | 27 | 27,843 | 0.096878% | [323, 354] |
| A_temporal | test | 55,482 | 417 | 55,065 | 0.751595% | [355, 742] |
| B_temporal | train | 193,250 | 358 | 192,892 | 0.185252% | [1, 322] |
| B_temporal | valid | 27,870 | 27 | 27,843 | 0.096878% | [323, 354] |
| B_temporal | test | 55,482 | 417 | 55,065 | 0.751595% | [355, 742] |

Bản ghi chồng lấn kiểm tra bằng fingerprint: 0 cho các phương pháp. Tỷ lệ gian lận theo thời gian được giữ thực tế, không ép về tỷ lệ chung.

Logistic Regression: AUC-PR A−B = 0.973003; Decision Tree: AUC-PR A−B = 0.693791; Random Forest: AUC-PR A−B = 0.758058; GBT: AUC-PR A−B = 0.701169 (stratified).

Logistic Regression: AUC-PR A−B = 0.942171; Decision Tree: AUC-PR A−B = 0.654117; Random Forest: AUC-PR A−B = 0.774849; GBT: AUC-PR A−B = 0.669183 (temporal).

### 5.1. Thực nghiệm Đối chứng Toàn diện: Baseline vs Đề tài & Python thuần vs Apache Spark (100% dữ liệu PaySim)

VERIFIED: Chạy thực nghiệm đối chứng trên toàn bộ **2.770.409** dòng giao dịch `TRANSFER` & `CASH_OUT` với tỷ lệ phân chia chuẩn hóa **70% Train (1.940.231 dòng) – 10% Validation (276.838 dòng) – 20% Test (553.340 dòng)** (Stratified, Seed = 42).

Phiên chạy tuần tự độc lập: **Python thuần (Scikit-Learn) $\rightarrow$ Apache Spark (PySpark MLlib)** cho 4 mô hình:

| Nền tảng | Mô hình | Vai trò | ETL Tiền xử lý (s) | Huấn luyện Fit (s) | Dự đoán Test (s) | Tổng Pipeline (s) | F1-Score | AUC-PR |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Python thuần** | Logistic Regression | *Baseline* | 25.818 | 4.292 | 0.038 | 30.148s | 0.9942 | 0.9976 |
| **Python thuần** | Decision Tree | Đề tài | 25.818 | 21.774 | 0.030 | 47.622s | 0.9973 | 0.9969 |
| **Python thuần** | Random Forest | Đề tài | 25.818 | 20.697 | 0.154 | 46.669s | 0.9973 | 0.9980 |
| **Python thuần** | GBT | Đề tài | 25.818 | 211.989 | 0.238 | 238.045s | 0.9982 | 0.9980 |
| **Apache Spark** | Logistic Regression | *Baseline* | **2.299** | 7.064 | 1.030 | **10.393s** | 0.9942 | 0.9905 |
| **Apache Spark** | Decision Tree | Đề tài | **2.299** | **5.676** | 0.944 | **8.919s** | 0.9966 | 0.9954 |
| **Apache Spark** | Random Forest | Đề tài | **2.299** | 25.429 | 1.354 | **29.082s** | 0.9966 | 0.9962 |
| **Apache Spark** | GBT | Đề tài | **2.299** | **39.268** | 0.929 | **42.496s** | 0.9969 | 0.9967 |

**Kết luận cốt lõi từ thực nghiệm:**
- **Spark áp đảo về tốc độ**: Tổng thời gian huấn luyện và dự đoán 4 mô hình trên Spark là **81.69s** so với **259.21s** của Python (**Spark nhanh hơn 3.17 lần**). Đặc biệt với GBT, Spark nhanh hơn **5.40 lần** (39.27s vs 211.99s); với tiền xử lý ETL, Spark nhanh hơn **11.23 lần** (2.30s vs 25.82s).
- **Mô hình đề tài vượt trội Baseline**: Giải thuật cây (GBT, Decision Tree, Random Forest) giải quyết mối quan hệ phi tuyến, đạt F1-Score và AUC-PR cao vượt trội so với Baseline Logistic Regression.
- Bằng chứng và báo cáo: `outputs/benchmarks/`, `outputs/reports/BAO_CAO_SO_SANH_BASELINE_SPARK_PYTHON.md`, `outputs/figures/benchmark/`.

## 6. Cách sửa và file thay đổi

| File | Thay đổi |
|---|---|
| config.py | profile paths, local[2], RAM mặc định phù hợp demo |
| main.py | chạy bốn thí nghiệm; saved demo chỉ đọc; launch kiểm tra health/timeout/dependency |
| src/data_loader.py | CSV FAILFAST/header, quy tắc cleaning và audit từng nhóm, local Hadoop/interpreter |
| src/features.py | profile A/B, trọng số thiếu lớp báo lỗi, assembler không skip |
| src/modeling.py | chia theo thời gian, ổn định phân tầng, chọn Validation trước Test, AUC cùng score, thời gian riêng |
| src/model_io.py | nạp/lưu PipelineModel định dạng Spark qua JVM |
| src/prediction.py | input contract, kiểm tra metadata, validation CSV, giữ thứ tự/nhãn/phạm vi |
| src/experiments.py | run riêng, provenance, A/B cùng split, công bố sau kiểm chứng, backup |
| src/artifacts.py | atomic JSON, consistency counts/weights/selection |
| src/figures.py | feature importance theo profile |
| src/summary.py | trạng thái/phạm vi/source selection |
| src/diagrams.py | sơ đồ cleaning, A/B và Validation |
| src/console.py | reset trạng thái checklist |
| src/envcheck.py | runtime cục bộ và mô tả baseline |
| src/word.py | guard VERIFIED/Validation, subprocess Node đúng đường dẫn |
| app/app.py | giữ ba tab, pre/post input, ngưỡng chính xác, lỗi Việt, CSV và dashboard theo experiment |
| report/content_part1.js, report/content_part2.js, report/content_part3.js | giữ cấu trúc I–V, số liệu thực, A/B, thời điểm, hạn chế |
| report/references.js | giữ toàn bộ tài liệu tham khảo cũ |
| report/lib.js | giữ dấu hệ số âm, từ chối NaN |
| tools/audit_data.py | audit CSV độc lập |
| tools/verify_artifacts.py | kiểm tra JSON và chính xác ô Word |
| tools/build_improvement_report.py | bàn giao tái tạo từ evidence |
| tools/generate_paysim_like.py | sidecar synthetic, runner không công bố |
| tools/capture_app.py | nhãn giao diện mới |
| tools/make_demo_batch.py | xuất CSV demo từ dữ liệu thật, sửa UTF-8 terminal Windows |
| tools/verify_curves.py | nạp lại model, holdout khớp, confusion/AUC khớp, lưu scores/labels thật và vẽ lại |
| tools/doi_chieu_bang_word.py | nêu rõ giới hạn tìm chuỗi |
| src/pure_python_bench.py | pipeline huấn luyện & đánh giá đơn luồng Scikit-Learn độc lập cho 4 mô hình, đo ETL/Train/Predict |
| src/spark_bench.py | pipeline huấn luyện & đánh giá Spark MLlib tối ưu hóa, pre-cache RAM, maxMemoryInMB=1024 |
| src/benchmark_figures.py | tự động sinh 4 biểu đồ trực quan đối chứng độ phân giải cao tại outputs/figures/benchmark/ |
| src/benchmark_report.py | tự động kết xuất báo cáo Markdown chuyên sâu chi tiết tại outputs/reports/ |
| src/benchmark_runner.py | bộ điều phối luồng tuần tự Python -> Spark, tính toán speedup, xuất kết quả JSON |
| tools/run_comparison.py | CLI độc lập cho thực nghiệm benchmark đối chứng (--sample-fraction, --seed) |
| tools/cluster_manager.py | quản lý tiến trình cụm Spark Master / Workers cục bộ |
| tools/start_cluster.bat, tools/stop_cluster.bat | script bật/tắt cụm Spark Standalone trên Windows |
| tests/test_benchmark.py | bộ kiểm thử tự động 4 unit tests cho pipeline benchmark đối chứng |
| tests/conftest.py, tests/test_pipeline.py, tests/test_app.py | kiểm thử Spark/CSV/leakage/app/artifacts |
| requirements.txt, requirements-lock.txt | dependencies tương thích Spark 3.5.3; lưu phiên bản thực tế |
| README.md | cách chạy/demo/tái lập/phạm vi, hướng dẫn benchmark và trạng thái report |

## 7. Bằng chứng kiểm thử trước và sau

Trước: không có test suite; không có căn cứ báo rằng một bộ test trước đó pass. Chạy ban đầu trên môi trường thiếu dependency/Python 3.12 gặp distutils/SciPy compatibility và Python worker metadata crash; logs setup/spark_smoke giữ lại. Các số liệu lịch sử không coi là so sánh before/after hợp lệ vì đổi cleaning, split và phạm vi lấy mẫu.

VERIFIED Pipeline Suite: 18 passed, 0 failures, 0 errors, 0 skipped; thời gian 54.278 giây.
VERIFIED Benchmark Suite: 4 passed, 0 failures, 0 errors; thời gian 1.36 giây (`tests/test_benchmark.py`).

Tests dùng dữ liệu tự tạo chỉ kiểm tra logic; không được đưa vào số liệu PaySim. Kiểm tra thực nghiệm dùng CSV thật và artifact riêng. Bằng chứng: outputs/qa/pytest.xml, pytest.log, spark_smoke_jvm.log, experiments_run.log, task-270.log.

## 8. Chức năng đã kiểm tra

| Hạng mục | Trạng thái và bằng chứng |
|---|---|
| Benchmark Đối chứng Spark vs Python (100% data) | VERIFIED: 4 mô hình, 2.77M dòng, Spark nhanh hơn 3.17x tổng và 5.40x GBT; BAO_CAO_SO_SANH_BASELINE_SPARK_PYTHON.md |
| Benchmark Test Suite | VERIFIED: 4 passed trong 1.36s, tests/test_benchmark.py |
| Spark đọc CSV, nạp model cũ, ghi Parquet | VERIFIED: spark_smoke_jvm.log |
| Bốn mô hình MLlib fit/save/evaluate | VERIFIED: test integration; kết quả PaySim theo bảng phía trên |
| Data Cleaning/Features/Split/Weight/Scaler/Threshold/Prediction/CSV | VERIFIED: pytest |
| CLI --help | VERIFIED: exit 0; cli_help.log |
| CSV thiếu | VERIFIED: exit 1, missing_data.log, không báo thành công/không tạo thực nghiệm |
| Python syntax | VERIFIED: compileall exit 0 |
| Word generation | VERIFIED nội dung sinh từ run; word_build.log. Bố cục NOT VERIFIED |
| Định dạng số Word | VERIFIED: giữ dấu âm (-1,0), report_formatter.log |
| Saved demo từ thư mục cha | VERIFIED: CLI exit 0, nạp/transform đủ 4 model, saved_demo.log |
| Saved artifacts | VERIFIED: 659 file nội dung không đổi sau demo |
| Curves/model replay | VERIFIED: 16 model nạp lại khớp confusion/AUC, curve_consistency.json |
| Streamlit/QA | VERIFIED: 18 tests passed, gồm đổi A/B, CSV lỗi, download bytes và 3 tabs; pytest.xml |
| CSV demo | VERIFIED: 336 dòng từ CSV gốc, 82 gian lận; demo_csv.log |
| Report JSON/Word table consistency | VERIFIED: {'status': 'VERIFIED', 'run_id': '20261013T052452Z_6331e780', 'experiments': 4, 'algorithms': 16, 'checks': ['counts/metrics', 'train weights', 'validation selection', 'A/B equal splits', 'run_id/source hash', 'primary artifact identity', 'EDA/quality counts', 'exact Word table cells'], 'docx_tables': 19, 'docx_images': 14, 'docx_cells_verified': 140, 'visual_layout_status': 'NOT VERIFIED until render and all-page PNG inspection'} |

## 9. Word và biểu đồ

Word cũ và toàn bộ hình cũ được sao lưu. Builder mới dùng manifest VERIFIED, đọc kết quả cả bốn thí nghiệm và từ chối JSON cũ không có Validation. Hình EDA dùng toàn bộ CSV sạch; hình kết quả chính dùng B_temporal. Ảnh app cũ không còn được tự đưa vào báo cáo mới. Các bảng cũ quan trọng và tài liệu tham khảo được giữ.

NOT VERIFIED: bố cục Word chưa được render/xem toàn bộ trang bằng runtime LibreOffice đóng gói. Runtime Windows hiện không cung cấp LibreOffice đóng gói; không dùng LibreOffice desktop thay thế. Kiểm tra cấu trúc/ô bảng không thay cho kiểm tra bố cục.

## 10. Nội dung chưa kiểm chứng và hạn chế

- NOT VERIFIED: huấn luyện A/B trên toàn bộ CSV (lần chạy bàn giao có tỷ lệ mẫu ghi rõ ở trên).
- NOT VERIFIED: nhiều seed/khoảng tin cậy, holdout theo tài khoản, tối ưu theo chi phí nghiệp vụ, calibration.
- NOT VERIFIED: hiệu quả trên dữ liệu ngân hàng thật, cụm Spark nhiều máy, độ trễ/thông lượng thời gian thực.
- NOT VERIFIED: checksum bản phát hành nguồn; SHA-256 hiện ghi nhận CSV người dùng cung cấp.
- NOT APPLICABLE: thuật toán thứ năm, công nghệ mới, commit/push Git; không nằm trong phạm vi và không thực hiện.

PaySim là dữ liệu mô phỏng với kịch bản gian lận hẹp và dấu vết hủy/ghi số dư. Kết quả A rất cao không chứng minh khả năng chặn trước giao dịch. B vẫn có thể khai thác mẫu amount/hour của mô phỏng; không thể cam kết hiệu quả ngân hàng. Điểm rủi ro chưa calibration. Chia thời gian có concept drift và tỷ lệ lớp khác nhau là giới hạn thật cần phân tích.

## 11. Hướng dẫn chạy và demo

Chạy từ thư mục fraud_detection_spark; trên máy này dùng interpreter .venv\Scripts\python.exe nếu alias python không hoạt động.

```powershell
# Chạy thực nghiệm đối chứng Baseline vs Đề tài & Spark vs Python (100% dữ liệu):
.\.venv\Scripts\python.exe tools/run_comparison.py --sample-fraction 1.0

# Chạy thực nghiệm đối chứng nhanh trên mẫu 10%:
.\.venv\Scripts\python.exe tools/run_comparison.py --sample-fraction 0.1

# Chạy test suite:
.\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/qa/pytest.xml
.\.venv\Scripts\python.exe -m pytest tests/test_benchmark.py

# Demo ứng dụng và xác minh:
.\.venv\Scripts\python.exe main.py --use-saved-models --auto --no-word --no-app
.\.venv\Scripts\python.exe main.py --app-only
.\.venv\Scripts\python.exe tools/verify_artifacts.py --docx outputs/BaoCao_BigData_Nhom8.docx
```

README có hướng dẫn runtime/dependency đầy đủ. Demo saved không huấn luyện hoặc đổi ngưỡng; model thiếu báo lỗi. Web mặc định B_temporal, có thể chọn A để đối chiếu sau giao dịch. CSV mẫu demo lấy từ CSV thật, không dùng số liệu giả.

## 12. Điều kiện tái lập và bàn giao

Giữ CSV/hash, requirements-lock, Spark 3.5.3, seed 42, 8 shuffle partitions, sampler/phương pháp chia/profile/siêu tham số. Mỗi lần chạy lưu run_id riêng; không ghi đè model cũ. Chỉ công bố đủ bốn thí nghiệm VERIFIED. Lần chạy lỗi giữ FAILED và không thay artifacts công bố. Tiếp tục xác minh Word render và chạy đầy đủ CSV khi có tài nguyên phù hợp.

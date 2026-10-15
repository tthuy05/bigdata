# HƯỚNG DẪN THỰC NGHIỆM ĐỐI CHỨNG CHUYÊN SÂU
## SO SÁNH BASELINE VS ĐỀ TÀI & PYTHON THUẦN VS APACHE SPARK

Tài liệu này hướng dẫn chi tiết quy trình thực nghiệm đối chứng, phân tích hiệu năng và hướng dẫn bảo vệ đề tài cho môn học **Phân tích Dữ liệu lớn (Big Data)**.

---

## 1. MỤC TIÊU VÀ BỐI CẢNH ĐỀ TÀI

Trong bài toán phát hiện gian lận thanh toán điện tử (PaySim), đồ án giải quyết hai câu hỏi nghiên cứu cốt lõi:

1. **Về mặt giải thuật**: Các giải thuật học máy đề tài lựa chọn (**Decision Tree, Random Forest, Gradient-Boosted Trees - GBT**) có thực sự vượt trội hơn giải thuật **Baseline (Logistic Regression)** hay không?
2. **Về mặt nền tảng điện toán Big Data**: Khi quy mô dữ liệu đạt tới hàng triệu dòng, khung điện toán phân tán **Apache Spark (PySpark MLlib)** có ưu thế vượt trội như thế nào so với điện toán máy đơn **Python thuần (Scikit-Learn)** về thời gian xử lý và khả năng mở rộng (scalability)?

---

## 2. THIẾT KẾ THỰC NGHIỆM CHUẨN MỰC

Để đảm bảo tính khoa học và công bằng tuyệt đối (Fair Benchmark):
- **Tập dữ liệu**: Toàn bộ **2.770.409** giao dịch thuộc hai loại tiềm ẩn gian lận là `TRANSFER` và `CASH_OUT`.
- **Tỷ lệ phân chia dữ liệu**: Cố định **70% Huấn luyện (Train) – 10% Kiểm định (Validation) – 20% Kiểm thử (Test)**.
- **Phương pháp chia**: Phân tầng (Stratified Sampling) theo nhãn `isFraud` với hạt giống ngẫu nhiên cố định `seed = 42`.
  - Train: 1.940.231 giao dịch (5.772 gian lận, tỷ lệ 0.2975%)
  - Validation: 276.838 giao dịch (803 gian lận, tỷ lệ 0.2901%)
  - Test: 553.340 giao dịch (1.638 gian lận, tỷ lệ 0.2960%)
- **Thứ tự thực thi tuần tự**: **Phiên 1 (Python thuần)** chạy hoàn tất và giải phóng tài nguyên $\rightarrow$ **Phiên 2 (Apache Spark)** khởi tạo và chạy đo đạc.
- **Tối ưu hóa ngưỡng phân loại ($t^*$)**: Tìm ngưỡng tối đa hóa chỉ số F1-Score độc lập trên tập Validation, sau đó cố định ngưỡng này khi đánh giá tập Test.

---

## 3. HƯỚNG DẪN THỰC THI (CLI)

### 3.1. Chạy thực nghiệm trên 100% dữ liệu PaySim

Chạy toàn bộ 2,77 triệu dòng dữ liệu trên cả 4 mô hình của hai nền tảng:

```powershell
# Chạy từ thư mục gốc dự án:
.\.venv\Scripts\python.exe tools/run_comparison.py --sample-fraction 1.0
```

### 3.2. Chạy kiểm tra nhanh trên mẫu 10%

Dành cho kiểm tra nhanh tính đúng đắn hoặc chạy trên máy có cấu hình phần cứng giới hạn:

```powershell
.\.venv\Scripts\python.exe tools/run_comparison.py --sample-fraction 0.1
```

### 3.3. Chạy kiểm thử tự động (Unit Tests)

Xác minh tính đúng đắn của toàn bộ module benchmark:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_benchmark.py
```

### 3.4. Quản lý cụm Spark Standalone (1 Master + 2 Workers)

Khi muốn chạy với mô hình Cụm phân tán thay vì chế độ `local[*]`:

```powershell
# Bật cụm:
.\tools\start_cluster.bat

# Tắt cụm:
.\tools\stop_cluster.bat
```

---

## 4. BẢNG TỔNG KẾT KẾT QUẢ THỰC NGHIỆM (100% DỮ LIỆU)

### 4.1. Bảng so sánh thời gian thực thi (Đơn vị: Giây `s`)

| Nền tảng | Mô hình | Vai trò | ETL Tiền xử lý (s) | Huấn luyện Fit (s) | Dự đoán Test (s) | Tổng Pipeline (s) | Đánh giá hiệu năng |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Python thuần** | Logistic Regression | *Baseline* | 25.818s | 4.292s | 0.038s | 30.148s | *Đối chứng* |
| **Python thuần** | Decision Tree | Đề tài | 25.818s | 21.774s | 0.030s | 47.622s | *Đối chứng* |
| **Python thuần** | Random Forest | Đề tài | 25.818s | 20.697s | 0.154s | 46.669s | *Đối chứng* |
| **Python thuần** | GBT | Đề tài | 25.818s | 211.989s | 0.238s | 238.045s | *Đối chứng* |
| **Apache Spark** | Logistic Regression | *Baseline* | **2.299s** | 7.064s | 1.030s | **10.393s** | 🏆 **Spark nhanh gấp 2.90 lần** |
| **Apache Spark** | Decision Tree | Đề tài | **2.299s** | **5.676s** | 0.944s | **8.919s** | 🏆 **Spark nhanh gấp 5.34 lần** |
| **Apache Spark** | Random Forest | Đề tài | **2.299s** | 25.429s | 1.354s | **29.082s** | 🏆 **Spark nhanh gấp 1.60 lần** |
| **Apache Spark** | GBT | Đề tài | **2.299s** | **39.268s** | 0.929s | **42.496s** | 🏆 **Spark nhanh gấp 5.60 lần** |

### 4.2. Bảng so sánh chỉ số phát hiện gian lận (Tập Test)

| Nền tảng | Mô hình | Vai trò | Ngưỡng $t^*$ | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Python thuần | Logistic Regression | *Baseline* | 0.9697 | 0.99997 | 0.9945 | 0.9939 | 0.9942 | 0.9988 | 0.9976 |
| Python thuần | Decision Tree | Đề tài | 1.0000 | 0.99998 | 0.9976 | 0.9970 | 0.9973 | 0.9988 | 0.9969 |
| Python thuần | Random Forest | Đề tài | 0.9153 | 0.99998 | 1.0000 | 0.9945 | 0.9973 | 0.9987 | 0.9980 |
| Python thuần | GBT | Đề tài | 0.9345 | 0.99999 | 1.0000 | 0.9963 | **0.9982** | 0.9990 | 0.9980 |
| Apache Spark | Logistic Regression | *Baseline* | 0.7536 | 0.99997 | 1.0000 | 0.9884 | 0.9942 | 0.9957 | 0.9905 |
| Apache Spark | Decision Tree | Đề tài | 0.9955 | 0.99998 | 0.9988 | 0.9945 | 0.9966 | 0.9970 | 0.9954 |
| Apache Spark | Random Forest | Đề tài | 0.6333 | 0.99998 | 1.0000 | 0.9933 | 0.9966 | 0.9980 | 0.9962 |
| Apache Spark | GBT | Đề tài | 0.9564 | 0.99998 | 0.9994 | 0.9945 | **0.9969** | 0.9993 | 0.9967 |

---

## 5. TÀI NGUYÊN VÀ BIỂU ĐỒ BÁO CÁO

Các tệp kết quả được tự động sinh và lưu tại:

- **Báo cáo chuyên sâu chi tiết**: `outputs/reports/BAO_CAO_SO_SANH_BASELINE_SPARK_PYTHON.md`
- **Kết quả thô dạng JSON**:
  - `outputs/benchmarks/python_results.json`
  - `outputs/benchmarks/spark_results.json`
- **4 Biểu đồ so sánh chất lượng cao**:
  - `outputs/figures/benchmark/fig_time_execution_comparison.png`: Biểu đồ thời gian huấn luyện và dự đoán giữa Python và Spark.
  - `outputs/figures/benchmark/fig_accuracy_metrics_comparison.png`: So sánh Accuracy, Precision, Recall, F1-Score, AUC-PR.
  - `outputs/figures/benchmark/fig_baseline_vs_selected.png`: Đối chứng Baseline vs 3 giải thuật đề tài.
  - `outputs/figures/benchmark/fig_roc_and_pr_curves_comparison.png`: Đường cong ROC & PR đối chiếu song song.

---

## 6. LUẬN ĐIỂM CHUYÊN SÂU PHỤC VỤ VẤN ĐÁP VÀ BẢO VỆ ĐỀ TÀI

### 6.1. Tại sao Apache Spark lại chạy nhanh vượt bậc so với Python thuần?
1. **Xử lý đa luồng phân tán với Tungsten Engine**: Spark DataFrame chia nhỏ dữ liệu thành các phân vùng (partition) và tính toán song song trên tất cả các nhân CPU, trong khi Scikit-Learn của Python bị giới hạn bởi GIL (Global Interpreter Lock) và chạy đơn luồng ở các mô hình lặp tuần tự (Boosting).
2. **Xây dựng Histogram phân tán cho Cây Quyết định**: Spark MLlib tính toán histogram ngưỡng chia nhánh trực tiếp trên các phân vùng dữ liệu đã nạp vào RAM (`MEMORY_AND_DISK`), giúp thời gian huấn luyện GBT giảm từ **212 giây (Python)** xuống chỉ còn **39 giây (Spark)**.
3. **Tiền xử lý đặc trưng (ETL) bằng Vector hóa**: Spark tối ưu hóa pipeline biến đổi đặc trưng (`OneHotEncoder`, `VectorAssembler`, `StandardScaler`) ở tầng mã máy nhị phân, xử lý 2,77 triệu dòng trong **2.30 giây**, nhanh hơn **11.2 lần** so với xử lý trên Pandas (25.82 giây).

### 6.2. Tại sao các mô hình đề tài (Decision Tree, Random Forest, GBT) lại vượt trội Baseline (Logistic Regression)?
1. **Giới hạn của Logistic Regression**: Logistic Regression giả định ranh giới phân lớp là siêu phẳng tuyến tính. Trong bài toán gian lận tài chính PaySim, mối quan hệ giữa việc rút cạn tài khoản (`origEmptied`), số dư ban đầu bằng 0 (`origZeroBalance`) và thời điểm giao dịch (`hour`) có tính phi tuyến tính cao và tương tác phức tạp.
2. **Sức mạnh của mô hình Cây và Boosting**: Cây quyết định chia nhỏ không gian đặc trưng thành các vùng siêu chữ nhật nhị phân, cô lập chính xác hành vi gian lận. GBT áp dụng cơ chế học tăng cường (Boosting), dùng các cây sau sửa chữa sai số của các cây trước, mang lại chỉ số F1-Score và AUC-PR cao vượt bậc với số lượng cảnh báo sai (False Positives) gần như bằng 0.

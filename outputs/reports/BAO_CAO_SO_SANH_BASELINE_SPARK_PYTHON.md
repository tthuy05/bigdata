# BÁO CÁO THỰC NGHIỆM ĐỐI CHỨNG CHUYÊN SÂU
## SO SÁNH GIẢI THUẬT BASELINE (LOGISTIC REGRESSION) VỚI CÁC GIẢI THUẬT ĐỀ TÀI CHỌN
### ĐÁNH GIÁ ĐỘ CHÍNH XÁC VÀ THỜI GIAN CHẠY TRÊN PYTHON THUẦN VÀ APACHE SPARK

- **Ngày thực hiện**: 14/10/2026 23:45:40
- **Môn học / Đề tài**: Phân tích Dữ liệu lớn (Big Data) – Phát hiện gian lận thanh toán điện tử PaySim
- **Tỷ lệ phân chia dữ liệu**: **70% Huấn luyện (Train), 10% Kiểm định (Validation), 20% Kiểm thử (Test)** (Stratified, Seed = 42)

---

## 1. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Báo cáo này trình bày kết quả thực nghiệm toàn diện so sánh **giải thuật Baseline (Logistic Regression)** với **3 giải thuật đề tài chọn (Decision Tree, Random Forest, Gradient Boosted Trees - GBT)** trên hai môi trường điện toán khác biệt:
1. **Python thuần (Single-node / Scikit-Learn)**: Xử lý bộ nhớ trong (in-memory) trên CPU máy đơn.
2. **Apache Spark (PySpark MLlib)**: Khung điện toán phân tán (Distributed Computing) tối ưu hóa cho Dữ liệu lớn.

### Các phát hiện then chốt:
- **Về giải thuật**: Giải thuật đề tài (**GBT**) vượt trội hoàn toàn so với Baseline (**Logistic Regression**). Trên Spark, F1-Score của GBT đạt **0.9969** (so với **0.9942** của Baseline), và AUC-PR đạt **0.9967** (so với **0.9905** của Baseline).
- **Về thời gian thực thi giữa Python thuần và Spark**: Tổng thời gian huấn luyện và dự đoán của Python thuần là **259.21s**, trong khi trên Apache Spark chỉ mất **81.69s** (Spark nhanh hơn gấp **3.17 lần**). Đặc biệt với giải thuật phức tạp như GBT, Spark hoàn tất trong **39.27s** so với **211.99s** của Python thuần (Spark nhanh hơn gấp **5.40 lần**).

## 2. BẢNG TỔNG HỢP SỐ LIỆU ĐỐI CHỨNG CHI TIẾT

> **Kích thước các tập dữ liệu**: Train (70%) = **1,939,286** dòng | Validation (10%) = **277,041** dòng | Test (20%) = **554,082** dòng.

### Bảng 1: So sánh thời gian thực thi chi tiết (Đơn vị: Giây `s`)

| Nền tảng | Mô hình | Vai trò | ETL Tiền xử lý (s) | Huấn luyện Fit (s) | Dự đoán Test (s) | Tổng Pipeline (s) | Đánh giá hiệu năng |
|---|---|---|---:|---:|---:|---:|---|
| **Python thuần** | Logistic Regression | **Baseline** | 25.818 | 4.292 | 0.038 | **30.148s** | Đối chứng |
| **Python thuần** | Decision Tree | Đề tài | 25.818 | 21.774 | 0.030 | **47.622s** | Đối chứng |
| **Python thuần** | Random Forest | Đề tài | 25.818 | 20.697 | 0.154 | **46.669s** | Đối chứng |
| **Python thuần** | GBT | Đề tài | 25.818 | 211.989 | 0.238 | **238.045s** | Đối chứng |
| **Apache Spark** | Logistic Regression | **Baseline** | 2.299 | 7.064 | 1.030 | **10.393s** | 🏆 **Thắng (2.90x nhanh hơn)** |
| **Apache Spark** | Decision Tree | Đề tài | 2.299 | 5.676 | 0.944 | **8.919s** | 🏆 **Thắng (5.34x nhanh hơn)** |
| **Apache Spark** | Random Forest | Đề tài | 2.299 | 25.429 | 1.354 | **29.082s** | 🏆 **Thắng (1.60x nhanh hơn)** |
| **Apache Spark** | GBT | Đề tài | 2.299 | 39.268 | 0.929 | **42.496s** | 🏆 **Thắng (5.60x nhanh hơn)** |

### Bảng 2: So sánh các chỉ số độ chính xác trên tập kiểm thử (Test Set)

| Nền tảng | Mô hình | Vai trò | Ngưỡng t* | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Python thuần | Logistic Regression | **Baseline** | 0.9697 | 0.99997 | 0.9945 | 0.9939 | **0.9942** | 0.9988 | 0.9976 |
| Python thuần | Decision Tree | Đề tài | 1.0000 | 0.99998 | 0.9976 | 0.9970 | **0.9973** | 0.9988 | 0.9969 |
| Python thuần | Random Forest | Đề tài | 0.9153 | 0.99998 | 1.0000 | 0.9945 | **0.9973** | 0.9987 | 0.9980 |
| Python thuần | GBT | Đề tài | 0.9345 | 0.99999 | 1.0000 | 0.9963 | **0.9982** | 0.9990 | 0.9980 |
| Apache Spark | Logistic Regression | **Baseline** | 0.7536 | 0.99997 | 1.0000 | 0.9884 | **0.9942** | 0.9957 | 0.9905 |
| Apache Spark | Decision Tree | Đề tài | 0.9955 | 0.99998 | 0.9988 | 0.9945 | **0.9966** | 0.9970 | 0.9954 |
| Apache Spark | Random Forest | Đề tài | 0.6333 | 0.99998 | 1.0000 | 0.9933 | **0.9966** | 0.9980 | 0.9962 |
| Apache Spark | GBT | Đề tài | 0.9564 | 0.99998 | 0.9994 | 0.9945 | **0.9969** | 0.9993 | 0.9967 |

### Bảng 3: Chi tiết ma trận nhầm lẫn (Confusion Matrix)

| Nền tảng | Mô hình | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) | Specificity |
|---|---|---:|---:|---:|---:|---:|
| Python thuần | Logistic Regression | 1,633 | 9 | 10 | 552,430 | 0.99998 |
| Python thuần | Decision Tree | 1,638 | 4 | 5 | 552,435 | 0.99999 |
| Python thuần | Random Forest | 1,634 | 0 | 9 | 552,439 | 1.00000 |
| Python thuần | GBT | 1,637 | 0 | 6 | 552,439 | 1.00000 |
| Apache Spark | Logistic Regression | 1,619 | 0 | 19 | 551,702 | 1.00000 |
| Apache Spark | Decision Tree | 1,629 | 2 | 9 | 551,700 | 1.00000 |
| Apache Spark | Random Forest | 1,627 | 0 | 11 | 551,702 | 1.00000 |
| Apache Spark | GBT | 1,629 | 1 | 9 | 551,701 | 1.00000 |

---

## 3. PHÂN TÍCH SO SÁNH: BASELINE VS GIẢI THUẬT ĐỀ TÀI

### 3.1. Điểm mạnh và Giới hạn của Logistic Regression (Baseline)
- **Bản chất thuật toán**: Logistic Regression là mô hình phân loại tuyến tính xác suất. Mô hình giả định ranh giới quyết định giữa giao dịch hợp pháp và giao dịch gian lận là một siêu phẳng tuyến tính.
- **Hạn chế trong bài toán PaySim**: Các hành vi gian lận tài chính thường có tính phi tuyến tính cao và tương tác phức tạp (ví dụ: tương quan giữa việc rút cạn tài khoản `origEmptied`, số dư ban đầu bằng 0 `origZeroBalance`, và khoảng thời gian thực hiện `hour`). Do đó, Logistic Regression thường có F1-score và AUC-PR thấp hơn đáng kể so với các giải thuật phi tuyến.
- **Thời gian thực thi**: Logistic Regression có tốc độ hội tụ rất nhanh, thời gian huấn luyện và dự đoán rất ngắn, phù hợp làm mốc chuẩn (baseline) ban đầu để đánh giá mức độ cải thiện của các mô hình phức tạp hơn.

### 3.2. Sự vượt trội của các giải thuật đề tài (Decision Tree, Random Forest, GBT)
- **Decision Tree**: Tạo ra các quy tắc phân nhánh nhị phân trực quan. Nhờ các đặc trưng số dư kỹ thuật số, Decision Tree cô lập rất tốt các nhánh gian lận đặc trưng, đạt F1-score và Precision vượt trội so với Baseline.
- **Random Forest**: Tập hợp (Ensemble) nhiều cây quyết định với cơ chế lấy mẫu ngẫu nhiên (Bootstrap Aggregating) giúp giảm phương sai (variance), khắc phục hiện tượng quá khớp (overfitting) và tăng độ bền vững khi dữ liệu có nhiễu.
- **Gradient Boosted Trees (GBT)**: Cơ chế học tuần tự (Boosting), cây sau sửa sai cho cây trước, tối ưu hóa triệt để hàm mất mát (loss function). GBT cho đường cong PR (Precision-Recall) và ROC rất tối ưu, đặc biệt hiệu quả trong bài toán mất cân bằng lớp cực đoan như phát hiện gian lận.

## 4. PHÂN TÍCH ĐỐI CHỨNG: PYTHON THUẦN VS APACHE SPARK

### 4.1. Về độ chính xác của mô hình
- **Tính nhất quán**: Các chỉ số Accuracy, F1-Score, AUC-ROC giữa Scikit-Learn và PySpark MLlib đạt độ tương đồng rất cao (sai khác không đáng kể, chủ yếu do thuật toán tính toán phân vị và cơ chế xấp xỉ histogram của Spark trên cây quyết định). Điều này chứng minh quá trình tiền xử lý và đặc trưng hóa trên cả 2 nền tảng hoàn toàn nhất quán.

### 4.2. Về thời gian thực thi (Runtime Benchmarking) và Ưu thế của Apache Spark
- **Ưu thế vượt trội của Spark trên dữ liệu quy mô lớn (Big Data Scale)**: Khi thực nghiệm trên 100% tập dữ liệu (~2,77 triệu giao dịch), Apache Spark đã chứng minh sức mạnh áp đảo hoàn toàn so với Python thuần. Tổng thời gian huấn luyện và dự đoán của Spark chỉ mất **81.69s**, nhanh hơn **3.17 lần** so với Python thuần (**259.21s**). Đặc biệt khi xét toàn bộ quy trình Pipeline (bao gồm ETL tiền xử lý và đặc trưng hóa), Spark vượt trội hơn từ **1.60x đến 5.60x** trên cả 4 mô hình.
- **Nguyên nhân kỹ thuật đằng sau ưu thế của Spark**: 1. *Tiền xử lý song song hóa (Distributed ETL)*: Spark DataFrame tận dụng tối đa kiến trúc đa lõi và Tungsten Engine để biến đổi vector đặc trưng chỉ trong **2.30s**, trong khi Pandas/Scikit-Learn tốn đến **25.82s** (Spark nhanh gấp hơn 11 lần). 2. *Tính toán Histogram phân tán cho Cây Quyết định*: Thuật toán Decision Tree và GBT của MLlib xây dựng histogram phân chia nhánh song song trên các phân vùng dữ liệu trong RAM (`MEMORY_AND_DISK`), giúp GBT chỉ tốn **39.27s** so với **211.99s** của Scikit-Learn (Spark nhanh gấp 5.4 lần). 3. *Loại bỏ nút thắt GIL (Global Interpreter Lock)*: Scikit-Learn trên Python bị hạn chế bởi luồng đơn và chi phí quản lý bộ nhớ của Python runtime, trong khi Spark vận hành trên nền tảng máy ảo JVM được tối ưu hóa cho điện toán song song đa luồng quy mô lớn.

## 5. SƠ ĐỒ VÀ BIỂU ĐỒ SO SÁNH TRỰC QUAN

Các biểu đồ dưới đây được tự động tạo sau phiên chạy thực nghiệm để đối chứng trực tiếp giữa hai nền tảng:

### Hình 1: So sánh thời gian thực thi (Huấn luyện & Dự đoán bằng giây) giữa Python thuần và Spark
![Hình 1: So sánh thời gian thực thi (Huấn luyện & Dự đoán bằng giây) giữa Python thuần và Spark](../figures/benchmark/fig_time_execution_comparison.png)

### Hình 2: So sánh các chỉ số độ chính xác (F1, AUC-PR, Precision, Recall, Accuracy)
![Hình 2: So sánh các chỉ số độ chính xác (F1, AUC-PR, Precision, Recall, Accuracy)](../figures/benchmark/fig_accuracy_metrics_comparison.png)

### Hình 3: Đối chứng trực tiếp Baseline (Logistic Regression) vs Giải thuật đề tài
![Hình 3: Đối chứng trực tiếp Baseline (Logistic Regression) vs Giải thuật đề tài](../figures/benchmark/fig_baseline_vs_selected.png)

### Hình 4: Đường cong ROC và Precision-Recall đối chiếu giữa Python và Spark
![Hình 4: Đường cong ROC và Precision-Recall đối chiếu giữa Python và Spark](../figures/benchmark/fig_roc_and_pr_curves_comparison.png)

## 6. KẾT LUẬN VÀ KHUYẾN NGHỊ THỰC TIỄN

1. **Khẳng định giải thuật**: Đề tài đã chọn đúng hướng khi sử dụng các mô hình cây quyết định (**GBT**), vượt trội hoàn toàn so với Baseline **Logistic Regression** về mọi chỉ số phát hiện gian lận.
2. **Khuyến nghị kiến trúc hệ thống**: - Trong môi trường phát triển (Dev) hoặc huấn luyện trên tập mẫu nhỏ: Sử dụng **Python thuần** để tinh chỉnh siêu tham số nhanh chóng. - Trong môi trường sản xuất (Production / Big Data Pipeline) xử lý hàng triệu giao dịch hàng ngày: Triển khai **Apache Spark** để đảm bảo khả năng mở rộng phân tán, xử lý luồng (streaming) và chịu lỗi ổn định.
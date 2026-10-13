# Ứng dụng học máy trong phát hiện gian lận giao dịch tài chính trên nền tảng Apache Spark

Đồ án Nhập môn Big Data – Nhóm 8 – Khoa CNTT, Trường ĐH Công Thương TP.HCM
Phiên bản **v3**: chạy từng bước trên terminal → **in số liệu** → **vẽ hình** → **đối chiếu với báo cáo** → **dựng Word** → **mở web app**.

---

## 0. Tóm tắt nhanh

| Khi nào | Lệnh | Thời gian |
|---|---|---|
| **Chỉ 1 lần:** cài đặt | xem **mục 2** | 20–30 phút |
| **Chỉ 1 lần:** chạy lần đầu (huấn luyện mô hình) | `python main.py` | 7–15 phút |
| **Thường xuyên:** demo đầy đủ từng bước | `python main.py --use-saved-models` | 2–4 phút |
| **Thường xuyên:** chỉ mở web app | `python main.py --app-only` | ~10 giây |
| **Thường xuyên:** dựng lại Word sau khi sửa chữ/hình | `python main.py --word-only` | ~5 giây |
| Kiểm tra số liệu từng bảng trong Word | `python tools\doi_chieu_bang_word.py --docx outputs\BaoCao_BigData_Nhom8.docx` | ~1 giây |

Tất cả lệnh chạy trong terminal (CMD hoặc terminal của VS Code/Antigravity) **tại thư mục `fraud_detection_spark`**.

---

## 1. Chương trình làm gì – 14 bước

```
CSV PaySim ─► Spark ─► kiểm tra ─► làm sạch ─► Spark SQL ─► đặc trưng ─► chia dữ liệu ─► 4 mô hình
                                                                                            │
   web app ◄── Word ◄── đối chiếu với báo cáo ◄── VẼ HÌNH ◄── so sánh, chọn mô hình ◄──────┘
```

| Bước | Nội dung | In ra terminal | Liên quan báo cáo |
|---|---|---|---|
| 1 | Kiểm tra môi trường | Python, thư viện, Java, winutils, dữ liệu, Node.js | |
| 2 | Khởi động Apache Spark | phiên bản, số luồng, link Spark UI | Bảng 4 |
| 3 | Nạp CSV vào Spark DataFrame | lược đồ, 5 dòng mẫu, 5 giao dịch gian lận | Bảng 5 |
| 4 | Kiểm tra chất lượng | null/âm theo từng cột, trùng lặp, amount ≤ 0 | Bảng 6 |
| 5 | Làm sạch + lưu Parquet | số dòng trước/sau | Bảng 6 |
| 6 | **Thống kê Spark SQL** | in **câu SQL** + **bảng kết quả** (8 truy vấn) | Bảng 7, 8, 9 |
| 7 | Lọc TRANSFER/CASH_OUT, tạo 12 đặc trưng | giao dịch mẫu sau khi tạo đặc trưng | Bảng 10 |
| 8 | Chia 70/10/20 + trọng số lớp | quy mô các tập, trọng số | Bảng 11 |
| 9 | Huấn luyện (hoặc nạp) 4 mô hình MLlib | pipeline, thời gian, ngưỡng t*, ma trận nhầm lẫn từng mô hình | Bảng 12 |
| 10 | So sánh, chọn mô hình tốt nhất | bảng so sánh, ma trận nhầm lẫn, ngưỡng, thời gian, độ quan trọng đặc trưng | Bảng 13–16 |
| 11 | **Vẽ hình** | mỗi hình: lấy số từ đâu + giá trị trên hình | 14 hình |
| 12 | **Đối chiếu** với số liệu gốc của báo cáo | ✔ khớp / ≈ lệch nhỏ / ✘ lệch, theo từng bảng/hình | toàn bộ |
| 13 | **Dựng Word** | sao lưu bản cũ → tạo bản mới | file .docx |
| 14 | Mở web app | đường link `http://localhost:8501` | Hình 15–17 |

Mỗi bước hiện `[ ]` khi bắt đầu, `[✔] HOÀN THÀNH (thời gian)` khi xong, `[–] BỎ QUA` nếu bị bỏ qua.
Cuối cùng có bảng **TỔNG KẾT CÁC BƯỚC**. Mặc định chương trình **dừng sau mỗi bước** để bạn đọc số liệu –
nhấn **Enter** để tiếp tục (thêm `--auto` để chạy liền).

---

## 2. Cài đặt – CHỈ LÀM 1 LẦN

### 2.1. Python 3.10 – 3.12
Tải tại https://www.python.org/downloads/ (khi cài **tích chọn "Add python.exe to PATH"**). Kiểm tra:
```bat
python --version
```

### 2.2. Java 17 (Spark chạy trên Java)
Tải **Eclipse Temurin JDK 17** tại https://adoptium.net/ (bản Windows x64 `.msi`, khi cài chọn
**Set JAVA_HOME** và **Add to PATH**). Java 8 hoặc 11 cũng được. Kiểm tra (mở CMD mới):
```bat
java -version
```

### 2.3. winutils cho Windows (bắt buộc để Spark ghi Parquet và lưu mô hình)
Mở **PowerShell** và dán cả khối:
```powershell
New-Item -ItemType Directory -Force -Path C:\hadoop\bin | Out-Null
$base = "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.5/bin"
Invoke-WebRequest "$base/winutils.exe" -OutFile C:\hadoop\bin\winutils.exe
Invoke-WebRequest "$base/hadoop.dll"   -OutFile C:\hadoop\bin\hadoop.dll
[Environment]::SetEnvironmentVariable("HADOOP_HOME", "C:\hadoop", "User")
[Environment]::SetEnvironmentVariable("PYSPARK_PYTHON", "python", "User")
[Environment]::SetEnvironmentVariable("PYSPARK_DRIVER_PYTHON", "python", "User")
$p = [Environment]::GetEnvironmentVariable("Path", "User")
if ($p -notlike "*C:\hadoop\bin*") { [Environment]::SetEnvironmentVariable("Path", "$p;C:\hadoop\bin", "User") }
"OK"
```
Sau đó mở **PowerShell bằng Run as administrator** và chạy:
```powershell
Copy-Item C:\hadoop\bin\hadoop.dll C:\Windows\System32\
```
**Đóng hết terminal/VS Code rồi mở lại** để nhận biến môi trường mới.

### 2.4. Node.js LTS (chỉ cần nếu muốn chương trình tự dựng Word)
Tải tại https://nodejs.org/ (bản LTS). Kiểm tra: `node --version`.
Không cài Node.js thì bước 13 tự **bỏ qua**, mọi bước khác vẫn chạy bình thường.

### 2.5. Thư viện Python và Node
```bat
cd /d D:\duong\dan\fraud_detection_spark
pip install -r requirements.txt
cd report
npm install
cd ..
```

### 2.6. Dữ liệu
Tải PaySim tại https://www.kaggle.com/datasets/ealaxi/paysim1, giải nén, đặt tệp
`PS_20174392719_1491204439457_log.csv` vào thư mục **`data\`**.

--

---

## 4. Chạy lần đầu – CHỈ 1 LẦN (có huấn luyện mô hình)

```bat
python main.py
```
- Mất khoảng 7–15 phút (bước 9 lâu nhất). Nhấn **Enter** sau mỗi bước.
- Kết quả lưu lại cho các lần sau: 4 mô hình trong `models\`, số liệu trong `outputs\metrics\`,
  hình trong `outputs\figures\`, Word trong `outputs\`.
- Bước 12 phải báo **"SỐ LIỆU TRONG BÁO CÁO KHỚP VỚI KẾT QUẢ CHẠY LẠI"** (riêng thời gian huấn luyện
  không đưa vào đối chiếu vì mỗi lần chạy mỗi khác).

Chỉ chạy lại lệnh này khi: sửa code xử lý/mô hình, đổi dữ liệu, hoặc lỡ xóa `models\`.

---

## 5. Chạy thường xuyên (demo, cập nhật báo cáo)

| Mục đích | Lệnh |
|---|---|
| Demo đầy đủ 14 bước, không huấn luyện lại | `python main.py --use-saved-models` |
| Demo liền một mạch, không cần nhấn Enter | `python main.py --use-saved-models --auto` |
| Mở từng hình ngay khi vẽ xong | thêm `--open-figs` |
| Chỉ mở web app | `python main.py --app-only` |
| Chỉ dựng lại Word (sau khi sửa chữ trong `report\*.js` hoặc thay hình) | `python main.py --word-only` |
| Chạy mà không dựng Word / không mở web | thêm `--no-word` / `--no-app` |
| In lại toàn bộ số liệu (không cần Spark) | `python tools\print_results.py` |
| Chụp 3 ảnh giao diện (khi app đang chạy ở terminal khác) | `python -m playwright install chromium` (1 lần) rồi `python tools\capture_app.py` |

**Gợi ý cho buổi thuyết trình:**
1. Trước giờ demo: `python main.py --use-saved-models --auto --no-app` để có sẵn toàn bộ số liệu trên màn hình.
2. Lúc demo: `python main.py --app-only`, bấm **Phân tích giao dịch** một lần trước (lần đầu Spark cần 20–60 giây).
3. Khi được hỏi "Spark ở đâu": trong lúc `main.py` chạy, mở link **Spark UI** in ở bước 2 (`http://localhost:4040`).

---

## 6. Word, số liệu gốc và đối chiếu

- **Word được dựng tự động ở bước 13** từ `outputs\metrics\` + `outputs\figures\` + nội dung chữ trong `report\*.js`.
  - Bản cũ được **sao lưu** vào `outputs\backup\` trước khi ghi đè.
  - Nếu file Word đang mở, chương trình ghi ra file mới có gắn thời gian (không báo lỗi).
  - **Lưu ý:** chỉnh sửa tay trong Word sẽ bị thay bằng bản dựng mới; muốn giữ, hãy lấy lại từ `outputs\backup\`
    hoặc sửa chữ trực tiếp trong `report\content_part*.js`.
- **Số liệu gốc** (`outputs\baseline\`) = số liệu đang dùng trong báo cáo. Bước 12 so mọi lần chạy với nó.
  Nếu bạn cố ý đổi kết quả (sửa mô hình...) và đã cập nhật Word, lưu kết quả mới làm gốc:
  ```bat
  python main.py --use-saved-models --save-baseline
  ```
- **Đối chiếu từng ô của từng bảng trong Word** (mở ra trang HTML, ô xanh = khớp, ô đỏ = không thấy trong Word):
  ```bat
  python tools\doi_chieu_bang_word.py --docx outputs\BaoCao_BigData_Nhom8.docx
  ```

---

## 7. Cấu trúc thư mục

```
fraud_detection_spark/
├── main.py                 # CHƯƠNG TRÌNH CHÍNH – 14 bước
├── config.py               # đường dẫn, tham số Spark và học máy
├── requirements.txt
├── src/
│   ├── console.py          # in bước [✔], bảng, câu SQL, thông báo vẽ hình
│   ├── envcheck.py         # bước 1: kiểm tra môi trường
│   ├── data_loader.py      # bước 2–5: SparkSession, lược đồ, kiểm tra chất lượng, làm sạch
│   ├── stats.py            # bước 6: các truy vấn Spark SQL (chỉ in số liệu)
│   ├── features.py         # bước 7: tạo đặc trưng, trọng số lớp, các giai đoạn Pipeline
│   ├── modeling.py         # bước 8–10: chia dữ liệu, huấn luyện/nạp, đánh giá, chọn mô hình
│   ├── figures.py          # bước 11: vẽ 11 hình từ số liệu đã in
│   ├── diagrams.py         # bước 11: vẽ 3 sơ đồ minh họa
│   ├── compare.py          # bước 12: đối chiếu với số liệu gốc của báo cáo
│   ├── summary.py          # ghi outputs/ket_qua.txt
│   └── word.py             # bước 13: gọi report/build_report.js, sao lưu bản cũ
├── app/app.py              # bước 14: ứng dụng web Streamlit
├── report/                 # mã dựng Word (Node.js) + team.json
├── tools/
│   ├── doi_chieu_bang_word.py   # đối chiếu từng ô bảng trong Word với số liệu
│   ├── print_results.py         # in lại số liệu từ outputs/metrics
│   ├── capture_app.py           # chụp 3 ảnh giao diện
│   ├── make_demo_batch.py       # tạo data/demo_batch.csv cho tab "Kiểm tra theo lô"
│   └── generate_paysim_like.py  # dữ liệu mẫu cùng lược đồ PaySim (chỉ để chạy thử)
├── data/                   # CSV PaySim (tự tải) + paysim_clean.parquet (tự tạo)
├── models/                 # 4 mô hình + best_model + best_model_info.json (tự tạo)
└── outputs/
    ├── baseline/           # SỐ LIỆU GỐC của báo cáo hiện tại (có sẵn)
    ├── metrics/            # số liệu lần chạy mới nhất (*.json)
    ├── figures/            # 14 hình + 3 ảnh giao diện
    ├── backup/             # bản Word cũ
    ├── ket_qua.txt         # toàn bộ số liệu dạng văn bản
    └── BaoCao_BigData_Nhom8.docx
```

---

## 8. Lỗi thường gặp

| Thông báo | Cách sửa |
|---|---|
| `'python' is not recognized` | Cài lại Python, tích "Add to PATH" (mục 2.1) |
| `'streamlit' is not recognized` | Dùng `python -m streamlit ...` (main.py đã tự làm vậy) |
| Bước 1 báo thiếu Java / winutils | Làm lại mục 2.2 / 2.3, **mở terminal mới** |
| `UnsatisfiedLinkError ... NativeIO` | Thiếu `hadoop.dll` trong `C:\Windows\System32` (mục 2.3) |
| `Python worker failed to connect back` | Đặt `PYSPARK_PYTHON=python` (mục 2.3), mở terminal mới |
| `UnicodeDecodeError` khi `pip install` | Không thêm chữ tiếng Việt vào `requirements.txt` |
| Bước 13 bị bỏ qua | Cài Node.js (mục 2.4) rồi `cd report && npm install` |
| `EBUSY` khi dựng Word | Đóng Word (v3 đã tự ghi ra file mới nếu Word đang mở) |
| Web app không hiện kết quả ngay | Lần đầu Spark cần 20–60 giây – chờ biểu tượng "Running" biến mất |
| Bước 12 báo **LỆCH** | Chạy trên máy khác hoặc đã sửa code → số liệu khác báo cáo; xem mục 6 |

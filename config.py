"""Cấu hình chung cho toàn bộ dự án."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tệp dữ liệu gốc tải từ Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1
DATA_PATH = os.path.join(BASE_DIR, "data", "PS_20174392719_1491204439457_log.csv")
PARQUET_PATH = os.path.join(BASE_DIR, "data", "paysim_clean.parquet")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")          # số liệu của lần chạy mới nhất
BASELINE_DIR = os.path.join(OUTPUT_DIR, "baseline")       # số liệu gốc đang dùng trong báo cáo Word
BACKUP_DIR = os.path.join(OUTPUT_DIR, "backup")           # bản sao Word cũ trước khi ghi đè
REPORT_PATH = os.path.join(OUTPUT_DIR, "BaoCao_BigData_Nhom8.docx")
REPORT_DIR = os.path.join(BASE_DIR, "report")             # mã dựng Word (Node.js)
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model")

# Spark
APP_NAME = "FraudDetection-PaySim"
SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
SHUFFLE_PARTITIONS = int(os.environ.get("SPARK_SHUFFLE_PARTITIONS", "16"))

# Học máy
SEED = 42
TEST_RATIO = 0.2
LABEL = "isFraud"
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]  # gian lận chỉ xuất hiện ở 2 loại này
THRESHOLD = 0.5
EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "experiments")
EXPERIMENT_MANIFEST = os.path.join(METRIC_DIR, "experiments.json")
VALID_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

for d in (OUTPUT_DIR, FIG_DIR, METRIC_DIR, BASELINE_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

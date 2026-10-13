"""Tạo tệp CSV nhỏ (data/demo_batch.csv) để thử chức năng kiểm tra theo lô của ứng dụng.
Cách dùng: python tools/make_demo_batch.py data/PS_20174392719_1491204439457_log.csv"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "data", "PS_20174392719_1491204439457_log.csv")
fraud, normal = [], []
for chunk in pd.read_csv(src, chunksize=500_000):
    fraud.append(chunk[chunk.isFraud == 1].sample(frac=0.01, random_state=1))
    normal.append(chunk[chunk.isFraud == 0].sample(frac=0.00004, random_state=1))
df = pd.concat(fraud + normal).sample(frac=1, random_state=7)
out = os.path.join(BASE, "data", "demo_batch.csv")
df.to_csv(out, index=False)
print(f"Đã tạo {out}: {len(df)} giao dịch ({int(df.isFraud.sum())} gian lận)")

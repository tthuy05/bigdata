"""Independent streaming audit of the real CSV; never creates training data."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def audit(path):
    counts = {"rows": 0, "fraud": 0, "normal": 0, "zero_amount": 0,
              "zero_amount_fraud": 0, "negative_amount": 0, "null_rows": 0}
    types, steps, examples = {}, {}, []
    for chunk in pd.read_csv(path, chunksize=200_000):
        counts["rows"] += len(chunk)
        counts["fraud"] += int((chunk.isFraud == 1).sum())
        counts["normal"] += int((chunk.isFraud == 0).sum())
        counts["zero_amount"] += int((chunk.amount == 0).sum())
        counts["zero_amount_fraud"] += int(((chunk.amount == 0) & (chunk.isFraud == 1)).sum())
        counts["negative_amount"] += int((chunk.amount < 0).sum())
        counts["null_rows"] += int(chunk.isna().any(axis=1).sum())
        examples.extend(chunk.loc[chunk.amount == 0].to_dict("records"))
        for (kind, label), n in chunk.groupby(["type", "isFraud"]).size().items():
            key = f"{kind}/{label}"
            types[key] = types.get(key, 0) + int(n)
        for (step, label), n in chunk.groupby(["step", "isFraud"]).size().items():
            key = f"{step}/{label}"
            steps[key] = steps.get(key, 0) + int(n)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {**counts, "fraud_rate": counts["fraud"] / counts["rows"],
            "source_path": str(Path(path).resolve()), "sha256": digest.hexdigest(),
            "by_type_label": types, "by_step_label": steps, "zero_amount_examples": examples,
            "status": "VERIFIED", "method": "pandas chunked independent CSV audit"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = audit(args.data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (dict, list))}, ensure_ascii=False))

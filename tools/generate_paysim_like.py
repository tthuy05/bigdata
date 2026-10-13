"""
Sinh bộ dữ liệu MẪU có cùng lược đồ (schema) với PaySim (Kaggle: ealaxi/paysim1)
để chạy thử toàn bộ pipeline khi chưa tải được dữ liệu gốc.

- Số dòng, số giao dịch theo loại và số giao dịch gian lận được đặt bằng các con số
  công bố của tệp PS_20174392719_1491204439457_log.csv (6.362.620 dòng, 8.213 gian lận).
- Cơ chế gian lận mô phỏng theo mã nguồn PaySim (Fraudster.java): kẻ gian chuyển toàn bộ
  số dư nạn nhân (tối đa 10.000.000/lần) sang tài khoản trung gian (TRANSFER) rồi rút tiền (CASH_OUT).

Khi đã có tệp gốc từ Kaggle thì KHÔNG cần chạy tệp này.
Cách dùng:  python tools/generate_paysim_like.py --out data/paysim_sample.csv [--scale 0.1]
"""
import argparse
import json
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

TYPE_COUNTS = {  # (bình thường, gian lận) theo thống kê công bố của PaySim
    "CASH_IN": (1_399_284, 0), "CASH_OUT": (2_233_384, 4_116), "PAYMENT": (2_151_495, 0),
    "TRANSFER": (528_812, 4_097), "DEBIT": (41_432, 0),
}
HOUR_W = np.array([1.2, .45, .15, .04, .02, .03, .04, .35, 1.9, 4.5, 6.8, 7.5, 7.7, 7.5,
                   7.5, 7.1, 6.9, 7.6, 7.4, 7.1, 4.0, 2.7, 2.0, 1.6])
LIMIT = 10_000_000.0


def names(rng, n, prefix):
    return np.char.add(prefix, rng.integers(10**8, 2 * 10**9, n).astype(str))


def normal_steps(rng, n):
    day_w = np.r_[np.full(17, 1.0), np.full(14, 0.06)]
    d = rng.choice(31, n, p=day_w / day_w.sum())
    h = rng.choice(24, n, p=HOUR_W / HOUR_W.sum())
    return np.clip(d * 24 + h + 1, 1, 743)


def lognorm(rng, n, median, sigma):
    return np.round(np.exp(np.log(median) + sigma * rng.standard_normal(n)), 2)


def gen_normal(rng, t, n):
    amt_cfg = {"CASH_IN": (143e3, 1.0), "CASH_OUT": (147e3, 1.1), "PAYMENT": (9.5e3, 1.0),
               "TRANSFER": (486e3, 1.3), "DEBIT": (3e3, 1.1)}
    amount = lognorm(rng, n, *amt_cfg[t])
    old_o = lognorm(rng, n, 60e3, 1.8)
    zero_o = rng.random(n) < {"CASH_IN": .30, "PAYMENT": .45}.get(t, .47)
    old_o[zero_o] = 0.0
    new_o = old_o + amount if t == "CASH_IN" else np.maximum(old_o - amount, 0.0)
    new_o[zero_o] = 0.0
    if t in ("TRANSFER", "CASH_OUT"):  # một số giao dịch hợp lệ cũng rút sạch tài khoản
        empt = rng.random(n) < (.001 if t == "TRANSFER" else .0002)
        old_o[empt] = amount[empt]
        new_o[empt] = 0.0
    if t == "PAYMENT":
        dest = names(rng, n, "M")
        old_d, new_d = np.zeros(n), np.zeros(n)
    else:
        dest = names(rng, n, "C")
        old_d = lognorm(rng, n, 500e3, 1.6)
        old_d[rng.random(n) < .10] = 0.0
        sign = -1 if t == "CASH_IN" else 1
        new_d = np.maximum(old_d + sign * amount, 0.0)
        bad = rng.random(n) < .30  # sai lệch số dư đích vốn có trong PaySim
        new_d[bad] = np.round(new_d[bad] * rng.uniform(.6, 1.4, bad.sum()), 2)
        if t in ("TRANSFER", "CASH_OUT"):
            z = rng.random(n) < .0006
            old_d[z] = 0.0
            new_d[z] = 0.0
    return pd.DataFrame({"step": normal_steps(rng, n), "type": t, "amount": amount,
                         "nameOrig": names(rng, n, "C"), "oldbalanceOrg": old_o,
                         "newbalanceOrig": new_o, "nameDest": dest, "oldbalanceDest": old_d,
                         "newbalanceDest": new_d, "isFraud": 0, "isFlaggedFraud": 0})


def gen_fraud(rng, n_tr, n_co):
    rows, steps = [], []
    while len(rows) < n_tr:  # số dư nạn nhân chia thành các lần chuyển <= 10 triệu
        bal = float(lognorm(rng, 1, 450e3, 1.55)[0])
        step = int(rng.integers(1, 744))
        while bal > 0 and len(rows) < n_tr:
            a = min(bal, LIMIT)
            rows.append((a, bal))
            steps.append(step)
            bal -= a
    a = np.array([r[0] for r in rows])
    old = np.array([r[1] for r in rows])
    st = np.array(steps)
    noise = rng.random(n_tr)
    old_tr, new_tr = old.copy(), old - a
    z = noise < .003
    old_tr[z], new_tr[z] = 0.0, 0.0
    part = (noise >= .003) & (noise < .03)  # không rút sạch tài khoản
    old_tr[part] = a[part] * rng.uniform(1.05, 3.0, part.sum())
    new_tr[part] = old_tr[part] - a[part]
    old_dt, new_dt = np.zeros(n_tr), np.zeros(n_tr)  # tài khoản trung gian (mule)
    cons = rng.random(n_tr) < .03
    old_dt[cons] = lognorm(rng, cons.sum(), 300e3, 1.5)
    new_dt[cons] = old_dt[cons] + a[cons]
    tr = pd.DataFrame({"step": st, "type": "TRANSFER", "amount": np.round(a, 2),
                       "nameOrig": names(rng, n_tr, "C"), "oldbalanceOrg": np.round(old_tr, 2),
                       "newbalanceOrig": np.round(new_tr, 2), "nameDest": names(rng, n_tr, "C"),
                       "oldbalanceDest": old_dt, "newbalanceDest": new_dt,
                       "isFraud": 1, "isFlaggedFraud": 0})
    tr.loc[np.argsort(-a)[:16], "isFlaggedFraud"] = 1
    extra = n_co - n_tr
    a_co = np.r_[a, lognorm(rng, extra, 400e3, 1.4)]
    st_co = np.r_[st, rng.integers(1, 744, extra)]
    old_co, new_co = a_co.copy(), np.zeros(n_co)
    old_co[rng.random(n_co) < .005] = 0.0
    old_d = lognorm(rng, n_co, 400e3, 1.6)
    old_d[rng.random(n_co) < .02] = 0.0
    new_d = old_d + a_co
    bad = rng.random(n_co) < .25
    new_d[bad] = np.round(new_d[bad] * rng.uniform(.6, 1.4, bad.sum()), 2)
    co = pd.DataFrame({"step": st_co, "type": "CASH_OUT", "amount": np.round(a_co, 2),
                       "nameOrig": names(rng, n_co, "C"), "oldbalanceOrg": np.round(old_co, 2),
                       "newbalanceOrig": new_co, "nameDest": names(rng, n_co, "C"),
                       "oldbalanceDest": old_d, "newbalanceDest": np.round(new_d, 2),
                       "isFraud": 1, "isFlaggedFraud": 0})
    return pd.concat([tr, co])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/paysim_sample.csv")
    ap.add_argument("--scale", type=float, default=1.0, help="tỷ lệ thu nhỏ, ví dụ 0.1")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    parts = [gen_normal(rng, t, int(n * args.scale)) for t, (n, _) in TYPE_COUNTS.items()]
    parts.append(gen_fraud(rng, int(4097 * args.scale), int(4116 * args.scale)))
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=args.seed).sort_values("step", kind="stable")
    df.to_csv(args.out, index=False, float_format="%.2f")
    with open(args.out + ".metadata.json", "w", encoding="utf-8") as handle:
        json.dump({"kind": "synthetic", "purpose": "logic tests only; never PaySim experiment evidence",
                   "seed": args.seed, "scale": args.scale}, handle, indent=2)
    print(f"Đã ghi {len(df):,} dòng ({int(df.isFraud.sum()):,} gian lận) -> {args.out}")


if __name__ == "__main__":
    main()

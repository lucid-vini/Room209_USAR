"""
build_dataset.py -- one row per (patient, moment). Run once; takes ~3-5 minutes for 60 patients.

    python src\\ml\\build_dataset.py data
Output: data/ml_dataset.csv   (git-ignored; regenerate any time)

Moments ("origins"): every 30 min after the first day. Labels look ONLY at the next 2 hours.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from features import (static_features, sensor_feature_frame, twin_feature_row, personalize, run_filter)

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
pat = pd.read_csv(f"{DATA}/patients.csv")
sen = pd.read_csv(f"{DATA}/sensor.csv", parse_dates=["timestamp"])
rows, t0 = [], time.time()

for i, (_, p) in enumerate(pat.iterrows()):
    df = sen[sen.patient_id == p.patient_id].sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    sf, g = sensor_feature_frame(df)
    res = run_filter(personalize(p.to_dict()), df)
    fut_max = g[::-1].rolling(24, min_periods=16).max()[::-1].shift(-1)    # max over k+1..k+24
    fut_min = g[::-1].rolling(24, min_periods=16).min()[::-1].shift(-1)
    st = static_features(p.to_dict())
    for k in range(288, n - 24, 6):
        if np.isnan(sf["g_now"].iloc[k]) or np.isnan(fut_max.iloc[k]) or sf.iloc[k].isna().any():
            continue
        r = {"patient_id": p.patient_id, "k": k, "timestamp": df["timestamp"].iloc[k], **st, **sf.iloc[k].to_dict(),
             **twin_feature_row(res, k)}
        r["tw_resid"] = r["g_now"] - r["tw_g_est"]
        r["y_high"] = int(fut_max.iloc[k] > 180)
        r["y_low"] = int(fut_min.iloc[k] < 70)
        for h in (30, 60, 120):
            r[f"g_future_{h}"] = g.iloc[k + h // 5]
        rows.append(r)
    print(f"[{i + 1}/{len(pat)}] {p.patient_id} done  ({time.time() - t0:.0f}s)", flush=True)

out = pd.DataFrame(rows)
out.to_csv(f"{DATA}/ml_dataset.csv", index=False)
print(f"\nSaved {DATA}/ml_dataset.csv: {len(out):,} rows, {out.patient_id.nunique()} patients")
print(f"overall: y_high rate {out.y_high.mean():.1%} | y_low rate {out.y_low.mean():.2%}")
print(f"new-high candidates (g_now<=180): {(out.g_now <= 180).sum():,} with {out[out.g_now <= 180].y_high.mean():.1%} positive")
print(f"new-low candidates  (g_now>=70) : {(out.g_now >= 70).sum():,} with {out[out.g_now >= 70].y_low.sum()} positives")

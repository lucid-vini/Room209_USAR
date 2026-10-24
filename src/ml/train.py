"""
train.py -- trains the risk models, runs the ablation, computes every metric, saves everything.

    python src\\ml\\train.py data

Outputs:  results/results_table.csv   (events: baselines + 3 LightGBM ablation rows, HIGH and LOW)
          results/forecast_table.csv  (RMSE + Clarke zones: persistence vs twin)
          results/shap_importance.csv, results/figures/*.png, results/summary.txt
          models/risk_high.txt, models/risk_low.txt (if enough events), models/feature_lists.json
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from features import STATIC_COLS, SENSOR_COLS, TWIN_COLS, FRIENDLY

SEED = 42
DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
for d in ("results/figures", "models"):
    os.makedirs(d, exist_ok=True)
df = pd.read_csv(f"{DATA}/ml_dataset.csv", parse_dates=["timestamp"])

# ------------------------------------------------------------ 1. split BY PATIENT
rng = np.random.default_rng(SEED)
pids = np.array(sorted(df.patient_id.unique()))
rng.shuffle(pids)
n = len(pids)
n_tr, n_va = int(round(0.70 * n)), int(round(0.15 * n))
split = {"train": pids[:n_tr], "val": pids[n_tr:n_tr + n_va], "test": pids[n_tr + n_va:]}
json.dump({k: list(v) for k, v in split.items()}, open("results/patient_split.json", "w"), indent=1)
print({k: len(v) for k, v in split.items()}, "patients in train/val/test")

FEATURE_SETS = {
    "LightGBM: sensor only": SENSOR_COLS,
    "LightGBM: sensor + health record": SENSOR_COLS + STATIC_COLS,
    "LightGBM: sensor + record + twin (hybrid)": SENSOR_COLS + STATIC_COLS + TWIN_COLS,
}


def threshold_for_recall(y, p, target=0.80):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr >= target)])


def metrics(y, p, thr):
    pred = p >= thr
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    return dict(sensitivity=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), specificity=tn / max(tn + fp, 1))


def auroc_ci(sub, score_col, y_col, B=200):
    """95% CI by resampling TEST PATIENTS (so uncertainty reflects having few patients)"""
    ids = sub.patient_id.unique()
    groups = {i: sub[sub.patient_id == i] for i in ids}
    vals = []
    for _ in range(B):
        s = pd.concat([groups[i] for i in rng.choice(ids, len(ids))])
        if s[y_col].nunique() == 2:
            vals.append(roc_auc_score(s[y_col], s[score_col]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def fit_lgb(X_tr, y_tr, X_va, y_va, seed=SEED):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                           random_state=seed, verbose=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc", callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


# ------------------------------------------------------------ 2. event models (HIGH and LOW)
results, models, curves = [], {}, {}
for target, y_col, cond in [("high >180 in next 2 h", "y_high", df.g_now <= 180),
                            ("low <70 in next 2 h", "y_low", df.g_now >= 70)]:
    d = df[cond].copy()                               # predict NEW events only
    d["extrap"] = d.g_now + 120 * d.g_slope_30m       # linear extrapolation score
    parts = {k: d[d.patient_id.isin(v)] for k, v in split.items()}
    tr, va, te = parts["train"], parts["val"], parts["test"]
    n_pos = {k: int(v[y_col].sum()) for k, v in parts.items()}
    n_pp = {k: int((v.groupby('patient_id')[y_col].sum() > 0).sum()) for k, v in parts.items()}
    print(f"  patients with >=1 event, train/val/test: {n_pp['train']}/{n_pp['val']}/{n_pp['test']}")
    print(f"\n[{target}] positives train/val/test: {n_pos['train']}/{n_pos['val']}/{n_pos['test']}")
    scorers = {}
    if target.startswith("high"):
        scorers["Rule: linear extrapolation"] = ("extrap", None)
        scorers["Twin physics only (forecast peak)"] = ("tw_fc_max", None)
    else:   # for lows the matching scores are the lowest values
        d["extrap_low"] = -d["extrap"]; d["tw_low"] = -d.tw_fc_60
        parts = {k: d[d.patient_id.isin(v)] for k, v in split.items()}
        tr, va, te = parts["train"], parts["val"], parts["test"]
        n_pp = {k: int((v.groupby('patient_id')[y_col].sum() > 0).sum()) for k, v in parts.items()}
        scorers["Rule: linear extrapolation"] = ("extrap_low", None)
        scorers["Twin physics only (forecast at +60 min)"] = ("tw_low", None)
    for name, (col, _) in scorers.items():
        if va[y_col].sum() < 2 or te[y_col].nunique() < 2:
            continue
        thr = threshold_for_recall(va[y_col], va[col])
        lo, hi = auroc_ci(te, col, y_col)
        results.append(dict(target=target, model=name, auroc=roc_auc_score(te[y_col], te[col]), ci_lo=lo, ci_hi=hi,
                            **metrics(te[y_col].to_numpy(), te[col].to_numpy(), thr), n_test=len(te), n_pos_test=n_pos["test"], n_pos_patients_test=n_pp["test"]))
    if n_pos["train"] < 40 or n_pos["val"] < 5 or n_pos["test"] < 5:
        print("  too few events for a reliable LightGBM model -> skipped (report this honestly)")
        results.append(dict(target=target, model="LightGBM (all three)", auroc=np.nan, n_test=len(te), n_pos_test=n_pos["test"], n_pos_patients_test=n_pp["test"],
                            note="too few events to train/evaluate"))
        continue
    for name, cols in FEATURE_SETS.items():
        m = fit_lgb(tr[cols], tr[y_col], va[cols], va[y_col])
        pv, pt = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
        thr = threshold_for_recall(va[y_col], pv)
        te = te.assign(_p=pt)
        lo, hi = auroc_ci(te, "_p", y_col)
        results.append(dict(target=target, model=name, auroc=roc_auc_score(te[y_col], pt), ci_lo=lo, ci_hi=hi,
                            threshold=thr, **metrics(te[y_col].to_numpy(), pt, thr), n_test=len(te), n_pos_test=n_pos["test"], n_pos_patients_test=n_pp["test"]))
        models[(y_col, name)] = (m, cols)
        if target.startswith("high"):
            curves[name] = roc_curve(te[y_col], pt)
    # label-shuffle sanity check on the hybrid model (AUROC should be ~0.5)
    cols = FEATURE_SETS["LightGBM: sensor + record + twin (hybrid)"]
    ys = tr[y_col].sample(frac=1.0, random_state=SEED).to_numpy()
    ms = fit_lgb(tr[cols], ys, va[cols], va[y_col])
    results.append(dict(target=target, model="SANITY: hybrid with SHUFFLED labels",
                        auroc=roc_auc_score(te[y_col], ms.predict_proba(te[cols])[:, 1]), n_test=len(te), n_pos_test=n_pos["test"], n_pos_patients_test=n_pp["test"]))
    # save the hybrid model for the dashboard
    hyb = "LightGBM: sensor + record + twin (hybrid)"
    m, cols = models[(y_col, hyb)]
    m.booster_.save_model(f"models/risk_{'high' if y_col == 'y_high' else 'low'}.txt")
    thr_saved = [r for r in results if r["target"] == target and r["model"] == hyb][0]["threshold"]
    json.dump({"features": cols, "threshold": thr_saved}, open(f"models/meta_{'high' if y_col == 'y_high' else 'low'}.json", "w"))

res = pd.DataFrame(results)
res["caution"] = np.where(res.target.str.startswith("low"), "EXPLORATORY: lows come from very few patients", "")
res.round(3).to_csv("results/results_table.csv", index=False)

# ------------------------------------------------------------ 3. forecast quality: persistence vs twin
te_all = df[df.patient_id.isin(split["test"])]


def clarke_zone(ref, pred):
    if (ref <= 70 and pred <= 70) or (pred <= 1.2 * ref and pred >= 0.8 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if ((70 <= ref <= 290) and pred >= ref + 110) or ((130 <= ref <= 180) and (pred <= (7 / 5) * ref - 182)):
        return "C"
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or ((175 / 3 <= ref <= 70) and pred >= (6 / 5) * ref):
        return "D"
    return "B"


frows = []
for h in (30, 60, 120):
    s = te_all.dropna(subset=[f"g_future_{h}"])
    ref = s[f"g_future_{h}"].to_numpy()
    for name, pred in [("Persistence (glucose stays the same)", s.g_now.to_numpy()), ("Twin forecast (no future meals assumed)", s[f"tw_fc_{h}"].to_numpy())]:
        z = pd.Series([clarke_zone(a, b) for a, b in zip(ref, pred)]).value_counts(normalize=True)
        frows.append(dict(horizon_min=h, model=name, rmse=float(np.sqrt(np.mean((ref - pred) ** 2))),
                          clarke_A_pct=100 * z.get("A", 0), clarke_AB_pct=100 * (z.get("A", 0) + z.get("B", 0)), n=len(s)))
ft = pd.DataFrame(frows)
ft.round(1).to_csv("results/forecast_table.csv", index=False)

# ------------------------------------------------------------ 4. SHAP + figures
m, cols = models[("y_high", "LightGBM: sensor + record + twin (hybrid)")]
d = df[(df.g_now <= 180) & df.patient_id.isin(split["test"])][cols]
Xs = d.sample(n=min(2000, len(d)), random_state=SEED)
sv = shap.TreeExplainer(m).shap_values(Xs)
if isinstance(sv, list):
    sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3:
    sv = sv[..., 1]
imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv).mean(0)})
imp["friendly"] = imp.feature.map(FRIENDLY)
imp = imp.groupby("friendly", as_index=False)["mean_abs_shap"].sum().sort_values("mean_abs_shap", ascending=False)  # sin+cos -> one 'time of day'
imp.round(4).to_csv("results/shap_importance.csv", index=False)
top = imp.head(12)[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top.friendly, top.mean_abs_shap, color="#1f77b4")
ax.set_xlabel("mean |SHAP value| (average influence on predicted high-glucose risk)")
ax.set_title("What drives the hybrid model's high-glucose alerts"); fig.tight_layout()
fig.savefig("results/figures/shap_importance.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
for name, (fpr, tpr, _) in curves.items():
    ax.plot(fpr, tpr, label=name)
ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("False alarm rate"); ax.set_ylabel("Sensitivity (events caught)")
ax.set_title("ROC curves, held-out test patients (high glucose)"); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("results/figures/roc_high.png", dpi=150); plt.close(fig)

s = te_all.dropna(subset=["g_future_60"])
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for a, (nm, col) in zip(ax, [("Persistence", "g_now"), ("Twin forecast", "tw_fc_60")]):
    a.scatter(s.g_future_60, s[col], s=2, alpha=0.25)
    a.plot([40, 350], [40, 350], "k-", lw=1); a.set_xlabel("Actual glucose at +60 min (mg/dL)"); a.set_title(nm)
ax[0].set_ylabel("Predicted (mg/dL)"); fig.tight_layout(); fig.savefig("results/figures/forecast_scatter_60.png", dpi=150); plt.close(fig)

with open("results/summary.txt", "w") as f:
    f.write(res.round(3).to_string(index=False) + "\n\n" + ft.round(1).to_string(index=False) + "\n\n" + imp.head(10).round(3).to_string(index=False))
print("\n=== EVENT RESULTS (held-out test patients) ===")
print(res.round(3).drop(columns=[c for c in ["note"] if c in res.columns]).to_string(index=False))
print("\n=== FORECAST RESULTS ===")
print(ft.round(1).to_string(index=False))
print("\nTop SHAP features:\n", imp.head(8)[["friendly", "mean_abs_shap"]].round(3).to_string(index=False))

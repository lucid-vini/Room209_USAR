"""Quick evaluation of the twin: forecast RMSE vs persistence, and sensitivity recovery."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from twin import personalize, run_filter, forecast

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
N_PAT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
pat = pd.read_csv(f"{DATA}/patients.csv")
hid = pd.read_csv(f"{DATA}/patients_hidden.csv").set_index("patient_id")
sen = pd.read_csv(f"{DATA}/sensor.csv", parse_dates=["timestamp"])
H = {30: 6, 60: 12, 120: 24}
rows, sens_est, sens_prior, ir_true = [], [], [], []
t0 = time.time()
for _, p in pat.head(N_PAT).iterrows():
    df = sen[sen.patient_id == p.patient_id].reset_index(drop=True)
    params = personalize(p.to_dict())
    res = run_filter(params, df)
    sens_est.append(np.exp(res.xs[-288 * 7:, 5]).mean())
    sens_prior.append(params["s0"])
    ir_true.append(hid.loc[p.patient_id, "true_insulin_resistance"])
    g = df["glucose_mgdl"].to_numpy()
    carbs = df["carbs_g"].to_numpy()
    for k in range(288, len(df) - 24, 12):                 # hourly forecast origins after day 1
        if np.isnan(g[k]):
            continue
        st = res.state_at(k)
        fc_none = forecast(st, 120, None, with_band=False)["g_pred"].to_numpy()
        meals = [{"t": (j - k) * 5, "carbs": carbs[j]} for j in range(k + 1, k + 25) if carbs[j] > 0]
        fc_meal = forecast(st, 120, {"meals": meals}, with_band=False)["g_pred"].to_numpy()
        for hz, ns in H.items():
            if np.isnan(g[k + ns]):
                continue
            rows.append((hz, g[k + ns] - g[k], g[k + ns] - fc_none[ns - 1], g[k + ns] - fc_meal[ns - 1]))
r = pd.DataFrame(rows, columns=["h", "persist", "twin_nomeal", "twin_meals_known"])
print(f"{len(r)//3} forecast origins, {time.time()-t0:.0f}s")
print(r.groupby("h").agg(lambda s: float(np.sqrt((s ** 2).mean()))).round(1).rename(columns=lambda c: "RMSE " + c))
print("\nsensitivity recovery (higher true insulin resistance -> lower sensitivity; want NEGATIVE corr):")
print("  corr(prior s0, true IR)     =", round(np.corrcoef(sens_prior, ir_true)[0, 1], 2))
print("  corr(filtered s, true IR)   =", round(np.corrcoef(sens_est, ir_true)[0, 1], 2))

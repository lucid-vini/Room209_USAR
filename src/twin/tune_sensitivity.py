"""
tune_sensitivity.py -- how fast should the filter let a patient's insulin sensitivity change?

The filter has one knob for this: process noise on logS (ls_sd).
  small  -> sensitivity barely moves, stays near the EHR-based prior
  large  -> sensitivity chases every glucose wiggle (noisy, can hit the clip limits)
We test several values and report (a) forecast RMSE and (b) how well the filtered sensitivity
matches the hidden truth (patients_hidden.csv -- used ONLY for checking, never for fitting).

Run:  python src\\twin\\tune_sensitivity.py data 20
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from twin import personalize, run_filter, forecast

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
N_PAT = int(sys.argv[2]) if len(sys.argv) > 2 else 20
pat = pd.read_csv(f"{DATA}/patients.csv").head(N_PAT)
hid = pd.read_csv(f"{DATA}/patients_hidden.csv").set_index("patient_id")
sen = pd.read_csv(f"{DATA}/sensor.csv", parse_dates=["timestamp"])
groups = {pid: d.reset_index(drop=True) for pid, d in sen.groupby("patient_id")}

print(f"{'ls_sd':>8} {'corr(s,IR)':>11} {'RMSE60':>8} {'RMSE120':>8}   (meals-logged forecasts; corr should be NEGATIVE)")
for ls_sd in [0.004, 0.002, 0.001, 0.0005, 0.0002]:
    Q = np.diag([2.0 ** 2, 0.02 ** 2, 0.02 ** 2, 1.5 ** 2, 0.5 ** 2, ls_sd ** 2])
    s_est, ir, err60, err120 = [], [], [], []
    for _, p in pat.iterrows():
        df = groups[p.patient_id]
        res = run_filter(personalize(p.to_dict()), df, Q=Q)
        s_est.append(np.exp(res.xs[-288 * 7:, 5]).mean())
        ir.append(hid.loc[p.patient_id, "true_insulin_resistance"])
        g, carbs = df["glucose_mgdl"].to_numpy(), df["carbs_g"].to_numpy()
        for k in range(288, len(df) - 24, 36):               # every 3 h (faster)
            if np.isnan(g[k]):
                continue
            meals = [{"t": (j - k) * 5, "carbs": carbs[j]} for j in range(k + 1, k + 25) if carbs[j] > 0]
            fc = forecast(res.state_at(k), 120, {"meals": meals}, with_band=False)["g_pred"].to_numpy()
            if not np.isnan(g[k + 12]): err60.append(g[k + 12] - fc[11])
            if not np.isnan(g[k + 24]): err120.append(g[k + 24] - fc[23])
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
    print(f"{ls_sd:>8} {np.corrcoef(s_est, ir)[0, 1]:>11.2f} {rm(err60):>8.1f} {rm(err120):>8.1f}")

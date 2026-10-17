"""
plot_patient.py -- makes the figures for slides 11, 16 and 18.

Run:  python src\\twin\\plot_patient.py data P001
Saves PNGs into docs\\figures\\
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from twin import personalize, run_filter, forecast, whatif

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
PID = sys.argv[2] if len(sys.argv) > 2 else "P001"
os.makedirs("docs/figures", exist_ok=True)

pat = pd.read_csv(f"{DATA}/patients.csv").set_index("patient_id")
df = pd.read_csv(f"{DATA}/sensor.csv", parse_dates=["timestamp"])
df = df[df.patient_id == PID].reset_index(drop=True)
res = run_filter(personalize(pat.loc[PID].to_dict()), df)
f = res.df

# ---- Figure 1: raw CGM vs filtered estimate (slide 11) ----
a, b = 288 * 4, 288 * 6                                   # two days
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(df.timestamp[a:b], df.glucose_mgdl[a:b], color="#bbbbbb", lw=1, label="Raw CGM sensor")
ax.plot(f.timestamp[a:b], f.g_est[a:b], color="#1f77b4", lw=2, label="Twin (Kalman-filtered) estimate")
ax.fill_between(f.timestamp[a:b], (f.g_est - 1.96 * f.g_std)[a:b], (f.g_est + 1.96 * f.g_std)[a:b],
                color="#1f77b4", alpha=0.15, label="95% uncertainty")
for t, c in zip(df.timestamp[a:b][df.carbs_g[a:b] > 0], df.carbs_g[a:b][df.carbs_g[a:b] > 0]):
    ax.axvline(t, color="orange", alpha=0.4, lw=1)
ax.axhline(180, ls="--", color="red", lw=0.8); ax.axhline(70, ls="--", color="red", lw=0.8)
ax.set_ylabel("Glucose (mg/dL)"); ax.set_title(f"{PID}: raw sensor vs twin estimate (orange lines = logged meals)")
ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig("docs/figures/fig1_filter.png", dpi=150); plt.close(fig)

# ---- Figure 2: 2-hour forecast made after dinner vs what really happened (slide 16) ----
day = 5
k = int(np.where((df.timestamp.dt.day == day) & (df.timestamp.dt.hour == 20) & (df.timestamp.dt.minute == 30))[0][0])
st = res.state_at(k)
fc = forecast(st, 120, None)
lo, hi = k - 36, k + 25
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(df.timestamp[lo:hi], df.glucose_mgdl[lo:hi], color="black", lw=2, label="Actual glucose")
ax.plot(fc.timestamp, fc.g_pred, "--", color="#d62728", lw=2, label="Twin forecast (made at the vertical line)")
ax.fill_between(fc.timestamp, fc.g_lo, fc.g_hi, color="#d62728", alpha=0.15)
ax.axvline(df.timestamp[k], color="gray", ls=":"); ax.axhline(180, ls="--", color="red", lw=0.8)
ax.set_ylabel("Glucose (mg/dL)"); ax.set_title(f"{PID}: 2-hour forecast vs actual (no future meals assumed)")
ax.legend(); fig.tight_layout(); fig.savefig("docs/figures/fig2_forecast.png", dpi=150); plt.close(fig)

# ---- Figure 3: what-if, walk after a planned snack (slide 14 / 18) ----
w = whatif(st, {"planned_meals": [{"t": 30, "carbs": 25}], "walk": {"t": 10, "minutes": 20}})
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(w.timestamp, w.g_base, color="#d62728", lw=2, label="Do nothing different")
ax.plot(w.timestamp, w.g_whatif, color="#2ca02c", lw=2, label="20-minute walk after dinner")
ax.axhline(180, ls="--", color="red", lw=0.8)
ax.set_ylabel("Glucose (mg/dL)"); ax.set_title(f"{PID}: what-if simulation (illustrative, not clinical advice)")
ax.legend(); fig.tight_layout(); fig.savefig("docs/figures/fig3_whatif.png", dpi=150); plt.close(fig)
print("peak without walk:", round(w.g_base.max()), "| with walk:", round(w.g_whatif.max()))
print("Saved 3 figures in docs/figures/")

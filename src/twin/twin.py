import math
import numpy as np
import pandas as pd
from model import DEFAULTS, DT, step, N_STATE
from kalman import EKF, numerical_jacobian
 
STEPS_PER_HOUR = int(60 / DT)
 
 
# ------------------------------------------------------------------ personalisation
def personalize(profile: dict) -> dict:
    """EHR -> model parameters.  A literature-informed PRIOR (a sensible starting guess),
    NOT a clinically validated mapping.  The Kalman filter then refines sensitivity from sensor data.
    Directions of effect: higher BMI, higher HbA1c, longer diabetes duration and a family history
    all lower insulin sensitivity.  Cohort centres/spreads below are rough T2D-cohort values."""
    z_bmi = (profile["bmi"] - 27.5) / 4.5
    z_a1c = (profile["hba1c"] - 7.3) / 1.0
    z_yrs = (profile["diabetes_years"] - 6.0) / 5.0
    fam = profile.get("family_history", 0)
    log_s0 = -(0.25 * z_bmi + 0.25 * z_a1c + 0.10 * z_yrs + 0.10 * fam)
    log_s0 = float(np.clip(log_s0, -0.9, 0.9))
    e_ag = 28.7 * profile["hba1c"] - 46.7              # estimated average glucose from HbA1c (ADAG formula)
    Gb = float(np.clip(e_ag - 28.0, 80.0, 200.0))      # fasting level sits below the 24-h average
    p = dict(DEFAULTS)
    p.update(Gb=Gb, log_s0=log_s0, s0=math.exp(log_s0))
    return p
 
 
# ------------------------------------------------------------------ inputs from sensor data
def exercise_from_steps(steps):
    """steps per 5 min -> exercise intensity 0..1 (a brisk walk is ~500 steps / 5 min)"""
    return np.clip((np.asarray(steps, dtype=float) - 250.0) / 300.0, 0.0, 1.0)
 
 
def sleep_hours_last_night(sensor_df, upto_idx):
    """hours of sleep (stage>0) in the 24 h before index upto_idx"""
    lo = max(0, upto_idx - 288)
    return float((sensor_df["sleep_stage"].iloc[lo:upto_idx + 1] > 0).sum() * DT / 60.0)
 
 
# ------------------------------------------------------------------ filtering
class FilterResult:
    def __init__(self, params, timestamps, xs, Ps, y, sleep_df=None):
        self.params, self.ts, self.xs, self.Ps, self.y = params, timestamps, xs, Ps, y
        self.sleep_df = sleep_df
        self.df = pd.DataFrame({
            "timestamp": timestamps,
            "g_est": xs[:, 0],
            "insulin_action_est": xs[:, 1] * 1e-3,
            "sensitivity_est": np.exp(xs[:, 5]),
            "g_std": np.sqrt(Ps[:, 0, 0]),
            "insulin_est": xs[:, 2],
            "gut_carbs_est": xs[:, 3] + xs[:, 4],
        })
 
    def state_at(self, k):
        k = int(k)
        sl = None
        if self.sleep_df is not None:
            sl = sleep_hours_last_night(self.sleep_df, k)
        return dict(t=self.ts[k], x=self.xs[k].copy(), P=self.Ps[k].copy(), params=self.params,
                    sleep_hours=sl, index=k)
 
    def last_state(self):
        return self.state_at(len(self.ts) - 1)
 
 
def run_filter(params, sensor_df, R=9.0, Q=None, P0=None):
    """sensor_df: one patient's rows (sorted by time) with glucose_mgdl, steps, carbs_g"""
    sensor_df = sensor_df.reset_index(drop=True)
    y = sensor_df["glucose_mgdl"].to_numpy(dtype=float)
    carbs = sensor_df["carbs_g"].to_numpy(dtype=float)
    e = exercise_from_steps(sensor_df["steps"].to_numpy())
    n = len(sensor_df)
    first = y[~np.isnan(y)][0]
    x0 = [first, 0.0, 0.0, 0.0, 0.0, params["log_s0"]]
    ekf = EKF(params, x0, P0=P0, R=R, Q=Q)
    xs = np.zeros((n, N_STATE))
    Ps = np.zeros((n, N_STATE, N_STATE))
    for k in range(n):
        if k > 0:
            ekf.predict(carbs[k - 1], e[k - 1])        # transition k-1 -> k uses inputs at k-1
        ekf.update(y[k])
        xs[k], Ps[k] = ekf.x, ekf.P
    return FilterResult(params, sensor_df["timestamp"].to_numpy(), xs, Ps, y, sleep_df=sensor_df)
 
 
# ------------------------------------------------------------------ forecasting
def _scenario_inputs(n_steps, scenario):
    """turn a scenario dict into per-step carbs and exercise arrays (t=0 is 'now')"""
    carbs = np.zeros(n_steps)
    ex = np.zeros(n_steps)
    if scenario:
        for m in scenario.get("meals", []):                 # {"t": minutes from now, "carbs": grams}
            k = int(round(m["t"] / DT))
            if 0 <= k < n_steps:
                carbs[k] += m["carbs"]
        w = scenario.get("walk")                              # {"t": start min from now, "minutes": duration}
        if w and w.get("minutes", 0) > 0:
            k0 = int(round(w["t"] / DT))
            k1 = k0 + max(1, int(round(w["minutes"] / DT)))
            ex[max(k0, 0):min(k1, n_steps)] = 1.0
    return carbs, ex
 
 
def forecast(state, horizon_min=120, scenario=None, with_band=True):
    """Roll the twin forward.  Without a scenario it assumes NO future meals or exercise.
    Returns timestamps at +5, +10, ... minutes."""
    n_steps = int(horizon_min / DT)
    params = dict(state["params"])
    x = np.array(state["x"], dtype=float)
    Pm = np.array(state["P"], dtype=float)
    if scenario and scenario.get("sleep_hours") is not None and state.get("sleep_hours") is not None:
        # ASSUMPTION: each hour of sleep under 7 h lowers insulin sensitivity ~6 %.
        cur = 1 + 0.06 * max(0.0, 7.0 - state["sleep_hours"])
        new = 1 + 0.06 * max(0.0, 7.0 - scenario["sleep_hours"])
        x[5] = float(np.clip(x[5] + math.log(cur / new), -1.5, 1.5))
    carbs, ex = _scenario_inputs(n_steps, scenario)
    Q = np.diag([2.0 ** 2, 0.02 ** 2, 0.02 ** 2, 1.5 ** 2, 0.5 ** 2, 0.004 ** 2])
    g, sd = [], []
    for k in range(n_steps):
        if with_band:
            F, fx = numerical_jacobian(x, carbs[k], ex[k], params)
            Pm = F @ Pm @ F.T + Q
            x = fx
            sd.append(math.sqrt(max(Pm[0, 0], 0.0) + 9.0))
        else:
            x = np.array(step(list(x), carbs[k], ex[k], params))
        g.append(x[0])
    ts = pd.to_datetime(state["t"]) + pd.to_timedelta(np.arange(1, n_steps + 1) * DT, unit="m")
    g = np.array(g)
    out = pd.DataFrame({"timestamp": ts, "g_pred": g})
    if with_band:
        sd = np.array(sd)
        out["g_lo"], out["g_hi"] = g - 1.96 * sd, g + 1.96 * sd
    return out
 
 
def whatif(state, scenario, horizon_min=120):
    """Counterfactual simulation.
    scenario = {
      "planned_meals": [{"t": 20, "carbs": 30}],     # what is expected to happen (baseline)
      "skip_meals": False,                           # what-if: skip those planned meals
      "walk": {"t": 30, "minutes": 20},              # what-if: add a walk
      "sleep_hours": 6.0 }                           # what-if: last night's sleep was this long
    Returns baseline vs what-if curves."""
    base_sc = {"meals": scenario.get("planned_meals", [])}
    alt_sc = {"meals": [] if scenario.get("skip_meals") else scenario.get("planned_meals", []),
              "walk": scenario.get("walk"), "sleep_hours": scenario.get("sleep_hours")}
    b = forecast(state, horizon_min, base_sc, with_band=False)
    a = forecast(state, horizon_min, alt_sc, with_band=False)
    out = pd.DataFrame({"timestamp": b["timestamp"], "g_base": b["g_pred"], "g_whatif": a["g_pred"]})
    out["delta"] = out["g_whatif"] - out["g_base"]
    return out

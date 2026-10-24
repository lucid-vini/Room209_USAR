"""
pipeline.py -- glue between the twin, the ML model and the dashboard.

    from pipeline import list_patients, moments, predict_patient, run_whatif

predict_patient(patient_id, k) returns ONE dictionary with everything the dashboard shows
(same shape as app/mock_outputs.json, plus a few extras).  k = row index of "now" in that patient's data.
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "ml"))
sys.path.insert(0, os.path.join(ROOT, "src", "twin"))
import numpy as np
import pandas as pd
from features import features_at, personalize, run_filter, FRIENDLY      # noqa: E402
from twin import whatif                                                   # noqa: E402
from risk import predict_risk, explain, _load                             # noqa: E402

DATA = os.path.join(ROOT, "data")
_store = {}


def _tables():
    if "pat" not in _store:
        _store["pat"] = pd.read_csv(os.path.join(DATA, "patients.csv")).set_index("patient_id", drop=False)
        sen = pd.read_csv(os.path.join(DATA, "sensor.csv"), parse_dates=["timestamp"])
        _store["sen"] = {pid: d.sort_values("timestamp").reset_index(drop=True) for pid, d in sen.groupby("patient_id")}
    return _store["pat"], _store["sen"]


def list_patients():
    pat, _ = _tables()
    return list(pat.index)


def _patient(pid):
    """profile, sensor dataframe, and the (cached) Kalman filter run for one patient"""
    pat, sen = _tables()
    if pid not in _store:
        profile = {k: (v.item() if hasattr(v, "item") else v) for k, v in pat.loc[pid].to_dict().items()}
        df = sen[pid]
        _store[pid] = (profile, df, run_filter(personalize(profile), df))
    return _store[pid]


def moments(pid):
    """valid 'now' moments (every 30 min, after day 1, leaving 2 h of future data). returns [(k, timestamp)]"""
    _, df, _ = _patient(pid)
    return [(k, df["timestamp"].iloc[k]) for k in range(288, len(df) - 24, 6)]


def _clean(v):
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 2)


def predict_patient(pid, k, history_steps=72):
    profile, df, res = _patient(pid)
    feats = features_at(profile, df, k, res=res)
    risk = predict_risk(feats)
    reasons = explain(feats, top=5)
    st = res.state_at(k)
    from twin import forecast
    fc = forecast(st, 120, None)
    hist = df.iloc[max(0, k - history_steps):k + 1]
    fut = df.iloc[k + 1:k + 25]
    g_now = float(feats["g_now"].iloc[0])
    meta_h = _load("high")[1] if _load("high") else {"threshold": 0.5}
    return {
        "patient_id": pid, "k": int(k), "now": str(df["timestamp"].iloc[k]), "patient": profile,
        "history": {"timestamp": [str(t) for t in hist["timestamp"]], "glucose_mgdl": [_clean(v) for v in hist["glucose_mgdl"]]},
        "state": {"g_est": _clean(st["x"][0]), "g_now": _clean(g_now), "sensitivity_est": _clean(np.exp(st["x"][5])),
                  "sleep_hours_last_24h": _clean(st["sleep_hours"])},
        "forecast": {"timestamp": [str(t) for t in fc["timestamp"]], "g_pred": [_clean(v) for v in fc["g_pred"]],
                     "g_lo": [_clean(v) for v in fc["g_lo"]], "g_hi": [_clean(v) for v in fc["g_hi"]]},
        "ml": {"p_high": risk.get("p_high"), "p_low": risk.get("p_low"), "alert_high": risk.get("alert_high"),
               "alert_low": risk.get("alert_low"), "threshold_high": meta_h["threshold"],
               "high_applicable": g_now <= 180, "low_applicable": g_now >= 70,
               "top_reasons": [[n, round(v, 3)] for n, v in reasons]},
        "actual_future": {"timestamp": [str(t) for t in fut["timestamp"]], "glucose_mgdl": [_clean(v) for v in fut["glucose_mgdl"]]},
    }


def run_whatif(pid, k, snack_carbs=0, snack_in_min=20, skip_snack=False, walk_min=0, walk_start_min=10, sleep_hours=None):
    """counterfactual: planned snack (baseline) vs. what-if (skip snack / walk / different sleep)"""
    _, _, res = _patient(pid)
    st = res.state_at(k)
    scenario = {
        "planned_meals": [{"t": snack_in_min, "carbs": snack_carbs}] if snack_carbs > 0 else [],
        "skip_meals": bool(skip_snack),
        "walk": {"t": walk_start_min, "minutes": walk_min} if walk_min > 0 else None,
        "sleep_hours": sleep_hours,
    }
    return whatif(st, scenario)

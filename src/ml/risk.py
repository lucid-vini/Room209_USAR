"""
risk.py -- the two functions the dashboard / glue code calls.

    from risk import predict_risk, explain
    p = predict_risk(features_row)    # -> {"p_high": 0.62, "p_low": 0.03 or None, "alert_high": True}
    r = explain(features_row)         # -> [("rapid rise/fall in last 30 min", 0.21), ...] top 5

`features_row` is ONE-row DataFrame from features.features_at(...) (or a dict with the same keys).
"""
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from features import FRIENDLY

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
_cache = {}


def _load(which):
    if which not in _cache:
        mp, jp = os.path.join(_ROOT, f"risk_{which}.txt"), os.path.join(_ROOT, f"meta_{which}.json")
        if not (os.path.exists(mp) and os.path.exists(jp)):
            _cache[which] = None
        else:
            meta = json.load(open(jp))
            _cache[which] = (lgb.Booster(model_file=mp), meta)
    return _cache[which]


def _frame(features):
    return features if isinstance(features, pd.DataFrame) else pd.DataFrame([features])


def predict_risk(features):
    X = _frame(features)
    out = {}
    for which in ("high", "low"):
        loaded = _load(which)
        if loaded is None:
            out[f"p_{which}"] = None
            continue
        booster, meta = loaded
        out[f"p_{which}"] = float(booster.predict(X[meta["features"]])[0])
        out[f"alert_{which}"] = bool(out[f"p_{which}"] >= meta["threshold"])
    return out


def explain(features, top=5, which="high"):
    """SHAP: how much each input pushed THIS prediction up (+) or down (-)."""
    X = _frame(features)
    booster, meta = _load(which)
    sv = shap.TreeExplainer(booster).shap_values(X[meta["features"]])
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    contrib = pd.Series(sv[0], index=meta["features"])
    grouped = contrib.groupby(contrib.index.map(lambda c: FRIENDLY.get(c, c))).sum()   # merge sin+cos
    grouped = grouped.reindex(grouped.abs().sort_values(ascending=False).index)[:top]
    return [(k, float(v)) for k, v in grouped.items()]

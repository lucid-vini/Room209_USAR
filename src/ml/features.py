"""
features.py -- turns raw sensor rows into the numbers the ML model sees at a moment "now".

RULE: every feature at origin k may only use data at index <= k (no peeking into the future).
Labels (what happens next) are built separately in build_dataset.py.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "twin"))
from twin import personalize, run_filter, forecast  # noqa: E402

STATIC_COLS = ["age", "sex_male", "bmi", "hba1c", "diabetes_years", "family_history", "on_metformin", "systolic_bp"]
SENSOR_COLS = ["g_now", "g_mean_1h", "g_std_1h", "g_mean_3h", "g_std_3h", "g_min_3h", "g_max_3h", "g_slope_30m",
               "min_since_meal", "carbs_3h", "steps_1h", "steps_3h", "hr_mean_1h", "sleep_h_24h", "hour_sin", "hour_cos"]
TWIN_COLS = ["tw_g_est", "tw_resid", "tw_sensitivity", "tw_insulin_action", "tw_insulin", "tw_gut_carbs",
             "tw_fc_max", "tw_fc_30", "tw_fc_60", "tw_fc_120", "tw_fc_peak_min"]

FRIENDLY = {
    "g_now": "current glucose", "g_mean_1h": "average glucose, last hour", "g_std_1h": "glucose variability, last hour",
    "g_mean_3h": "average glucose, last 3 hours", "g_std_3h": "glucose variability, last 3 hours",
    "g_min_3h": "lowest glucose, last 3 hours", "g_max_3h": "highest glucose, last 3 hours",
    "g_slope_30m": "rapid rise/fall in last 30 min", "min_since_meal": "time since last logged meal",
    "carbs_3h": "carbs eaten in last 3 hours", "steps_1h": "steps in last hour", "steps_3h": "steps in last 3 hours",
    "hr_mean_1h": "heart rate, last hour", "sleep_h_24h": "sleep in last 24 hours",
    "hour_sin": "time of day", "hour_cos": "time of day",
    "age": "age", "sex_male": "sex", "bmi": "BMI", "hba1c": "HbA1c (long-term control)",
    "diabetes_years": "years with diabetes", "family_history": "family history", "on_metformin": "metformin use",
    "systolic_bp": "blood pressure",
    "tw_g_est": "twin's filtered glucose", "tw_resid": "sensor vs twin mismatch", "tw_sensitivity": "estimated insulin sensitivity",
    "tw_insulin_action": "estimated insulin action", "tw_insulin": "estimated insulin level",
    "tw_gut_carbs": "carbs still being absorbed (twin)", "tw_fc_max": "twin's forecast peak (next 2 h)",
    "tw_fc_30": "twin's forecast at +30 min", "tw_fc_60": "twin's forecast at +60 min",
    "tw_fc_120": "twin's forecast at +120 min", "tw_fc_peak_min": "time to twin's forecast peak",
}


def static_features(profile):
    return {"age": profile["age"], "sex_male": int(profile["sex"] == "M"), "bmi": profile["bmi"],
            "hba1c": profile["hba1c"], "diabetes_years": profile["diabetes_years"],
            "family_history": profile["family_history"], "on_metformin": profile["on_metformin"],
            "systolic_bp": profile["systolic_bp"]}


def sensor_feature_frame(df):
    """vectorised sensor features for EVERY row of one patient (each row uses only past/current data)."""
    df = df.reset_index(drop=True)
    n = len(df)
    g = df["glucose_mgdl"].interpolate(limit=6, limit_area="inside")     # fill gaps up to 30 min only
    out = pd.DataFrame(index=df.index)
    out["g_now"] = g
    out["g_mean_1h"] = g.rolling(12, min_periods=8).mean()
    out["g_std_1h"] = g.rolling(12, min_periods=8).std()
    out["g_mean_3h"] = g.rolling(36, min_periods=24).mean()
    out["g_std_3h"] = g.rolling(36, min_periods=24).std()
    out["g_min_3h"] = g.rolling(36, min_periods=24).min()
    out["g_max_3h"] = g.rolling(36, min_periods=24).max()
    out["g_slope_30m"] = (g - g.shift(6)) / 30.0                          # mg/dL per minute
    idx = pd.Series(np.where(df["carbs_g"] > 0, np.arange(n), np.nan)).ffill()
    out["min_since_meal"] = ((np.arange(n) - idx) * 5).fillna(1440).clip(upper=1440)
    out["carbs_3h"] = df["carbs_g"].rolling(36, min_periods=1).sum()
    out["steps_1h"] = df["steps"].rolling(12, min_periods=1).sum()
    out["steps_3h"] = df["steps"].rolling(36, min_periods=1).sum()
    out["hr_mean_1h"] = df["heart_rate"].rolling(12, min_periods=1).mean()
    out["sleep_h_24h"] = (df["sleep_stage"] > 0).astype(float).rolling(288, min_periods=1).sum() * 5 / 60
    hod = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hod / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hod / 24)
    return out, g


def twin_feature_row(res, k):
    """twin outputs at origin k. The filter is causal (only data <= k), so no leakage."""
    st = res.state_at(k)
    fc = forecast(st, 120, None, with_band=False)["g_pred"].to_numpy()
    x = st["x"]
    return {"tw_g_est": x[0], "tw_sensitivity": float(np.exp(x[5])), "tw_insulin_action": x[1] * 1e-3,
            "tw_insulin": x[2], "tw_gut_carbs": x[3] + x[4], "tw_fc_max": fc.max(),
            "tw_fc_30": fc[5], "tw_fc_60": fc[11], "tw_fc_120": fc[23], "tw_fc_peak_min": (int(fc.argmax()) + 1) * 5}


def features_at(profile, df, k, res=None):
    """ONE row of all features at index k of a patient's data (used by the dashboard glue code)."""
    if res is None:
        res = run_filter(personalize(profile), df)
    sf, _ = sensor_feature_frame(df)
    row = {**static_features(profile), **sf.iloc[k].to_dict(), **twin_feature_row(res, k)}
    row["tw_resid"] = row["g_now"] - row["tw_g_est"]
    return pd.DataFrame([row])

import argparse
import os
import numpy as np
import pandas as pd
 
STEP = 5                      # minutes per sample
PER_DAY = 24 * 60 // STEP     # 288 samples per day
 
 
def make_profiles(n, rng):
    prof, hidden = [], []
    for i in range(n):
        pid = f"P{i + 1:03d}"
        age = int(np.clip(rng.normal(52, 11), 28, 80))
        sex = str(rng.choice(["M", "F"]))
        bmi = float(np.clip(rng.normal(27.5, 4.5), 18.5, 42))
        years = int(np.clip(rng.gamma(2.0, 3.0), 0, 25))
        fam = int(rng.random() < 0.55)
        # latent insulin resistance (1.0 = average T2D patient); higher = worse
        ir = 1.0 + 0.045 * (bmi - 27.5) + 0.012 * (age - 52) + 0.015 * years + 0.10 * fam + rng.normal(0, 0.12)
        ir = float(np.clip(ir, 0.6, 1.8))
        metformin = int(rng.random() < min(0.95, 0.35 + 0.04 * years))
        hba1c = float(np.clip(6.6 + 1.8 * (ir - 1.0) - 0.4 * metformin + rng.normal(0, 0.5), 5.6, 11.5))
        sbp = int(np.clip(110 + 0.4 * (age - 40) + 0.6 * (bmi - 25) + rng.normal(0, 8), 95, 190))
        prof.append(dict(patient_id=pid, age=age, sex=sex, bmi=round(bmi, 1), hba1c=round(hba1c, 1),
                         diabetes_years=years, family_history=fam, on_metformin=metformin, systolic_bp=sbp))
        e_ag = 28.7 * hba1c - 46.7   # standard estimated-average-glucose formula (ADAG)
        hidden.append(dict(patient_id=pid, true_insulin_resistance=round(ir, 3), eag_mgdl=round(e_ag, 1)))
    return pd.DataFrame(prof), pd.DataFrame(hidden)
 
 
def kernel(peak_min, length_steps):
    """Gamma-like bump, peak value 1 at peak_min minutes."""
    t = np.arange(length_steps) * STEP
    return (t / peak_min) * np.exp(1 - t / peak_min)
 
 
def add_effect(arr, start, amp, peak_min, length=96):
    if start < 0 or start >= len(arr):
        return
    end = min(len(arr), start + length)
    arr[start:end] += amp * kernel(peak_min, length)[: end - start]
 
 
def simulate_patient(p, h, days, rng, start_ts):
    n = days * PER_DAY
    ir0 = h["true_insulin_resistance"]
    bmi = p["bmi"]
 
    #  sleep schedule and stages 
    stage = np.zeros(n, dtype=int)            # 0 awake, 1 light, 2 deep, 3 REM
    in_sleep = np.zeros(n, dtype=bool)
    sleep_hours = np.full(days, 7.0)          # hours slept the night BEFORE day d
    sleep_start_idx = []
    for d in range(days):
        s_min = rng.normal(23 * 60, 45)
        dur_h = float(np.clip(rng.normal(6.8, 1.0), 4.5, 8.8))
        s = d * PER_DAY + int(s_min // STEP)
        L = int(dur_h * 60 // STEP)
        sleep_start_idx.append(s)
        if d + 1 < days:
            sleep_hours[d + 1] = dur_h
        for k in range(L):
            idx = s + k
            if idx >= n:
                break
            in_sleep[idx] = True
            ph = (k * STEP) % 90
            cyc = (k * STEP) // 90
            if ph < 10:
                st = 1
            elif ph < 40:
                st = 2 if cyc < 3 else 1
            elif ph < 65:
                st = 1
            else:
                st = 3
            if rng.random() < 0.02:
                st = 0
            stage[idx] = st
    # ir per day: poor sleep -> lower insulin sensitivity -> higher resistance (ASSUMPTION)
    ir_day = ir0 * (1 + 0.06 * np.maximum(0, 7.0 - sleep_hours))
 
    #  meals 
    glucose = np.zeros(n)
    carbs_logged = np.zeros(n)
    meal_defs = [  # name, mean minute, sd, carbs mean, carbs sd, probability
        ("breakfast", 8 * 60, 40, 55, 12, 1.0),
        ("lunch", 13.5 * 60, 50, 80, 15, 1.0),
        ("dinner", 20 * 60, 50, 70, 15, 1.0),
        ("snack", 16.5 * 60, 50, 20, 7, 0.5),
        ("late_snack", 22.5 * 60, 30, 25, 8, 0.15),
    ]
    dinner_idx = {}
    for d in range(days):
        for name, mu, sd, cm, cs, pr in meal_defs:
            if rng.random() > pr:
                continue
            idx = d * PER_DAY + int(round(rng.normal(mu, sd) / STEP))
            if idx < 0 or idx >= n - 1:
                continue
            carbs = float(np.clip(rng.normal(cm, cs), 8, 140))
            peak = 45 + 20 * ir_day[d] + rng.normal(0, 5)
            amp = carbs * 0.9 * ir_day[d] * rng.lognormal(0, 0.15)
            add_effect(glucose, idx, amp, peak)
            if rng.random() > 0.10:           # 10% of meals are UNLOGGED (realistic gap)
                carbs_logged[idx] += round(carbs, 0)
            if name == "dinner":
                dinner_idx[d] = idx
 
    #  exercise (walks) 
    steps_walk = np.zeros(n)
    hr_walk = np.zeros(n)
    for d in range(days):
        walks = []
        if d in dinner_idx and rng.random() < 0.30:
            walks.append(dinner_idx[d] + int(rng.integers(3, 9)))        # walk after dinner
        if rng.random() < 0.20:
            walks.append(d * PER_DAY + int((6.5 * 60 + rng.normal(0, 20)) // STEP))
        for w in walks:
            dur = int(rng.integers(15, 41))                              # minutes
            L = max(1, dur // STEP)
            if w < 0 or w + L >= n:
                continue
            steps_walk[w:w + L] += rng.normal(550, 60, L)
            hr_walk[w:w + L] += rng.normal(35, 4, L)
            add_effect(glucose, w, -(0.9 * dur) * (0.7 + 0.3 * rng.random()), 40, length=60)
 
    #  dawn phenomenon + slow drift 
    t_of_day = (np.arange(n) % PER_DAY) * STEP
    day_idx = np.arange(n) // PER_DAY
    dawn = 10 * ir_day[day_idx] * np.exp(-0.5 * ((t_of_day - 6.5 * 60) / 60) ** 2)
    drift = np.zeros(n)
    for i in range(1, n):
        drift[i] = 0.97 * drift[i - 1] + rng.normal(0, 2.0)
    raw = glucose + dawn + drift
 
    #  calibrate so the CGM mean is consistent with the patient's HbA1c (+ real-world mismatch) 
    target_mean = h["eag_mgdl"] + rng.normal(0, 8)
    base = target_mean - raw.mean()
    g = raw + base + rng.normal(0, 3.0, n)                               # sensor noise
    g = np.clip(g, 40, 400)
 
    #  sensor dropouts 
    for _ in range(int(days * 0.7)):
        s = int(rng.integers(0, n - 10))
        g[s:s + int(rng.integers(2, 9))] = np.nan
 
    #  steps 
    awake = ~in_sleep
    active = rng.random(n) < 0.60
    steps = np.where(awake & active, rng.gamma(2.0, 28.0, n), 0.0) + steps_walk
    steps = np.round(steps).astype(int)
 
    #  heart rate 
    rest = 66 + 0.35 * (bmi - 27)
    circ = -8 * np.cos(2 * np.pi * (t_of_day - 15 * 60) / (24 * 60)) * 0.5 - 4  # lower at night
    hr_noise = np.zeros(n)
    for i in range(1, n):
        hr_noise[i] = 0.9 * hr_noise[i - 1] + rng.normal(0, 1.5)
    hr = rest + circ + hr_noise + np.minimum(0.25 * steps, 40) * 0.5 + hr_walk
    hr = np.where(stage == 2, hr - 8, np.where(in_sleep, hr - 5, hr))
    hr = np.clip(np.round(hr), 40, 180).astype(int)
 
    ts = pd.date_range(start_ts, periods=n, freq=f"{STEP}min")
    return pd.DataFrame({
        "patient_id": p["patient_id"], "timestamp": ts,
        "glucose_mgdl": np.round(g, 1), "heart_rate": hr, "steps": steps,
        "sleep_stage": stage, "carbs_g": carbs_logged,
    })
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=60)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    ap.add_argument("--start", default="2026-09-01")
    a = ap.parse_args()
 
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    prof, hidden = make_profiles(a.n_patients, rng)
    frames = []
    for (_, p), (_, h) in zip(prof.iterrows(), hidden.iterrows()):
        frames.append(simulate_patient(p.to_dict(), h.to_dict(), a.days, rng, a.start))
    sensor = pd.concat(frames, ignore_index=True)
 
    prof.to_csv(os.path.join(a.out, "patients.csv"), index=False)
    hidden.to_csv(os.path.join(a.out, "patients_hidden.csv"), index=False)
    sensor.to_csv(os.path.join(a.out, "sensor.csv"), index=False)
 
    #  sanity report 
    g = sensor["glucose_mgdl"]
    print(f"patients: {len(prof)} | rows: {len(sensor):,} | missing glucose: {g.isna().mean():.2%}")
    print(f"glucose mean {g.mean():.0f}, min {g.min():.0f}, max {g.max():.0f}")
    print(f"time >180: {(g > 180).mean():.1%} | time <70: {(g < 70).mean():.2%} | in 70-180: {g.between(70, 180).mean():.1%}")
    print(f"avg steps/day: {sensor.groupby('patient_id')['steps'].sum().mean() / a.days:.0f}")
    print(f"avg HR: {sensor['heart_rate'].mean():.0f} bpm | avg carbs/day: {sensor['carbs_g'].sum() / (len(prof) * a.days):.0f} g")
    chk = sensor.groupby("patient_id")["glucose_mgdl"].mean().to_frame("cgm_mean").join(
        prof.set_index("patient_id")["hba1c"])
    chk["eag"] = 28.7 * chk["hba1c"] - 46.7
    print(f"corr(HbA1c, CGM mean) = {chk['hba1c'].corr(chk['cgm_mean']):.2f}")
    print(f"patients with any glucose <70: {(sensor.assign(lo=g < 70).groupby('patient_id')['lo'].any()).sum()} / {len(prof)}")
    print("Saved to", os.path.abspath(a.out))
 
 
if __name__ == "__main__":
    main()

"""
Doctor dashboard for the Digital Twin (Type 2 Diabetes).   Run from the project root:

    streamlit run app\\dashboard.py
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pipeline as pl

st.set_page_config(page_title="Digital Twin: Type 2 Diabetes", layout="wide")


@st.cache_data(show_spinner="Running the twin...")
def get_view(pid, k):
    return pl.predict_patient(pid, k)


@st.cache_data(show_spinner=False)
def get_moments(pid):
    return pl.moments(pid)


@st.cache_data(show_spinner="Simulating...")
def get_whatif(pid, k, snack, skip, walk, sleep):
    return pl.run_whatif(pid, k, snack_carbs=snack, skip_snack=skip, walk_min=walk, sleep_hours=sleep)


# ------------------------------------------------------------------ sidebar: pick patient and moment
st.sidebar.title("Virtual patient")
pid = st.sidebar.selectbox("Patient", pl.list_patients())
ms = get_moments(pid)
labels = [t.strftime("%a %d %b, %H:%M") for _, t in ms]
default = next((i for i, (_, t) in enumerate(ms) if t.day == 5 and t.hour == 20 and t.minute == 30), len(ms) // 2)
choice = st.sidebar.select_slider("Moment in time (\"now\")", options=list(range(len(ms))), value=default,
                                  format_func=lambda i: labels[i])
k = ms[choice][0]
show_truth = st.sidebar.checkbox("Show what actually happened (validation only)", value=True)
st.sidebar.caption("Proof of concept on synthetic data. Not medical advice.")

v = get_view(pid, k)
p, ml, state = v["patient"], v["ml"], v["state"]

# ------------------------------------------------------------------ header
st.title("Digital Twin: Type 2 Diabetes")
c = st.columns(6)
c[0].metric("Patient", pid)
c[1].metric("Age / sex", f"{p['age']} / {p['sex']}")
c[2].metric("BMI", p["bmi"])
c[3].metric("HbA1c", f"{p['hba1c']}%")
c[4].metric("Years with diabetes", p["diabetes_years"])
c[5].metric("Glucose now", f"{state['g_now']:.0f} mg/dL")

left, right = st.columns([2.2, 1])

# ------------------------------------------------------------------ main chart
with left:
    st.subheader("Glucose: recent history and 2-hour forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v["history"]["timestamp"], y=v["history"]["glucose_mgdl"], name="Sensor (history)",
                             line=dict(color="#333", width=2)))
    f = v["forecast"]
    fig.add_trace(go.Scatter(x=f["timestamp"] + f["timestamp"][::-1], y=f["g_hi"] + f["g_lo"][::-1], fill="toself",
                             fillcolor="rgba(214,39,40,0.15)", line=dict(width=0), name="95% uncertainty", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["timestamp"], y=f["g_pred"], name="Twin forecast", line=dict(color="#d62728", width=3, dash="dash")))
    if show_truth:
        a = v["actual_future"]
        fig.add_trace(go.Scatter(x=a["timestamp"], y=a["glucose_mgdl"], name="What actually happened",
                                 line=dict(color="#888", width=2, dash="dot")))
    fig.add_hline(y=180, line_dash="dash", line_color="red", annotation_text="High (180)")
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Low (70)")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="mg/dL", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    st.caption("The forecast assumes no future meals or exercise unless you add them in the what-if panel below.")

# ------------------------------------------------------------------ risk panel
with right:
    st.subheader("Risk in the next 2 hours")
    if not ml["high_applicable"]:
        st.error("Glucose is already above 180 mg/dL.")
    else:
        ph, thr = ml["p_high"], ml["threshold_high"]
        status = "HIGH ALERT" if ml["alert_high"] else ("Watch" if ph >= 0.5 * thr else "Low risk")
        color = {"HIGH ALERT": "red", "Watch": "orange", "Low risk": "green"}[status]
        st.markdown(f"### :{color}[{status}]")
        st.metric("Risk score: going above 180", f"{ph:.0%}")
        st.progress(min(max(ph, 0.0), 1.0))
        st.caption(f"Model score, not a calibrated probability (the model was trained with balanced class weights). Alert threshold: {thr:.0%}, chosen to catch about 80% of events on validation patients.")
    if ml["low_applicable"] and ml["p_low"] is not None:
        st.metric("Risk score: going below 70 (exploratory)", f"{ml['p_low']:.0%}")
        st.caption("Low events are rare in the synthetic data, so this estimate is experimental.")
    st.subheader("Why?")
    rs = pd.DataFrame(ml["top_reasons"], columns=["factor", "effect"])
    rs["direction"] = np.where(rs.effect >= 0, "raises risk", "lowers risk")
    rfig = go.Figure(go.Bar(x=rs.effect[::-1], y=rs.factor[::-1], orientation="h",
                            marker_color=["#d62728" if e >= 0 else "#2ca02c" for e in rs.effect[::-1]]))
    rfig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="push on risk (SHAP)")
    st.plotly_chart(rfig, width="stretch")

# ------------------------------------------------------------------ what-if
st.divider()
st.subheader("What if? Try a change on the virtual patient")
w = st.columns(4)
snack = w[0].slider("Planned snack in 20 min (g carbs)", 0, 60, 25, step=5)
skip = w[0].checkbox("Skip that snack")
walk = w[1].slider("Walk, minutes (starts in 10 min)", 0, 45, 20, step=5)
sleep_default = float(np.clip(round((state["sleep_hours_last_24h"] or 7) * 2) / 2, 4, 9))
sleep = w[2].slider("Last night's sleep (hours)", 4.0, 9.0, sleep_default, step=0.5)
res = get_whatif(pid, k, snack, skip, walk, sleep)
base_pk, new_pk = res.g_base.max(), res.g_whatif.max()
w[3].metric("Predicted peak", f"{new_pk:.0f} mg/dL", f"{new_pk - base_pk:+.0f} vs baseline", delta_color="inverse")
w[3].caption("Baseline = planned snack, no walk, actual sleep.")
wf = go.Figure()
wf.add_trace(go.Scatter(x=res.timestamp, y=res.g_base, name="Baseline", line=dict(color="#d62728", width=3)))
wf.add_trace(go.Scatter(x=res.timestamp, y=res.g_whatif, name="What-if", line=dict(color="#2ca02c", width=3)))
wf.add_hline(y=180, line_dash="dash", line_color="red")
wf.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="mg/dL", legend=dict(orientation="h", y=-0.2))
st.plotly_chart(wf, width="stretch")
st.caption("Illustrative simulation. Walk and meal effects come from the twin's equations; the sleep effect is an assumption (about 6% lower insulin sensitivity per hour under 7 h).")

# ------------------------------------------------------------------ results tab
st.divider()
with st.expander("How well does it work? (held-out test patients)"):
    for name, title in [("results_table.csv", "Event prediction (alerts)"), ("forecast_table.csv", "Glucose forecast accuracy")]:
        fp = os.path.join(ROOT, "results", name)
        if os.path.exists(fp):
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(fp), width="stretch")
    for img in ["roc_high.png", "shap_importance.png", "forecast_scatter_60.png"]:
        fp = os.path.join(ROOT, "results", "figures", img)
        if os.path.exists(fp):
            st.image(fp)
st.caption("Data: synthetic EHR and synthetic CGM/wearable data. Proof of concept; not a medical device.")

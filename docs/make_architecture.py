"""
make_architecture.py -- draws the architecture diagram.   Run from the project root:
    python docs\\make_architecture.py
Outputs docs/architecture.pdf and docs/architecture.png  (edit the labels below and rerun to change it)
"""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COL = {"data": "#d6e6f5", "fuse": "#e4e4e4", "twin": "#fde5c8", "ml": "#d9efd5", "ui": "#e5dcf2", "eval": "#f7f7f7"}
fig, ax = plt.subplots(figsize=(16, 8.6))
ax.set_xlim(0, 120); ax.set_ylim(0, 66); ax.axis("off")


def box(x, y, w, h, title, body="", color="#fff", tsize=12.5, bsize=10.4, width=None, bold_edge=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=color,
                                ec="#444" if bold_edge else "#777", lw=1.8 if bold_edge else 1.1))
    wrap = width or max(int(w * 1.25), 12)
    ty = y + h - 1.6
    ax.text(x + w / 2, ty, title, ha="center", va="top", fontsize=tsize, fontweight="bold")
    if body:
        ax.text(x + w / 2, ty - 3.0 - 1.9 * (len(textwrap.wrap(title, wrap)) - 1), textwrap.fill(body, wrap), ha="center", va="top", fontsize=bsize, color="#222")


def arrow(p0, p1, label="", both=False, rad=0.0, lx=0, ly=0.8):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", lw=1.7, color="#333",
                                                       connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly, label, ha="center", fontsize=9.5, style="italic", color="#333",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


def poly(points, label="", lx=0, ly=0.8):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#333", lw=1.7, solid_capstyle="butt")
    arrow(points[-2], points[-1])
    if label:
        mx, my = (points[1][0] + points[2][0]) / 2, (points[1][1] + points[2][1]) / 2
        ax.text(mx + lx, my + ly, label, ha="center", fontsize=8, style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


ax.text(60, 64.6, "Digital Twin for Type 2 Diabetes: System Architecture", ha="center", fontsize=19, fontweight="bold")

# ---- inputs
box(1, 40, 18, 14, "Static data", "Electronic Health Record (synthetic): age, sex, BMI, HbA1c, diabetes years, family history, medication, BP", COL["data"])
box(1, 14, 18, 18, "Dynamic data", "CGM glucose every 5 min, heart rate, steps, sleep stage, meal log (synthetic)", COL["data"])

# ---- fusion
box(24, 14, 16, 40, "Preprocessing & fusion", "Clean and fill short gaps. Align on a 5-minute grid. Engineer features (rolling stats, rate of change, time since meal). Link each sensor window to the patient's profile.", COL["fuse"], width=17)

# ---- twin
ax.add_patch(FancyBboxPatch((44.5, 11.5), 28.5, 44.5, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#fff8ee", ec="#c77a1a", lw=2.2))
ax.text(58.75, 54.6, "DIGITAL TWIN CORE", ha="center", fontsize=13, fontweight="bold", color="#8a4b00")
box(46.5, 40.5, 24.5, 11, "Personalisation", "Health record sets starting insulin sensitivity and basal glucose (literature-informed prior)", COL["twin"])
box(46.5, 26.5, 24.5, 11.5, "Physiological model", "Bergman-style ODE: glucose, insulin, insulin action, 2-stage gut; meal and exercise inputs", COL["twin"])
box(46.5, 13, 24.5, 11.5, "Extended Kalman filter", "Predict with the model, compare with CGM, correct. Learns hidden states and effective insulin sensitivity", COL["twin"])
arrow((58.75, 40.5), (58.75, 38.0))
arrow((56.5, 26.5), (56.5, 24.5), both=True)
arrow((61.0, 26.5), (61.0, 24.5), both=True)

# ---- right-hand blocks
box(78, 42, 19, 12, "2-hour forecast", "Glucose forecast with 95% uncertainty band", COL["twin"])
box(78, 27, 19, 11.5, "What-if engine", "Counterfactual simulation: walk, skip a snack, different sleep", COL["twin"])
box(78, 11.5, 19, 13.5, "ML risk model", "LightGBM on sensor + record + twin outputs. Alert if glucose will leave 70-180 in 2 h. SHAP gives the reasons.", COL["ml"])

# ---- dashboard
box(101, 14, 18.5, 40, "Doctor dashboard", "Streamlit web app: patient summary, history + forecast with uncertainty, risk status with reasons, what-if sliders that re-run the twin.", COL["ui"], tsize=13)

# ---- arrows
arrow((19.3, 47), (23.7, 47))
arrow((19.3, 23), (23.7, 23))
poly([(40.3, 46), (42.5, 46), (42.5, 46), (46.2, 46)], "patient profile", lx=0, ly=1.0)
arrow((40.3, 22), (46.2, 22), "sensor stream", ly=1.2)
arrow((71.2, 47), (77.7, 47))
arrow((71.2, 33), (77.7, 33))
poly([(71.4, 17.5), (77.7, 17.5)])
ax.text(74.5, 18.6, "twin outputs", ha="center", fontsize=9, style="italic", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
poly([(32, 14), (32, 7.2), (87.5, 7.2), (87.5, 11.2)], "sensor + health-record features", lx=-8, ly=0.9)
arrow((97.3, 48), (100.7, 48))
arrow((97.3, 33), (100.7, 33), both=True)
arrow((97.3, 18), (100.7, 18))
ax.text(99, 34.2, "sliders", ha="center", fontsize=9, style="italic")

# ---- evaluation banner
ax.add_patch(FancyBboxPatch((1, 0.6), 118.5, 4.4, boxstyle="round,pad=0.2,rounding_size=1", fc=COL["eval"], ec="#999", lw=1))
ax.text(60, 2.8, "Evaluation: patient-level train / validation / test split  |  baselines  |  ablation (sensor only, + health record, + twin)  |  RMSE, AUROC, sensitivity, precision  |  Clarke error grid  |  SHAP",
        ha="center", va="center", fontsize=10.5)
fig.tight_layout(pad=0.5)
fig.savefig("docs/architecture.pdf"); fig.savefig("docs/architecture.png", dpi=170)
print("Saved docs/architecture.pdf and docs/architecture.png")

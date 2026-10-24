"""
make_results_md.py -- writes the Results tables into README.md straight from results/*.csv,
so the README numbers always match the repo.   Run from the project root (after train.py):

    python src\\make_results_md.py
It replaces everything between <!-- RESULTS_START --> and <!-- RESULTS_END --> in README.md.
"""
import os
import numpy as np
import pandas as pd

R = "results"


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(x, d=2):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


res = pd.read_csv(f"{R}/results_table.csv")
ft = pd.read_csv(f"{R}/forecast_table.csv")
parts = []
for prefix, title, note in [
    ("high", "High glucose: will glucose go above 180 mg/dL in the next 2 hours?",
     "Predicts new events only (patients currently at or below 180). Alert threshold chosen on validation patients to catch about 80% of events."),
    ("low", "Low glucose: will glucose go below 70 mg/dL in the next 2 hours? (EXPLORATORY)",
     "Low events are rare and come from very few patients in the synthetic data, so treat these rows as a prototype, not a result."),
]:
    d = res[res.target.str.startswith(prefix)].copy()
    rows = []
    for _, r in d.iterrows():
        if pd.isna(r.auroc):
            rows.append([r.model, "n/a", "n/a", "n/a", "n/a"]); continue
        ci = f"{fmt(r.ci_lo)} to {fmt(r.ci_hi)}" if "ci_lo" in d and not pd.isna(r.get("ci_lo")) else "n/a"
        rows.append([r.model, fmt(r.auroc), ci, fmt(r.get("sensitivity")), fmt(r.get("precision"))])
    npos = int(d.n_pos_test.iloc[0]) if len(d) else 0
    parts.append(f"**{title}**\n\n{md_table(rows, ['Model', 'AUROC', '95% CI (resampling test patients)', 'Sensitivity', 'Precision'])}\n\n"
                 f"_{note} Test set: {int(d.n_test.iloc[0])} moments, {npos} events._\n")

rows = [[int(r.horizon_min), r.model, f"{r.rmse:.1f}", f"{r.clarke_A_pct:.0f}%", f"{r.clarke_AB_pct:.0f}%"] for _, r in ft.iterrows()]
parts.append("**Glucose forecast accuracy (held-out test patients)**\n\n" +
             md_table(rows, ["Horizon (min)", "Model", "RMSE (mg/dL)", "Clarke zone A", "Clarke zones A+B"]) +
             "\n\n_RMSE = average forecast error (lower is better). Clarke zones A and B are clinically safe errors. "
             "The twin forecast assumes no future meals or exercise._\n")
figs = [("ROC curves", "results/figures/roc_high.png"), ("What drives the alerts (SHAP)", "results/figures/shap_importance.png"),
        ("Forecast vs actual at +60 min", "results/figures/forecast_scatter_60.png")]
parts.append("\n".join(f"![{n}]({p})" for n, p in figs if os.path.exists(p)))
block = "<!-- RESULTS_START -->\n" + "\n".join(parts) + "\n<!-- RESULTS_END -->"

txt = open("README.md", encoding="utf-8").read()
a, b = txt.index("<!-- RESULTS_START -->"), txt.index("<!-- RESULTS_END -->") + len("<!-- RESULTS_END -->")
open("README.md", "w", encoding="utf-8").write(txt[:a] + block + txt[b:])
print("README.md results section updated from results/*.csv")

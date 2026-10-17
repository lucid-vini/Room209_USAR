"""
model.py -- physiological glucose-insulin model (Bergman-style minimal model + gut + exercise)

State vector x = [G, X, I, D1, D2, logS]
  G    plasma glucose (mg/dL)
  X    "insulin action" (scaled x1000, 1/min): how hard insulin is pushing glucose down
  I    plasma insulin ABOVE basal (uU/mL)
  D1   carbs in the stomach (g)        } two-stage gut: a meal enters D1, moves to D2,
  D2   carbs in the intestine (g)      } then appears in blood as Ra
  logS log of the patient's insulin-sensitivity multiplier (1.0 = reference T2D patient)

Inputs per 5-min step: carbs (g eaten at the start of the step) and e (exercise intensity 0..1)
"""
import math
import numpy as np

DT = 5.0   # minutes per step
N_STATE = 6

DEFAULTS = dict(
    p1=0.0015,    # glucose effectiveness: glucose clears itself toward basal (1/min)
    p2=0.0575,    # decay rate of insulin action (1/min)
    p3=0.138,     # how strongly insulin builds insulin action
    n=0.175,      # insulin clearance (1/min)
    gam=0.0133,   # insulin secretion per mg/dL of glucose above basal
    tau_g=33.4,   # gut transit time constant (min)
    gain=2.44,    # mg/dL of glucose rise per gram of carbs (before insulin acts)
    kx=0.61,      # how strongly exercise boosts insulin-independent glucose uptake
    Gb=110.0,     # basal (fasting) glucose, mg/dL (overridden per patient by personalize())
)


def _sp(u):
    """smooth version of max(u, 0) so the filter's Jacobian is well-behaved"""
    return 0.5 * (u + math.sqrt(u * u + 25.0))


def deriv(x, e, P):
    G, X, I, D1, D2, ls = x
    s = math.exp(ls)
    Ra = P["gain"] * D2 / P["tau_g"]
    dG = -P["p1"] * (G - P["Gb"]) - 1e-3 * X * G + Ra
    dX = -P["p2"] * X + P["p3"] * s * I + P["kx"] * e
    dI = -P["n"] * I + P["gam"] * _sp(G - P["Gb"])
    dD1 = -D1 / P["tau_g"]
    dD2 = (D1 - D2) / P["tau_g"]
    return [dG, dX, dI, dD1, dD2, 0.0]


def step(x, carbs, e, P):
    """advance the model by DT minutes with one RK4 step. x is a list of 6 floats."""
    y = list(x)
    y[3] += carbs                                   # meal enters the stomach
    k1 = deriv(y, e, P)
    k2 = deriv([y[i] + 0.5 * DT * k1[i] for i in range(6)], e, P)
    k3 = deriv([y[i] + 0.5 * DT * k2[i] for i in range(6)], e, P)
    k4 = deriv([y[i] + DT * k3[i] for i in range(6)], e, P)
    out = [y[i] + DT / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(6)]
    out[3] = max(out[3], 0.0)
    out[4] = max(out[4], 0.0)
    return out


def simulate(x0, carbs_seq, e_seq, P):
    """run the model open-loop. returns array (len+1, 6)."""
    xs = [list(x0)]
    for c, e in zip(carbs_seq, e_seq):
        xs.append(step(xs[-1], c, e, P))
    return np.array(xs)

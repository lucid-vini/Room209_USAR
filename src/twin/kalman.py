"""
kalman.py -- Extended Kalman Filter (EKF) for the glucose-insulin model

Why EKF? The model is nonlinear (insulin action X multiplies glucose G, and S = exp(logS)).
An EKF linearises the model around the current estimate at every step (that linearisation is the
Jacobian F), then runs the usual Kalman predict/correct loop.

Each step:
  PREDICT: x <- f(x, inputs);  P <- F P F^T + Q      (Q = process noise: "how wrong is my model?")
  UPDATE : K = P H^T (H P H^T + R)^-1                (R = measurement noise: "how wrong is the sensor?")
           x <- x + K (y - H x);  P <- (I-KH) P (I-KH)^T + K R K^T
K (the Kalman gain) decides how much to trust the sensor vs the model.
"""
import numpy as np
from model import step, N_STATE

LS_MIN, LS_MAX = -1.5, 1.5     # allowed range of log-sensitivity


def numerical_jacobian(x, carbs, e, P, fx=None):
    x = np.asarray(x, dtype=float)
    fx = np.array(step(list(x), carbs, e, P)) if fx is None else fx
    F = np.zeros((N_STATE, N_STATE))
    for j in range(N_STATE):
        eps = 1e-4 * max(1.0, abs(x[j]))
        xp = x.copy()
        xp[j] += eps
        F[:, j] = (np.array(step(list(xp), carbs, e, P)) - fx) / eps
    return F, fx


class EKF:
    def __init__(self, params, x0, P0=None, Q=None, R=9.0):
        self.params = params
        self.x = np.array(x0, dtype=float)
        # initial uncertainty: glucose +-5, others small, sensitivity +-0.3 (log scale)
        self.P = np.diag([25.0, 1.0, 1.0, 4.0, 4.0, 0.09]) if P0 is None else np.array(P0, dtype=float)
        # process noise per 5-min step. D1 noise lets the filter absorb UNLOGGED meals.
        self.Q = np.diag([2.0 ** 2, 0.02 ** 2, 0.02 ** 2, 1.5 ** 2, 0.5 ** 2, 0.004 ** 2]) if Q is None else np.array(Q)
        self.R = R
        self.H = np.zeros((1, N_STATE))
        self.H[0, 0] = 1.0

    def predict(self, carbs, e):
        F, fx = numerical_jacobian(self.x, carbs, e, self.params)
        self.x = fx
        self.P = F @ self.P @ F.T + self.Q
        self._fix()

    def update(self, y):
        if y is None or np.isnan(y):
            return 0.0                              # sensor dropout: skip the correction
        S = float((self.H @ self.P @ self.H.T)[0, 0] + self.R)
        K = (self.P @ self.H.T) / S
        innov = y - float(self.x[0])
        self.x = self.x + (K.flatten() * innov)
        I_KH = np.eye(N_STATE) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + (K * self.R) @ K.T     # Joseph form (stays positive-definite)
        self._fix()
        return innov

    def _fix(self):
        self.x[2] = max(self.x[2], 0.0)
        self.x[3] = max(self.x[3], 0.0)
        self.x[4] = max(self.x[4], 0.0)
        self.x[5] = float(np.clip(self.x[5], LS_MIN, LS_MAX))
        self.P = 0.5 * (self.P + self.P.T)

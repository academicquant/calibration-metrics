"""Calibration metrics for binary probability forecasts, in NumPy.

ece(p, y, m)                expected calibration error with m equal-width bins
reliability_table(p, y, m)  per-bin count, mean prediction and observed frequency
brier(p, y)                 mean squared error of the probabilities
log_loss(p, y, eps)         cross-entropy, with p clipped to [eps, 1 - eps]
"""
import numpy as np


def _bins(p, m):
    # p = 1.0 would otherwise land in bin m, one past the end
    return np.minimum((p * m).astype(int), m - 1)


def reliability_table(p, y, m=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("p and y must be 1-d arrays of the same length")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    b = _bins(p, m)
    rows = []
    for k in range(m):
        s = b == k
        n = int(s.sum())
        rows.append((k / m, (k + 1) / m, n, float(p[s].mean()) if n else None, float(y[s].mean()) if n else None))
    return rows


def ece(p, y, m=10):
    n = len(p)
    return sum(c / n * abs(obs - pred) for _, _, c, pred, obs in reliability_table(p, y, m) if c)


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps=1e-12):
    p, y = np.clip(np.asarray(p, float), eps, 1 - eps), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

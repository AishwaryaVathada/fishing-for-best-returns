from __future__ import annotations

import numpy as np


def cvar(x: np.ndarray, alpha: float = 0.1) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    q = np.quantile(x, alpha)
    return float(np.mean(x[x <= q]))


def summarize_returns(returns: np.ndarray) -> dict:
    r = np.asarray(returns, dtype=np.float64)
    return {
        "mean": float(np.mean(r)),
        "std": float(np.std(r)),
        "q05": float(np.quantile(r, 0.05)),
        "q50": float(np.quantile(r, 0.50)),
        "q95": float(np.quantile(r, 0.95)),
        "cvar10": float(cvar(r, 0.10)),
    }

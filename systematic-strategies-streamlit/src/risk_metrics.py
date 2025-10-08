import numpy as np

EPS = 1e-12

def _to_1d(x):
    """Cast to 1D float array and drop NaNs."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[~np.isnan(arr)]

def sharpe_ratio(returns, rf=0.0, eps=EPS):
    r = _to_1d(returns) - rf
    if r.size == 0:
        return 0.0
    denom = np.std(r, ddof=1)  # unbiased std
    return float(np.mean(r) / denom) if denom > eps else 0.0

def sortino_ratio(returns, rf=0.0, eps=EPS):
    r = _to_1d(returns) - rf
    if r.size == 0:
        return 0.0
    downside = r[r < 0.0]
    denom = np.std(downside, ddof=1) if downside.size > 1 else 0.0
    return float(np.mean(r) / denom) if denom > eps else 0.0

def max_drawdown(equity_curve, eps=EPS):
    ec = _to_1d(equity_curve)
    if ec.size == 0:
        return 0.0
    peak = np.maximum.accumulate(ec)
    # guard against divide by zero if the first value is 0
    peak = np.where(peak < eps, eps, peak)
    dd = (ec - peak) / peak
    return float(dd.min())

def value_at_risk(returns, alpha=0.05):
    r = _to_1d(returns)
    if r.size == 0:
        return float("nan")
    # alpha quantile (e.g., 5% left tail)
    return float(np.quantile(r, alpha))

def conditional_var(returns, alpha=0.05):
    r = _to_1d(returns)
    if r.size == 0:
        return float("nan")
    var = value_at_risk(r, alpha)
    tail = r[r <= var]
    return float(tail.mean()) if tail.size > 0 else float(var)

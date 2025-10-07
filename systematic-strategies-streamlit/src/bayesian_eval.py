import numpy as np

def monte_carlo_metrics(returns, n_sim=500):
    metrics = []
    for _ in range(n_sim):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        metrics.append({
            "sharpe": np.mean(sample) / np.std(sample),
            "drawdown": np.min((sample.cumsum() - np.maximum.accumulate(sample.cumsum())))
        })
    return metrics

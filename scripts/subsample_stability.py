import numpy as np
import pandas as pd
from ml_tfhp.models.ml_tfhp_model import MLTFHPModel

df = pd.read_csv('data/real_btcusdt_trades.csv')
t_all = df['time_sec'].values[:10000]

subsamples = [
    ("Full Sample (1-10,000)", t_all[:10000]),
    ("Subsample 1 (1-5,000)", t_all[:5000]),
    ("Subsample 2 (5,001-10,000)", t_all[5000:10000] - t_all[5000]),
]

print("Subsample Stability Analysis:")
print("| Sample Regime | Trades | Window (s) | $\\ln \\hat{L}$ | $\\mu$ | $\\alpha$ | $\\beta$ | $\\gamma$ | $\\eta$ | $n^*$ |")
print("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

for name, ts in subsamples:
    T = float(ts[-1])
    model = MLTFHPModel(M=12)
    model.fit(ts, T=T, n_starts=1, compute_inference=False, seed=42)
    mu, alpha, beta, gamma, eta = model.params
    ll = model.log_likelihood_
    br = model.branching_ratio_
    print(f"| {name} | {len(ts)} | {T:.1f} | {ll:.2f} | {mu:.4f} | {alpha:.4f} | {beta:.4f} | {gamma:.4f} | {eta:.4f} | {br:.4f} |")

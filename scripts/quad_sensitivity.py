import numpy as np
import pandas as pd
from ml_tfhp.models.ml_tfhp_model import MLTFHPModel

df = pd.read_csv('data/real_btcusdt_trades.csv')
t_norm = df['time_sec'].values[:10000]
T = float(t_norm[-1])

print(f"Events: {len(t_norm)}, T = {T:.2f} s")

results = {}
for M in [8, 10, 12, 16, 20]:
    model = MLTFHPModel(M=M)
    model.fit(t_norm, T=T, n_starts=1, compute_inference=False, seed=42)
    mu, alpha, beta, gamma, eta = model.params
    ll = model.log_likelihood_
    br = model.branching_ratio_
    results[M] = (ll, mu, alpha, beta, gamma, eta, br)

ref_ll = results[20][0]
print("\nQuadrature Sensitivity across M in [8, 10, 12, 16, 20]:")
print("| $M$ | $\\ln \\hat{L}$ | $\\mu$ | $\\alpha$ | $\\beta$ | $\\gamma$ | $\\eta$ | $n^*$ | Rel Error vs $M=20$ |")
print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
for M in [8, 10, 12, 16, 20]:
    ll, mu, a, b, g, e, br = results[M]
    rel_err = abs(ll - ref_ll) / abs(ref_ll)
    print(f"| {M:2d} | {ll:.2f} | {mu:.4f} | {a:.4f} | {b:.4f} | {g:.4f} | {e:.4f} | {br:.4f} | {rel_err:.2e} |")

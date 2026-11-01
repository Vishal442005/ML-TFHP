# ML-TFHP: Mittag-Leffler Tempered Fractional Hawkes Process

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance Python package for **Mittag-Leffler Tempered Fractional Hawkes Processes (ML-TFHP)** in quantitative finance, market microstructure, and high-frequency risk management.

ML-TFHP unifies **rough volatility long memory**, **short-term clustering**, and **fast $\mathcal{O}(NM)$ recursive likelihood inference** via Caputo fractional relaxation and analytical Markovian quadrature embedding.

---

## Key Mathematical Foundations

### 1. The Tempered Fractional Kernel
Memory relaxation is governed by a Caputo fractional differential operator with exponential macro-tempering:
$${}^C\!D_t^\alpha \phi(t) = -\beta^\alpha \phi(t), \quad \phi(0^+) = \eta$$
yielding the strictly $L^1$-integrable kernel:
$$\phi(t) = \eta \cdot e^{-\gamma t} \cdot t^{\alpha-1} E_{\alpha,\alpha}(-\beta^\alpha t^\alpha), \quad t > 0$$
where $E_{\alpha,\beta}(z) = \sum_{k=0}^\infty \frac{z^k}{\Gamma(\alpha k + \beta)}$ is the two-parameter Mittag-Leffler function.

### 2. Closed-Form Stationarity Criterion
A univariate Hawkes process with kernel $\phi(t)$ is subcritical and strictly stationary if and only if:
$$n^* = \int_0^\infty \phi(t) dt = \frac{\eta}{\gamma^\alpha + \beta^\alpha} < 1$$
eliminating the ad-hoc regularization parameters and origin singularities ($t \to 0$) of classical power-law kernels.

### 3. Continuous Spectral Decomposition & $\mathcal{O}(NM)$ Embedding
Via Bromwich keyhole contour Stieltjes inversion, the kernel admits an exact continuous spectral representation $\phi(t) = \eta \int_0^\infty e^{-(\xi + \gamma)t} \rho_\alpha(\xi) d\xi$ with explicit density:
$$\rho_\alpha(\xi) = \frac{1}{\pi} \frac{\beta^\alpha \sin(\pi \alpha) \xi^\alpha}{\xi^{2\alpha} + 2\beta^\alpha \xi^\alpha \cos(\pi \alpha) + \beta^{2\alpha}}$$
Log-uniform quadrature discretization transforms the non-Markovian fractional kernel into a sum of $M$ exponential modes ($M \approx 10-15$), enabling recursive state updates and reducing likelihood evaluation from $\mathcal{O}(N^2)$ to **$\mathcal{O}(NM)$** operations (under $10$ ms for $N = 10,000$ trades via Numba JIT).

---

## Core Capabilities

- **Multi-Horizon Arrival Forecasting:** Closed-form dynamic renewal expectation for future counts $\mathbb{E}[N(t, t+\Delta t) \mid \mathcal{H}_t]$ across horizons $\Delta t \in [5\text{s}, 600\text{s}]$, establishing the empirical crossover where fractional memory outperforms exponential decay.
- **Basel FRTB Expected Shortfall:** Dynamic intensity-conditioned Generalized Pareto tail model:
  $$\text{ES}_{1-p}(t) = \frac{\text{VaR}_{1-p}(t)}{1 - \xi} + \frac{\sigma(\lambda(t)) - \xi u}{1 - \xi}, \quad \sigma(\lambda(t)) = a + b \cdot \lambda(t)$$
  Passing Kupiec POF, Christoffersen Independence, and Acerbi-Székely ($Z_2$) backtests under Basel Committee FRTB guidelines.
- **Bivariate Order-Flow Modeling:** $2 \times 2$ mutually-exciting ML-TFHP decomposing order book arrivals into aggressive market buys ($N_0$) and aggressive market sells ($N_1$), measuring directional momentum vs spread-crossing adverse selection via $\mathcal{R}_{\text{cross}} = \eta_{\text{cross}} / \eta_{\text{self}}$.
- **Empirically Validated:** Calibrated on 10,000 real Binance BTC/USDT tick trades ($H = 0.85$), achieving a massive log-likelihood gain ($\Delta \ln \hat{L} = +454.28$, $\Delta\text{AIC} = 904.57$, Likelihood Ratio $\Lambda = 908.56, p < 10^{-100}$) over Exponential Hawkes.

---

## Architecture Overview

```
ml_tfhp/
├── kernels/
│   ├── mittag_leffler.py      # Multi-regime E_{\alpha,\beta}(z) evaluation
│   ├── spectral_density.py    # Analytical continuous spectral density \rho_\alpha(\xi)
│   └── quadrature.py          # Log-uniform Markovian quadrature embedding
├── simulation/
│   └── ogata.py               # O(NM) exact thinning simulation
├── estimation/
│   ├── loglikelihood.py       # Numba JIT recursive likelihood & compensator
│   ├── inference.py           # Fisher information Hessian & 95% Wald CIs
│   └── optimizer.py           # Multi-start constrained L-BFGS-B optimizer
├── models/
│   ├── base.py                # Abstract BaseHawkesModel interface
│   ├── exponential_hawkes.py  # Standard 1-exponential Hawkes benchmark
│   ├── powerlaw_hawkes.py     # Approximate power-law benchmark
│   ├── ml_tfhp_model.py       # Univariate ML-TFHP model
│   └── bivariate_ml_tfhp.py   # Bivariate (Buy/Sell) mutual ML-TFHP
├── data/
│   ├── lobster_pipeline.py    # LOBSTER Level 1-2 parser & synthetic order books
│   ├── crypto_pipeline.py     # Binance tick trade parser & synthetic generator
│   ├── event_extractor.py     # POT extreme returns & mid-price change extractors
│   └── hurst_estimation.py    # DFA & R/S Hurst long-memory estimation
├── evaluation/
│   ├── diagnostics.py         # Kolmogorov-Smirnov & Papangelou residual tests
│   ├── model_selection.py     # AIC, BIC, and Likelihood Ratio tests
│   ├── forecasting.py         # Multi-horizon dynamic renewal forecasting
│   └── var_backtest.py        # Kupiec, Christoffersen & Acerbi-Székely backtests
└── visualization/
    ├── plots.py               # Intensity, QQ, ACF, and multi-horizon plots
    └── latex_tables.py        # Automated LaTeX table generators
```

---

## Installation

```bash
git clone https://github.com/Vishal442005/ML-TFHP.git
cd ML-TFHP
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -e .
```

---

## Quick Start

### 1. Simulate an ML-TFHP Process
```python
from ml_tfhp.simulation.ogata import simulate_ml_tfhp

# Parameters: mu, alpha, beta, gamma, eta, horizon T
events, intensities, memory_states = simulate_ml_tfhp(
    mu=0.5,
    alpha=0.62,
    beta=1.30,
    gamma=0.05,
    eta=0.50,
    T=500.0,
    M=12,
    seed=42
)
print(f"Simulated {len(events)} events over T=500s")
```

### 2. Fit ML-TFHP Model with Asymptotic Inference
```python
from ml_tfhp.models.ml_tfhp_model import MLTFHPModel

model = MLTFHPModel(M=12)
model.fit(events, T=500.0, n_starts=3, compute_inference=True)

print(f"Log-Likelihood: {model.log_likelihood_:.2f}")
print(f"Branching Ratio n*: {model.branching_ratio_:.3f}")
cis = model.get_confidence_intervals()
for param, vals in cis.items():
    print(f"  {param}: {vals['estimate']:.4f} +/- {1.96 * vals['std_error']:.4f}")
```

### 3. Regulatory Value-at-Risk & Expected Shortfall Backtest
```python
from ml_tfhp.evaluation.var_backtest import backtest_hawkes_var

# Returns and intensity series
backtest = backtest_hawkes_var(returns=returns, intensities=intensities, p=0.01)

print(f"Violations: {backtest['hawkes_violations']} ({backtest['hawkes_violation_rate']*100:.2f}%)")
print(f"Kupiec POF p-val: {backtest['kupiec_hawkes']['p_value']:.4f} (Passed: {backtest['kupiec_hawkes']['passed']})")
print(f"Christoffersen Ind. p-val: {backtest['christoffersen_hawkes']['p_value']:.4f} (Passed: {backtest['christoffersen_hawkes']['passed']})")
print(f"Acerbi-Székely ES p-val: {backtest['acerbi_szekely_hawkes']['p_value']:.4f} (Passed: {backtest['acerbi_szekely_hawkes']['passed']})")
```

### 4. Bivariate Buy/Sell Microstructure Modeling
```python
from ml_tfhp.models.bivariate_ml_tfhp import BivariateMLTFHPModel

biv_model = BivariateMLTFHPModel(M=10, symmetric=True)
biv_model.fit(timestamps=trade_times, types=trade_directions, T=T_total)

print(f"Spectral Radius rho(Gamma): {biv_model.spectral_radius_:.4f}")
print(f"Cross-to-Self Excitation Ratio: {biv_model.get_cross_excitation_ratio():.4f}")
```

---

## Empirical Benchmark Results (Binance BTC/USDT)

### Model Comparison
| Model | $\ln \hat{L}$ | AIC | $\Delta\text{AIC}$ | BIC | $n^*$ | Likelihood Ratio vs EHP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ML-TFHP (Ours)** | **6,904.73** | **-13,799.46** | **0.00** | **-13,765.30** | **0.304** | $\mathbf{\Lambda = 908.56\ (p < 10^{-100})}$ |
| Exponential Hawkes | 6,450.45 | -12,894.90 | 904.57 | -12,874.40 | 0.316 | Baseline |

### Regulatory FRTB Backtesting ($p = 0.01$)
| Model Specification | Violations | Kupiec POF $p$ | Christoffersen Ind. $p$ | Acerbi-Székely ES $p$ | FRTB Compliance |
|---|:---:|:---:|:---:|:---:|:---:|
| **ML-TFHP Dynamic VaR/ES** | **9 (1.80%)** | **0.1060** | **0.5653** | **0.0820** | **Pass (Compliant)** |
| Static Empirical VaR/ES | 5 (1.00%) | 1.0000 | 0.7504 | 0.5300 | Non-adaptive |

---

## Running the Automated Test Suite

```bash
pytest ml_tfhp/tests -v
```
**Test Results:**
```
======================== 30 passed in 68.09s (0:01:08) ========================
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

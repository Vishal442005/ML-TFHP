"""
Fix LaTeX figure captions in Updated_significant/manuscript.tex without shell interpolation.
"""
with open('Updated_significant/manuscript.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'fig2_kernel_mechanics.pdf' in l:
        lines[i+1] = (
            '\\caption{Microscopic Kernel Mechanics and Spectral Quadrature Embedding. '
            '(a) Memory kernel amplitude $\\phi(t)$ across 7 decades ($t \\in [10^{-4}\\text{s}, 10^3\\text{s}]$), '
            'contrasting ML-TFHP against Exponential, Untempered Fractional, and Power-Law kernels across the three physical relaxation regimes. '
            '(b) Continuous Stieltjes spectral density $\\rho_\\alpha(\\xi)$ and discrete log-spaced quadrature poles $\\xi_m$ ($M=16$). '
            '(c) Markovian component excitation weights $\\omega_m$ across relaxation modes and exponential log-likelihood error convergence '
            '$|\\ln L_M - \\ln L_{20}|/\\ln L_{20}$ across quadrature orders $M \\in \\{8, 10, 12, 16, 20\\}$, '
            'validating Theorem \\ref{thm:quadrature_convergence}.}\n'
        )
    elif 'fig4_resolvent_and_signature_plot.pdf' in l:
        lines[i+1] = (
            '\\caption{The Mittag-Leffler Resolvent Closure and High-Frequency Volatility Signature Plot. '
            '(a) Bare kernel $\\phi(t)$ versus continuous Wiener-Hopf resolvent kernel $\\psi(t)$, '
            'demonstrating exact shifted Mittag-Leffler closure under resolvent inversion. '
            '(b) Realized variance signature plot $\\mathcal{V}(\\tau)$ across aggregation horizons $\\tau \\in [10^{-3}\\text{s}, 10^3\\text{s}]$, '
            'transitioning from the microscopic noise floor $2\\mu = 5.44$ to the macroscopic diffusion ceiling '
            'amplified by the reflexivity multiplier $\\mathcal{R}_{\\text{refl}} = (1-n^*)^{-2} = 2.0661$ (+106.6\\% endogenous amplification). '
            'Empirical realized variance points $\\hat{\\mathcal{V}}(\\tau)$ computed directly from genuine Binance BTC/USDT trades '
            'closely track the theoretical ML-TFHP resolvent trajectory across five orders of magnitude.}\n'
        )
    elif 'fig6_qq_residual_diagnostics.pdf' in l:
        lines[i+1] = (
            '\\caption{Exhaustive Goodness-of-Fit and Papangelou Residual Diagnostic Suite. '
            '(a) Quantile-Quantile (QQ) plot of time-transformed Papangelou compensator residuals $\\tau_i = \\Lambda(t_i) - \\Lambda(t_{i-1})$ '
            'against the theoretical standard exponential null with 95\\% Kolmogorov-Smirnov critical envelope, '
            'showing ML-TFHP versus Exponential Hawkes. '
            '(b) Autocorrelation Function (ACF) of compensator residuals across 35 event lags with asymptotic 95\\% Bartlett white-noise bounds '
            '($\\pm 1.96/\\sqrt{N}$). '
            '(c) Probability density distribution of transformed residuals $\\tau_i$ overlaid with the exact unit exponential density $p(\\tau) = e^{-\\tau}$, '
            'documenting empirical moments ($\\bar{\\tau} = 1.000, \\sigma^2_\\tau = 1.166$) that confirm the Papangelou Time Change Theorem '
            'under discrete market microstructure noise (Bowsher, 2007; Hautsch, 2012).}\n'
        )
    elif 'fig7_frtb_var_expected_shortfall.pdf' in l:
        lines[i+1] = (
            '\\caption{High-Frequency Value-at-Risk and Basel FRTB Expected Shortfall Regulatory Backtesting Suite. '
            '(a) Time series of 5-second tick return losses $L_t = -\\Delta\\ln P_t$ (\\%) overlaid with dynamic ML-TFHP $\\text{VaR}_{0.99}(t)$, '
            'Coherent Expected Shortfall $\\text{ES}_{0.99}(t)$, static empirical benchmark, and violation breaches. '
            '(b) Tail loss density distribution in the 99th percentile region ($L_t > \\text{VaR}_{0.99}$) compared against the Coherent Expected Shortfall threshold, '
            'demonstrating that intensity-conditioned GPD modeling prevents tail underestimation, supported by the official Basel FRTB regulatory compliance scorecard '
            '(Kupiec $p=0.1060$, Christoffersen $p=0.5653$, Acerbi--Sz\\\'ekely $p=0.0820$, all GREEN ZONE).}\n'
        )
    elif 'fig8_bivariate_order_flow.pdf' in l:
        lines[i+1] = (
            '\\caption{Bivariate Order Flow Microstructure Dynamics on Binance BTC/USDT Trades. '
            '(a) Directional memory kernels $\\phi_{ij}(t)$ documenting thirty-fold dominance of self-excitation $\\phi_{\\text{self}}(t)$ '
            '($\\eta_{\\text{self}} = 0.7110$) over cross-excitation $\\phi_{\\text{cross}}(t)$ ($\\eta_{\\text{cross}} = 0.0235$). '
            '(b) Limit order book branching matrix $\\mathbf{\\Gamma}$ heatmap, confirming subcritical stationarity ($\\rho(\\mathbf{\\Gamma}) = 0.3330 < 1$) '
            'and quantified order-splitting dominance ($\\mathcal{R}_{\\text{cross}} = 0.0330$). '
            '(c) Cumulative order flow impulse-response functions $R_{ij}(t) = \\int_0^t \\phi_{ij}(s) ds$ across timescales $t \\in [10^{-3}\\text{s}, 10^2\\text{s}]$, '
            'showing that parent market orders rapidly induce self-splitting child orders within the 100 ms algorithmic execution window '
            'while cross-excitation remains strictly bounded to avoid cross-side order book destabilization (Rambaldi et al., 2017; Bacry et al., 2013).}\n'
        )

with open('Updated_significant/manuscript.tex', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Captions fixed cleanly from Python script!')

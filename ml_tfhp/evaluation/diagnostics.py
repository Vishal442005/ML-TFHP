"""
Residual Diagnostics based on the Papangelou Random Time Change Theorem.

Tests:
1. Kolmogorov-Smirnov (KS) test on transformed residuals u_i = 1 - exp(-tau_i) ~ Uniform(0,1).
2. Ljung-Box test for serial autocorrelation in residuals.
3. QQ plot coordinate computation vs theoretical Exp(1) quantiles.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
from scipy.stats import kstest, chi2


def ks_test_residuals(residuals: np.ndarray) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test on compensator increments tau_i against Exp(1).

    Parameters
    ----------
    residuals : ndarray
        1D array of non-negative compensator residuals tau_i.

    Returns
    -------
    ks_stat : float
        Kolmogorov-Smirnov test statistic.
    p_value : float
        Two-sided p-value (p > 0.05 indicates failure to reject null of Exp(1)).
    """
    res = np.asarray(residuals, dtype=np.float64)
    res_clean = res[res > 0]
    # Transform to Uniform(0, 1)
    u = 1.0 - np.exp(-res_clean)
    ks_res = kstest(u, "uniform")
    return float(ks_res.statistic), float(ks_res.pvalue)


def ljung_box_residuals(residuals: np.ndarray, lags: List[int] = None) -> Dict[int, Dict[str, float]]:
    """
    Compute Ljung-Box test for absence of autocorrelation in residuals at specified lags.

    Q(h) = n * (n + 2) * sum_{k=1}^h rho_k^2 / (n - k) ~ chi2(h)
    """
    if lags is None:
        lags = [5, 10, 20]

    res = np.asarray(residuals, dtype=np.float64)
    n = len(res)
    mean_res = np.mean(res)
    var_res = np.var(res)

    if var_res < 1e-12:
        return {lag: {"stat": 0.0, "p_value": 1.0} for lag in lags}

    # Compute sample autocorrelations up to max(lags)
    max_lag = max(lags)
    rho = np.empty(max_lag, dtype=np.float64)
    for k in range(1, max_lag + 1):
        cov_k = np.sum((res[:-k] - mean_res) * (res[k:] - mean_res)) / n
        rho[k - 1] = cov_k / var_res

    results = {}
    for h in lags:
        q_stat = 0.0
        for k in range(1, h + 1):
            q_stat += (rho[k - 1] ** 2) / (n - k)
        q_stat *= n * (n + 2)
        p_val = 1.0 - chi2.cdf(q_stat, df=h)
        results[h] = {"stat": float(q_stat), "p_value": float(p_val)}

    return results


def compute_qq_plot_data(residuals: np.ndarray, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute theoretical vs empirical quantiles of residuals for QQ-plotting against Exp(1).

    Returns
    -------
    theoretical_quantiles : ndarray
        -ln(1 - p_k)
    empirical_quantiles : ndarray
        Sorted sample quantiles.
    """
    res = np.sort(np.asarray(residuals, dtype=np.float64))
    n = len(res)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = -np.log(1.0 - probs)

    if n > n_points:
        indices = np.linspace(0, n - 1, n_points, dtype=int)
        return theoretical_quantiles[indices], res[indices]
    return theoretical_quantiles, res


def evaluate_goodness_of_fit(residuals: np.ndarray) -> Dict[str, Any]:
    """
    Run complete diagnostic test battery on model residuals.
    """
    ks_stat, ks_pval = ks_test_residuals(residuals)
    lb_dict = ljung_box_residuals(residuals, lags=[5, 10, 20])
    mean_val = float(np.mean(residuals))
    var_val = float(np.var(residuals))

    return {
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pval,
        "mean_residual": mean_val,
        "var_residual": var_val,
        "ljung_box": lb_dict,
        "passes_ks": ks_pval > 0.05,
    }

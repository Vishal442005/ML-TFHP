"""
Statistical Model Comparison: AIC, BIC, Likelihood Ratio Tests, and Vuong Non-Nested Tests.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.stats import chi2, norm


def likelihood_ratio_test(
    ll_restricted: float,
    ll_full: float,
    df: int = 1
) -> Tuple[float, float]:
    """
    Compute Likelihood Ratio Test (LRT) statistic and p-value for nested models:
        Lambda = 2 * (LL_full - LL_restricted) ~ chi2(df)

    Parameters
    ----------
    ll_restricted : float
        Log-likelihood of simpler (nested) model.
    ll_full : float
        Log-likelihood of richer (unrestricted) model.
    df : int
        Difference in number of free parameters.

    Returns
    -------
    lr_stat : float
    p_value : float
    """
    lr_stat = 2.0 * max(0.0, ll_full - ll_restricted)
    p_val = 1.0 - chi2.cdf(lr_stat, df=df)
    return float(lr_stat), float(p_val)


def vuong_test(
    pointwise_ll_1: np.ndarray,
    pointwise_ll_2: np.ndarray
) -> Tuple[float, float, str]:
    """
    Vuong test for non-nested model selection.

    m_i = ll_1(i) - ll_2(i)
    V = sqrt(N) * mean(m) / std(m) ~ N(0, 1)

    Returns
    -------
    v_stat : float
    p_value : float
    preferred_model : str ('Model 1', 'Model 2', or 'Indistinguishable')
    """
    m = np.asarray(pointwise_ll_1) - np.asarray(pointwise_ll_2)
    n = len(m)
    mean_m = np.mean(m)
    std_m = np.std(m, ddof=1)

    if std_m < 1e-12:
        return 0.0, 1.0, "Indistinguishable"

    v_stat = np.sqrt(n) * (mean_m / std_m)
    p_val = 2.0 * (1.0 - norm.cdf(abs(v_stat)))

    if p_val < 0.05:
        preferred = "Model 1" if v_stat > 0 else "Model 2"
    else:
        preferred = "Indistinguishable"

    return float(v_stat), float(p_val), preferred


def compare_models(models: list) -> List[Dict[str, Any]]:
    """
    Tabulate comparative metrics across multiple fitted Hawkes process models.
    """
    rows = []
    min_aic = min(m.aic_ for m in models if m.is_fitted)
    min_bic = min(m.bic_ for m in models if m.is_fitted)

    for m in models:
        if not m.is_fitted:
            continue
        resids = m.compute_residuals()
        from .diagnostics import ks_test_residuals
        ks_stat, ks_p = ks_test_residuals(resids)

        rows.append({
            "model": m.name,
            "log_likelihood": float(m.log_likelihood_),
            "aic": float(m.aic_),
            "delta_aic": float(m.aic_ - min_aic),
            "bic": float(m.bic_),
            "delta_bic": float(m.bic_ - min_bic),
            "branching_ratio": float(m.branching_ratio_),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "passes_ks": ks_p > 0.05,
        })

    # Sort by AIC ascending
    rows.sort(key=lambda x: x["aic"])
    return rows

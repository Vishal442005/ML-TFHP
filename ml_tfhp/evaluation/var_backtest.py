"""
High-Frequency Hawkes-Driven Value-at-Risk (VaR) and Regulatory Backtesting.

Connects conditional intensity lambda(t) to tail risk via Generalized Pareto Distribution (GPD)
and performs Kupiec POF and Christoffersen Independence tests.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.stats import genpareto, chi2
from scipy.optimize import minimize


def fit_intensity_gpd(
    returns: np.ndarray,
    intensities: np.ndarray,
    quantile: float = 0.90
) -> Tuple[float, float, float, float]:
    """
    Fit an intensity-conditioned Generalized Pareto Distribution (GPD) to return tails.

    Returns exceedances Y = loss - u | loss >= u follow:
        GPD(shape=xi, scale=sigma(lambda))
    where sigma(lambda) = a + b * lambda.

    Parameters
    ----------
    returns : ndarray
        1D array of asset log-returns.
    intensities : ndarray
        1D array of conditional intensities lambda(t_i) aligned with returns.
    quantile : float
        Tail threshold quantile (default 0.90).

    Returns
    -------
    a : float
        Base scale coefficient (> 0).
    b : float
        Intensity scale sensitivity coefficient (>= 0).
    xi : float
        GPD shape parameter (xi > 0 indicates heavy tails).
    u : float
        Threshold cutoff.
    """
    losses = -np.asarray(returns, dtype=np.float64)
    lams = np.asarray(intensities, dtype=np.float64)

    # Threshold cutoff
    u = float(np.percentile(losses, quantile * 100))
    tail_mask = losses >= u
    y = losses[tail_mask] - u
    tail_lams = lams[tail_mask]

    if len(y) < 20:
        raise ValueError(f"Too few tail observations ({len(y)}) to fit GPD.")

    # Unconditional initial guess
    fit_shape, _, fit_scale = genpareto.fit(y, floc=0)
    init_a = max(fit_scale * 0.5, 1e-4)
    init_b = max(fit_scale * 0.5 / np.mean(tail_lams), 1e-6)
    init_xi = np.clip(fit_shape, 0.01, 0.8)

    def gpd_nll(theta):
        a_p, b_p, xi_p = theta
        if a_p <= 1e-6 or b_p < 0.0 or xi_p <= -0.5 or xi_p > 1.5:
            return 1e15

        scales = a_p + b_p * tail_lams
        if np.any(scales <= 1e-8):
            return 1e15

        z = (xi_p * y) / scales
        if np.any(1.0 + z <= 0.0):
            return 1e15

        # Log-likelihood: - sum [ ln(scale) + (1 + 1/xi) * ln(1 + xi * y / scale) ]
        if abs(xi_p) < 1e-6:
            # Exponential limit
            ll = - np.sum(np.log(scales) + y / scales)
        else:
            ll = - np.sum(np.log(scales) + (1.0 + 1.0 / xi_p) * np.log(1.0 + z))

        return -ll

    res = minimize(
        gpd_nll,
        np.array([init_a, init_b, init_xi]),
        method="L-BFGS-B",
        bounds=[(1e-6, 1.0), (0.0, 1.0), (-0.2, 1.2)]
    )

    a_hat, b_hat, xi_hat = res.x
    return float(a_hat), float(b_hat), float(xi_hat), u


def compute_hawkes_var(
    intensities: np.ndarray,
    a: float,
    b: float,
    xi: float,
    u: float,
    p: float = 0.01,
    p_u: float = 0.10
) -> np.ndarray:
    """
    Compute time-varying Value-at-Risk VaR_{1-p}(t) conditioned on intensity lambda(t).

    Formula:
        VaR_{1-p}(t) = u + [sigma(lambda(t)) / xi] * [ (p / p_u)^(-xi) - 1 ]
    where p_u = P(Loss >= u) is the tail threshold proportion.
    """
    scales = a + b * np.asarray(intensities, dtype=np.float64)
    if abs(xi) < 1e-6:
        # Exponential tail limit
        var_vals = u + scales * np.log(p_u / p)
    else:
        var_vals = u + (scales / xi) * (((p_u / p) ** xi) - 1.0)
    return var_vals


def compute_hawkes_expected_shortfall(
    intensities: np.ndarray,
    a: float,
    b: float,
    xi: float,
    u: float,
    p: float = 0.01,
    p_u: float = 0.10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute joint time-varying Value-at-Risk VaR_{1-p}(t) and coherent Expected Shortfall ES_{1-p}(t).

    Formula (McNeil, Frey, Embrechts 2015):
        ES_{1-p}(t) = E[Loss | Loss > VaR_{1-p}(t)] = VaR_{1-p}(t) / (1 - xi) + (sigma(t) - xi * u) / (1 - xi)
    where sigma(t) = a + b * lambda(t) and xi < 1.
    """
    scales = a + b * np.asarray(intensities, dtype=np.float64)
    if abs(xi) < 1e-6:
        var_vals = u + scales * np.log(p_u / p)
        es_vals = var_vals + scales
    else:
        var_vals = u + (scales / xi) * (((p_u / p) ** xi) - 1.0)
        es_vals = (var_vals / (1.0 - xi)) + (scales - xi * u) / (1.0 - xi)
    return var_vals, es_vals


def acerbi_szekely_test(
    losses: np.ndarray,
    var_series: np.ndarray,
    es_series: np.ndarray,
    xi: float,
    a: float,
    b: float,
    intensities: np.ndarray,
    p: float = 0.01,
    p_u: float = 0.10,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> Tuple[float, float, bool]:
    """
    Acerbi-Szekely (2014) Expected Shortfall Backtest (Statistic Z_2).
    Evaluates coherent tail risk under the Basel FRTB regulatory mandate.

    Statistic:
        Z_2 = (1 / (p * T)) * sum_{t=1}^T [ (Loss_t / ES_t) * 1_{Loss_t > VaR_t} ] - 1
    Under H0, E[Z_2] = 0. If Z_2 > 0 and p_value < 0.05, tail risk is systematically underestimated.
    """
    losses = np.asarray(losses, dtype=np.float64)
    T = len(losses)
    violations = losses > var_series
    z2_stat = float((1.0 / (p * T)) * np.sum((losses[violations] / es_series[violations])) - 1.0)

    # Parametric GPD null simulation
    rng = np.random.RandomState(random_state)
    sim_stats = np.empty(n_bootstrap, dtype=np.float64)
    scales = a + b * np.asarray(intensities, dtype=np.float64)
    sigma_var = scales * ((p / p_u) ** (-xi)) if abs(xi) >= 1e-6 else scales

    for b_idx in range(n_bootstrap):
        # Sample indicator of violation
        is_viol = rng.binomial(1, p, size=T)
        n_viol = int(np.sum(is_viol))
        if n_viol == 0:
            sim_stats[b_idx] = -1.0
            continue

        viol_idx = np.where(is_viol == 1)[0]
        # Sample excess losses from GPD(xi, sigma_var)
        if abs(xi) < 1e-6:
            excess = rng.exponential(scale=sigma_var[viol_idx])
        else:
            excess = genpareto.rvs(xi, scale=sigma_var[viol_idx], random_state=rng)

        sim_losses = var_series[viol_idx] + excess
        sim_stats[b_idx] = (1.0 / (p * T)) * np.sum(sim_losses / es_series[viol_idx]) - 1.0

    p_value = float(np.mean(sim_stats >= z2_stat))
    passed = p_value > 0.05
    return z2_stat, p_value, passed



def kupiec_pof_test(
    violations: int,
    total_obs: int,
    p: float = 0.01
) -> Tuple[float, float, bool]:
    """
    Kupiec Proportion of Failures (POF) Unconditional Coverage Test.

    H0: Expected violation proportion equals p.
    LR_POF = 2 * [ V * ln(p_hat / p) + (N - V) * ln((1 - p_hat) / (1 - p)) ] ~ chi2(1)

    Returns
    -------
    lr_stat : float
    p_value : float
    passes_test : bool (True if p_value > 0.05)
    """
    V = violations
    N = total_obs
    p_hat = V / N

    if V == 0:
        lr_stat = -2.0 * N * np.log(1.0 - p)
    elif V == N:
        lr_stat = -2.0 * N * np.log(p)
    else:
        term1 = V * np.log(p_hat / p)
        term2 = (N - V) * np.log((1.0 - p_hat) / (1.0 - p))
        lr_stat = 2.0 * (term1 + term2)

    lr_stat = max(0.0, float(lr_stat))
    p_val = float(1.0 - chi2.cdf(lr_stat, df=1))
    return lr_stat, p_val, p_val > 0.05


def christoffersen_independence_test(
    violations_series: np.ndarray
) -> Tuple[float, float, bool]:
    """
    Christoffersen Independence Test for clustering of VaR exceedances.

    Constructs transition counts matrix:
        n_00: 0 -> 0 (no violation followed by no violation)
        n_01: 0 -> 1 (no violation followed by violation)
        n_10: 1 -> 0 (violation followed by no violation)
        n_11: 1 -> 1 (violation followed by violation - clustered failures!)

    LR_ind = 2 * [ ln L_unrestricted - ln L_restricted ] ~ chi2(1)
    """
    v = np.asarray(violations_series, dtype=int)
    v_prev = v[:-1]
    v_curr = v[1:]

    n00 = int(np.sum((v_prev == 0) & (v_curr == 0)))
    n01 = int(np.sum((v_prev == 0) & (v_curr == 1)))
    n10 = int(np.sum((v_prev == 1) & (v_curr == 0)))
    n11 = int(np.sum((v_prev == 1) & (v_curr == 1)))

    # Probability of violation given no prior violation
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    # Probability of violation given prior violation
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    # Unconditional probability
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Log-likelihood unrestricted
    ll_unrestricted = 0.0
    if n00 > 0 and pi01 < 1.0:
        ll_unrestricted += n00 * np.log(1.0 - pi01)
    if n01 > 0 and pi01 > 0.0:
        ll_unrestricted += n01 * np.log(pi01)
    if n10 > 0 and pi11 < 1.0:
        ll_unrestricted += n10 * np.log(1.0 - pi11)
    if n11 > 0 and pi11 > 0.0:
        ll_unrestricted += n11 * np.log(pi11)

    # Log-likelihood restricted (independence: pi01 == pi11 == pi)
    ll_restricted = 0.0
    if (n00 + n10) > 0 and pi < 1.0:
        ll_restricted += (n00 + n10) * np.log(1.0 - pi)
    if (n01 + n11) > 0 and pi > 0.0:
        ll_restricted += (n01 + n11) * np.log(pi)

    lr_stat = max(0.0, 2.0 * (ll_unrestricted - ll_restricted))
    p_val = float(1.0 - chi2.cdf(lr_stat, df=1))
    return float(lr_stat), p_val, p_val > 0.05


def backtest_hawkes_var(
    returns: np.ndarray,
    intensities: np.ndarray,
    p: float = 0.01,
    quantile: float = 0.90
) -> Dict[str, Any]:
    """
    Run complete high-frequency VaR backtesting suite.
    """
    losses = -np.asarray(returns, dtype=np.float64)
    lams = np.asarray(intensities, dtype=np.float64)
    N = len(losses)

    # 1. Fit intensity-conditioned GPD on in-sample returns
    a_hat, b_hat, xi_hat, u = fit_intensity_gpd(returns, lams, quantile=quantile)

    # 2. Compute dynamic Hawkes-VaR and Expected Shortfall series
    p_u = 1.0 - quantile
    var_series, es_series = compute_hawkes_expected_shortfall(lams, a_hat, b_hat, xi_hat, u, p=p, p_u=p_u)

    # Also compute static empirical VaR and ES baseline for comparison
    static_var = float(np.percentile(losses, (1.0 - p) * 100))
    static_es = float(np.mean(losses[losses > static_var])) if np.any(losses > static_var) else static_var * 1.2
    static_es_series = np.full(N, static_es)
    static_var_series = np.full(N, static_var)

    # 3. Identify violations
    hawkes_violations = (losses > var_series).astype(int)
    static_violations = (losses > static_var).astype(int)

    n_hawkes = int(np.sum(hawkes_violations))
    n_static = int(np.sum(static_violations))

    # 4. Run Kupiec POF tests
    lr_kup_h, p_kup_h, pass_kup_h = kupiec_pof_test(n_hawkes, N, p=p)
    lr_kup_s, p_kup_s, pass_kup_s = kupiec_pof_test(n_static, N, p=p)

    # 5. Run Christoffersen Independence tests
    lr_ind_h, p_ind_h, pass_ind_h = christoffersen_independence_test(hawkes_violations)
    lr_ind_s, p_ind_s, pass_ind_s = christoffersen_independence_test(static_violations)

    # Combined conditional coverage LR_CC = LR_POF + LR_ind ~ chi2(2)
    lr_cc_h = lr_kup_h + lr_ind_h
    p_cc_h = float(1.0 - chi2.cdf(lr_cc_h, df=2))

    # 6. Run Acerbi-Szekely Expected Shortfall (ES) Backtest
    z2_h, p_es_h, pass_es_h = acerbi_szekely_test(
        losses=losses,
        var_series=var_series,
        es_series=es_series,
        xi=xi_hat,
        a=a_hat,
        b=b_hat,
        intensities=lams,
        p=p,
        p_u=p_u
    )
    z2_s, p_es_s, pass_es_s = acerbi_szekely_test(
        losses=losses,
        var_series=static_var_series,
        es_series=static_es_series,
        xi=xi_hat,
        a=a_hat,
        b=b_hat,
        intensities=lams,
        p=p,
        p_u=p_u
    )

    return {
        "N": N,
        "target_p": p,
        "expected_violations": int(round(N * p)),
        "hawkes_violations": n_hawkes,
        "hawkes_violation_rate": float(n_hawkes / N),
        "static_violations": n_static,
        "static_violation_rate": float(n_static / N),
        "kupiec_hawkes": {"stat": lr_kup_h, "p_value": p_kup_h, "passed": pass_kup_h},
        "kupiec_static": {"stat": lr_kup_s, "p_value": p_kup_s, "passed": pass_kup_s},
        "christoffersen_hawkes": {"stat": lr_ind_h, "p_value": p_ind_h, "passed": pass_ind_h},
        "christoffersen_static": {"stat": lr_ind_s, "p_value": p_ind_s, "passed": pass_ind_s},
        "conditional_coverage_hawkes": {"stat": lr_cc_h, "p_value": p_cc_h, "passed": p_cc_h > 0.05},
        "acerbi_szekely_hawkes": {"stat": z2_h, "p_value": p_es_h, "passed": pass_es_h},
        "acerbi_szekely_static": {"stat": z2_s, "p_value": p_es_s, "passed": pass_es_s},
        "gpd_params": {"a": a_hat, "b": b_hat, "xi": xi_hat, "u": u},
        "var_series": var_series,
        "es_series": es_series,
    }

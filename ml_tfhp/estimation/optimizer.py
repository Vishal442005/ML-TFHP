"""
Constrained Multi-Start Maximum Likelihood Estimation for Hawkes Processes.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import minimize
from .loglikelihood import ml_tfhp_loglikelihood, compute_recursive_loglikelihood_numba
from ..kernels.spectral_density import branching_ratio


def fit_ml_tfhp(
    events: np.ndarray,
    T: float,
    M: int = 10,
    xi_min: float = 1e-4,
    xi_max: float = 1e2,
    n_starts: int = 5,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit the Mittag-Leffler Tempered Fractional Hawkes Process via multi-start L-BFGS-B.

    Parameters
    ----------
    events : ndarray
        1D array of sorted event timestamps.
    T : float
        Observation window end.
    M : int
        Number of quadrature components.
    n_starts : int
        Number of random restarts to avoid local minima.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict with estimated parameters, log-likelihood, AIC, BIC, and convergence flags.
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(events)
    if N == 0:
        raise ValueError("Cannot fit on empty event array.")

    # Objective: Negative Log-Likelihood
    def objective(theta):
        ll = ml_tfhp_loglikelihood(theta, events, T, M, xi_min, xi_max)
        return -ll

    # Parameter bounds: (mu, alpha, beta, gamma, eta)
    bounds = [
        (1e-4, 50.0),    # mu
        (0.05, 0.95),    # alpha
        (1e-3, 50.0),    # beta
        (0.0, 20.0),     # gamma
        (1e-4, 50.0),    # eta
    ]

    # Generate diverse initial parameter guesses
    empirical_rate = N / T
    initial_guesses = [
        # 1. Conservative baseline: high mu, moderate alpha
        np.array([empirical_rate * 0.7, 0.6, 1.0, 0.05, 0.2]),
        # 2. Reflexive regime: lower mu, strong excitation
        np.array([empirical_rate * 0.3, 0.65, 1.5, 0.02, 0.7]),
        # 3. Rough regime: alpha = 0.6 (Hurst ~ 0.1)
        np.array([empirical_rate * 0.5, 0.60, 2.0, 0.1, 0.5]),
        # 4. Long memory: alpha = 0.4
        np.array([empirical_rate * 0.5, 0.40, 1.0, 0.01, 0.3]),
    ]

    # Random restarts for remaining
    for _ in range(max(0, n_starts - len(initial_guesses))):
        init_mu = np.random.uniform(empirical_rate * 0.2, empirical_rate * 0.8)
        init_alpha = np.random.uniform(0.3, 0.8)
        init_beta = np.random.uniform(0.5, 3.0)
        init_gamma = np.random.uniform(0.01, 0.5)
        # Ensure initial branching ratio < 0.9
        max_eta = 0.8 * ((init_gamma ** init_alpha) + (init_beta ** init_alpha))
        init_eta = np.random.uniform(0.1 * max_eta, max_eta)
        initial_guesses.append(np.array([init_mu, init_alpha, init_beta, init_gamma, init_eta]))

    best_opt = None
    best_nll = np.inf

    for init_theta in initial_guesses:
        try:
            res = minimize(
                objective,
                init_theta,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-8, "gtol": 1e-6}
            )
            if res.fun < best_nll and not np.isnan(res.fun):
                best_nll = res.fun
                best_opt = res
        except Exception:
            continue

    if best_opt is None:
        raise RuntimeError("ML-TFHP optimization failed across all initial starts.")

    theta_hat = best_opt.x
    mu_hat, alpha_hat, beta_hat, gamma_hat, eta_hat = theta_hat
    n_star_hat = branching_ratio(alpha_hat, beta_hat, gamma_hat, eta_hat)

    log_lik = -best_nll
    k = 5  # Number of free parameters
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(N) - 2 * log_lik

    return {
        "params": theta_hat,
        "mu": mu_hat,
        "alpha": alpha_hat,
        "beta": beta_hat,
        "gamma": gamma_hat,
        "eta": eta_hat,
        "branching_ratio": n_star_hat,
        "log_likelihood": log_lik,
        "aic": aic,
        "bic": bic,
        "success": best_opt.success,
        "message": best_opt.message,
        "n_events": N,
        "T": T,
    }


def fit_exponential_hawkes(
    events: np.ndarray,
    T: float,
    n_starts: int = 3,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Fit baseline Exponential Hawkes Process (EHP):
        phi(t) = alpha_ehp * beta_ehp * exp(-beta_ehp * t)
        Branching ratio n* = alpha_ehp < 1.0.
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(events)
    empirical_rate = N / T

    def ehp_nll(params):
        mu, alpha_p, beta_p = params
        if mu <= 1e-6 or alpha_p <= 1e-6 or alpha_p >= 0.999 or beta_p <= 1e-4:
            return 1e15

        # Single exponential mode
        weights = np.array([alpha_p * beta_p], dtype=np.float64)
        zetas = np.array([beta_p], dtype=np.float64)
        ll, _ = compute_recursive_loglikelihood_numba(events, T, mu, weights, zetas)
        return -ll

    bounds = [(1e-4, 50.0), (1e-4, 0.99), (1e-3, 50.0)]
    initial_guesses = [
        np.array([empirical_rate * 0.6, 0.4, 1.0]),
        np.array([empirical_rate * 0.3, 0.7, 2.0]),
        np.array([empirical_rate * 0.8, 0.2, 0.5]),
    ]

    best_opt = None
    best_nll = np.inf

    for init_p in initial_guesses:
        try:
            res = minimize(
                ehp_nll,
                init_p,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 150}
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_opt = res
        except Exception:
            continue

    if best_opt is None:
        raise RuntimeError("EHP optimization failed.")

    mu_hat, alpha_hat, beta_hat = best_opt.x
    log_lik = -best_nll
    k = 3
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(N) - 2 * log_lik

    return {
        "params": best_opt.x,
        "mu": mu_hat,
        "alpha": alpha_hat,
        "beta": beta_hat,
        "branching_ratio": alpha_hat,
        "log_likelihood": log_lik,
        "aic": aic,
        "bic": bic,
        "success": best_opt.success,
        "n_events": N,
        "T": T,
    }

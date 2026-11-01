"""
Statistical Inference: Hessian, Observed Fisher Information, and Standard Errors.
"""

from typing import Tuple, Dict, Any, Callable
import numpy as np


def compute_hessian_central_diff(
    func: Callable[[np.ndarray], float],
    theta: np.ndarray,
    eps: float = 1e-4
) -> np.ndarray:
    """
    Compute the Hessian matrix of func at theta using symmetric finite differences.

    H_{ij} = [f(theta + h_i e_i + h_j e_j) - f(theta + h_i e_i - h_j e_j)
              - f(theta - h_i e_i + h_j e_j) + f(theta - h_i e_i - h_j e_j)] / (4 * h_i * h_j)
    """
    p = len(theta)
    H = np.zeros((p, p), dtype=np.float64)
    # Scale step sizes proportional to parameter magnitude
    h = eps * np.maximum(np.abs(theta), 1e-2)

    f0 = func(theta)

    for i in range(p):
        for j in range(i, p):
            if i == j:
                # Diagonal: standard 3-point central difference
                theta_plus = theta.copy()
                theta_plus[i] += h[i]
                theta_minus = theta.copy()
                theta_minus[i] -= h[i]
                H[i, i] = (func(theta_plus) - 2.0 * f0 + func(theta_minus)) / (h[i] ** 2)
            else:
                # Off-diagonal: 4-point central difference
                theta_pp = theta.copy()
                theta_pp[i] += h[i]
                theta_pp[j] += h[j]

                theta_pm = theta.copy()
                theta_pm[i] += h[i]
                theta_pm[j] -= h[j]

                theta_mp = theta.copy()
                theta_mp[i] -= h[i]
                theta_mp[j] += h[j]

                theta_mm = theta.copy()
                theta_mm[i] -= h[i]
                theta_mm[j] -= h[j]

                hij = (func(theta_pp) - func(theta_pm) - func(theta_mp) + func(theta_mm)) / (4.0 * h[i] * h[j])
                H[i, j] = hij
                H[j, i] = hij

    return H


def compute_hessian_and_se(
    loglik_func: Callable[[np.ndarray], float],
    theta_hat: np.ndarray,
    param_names: list = None
) -> Dict[str, Any]:
    """
    Compute observed Fisher information, parameter covariance, and 95% Wald CIs.

    Parameters
    ----------
    loglik_func : callable
        Function returning log-likelihood given parameter vector theta.
    theta_hat : ndarray
        Maximum likelihood estimate.
    param_names : list of str, optional
        Names of the parameters.

    Returns
    -------
    dict with keys:
        'hessian': Hessian matrix of log-likelihood
        'fisher_info': Observed Fisher information (-Hessian)
        'cov': Covariance matrix (inverse Fisher info)
        'se': Standard errors
        'ci_lower': 95% lower bounds
        'ci_upper': 95% upper bounds
    """
    p = len(theta_hat)
    if param_names is None:
        param_names = [f"theta_{i}" for i in range(p)]

    # Compute Hessian of log-likelihood (expected to be negative definite at max)
    H = compute_hessian_central_diff(loglik_func, theta_hat)
    fisher_info = -H

    # Regularized inversion to handle near-singular Hessian
    try:
        # Check condition number
        cond = np.linalg.cond(fisher_info)
        if cond > 1e12 or np.isnan(cond):
            cov = np.linalg.pinv(fisher_info)
        else:
            cov = np.linalg.inv(fisher_info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher_info)

    # Standard errors
    var = np.diag(cov)
    # Clip negative variances due to numerical noise
    var_clipped = np.maximum(var, 1e-12)
    se = np.sqrt(var_clipped)

    ci_lower = theta_hat - 1.96 * se
    ci_upper = theta_hat + 1.96 * se

    return {
        "hessian": H,
        "fisher_info": fisher_info,
        "cov": cov,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "param_names": param_names,
    }

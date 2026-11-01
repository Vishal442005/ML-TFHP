"""
Log-Uniform Markovian Quadrature for Mittag-Leffler Tempered Fractional Kernels.

Transforms the non-Markovian fractional kernel phi(t) into a sum of M exponentials:
    phi(t) approx sum_{m=1}^M w_m * exp(-zeta_m * t)
where zeta_m = xi_m + gamma.
"""

from typing import Tuple, Union, Optional
import numpy as np
from .spectral_density import spectral_density_ml, branching_ratio


def compute_log_quadrature(
    alpha: float,
    beta: float,
    gamma_param: float,
    eta: float,
    M: int = 10,
    xi_min: float = 1e-4,
    xi_max: float = 1e2,
    normalize_branching_ratio: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute log-uniformly spaced nodes and weights for the Markovian
    exponential sum approximation of the ML-TFHP kernel.

    Parameters
    ----------
    alpha : float
        Fractional exponent in (0, 1).
    beta : float
        Decay rate parameter (> 0).
    gamma_param : float
        Exponential tempering rate (>= 0).
    eta : float
        Excitation amplitude (> 0).
    M : int
        Number of exponential modes (typically 8 to 12).
    xi_min : float
        Minimum relaxation frequency (default 1e-4, corresponding to horizon ~ 1e4 s).
    xi_max : float
        Maximum relaxation frequency (default 1e2, corresponding to inter-arrival ~ 1e-2 s).
    normalize_branching_ratio : bool
        If True, rescales weights so sum_m (w_m / zeta_m) matches the exact
        theoretical branching ratio eta / (gamma^alpha + beta^alpha).

    Returns
    -------
    weights : np.ndarray of shape (M,)
        Component excitation weights w_m.
    zetas : np.ndarray of shape (M,)
        Effective decay rates zeta_m = xi_m + gamma.
    """
    if M < 2:
        raise ValueError("M must be at least 2 for log-quadrature.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if gamma_param < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma_param}")
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")

    # Generate M log-uniform nodes
    log_xi = np.linspace(np.log(xi_min), np.log(xi_max), M)
    xis = np.exp(log_xi)
    delta_log_xi = (np.log(xi_max) - np.log(xi_min)) / (M - 1)

    # Compute raw spectral densities and midpoint weights
    rhos = spectral_density_ml(xis, alpha, beta)
    # Jacobin factor d(xi) = xi * d(ln xi)
    weights = eta * rhos * xis * delta_log_xi
    zetas = xis + gamma_param

    if normalize_branching_ratio:
        # Theoretical exact L1 norm
        exact_norm = branching_ratio(alpha, beta, gamma_param, eta)
        # Approximate L1 norm = sum_m (w_m / zeta_m)
        approx_norm = np.sum(weights / zetas)
        if approx_norm > 1e-12:
            scale = exact_norm / approx_norm
            weights = weights * scale

    return weights, zetas


def approximate_kernel(
    t: Union[float, np.ndarray],
    weights: np.ndarray,
    zetas: np.ndarray
) -> Union[float, np.ndarray]:
    """
    Evaluate the Markovian sum-of-exponentials approximation at time lag(s) t:
        phi_approx(t) = sum_{m=1}^M w_m * exp(-zeta_m * t)

    Parameters
    ----------
    t : float or ndarray
        Time lag(s) (t >= 0).
    weights : ndarray of shape (M,)
        Component weights w_m.
    zetas : ndarray of shape (M,)
        Effective decay rates zeta_m.

    Returns
    -------
    phi_approx : float or ndarray
        Approximated kernel value(s).
    """
    t_arr = np.asarray(t, dtype=np.float64)
    # Shape: (len(t), M)
    # Using outer product: exp(-zeta_m * t)
    if t_arr.ndim == 0:
        return float(np.sum(weights * np.exp(-zetas * t_arr)))
    
    # Vectorized across time lags
    # (N, 1) * (1, M) -> (N, M)
    decay_matrix = np.exp(-np.outer(t_arr, zetas))
    phi_vals = decay_matrix @ weights
    return phi_vals


def evaluate_approximation_error(
    alpha: float,
    beta: float,
    gamma_param: float,
    eta: float,
    M: int = 10,
    t_min: float = 1e-2,
    t_max: float = 1e3,
    n_points: int = 200
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare exact ML-TFHP kernel vs sum-of-exponentials approximation
    across log-spaced evaluation points.

    Returns
    -------
    max_rel_err : float
        Maximum relative error over the evaluation range.
    mean_rel_err : float
        Mean relative error.
    t_grid : ndarray
        Time evaluation grid.
    exact_vals : ndarray
        Exact kernel values.
    approx_vals : ndarray
        Approximated kernel values.
    """
    from .mittag_leffler import ml_tfhp_kernel_exact

    weights, zetas = compute_log_quadrature(
        alpha=alpha,
        beta=beta,
        gamma_param=gamma_param,
        eta=eta,
        M=M,
        xi_min=0.1 / t_max,
        xi_max=10.0 / t_min,
        normalize_branching_ratio=True
    )

    t_grid = np.geomspace(t_min, t_max, n_points)
    exact_vals = ml_tfhp_kernel_exact(t_grid, mu=1.0, alpha=alpha, beta=beta, gamma_param=gamma_param, eta=eta)
    approx_vals = approximate_kernel(t_grid, weights, zetas)

    rel_errors = np.abs(exact_vals - approx_vals) / np.maximum(exact_vals, 1e-15)
    max_rel_err = float(np.max(rel_errors))
    mean_rel_err = float(np.mean(rel_errors))

    return max_rel_err, mean_rel_err, t_grid, exact_vals, approx_vals

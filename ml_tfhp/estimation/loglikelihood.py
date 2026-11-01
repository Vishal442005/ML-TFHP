"""
Numba-Accelerated O(NM) Recursive Log-Likelihood and Compensator Computation.
"""

from typing import Tuple
import numpy as np
import numba
from ..kernels.quadrature import compute_log_quadrature
from ..kernels.spectral_density import branching_ratio


@numba.jit(nopython=True, fastmath=True)
def compute_recursive_loglikelihood_numba(
    events: np.ndarray,
    T: float,
    mu: float,
    weights: np.ndarray,
    zetas: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute log-likelihood and event intensities in O(N * M) using Numba JIT.

    Parameters
    ----------
    events : ndarray
        1D array of sorted event timestamps in [0, T].
    T : float
        Observation window end (T >= events[-1]).
    mu : float
        Baseline immigrant arrival rate.
    weights : ndarray
        Quadrature weights w_m of shape (M,).
    zetas : ndarray
        Effective decay rates zeta_m of shape (M,).

    Returns
    -------
    log_lik : float
        Log-likelihood value ln L(theta).
    intensities : ndarray
        Array of conditional intensities lambda(t_i) for each event.
    """
    N = len(events)
    M = len(weights)
    intensities = np.empty(N, dtype=np.float64)

    if N == 0:
        return -mu * T, intensities

    # 1. Recursive intensity evaluation: lambda(t_i)
    # A_m(1) = 0
    A = np.zeros(M, dtype=np.float64)
    intensities[0] = mu
    log_sum = np.log(mu)

    for i in range(1, N):
        delta = events[i] - events[i - 1]
        # Propagate latent memory state: A_m(i) = (A_m(i-1) + 1) * exp(-zeta_m * delta)
        for m in range(M):
            A[m] = (A[m] + 1.0) * np.exp(-zetas[m] * delta)

        # Intensity at event t_i
        lam = mu
        for m in range(M):
            lam += weights[m] * A[m]

        if lam <= 0.0 or np.isnan(lam):
            return -1e15, intensities

        intensities[i] = lam
        log_sum += np.log(lam)

    # 2. Compensator evaluation: int_0^T lambda(s) ds
    # int_0^T lambda(s) ds = mu * T + sum_m (w_m / zeta_m) * sum_i (1 - exp(-zeta_m * (T - t_i)))
    comp = mu * T
    for m in range(M):
        zeta_inv = weights[m] / zetas[m]
        sum_exp = 0.0
        for i in range(N):
            sum_exp += 1.0 - np.exp(-zetas[m] * (T - events[i]))
        comp += zeta_inv * sum_exp

    log_lik = log_sum - comp
    return log_lik, intensities


@numba.jit(nopython=True, fastmath=True)
def compute_compensator_residuals_numba(
    events: np.ndarray,
    mu: float,
    weights: np.ndarray,
    zetas: np.ndarray
) -> np.ndarray:
    """
    Compute inter-event compensator increments (Papangelou residuals) in O(N * M):
        tau_i = int_{t_{i-1}}^{t_i} lambda(s) ds

    Under the true data-generating process, tau_i ~ i.i.d. Exp(1).
    """
    N = len(events)
    M = len(weights)
    residuals = np.empty(N, dtype=np.float64)

    if N == 0:
        return residuals

    # First interval [0, t_1]
    residuals[0] = mu * events[0]

    A = np.zeros(M, dtype=np.float64)

    for i in range(1, N):
        delta = events[i] - events[i - 1]
        # Over (t_{i-1}, t_i), post-jump memory was A_m + 1.0
        tau = mu * delta
        for m in range(M):
            post_jump = A[m] + 1.0
            decay = np.exp(-zetas[m] * delta)
            tau += (weights[m] / zetas[m]) * post_jump * (1.0 - decay)
            # Update A for next step
            A[m] = post_jump * decay

        residuals[i] = tau

    return residuals


def ml_tfhp_loglikelihood(
    params: np.ndarray,
    events: np.ndarray,
    T: float,
    M: int = 10,
    xi_min: float = 1e-4,
    xi_max: float = 1e2
) -> float:
    """
    Evaluate log-likelihood for ML-TFHP parameter vector theta = (mu, alpha, beta, gamma, eta).

    Returns negative log-likelihood if used for minimization.
    """
    mu, alpha, beta, gamma_param, eta = params

    # Enforce parameter boundaries
    if mu <= 1e-8 or alpha <= 0.01 or alpha >= 0.99 or beta <= 1e-4 or gamma_param < 0.0 or eta <= 1e-8:
        return -1e15

    # Enforce stationarity: n* < 1.0
    n_star = branching_ratio(alpha, beta, gamma_param, eta)
    if n_star >= 0.999:
        # Quadratic penalty for violating stationarity
        return -1e15 - 1e6 * (n_star - 0.999) ** 2

    weights, zetas = compute_log_quadrature(
        alpha=alpha,
        beta=beta,
        gamma_param=gamma_param,
        eta=eta,
        M=M,
        xi_min=xi_min,
        xi_max=xi_max,
        normalize_branching_ratio=True
    )

    ll, _ = compute_recursive_loglikelihood_numba(events, T, mu, weights, zetas)
    return float(ll)

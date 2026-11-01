"""
Ogata Thinning Algorithm for Hawkes Processes with Exponential and Fractional Kernels.

Implements fast O(N * M) exact simulation via recursive state decay.
"""

from typing import Tuple, Optional
import numpy as np
from ..kernels.quadrature import compute_log_quadrature
from ..kernels.spectral_density import branching_ratio


def simulate_ml_tfhp(
    mu: float,
    alpha: float,
    beta: float,
    gamma_param: float,
    eta: float,
    T: float,
    M: int = 10,
    xi_min: float = 1e-4,
    xi_max: float = 1e2,
    seed: Optional[int] = None,
    max_events: int = 1_000_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate event arrival timestamps for the ML-TFHP model on [0, T]
    using Ogata's thinning method with recursive Markovian states.

    Parameters
    ----------
    mu : float
        Baseline immigrant arrival rate (> 0).
    alpha : float
        Fractional memory parameter in (0, 1).
    beta : float
        Kernel scale/decay parameter (> 0).
    gamma_param : float
        Tempering rate (>= 0).
    eta : float
        Excitation amplitude (> 0).
    T : float
        Total simulation time horizon.
    M : int
        Number of exponential quadrature components (default 10).
    xi_min : float
        Minimum quadrature relaxation frequency.
    xi_max : float
        Maximum quadrature relaxation frequency.
    seed : int, optional
        Random seed for reproducibility.
    max_events : int
        Safety cap on generated events.

    Returns
    -------
    events : np.ndarray
        Sorted 1D array of event arrival timestamps t_i in [0, T].
    intensities : np.ndarray
        Conditional intensity values evaluated immediately before each event: lambda(t_i^-).
    states : np.ndarray of shape (len(events), M)
        Latent memory states at each event.
    """
    if seed is not None:
        np.random.seed(seed)

    n_star = branching_ratio(alpha, beta, gamma_param, eta)
    if n_star >= 1.0:
        raise ValueError(
            f"Non-stationary parameter combination: branching ratio n* = {n_star:.4f} >= 1.0. "
            "Process will explode."
        )

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

    t = 0.0
    A = np.zeros(M, dtype=np.float64)
    lambda_bar = mu
    
    event_list = []
    intensity_list = []
    state_list = []

    while t < T and len(event_list) < max_events:
        # 1. Sample inter-arrival candidate from Poisson process with rate lambda_bar
        if lambda_bar <= 0:
            lambda_bar = mu
        delta_t = np.random.exponential(1.0 / lambda_bar)
        t_cand = t + delta_t

        if t_cand >= T:
            break

        # 2. Propagate latent memory state to candidate time
        A_cand = A * np.exp(-zetas * (t_cand - t))
        lambda_cand = mu + np.sum(weights * A_cand)

        # 3. Acceptance-rejection check
        u = np.random.uniform(0.0, 1.0)
        if u <= (lambda_cand / lambda_bar):
            # Accept event
            t = t_cand
            event_list.append(t)
            intensity_list.append(lambda_cand)
            state_list.append(A_cand.copy())
            
            # Post-jump state update: each latent memory increases by 1
            A = A_cand + 1.0
            # New upper bound immediately after event
            lambda_bar = lambda_cand + np.sum(weights)
        else:
            # Reject candidate: since kernel is monotonically decreasing between events,
            # lambda_cand is a valid, tighter upper bound for times > t_cand
            t = t_cand
            A = A_cand
            lambda_bar = lambda_cand

    return np.array(event_list), np.array(intensity_list), np.array(state_list)


def simulate_exponential_hawkes(
    mu: float,
    alpha_ehp: float,
    beta_ehp: float,
    T: float,
    seed: Optional[int] = None,
    max_events: int = 1_000_000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard Exponential Hawkes baseline simulation for validation and comparison.
    Kernel: phi(t) = alpha_ehp * beta_ehp * exp(-beta_ehp * t)
    Branching ratio: n* = alpha_ehp < 1.0.
    """
    if seed is not None:
        np.random.seed(seed)

    if alpha_ehp >= 1.0:
        raise ValueError(f"Non-stationary EHP: alpha = {alpha_ehp} >= 1.0")

    t = 0.0
    R = 0.0
    lambda_bar = mu
    events = []
    intensities = []

    while t < T and len(events) < max_events:
        delta_t = np.random.exponential(1.0 / lambda_bar)
        t_cand = t + delta_t
        if t_cand >= T:
            break

        R_cand = R * np.exp(-beta_ehp * (t_cand - t))
        lambda_cand = mu + alpha_ehp * beta_ehp * R_cand

        u = np.random.uniform(0.0, 1.0)
        if u <= (lambda_cand / lambda_bar):
            t = t_cand
            events.append(t)
            intensities.append(lambda_cand)
            R = R_cand + 1.0
            lambda_bar = lambda_cand + alpha_ehp * beta_ehp
        else:
            t = t_cand
            R = R_cand
            lambda_bar = lambda_cand

    return np.array(events), np.array(intensities)

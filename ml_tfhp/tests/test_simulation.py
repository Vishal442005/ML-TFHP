"""
Unit tests for Ogata thinning simulation.
"""

import numpy as np
import pytest
from ml_tfhp.simulation.ogata import simulate_ml_tfhp, simulate_exponential_hawkes
from ml_tfhp.kernels.spectral_density import branching_ratio


def test_simulation_generates_ordered_events():
    """Verify simulation generates strictly increasing timestamps within [0, T]."""
    mu = 0.5
    alpha = 0.6
    beta = 1.0
    gamma_param = 0.1
    eta = 0.4
    T = 200.0

    events, intensities, states = simulate_ml_tfhp(
        mu=mu,
        alpha=alpha,
        beta=beta,
        gamma_param=gamma_param,
        eta=eta,
        T=T,
        seed=42
    )

    assert len(events) > 0
    assert np.all(events >= 0.0)
    assert np.all(events <= T)
    assert np.all(np.diff(events) > 0.0)  # Strictly ascending
    assert len(intensities) == len(events)
    assert len(states) == len(events)
    assert np.all(intensities >= mu)


def test_simulation_empirical_mean_intensity():
    """
    Verify empirical event rate N(T)/T is close to theoretical stationary mean:
        E[lambda] = mu / (1 - n*)
    """
    mu = 1.0
    alpha = 0.7
    beta = 2.0
    gamma_param = 0.5
    eta = 0.8
    T = 2000.0  # Long horizon for law of large numbers

    n_star = branching_ratio(alpha, beta, gamma_param, eta)
    expected_lambda = mu / (1.0 - n_star)

    events, _, _ = simulate_ml_tfhp(
        mu=mu,
        alpha=alpha,
        beta=beta,
        gamma_param=gamma_param,
        eta=eta,
        T=T,
        seed=123
    )

    empirical_lambda = len(events) / T
    # Relative difference should be within 10% on long sample
    rel_diff = abs(empirical_lambda - expected_lambda) / expected_lambda
    assert rel_diff < 0.10


def test_exponential_hawkes_baseline_simulation():
    """Verify standard exponential Hawkes simulation produces valid sequence."""
    mu = 0.8
    alpha_ehp = 0.5
    beta_ehp = 2.0
    T = 300.0

    events, intensities = simulate_exponential_hawkes(
        mu=mu,
        alpha_ehp=alpha_ehp,
        beta_ehp=beta_ehp,
        T=T,
        seed=42
    )

    assert len(events) > 0
    assert np.all(np.diff(events) > 0.0)
    expected_lambda = mu / (1.0 - alpha_ehp)  # 0.8 / 0.5 = 1.6
    empirical_lambda = len(events) / T
    assert abs(empirical_lambda - expected_lambda) / expected_lambda < 0.15

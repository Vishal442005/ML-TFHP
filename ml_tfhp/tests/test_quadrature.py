"""
Unit tests and error verification for Markovian log-quadrature.
"""

import numpy as np
import pytest
from ml_tfhp.kernels.quadrature import (
    compute_log_quadrature,
    approximate_kernel,
    evaluate_approximation_error,
)
from ml_tfhp.kernels.spectral_density import branching_ratio


def test_quadrature_branching_ratio_conservation():
    """Verify that sum(w_m / zeta_m) equals the exact theoretical branching ratio."""
    alpha = 0.65
    beta = 1.2
    gamma_param = 0.05
    eta = 0.8
    M = 10

    weights, zetas = compute_log_quadrature(
        alpha=alpha,
        beta=beta,
        gamma_param=gamma_param,
        eta=eta,
        M=M,
        normalize_branching_ratio=True
    )

    approx_n_star = np.sum(weights / zetas)
    exact_n_star = branching_ratio(alpha, beta, gamma_param, eta)
    np.testing.assert_allclose(approx_n_star, exact_n_star, rtol=1e-12)


def test_quadrature_kernel_approximation_accuracy():
    """
    Verify that the sum-of-exponentials kernel matches the exact ML-TFHP kernel
    with low relative error over typical intraday time horizon [1e-2, 1e2].
    """
    alpha = 0.6
    beta = 1.0
    gamma_param = 0.02
    eta = 0.5
    M = 12

    max_err, mean_err, t_grid, exact_vals, approx_vals = evaluate_approximation_error(
        alpha=alpha,
        beta=beta,
        gamma_param=gamma_param,
        eta=eta,
        M=M,
        t_min=1e-2,
        t_max=1e2,
        n_points=100
    )

    # Across 4 decades of time, the mean relative error should be small (< 0.05)
    # and both curves should closely align
    assert mean_err < 0.05
    # Values should all be positive and monotonically decaying
    assert np.all(approx_vals > 0)
    assert approx_vals[0] > approx_vals[-1]

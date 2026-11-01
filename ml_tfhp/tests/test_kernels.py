"""
Unit tests for Mittag-Leffler functions and exact kernel evaluations.
"""

import numpy as np
import pytest
from ml_tfhp.kernels.mittag_leffler import mittag_leffler, ml_tfhp_kernel_exact
from ml_tfhp.kernels.spectral_density import spectral_density_ml, branching_ratio, is_stationary


def test_mittag_leffler_exponential_case():
    """When alpha=1, beta=1, E_{1,1}(z) must equal exp(z)."""
    z_vals = np.array([0.0, -0.5, -1.0, -2.5, -5.0, -10.0])
    ml_vals = mittag_leffler(1.0, 1.0, z_vals)
    expected = np.exp(z_vals)
    np.testing.assert_allclose(ml_vals, expected, rtol=1e-10, atol=1e-12)


def test_mittag_leffler_e12_case():
    """When alpha=1, beta=2, E_{1,2}(z) must equal (exp(z) - 1) / z."""
    z_vals = np.array([-0.1, -0.5, -1.0, -2.0, -5.0])
    ml_vals = mittag_leffler(1.0, 2.0, z_vals)
    expected = (np.exp(z_vals) - 1.0) / z_vals
    np.testing.assert_allclose(ml_vals, expected, rtol=1e-10, atol=1e-12)


def test_mittag_leffler_continuity_near_boundary():
    """Test smooth continuity between series and asymptotic expansion around x=15."""
    alpha = 0.6
    beta = 0.6
    # Right below and above 15
    z1 = -14.9
    z2 = -15.1
    v1 = mittag_leffler(alpha, beta, z1)
    v2 = mittag_leffler(alpha, beta, z2)
    assert abs(v1 - v2) < 0.01
    assert v1 > 0
    assert v2 > 0


def test_spectral_density_positivity():
    """Spectral density rho_alpha(xi) must be strictly positive for all xi > 0."""
    xi = np.geomspace(1e-5, 1e3, 50)
    alpha = 0.7
    beta = 1.5
    rho = spectral_density_ml(xi, alpha, beta)
    assert np.all(rho > 0.0)


def test_branching_ratio_formula():
    """Test closed-form branching ratio calculation."""
    alpha = 0.5
    beta = 4.0   # 4^0.5 = 2.0
    gamma = 9.0  # 9^0.5 = 3.0
    eta = 2.5
    # n* = 2.5 / (3.0 + 2.0) = 0.5
    n_star = branching_ratio(alpha, beta, gamma, eta)
    assert abs(n_star - 0.5) < 1e-12
    assert is_stationary(alpha, beta, gamma, eta)


def test_non_stationary_detection():
    """Detect when eta >= gamma^alpha + beta^alpha."""
    alpha = 0.5
    beta = 1.0
    gamma = 0.0
    eta = 1.2  # 1.2 / 1.0^0.5 = 1.2 > 1.0
    assert not is_stationary(alpha, beta, gamma, eta)

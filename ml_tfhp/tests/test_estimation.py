"""
Unit tests for estimation, baseline models, and statistical inference.
"""

import numpy as np
import pytest
from ml_tfhp.simulation.ogata import simulate_exponential_hawkes, simulate_ml_tfhp
from ml_tfhp.models.exponential_hawkes import ExponentialHawkesModel
from ml_tfhp.models.powerlaw_hawkes import PowerLawHawkesModel
from ml_tfhp.models.ml_tfhp_model import MLTFHPModel
from ml_tfhp.evaluation.diagnostics import ks_test_residuals, ljung_box_residuals
from ml_tfhp.evaluation.model_selection import compare_models, likelihood_ratio_test


def test_exponential_hawkes_parameter_recovery():
    """Verify Exponential Hawkes fits and recovers parameters on synthetic data."""
    true_mu = 1.0
    true_alpha = 0.5
    true_beta = 2.0
    T = 600.0

    events, _ = simulate_exponential_hawkes(true_mu, true_alpha, true_beta, T=T, seed=123)
    assert len(events) > 100

    model = ExponentialHawkesModel()
    model.fit(events, T=T, seed=42)

    assert model.is_fitted
    mu_hat, alpha_hat, beta_hat = model.params

    # Parameter recovery within reasonable tolerance on 600 time units
    assert abs(mu_hat - true_mu) / true_mu < 0.25
    assert abs(alpha_hat - true_alpha) / true_alpha < 0.25

    # Check residuals satisfy Papangelou theorem
    residuals = model.compute_residuals()
    assert len(residuals) == len(events)
    assert np.all(residuals >= 0.0)
    ks_stat, p_val = ks_test_residuals(residuals)
    assert p_val > 0.01  # Should not be rejected


def test_powerlaw_hawkes_fitting():
    """Verify Power-Law Hawkes model fits and evaluates intensity."""
    events, _ = simulate_exponential_hawkes(1.0, 0.4, 2.0, T=300.0, seed=42)
    model = PowerLawHawkesModel()
    model.fit(events, T=300.0, seed=42)

    assert model.is_fitted
    assert model.branching_ratio_ < 1.0
    assert model.aic_ is not None

    residuals = model.compute_residuals()
    assert len(residuals) == len(events)


def test_ml_tfhp_model_fitting_and_inference():
    """Verify ML-TFHP model fits, computes Fisher information and standard errors."""
    true_mu = 0.8
    true_alpha = 0.65
    true_beta = 1.5
    true_gamma = 0.1
    true_eta = 0.6
    T = 400.0

    events, _, _ = simulate_ml_tfhp(
        mu=true_mu,
        alpha=true_alpha,
        beta=true_beta,
        gamma_param=true_gamma,
        eta=true_eta,
        T=T,
        seed=123
    )

    model = MLTFHPModel(M=8)
    model.fit(events, T=T, n_starts=3, compute_inference=True, seed=42)

    assert model.is_fitted
    assert model.branching_ratio_ < 1.0
    assert model.log_likelihood_ is not None

    # Check inference
    ci = model.get_confidence_intervals()
    assert "mu" in ci
    assert "alpha" in ci
    assert ci["alpha"]["std_error"] > 0
    assert ci["alpha"]["ci_95_lower"] < ci["alpha"]["ci_95_upper"]

    # Check intensity prediction
    t_eval = np.linspace(10.0, 50.0, 20)
    lam = model.predict_intensity(t_eval)
    assert len(lam) == len(t_eval)
    assert np.all(lam >= model.params[0])


def test_model_comparison_framework():
    """Verify nested comparison and ranking across EHP, PLHP, and ML-TFHP."""
    events, _, _ = simulate_ml_tfhp(
        mu=0.7,
        alpha=0.6,
        beta=1.2,
        gamma_param=0.05,
        eta=0.5,
        T=300.0,
        seed=77
    )

    ehp = ExponentialHawkesModel().fit(events, T=300.0, seed=42)
    ml_tfhp = MLTFHPModel(M=8).fit(events, T=300.0, n_starts=2, compute_inference=False, seed=42)

    comp_table = compare_models([ehp, ml_tfhp])
    assert len(comp_table) == 2
    assert "delta_aic" in comp_table[0]
    assert comp_table[0]["delta_aic"] == 0.0  # Best model has delta_aic == 0

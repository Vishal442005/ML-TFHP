"""
Unit tests for Bivariate ML-TFHP Model (Simulation, Estimation, Residuals, Stationarity).
"""

import numpy as np
import pytest
from ml_tfhp.models.bivariate_ml_tfhp import BivariateMLTFHPModel


def test_bivariate_simulation_and_stationarity():
    """Verify bivariate simulation generates valid timestamps and checks spectral radius."""
    model = BivariateMLTFHPModel(M=8, symmetric=True)
    ts, types, Gamma = model.simulate(
        mu0=0.4, mu1=0.4,
        alpha=0.6, beta=1.2, gamma=0.05,
        eta00=0.25, eta01=0.10, eta10=0.10, eta11=0.25,
        T=150.0, seed=42
    )

    assert len(ts) > 20
    assert len(types) == len(ts)
    assert np.all(np.diff(ts) >= 0)  # Monotonic
    assert set(types).issubset({0, 1})

    # Spectral radius must be strictly < 1
    eigvals = np.linalg.eigvals(Gamma)
    spec_rad = np.max(np.abs(eigvals))
    assert spec_rad < 1.0


def test_bivariate_supercritical_rejection():
    """Verify that supercritical parameters (spectral radius >= 1) raise ValueError."""
    model = BivariateMLTFHPModel(M=8, symmetric=True)
    with pytest.raises(ValueError, match="supercritical"):
        model.simulate(
            mu0=0.5, mu1=0.5,
            alpha=0.6, beta=0.8, gamma=0.01,
            eta00=1.5, eta01=0.8, eta10=0.8, eta11=1.5,  # Too high excitation!
            T=100.0, seed=42
        )


def test_bivariate_fitting_symmetric():
    """Verify symmetric Bivariate ML-TFHP fitting on simulated order flow."""
    model_sim = BivariateMLTFHPModel(M=8, symmetric=True)
    ts, types, _ = model_sim.simulate(
        mu0=0.5, mu1=0.5,
        alpha=0.55, beta=1.0, gamma=0.05,
        eta00=0.2, eta01=0.08, eta10=0.08, eta11=0.2,
        T=200.0, seed=42
    )

    model_fit = BivariateMLTFHPModel(M=8, symmetric=True)
    model_fit.fit(ts, types, T=200.0, n_starts=2, seed=42)

    assert model_fit.is_fitted
    assert model_fit.spectral_radius_ < 1.0
    assert model_fit.log_likelihood_ is not None
    assert model_fit.aic_ is not None
    assert model_fit.get_cross_excitation_ratio() > 0.0

    # Check intensities and residuals
    intensities, res0, res1 = model_fit.predict_intensities_and_residuals(ts, types, T=200.0)
    assert len(intensities) == len(ts)
    assert np.all(intensities > 0.0)
    assert len(res0) == np.sum(types == 0)
    assert len(res1) == np.sum(types == 1)

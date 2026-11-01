"""
Unit tests for Hawkes-VaR Risk Backtesting, Out-of-Sample Forecasting, and LaTeX table generators.
"""

import numpy as np
import pytest
from ml_tfhp.simulation.ogata import simulate_ml_tfhp
from ml_tfhp.models.ml_tfhp_model import MLTFHPModel
from ml_tfhp.models.exponential_hawkes import ExponentialHawkesModel
from ml_tfhp.evaluation.var_backtest import (
    fit_intensity_gpd,
    compute_hawkes_var,
    compute_hawkes_expected_shortfall,
    acerbi_szekely_test,
    kupiec_pof_test,
    christoffersen_independence_test,
    backtest_hawkes_var,
)
from ml_tfhp.evaluation.forecasting import (
    evaluate_out_of_sample_forecasting,
    evaluate_multi_horizon_forecasting,
)
from ml_tfhp.visualization.latex_tables import (
    generate_model_comparison_latex_table,
    generate_parameter_estimates_latex_table,
    generate_var_backtest_latex_table,
)


def test_kupiec_pof_exact_values():
    """Verify Kupiec POF formula returns expected statistic when violations match target."""
    # 10 violations out of 1000 with p=0.01 is perfect calibration
    lr_stat, p_val, passed = kupiec_pof_test(10, 1000, p=0.01)
    assert abs(lr_stat) < 1e-6
    assert p_val > 0.99
    assert passed

    # 40 violations out of 1000 is 4x excessive -> must reject null
    lr_stat_rej, p_val_rej, passed_rej = kupiec_pof_test(40, 1000, p=0.01)
    assert lr_stat_rej > 20.0
    assert p_val_rej < 1e-4
    assert not passed_rej


def test_christoffersen_independence_test():
    """Verify Christoffersen independence test detects clustered violations."""
    # Independent sequence: 01000000000100000000
    indep_seq = np.zeros(200, dtype=int)
    indep_seq[np.linspace(10, 190, 8, dtype=int)] = 1
    lr_stat, p_val, passed = christoffersen_independence_test(indep_seq)
    assert passed
    assert p_val > 0.05

    # Clustered sequence: consecutive violations 1,1,1,1
    clustered_seq = np.zeros(200, dtype=int)
    clustered_seq[50:54] = 1
    clustered_seq[120:124] = 1
    lr_stat_c, p_val_c, passed_c = christoffersen_independence_test(clustered_seq)
    assert not passed_c  # Detects violation clustering!


def test_hawkes_var_backtest_pipeline():
    """Verify end-to-end Hawkes-VaR backtest on synthetic returns."""
    np.random.seed(42)
    N = 1000
    # Synthetic intensities
    lams = np.random.uniform(0.5, 3.0, size=N)
    # Conditional returns with volatility scaling with lambda
    sigma = 0.005 + 0.003 * lams
    returns = np.random.normal(0, sigma)

    res = backtest_hawkes_var(returns, lams, p=0.01, quantile=0.90)

    assert res["N"] == N
    assert "hawkes_violations" in res
    assert "kupiec_hawkes" in res
    assert "christoffersen_hawkes" in res
    assert len(res["var_series"]) == N


def test_out_of_sample_forecasting():
    """Verify rolling out-of-sample event forecast evaluation."""
    events, _, _ = simulate_ml_tfhp(0.8, 0.62, 1.3, 0.05, 0.5, T=400.0, seed=42)
    model = MLTFHPModel(M=8).fit(events, T=250.0, n_starts=2, compute_inference=False, seed=42)

    forecast_results = evaluate_out_of_sample_forecasting(
        events,
        T_train=250.0,
        T_total=400.0,
        delta_t=10.0,
        models=[model]
    )

    assert "ML-TFHP" in forecast_results
    assert forecast_results["ML-TFHP"]["rmse"] >= 0.0
    assert forecast_results["ML-TFHP"]["mae"] >= 0.0


def test_latex_table_generators():
    """Verify LaTeX table generation produces valid LaTeX syntax."""
    events, _, _ = simulate_ml_tfhp(0.8, 0.62, 1.3, 0.05, 0.5, T=300.0, seed=42)
    m1 = ExponentialHawkesModel().fit(events, T=300.0, seed=42)
    m2 = MLTFHPModel(M=8).fit(events, T=300.0, n_starts=2, compute_inference=True, seed=42)

    # 1. Model comparison table
    latex_comp = generate_model_comparison_latex_table([m1, m2])
    assert "\\begin{table}" in latex_comp
    assert "\\end{table}" in latex_comp
    assert "ML-TFHP" in latex_comp

    # 2. Parameter table
    latex_params = generate_parameter_estimates_latex_table(m2)
    assert "\\begin{table}" in latex_params
    assert "\\alpha" in latex_params

    # 3. VaR table
    dummy_backtest = {
        "hawkes_violations": 12,
        "hawkes_violation_rate": 0.012,
        "static_violations": 18,
        "static_violation_rate": 0.018,
        "kupiec_hawkes": {"p_value": 0.55, "passed": True},
        "kupiec_static": {"p_value": 0.02, "passed": False},
        "christoffersen_hawkes": {"p_value": 0.40, "passed": True},
        "christoffersen_static": {"p_value": 0.01, "passed": False},
    }
    latex_var = generate_var_backtest_latex_table(dummy_backtest)
    assert "ML-TFHP Dynamic VaR" in latex_var
    assert "Static Empirical VaR" in latex_var


def test_expected_shortfall_and_acerbi_szekely():
    """Verify Expected Shortfall calculation and Acerbi-Szekely backtest."""
    np.random.seed(42)
    lams = np.array([0.5, 1.0, 2.0, 3.5])
    var_vals, es_vals = compute_hawkes_expected_shortfall(
        intensities=lams,
        a=0.01,
        b=0.005,
        xi=0.2,
        u=0.02,
        p=0.01,
        p_u=0.10
    )
    # ES must strictly exceed VaR everywhere for non-degenerate distributions
    assert np.all(es_vals > var_vals)

    # Test Acerbi-Szekely backtest under well-calibrated synthetic data
    N = 1000
    lams_full = np.random.uniform(0.5, 2.5, size=N)
    var_f, es_f = compute_hawkes_expected_shortfall(lams_full, 0.01, 0.005, 0.2, 0.02, p=0.01, p_u=0.10)
    # Synthetic losses with 1% exceedances
    losses = np.random.normal(0, 0.008, size=N)
    z2, p_val, passed = acerbi_szekely_test(
        losses=losses,
        var_series=var_f,
        es_series=es_f,
        xi=0.2,
        a=0.01,
        b=0.005,
        intensities=lams_full,
        p=0.01,
        n_bootstrap=200
    )
    assert isinstance(z2, float)
    assert 0.0 <= p_val <= 1.0
    assert isinstance(passed, (bool, np.bool_))


def test_multi_horizon_forecasting():
    """Verify multi-horizon forecasting sweeps over multiple delta_t windows."""
    events, _, _ = simulate_ml_tfhp(0.8, 0.62, 1.3, 0.05, 0.5, T=400.0, seed=42)
    m = MLTFHPModel(M=8).fit(events, T=250.0, n_starts=1, compute_inference=False, seed=42)
    horizons = [10.0, 30.0]
    res = evaluate_multi_horizon_forecasting(events, T_train=250.0, T_total=400.0, horizons=horizons, models=[m])
    assert 10.0 in res
    assert 30.0 in res
    assert "ML-TFHP" in res[10.0]
    assert res[10.0]["ML-TFHP"]["rmse"] >= 0.0


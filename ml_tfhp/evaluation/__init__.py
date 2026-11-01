"""
Evaluation, Goodness-of-Fit Diagnostics, Model Selection, VaR Backtesting, and Forecasting.
"""

from .diagnostics import (
    evaluate_goodness_of_fit,
    ks_test_residuals,
    ljung_box_residuals,
    compute_qq_plot_data,
)
from .model_selection import (
    compare_models,
    likelihood_ratio_test,
    vuong_test,
)
from .var_backtest import (
    fit_intensity_gpd,
    compute_hawkes_var,
    kupiec_pof_test,
    christoffersen_independence_test,
    backtest_hawkes_var,
)
from .forecasting import (
    predict_future_event_count,
    evaluate_out_of_sample_forecasting,
)

__all__ = [
    "evaluate_goodness_of_fit",
    "ks_test_residuals",
    "ljung_box_residuals",
    "compute_qq_plot_data",
    "compare_models",
    "likelihood_ratio_test",
    "vuong_test",
    "fit_intensity_gpd",
    "compute_hawkes_var",
    "kupiec_pof_test",
    "christoffersen_independence_test",
    "backtest_hawkes_var",
    "predict_future_event_count",
    "evaluate_out_of_sample_forecasting",
]

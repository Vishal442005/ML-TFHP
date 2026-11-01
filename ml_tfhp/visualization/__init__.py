"""
Visualization, plotting, and automated LaTeX table generation.
"""

from .plots import plot_intensity_path, plot_qq_residuals, plot_acf_residuals
from .latex_tables import (
    generate_model_comparison_latex_table,
    generate_parameter_estimates_latex_table,
    generate_var_backtest_latex_table,
)

__all__ = [
    "plot_intensity_path",
    "plot_qq_residuals",
    "plot_acf_residuals",
    "generate_model_comparison_latex_table",
    "generate_parameter_estimates_latex_table",
    "generate_var_backtest_latex_table",
]

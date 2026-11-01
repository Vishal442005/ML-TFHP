"""
Generate Publication-Ready Figures:
1. figures/intensity_comparison.png - lambda(t) paths during a clustering burst
2. figures/qq_plot_comparison.png - Q-Q plot comparing EHP, PLHP, ML-TFHP against Exp(1)
3. figures/residuals_acf.png - Autocorrelation function of ML-TFHP compensator residuals
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from ml_tfhp.simulation.ogata import simulate_ml_tfhp
from ml_tfhp.models.exponential_hawkes import ExponentialHawkesModel
from ml_tfhp.models.powerlaw_hawkes import PowerLawHawkesModel
from ml_tfhp.models.ml_tfhp_model import MLTFHPModel
from ml_tfhp.visualization.plots import (
    plot_intensity_path,
    plot_qq_residuals,
    plot_acf_residuals,
)


def generate_all_figures():
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    print("Generating simulation data and fitting models...")
    events, _, _ = simulate_ml_tfhp(
        mu=0.7,
        alpha=0.62,
        beta=1.3,
        gamma_param=0.03,
        eta=0.6,
        T=600.0,
        seed=42
    )

    print(f"Total events: {len(events)}")
    ehp = ExponentialHawkesModel().fit(events, T=600.0, seed=42)
    plhp = PowerLawHawkesModel().fit(events, T=600.0, seed=42)
    ml_tfhp = MLTFHPModel(M=10).fit(events, T=600.0, n_starts=3, seed=42)

    # 1. Plot intensity comparison over a high-clustering 30-second window
    t_start, t_end = 150.0, 185.0
    intensity_file = str(output_dir / "intensity_comparison.png")
    print(f"Plotting intensity comparison -> {intensity_file}")
    plot_intensity_path(events, [ehp, plhp, ml_tfhp], t_start, t_end, save_path=intensity_file)

    # 2. Plot QQ-plots
    qq_file = str(output_dir / "qq_plot_comparison.png")
    print(f"Plotting QQ residuals -> {qq_file}")
    plot_qq_residuals([ehp, plhp, ml_tfhp], save_path=qq_file)

    # 3. Plot ML-TFHP residual ACF
    acf_file = str(output_dir / "residuals_acf.png")
    print(f"Plotting ACF residuals -> {acf_file}")
    residuals = ml_tfhp.compute_residuals()
    plot_acf_residuals(residuals, max_lag=25, title="ML-TFHP Compensator Residuals ACF", save_path=acf_file)

    print("\nAll publication figures successfully generated in figures/ directory!")


if __name__ == "__main__":
    generate_all_figures()

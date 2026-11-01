"""
Publication-Quality Diagnostic and Benchmark Visualizations.
"""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from ..evaluation.diagnostics import compute_qq_plot_data


def plot_intensity_path(
    events: np.ndarray,
    models: list,
    t_start: float,
    t_end: float,
    n_points: int = 500,
    save_path: Optional[str] = None
):
    """
    Plot conditional intensity paths lambda(t) for competing models during a market burst.
    """
    t_grid = np.linspace(t_start, t_end, n_points)
    plt.figure(figsize=(12, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    linestyles = ["-", "--", "-.", ":"]

    for i, m in enumerate(models):
        lam = m.predict_intensity(t_grid, events=events)
        plt.plot(t_grid, lam, label=m.name, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=1.8)

    # Plot event ticks along bottom
    window_events = events[(events >= t_start) & (events <= t_end)]
    if len(window_events) > 0:
        plt.scatter(window_events, np.zeros_like(window_events), color="black", marker="|", s=100, label="Events (arrivals)")

    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Conditional Intensity $\\lambda(t)$", fontsize=12)
    plt.title(f"Instantaneous Event Intensity $\\lambda(t)$ Comparison [{t_start:.1f}s - {t_end:.1f}s]", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_qq_residuals(
    models: list,
    save_path: Optional[str] = None
):
    """
    Plot comparative QQ-plots of compensator residuals against Exp(1) 45-degree diagonal.
    """
    plt.figure(figsize=(7, 6))

    colors = ["#d62728", "#ff7f0e", "#1f77b4"]
    max_val = 0.0

    for i, m in enumerate(models):
        res = m.compute_residuals()
        theo, emp = compute_qq_plot_data(res, n_points=120)
        max_val = max(max_val, np.max(theo), np.max(emp))
        plt.plot(theo, emp, label=f"{m.name}", color=colors[i % len(colors)], linewidth=2)

    # 45-degree reference line
    plt.plot([0, max_val], [0, max_val], "k--", label="Theoretical Exp(1) [Ideal]", linewidth=1.5, alpha=0.7)

    plt.xlabel("Theoretical Quantiles $-\\ln(1 - p_k)$", fontsize=12)
    plt.ylabel("Empirical Residual Quantiles $\\tau_{(k)}$", fontsize=12)
    plt.title("Goodness-of-Fit: Residual Q-Q Plot", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_acf_residuals(
    residuals: np.ndarray,
    max_lag: int = 30,
    title: str = "Residual Autocorrelation Function (ACF)",
    save_path: Optional[str] = None
):
    """
    Plot autocorrelation function of compensator residuals with 95% confidence bounds.
    """
    n = len(residuals)
    mean_res = np.mean(residuals)
    var_res = np.var(residuals)

    lags = np.arange(1, max_lag + 1)
    acf_vals = np.empty(max_lag)

    for k in lags:
        acf_vals[k - 1] = np.sum((residuals[:-k] - mean_res) * (residuals[k:] - mean_res)) / (n * var_res)

    ci_bound = 1.96 / np.sqrt(n)

    plt.figure(figsize=(9, 4.5))
    plt.bar(lags, acf_vals, width=0.4, color="#1f77b4", alpha=0.8, label="Sample ACF")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axhline(ci_bound, color="red", linestyle="--", label="95% Confidence Band (White Noise)")
    plt.axhline(-ci_bound, color="red", linestyle="--")

    plt.xlabel("Lag (h)", fontsize=12)
    plt.ylabel("Autocorrelation $\\rho(h)$", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

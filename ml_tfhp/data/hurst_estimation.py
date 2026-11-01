"""
Hurst Exponent Estimation via Detrended Fluctuation Analysis (DFA) and Rescaled Range (R/S).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def estimate_hurst_dfa(
    series: np.ndarray,
    order: int = 1,
    scale_min: int = 10,
    scale_max: Optional[int] = None,
    n_scales: int = 20
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Estimate the Hurst exponent H of a time series using Detrended Fluctuation Analysis (DFA).

    Parameters
    ----------
    series : ndarray
        1D financial series (e.g. log-returns, realized volatility, or mid-price increments).
    order : int
        Polynomial detrending order (order=1 is standard linear DFA).
    scale_min : int
        Minimum segment length s (default 10).
    scale_max : int, optional
        Maximum segment length s (default N // 4).
    n_scales : int
        Number of log-spaced scale evaluations.

    Returns
    -------
    H : float
        Estimated Hurst exponent (slope of log F(s) vs log s).
    scales : ndarray
        Array of segment scales s.
    fluctuations : ndarray
        Fluctuation function values F(s).
    r_squared : float
        R^2 goodness-of-fit of the linear power-law regression.
    """
    x = np.asarray(series, dtype=np.float64)
    N = len(x)
    if N < 100:
        raise ValueError(f"Series too short for reliable DFA (N={N} < 100).")

    if scale_max is None:
        scale_max = max(scale_min + 10, N // 4)

    # 1. Integrate the centered profile: Y(k) = sum_{i=1}^k (x_i - mean(x))
    mean_x = np.mean(x)
    Y = np.cumsum(x - mean_x)

    # 2. Log-spaced scale selection
    scales = np.unique(np.geomspace(scale_min, scale_max, n_scales).astype(int))
    fluctuations = np.empty(len(scales), dtype=np.float64)

    for idx, s in enumerate(scales):
        n_segs = N // s
        if n_segs == 0:
            continue

        # Forward segments
        forward_vars = np.empty(n_segs, dtype=np.float64)
        t_seg = np.arange(s)
        for v in range(n_segs):
            seg_Y = Y[v * s : (v + 1) * s]
            poly = np.polyfit(t_seg, seg_Y, deg=order)
            trend = np.polyval(poly, t_seg)
            forward_vars[v] = np.mean((seg_Y - trend) ** 2)

        # Backward segments (to handle remainder at the end of the array)
        backward_vars = np.empty(n_segs, dtype=np.float64)
        for v in range(n_segs):
            seg_Y = Y[N - (v + 1) * s : N - v * s]
            poly = np.polyfit(t_seg, seg_Y, deg=order)
            trend = np.polyval(poly, t_seg)
            backward_vars[v] = np.mean((seg_Y - trend) ** 2)

        all_vars = np.concatenate([forward_vars, backward_vars])
        fluctuations[idx] = np.sqrt(np.mean(all_vars))

    # 3. Fit log F(s) = H * log(s) + C via ordinary least squares
    log_scales = np.log(scales)
    log_fluct = np.log(np.maximum(fluctuations, 1e-15))

    p = np.polyfit(log_scales, log_fluct, deg=1)
    H = float(p[0])

    # Goodness-of-fit R^2
    fitted = np.polyval(p, log_scales)
    ss_tot = np.sum((log_fluct - np.mean(log_fluct)) ** 2)
    ss_res = np.sum((log_fluct - fitted) ** 2)
    r_squared = float(1.0 - (ss_res / max(ss_tot, 1e-15)))

    return H, scales, fluctuations, r_squared


def estimate_hurst_rs(
    series: np.ndarray,
    scale_min: int = 10,
    scale_max: Optional[int] = None,
    n_scales: int = 20
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate Hurst exponent via classical Rescaled Range (R/S) analysis.
    """
    x = np.asarray(series, dtype=np.float64)
    N = len(x)
    if scale_max is None:
        scale_max = max(scale_min + 10, N // 2)

    scales = np.unique(np.geomspace(scale_min, scale_max, n_scales).astype(int))
    rs_vals = np.empty(len(scales), dtype=np.float64)

    for idx, s in enumerate(scales):
        n_segs = N // s
        if n_segs == 0:
            continue
        rs_segs = np.empty(n_segs, dtype=np.float64)
        for v in range(n_segs):
            seg = x[v * s : (v + 1) * s]
            m = np.mean(seg)
            std = np.std(seg, ddof=1)
            if std < 1e-12:
                std = 1e-12
            cum_dev = np.cumsum(seg - m)
            R = np.max(cum_dev) - np.min(cum_dev)
            rs_segs[v] = R / std

        rs_vals[idx] = np.mean(rs_segs)

    log_scales = np.log(scales)
    log_rs = np.log(np.maximum(rs_vals, 1e-15))
    p = np.polyfit(log_scales, log_rs, deg=1)
    H = float(p[0])

    return H, scales, rs_vals


def hurst_to_fractional_alpha(H: float, regime: str = "rough") -> float:
    """
    Convert estimated Hurst exponent H into the fractional Hawkes parameter alpha.

    Parameters
    ----------
    H : float
        Hurst exponent in (0, 1).
    regime : str
        'rough' : H_vol = alpha - 0.5 => alpha = H + 0.5 in (0.5, 1.0)
        'subcritical' : H_count = alpha + 0.5 => alpha = H - 0.5 in (0.0, 0.5)

    Returns
    -------
    alpha : float
        Implied fractional exponent in (0.01, 0.99).
    """
    if regime == "rough":
        # Rough volatility regime (Jaisson-Rosenbaum)
        alpha = H + 0.5
    elif regime == "subcritical":
        # Subcritical counting process long memory
        alpha = H - 0.5
    else:
        raise ValueError(f"Unknown regime '{regime}', choose 'rough' or 'subcritical'.")

    return float(np.clip(alpha, 0.05, 0.95))

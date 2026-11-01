"""
Event Extraction Pipeline: Peaks-Over-Threshold (POT) and Mid-Price Change Jumps.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


def extract_peaks_over_threshold(
    timestamps: np.ndarray,
    prices: np.ndarray,
    dt_sec: float = 5.0,
    quantile: float = 0.95,
    downside_only: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Extract timestamps of extreme return events using the Peaks-Over-Threshold (POT) method.

    Parameters
    ----------
    timestamps : ndarray
        Continuous timestamps in seconds.
    prices : ndarray
        Asset trade or mid-prices.
    dt_sec : float
        Binning window in seconds for return calculation.
    quantile : float
        Tail threshold percentile (default 0.95 for 95th percentile).
    downside_only : bool
        If True, only extract flash-crash / downward price drops (returns < -threshold).

    Returns
    -------
    event_times : ndarray
        Timestamps of extreme events t_i.
    jump_sizes : ndarray
        Magnitude of return jump at each event.
    threshold : float
        The empirical cutoff value u.
    binned_returns : ndarray
        Full series of binned log-returns.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    px = np.asarray(prices, dtype=np.float64)

    # Clean non-positive or NaN prices
    valid = (px > 0) & (~np.isnan(px)) & (~np.isnan(ts))
    ts = ts[valid]
    px = px[valid]

    # Create uniform time grid
    t_min = ts[0]
    t_max = ts[-1]
    grid_times = np.arange(t_min, t_max, dt_sec)

    # Forward-fill prices onto grid
    idx = np.searchsorted(ts, grid_times, side="right") - 1
    idx = np.clip(idx, 0, len(px) - 1)
    grid_prices = px[idx]

    # Compute log-returns
    log_returns = np.diff(np.log(grid_prices))
    event_grid_times = grid_times[1:]

    if downside_only:
        # Negative returns
        loss = -log_returns
        u = float(np.percentile(loss, quantile * 100))
        mask = loss >= u
        event_times = event_grid_times[mask]
        jump_sizes = loss[mask]
    else:
        # Absolute volatility shocks
        abs_ret = np.abs(log_returns)
        u = float(np.percentile(abs_ret, quantile * 100))
        mask = abs_ret >= u
        event_times = event_grid_times[mask]
        jump_sizes = abs_ret[mask]

    # Scale time so initial event is referenced from t=0
    return event_times, jump_sizes, u, log_returns


def extract_midprice_jumps(
    timestamps: np.ndarray,
    ask_prices: np.ndarray,
    bid_prices: np.ndarray,
    min_price_change: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract timestamps where the Limit Order Book mid-price changes.

    P_mid(t) = (P_ask(t) + P_bid(t)) / 2

    Returns
    -------
    jump_times : ndarray
        Timestamps where mid-price shifted.
    jump_sizes : ndarray
        Signed mid-price differences: P_mid(t) - P_mid(t - 1).
    mid_prices : ndarray
        Full mid-price series.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    asks = np.asarray(ask_prices, dtype=np.float64)
    bids = np.asarray(bid_prices, dtype=np.float64)

    mid_prices = 0.5 * (asks + bids)
    diffs = np.diff(mid_prices)

    # Find indices where price moved
    jump_indices = np.where(np.abs(diffs) >= min_price_change)[0] + 1

    jump_times = ts[jump_indices]
    jump_sizes = diffs[jump_indices - 1]

    return jump_times, jump_sizes, mid_prices

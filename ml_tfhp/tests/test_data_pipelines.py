"""
Unit tests for Data Engineering, DFA Hurst Exponent, and Event Extraction pipelines.
"""

import numpy as np
import pytest
from ml_tfhp.data.hurst_estimation import (
    estimate_hurst_dfa,
    estimate_hurst_rs,
    hurst_to_fractional_alpha,
)
from ml_tfhp.data.event_extractor import (
    extract_peaks_over_threshold,
    extract_midprice_jumps,
)
from ml_tfhp.data.lobster_pipeline import (
    generate_synthetic_lobster_data,
    parse_lobster_message_file,
    parse_lobster_orderbook_file,
)
from ml_tfhp.data.crypto_pipeline import (
    generate_synthetic_crypto_trades,
    parse_binance_trades_csv,
)


def test_dfa_hurst_white_noise():
    """DFA on white noise must yield H approx 0.50 (+/- 0.08)."""
    np.random.seed(42)
    noise = np.random.normal(0, 1, size=4000)
    H, scales, fluct, r2 = estimate_hurst_dfa(noise, scale_min=10, n_scales=15)

    assert 0.42 < H < 0.58
    assert r2 > 0.95  # Clean linear power-law scaling


def test_hurst_to_alpha_mapping():
    """Verify mapping between Hurst exponent and fractional Hawkes alpha."""
    # Rough volatility regime: H = 0.12 => alpha = 0.12 + 0.5 = 0.62
    alpha_rough = hurst_to_fractional_alpha(0.12, regime="rough")
    assert abs(alpha_rough - 0.62) < 1e-6

    # Subcritical counting regime: H = 0.85 => alpha = 0.85 - 0.5 = 0.35
    alpha_sub = hurst_to_fractional_alpha(0.85, regime="subcritical")
    assert abs(alpha_sub - 0.35) < 1e-6


def test_peaks_over_threshold_extraction():
    """Verify POT extraction isolates extreme returns above 95th percentile."""
    t = np.linspace(0, 1000, 2000)
    # Price path with sudden jumps
    prices = 100.0 + np.cumsum(np.random.normal(0, 0.1, size=2000))
    # Add 5 artificial flash crash spikes
    jump_indices = [300, 700, 1100, 1500, 1800]
    for idx in jump_indices:
        prices[idx:] -= 5.0

    event_times, jump_sizes, u, returns = extract_peaks_over_threshold(
        t, prices, dt_sec=5.0, quantile=0.95, downside_only=True
    )

    assert len(event_times) > 0
    assert len(event_times) == len(jump_sizes)
    assert np.all(jump_sizes >= u)
    assert u > 0.0


def test_lobster_pipeline_generation_and_jumps():
    """Verify synthetic LOBSTER generation and mid-price jump extraction."""
    msg_df, ob_df = generate_synthetic_lobster_data(
        n_events=1000,
        initial_price=150.0,
        T_seconds=600.0,
        seed=42
    )

    assert len(msg_df) > 100
    assert len(msg_df) == len(ob_df)
    assert "mid_price" in ob_df.columns
    assert "spread" in ob_df.columns

    # Extract mid-price jumps
    jump_times, jump_sizes, mid = extract_midprice_jumps(
        msg_df["time"].values,
        ob_df["ask_price_1"].values,
        ob_df["bid_price_1"].values
    )

    assert len(jump_times) > 0
    assert len(jump_times) == len(jump_sizes)
    assert np.all(np.abs(jump_sizes) > 0.0)


def test_crypto_pipeline_generation():
    """Verify Binance format crypto trade generation."""
    trades_df = generate_synthetic_crypto_trades(
        n_trades=1500,
        initial_price=60000.0,
        T_seconds=900.0,
        seed=42
    )

    assert len(trades_df) == 1500
    assert "is_buyer_maker" in trades_df.columns
    assert "price" in trades_df.columns
    assert "qty" in trades_df.columns
    assert np.all(trades_df["price"] > 0)

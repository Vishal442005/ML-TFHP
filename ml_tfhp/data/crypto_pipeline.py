"""
Cryptocurrency High-Frequency Trade Parser and Synthetic Generator (Binance Format).
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


def parse_binance_trades_csv(filepath: str) -> pd.DataFrame:
    """
    Parse standard Binance historical public trade CSV.

    Columns:
    - id: Trade ID
    - price: Execution price
    - qty: Base asset volume
    - quote_qty: Quote asset volume (price * qty)
    - time: Unix timestamp in milliseconds
    - is_buyer_maker: True if seller is taker (downward sell shock), False if buyer is taker
    - is_best_match: True if match was best
    """
    # Try reading with or without headers
    df = pd.read_csv(filepath)
    if "is_buyer_maker" not in df.columns:
        # Unlabelled public archive format
        cols = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]
        df = pd.read_csv(filepath, header=None, names=cols)

    # Convert milliseconds to seconds relative to start
    df["time_sec"] = (df["time"] - df["time"].iloc[0]) / 1000.0
    return df


def generate_synthetic_crypto_trades(
    n_trades: int = 5000,
    initial_price: float = 65000.0,
    T_seconds: float = 3600.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic cryptocurrency tick-level trades with rough volatility bursts
    and heavy-tailed metaorder sizes.
    """
    if seed is not None:
        np.random.seed(seed)

    from ..simulation.ogata import simulate_ml_tfhp

    # Generate bursty arrival times
    arrivals, _, _ = simulate_ml_tfhp(
        mu=0.8,
        alpha=0.60,
        beta=1.2,
        gamma_param=0.01,
        eta=0.65,
        T=T_seconds,
        seed=seed,
        max_events=n_trades
    )

    N = len(arrivals)
    timestamps_ms = 1700000000000 + (arrivals * 1000.0).astype(np.int64)

    # Direction: is_buyer_maker (True = Sell Market Order, False = Buy Market Order)
    is_buyer_maker = np.random.choice([True, False], size=N, p=[0.52, 0.48])

    # Power-law trade volumes (metaorder Pareto distribution)
    # Pareto with shape 1.8 (heavy-tailed)
    volumes = np.random.pareto(a=1.8, size=N) * 0.1 + 0.005

    # Geometric Brownian Motion with rough jump bursts
    current_price = initial_price
    prices = np.empty(N, dtype=np.float64)

    for i in range(N):
        direction_sign = -1.0 if is_buyer_maker[i] else 1.0
        # Price impact proportional to sqrt(volume) (Kyle's lambda / square root law)
        impact = direction_sign * 0.0002 * np.sqrt(volumes[i]) * current_price
        # Small diffusion step
        diff_noise = np.random.normal(0, 0.0001 * current_price)
        current_price = max(100.0, current_price + impact + diff_noise)
        prices[i] = current_price

    df = pd.DataFrame({
        "id": np.arange(100000, 100000 + N),
        "price": np.round(prices, 2),
        "qty": np.round(volumes, 4),
        "quote_qty": np.round(prices * volumes, 2),
        "time": timestamps_ms,
        "is_buyer_maker": is_buyer_maker,
        "is_best_match": True,
        "time_sec": arrivals,
    })

    return df

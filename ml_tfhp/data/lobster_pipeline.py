"""
LOBSTER Limit Order Book Parser and Synthetic Generator.

Handles Level 1 and Level 2 LOBSTER message and orderbook CSV files.
"""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd


def parse_lobster_message_file(filepath: str) -> pd.DataFrame:
    """
    Parse standard LOBSTER message file.

    Format columns:
    1. Time: Seconds after midnight (with decimal precision up to nanoseconds)
    2. Type: 1 (Submission), 2 (Partial cancel), 3 (Deletion), 4 (Visible exec),
             5 (Hidden exec), 6 (Cross trade), 7 (Trading halt)
    3. OrderID: Unique order ID
    4. Size: Number of shares
    5. Price: Dollar price times 10,000 (e.g. 1000000 = $100.00)
    6. Direction: -1 (Sell limit), 1 (Buy limit)
    """
    cols = ["time", "event_type", "order_id", "size", "price", "direction"]
    df = pd.read_csv(filepath, header=None, names=cols)
    df["price_dollar"] = df["price"] / 10000.0
    return df


def parse_lobster_orderbook_file(filepath: str, levels: int = 1) -> pd.DataFrame:
    """
    Parse standard LOBSTER orderbook file.

    Columns for Level 1:
        Ask Price 1, Ask Size 1, Bid Price 1, Bid Size 1
    Prices are multiplied by 10,000.
    """
    df = pd.read_csv(filepath, header=None)
    cols = []
    for lv in range(1, levels + 1):
        cols.extend([f"ask_price_{lv}", f"ask_size_{lv}", f"bid_price_{lv}", f"bid_size_{lv}"])

    df = df.iloc[:, : len(cols)]
    df.columns = cols

    # Rescale prices to dollars
    for lv in range(1, levels + 1):
        df[f"ask_price_{lv}"] = df[f"ask_price_{lv}"] / 10000.0
        df[f"bid_price_{lv}"] = df[f"bid_price_{lv}"] / 10000.0

    df["mid_price"] = 0.5 * (df["ask_price_1"] + df["bid_price_1"])
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    return df


def generate_synthetic_lobster_data(
    n_events: int = 5000,
    initial_price: float = 150.00,
    tick_size: float = 0.01,
    T_seconds: float = 3600.0,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic synthetic LOBSTER-formatted message and orderbook DataFrames
    with fractional volatility clustering and price impact.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate clustered event arrival timestamps
    from ..simulation.ogata import simulate_ml_tfhp

    arrival_times, _, _ = simulate_ml_tfhp(
        mu=0.5,
        alpha=0.62,
        beta=1.5,
        gamma_param=0.02,
        eta=0.55,
        T=T_seconds,
        seed=seed,
        max_events=n_events
    )

    N = len(arrival_times)
    # Start at 9:30 AM = 34200 seconds after midnight
    times = 34200.0 + arrival_times

    # Event types: 1 (Submission 40%), 3 (Deletion 35%), 4 (Market Execution 25%)
    types = np.random.choice([1, 3, 4], size=N, p=[0.40, 0.35, 0.25])
    directions = np.random.choice([1, -1], size=N)
    sizes = np.random.choice([100, 200, 500, 1000], size=N, p=[0.6, 0.25, 0.1, 0.05])

    # Dynamic mid-price evolution driven by market orders (type 4)
    price_cents = int(round(initial_price / tick_size))
    prices_list = []
    ask1_list = []
    bid1_list = []

    for t, evt_type, d in zip(times, types, directions):
        if evt_type == 4:
            # Market order execution moves price in trade direction
            price_cents += d * np.random.choice([0, 1, 2], p=[0.6, 0.35, 0.05])

        mid = price_cents * tick_size
        ask = mid + 0.5 * tick_size
        bid = mid - 0.5 * tick_size

        prices_list.append(int(round(mid * 10000)))
        ask1_list.append(ask)
        bid1_list.append(bid)

    message_df = pd.DataFrame({
        "time": times,
        "event_type": types,
        "order_id": np.arange(1000, 1000 + N),
        "size": sizes,
        "price": prices_list,
        "direction": directions,
        "price_dollar": np.array(prices_list) / 10000.0
    })

    orderbook_df = pd.DataFrame({
        "ask_price_1": ask1_list,
        "ask_size_1": np.random.randint(5, 50, size=N) * 100,
        "bid_price_1": bid1_list,
        "bid_size_1": np.random.randint(5, 50, size=N) * 100,
        "mid_price": 0.5 * (np.array(ask1_list) + np.array(bid1_list)),
        "spread": np.array(ask1_list) - np.array(bid1_list)
    })

    return message_df, orderbook_df

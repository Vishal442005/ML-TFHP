"""
Live Market Data Downloader and Empirical Benchmark Pipeline:
Fetches real tick-by-tick trades from Binance Public API (BTC/USDT),
extracts empirical market shocks, runs DFA Hurst estimation, fits ML-TFHP,
and performs regulatory Hawkes-VaR backtesting on real historical market data.
"""

import os
import sys
import json
import time
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ml_tfhp.data.event_extractor import extract_peaks_over_threshold
from ml_tfhp.data.hurst_estimation import estimate_hurst_dfa, hurst_to_fractional_alpha
from ml_tfhp.models.exponential_hawkes import ExponentialHawkesModel
from ml_tfhp.models.powerlaw_hawkes import PowerLawHawkesModel
from ml_tfhp.models.ml_tfhp_model import MLTFHPModel
from ml_tfhp.evaluation.model_selection import compare_models, likelihood_ratio_test
from ml_tfhp.evaluation.var_backtest import backtest_hawkes_var


def fetch_binance_agg_trades(symbol: str = "BTCUSDT", target_trades: int = 10000) -> pd.DataFrame:
    """
    Fetch consecutive historical aggregated trades from Binance public REST API.
    """
    print(f"Fetching {target_trades} real trades for {symbol} from Binance Public API...")
    base_url = "https://api.binance.com/api/v3/aggTrades"

    # 1. Fetch latest trade ID
    latest_req = urllib.request.Request(f"{base_url}?symbol={symbol}&limit=1", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(latest_req, timeout=10) as resp:
        latest_data = json.loads(resp.read().decode())
        latest_id = latest_data[-1]["a"]

    start_id = latest_id - target_trades
    all_trades = []
    current_id = start_id
    batch_size = 1000

    while len(all_trades) < target_trades:
        url = f"{base_url}?symbol={symbol}&limit={batch_size}&fromId={current_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                batch = json.loads(resp.read().decode())
                if not batch:
                    break
                all_trades.extend(batch)
                current_id = batch[-1]["a"] + 1
                time.sleep(0.05)
                print(f"  Fetched {len(all_trades)} / {target_trades} trades...")
        except Exception as e:
            print(f"  Warning during fetch: {e}")
            break

    # Format into DataFrame
    records = []
    for tr in all_trades[:target_trades]:
        records.append({
            "trade_id": tr["a"],
            "price": float(tr["p"]),
            "qty": float(tr["q"]),
            "time_ms": int(tr["T"]),
            "is_buyer_maker": bool(tr["m"]),
        })

    df = pd.DataFrame(records)
    t0 = df["time_ms"].iloc[0]
    df["time_sec"] = (df["time_ms"] - t0) / 1000.0
    return df


def run_real_market_empirical_benchmark():
    print("=" * 80)
    print("EMPIRICAL BENCHMARK ON REAL-WORLD BINANCE HIGH-FREQUENCY CRYPTO DATA")
    print("=" * 80)

    # 1. Fetch real trades
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_file = data_dir / "real_btcusdt_trades.csv"

    # If existing file has fewer than 5,000 trades, re-download
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        if len(df) < 5000:
            df = fetch_binance_agg_trades("BTCUSDT", target_trades=10000)
            df.to_csv(csv_file, index=False)
            print(f"Saved {len(df)} real trades to {csv_file}")
        else:
            print(f"Loading {len(df)} existing real trades from {csv_file}...")
    else:
        df = fetch_binance_agg_trades("BTCUSDT", target_trades=10000)
        df.to_csv(csv_file, index=False)
        print(f"Saved {len(df)} real trades to {csv_file}")

    N_trades = len(df)
    T_span = df["time_sec"].iloc[-1]
    print(f"\n[1] Real Market Data Summary:")
    print(f"    Total Trades Analyzed : {N_trades} ticks")
    print(f"    Observation Duration  : {T_span:.1f} seconds ({T_span/60.0:.1f} minutes)")
    print(f"    Price Range           : ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
    print(f"    Average Trade Size    : {df['qty'].mean():.4f} BTC")

    # 2. Estimate empirical Hurst exponent (DFA) on real price increments
    log_prices = np.log(df["price"].values)
    log_diffs = np.diff(log_prices)
    active_diffs = log_diffs[np.abs(log_diffs) > 1e-8]

    H_real, scales, fluct, r2 = estimate_hurst_dfa(active_diffs, scale_min=10, n_scales=15)
    implied_alpha = hurst_to_fractional_alpha(H_real, regime="rough")

    print(f"\n[2] Empirical Hurst Scaling on Real Returns:")
    print(f"    DFA Hurst Exponent H  : {H_real:.4f} (R^2 = {r2:.4f})")
    print(f"    Market Roughness      : {'Rough Volatility (H < 0.5)' if H_real < 0.5 else 'Subcritical Memory (H >= 0.5)'}")
    print(f"    Implied Hawkes Alpha  : {implied_alpha:.4f}")

    # 3. Extract real Peaks-Over-Threshold market shocks
    dt_sec = 5.0
    quantile = 0.90
    event_times, jump_sizes, u_thresh, returns = extract_peaks_over_threshold(
        df["time_sec"].values,
        df["price"].values,
        dt_sec=dt_sec,
        quantile=quantile,
        downside_only=False
    )

    print(f"\n[3] Peaks-Over-Threshold (POT) Real Market Shock Extraction:")
    print(f"    Time Bin Resolution   : {dt_sec:.1f} seconds")
    print(f"    Threshold Percentile  : {quantile*100:.0f}th percentile (|r| >= {u_thresh*100:.4f}%)")
    print(f"    Total Real Shocks     : {len(event_times)} shock arrivals")

    # Re-reference timestamps from t=0
    T_horizon = event_times[-1] - event_times[0] + 5.0
    shifted_times = event_times - event_times[0] + 0.5

    # 4. Fit competing models on real market shocks
    print(f"\n[4] Fitting Competing Models on {len(shifted_times)} Real Market Shock Arrivals:")
    print("    - Fitting Exponential Hawkes Process (EHP)...")
    ehp = ExponentialHawkesModel().fit(shifted_times, T=T_horizon, seed=42)

    print("    - Fitting Power-Law Hawkes Process (PLHP)...")
    plhp = PowerLawHawkesModel().fit(shifted_times, T=T_horizon, seed=42)

    print("    - Fitting Proposed ML-TFHP Model (with Fisher Hessian inference)...")
    ml_tfhp = MLTFHPModel(M=10).fit(shifted_times, T=T_horizon, n_starts=4, compute_inference=True, seed=42)

    # 5. Comparative benchmark table
    print("\n[5] Real-World Model Selection Benchmark Table:")
    print("-" * 88)
    print(f"{'Model':<18} | {'Log-Lik':<10} | {'AIC':<10} | {'d_AIC':<8} | {'BIC':<10} | {'n*':<6} | {'KS p-val':<8} | {'Status':<6}")
    print("-" * 88)

    models = [ehp, plhp, ml_tfhp]
    comp = compare_models(models)
    for row in comp:
        print(
            f"{row['model']:<18} | "
            f"{row['log_likelihood']:<10.2f} | "
            f"{row['aic']:<10.2f} | "
            f"{row['delta_aic']:<8.2f} | "
            f"{row['bic']:<10.2f} | "
            f"{row['branching_ratio']:<6.3f} | "
            f"{row['ks_pvalue']:<8.4f} | "
            f"{'PASS' if row['passes_ks'] else 'REJECT':<6}"
        )
    print("-" * 88)

    # Likelihood Ratio Test
    lr_stat, lr_p = likelihood_ratio_test(ehp.log_likelihood_, ml_tfhp.log_likelihood_, df=2)
    print(f"\n[6] Likelihood Ratio Test (EHP vs ML-TFHP on Real Data):")
    print(f"    LR Statistic = {lr_stat:.4f}, p-value = {lr_p:.4e}")
    if lr_p < 0.05:
        print("    -> ML-TFHP statistically outperforms Exponential Hawkes on REAL market data!")

    # 7. Real High-Frequency Hawkes-VaR Backtesting
    print("\n[7] High-Frequency Hawkes-VaR Backtesting on Real Return Series:")
    t_ret_grid = np.linspace(shifted_times[0], shifted_times[-1], len(returns))
    real_lams = ml_tfhp.predict_intensity(t_ret_grid, events=shifted_times)

    backtest = backtest_hawkes_var(returns, real_lams, p=0.01, quantile=0.85)
    print(f"    Total Test Return Windows : {backtest['N']}")
    print(f"    Hawkes-VaR Violations     : {backtest['hawkes_violations']} ({backtest['hawkes_violation_rate']*100:.2f}%) [Target: 1.0%]")
    print(f"    Static VaR Violations     : {backtest['static_violations']} ({backtest['static_violation_rate']*100:.2f}%)")
    print(f"    Kupiec POF Test           : LR = {backtest['kupiec_hawkes']['stat']:.4f}, p-val = {backtest['kupiec_hawkes']['p_value']:.4f} -> {'PASS' if backtest['kupiec_hawkes']['passed'] else 'FAIL'}")
    print(f"    Christoffersen Indep.     : LR = {backtest['christoffersen_hawkes']['stat']:.4f}, p-val = {backtest['christoffersen_hawkes']['p_value']:.4f} -> {'PASS (No Clustering)' if backtest['christoffersen_hawkes']['passed'] else 'FAIL'}")

    print("\n" + "=" * 80)
    print("REAL-WORLD MARKET DATA BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_real_market_empirical_benchmark()

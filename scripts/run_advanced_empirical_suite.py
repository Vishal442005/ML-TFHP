"""
Comprehensive Advanced Empirical Benchmarking Suite for Top-Tier Publication.
Executes:
1. Multi-horizon forecasting sweep (5s to 600s) comparing ML-TFHP vs EHP vs Poisson.
2. Basel FRTB Expected Shortfall (ES) & Acerbi-Szekely regulatory backtest.
3. Bivariate (Buy/Sell) ML-TFHP order flow fitting on 10,000 real Binance tick trades.
4. Generates multi_horizon_forecasting_crossover.png and complete LaTeX submission tables.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_tfhp.models.ml_tfhp_model import MLTFHPModel
from ml_tfhp.models.exponential_hawkes import ExponentialHawkesModel
from ml_tfhp.models.bivariate_ml_tfhp import BivariateMLTFHPModel
from ml_tfhp.data.event_extractor import extract_peaks_over_threshold
from ml_tfhp.evaluation.forecasting import evaluate_multi_horizon_forecasting
from ml_tfhp.evaluation.var_backtest import backtest_hawkes_var
from ml_tfhp.visualization.latex_tables import (
    generate_model_comparison_latex_table,
    generate_parameter_estimates_latex_table,
    generate_var_backtest_latex_table,
    generate_multi_horizon_forecasting_latex_table,
)


def main():
    print("=" * 70)
    print("RUNNING ADVANCED EMPIRICAL BENCHMARKING SUITE")
    print("=" * 70)

    # 1. Load Real Binance Data
    data_path = "data/real_btcusdt_trades.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}")

    df = pd.read_csv(data_path)
    timestamps = df["time_sec"].values
    T_total = float(timestamps[-1])
    N_total = len(timestamps)
    print(f"Loaded {N_total} real trades spanning {T_total:.2f} s ({T_total/60:.1f} min).")

    # Split train / test for out-of-sample forecasting (70% train, 30% test)
    T_train = T_total * 0.70
    train_events = timestamps[timestamps <= T_train]
    print(f"Training on {len(train_events)} events (T <= {T_train:.1f}s), testing on {N_total - len(train_events)} events.")

    # 2. Fit Univariate Models
    print("\n--- Fitting ML-TFHP Model ---")
    ml_model = MLTFHPModel(M=12).fit(train_events, T=T_train, n_starts=3, compute_inference=True, seed=42)
    print(f"ML-TFHP fit complete: LogLik={ml_model.log_likelihood_:.2f}, n*={ml_model.branching_ratio_:.3f}")

    print("\n--- Fitting Exponential Hawkes Model ---")
    ehp_model = ExponentialHawkesModel().fit(train_events, T=T_train, seed=42)
    print(f"EHP fit complete: LogLik={ehp_model.log_likelihood_:.2f}, n*={ehp_model.branching_ratio_:.3f}")

    # Dummy class for Poisson baseline
    class PoissonBaseline:
        def __init__(self, rate):
            self.name = "Poisson"
            self.rate = rate
            self.is_fitted = True
            self.params = np.array([rate])
            self.log_likelihood_ = float(len(train_events) * np.log(rate) - rate * T_train)
            self.aic_ = float(2.0 * 1 - 2.0 * self.log_likelihood_)
            self.bic_ = float(np.log(len(train_events)) - 2.0 * self.log_likelihood_)
            self.branching_ratio_ = 0.0
            self.n_params = 1
            self.inference_ = None

    poisson_model = PoissonBaseline(len(train_events) / T_train)

    # 3. Multi-Horizon Forecasting Evaluation
    print("\n--- Evaluating Multi-Horizon Forecasting Crossover ---")
    horizons = [5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0]
    multi_results = evaluate_multi_horizon_forecasting(
        events=timestamps,
        T_train=T_train,
        T_total=T_total,
        horizons=horizons,
        models=[poisson_model, ehp_model, ml_model]
    )

    print("\nHorizon Results:")
    print(f"{'Horizon':<10} | {'Poisson RMSE':<14} | {'EHP RMSE':<12} | {'ML-TFHP RMSE':<14} | {'Winner':<10}")
    print("-" * 70)
    for h in horizons:
        p_rmse = multi_results[h]["Poisson"]["rmse"]
        e_rmse = multi_results[h]["ExponentialHawkes"]["rmse"]
        m_rmse = multi_results[h]["ML-TFHP"]["rmse"]
        winner = "ML-TFHP" if m_rmse < e_rmse else "EHP"
        print(f"{h:<10.1f} | {p_rmse:<14.3f} | {e_rmse:<12.3f} | {m_rmse:<14.3f} | {winner:<10}")

    # Plot Multi-Horizon RMSE and MAE
    os.makedirs("figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    h_arr = np.array(horizons)
    p_rmse = [multi_results[h]["Poisson"]["rmse"] for h in horizons]
    e_rmse = [multi_results[h]["ExponentialHawkes"]["rmse"] for h in horizons]
    m_rmse = [multi_results[h]["ML-TFHP"]["rmse"] for h in horizons]

    p_mae = [multi_results[h]["Poisson"]["mae"] for h in horizons]
    e_mae = [multi_results[h]["ExponentialHawkes"]["mae"] for h in horizons]
    m_mae = [multi_results[h]["ML-TFHP"]["mae"] for h in horizons]

    ax1.plot(h_arr, p_rmse, 'r:s', label='Poisson Baseline', lw=1.5, markersize=5)
    ax1.plot(h_arr, e_rmse, 'b--o', label='Exponential Hawkes', lw=2.0, markersize=6)
    ax1.plot(h_arr, m_rmse, 'g-^', label='ML-TFHP (Ours)', lw=2.5, markersize=7)
    ax1.set_xlabel(r'Forecast Horizon $\Delta t$ (seconds)', fontsize=12)
    ax1.set_ylabel('Root Mean Square Error (RMSE)', fontsize=12)
    ax1.set_title('Out-of-Sample Arrival Count RMSE', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which='both', ls='--', alpha=0.5)
    ax1.legend(fontsize=10)

    ax2.plot(h_arr, p_mae, 'r:s', label='Poisson Baseline', lw=1.5, markersize=5)
    ax2.plot(h_arr, e_mae, 'b--o', label='Exponential Hawkes', lw=2.0, markersize=6)
    ax2.plot(h_arr, m_mae, 'g-^', label='ML-TFHP (Ours)', lw=2.5, markersize=7)
    ax2.set_xlabel(r'Forecast Horizon $\Delta t$ (seconds)', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax2.set_title('Out-of-Sample Arrival Count MAE', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, which='both', ls='--', alpha=0.5)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig_path = "figures/multi_horizon_forecasting_crossover.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved figure: {fig_path}")

    # 4. Value-at-Risk and Basel Expected Shortfall Backtest
    print("\n--- Running VaR and FRTB Expected Shortfall Backtest ---")
    bin_dt = 5.0
    _, _, _, returns = extract_peaks_over_threshold(
        timestamps=timestamps,
        prices=df["price"].values,
        dt_sec=bin_dt,
        quantile=0.90
    )
    t_ret = np.arange(timestamps[0] + bin_dt, timestamps[-1], bin_dt)[:len(returns)]

    # Conditional intensities aligned with return timestamps
    lams = ml_model.predict_intensity(t_ret, timestamps)

    backtest_res = backtest_hawkes_var(
        returns=returns,
        intensities=lams,
        p=0.01,
        quantile=0.90
    )
    print(f"Total Returns N = {backtest_res['N']}, Violations = {backtest_res['hawkes_violations']} (rate: {backtest_res['hawkes_violation_rate']*100:.2f}%)")
    print(f"Kupiec POF p-value: {backtest_res['kupiec_hawkes']['p_value']:.4f} (Passed: {backtest_res['kupiec_hawkes']['passed']})")
    print(f"Christoffersen Ind. p-value: {backtest_res['christoffersen_hawkes']['p_value']:.4f} (Passed: {backtest_res['christoffersen_hawkes']['passed']})")
    print(f"Acerbi-Szekely ES p-value: {backtest_res['acerbi_szekely_hawkes']['p_value']:.4f} (Passed: {backtest_res['acerbi_szekely_hawkes']['passed']})")

    # 5. Bivariate Mutual Excitation Fitting on Real Buyer/Seller Maker Flow
    print("\n--- Fitting Bivariate ML-TFHP on Buyer/Seller Aggressor Flow ---")
    types = (~df["is_buyer_maker"]).astype(int).values  # 0: Aggressive Sell (buyer maker), 1: Aggressive Buy (seller maker)
    # Let's verify mapping: is_buyer_maker=False means aggressive BUY (channel 0)
    channel_types = np.where(df["is_buyer_maker"].values, 1, 0) # 0=Buy, 1=Sell

    biv_model = BivariateMLTFHPModel(M=10, symmetric=True)
    biv_model.fit(timestamps, channel_types, T=T_total, n_starts=3, seed=42)

    mu_b, alpha_b, beta_b, gamma_b, eta_self, eta_cross = biv_model.params
    cross_ratio = biv_model.get_cross_excitation_ratio()
    spec_rad = biv_model.spectral_radius_

    print(f"Bivariate Fit Complete:")
    print(f"  mu: {mu_b:.4f}")
    print(f"  alpha: {alpha_b:.4f}, beta: {beta_b:.4f}, gamma: {gamma_b:.4f}")
    print(f"  eta_self: {eta_self:.4f}, eta_cross: {eta_cross:.4f}")
    print(f"  Spectral Radius rho(Gamma): {spec_rad:.4f}")
    print(f"  Cross/Self Excitation Ratio: {cross_ratio:.4f}")

    # 6. Generate Complete LaTeX Tables for Manuscript
    os.makedirs("reports", exist_ok=True)
    tab1 = generate_model_comparison_latex_table([ehp_model, ml_model])
    tab2 = generate_parameter_estimates_latex_table(ml_model)
    tab3 = generate_var_backtest_latex_table(backtest_res)
    tab4 = generate_multi_horizon_forecasting_latex_table(multi_results)

    # Format Table 5 (Bivariate Results)
    tab5 = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Bivariate ML-TFHP Order Flow Parameter Estimates on Binance BTC/USDT Trades}",
        "\\label{tab:bivariate_estimates}",
        "\\begin{tabular}{lcc}",
        "\\hline\\hline",
        "\\textbf{Parameter / Metric} & \\textbf{Estimate} & \\textbf{Description} \\\\",
        "\\hline",
        f"Baseline Arrival Rate $\\mu$ & {mu_b:.4f} & Exogenous order arrival rate \\\\",
        f"Fractional Memory $\\alpha$ & {alpha_b:.4f} & Sub-diffusive long memory \\\\",
        f"Decay Scale $\\beta$ & {beta_b:.4f} & Microstructural relaxation speed \\\\",
        f"Tempering Rate $\\gamma$ & {gamma_b:.4f} & Macro-economic cutoff frequency \\\\",
        f"Self-Excitation $\\eta_{{\\text{{self}}}}$ & {eta_self:.4f} & Order-splitting cascade intensity \\\\",
        f"Cross-Excitation $\\eta_{{\\text{{cross}}}}$ & {eta_cross:.4f} & Spread-crossing / adverse selection \\\\",
        "\\hline",
        f"Spectral Radius $\\rho(\\mathbf{{\\Gamma}})$ & \\textbf{{{spec_rad:.4f}}} & Stationarity condition ($\\rho < 1$) \\\\",
        f"Cross-Excitation Ratio $\\mathcal{{R}}_{{\\text{{cross}}}}$ & \\textbf{{{cross_ratio:.4f}}} & Ratio $\\eta_{{\\text{{cross}}}} / \\eta_{{\\text{{self}}}}$ \\\\",
        "\\hline\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ]
    tab5_str = "\n".join(tab5)

    all_tables = "\n\n".join([tab1, tab2, tab3, tab4, tab5_str])
    tables_path = "reports/submission_tables.tex"
    with open(tables_path, "w", encoding="utf-8") as f:
        f.write(all_tables)

    print(f"\nWrote all 5 publication tables to {tables_path}")
    print("=" * 70)
    print("ALL ADVANCED EMPIRICAL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

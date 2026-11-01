"""
Verification and Benchmark Runner for Week 4 & Week 5 Deliverables:
- Out-of-sample forward-chaining event forecasting (RMSE, MAE)
- High-Frequency Hawkes-VaR Backtesting (Kupiec POF & Christoffersen Independence)
- Automated LaTeX Table Generation for academic publication
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from ml_tfhp.simulation.ogata import simulate_ml_tfhp
from ml_tfhp.models.exponential_hawkes import ExponentialHawkesModel
from ml_tfhp.models.ml_tfhp_model import MLTFHPModel
from ml_tfhp.evaluation.forecasting import evaluate_out_of_sample_forecasting
from ml_tfhp.evaluation.var_backtest import backtest_hawkes_var
from ml_tfhp.visualization.latex_tables import (
    generate_model_comparison_latex_table,
    generate_parameter_estimates_latex_table,
    generate_var_backtest_latex_table,
)


def run_week4_week5_benchmarks():
    print("=" * 80)
    print("WEEKS 4 & 5: RISK MANAGEMENT, FORECASTING & PUBLICATION SUITE REPORT")
    print("=" * 80)

    # 1. Simulate extended market session with volatility clusters
    print("\n[1] Simulating High-Volatility Trading Session (T=1200 s):")
    T_total = 1200.0
    T_train = 800.0   # Train on first 800s (~67%), test on remaining 400s (~33%)

    events, intensities_sim, _ = simulate_ml_tfhp(
        mu=0.75,
        alpha=0.62,
        beta=1.3,
        gamma_param=0.03,
        eta=0.60,
        T=T_total,
        seed=2026
    )

    print(f"    Total Events Generated       : {len(events)} arrivals on [0, {T_total:.0f} s]")
    print(f"    In-Sample Train Set [0, {T_train:.0f}s] : {np.sum(events <= T_train)} arrivals")
    print(f"    Out-of-Sample Test [>{T_train:.0f}s] : {np.sum(events > T_train)} arrivals")

    # 2. Fit models on training segment
    print("\n[2] Model Fitting on Training Horizon [0, 800s]:")
    print("    - Fitting Baseline EHP...")
    ehp = ExponentialHawkesModel().fit(events, T=T_train, seed=42)

    print("    - Fitting Proposed ML-TFHP Model (with Fisher Hessian inference)...")
    ml_tfhp = MLTFHPModel(M=10).fit(events, T=T_train, n_starts=3, compute_inference=True, seed=42)

    # 3. Out-of-sample forward-chaining event forecasting
    print("\n[3] Out-of-Sample Event Arrival Forecasting (Horizon Delta_t = 15s):")
    print("-" * 65)
    print(f"{'Model':<25} | {'RMSE (arrivals/15s)':<20} | {'MAE':<12}")
    print("-" * 65)

    forecast_res = evaluate_out_of_sample_forecasting(
        events,
        T_train=T_train,
        T_total=T_total,
        delta_t=15.0,
        models=[ehp, ml_tfhp]
    )

    for name, metrics in forecast_res.items():
        print(f"{name:<25} | {metrics['rmse']:<20.4f} | {metrics['mae']:<12.4f}")
    print("-" * 65)

    # 4. High-Frequency Hawkes-VaR Backtesting at 99% Confidence
    print("\n[4] High-Frequency Hawkes-VaR Backtesting (99% Confidence Level):")
    # Generate synthetic price returns driven by instantaneous intensity
    N_steps = 1500
    t_eval = np.linspace(10.0, T_total, N_steps)
    # Predict dynamic intensities from fitted ML-TFHP
    dynamic_lams = ml_tfhp.predict_intensity(t_eval, events=events)

    # Conditional asset returns: volatility scales with lambda(t)
    sigma_t = 0.0004 + 0.0003 * dynamic_lams
    returns = np.random.normal(0, sigma_t)

    backtest_results = backtest_hawkes_var(returns, dynamic_lams, p=0.01, quantile=0.90)

    print(f"    Total Test Observations      : {backtest_results['N']}")
    print(f"    Nominal Target Failure Rate  : {backtest_results['target_p']*100:.1f}% ({backtest_results['expected_violations']} expected)")
    print(f"    Hawkes-VaR Violations        : {backtest_results['hawkes_violations']} ({backtest_results['hawkes_violation_rate']*100:.2f}%)")
    print(f"    Static VaR Violations        : {backtest_results['static_violations']} ({backtest_results['static_violation_rate']*100:.2f}%)")
    print(f"    Kupiec POF Test (Hawkes-VaR) : LR = {backtest_results['kupiec_hawkes']['stat']:.4f}, p-val = {backtest_results['kupiec_hawkes']['p_value']:.4f} -> {'PASS' if backtest_results['kupiec_hawkes']['passed'] else 'FAIL'}")
    print(f"    Christoffersen Indep. Test   : LR = {backtest_results['christoffersen_hawkes']['stat']:.4f}, p-val = {backtest_results['christoffersen_hawkes']['p_value']:.4f} -> {'PASS (No Clustering)' if backtest_results['christoffersen_hawkes']['passed'] else 'FAIL'}")
    print(f"    Conditional Coverage LR_CC   : LR = {backtest_results['conditional_coverage_hawkes']['stat']:.4f}, p-val = {backtest_results['conditional_coverage_hawkes']['p_value']:.4f}")

    # 5. Automated LaTeX Table Generation
    print("\n[5] Generating Journal Submission LaTeX Tables...")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    tex_comparison = generate_model_comparison_latex_table([ehp, ml_tfhp])
    tex_params = generate_parameter_estimates_latex_table(ml_tfhp)
    tex_var = generate_var_backtest_latex_table(backtest_results)

    combined_tex = f"""% =========================================================================
% ML-TFHP Academic Paper Submission Tables (Generated Automatically)
% =========================================================================

% Table 1: Model Comparison
{tex_comparison}

\\vspace{{1cm}}

% Table 2: Parameter Estimates
{tex_params}

\\vspace{{1cm}}

% Table 3: Regulatory VaR Backtesting
{tex_var}
"""
    output_tex_path = reports_dir / "submission_tables.tex"
    with open(output_tex_path, "w", encoding="utf-8") as f:
        f.write(combined_tex)

    print(f"    -> Exported all tables to: {output_tex_path}")

    print("\n" + "=" * 80)
    print("ALL WEEK 4 & WEEK 5 BENCHMARKS SUCCESSFULLY EXECUTED")
    print("=" * 80)


if __name__ == "__main__":
    run_week4_week5_benchmarks()

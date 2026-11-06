"""
Comprehensive Empirical Verification Script.
Computes and verifies ALL empirical values across all 8 tables and 7 figures.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ml_tfhp.models.ml_tfhp_model import MLTFHPModel
from ml_tfhp.models.exponential_hawkes import ExponentialHawkesModel
from ml_tfhp.models.bivariate_ml_tfhp import BivariateMLTFHPModel
from ml_tfhp.data.event_extractor import extract_peaks_over_threshold
from ml_tfhp.evaluation.var_backtest import (
    backtest_hawkes_var,
    kupiec_pof_test,
    christoffersen_independence_test,
    acerbi_szekely_test,
    fit_intensity_gpd
)
from ml_tfhp.evaluation.forecasting import evaluate_multi_horizon_forecasting

def run_audit():
    print("=" * 80)
    print("STEP 1: DATA DESCRIPTION & DFA MEMORY")
    print("=" * 80)
    df = pd.read_csv("data/real_btcusdt_trades.csv")
    print(f"Total rows in CSV: {len(df)}")
    
    # 10,000 trades sample as used in manuscript
    df_10k = df.iloc[:10000].copy()
    timestamps = df_10k["time_sec"].values
    prices = df_10k["price"].values
    is_buyer_maker = df_10k["is_buyer_maker"].values
    
    T_total = float(timestamps[-1])
    N_total = len(timestamps)
    price_min = float(prices.min())
    price_max = float(prices.max())
    n_buys = int(np.sum(~is_buyer_maker))  # taker buys
    n_sells = int(np.sum(is_buyer_maker))  # taker sells
    
    print(f"N_total: {N_total}")
    print(f"T_total: {T_total:.2f} s ({T_total/60.0:.2f} minutes)")
    print(f"Price range: ${price_min:.2f} - ${price_max:.2f}")
    print(f"Taker Buys: {n_buys}, Taker Sells: {n_sells}")
    
    # In-sample 70%
    T_train = T_total * 0.70
    train_mask = timestamps <= T_train
    train_events = timestamps[train_mask]
    N_train = len(train_events)
    test_events = timestamps[~train_mask]
    N_test = len(test_events)
    print(f"In-sample T_train: {T_train:.2f} s, N_train: {N_train} trades ({N_train/N_total*100:.1f}%)")
    print(f"Out-of-sample: {N_test} trades ({N_test/N_total*100:.1f}%)")
    
    # DFA on order signs
    signs = np.where(is_buyer_maker, -1.0, 1.0)
    cum_signs = np.cumsum(signs)
    
    # Simple DFA calculation
    def calc_dfa(x, scales=None):
        if scales is None:
            scales = np.floor(2 ** np.linspace(3, 10, 15)).astype(int)
        F = []
        for s in scales:
            n_segments = len(x) // s
            res = 0.0
            for i in range(n_segments):
                seg = x[i*s : (i+1)*s]
                t = np.arange(s)
                p = np.polyfit(t, seg, 1)
                fit = np.polyval(p, t)
                res += np.sum((seg - fit) ** 2)
            F.append(np.sqrt(res / (n_segments * s)))
        log_s = np.log10(scales)
        log_F = np.log10(F)
        slope, intercept, r_val, p_val, std_err = stats.linregress(log_s, log_F)
        return slope, r_val**2
        
    H_dfa, r2_dfa = calc_dfa(cum_signs)
    print(f"DFA on cumulative signed order flow: H = {H_dfa:.4f}, R^2 = {r2_dfa:.4f}")

    print("\n" + "=" * 80)
    print("STEP 2: IN-SAMPLE MODEL CALIBRATION (TABLE 2 & TABLE 3)")
    print("=" * 80)
    # Fit ML-TFHP (M=12)
    ml_model = MLTFHPModel(M=12).fit(train_events, T=T_train, n_starts=5, compute_inference=True, seed=42)
    mu_ml, alpha_ml, beta_ml, gamma_ml, eta_ml = ml_model.params
    n_star_ml = ml_model.branching_ratio_
    ll_ml = ml_model.log_likelihood_
    aic_ml = ml_model.aic_
    bic_ml = ml_model.bic_
    ci_info = ml_model.get_confidence_intervals()
    print("Parameter CIs:")
    for k, v in ci_info.items():
        print(f"  {k}: {v['estimate']:.4f} (SE: {v['std_error']:.4e}, 95% CI: [{v['ci_95_lower']:.4f}, {v['ci_95_upper']:.4f}])")
    
    print(f"ML-TFHP (M=12):")
    print(f"  params: mu={mu_ml:.4f}, alpha={alpha_ml:.4f}, beta={beta_ml:.4f}, gamma={gamma_ml:.4f}, eta={eta_ml:.4f}")
    print(f"  branching ratio n*: {n_star_ml:.4f}")
    print(f"  log-likelihood: {ll_ml:.2f}")
    print(f"  AIC: {aic_ml:.2f}, BIC: {bic_ml:.2f}")
    
    # Fit Exponential Hawkes
    ehp_model = ExponentialHawkesModel().fit(train_events, T=T_train, seed=42)
    mu_exp, alpha_exp, beta_exp = ehp_model.params
    n_star_exp = ehp_model.branching_ratio_
    ll_exp = ehp_model.log_likelihood_
    aic_exp = ehp_model.aic_
    bic_exp = ehp_model.bic_
    print(f"Exponential Hawkes:")
    print(f"  params: mu={mu_exp:.4f}, alpha={alpha_exp:.4f}, beta={beta_exp:.4f}")
    print(f"  branching ratio n*: {n_star_exp:.4f}")
    print(f"  log-likelihood: {ll_exp:.2f}")
    print(f"  AIC: {aic_exp:.2f}, BIC: {bic_exp:.2f}")
    
    # Bi-Exponential Hawkes
    def bi_exp_ll(params):
        mu, a1, b1, a2, b2 = params
        if mu <= 0 or a1 < 0 or b1 <= 0 or a2 < 0 or b2 <= 0: return 1e12
        if (a1/b1 + a2/b2) >= 0.999: return 1e12
        ll = np.log(mu)
        r1, r2 = 0.0, 0.0
        for i in range(1, N_train):
            dt = train_events[i] - train_events[i-1]
            r1 = (r1 + a1) * np.exp(-b1 * dt)
            r2 = (r2 + a2) * np.exp(-b2 * dt)
            lam = mu + r1 + r2
            if lam <= 0: return 1e12
            ll += np.log(lam)
        comp = mu * T_train + (a1/b1) * np.sum(1.0 - np.exp(-b1 * (T_train - train_events))) + (a2/b2) * np.sum(1.0 - np.exp(-b2 * (T_train - train_events)))
        return -(ll - comp)
        
    res_bi = minimize(bi_exp_ll, [2.5, 0.5, 10.0, 0.2, 1.0], method='L-BFGS-B', bounds=[(1e-4, 50), (0, 50), (1e-3, 50), (0, 50), (1e-3, 20)])
    ll_bi = -res_bi.fun
    k_bi = 5
    aic_bi = 2*k_bi - 2*ll_bi
    bic_bi = k_bi*np.log(N_train) - 2*ll_bi
    n_star_bi = res_bi.x[1]/res_bi.x[2] + res_bi.x[3]/res_bi.x[4]
    print(f"Bi-Exponential Hawkes:")
    print(f"  params: {res_bi.x}")
    print(f"  branching ratio n*: {n_star_bi:.4f}")
    print(f"  log-likelihood: {ll_bi:.2f}")
    print(f"  AIC: {aic_bi:.2f}, BIC: {bic_bi:.2f}")
    
    # LRT & Delta AIC
    delta_ll = ll_ml - ll_exp
    delta_aic = aic_exp - aic_ml
    lrt_stat = 2.0 * delta_ll
    p_lrt = stats.chi2.sf(lrt_stat, df=2)
    print(f"Model Comparison vs Exponential Hawkes:")
    print(f"  Delta ln L: {delta_ll:+.2f}")
    print(f"  Delta AIC: {delta_aic:+.2f}")
    print(f"  LRT statistic: {lrt_stat:.2f}, p-val: {p_lrt}")
    
    # Papangelou Residuals and KS test
    res_ml = ml_model.compute_residuals()
    ks_stat_ml, ks_pval_ml = stats.kstest(res_ml, 'expon')
    res_exp = ehp_model.compute_residuals()
    ks_stat_exp, ks_pval_exp = stats.kstest(res_exp, 'expon')
    print(f"KS Test on Residuals:")
    print(f"  ML-TFHP: KS stat = {ks_stat_ml:.4f}, p-val = {ks_pval_ml:.4e}")
    print(f"  Exponential Hawkes: KS stat = {ks_stat_exp:.4f}, p-val = {ks_pval_exp:.4e}")

    # Reflexivity
    refl_mult = 1.0 / ((1.0 - n_star_ml) ** 2)
    var_amp = (refl_mult - 1.0) * 100.0
    print(f"Market Reflexivity Multiplier: R_refl = (1 - n*)^(-2) = {refl_mult:.4f} (+{var_amp:.1f}% variance amplification)")

    print("\n" + "=" * 80)
    print("STEP 3: MULTI-HORIZON FORECASTING (TABLE 5 & FIGURE 3)")
    print("=" * 80)
    horizons = [5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0]
    
    class PoissonBaseline:
        def __init__(self, rate):
            self.name = "Poisson"
            self.rate = rate
            self.is_fitted = True
            self.params = np.array([rate])
            self.log_likelihood_ = float(N_train * np.log(rate) - rate * T_train)
            self.branching_ratio_ = 0.0
        def predict_intensity(self, t, events=None):
            return np.full_like(t, self.rate)
        def predict_counts(self, current_time, horizon, history_events=None):
            return np.full_like(horizon, self.rate * horizon)

    uncond_rate = float(N_train / T_train)
    p_model = PoissonBaseline(uncond_rate)
    
    multi_results = evaluate_multi_horizon_forecasting(
        events=timestamps,
        T_train=T_train,
        T_total=T_total,
        horizons=horizons,
        models=[p_model, ehp_model, ml_model]
    )
    
    for h in horizons:
        p_res = multi_results[h]["Poisson"]
        e_res = multi_results[h]["ExponentialHawkes"]
        m_res = multi_results[h]["ML-TFHP"]
        print(f"Horizon {h:3.0f}s: Poisson [RMSE={p_res['rmse']:.3f}, MAE={p_res['mae']:.3f}] | EHP [RMSE={e_res['rmse']:.3f}, MAE={e_res['mae']:.3f}] | ML-TFHP [RMSE={m_res['rmse']:.3f}, MAE={m_res['mae']:.3f}]")

    print("\n" + "=" * 80)
    print("STEP 4: VALUE-AT-RISK & EXPECTED SHORTFALL REGULATORY BACKTESTING (TABLE 4 & FIGURE 6)")
    print("=" * 80)
    bin_dt = 5.0
    _, _, _, returns = extract_peaks_over_threshold(
        timestamps=timestamps,
        prices=prices,
        dt_sec=bin_dt,
        quantile=0.90
    )
    t_ret = np.arange(timestamps[0] + bin_dt, timestamps[-1], bin_dt)[:len(returns)]
    lams_ret = ml_model.predict_intensity(t_ret, timestamps)
    
    backtest_res = backtest_hawkes_var(
        returns=returns,
        intensities=lams_ret,
        p=0.01,
        quantile=0.90
    )
    
    losses = -returns * 100.0
    var_hawkes = backtest_res["var_series"] * 100.0
    es_hawkes = backtest_res["es_series"] * 100.0
    violations_all = losses > var_hawkes
    
    # Full session:
    n_all = backtest_res["N"]
    n_viol_all = backtest_res["hawkes_violations"]
    rate_all = backtest_res["hawkes_violation_rate"] * 100.0
    kup_all = backtest_res['kupiec_hawkes']
    chr_all = backtest_res['christoffersen_hawkes']
    ace_all = backtest_res['acerbi_szekely_hawkes']
    
    n_viol_s_all = backtest_res["static_violations"]
    rate_s_all = backtest_res["static_violation_rate"] * 100.0
    kup_s_all = backtest_res['kupiec_static']
    chr_s_all = backtest_res['christoffersen_static']
    ace_s_all = backtest_res['acerbi_szekely_static']

    print(f"Full Session (N = {n_all} intervals, dt = 5s, total = {n_all*bin_dt:.1f}s):")
    print(f"  ML-TFHP Dynamic: Violations = {n_viol_all} ({rate_all:.2f}%) | Kupiec LR = {kup_all['stat']:.4f}, p = {kup_all['p_value']:.4f} | Christoffersen LR = {chr_all['stat']:.4f}, p = {chr_all['p_value']:.4f} | Acerbi-Szekely Z2 = {ace_all['stat']:.4f}, p = {ace_all['p_value']:.4f}")
    print(f"  Static Empirical: Violations = {n_viol_s_all} ({rate_s_all:.2f}%) | Kupiec LR = {kup_s_all['stat']:.4f}, p = {kup_s_all['p_value']:.4f} | Christoffersen LR = {chr_s_all['stat']:.4f}, p = {chr_s_all['p_value']:.4f} | Acerbi-Szekely Z2 = {ace_s_all['stat']:.4f}, p = {ace_s_all['p_value']:.4f}")

    # Strict OOS partition (t > T_train):
    oos_mask = t_ret > T_train
    losses_oos = -returns[oos_mask]
    var_oos = backtest_res["var_series"][oos_mask]
    es_oos = backtest_res["es_series"][oos_mask]
    lams_oos = lams_ret[oos_mask]
    viol_oos = (losses_oos > var_oos).astype(int)
    n_oos = len(losses_oos)
    n_viol_oos = int(np.sum(viol_oos))
    rate_oos = n_viol_oos / n_oos * 100.0
    kup_stat_oos, kup_pval_oos, kup_pass_oos = kupiec_pof_test(n_viol_oos, n_oos, 0.01)
    chr_stat_oos, chr_pval_oos, chr_pass_oos = christoffersen_independence_test(viol_oos)
    gpd_params = backtest_res["gpd_params"]
    z2_oos, p_es_oos, pass_es_oos = acerbi_szekely_test(
        losses=losses_oos,
        var_series=var_oos,
        es_series=es_oos,
        xi=gpd_params["xi"],
        a=gpd_params["a"],
        b=gpd_params["b"],
        intensities=lams_oos,
        p=0.01,
        p_u=0.10
    )
    
    # Static VaR OOS
    static_var_val = float(np.percentile(-returns, 99.0))
    static_es_val = float(np.mean(-returns[-returns > static_var_val])) if np.any(-returns > static_var_val) else static_var_val * 1.2
    viol_s_oos = (-returns[oos_mask] > static_var_val).astype(int)
    n_viol_s_oos = int(np.sum(viol_s_oos))
    rate_s_oos = n_viol_s_oos / n_oos * 100.0
    kup_stat_s_oos, kup_pval_s_oos, kup_pass_s_oos = kupiec_pof_test(n_viol_s_oos, n_oos, 0.01)
    chr_stat_s_oos, chr_pval_s_oos, chr_pass_s_oos = christoffersen_independence_test(viol_s_oos)
    z2_s_oos, p_es_s_oos, pass_es_s_oos = acerbi_szekely_test(
        losses=losses_oos,
        var_series=np.full(n_oos, static_var_val),
        es_series=np.full(n_oos, static_es_val),
        xi=gpd_params["xi"],
        a=gpd_params["a"],
        b=gpd_params["b"],
        intensities=lams_oos,
        p=0.01,
        p_u=0.10
    )

    print(f"Strict Out-of-Sample Window (N = {n_oos} intervals, dt = 5s, total = {n_oos*bin_dt:.1f}s):")
    print(f"  ML-TFHP Dynamic: Violations = {n_viol_oos} ({rate_oos:.2f}%) | Kupiec LR = {kup_stat_oos:.4f}, p = {kup_pval_oos:.4f} | Christoffersen LR = {chr_stat_oos:.4f}, p = {chr_pval_oos:.4f} | Acerbi-Szekely Z2 = {z2_oos:.4f}, p = {p_es_oos:.4f}")
    print(f"  Static Empirical: Violations = {n_viol_s_oos} ({rate_s_oos:.2f}%) | Kupiec LR = {kup_stat_s_oos:.4f}, p = {kup_pval_s_oos:.4f} | Christoffersen LR = {chr_stat_s_oos:.4f}, p = {chr_pval_s_oos:.4f} | Acerbi-Szekely Z2 = {z2_s_oos:.4f}, p = {p_es_s_oos:.4f}")

    print("\n" + "=" * 80)
    print("STEP 5: BIVARIATE MODEL ESTIMATION (TABLE 6 & FIGURE 7)")
    print("=" * 80)
    biv_model = BivariateMLTFHPModel(M=10).fit(timestamps, is_buyer_maker, T=T_total, n_starts=3, seed=42)
    mu_b, alpha_b, beta_b, gamma_b, eta_self, eta_cross = biv_model.params
    gamma_denom = gamma_b**alpha_b + beta_b**alpha_b
    gamma_mat = np.array([
        [eta_self / gamma_denom, eta_cross / gamma_denom],
        [eta_cross / gamma_denom, eta_self / gamma_denom]
    ])
    spec_rad = float(np.max(np.abs(np.linalg.eigvals(gamma_mat))))
    cross_ratio = float(eta_cross / eta_self)
    ll_biv = float(biv_model.log_likelihood_)
    
    print(f"Bivariate ML-TFHP Estimates:")
    print(f"  mu: {mu_b:.4f}")
    print(f"  alpha: {alpha_b:.4f}")
    print(f"  beta: {beta_b:.4f}")
    print(f"  gamma: {gamma_b:.4f}")
    print(f"  eta_self: {eta_self:.4f}")
    print(f"  eta_cross: {eta_cross:.4f}")
    print(f"  Branching matrix Gamma:\n{gamma_mat}")
    print(f"  Spectral radius: {spec_rad:.4f}")
    print(f"  Cross-to-self ratio R_cross: {cross_ratio:.4f} (1/R_cross = {1.0/cross_ratio:.1f}x)")
    print(f"  Log-likelihood: {ll_biv:.2f}")

    print("\n" + "=" * 80)
    print("STEP 6: QUADRATURE SENSITIVITY (TABLE 7)")
    print("=" * 80)
    m_nodes = [8, 10, 12, 16, 20]
    quad_results = {}
    for M in m_nodes:
        m_fit = MLTFHPModel(M=M).fit(timestamps, T=T_total, n_starts=3, seed=42)
        quad_results[M] = {
            'll': m_fit.log_likelihood_,
            'params': m_fit.params,
            'n_star': m_fit.branching_ratio_
        }
        p = m_fit.params
        print(f"M={M:2d}: ln L = {m_fit.log_likelihood_:.2f}, mu={p[0]:.4f}, alpha={p[1]:.4f}, beta={p[2]:.4f}, gamma={p[3]:.4f}, eta={p[4]:.4f}, n*={m_fit.branching_ratio_:.4f}")
        
    ref_ll = quad_results[20]['ll']
    for M in m_nodes:
        rel_err = abs(quad_results[M]['ll'] - ref_ll) / abs(ref_ll)
        print(f"M={M:2d}: Rel Error vs M=20 = {rel_err:.2e}")

    print("\n" + "=" * 80)
    print("STEP 7: TEMPORAL SUBSAMPLE STABILITY (TABLE 8)")
    print("=" * 80)
    # Full 10,000
    m_full = MLTFHPModel(M=12).fit(timestamps, T=T_total, n_starts=3, seed=42)
    p_full = m_full.params
    n_full = m_full.branching_ratio_
    print(f"Full Sample (N=10,000, T={T_total:.1f}s): mu={p_full[0]:.4f}, alpha={p_full[1]:.4f}, beta={p_full[2]:.4f}, gamma={p_full[3]:.4f}, eta={p_full[4]:.4f}, n*={n_full:.4f}")

    # Subsample 1: 1-5,000
    ts_sub1 = timestamps[:5000]
    T_sub1 = float(ts_sub1[-1])
    m_sub1 = MLTFHPModel(M=12).fit(ts_sub1, T=T_sub1, n_starts=3, seed=42)
    p_sub1 = m_sub1.params
    n_sub1 = m_sub1.branching_ratio_
    print(f"Subsample 1 (N=5,000, T={T_sub1:.1f}s): mu={p_sub1[0]:.4f}, alpha={p_sub1[1]:.4f}, beta={p_sub1[2]:.4f}, gamma={p_sub1[3]:.4f}, eta={p_sub1[4]:.4f}, n*={n_sub1:.4f}")

    # Subsample 2: 5,001-10,000
    ts_sub2 = timestamps[5000:] - timestamps[5000]  # relative times
    T_sub2 = float(ts_sub2[-1])
    m_sub2 = MLTFHPModel(M=12).fit(ts_sub2, T=T_sub2, n_starts=3, seed=42)
    p_sub2 = m_sub2.params
    n_sub2 = m_sub2.branching_ratio_
    print(f"Subsample 2 (N=5,000, T={T_sub2:.1f}s): mu={p_sub2[0]:.4f}, alpha={p_sub2[1]:.4f}, beta={p_sub2[2]:.4f}, gamma={p_sub2[3]:.4f}, eta={p_sub2[4]:.4f}, n*={n_sub2:.4f}")
    
    alpha_diff = abs(p_sub1[1] - p_sub2[1]) / p_sub1[1] * 100.0
    n_diff = abs(n_sub1 - n_sub2) / n_sub1 * 100.0
    print(f"Stability variation: alpha diff = {alpha_diff:.2f}%, n* diff = {n_diff:.2f}%")

if __name__ == "__main__":
    run_audit()

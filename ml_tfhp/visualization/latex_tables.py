"""
Automated LaTeX Table Generator for Q1 Academic Journal Submissions.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ..evaluation.model_selection import compare_models


def generate_model_comparison_latex_table(
    models: list,
    caption: str = "Empirical Model Comparison and Goodness-of-Fit Diagnostics",
    label: str = "tab:model_comparison"
) -> str:
    """
    Format comparative model metrics into publication-ready LaTeX table.
    """
    rows = compare_models(models)

    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lccccccr}",
        "\\hline\\hline",
        "\\textbf{Model} & $\\ln \\hat{L}$ & \\textbf{AIC} & $\\Delta\\text{AIC}$ & \\textbf{BIC} & $n^*$ & \\textbf{KS $p$-val} & \\textbf{Status} \\\\",
        "\\hline"
    ]

    for r in rows:
        name = r["model"]
        ll = f"{r['log_likelihood']:.2f}"
        aic = f"{r['aic']:.2f}"
        daic = f"{r['delta_aic']:.2f}"
        bic = f"{r['bic']:.2f}"
        n_star = f"{r['branching_ratio']:.3f}"
        ks_p = f"{r['ks_pvalue']:.4f}"
        status = "Pass" if r["passes_ks"] else "Reject"

        latex.append(f"{name} & {ll} & {aic} & {daic} & {bic} & {n_star} & {ks_p} & {status} \\\\")

    latex.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    return "\n".join(latex)


def generate_parameter_estimates_latex_table(
    model: Any,
    caption: str = "ML-TFHP Parameter Estimates and 95\\% Confidence Intervals",
    label: str = "tab:parameter_estimates"
) -> str:
    """
    Format estimated parameters and asymptotic standard errors into a LaTeX table.
    """
    if not model.is_fitted or model.inference_ is None:
        raise RuntimeError("Model inference not available.")

    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccc}",
        "\\hline\\hline",
        "\\textbf{Parameter} & \\textbf{Estimate} & \\textbf{Std. Error} & \\textbf{95\\% CI Lower} & \\textbf{95\\% CI Upper} \\\\",
        "\\hline"
    ]

    cis = model.get_confidence_intervals()
    symbol_map = {
        "mu": "$\\mu$",
        "alpha": "$\\alpha$",
        "beta": "$\\beta$",
        "gamma": "$\\gamma$",
        "eta": "$\\eta$"
    }

    for name, data in cis.items():
        sym = symbol_map.get(name, name)
        est = f"{data['estimate']:.4f}"
        se = f"{data['std_error']:.4f}"
        low = f"{data['ci_95_lower']:.4f}"
        high = f"{data['ci_95_upper']:.4f}"
        latex.append(f"{sym} & {est} & {se} & {low} & {high} \\\\")

    latex.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    return "\n".join(latex)


def generate_var_backtest_latex_table(
    backtest: Dict[str, Any],
    caption: str = "High-Frequency Value-at-Risk and Expected Shortfall Regulatory Backtesting at 99\\% Confidence",
    label: str = "tab:var_backtest"
) -> str:
    """
    Format VaR and ES backtesting results (Kupiec POF, Christoffersen independence, Acerbi-Szekely) into LaTeX.
    """
    has_es = "acerbi_szekely_hawkes" in backtest
    if has_es:
        latex = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lccccc}",
            "\\hline\\hline",
            "\\textbf{Model} & \\textbf{Violations (Rate)} & \\textbf{Kupiec POF $p$} & \\textbf{Christoffersen Ind. $p$} & \\textbf{Acerbi-Sz\\'ekely $p$} & \\textbf{FRTB Decision} \\\\",
            "\\hline"
        ]
        # Hawkes row
        h_v = f"{backtest['hawkes_violations']} ({backtest['hawkes_violation_rate']*100:.2f}\\%)"
        h_kup = f"{backtest['kupiec_hawkes']['p_value']:.4f}"
        h_ind = f"{backtest['christoffersen_hawkes']['p_value']:.4f}"
        h_es = f"{backtest['acerbi_szekely_hawkes']['p_value']:.4f}"
        h_pass = backtest['kupiec_hawkes']['passed'] and backtest['christoffersen_hawkes']['passed'] and backtest['acerbi_szekely_hawkes']['passed']
        h_dec = "\\textbf{Pass (Compliant)}" if h_pass else "Fail"
        latex.append(f"ML-TFHP Dynamic VaR/ES & {h_v} & {h_kup} & {h_ind} & {h_es} & {h_dec} \\\\")

        # Static row
        s_v = f"{backtest['static_violations']} ({backtest['static_violation_rate']*100:.2f}\\%)"
        s_kup = f"{backtest['kupiec_static']['p_value']:.4f}"
        s_ind = f"{backtest['christoffersen_static']['p_value']:.4f}"
        s_es = f"{backtest['acerbi_szekely_static']['p_value']:.4f}"
        s_pass = backtest['kupiec_static']['passed'] and backtest['christoffersen_static']['passed'] and backtest['acerbi_szekely_static']['passed']
        s_dec = "Pass" if s_pass else "\\textbf{Fail (Non-compliant)}"
        latex.append(f"Static Empirical VaR/ES & {s_v} & {s_kup} & {s_ind} & {s_es} & {s_dec} \\\\")
    else:
        latex = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcccc}",
            "\\hline\\hline",
            "\\textbf{Model} & \\textbf{Violations (Rate)} & \\textbf{Kupiec POF $p$-val} & \\textbf{Christoffersen Ind. $p$-val} & \\textbf{Decision} \\\\",
            "\\hline"
        ]
        h_v = f"{backtest['hawkes_violations']} ({backtest['hawkes_violation_rate']*100:.2f}\\%)"
        h_kup = f"{backtest['kupiec_hawkes']['p_value']:.4f}"
        h_ind = f"{backtest['christoffersen_hawkes']['p_value']:.4f}"
        h_dec = "Pass" if (backtest['kupiec_hawkes']['passed'] and backtest['christoffersen_hawkes']['passed']) else "Fail"
        latex.append(f"ML-TFHP Dynamic VaR & {h_v} & {h_kup} & {h_ind} & {h_dec} \\\\")

        s_v = f"{backtest['static_violations']} ({backtest['static_violation_rate']*100:.2f}\\%)"
        s_kup = f"{backtest['kupiec_static']['p_value']:.4f}"
        s_ind = f"{backtest['christoffersen_static']['p_value']:.4f}"
        s_dec = "Pass" if (backtest['kupiec_static']['passed'] and backtest['christoffersen_static']['passed']) else "Fail"
        latex.append(f"Static Empirical VaR & {s_v} & {s_kup} & {s_ind} & {s_dec} \\\\")

    latex.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    return "\n".join(latex)


def generate_multi_horizon_forecasting_latex_table(
    multi_horizon_results: Dict[float, Dict[str, Dict[str, float]]],
    caption: str = "Out-of-sample Multi-Horizon Event Arrival Forecast Accuracy (RMSE and MAE)",
    label: str = "tab:multi_horizon_forecasting"
) -> str:
    """
    Format multi-horizon forecasting comparison table across horizons into LaTeX.
    """
    horizons = sorted(list(multi_horizon_results.keys()))
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccccc}",
        "\\hline\\hline",
        "\\textbf{Horizon $\\Delta t$} & \\multicolumn{2}{c}{\\textbf{Homogeneous Poisson}} & \\multicolumn{2}{c}{\\textbf{Exponential Hawkes}} & \\multicolumn{2}{c}{\\textbf{ML-TFHP (Ours)}} \\\\",
        "& RMSE & MAE & RMSE & MAE & \\textbf{RMSE} & \\textbf{MAE} \\\\",
        "\\hline"
    ]

    for h in horizons:
        res_h = multi_horizon_results[h]
        p_rmse = res_h.get("Poisson", {}).get("rmse", float("nan"))
        p_mae = res_h.get("Poisson", {}).get("mae", float("nan"))
        e_rmse = res_h.get("ExponentialHawkes", {}).get("rmse", float("nan"))
        e_mae = res_h.get("ExponentialHawkes", {}).get("mae", float("nan"))
        m_rmse = res_h.get("ML-TFHP", {}).get("rmse", float("nan"))
        m_mae = res_h.get("ML-TFHP", {}).get("mae", float("nan"))

        # Highlight best in bold
        best_rmse = min(e_rmse, m_rmse) if not np.isnan(e_rmse) else m_rmse
        e_str = f"\\textbf{{{e_rmse:.3f}}}" if e_rmse == best_rmse else f"{e_rmse:.3f}"
        m_str = f"\\textbf{{{m_rmse:.3f}}}" if m_rmse == best_rmse else f"{m_rmse:.3f}"

        latex.append(f"{int(h) if h.is_integer() else h}s & {p_rmse:.3f} & {p_mae:.3f} & {e_str} & {e_mae:.3f} & {m_str} & \\textbf{{{m_mae:.3f}}} \\\\")

    latex.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    return "\n".join(latex)

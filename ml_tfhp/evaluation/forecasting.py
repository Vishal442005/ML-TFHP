"""
Out-of-Sample Forecasting for Hawkes Process Event Intensities and Counts.
Supports both naive short-term and closed-form dynamic renewal renewal expectations
across multi-horizon forecasting sweeps.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import scipy.linalg as la


def predict_future_event_count(
    t_start: float,
    delta_t: float,
    events_history: np.ndarray,
    mu: float,
    weights: np.ndarray,
    zetas: np.ndarray,
    exact_renewal: bool = True
) -> float:
    """
    Predict the expected number of events in future interval [t_start, t_start + delta_t]:
        E[N(t_start, t_start + delta_t) | H_{t_start}] = int_{0}^{delta_t} E[lambda(t_start + s) | H_{t_start}] ds

    When exact_renewal is True, accounts for endogenous mutual excitation over [t_start, t_start + delta_t]
    via the closed-form matrix differential equation for Markovian state vectors A(t):
        d E[A(s)]/ds = (1 w^T - diag(zeta)) E[A(s)] + mu 1.
    As delta_t -> infinity, converges analytically to [mu / (1 - n^*)] * delta_t.
    """
    past = events_history[events_history < t_start]
    M = len(weights)
    n_star = float(np.sum(weights / zetas))
    if len(past) == 0:
        if n_star < 0.9999:
            return float(mu / (1.0 - n_star) * delta_t)
        return float(mu * delta_t)

    # Compute memory state at t_start
    lags = t_start - past
    decay = np.exp(-np.outer(lags, zetas))  # (n_past, M)
    A0 = np.sum(decay, axis=0)              # shape (M,)

    if not exact_renewal or n_star >= 0.9999:
        # Naive first-order expectation (neglects subsequent arrivals in [t, t + delta_t])
        expected_count = mu * delta_t + np.sum((weights * A0 / zetas) * (1.0 - np.exp(-zetas * delta_t)))
        return float(expected_count)

    # Exact closed-form renewal integration
    M_dyn = np.ones((M, 1)) @ weights.reshape(1, M) - np.diag(zetas)
    try:
        inv_M = la.inv(M_dyn)
        expM = la.expm(M_dyn * delta_t)
        term1 = inv_M @ (expM - np.eye(M)) @ A0
        term2 = inv_M @ (inv_M @ (expM - np.eye(M)) - delta_t * np.eye(M)) @ (mu * np.ones(M))
        int_A = term1 + term2
        expected_count = mu * delta_t + np.dot(weights, int_A)
        return max(0.0, float(expected_count))
    except Exception:
        # Fallback to naive first-order
        expected_count = mu * delta_t + np.sum((weights * A0 / zetas) * (1.0 - np.exp(-zetas * delta_t)))
        return float(expected_count)


def evaluate_out_of_sample_forecasting(
    events: np.ndarray,
    T_train: float,
    T_total: float,
    delta_t: float,
    models: list
) -> Dict[str, Dict[str, float]]:
    """
    Perform rolling forward-chaining evaluation of future event arrival counts.
    """
    train_events = events[events <= T_train]
    eval_times = np.arange(T_train, T_total - delta_t, delta_t)
    actual_counts = np.empty(len(eval_times), dtype=float)

    for i, t in enumerate(eval_times):
        actual_counts[i] = np.sum((events >= t) & (events < t + delta_t))

    results = {}
    for m in models:
        predicted_counts = np.empty(len(eval_times), dtype=float)

        if m.name == "ML-TFHP":
            mu = m.params[0]
            weights = m.weights_
            zetas = m.zetas_
            for i, t in enumerate(eval_times):
                predicted_counts[i] = predict_future_event_count(t, delta_t, events, mu, weights, zetas)
        elif m.name == "ExponentialHawkes":
            mu, alpha, beta = m.params
            weights = np.array([alpha * beta])
            zetas = np.array([beta])
            for i, t in enumerate(eval_times):
                predicted_counts[i] = predict_future_event_count(t, delta_t, events, mu, weights, zetas)
        else:
            # Baseline: Poisson rate
            poisson_rate = len(train_events) / T_train
            predicted_counts[:] = poisson_rate * delta_t

        rmse = float(np.sqrt(np.mean((actual_counts - predicted_counts) ** 2)))
        mae = float(np.mean(np.abs(actual_counts - predicted_counts)))
        results[m.name] = {"rmse": rmse, "mae": mae}

    return results


def evaluate_multi_horizon_forecasting(
    events: np.ndarray,
    T_train: float,
    T_total: float,
    horizons: List[float],
    models: list
) -> Dict[float, Dict[str, Dict[str, float]]]:
    """
    Evaluate forecasting performance across multiple time horizons delta_t.
    Demonstrates long-memory Hawkes superiority over exponential decay at medium and long horizons.

    Parameters
    ----------
    events : ndarray
        Event arrival times.
    T_train : float
        Training end time.
    T_total : float
        Total observation time.
    horizons : list of float
        List of forecast windows delta_t (e.g. [5.0, 15.0, 60.0, 300.0, 600.0, 1200.0]).
    models : list
        List of fitted BaseHawkesModel instances.

    Returns
    -------
    dict mapping horizon -> model_name -> {'rmse': float, 'mae': float}.
    """
    horizon_results = {}
    for h in horizons:
        res_h = evaluate_out_of_sample_forecasting(
            events=events,
            T_train=T_train,
            T_total=T_total,
            delta_t=h,
            models=models
        )
        horizon_results[h] = res_h
    return horizon_results


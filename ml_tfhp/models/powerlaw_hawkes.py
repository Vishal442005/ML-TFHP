"""
Baseline 2: Power-Law Hawkes Process (PLHP).
Kernel: phi(t) = eta * (c + t)^(-(1 + delta))
Branching ratio: n* = eta / (delta * c^delta) < 1.0.
"""

from typing import Dict, Any, Optional
import numpy as np
import numba
from scipy.optimize import minimize
from .base import BaseHawkesModel


@numba.jit(nopython=True, fastmath=True)
def _powerlaw_loglikelihood_numba(
    events: np.ndarray,
    T: float,
    mu: float,
    eta: float,
    c: float,
    delta: float
) -> float:
    N = len(events)
    if N == 0:
        return -mu * T

    log_sum = 0.0
    p = 1.0 + delta

    for i in range(N):
        lam = mu
        t_i = events[i]
        for j in range(i):
            lam += eta * ((c + (t_i - events[j])) ** (-p))

        if lam <= 0.0 or np.isnan(lam):
            return -1e15
        log_sum += np.log(lam)

    # Compensator: int_0^T lambda(s) ds
    # Phi(u) = (eta / delta) * (c^(-delta) - (c + u)^(-delta))
    c_neg_delta = c ** (-delta)
    factor = eta / delta
    comp = mu * T

    for i in range(N):
        comp += factor * (c_neg_delta - ((c + (T - events[i])) ** (-delta)))

    return log_sum - comp


class PowerLawHawkesModel(BaseHawkesModel):
    """
    Power-Law Hawkes Process with regularized cutoff at zero.
    """

    def __init__(self):
        super().__init__(name="PowerLawHawkes")
        self.param_names = ["mu", "eta", "c", "delta"]

    def fit(self, events: np.ndarray, T: float, n_starts: int = 4, seed: Optional[int] = None) -> "PowerLawHawkesModel":
        if seed is not None:
            np.random.seed(seed)

        events_arr = np.sort(np.asarray(events, dtype=np.float64))
        events_arr = events_arr[(events_arr >= 0.0) & (events_arr <= T)]
        N = len(events_arr)
        empirical_rate = N / T

        def plhp_nll(theta):
            mu, eta, c, delta = theta
            if mu <= 1e-6 or eta <= 1e-6 or c <= 1e-5 or delta <= 0.01:
                return 1e15
            # Branching ratio
            n_star = eta / (delta * (c ** delta))
            if n_star >= 0.999:
                return 1e15 + 1e6 * (n_star - 0.999) ** 2

            ll = _powerlaw_loglikelihood_numba(events_arr, T, mu, eta, c, delta)
            return -ll

        bounds = [
            (1e-4, 50.0),    # mu
            (1e-4, 20.0),    # eta
            (1e-3, 10.0),    # c
            (0.05, 5.0),     # delta
        ]

        initial_guesses = [
            np.array([empirical_rate * 0.6, 0.3, 0.1, 0.5]),
            np.array([empirical_rate * 0.3, 0.5, 0.05, 0.8]),
            np.array([empirical_rate * 0.7, 0.2, 0.2, 1.2]),
        ]

        for _ in range(max(0, n_starts - len(initial_guesses))):
            init_mu = np.random.uniform(empirical_rate * 0.3, empirical_rate * 0.8)
            init_eta = np.random.uniform(0.1, 0.8)
            init_c = np.random.uniform(0.01, 0.5)
            init_delta = np.random.uniform(0.2, 1.5)
            initial_guesses.append(np.array([init_mu, init_eta, init_c, init_delta]))

        best_opt = None
        best_nll = np.inf

        for init_theta in initial_guesses:
            try:
                res = minimize(
                    plhp_nll,
                    init_theta,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 150}
                )
                if res.fun < best_nll and not np.isnan(res.fun):
                    best_nll = res.fun
                    best_opt = res
            except Exception:
                continue

        if best_opt is None:
            raise RuntimeError("PLHP fitting failed across all initial starts.")

        self.params = best_opt.x
        mu, eta, c, delta = self.params
        self.branching_ratio_ = float(eta / (delta * (c ** delta)))
        self.log_likelihood_ = -best_nll
        k = 4
        self.aic_ = 2 * k - 2 * self.log_likelihood_
        self.bic_ = k * np.log(N) - 2 * self.log_likelihood_
        self.events_ = events_arr
        self.T_ = T
        self.is_fitted = True
        return self

    def predict_intensity(self, t_eval: np.ndarray, events: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if events is None:
            events = self.events_

        mu, eta, c, delta = self.params
        p = 1.0 + delta
        t_arr = np.asarray(t_eval, dtype=np.float64)
        lam = np.full_like(t_arr, mu)

        for i, t in enumerate(t_arr):
            past = events[events < t]
            if len(past) > 0:
                lam[i] += np.sum(eta * ((c + (t - past)) ** (-p)))
        return lam

    def compute_residuals(self, events: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if events is None:
            events = self.events_

        mu, eta, c, delta = self.params
        N = len(events)
        residuals = np.empty(N, dtype=np.float64)
        residuals[0] = mu * events[0]
        factor = eta / delta
        c_neg_delta = c ** (-delta)

        for i in range(1, N):
            t_prev = events[i - 1]
            t_curr = events[i]
            # Baseline increment
            tau = mu * (t_curr - t_prev)
            # Past event kernel contributions integrated over [t_prev, t_curr]
            for j in range(i):
                t_j = events[j]
                # int_{t_prev}^{t_curr} (c + s - t_j)^(-(1+delta)) ds
                # = factor * [ (c + t_prev - t_j)^(-delta) - (c + t_curr - t_j)^(-delta) ]
                u_prev = c + t_prev - t_j
                u_curr = c + t_curr - t_j
                tau += factor * ((u_prev ** (-delta)) - (u_curr ** (-delta)))
            residuals[i] = tau

        return residuals

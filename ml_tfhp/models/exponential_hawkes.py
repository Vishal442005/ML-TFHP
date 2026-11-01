"""
Baseline 1: Standard Exponential Hawkes Process (EHP).
Kernel: phi(t) = alpha * beta * exp(-beta * t)
"""

from typing import Dict, Any, Optional
import numpy as np
from .base import BaseHawkesModel
from ..estimation.optimizer import fit_exponential_hawkes
from ..estimation.loglikelihood import compute_compensator_residuals_numba


class ExponentialHawkesModel(BaseHawkesModel):
    """
    Standard Exponential Hawkes Process model with recursive O(N) evaluation.
    """

    def __init__(self):
        super().__init__(name="ExponentialHawkes")
        self.param_names = ["mu", "alpha", "beta"]

    def fit(self, events: np.ndarray, T: float, n_starts: int = 4, seed: Optional[int] = None) -> "ExponentialHawkesModel":
        events_arr = np.sort(np.asarray(events, dtype=np.float64))
        events_arr = events_arr[(events_arr >= 0.0) & (events_arr <= T)]
        res = fit_exponential_hawkes(events_arr, T, n_starts=n_starts, seed=seed)

        self.params = res["params"]
        self.log_likelihood_ = res["log_likelihood"]
        self.aic_ = res["aic"]
        self.bic_ = res["bic"]
        self.branching_ratio_ = res["branching_ratio"]
        self.events_ = events_arr
        self.T_ = T
        self.is_fitted = True
        return self

    def predict_intensity(self, t_eval: np.ndarray, events: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if events is None:
            events = self.events_
        if events is None or len(events) == 0:
            return np.full_like(t_eval, self.params[0])

        mu, alpha, beta = self.params
        t_arr = np.asarray(t_eval, dtype=np.float64)
        lam = np.full_like(t_arr, mu)

        # Vectorized past event evaluation
        weight = alpha * beta
        for i, t in enumerate(t_arr):
            past = events[events < t]
            if len(past) > 0:
                lam[i] += weight * np.sum(np.exp(-beta * (t - past)))
        return lam

    def compute_residuals(self, events: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if events is None:
            events = self.events_

        mu, alpha, beta = self.params
        weights = np.array([alpha * beta], dtype=np.float64)
        zetas = np.array([beta], dtype=np.float64)
        return compute_compensator_residuals_numba(events, mu, weights, zetas)

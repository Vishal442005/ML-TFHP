"""
Proposed Model: Mittag-Leffler Tempered Fractional Hawkes Process (ML-TFHP).
Kernel: phi(t) = eta * exp(-gamma * t) * t^(alpha - 1) * E_{alpha, alpha}(- (beta * t)^alpha)
Stationarity: n* = eta / (gamma^alpha + beta^alpha) < 1.0.
"""

from typing import Dict, Any, Optional
import numpy as np
from .base import BaseHawkesModel
from ..estimation.optimizer import fit_ml_tfhp
from ..estimation.loglikelihood import (
    compute_compensator_residuals_numba,
    ml_tfhp_loglikelihood,
)
from ..estimation.inference import compute_hessian_and_se
from ..kernels.quadrature import compute_log_quadrature, approximate_kernel
from ..kernels.spectral_density import branching_ratio


class MLTFHPModel(BaseHawkesModel):
    """
    Mittag-Leffler Tempered Fractional Hawkes Process model.
    """

    def __init__(
        self,
        M: int = 10,
        xi_min: float = 1e-4,
        xi_max: float = 1e2
    ):
        super().__init__(name="ML-TFHP")
        self.param_names = ["mu", "alpha", "beta", "gamma", "eta"]
        self.M = M
        self.xi_min = xi_min
        self.xi_max = xi_max
        self.inference_: Optional[Dict[str, Any]] = None
        self.weights_: Optional[np.ndarray] = None
        self.zetas_: Optional[np.ndarray] = None

    def fit(
        self,
        events: np.ndarray,
        T: float,
        n_starts: int = 5,
        compute_inference: bool = True,
        seed: Optional[int] = None
    ) -> "MLTFHPModel":
        """
        Fit ML-TFHP model to event timestamps.
        """
        events_arr = np.sort(np.asarray(events, dtype=np.float64))
        events_arr = events_arr[(events_arr >= 0.0) & (events_arr <= T)]
        res = fit_ml_tfhp(
            events_arr,
            T,
            M=self.M,
            xi_min=self.xi_min,
            xi_max=self.xi_max,
            n_starts=n_starts,
            seed=seed
        )

        self.params = res["params"]
        self.log_likelihood_ = res["log_likelihood"]
        self.aic_ = res["aic"]
        self.bic_ = res["bic"]
        self.branching_ratio_ = res["branching_ratio"]
        self.events_ = events_arr
        self.T_ = T
        self.is_fitted = True

        # Precompute optimal quadrature components
        mu, alpha, beta, gamma_p, eta = self.params
        self.weights_, self.zetas_ = compute_log_quadrature(
            alpha=alpha,
            beta=beta,
            gamma_param=gamma_p,
            eta=eta,
            M=self.M,
            xi_min=self.xi_min,
            xi_max=self.xi_max,
            normalize_branching_ratio=True
        )

        # Compute asymptotic Fisher information and standard errors
        if compute_inference:
            def loglik_target(th):
                return ml_tfhp_loglikelihood(th, events_arr, T, self.M, self.xi_min, self.xi_max)

            self.inference_ = compute_hessian_and_se(
                loglik_target,
                self.params,
                param_names=self.param_names
            )

        return self

    def predict_intensity(self, t_eval: np.ndarray, events: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict conditional intensity lambda(t) at evaluation timestamps.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if events is None:
            events = self.events_

        mu = self.params[0]
        t_arr = np.asarray(t_eval, dtype=np.float64)
        lam = np.full_like(t_arr, mu)

        if events is None or len(events) == 0:
            return lam

        weights, zetas = self.weights_, self.zetas_

        for i, t in enumerate(t_arr):
            past = events[events < t]
            if len(past) > 0:
                lags = t - past
                # sum_m w_m sum_j exp(-zeta_m * lag_j)
                decay = np.exp(-np.outer(lags, zetas))  # shape: (n_past, M)
                lam[i] += float(np.sum(decay @ weights))

        return lam

    def compute_residuals(self, events: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Papangelou inter-event compensator residuals: tau_i ~ Exp(1).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if events is None:
            events = self.events_

        mu = self.params[0]
        return compute_compensator_residuals_numba(events, mu, self.weights_, self.zetas_)

    def get_confidence_intervals(self) -> Dict[str, Dict[str, float]]:
        """
        Return 95% confidence intervals and standard errors for all parameters.
        """
        if not self.is_fitted or self.inference_ is None:
            raise RuntimeError("Inference has not been computed.")

        ci_dict = {}
        for name, val, se, low, high in zip(
            self.param_names,
            self.params,
            self.inference_["se"],
            self.inference_["ci_lower"],
            self.inference_["ci_upper"],
        ):
            ci_dict[name] = {
                "estimate": float(val),
                "std_error": float(se),
                "ci_95_lower": float(low),
                "ci_95_upper": float(high),
            }
        return ci_dict

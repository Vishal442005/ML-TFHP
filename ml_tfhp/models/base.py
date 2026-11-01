"""
Abstract Base Class for Self-Exciting Point Process Models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseHawkesModel(ABC):
    """
    Abstract base class for Hawkes processes.
    """

    def __init__(self, name: str):
        self.name = name
        self.params: Optional[np.ndarray] = None
        self.param_names: list = []
        self.is_fitted: bool = False
        self.log_likelihood_: Optional[float] = None
        self.aic_: Optional[float] = None
        self.bic_: Optional[float] = None
        self.branching_ratio_: Optional[float] = None
        self.events_: Optional[np.ndarray] = None
        self.T_: Optional[float] = None

    @abstractmethod
    def fit(self, events: np.ndarray, T: float, **kwargs) -> "BaseHawkesModel":
        """Fit model parameters to event sequence."""
        pass

    @abstractmethod
    def predict_intensity(self, t_eval: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Evaluate conditional intensity lambda(t) across t_eval given past events."""
        pass

    @abstractmethod
    def compute_residuals(self, events: np.ndarray) -> np.ndarray:
        """Compute Papangelou inter-event compensator residuals tau_i."""
        pass

    def summary(self) -> Dict[str, Any]:
        """Return model estimation summary dictionary."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return {
            "model": self.name,
            "params": dict(zip(self.param_names, self.params)),
            "branching_ratio": self.branching_ratio_,
            "log_likelihood": self.log_likelihood_,
            "aic": self.aic_,
            "bic": self.bic_,
            "n_events": len(self.events_) if self.events_ is not None else 0,
            "T": self.T_,
        }

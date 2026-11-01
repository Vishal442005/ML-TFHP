"""
Analytical Spectral Density for Fractional Mittag-Leffler Kernels.

Implements the continuous spectral measure density rho_alpha(xi) corresponding
to the inverse Laplace transform of the Mittag-Leffler kernel via Bernstein's
representation for completely monotone kernels:
    phi_untempered(t) = eta * int_0^infty exp(-xi * t) * rho_alpha(xi) dxi
"""

from typing import Union
import numpy as np


def spectral_density_ml(
    xi: Union[float, np.ndarray],
    alpha: float,
    beta: float
) -> Union[float, np.ndarray]:
    """
    Evaluate the closed-form spectral density rho_alpha(xi) of the excitation kernel
    phi(t) = t^(alpha - 1) * E_{alpha, alpha}(- (beta * t)^alpha).

    Derived via Stieltjes transform inversion of L{phi}(s) = 1 / (s^alpha + beta^alpha):
        rho_alpha(xi) = (1 / pi) * [beta^alpha * sin(pi * alpha) * xi^alpha] /
                        [xi^(2 * alpha) + 2 * beta^alpha * xi^alpha * cos(pi * alpha) + beta^(2 * alpha)]

    Parameters
    ----------
    xi : float or ndarray
        Continuous relaxation rates (xi > 0).
    alpha : float
        Fractional exponent in (0, 1).
    beta : float
        Scale/decay parameter (> 0).

    Returns
    -------
    rho : float or ndarray
        Spectral density value(s) at xi.
    """
    xi_arr = np.maximum(np.asarray(xi, dtype=np.float64), 1e-16)
    
    beta_a = beta ** alpha
    beta_2a = beta_a ** 2
    sin_pa = np.sin(np.pi * alpha)
    cos_pa = np.cos(np.pi * alpha)
    
    xi_a = xi_arr ** alpha
    xi_2a = xi_a ** 2
    
    # Numerator is beta^alpha * sin(pi * alpha) * xi^alpha
    num = (1.0 / np.pi) * beta_a * sin_pa * xi_a
    denom = xi_2a + 2.0 * beta_a * xi_a * cos_pa + beta_2a
    
    denom = np.maximum(denom, 1e-30)
    rho = num / denom
    
    if np.isscalar(xi):
        return float(rho)
    return rho


def branching_ratio(
    alpha: float,
    beta: float,
    gamma_param: float,
    eta: float
) -> float:
    """
    Compute the closed-form branching ratio n* (L1 norm of the ML-TFHP kernel):
        n* = eta / (gamma^alpha + beta^alpha)

    Stationarity requires n* < 1.0.
    """
    denom = (gamma_param ** alpha) + (beta ** alpha)
    if denom <= 0:
        return np.inf
    return float(eta / denom)


def is_stationary(
    alpha: float,
    beta: float,
    gamma_param: float,
    eta: float,
    tolerance: float = 1e-6
) -> bool:
    """
    Check if the parameter configuration satisfies the stationarity condition n* < 1.
    """
    return branching_ratio(alpha, beta, gamma_param, eta) < (1.0 - tolerance)

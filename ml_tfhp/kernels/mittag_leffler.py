"""
Mittag-Leffler Function and Tempered Mittag-Leffler Kernel Evaluations.

Implements the two-parameter Mittag-Leffler function E_{alpha, beta}(z)
via series summation for small arguments, Gauss-Laguerre integral representation
for intermediate arguments, and asymptotic expansion for large negative arguments.
"""

from typing import Union
import numpy as np
from scipy.special import gamma, gammaln, roots_laguerre


# Precompute standard 24-point Gauss-Laguerre nodes and weights for intermediate evaluations
_GL_NODES, _GL_WEIGHTS = roots_laguerre(24)


def _rho_standard(xi: np.ndarray, alpha: float) -> np.ndarray:
    """Standard spectral density with c = 1."""
    sin_pa = np.sin(np.pi * alpha)
    cos_pa = np.cos(np.pi * alpha)
    xi_a = xi ** alpha
    num = (1.0 / np.pi) * sin_pa * xi_a
    denom = (xi_a ** 2) + 2.0 * xi_a * cos_pa + 1.0
    return num / np.maximum(denom, 1e-30)


def mittag_leffler_scalar(alpha: float, beta: float, z: float, tol: float = 1e-15) -> float:
    """
    Compute E_{alpha, beta}(z) for a scalar real argument z.
    """
    if abs(z) < 1e-15:
        return 1.0 / gamma(beta)
        
    # Classical exponential limits
    if abs(alpha - 1.0) < 1e-12:
        if abs(beta - 1.0) < 1e-12:
            return float(np.exp(z))
        elif abs(beta - 2.0) < 1e-12:
            return float(np.expm1(z) / z)

    # For negative real arguments z = -x (x > 0)
    if z < 0:
        x = -z
        
        # Intermediate/large regime for the symmetric case beta == alpha
        if abs(alpha - beta) < 1e-12:
            if x < 0.05:
                # Taylor series is only safe for very small x where terms don't alternate with large magnitude
                val = 0.0
                for k in range(30):
                    term = ((-x) ** k) / gamma(alpha * k + alpha)
                    val += term
                    if k > 3 and abs(term) < tol * max(abs(val), 1.0):
                        break
                return float(val)
                
            elif x < 35.0:
                # Gauss-Laguerre quadrature of the exact integral:
                # E_{alpha,alpha}(-x) = (1/x) * int_0^infty exp(-u) * rho_std(u * x^(-1/alpha)) du
                x_factor = x ** (-1.0 / alpha)
                u_scaled = _GL_NODES * x_factor
                rhos = _rho_standard(u_scaled, alpha)
                integral = np.sum(_GL_WEIGHTS * rhos)
                return float(integral / x)
                
            else:
                # Asymptotic expansion for x >= 35.0
                # E_{alpha,alpha}(-x) ~ - x^(-2)/Gamma(-alpha) + x^(-3)/Gamma(-2*alpha) - x^(-4)/Gamma(-3*alpha)
                val = 0.0
                for k in range(2, 6):
                    arg = alpha - alpha * k
                    if abs(arg - np.round(arg)) < 1e-12 and arg <= 0:
                        continue
                    term = ((-1.0) ** (k - 1)) * (x ** (-k)) / gamma(arg)
                    val += term
                return float(val)

        else:
            # General beta != alpha for negative z
            if x <= 4.0:
                val = 0.0
                for k in range(50):
                    term = ((-x) ** k) / gamma(alpha * k + beta)
                    val += term
                    if k > 5 and abs(term) < tol * max(abs(val), 1.0):
                        break
                return float(val)
            else:
                # Asymptotic expansion
                val = 0.0
                for k in range(1, 6):
                    arg = beta - alpha * k
                    if abs(arg - np.round(arg)) < 1e-12 and arg <= 0:
                        continue
                    term = ((-1.0) ** (k - 1)) * (x ** (-k)) / gamma(arg)
                    val += term
                return float(val)

    # Positive z: Taylor series with log-gamma
    val = 0.0
    for k in range(150):
        log_term = k * np.log(z + 1e-300) - gammaln(alpha * k + beta)
        if log_term > 700:
            return float(np.inf)
        term = np.exp(log_term)
        val += term
        if k > 5 and abs(term) < tol * val:
            break
            
    return float(val)


def mittag_leffler(alpha: float, beta: float, z: Union[float, np.ndarray], tol: float = 1e-15) -> Union[float, np.ndarray]:
    """
    Vectorized computation of the Mittag-Leffler function E_{alpha, beta}(z).
    """
    if np.isscalar(z):
        return mittag_leffler_scalar(alpha, beta, float(z), tol=tol)
    
    z_arr = np.asarray(z, dtype=np.float64)
    res = np.empty_like(z_arr)
    it = np.nditer([z_arr, res], op_flags=[['readonly'], ['writeonly', 'allocate']])
    for x, y in it:
        y[...] = mittag_leffler_scalar(alpha, beta, float(x), tol=tol)
    return res


def ml_tfhp_kernel_exact(
    t: Union[float, np.ndarray],
    mu: float,
    alpha: float,
    beta: float,
    gamma_param: float,
    eta: float
) -> Union[float, np.ndarray]:
    """
    Evaluate the exact analytical ML-TFHP excitation kernel:
        phi(t) = eta * exp(-gamma * t) * t^(alpha - 1) * E_{alpha, alpha}(- (beta * t)^alpha)
    """
    t_arr = np.maximum(np.asarray(t, dtype=np.float64), 1e-14)
    z = - ((beta * t_arr) ** alpha)
    ml_vals = mittag_leffler(alpha, alpha, z)
    phi = eta * np.exp(-gamma_param * t_arr) * (t_arr ** (alpha - 1.0)) * ml_vals
    if np.isscalar(t):
        return float(phi)
    return phi

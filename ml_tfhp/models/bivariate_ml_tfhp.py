"""
Bivariate Mittag-Leffler Tempered Fractional Hawkes Process (Bivariate ML-TFHP).

Models mutually-exciting high-frequency order book flow:
    Channel 0: Buyer-initiated (aggressive buy / ask hit) trades (+)
    Channel 1: Seller-initiated (aggressive sell / bid lift) trades (-)

Kernel Matrix:
    Phi(t) = [ phi_{ij}(t) ]_{i, j in {0, 1}}
where:
    phi_{ij}(t) = eta_{ij} * e^{-gamma * t} * t^{alpha - 1} * E_{alpha, alpha}(-beta^alpha * t^alpha)
Branching Matrix:
    Gamma_{ij} = eta_{ij} / (gamma^alpha + beta^alpha)
Stability Condition:
    rho(Gamma) < 1 (spectral radius of Gamma strictly less than 1).
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import numba as nb
from scipy.optimize import minimize

from ..kernels.quadrature import compute_log_quadrature
from ..kernels.spectral_density import branching_ratio


@nb.njit(fastmath=True)
def _bivariate_loglik_numba(
    timestamps: np.ndarray,
    types: np.ndarray,
    T: float,
    mu0: float,
    mu1: float,
    w0: np.ndarray,
    zetas: np.ndarray,
    eta00: float,
    eta01: float,
    eta10: float,
    eta11: float
) -> float:
    """
    Numba JIT accelerated recursive log-likelihood for bivariate Hawkes process.
    Time complexity: O(N * M) where N = len(timestamps), M = len(zetas).
    """
    N = len(timestamps)
    M = len(zetas)
    A0 = np.zeros(M, dtype=np.float64)
    A1 = np.zeros(M, dtype=np.float64)

    ll = 0.0
    last_t = 0.0
    n0 = 0
    n1 = 0

    for i in range(N):
        t = timestamps[i]
        dt = t - last_t
        y = types[i]

        for m in range(M):
            e_decay = np.exp(-zetas[m] * dt)
            A0[m] *= e_decay
            A1[m] *= e_decay

        last_t = t

        lam = 0.0
        if y == 0:
            n0 += 1
            lam = mu0
            for m in range(M):
                lam += w0[m] * (eta00 * A0[m] + eta01 * A1[m])
            for m in range(M):
                A0[m] += 1.0
        else:
            n1 += 1
            lam = mu1
            for m in range(M):
                lam += w0[m] * (eta10 * A0[m] + eta11 * A1[m])
            for m in range(M):
                A1[m] += 1.0

        if lam <= 1e-12:
            return -1e15

        ll += np.log(lam)

    # Final interval decay to horizon T
    dt_end = T - last_t
    for m in range(M):
        e_decay = np.exp(-zetas[m] * dt_end)
        A0[m] *= e_decay
        A1[m] *= e_decay

    # Exact closed-form compensators
    comp0 = mu0 * T
    comp1 = mu1 * T
    for m in range(M):
        S0 = n0 - A0[m]
        S1 = n1 - A1[m]
        w_over_zeta = w0[m] / zetas[m]
        comp0 += w_over_zeta * (eta00 * S0 + eta01 * S1)
        comp1 += w_over_zeta * (eta10 * S0 + eta11 * S1)

    return ll - (comp0 + comp1)


@nb.njit(fastmath=True)
def _bivariate_intensities_and_residuals(
    timestamps: np.ndarray,
    types: np.ndarray,
    T: float,
    mu0: float,
    mu1: float,
    w0: np.ndarray,
    zetas: np.ndarray,
    eta00: float,
    eta01: float,
    eta10: float,
    eta11: float
):
    """
    Compute conditional intensities lambda_0(t_k), lambda_1(t_k) and
    time-transformed Papangelou residuals for goodness-of-fit testing.
    """
    N = len(timestamps)
    M = len(zetas)
    A0 = np.zeros(M, dtype=np.float64)
    A1 = np.zeros(M, dtype=np.float64)

    intensities = np.empty(N, dtype=np.float64)
    residuals_0 = []
    residuals_1 = []

    last_t = 0.0
    last_int_comp0 = 0.0
    last_int_comp1 = 0.0
    cum_comp0 = 0.0
    cum_comp1 = 0.0

    for i in range(N):
        t = timestamps[i]
        dt = t - last_t
        y = types[i]

        # Integrate compensator between last_t and t
        for m in range(M):
            e_decay = np.exp(-zetas[m] * dt)
            A0_m_prev = A0[m]
            A1_m_prev = A1[m]
            decay_integral = (1.0 - e_decay) / zetas[m]

            cum_comp0 += (w0[m] * eta00 * A0_m_prev + w0[m] * eta01 * A1_m_prev) * decay_integral
            cum_comp1 += (w0[m] * eta10 * A0_m_prev + w0[m] * eta11 * A1_m_prev) * decay_integral

            A0[m] *= e_decay
            A1[m] *= e_decay

        cum_comp0 += mu0 * dt
        cum_comp1 += mu1 * dt
        last_t = t

        if y == 0:
            lam = mu0
            for m in range(M):
                lam += w0[m] * (eta00 * A0[m] + eta01 * A1[m])
                A0[m] += 1.0
            intensities[i] = lam
            residuals_0.append(cum_comp0 - last_int_comp0)
            last_int_comp0 = cum_comp0
        else:
            lam = mu1
            for m in range(M):
                lam += w0[m] * (eta10 * A0[m] + eta11 * A1[m])
                A1[m] += 1.0
            intensities[i] = lam
            residuals_1.append(cum_comp1 - last_int_comp1)
            last_int_comp1 = cum_comp1

    return intensities, np.array(residuals_0), np.array(residuals_1)


class BivariateMLTFHPModel:
    """
    Bivariate Mittag-Leffler Tempered Fractional Hawkes Process Model.
    """

    def __init__(self, M: int = 10, symmetric: bool = True):
        self.name = "Bivariate-ML-TFHP"
        self.M = M
        self.symmetric = symmetric
        self.params: Optional[np.ndarray] = None
        self.param_names: List[str] = (
            ["mu", "alpha", "beta", "gamma", "eta_self", "eta_cross"]
            if symmetric
            else ["mu0", "mu1", "alpha", "beta", "gamma", "eta00", "eta01", "eta10", "eta11"]
        )
        self.is_fitted: bool = False
        self.log_likelihood_: Optional[float] = None
        self.aic_: Optional[float] = None
        self.bic_: Optional[float] = None
        self.w0_: Optional[np.ndarray] = None
        self.zetas_: Optional[np.ndarray] = None
        self.branching_matrix_: Optional[np.ndarray] = None
        self.spectral_radius_: Optional[float] = None
        self.n_params = len(self.param_names)

    def simulate(
        self,
        mu0: float,
        mu1: float,
        alpha: float,
        beta: float,
        gamma: float,
        eta00: float,
        eta01: float,
        eta10: float,
        eta11: float,
        T: float = 200.0,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate bivariate event times and types via exact Ogata thinning.
        """
        w0, zetas = compute_log_quadrature(alpha, beta, gamma, eta=1.0, M=self.M)
        denom = gamma**alpha + beta**alpha
        Gamma = np.array([[eta00, eta01], [eta10, eta11]]) / denom
        eigvals = np.linalg.eigvals(Gamma)
        spec_rad = float(np.max(np.abs(eigvals)))
        if spec_rad >= 1.0:
            raise ValueError(f"Bivariate system is supercritical: rho(Gamma) = {spec_rad:.4f} >= 1")

        rng = np.random.RandomState(seed)
        A0 = np.zeros(self.M)
        A1 = np.zeros(self.M)
        t = 0.0

        ts = []
        types = []

        while t < T:
            lam0 = mu0 + np.sum(w0 * (eta00 * A0 + eta01 * A1))
            lam1 = mu1 + np.sum(w0 * (eta10 * A0 + eta11 * A1))
            lam_total = lam0 + lam1

            lam_bar = max(lam_total, 1e-4)
            u = rng.uniform(0, 1)
            w = -np.log(u) / lam_bar
            t_cand = t + w
            if t_cand > T:
                break

            dt = t_cand - t
            decay = np.exp(-zetas * dt)
            cand_A0 = A0 * decay
            cand_A1 = A1 * decay
            cand_lam0 = mu0 + np.sum(w0 * (eta00 * cand_A0 + eta01 * cand_A1))
            cand_lam1 = mu1 + np.sum(w0 * (eta10 * cand_A0 + eta11 * cand_A1))
            cand_total = cand_lam0 + cand_lam1

            d = rng.uniform(0, 1)
            if d <= cand_total / lam_bar:
                t = t_cand
                p0 = cand_lam0 / cand_total
                if rng.uniform(0, 1) <= p0:
                    y = 0
                    A0 = cand_A0 + 1.0
                    A1 = cand_A1
                else:
                    y = 1
                    A0 = cand_A0
                    A1 = cand_A1 + 1.0
                ts.append(t)
                types.append(y)
            else:
                t = t_cand
                A0 = cand_A0
                A1 = cand_A1

        return np.array(ts), np.array(types), Gamma

    def fit(
        self,
        timestamps: np.ndarray,
        types: np.ndarray,
        T: Optional[float] = None,
        n_starts: int = 3,
        seed: int = 42
    ) -> "BivariateMLTFHPModel":
        """
        Fit Bivariate ML-TFHP parameters via Maximum Likelihood Estimation.
        """
        ts = np.asarray(timestamps, dtype=np.float64)
        ys = np.asarray(types, dtype=np.int64)

        if T is None:
            T = float(ts[-1]) if len(ts) > 0 else 1.0

        mask = (ts >= 0.0) & (ts <= T)
        ts = ts[mask]
        ys = ys[mask]
        N = len(ts)
        if N < 20:
            raise ValueError("Too few bivariate events to fit model.")

        mean_rate = N / T

        if self.symmetric:
            bounds = [
                (1e-4, mean_rate * 2.0),  # mu
                (0.05, 0.95),             # alpha
                (0.01, 10.0),             # beta
                (1e-4, 5.0),              # gamma
                (1e-4, 5.0),              # eta_self
                (1e-4, 5.0),              # eta_cross
            ]

            def objective(theta):
                mu, alpha, beta, gamma, eta_self, eta_cross = theta
                denom = gamma**alpha + beta**alpha
                if (eta_self + eta_cross) / denom >= 0.999:
                    return 1e12

                try:
                    w0, zetas = compute_log_quadrature(alpha, beta, gamma, eta=1.0, M=self.M)
                    ll = _bivariate_loglik_numba(
                        ts, ys, T,
                        mu, mu,
                        w0, zetas,
                        eta_self, eta_cross, eta_cross, eta_self
                    )
                    return -ll if np.isfinite(ll) else 1e12
                except Exception:
                    return 1e12

            rng = np.random.RandomState(seed)
            starts = [
                np.array([mean_rate * 0.25, 0.60, 1.0, 0.05, 0.3, 0.1]),
                np.array([mean_rate * 0.35, 0.40, 0.5, 0.02, 0.2, 0.05]),
                np.array([mean_rate * 0.20, 0.75, 2.0, 0.10, 0.4, 0.15]),
            ]
            for _ in range(max(0, n_starts - len(starts))):
                rand_s = np.array([
                    rng.uniform(mean_rate * 0.1, mean_rate * 0.5),
                    rng.uniform(0.2, 0.8),
                    rng.uniform(0.1, 3.0),
                    rng.uniform(0.01, 0.5),
                    rng.uniform(0.05, 0.5),
                    rng.uniform(0.01, 0.2),
                ])
                starts.append(rand_s)

        else:
            bounds = [
                (1e-4, mean_rate * 2.0),  # mu0
                (1e-4, mean_rate * 2.0),  # mu1
                (0.05, 0.95),             # alpha
                (0.01, 10.0),             # beta
                (1e-4, 5.0),              # gamma
                (1e-4, 5.0),              # eta00
                (1e-4, 5.0),              # eta01
                (1e-4, 5.0),              # eta10
                (1e-4, 5.0),              # eta11
            ]

            def objective(theta):
                mu0, mu1, alpha, beta, gamma, eta00, eta01, eta10, eta11 = theta
                denom = gamma**alpha + beta**alpha
                Gamma = np.array([[eta00, eta01], [eta10, eta11]]) / denom
                eigvals = np.linalg.eigvals(Gamma)
                if np.max(np.abs(eigvals)) >= 0.999:
                    return 1e12

                try:
                    w0, zetas = compute_log_quadrature(alpha, beta, gamma, eta=1.0, M=self.M)
                    ll = _bivariate_loglik_numba(
                        ts, ys, T,
                        mu0, mu1,
                        w0, zetas,
                        eta00, eta01, eta10, eta11
                    )
                    return -ll if np.isfinite(ll) else 1e12
                except Exception:
                    return 1e12

            rng = np.random.RandomState(seed)
            starts = [
                np.array([mean_rate * 0.25, mean_rate * 0.25, 0.60, 1.0, 0.05, 0.3, 0.1, 0.1, 0.3]),
                np.array([mean_rate * 0.35, mean_rate * 0.35, 0.40, 0.5, 0.02, 0.2, 0.05, 0.05, 0.2]),
            ]

        best_ll = -1e15
        best_x = None

        for x0 in starts[:n_starts]:
            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 150, "ftol": 1e-8}
            )
            if res.success and (-res.fun > best_ll):
                best_ll = -res.fun
                best_x = res.x

        if best_x is None:
            # Fallback to first start
            res = minimize(objective, starts[0], method="L-BFGS-B", bounds=bounds)
            best_ll = -res.fun
            best_x = res.x

        self.params = best_x
        self.log_likelihood_ = float(best_ll)
        k = len(best_x)
        self.aic_ = float(2.0 * k - 2.0 * best_ll)
        self.bic_ = float(k * np.log(N) - 2.0 * best_ll)

        # Store quadrature and branching properties
        if self.symmetric:
            mu, alpha, beta, gamma, eta_self, eta_cross = best_x
            w0, zetas = compute_log_quadrature(alpha, beta, gamma, eta=1.0, M=self.M)
            denom = gamma**alpha + beta**alpha
            n_self = eta_self / denom
            n_cross = eta_cross / denom
            Gamma = np.array([[n_self, n_cross], [n_cross, n_self]])
        else:
            mu0, mu1, alpha, beta, gamma, eta00, eta01, eta10, eta11 = best_x
            w0, zetas = compute_log_quadrature(alpha, beta, gamma, eta=1.0, M=self.M)
            denom = gamma**alpha + beta**alpha
            Gamma = np.array([[eta00, eta01], [eta10, eta11]]) / denom

        self.w0_ = w0
        self.zetas_ = zetas
        self.branching_matrix_ = Gamma
        self.spectral_radius_ = float(np.max(np.abs(np.linalg.eigvals(Gamma))))
        self.is_fitted = True
        return self

    def predict_intensities_and_residuals(
        self,
        timestamps: np.ndarray,
        types: np.ndarray,
        T: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute intensities lambda(t_k) and residuals for buy and sell channels.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing intensities.")

        ts = np.asarray(timestamps, dtype=np.float64)
        ys = np.asarray(types, dtype=np.int64)
        if T is None:
            T = float(ts[-1])

        if self.symmetric:
            mu, alpha, beta, gamma, eta_self, eta_cross = self.params
            return _bivariate_intensities_and_residuals(
                ts, ys, T,
                mu, mu,
                self.w0_, self.zetas_,
                eta_self, eta_cross, eta_cross, eta_self
            )
        else:
            mu0, mu1, alpha, beta, gamma, eta00, eta01, eta10, eta11 = self.params
            return _bivariate_intensities_and_residuals(
                ts, ys, T,
                mu0, mu1,
                self.w0_, self.zetas_,
                eta00, eta01, eta10, eta11
            )

    def get_cross_excitation_ratio(self) -> float:
        """
        Returns ratio of cross-excitation to self-excitation.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if self.symmetric:
            return float(self.params[5] / self.params[4])
        else:
            cross_avg = (self.params[6] + self.params[7]) / 2.0
            self_avg = (self.params[5] + self.params[8]) / 2.0
            return float(cross_avg / self_avg)

"""
Raw SVI (Stochastic Volatility Inspired) slice fit.

Fits one expiry's smile in total-variance space:
    w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
where k = ln(K/F) is log-moneyness and w = sigma_impl^2 * T is total
variance. Fitting in total variance (rather than vol directly) is what
lets `surface.py` interpolate across maturities while staying consistent
with the calendar no-arbitrage condition.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

# Parameter bounds passed to least_squares. `b >= 0` and `|rho| < 1` are
# required for the raw-SVI parameterization to even define a valid slice;
# the rest just keep the optimizer in a numerically sane region.
_LOWER_BOUNDS = np.array([-5.0, 1e-6, -0.999, -3.0, 1e-4])
_UPPER_BOUNDS = np.array([20.0, 20.0, 0.999, 3.0, 20.0])


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self):
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    def total_variance(self, k):
        return raw_svi(k, self.a, self.b, self.rho, self.m, self.sigma)

    def implied_vol(self, k, T):
        w = np.maximum(self.total_variance(k), 0.0)
        return np.sqrt(w / T)


class SVIFitResult:
    def __init__(self, params: SVIParams, k, w, weight, T, opt_result):
        self.params = params
        self.T = T
        model = params.total_variance(k)
        self.residuals = model - w
        self.weighted_rmse = float(np.sqrt(np.average(self.residuals ** 2, weights=weight)))
        self.success = opt_result.success


def raw_svi(k, a, b, rho, m, sigma):
    """The raw SVI total-variance function, vectorized over k."""
    k = np.asarray(k, dtype=float)
    x = k - m
    return a + b * (rho * x + np.sqrt(x ** 2 + sigma ** 2))


@dataclass
class NaturalSVIParams:
    """
    Natural SVI parameter set chi_N = {Delta, mu, rho, omega, zeta}
    (Gatheral & Jacquier 2013, Section 3.2), with omega >= 0, Delta in R,
    mu in R, |rho| < 1 and zeta > 0.
    """
    delta: float
    mu: float
    rho: float
    omega: float
    zeta: float

    def as_array(self):
        return np.array([self.delta, self.mu, self.rho, self.omega, self.zeta])

    def total_variance(self, k):
        return natural_svi(k, self.delta, self.mu, self.rho, self.omega, self.zeta)


def natural_svi(k, delta, mu, rho, omega, zeta):
    """
    Equation (3.2), the natural SVI parameterization of total variance:

        w(k; chi_N) = Delta + (omega/2) * {1 + zeta*rho*(k-mu)
                                           + sqrt((zeta*(k-mu) + rho)^2 + (1-rho^2))}
    """
    k = np.asarray(k, dtype=float)
    x = k - mu
    return delta + (omega / 2.0) * (
        1.0 + zeta * rho * x + np.sqrt((zeta * x + rho) ** 2 + (1.0 - rho ** 2))
    )


def natural_to_raw(params: NaturalSVIParams) -> SVIParams:
    """
    Lemma 3.1, Equation (3.3) -- natural SVI to raw SVI:

        (a, b, rho, m, sigma) = (Delta + (omega/2)(1-rho^2), omega*zeta/2, rho,
                                 mu - rho/zeta, sqrt(1-rho^2)/zeta)
    """
    delta, mu, rho = params.delta, params.mu, params.rho
    omega, zeta = params.omega, params.zeta
    root = np.sqrt(1.0 - rho ** 2)
    return SVIParams(
        a=float(delta + omega / 2.0 * (1.0 - rho ** 2)),
        b=float(omega * zeta / 2.0),
        rho=float(rho),
        m=float(mu - rho / zeta),
        sigma=float(root / zeta),
    )


def raw_to_natural(params: SVIParams) -> NaturalSVIParams:
    """
    Lemma 3.1, Equation (3.4) -- raw SVI to natural SVI:

        (Delta, mu, rho, omega, zeta) = (a - (omega/2)(1-rho^2),
                                         m + rho*sigma/sqrt(1-rho^2), rho,
                                         2*b*sigma/sqrt(1-rho^2),
                                         sqrt(1-rho^2)/sigma)

    (omega appears on the right of the Delta expression, so it is computed
    first.)
    """
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    root = np.sqrt(1.0 - rho ** 2)
    omega = 2.0 * b * sigma / root
    return NaturalSVIParams(
        delta=float(a - omega / 2.0 * (1.0 - rho ** 2)),
        mu=float(m + rho * sigma / root),
        rho=float(rho),
        omega=float(omega),
        zeta=float(root / sigma),
    )


def _initial_guess(k, w):
    """
    Heuristic starting point from the ATM total-variance level and a rough
    linear skew estimate. A live system would warm-start `x0` from the
    previous fit's converged parameters instead (smile parameters move
    little tick to tick) -- `fit_svi_slice`'s `x0` argument is the hook for
    that; this heuristic only runs when no warm start is supplied.
    """
    atm_idx = int(np.argmin(np.abs(k)))
    a0 = max(float(w[atm_idx]) * 0.9, 1e-4)

    if len(k) >= 2 and np.ptp(k) > 1e-8:
        slope, _ = np.polyfit(k, w, 1)
    else:
        slope = 0.0
    b0 = float(np.clip(abs(slope) * 0.5, 0.05, 5.0))
    rho0 = float(np.clip(-np.sign(slope) * 0.3, -0.9, 0.9)) if slope else -0.3
    m0 = 0.0
    sigma0 = float(np.clip(np.std(k) * 0.5, 0.05, 5.0)) if len(k) > 1 else 0.1

    x0 = np.array([a0, b0, rho0, m0, sigma0])
    return np.clip(x0, _LOWER_BOUNDS, _UPPER_BOUNDS)


def fit_svi_slice(k, mid_iv, weight, T, x0=None, validate=True):
    """
    Fit raw SVI parameters to one expiry's (k, mid_iv, weight) data via
    weighted nonlinear least squares (`scipy.optimize.least_squares`).

    If `validate` is True, the fitted slice is checked for butterfly
    (static) arbitrage via `arbitrage.check_butterfly` before being
    returned -- a failing fit raises `arbitrage.ArbitrageError` rather than
    silently handing a bad slice to the pricer. The calendar check needs
    neighboring expiries and so is run separately, across the full surface,
    by `surface.py`.
    """
    k = np.asarray(k, dtype=float)
    mid_iv = np.asarray(mid_iv, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if len(k) < 5:
        raise ValueError("need at least 5 points to fit the 5 raw-SVI parameters")

    w = mid_iv ** 2 * T
    if x0 is None:
        x0 = _initial_guess(k, w)

    sqrt_weight = np.sqrt(weight)

    def residuals(params):
        return sqrt_weight * (raw_svi(k, *params) - w)

    opt_result = least_squares(residuals, x0, bounds=(_LOWER_BOUNDS, _UPPER_BOUNDS))
    params = SVIParams(*opt_result.x)
    result = SVIFitResult(params, k, w, weight, T, opt_result)

    if validate:
        import arbitrage  # deferred: arbitrage.py imports SVIParams from here
        arbitrage.check_butterfly(params)

    return result

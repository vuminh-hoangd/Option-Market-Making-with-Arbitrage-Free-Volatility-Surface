"""
The volatility-arbitrage edge phi_edge(t,s) (paper eq. 10).

This is the economic heart of the model. Ignoring the discount kernel D,
phi_edge is exactly the time-t expected total volatility-arbitrage profit per
unit long position in the option held delta-hedged to expiry:

    E[ int_t^T (sigma^2 - sigma_imp^2)/2 * Gamma^$(u, S_u) du | S_t = s ]

i.e. the gamma-theta carry from being right about volatility. Its sign is the
sign of (sigma^2 - sigma_imp^2): if the market maker thinks realised vol will
beat implied, being long the option is profitable, phi_edge > 0, and
`quoting` turns that into a more aggressive bid.

Under the model's constant-sigma assumption the expectation above collapses
to a one-dimensional integral over time -- the S_u expectation is Gaussian and
done analytically -- so the only numerics here is a single `scipy.integrate.quad`.

# TODO (follow-up): non-constant real-world vol. A stochastic or local-vol
# path sigma(t, S_t) breaks the closed form below: the inner expectation no
# longer integrates out and phi_edge would have to be computed by Monte Carlo
# or by solving the pricing PDE. Deliberately out of scope for this pass.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from . import riccati

if TYPE_CHECKING:
    from .quoting import MMParams

# Absolute/relative tolerances for the edge quadrature. The integrand is
# smooth and bounded on [t, T], so quad converges quickly; these are tight
# enough that the quadrature error is far below any parameter uncertainty.
_QUAD_EPSABS = 1e-10
_QUAD_EPSREL = 1e-10


def _validate(t: float, s: float, K: float, tau: float, T: float, sigma_imp_val: float) -> None:
    if s <= 0.0:
        raise ValueError(f"spot must be positive, got s={s}")
    if K <= 0.0:
        raise ValueError(f"strike must be positive, got K={K}")
    if sigma_imp_val <= 0.0:
        raise ValueError(f"implied vol must be positive, got sigma_imp_val={sigma_imp_val}")
    if t > T:
        raise ValueError(f"t must not exceed the horizon T, got t={t}, T={T}")
    if T >= tau:
        raise ValueError(
            f"the market-making horizon must end strictly before expiry, got T={T}, tau={tau}"
        )
    if t < 0.0:
        raise ValueError(f"t must be non-negative, got t={t}")


def phi_edge(
    t: float,
    s: float,
    K: float,
    tau: float,
    T: float,
    sigma_imp_val: float,
    params: "MMParams",
) -> float:
    """
    phi_edge(t,s), paper eq. 10:

        K (sigma^2 - sigma_imp^2)/2
          * int_t^T D(t,u) / sqrt(sigma_imp^2 (tau-u) + sigma^2 (u-t))
                    * phi_pdf(z(u)) du

        z(u) = (ln(s/K) - sigma^2 (u-t)/2 - sigma_imp^2 (tau-u)/2)
               / sqrt(sigma_imp^2 (tau-u) + sigma^2 (u-t))

    The variance under the square root mixes the two vols by regime: over
    [t, u] the spot diffuses at the market maker's own `sigma`, and from u out
    to expiry the option is still marked at the market's `sigma_imp`.

    `sigma_imp_val` is frozen for the whole integration, matching the model's
    assumption that the surface does not move within [0, T].

    Returns exactly 0.0 at t == T (empty integration interval), and its sign
    always matches the sign of (sigma^2 - sigma_imp^2).
    """
    _validate(t, s, K, tau, T, sigma_imp_val)

    variance_spread = params.sigma ** 2 - sigma_imp_val ** 2
    if t == T or variance_spread == 0.0:
        return 0.0

    log_moneyness = np.log(s / K)
    sigma_sq = params.sigma ** 2
    sigma_imp_sq = sigma_imp_val ** 2

    def integrand(u: float) -> float:
        # Bounded away from zero on [t, T] because tau - u >= tau - T > 0.
        variance = sigma_imp_sq * (tau - u) + sigma_sq * (u - t)
        vol = np.sqrt(variance)
        z = (log_moneyness - 0.5 * sigma_sq * (u - t) - 0.5 * sigma_imp_sq * (tau - u)) / vol
        return riccati.discount_kernel(t, u, T, params) * norm.pdf(z) / vol

    integral, _abserr = quad(integrand, t, T, epsabs=_QUAD_EPSABS, epsrel=_QUAD_EPSREL)
    return float(K * variance_spread / 2.0 * integral)


# Number of Gauss-Legendre nodes used by `phi_edge_batch`. The integrand is
# smooth and bounded on [t, T], so this is far past the point of diminishing
# returns -- it agrees with the adaptive `quad` above to ~1e-12 relative.
_GL_NODES = 64
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GL_NODES)


def phi_edge_batch(
    t: float,
    spots: np.ndarray,
    K: float,
    tau: float,
    T: float,
    sigma_imp_val: float,
    params: "MMParams",
) -> np.ndarray:
    """
    `phi_edge` evaluated at many spots at once, for one time t.

    Same integral, but on a fixed Gauss-Legendre grid instead of an adaptive
    one, which lets the whole spot vector share a single set of nodes: D(t,u),
    the mixed variance and the quadrature weights depend only on u, so only
    z(u) has to be broadcast across spots. `sim` runs one of these per time
    step instead of one adaptive `quad` per (path, step), which is the
    difference between a simulation that finishes and one that does not.

    `phi_edge` remains the reference implementation; this is validated against
    it in the tests.
    """
    spots = np.asarray(spots, dtype=float)
    _validate(t, float(np.min(spots)), K, tau, T, sigma_imp_val)

    variance_spread = params.sigma ** 2 - sigma_imp_val ** 2
    if t == T or variance_spread == 0.0:
        return np.zeros_like(spots)

    sigma_sq = params.sigma ** 2
    sigma_imp_sq = sigma_imp_val ** 2

    # Map the Gauss-Legendre nodes from [-1, 1] onto [t, T].
    half_width = 0.5 * (T - t)
    u = 0.5 * (T + t) + half_width * _GL_X
    weights = half_width * _GL_W

    vol = np.sqrt(sigma_imp_sq * (tau - u) + sigma_sq * (u - t))
    kernel = np.array([riccati.discount_kernel(t, float(ui), T, params) for ui in u])
    offset = -0.5 * sigma_sq * (u - t) - 0.5 * sigma_imp_sq * (tau - u)

    # z has shape (n_spots, n_nodes).
    z = (np.log(spots / K)[:, None] + offset[None, :]) / vol[None, :]
    integral = norm.pdf(z) @ (weights * kernel / vol)
    return K * variance_spread / 2.0 * integral

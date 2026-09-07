"""
psi1(t,s), the coefficient of q in the value-function ansatz (paper eq. 9).

    psi1(t,s) = phi_edge(t,s)
        + 2 e^{-1} (lambda0_b - lambda0_a) int_t^T D(t,u) psi2(u) du
        + 2 e^{-1} (lambda0_b kappa_b - lambda0_a kappa_a) int_t^T D(t,u) psi2(u)^2 du

The first term is the volatility-arbitrage edge. The other two are order-flow
imbalance corrections: they vanish only when the book is symmetric on both
sides, and this module implements them in full -- the general asymmetric case
is the point, so there is no lambda0_b == lambda0_a shortcut anywhere.

Reading the signs: if the market receives more buy interest than sell
interest, the market maker expects to be pushed short over the horizon, and
these terms shade psi1 so that quotes lean against the incoming flow before
the inventory has actually accumulated.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.integrate import quad

from . import riccati
from .edge import phi_edge

if TYPE_CHECKING:
    from .quoting import MMParams

_QUAD_EPSABS = 1e-10
_QUAD_EPSREL = 1e-10


def _discounted_integral(
    t: float, T: float, params: "MMParams", weight: Callable[[float], float]
) -> float:
    """int_t^T D(t,u) * weight(psi2(u)) du, zero on an empty interval."""
    if t >= T:
        return 0.0

    def integrand(u: float) -> float:
        return riccati.discount_kernel(t, u, T, params) * weight(riccati.psi2(u, T, params))

    integral, _abserr = quad(integrand, t, T, epsabs=_QUAD_EPSABS, epsrel=_QUAD_EPSREL)
    return float(integral)


def flow_corrections(t: float, T: float, params: "MMParams") -> float:
    """
    The two order-flow-imbalance terms of eq. 9, i.e. psi1 minus its edge term:

        2 e^{-1} (lambda0_b - lambda0_a)                 int_t^T D(t,u) psi2(u)   du
      + 2 e^{-1} (lambda0_b kappa_b - lambda0_a kappa_a) int_t^T D(t,u) psi2(u)^2 du

    Split out from `psi1` because it depends only on t, never on the spot: a
    simulation sweeping many paths across a shared time grid can evaluate this
    once per time step instead of once per (path, step). Both terms are exactly
    zero for a perfectly symmetric book.
    """
    if t >= T:
        return 0.0

    two_over_e = 2.0 * np.exp(-1.0)

    intensity_imbalance = params.lambda0_b - params.lambda0_a
    elasticity_imbalance = (
        params.lambda0_b * params.kappa_b - params.lambda0_a * params.kappa_a
    )

    linear_term = 0.0
    if intensity_imbalance != 0.0:
        linear_term = two_over_e * intensity_imbalance * _discounted_integral(
            t, T, params, lambda p2: p2
        )

    quadratic_term = 0.0
    if elasticity_imbalance != 0.0:
        quadratic_term = two_over_e * elasticity_imbalance * _discounted_integral(
            t, T, params, lambda p2: p2 ** 2
        )

    return float(linear_term + quadratic_term)


def psi1(
    t: float,
    s: float,
    K: float,
    tau: float,
    T: float,
    sigma_imp_val: float,
    params: "MMParams",
) -> float:
    """
    psi1(t,s) per eq. 9, including both asymmetric-order-flow corrections.

    Equals 0.0 at t == T: the edge term vanishes and both correction integrals
    are over an empty interval, so a market maker at the end of the horizon has
    no forward-looking adjustment left to make.
    """
    edge_term = phi_edge(t, s, K, tau, T, sigma_imp_val, params)
    return float(edge_term + flow_corrections(t, T, params))

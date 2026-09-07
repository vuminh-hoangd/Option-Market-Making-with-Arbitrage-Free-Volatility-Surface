"""
Optimal bid/ask spreads and executable quotes (paper eq. 11).

The optimal spreads decompose into three financially distinct pieces, which is
the whole appeal of the model:

    delta_b* = 1/kappa_b - psi1(t,s) - (2q+1) psi2(t)
    delta_a* = 1/kappa_a + psi1(t,s) + (2q-1) psi2(t)

- `1/kappa`      the elasticity of the option's demand and supply. The more
                 sharply fill rates fall off with the spread (larger kappa),
                 the tighter the market maker can afford to quote.
- `psi1`         the market maker's edge: the expected volatility-arbitrage
                 profit, plus order-flow-imbalance corrections. It enters the
                 two sides with opposite signs, so a positive edge (the market
                 maker wants to be long vol) shifts BOTH quotes up -- a more
                 competitive bid and a less competitive ask.
- `psi2` terms   inventory control. psi2 < 0 and the q-dependence is linear
                 and opposite-signed across the sides, so accumulated length
                 (q > 0) widens the bid and tightens the ask, pushing the book
                 back toward flat.

Note these spreads are not theoretically optimal: they come from the paper's
second-order expansion of the exponential terms in the HJB equation, which is
accurate over a short market-making horizon -- the model's target regime.

# TODO (follow-up): PDE finite-difference validation gate. Solve HJB eq. 6
# numerically on the discrete q-grid and compare against these approximated
# quotes, as an accuracy check before any production use.
# TODO (follow-up): multi-option extension. The vector/matrix analogues
# (Theta2(t), theta1(t), risk-factor matrices A and B, eq. 12-17) are deferred
# until this single-option engine is validated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bs_mm, riccati
from .bs_mm import VolSurfaceLike
from .psi1 import psi1


@dataclass(frozen=True)
class MMParams:
    """
    Everything the quoting rule needs beyond the contract and the surface.

    Every field is an INPUT. Nothing in this package estimates any of them.

    # TODO (follow-up): estimators for sigma, lambda0_b/a and kappa_b/a.
    # sigma comes from the market maker's own volatility forecast; the flow
    # parameters are fit from fill data (see Fernandez-Tapia (2015) for the
    # standard approach). Both are computed externally and passed in here.
    """

    alpha: float      # terminal inventory penalty
    beta: float       # running inventory penalty
    lambda0_b: float  # bid-side base fill intensity, at zero spread
    lambda0_a: float  # ask-side base fill intensity, at zero spread
    kappa_b: float    # bid-side intensity decay
    kappa_a: float    # ask-side intensity decay
    sigma: float      # the market maker's own constant real-world vol estimate

    def __post_init__(self) -> None:
        # beta > 0 (not >= 0) because eta = sqrt(beta/Upsilon) degenerates the
        # closed-form Riccati solution at beta = 0; alpha > 0 for the same
        # reason on the terminal side.
        for name in (
            "alpha", "beta", "lambda0_b", "lambda0_a", "kappa_b", "kappa_a", "sigma",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"MMParams.{name} must be finite and strictly positive, got {value}"
                )


def optimal_spreads(
    t: float,
    s: float,
    q: float,
    K: float,
    tau: float,
    T: float,
    sigma_imp_val: float,
    params: MMParams,
) -> tuple[float, float]:
    """
    (delta_b*, delta_a*) at inventory q, per eq. 11.

    `sigma_imp_val` is passed in rather than looked up so that this whole
    layer stays independent of the surface -- `quote` is the function that
    does the lookup.

    The spreads are unbounded below: a large enough inventory, or a large
    enough edge, drives one side negative, which means the model wants to quote
    through the mark to shed risk or to capture vol. That is intended
    behaviour, so nothing is clamped here; a production caller should impose
    its own floor and position limits on top.
    """
    p1 = psi1(t, s, K, tau, T, sigma_imp_val, params)
    p2 = riccati.psi2(t, T, params)
    return spreads_from_coefficients(p1, p2, q, params)


def spreads_from_coefficients(
    psi1_val: float, psi2_val: float, q: float, params: MMParams
) -> tuple[float, float]:
    """
    Eq. 11 itself, once psi1 and psi2 are known.

    Exposed separately so `sim` can reuse the parts of psi1/psi2 that depend
    only on t across every path on a shared time grid, without restating the
    spread formula anywhere else.
    """
    delta_b_star = 1.0 / params.kappa_b - psi1_val - (2.0 * q + 1.0) * psi2_val
    delta_a_star = 1.0 / params.kappa_a + psi1_val + (2.0 * q - 1.0) * psi2_val
    return float(delta_b_star), float(delta_a_star)


def quote(
    surface: VolSurfaceLike,
    t: float,
    s: float,
    q: float,
    K: float,
    tau: float,
    T: float,
    params: MMParams,
) -> tuple[float, float]:
    """
    (bid, ask) around the mark-to-market option price:

        O   = BS_call(s, K, tau - t, sigma_imp(K, tau))
        bid = O - delta_b*
        ask = O + delta_a*

    `surface` is any calibrated object exposing `implied_vol(K, T)` -- see
    `bs_mm.VolSurfaceLike`. For a simulation loop, prefer
    `quote_at_implied_vol` and hoist the surface query out: sigma_imp is
    assumed constant over [0, T], so re-querying it every step is wasted work.
    """
    sigma_imp_val = bs_mm.sigma_imp(surface, K, tau)
    return quote_at_implied_vol(t, s, q, K, tau, T, sigma_imp_val, params)


def quote_at_implied_vol(
    t: float,
    s: float,
    q: float,
    K: float,
    tau: float,
    T: float,
    sigma_imp_val: float,
    params: MMParams,
) -> tuple[float, float]:
    """`quote` with the surface lookup already done and sigma_imp frozen."""
    option_price = bs_mm.call_price(s, K, tau - t, sigma_imp_val)
    delta_b_star, delta_a_star = optimal_spreads(
        t, s, q, K, tau, T, sigma_imp_val, params
    )
    return option_price - delta_b_star, option_price + delta_a_star

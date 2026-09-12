"""
Classical Avellaneda & Stoikov (2008), "High-frequency trading in a limit
order book", Quantitative Finance 8(3).

The original single-asset inventory model: a dealer quotes bid/ask around a
midprice `s_t` that follows arithmetic Brownian motion (`ds_t = sigma dB_t`,
no drift), with fills arriving as Cox processes at intensity
`lambda(delta) = A * exp(-k*delta)` on each side -- symmetric, unlike this
package's own asymmetric convention elsewhere. The dealer has CARA
(exponential) utility with risk-aversion `gamma`; inventory risk enters
through that utility, not through a quadratic running/terminal penalty the
way `option_market_making.quoting` does.

This is the paper's well-known CLOSED-FORM result (its eq. 25-29, the
small-time-remaining expansion), not the exact HJB solution (which requires
solving a nonlinear ODE system with no closed form). This is the version
almost universally meant by "the Avellaneda-Stoikov model" in practice, and
is offered here on the same terms `option_market_making` offers its own
second-order approximation: clearly flagged as an approximation, not silently
presented as exact.

    r(s,q,t)   = s - q * gamma * sigma**2 * (T-t)                (reservation price)
    delta(t)   = gamma*sigma**2*(T-t) + (2/gamma)*ln(1 + gamma/k)  (total spread)
    bid = r - delta/2,  ask = r + delta/2

Notably, the base intensity `A` does not appear anywhere in `r` or `delta` --
under this solution the quoted spread depends only on how fast fill
intensity decays with distance (`k`), never on the general level of order
flow. That is a real structural difference from `option_market_making`,
where `lambda0_b`/`lambda0_a` enter `psi_1`'s order-flow correction directly.
`A` only ever determines how often the resulting quotes actually get filled.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AvellanedaStoikovParams:
    """
    `gamma`   risk aversion (CARA utility parameter), gamma > 0.
    `k`       symmetric intensity decay, lambda(delta) = A*exp(-k*delta), k > 0.
    `A`       base intensity at zero spread, A > 0. Never enters `quote` --
              kept here only for documentation/fill-rate context and so a
              caller has one place to carry the full parameter set.
    """
    gamma: float
    k: float
    A: float = 1.0

    def __post_init__(self) -> None:
        for name in ("gamma", "k", "A"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"AvellanedaStoikovParams.{name} must be finite and positive, got {value}")


def reservation_price(s: float, q: float, sigma: float, gamma: float, tau: float) -> float:
    """
    r(s,q,t) = s - q*gamma*sigma**2*tau,  tau := T-t.

    The dealer's own indifference price for the asset given current
    inventory `q`: long inventory (q>0) marks the reservation price DOWN
    (the dealer is keener to sell than the mid suggests), short marks it up.
    Vectorized over any of its array arguments.
    """
    return s - q * gamma * sigma ** 2 * tau


def optimal_half_spread(sigma: float, gamma: float, k: float, tau: float) -> float:
    """
    Half of eq. 27's total optimal spread:

        delta/2 = [gamma*sigma**2*tau + (2/gamma)*ln(1+gamma/k)] / 2

    Independent of inventory `q` -- symmetric width around the reservation
    price, unlike `option_market_making`'s eq. 11 where the two sides get
    genuinely different (not just re-centred) treatment.
    """
    return 0.5 * (gamma * sigma ** 2 * tau + (2.0 / gamma) * np.log1p(gamma / k))


def quote(s: float, q: float, sigma: float, gamma: float, k: float, tau: float) -> tuple[float, float]:
    """
    (bid, ask), eq. 25/29: reservation price re-centred by the optimal half-spread.

    `tau` = T-t must be >= 0; at tau=0 the spread collapses to the pure
    liquidity term `(1/gamma)*ln(1+gamma/k)` and the reservation price
    collapses to `s` (no more time for inventory risk to accrue over).
    """
    if tau < 0.0:
        raise ValueError(f"tau (=T-t) must be non-negative, got {tau}")
    r = reservation_price(s, q, sigma, gamma, tau)
    half = optimal_half_spread(sigma, gamma, k, tau)
    return r - half, r + half

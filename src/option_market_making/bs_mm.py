"""
Black-Scholes pricing and Greeks for the single call being quoted, plus the
one bridge from this package to the calibrated volatility surface.

Rates and dividends are zero throughout, per the paper's Section "The
baseline model" ("Dividend and interest rates are assumed to be zero"), so
these are deliberately *not* thin wrappers around `vol_surface.bs` -- that
module carries r/q arguments and a call/put switch this model has no use
for, and the paper's own d2/dollar-gamma parameterization is stated in terms
of (s, K, tau - t, sigma_imp) directly.

Two distinct volatilities live in this model and must never be conflated:
- `sigma_imp`, the market-implied vol, is what marks the book and what all
  Greeks (including the hedge delta) are computed at;
- `sigma`, the market maker's own view (in `MMParams`), never appears here.
  It enters only through the edge term in `edge.py`.
"""
from typing import Protocol

import numpy as np
from scipy.stats import norm


class VolSurfaceLike(Protocol):
    """
    Structural type for anything this package will accept as a surface.

    `vol_surface.global_essvi.GlobalESSVISurface`,
    `vol_surface.surface.SSVIVolSurface` and `vol_surface.surface.VolSurface`
    all satisfy it -- they expose exactly this one method, in year-fractions,
    which is why nothing here needs to know which of the three it was handed.
    """

    def implied_vol(self, K: float, T: float) -> float: ...


def sigma_imp(surface: VolSurfaceLike, K: float, tau: float) -> float:
    """
    Market-implied volatility at (K, tau) from an already-calibrated surface.

    The `surface` argument is explicit rather than a module-level singleton
    so that this package stays free of global state and of any network call:
    the caller calibrates the surface once (`vol_surface.global_essvi.calibrate`
    or `vol_surface.surface.fit_ssvi_surface`) and threads the result through.

    The model assumes sigma_imp is unchanged over the market-making horizon
    [0, T] (the surface is "typically recalibrated less frequently relative to
    the refresh of the limit order quotes"), so callers should evaluate this
    once per contract and hold the value frozen -- see `quoting.quote`.
    """
    if K <= 0.0:
        raise ValueError(f"strike must be positive, got K={K}")
    if tau <= 0.0:
        raise ValueError(f"maturity must be positive, got tau={tau}")

    vol = float(surface.implied_vol(K, tau))
    if not np.isfinite(vol) or vol <= 0.0:
        raise ValueError(
            f"surface returned a non-positive/non-finite implied vol {vol} at "
            f"(K={K}, tau={tau}); the surface is probably being queried outside "
            f"its calibrated strike/maturity range"
        )
    return vol


def _check_tau_minus_t(tau_minus_t: float) -> float:
    """
    Guard the time-to-expiry that every formula below divides by.

    Near expiry d2 and the dollar gamma both blow up, so this raises rather
    than quietly handing back nan/inf. The market-making horizon T must sit
    strictly inside the option's life (T < tau) for the model to make sense
    at all, so hitting this is a caller bug, not a numerical edge case.
    """
    if not np.isfinite(tau_minus_t) or tau_minus_t <= 0.0:
        raise ValueError(
            f"time to expiry must be finite and strictly positive, got "
            f"tau_minus_t={tau_minus_t}; the market-making horizon T must "
            f"satisfy T < tau"
        )
    return float(tau_minus_t)


def _check_inputs(s: float, K: float, tau_minus_t: float, sigma_imp_val: float) -> None:
    if s <= 0.0:
        raise ValueError(f"spot must be positive, got s={s}")
    if K <= 0.0:
        raise ValueError(f"strike must be positive, got K={K}")
    if sigma_imp_val <= 0.0:
        raise ValueError(f"implied vol must be positive, got sigma_imp={sigma_imp_val}")
    _check_tau_minus_t(tau_minus_t)


def d2(s: float, K: float, tau_minus_t: float, sigma_imp_val: float) -> float:
    """
    d2 = (ln(s/K) - sigma_imp^2 (tau-t)/2) / (sigma_imp sqrt(tau-t)).

    The paper's d_2(t, s); with zero rates this is the standard BS d2.
    """
    _check_inputs(s, K, tau_minus_t, sigma_imp_val)
    return (np.log(s / K) - 0.5 * sigma_imp_val ** 2 * tau_minus_t) / (
        sigma_imp_val * np.sqrt(tau_minus_t)
    )


def d1(s: float, K: float, tau_minus_t: float, sigma_imp_val: float) -> float:
    """d1 = d2 + sigma_imp sqrt(tau-t)."""
    return d2(s, K, tau_minus_t, sigma_imp_val) + sigma_imp_val * np.sqrt(tau_minus_t)


def call_price(s: float, K: float, tau_minus_t: float, sigma_imp_val: float) -> float:
    """
    Mark-to-market value of the call, O(t,s) = BS_call(s, K, tau-t, sigma_imp),
    with zero rates and dividends.
    """
    _d1 = d1(s, K, tau_minus_t, sigma_imp_val)
    _d2 = _d1 - sigma_imp_val * np.sqrt(tau_minus_t)
    return float(s * norm.cdf(_d1) - K * norm.cdf(_d2))


def dollar_gamma(s: float, K: float, tau_minus_t: float, sigma_imp_val: float) -> float:
    """
    Dollar gamma, Gamma^$(t,s) = s^2 d_ss O(t,s) = K phi(d2) / (sigma_imp sqrt(tau-t)).

    Economically the change in delta notional per unit change in the
    underlying price; it is the weight the volatility-arbitrage P&L in
    `edge.py` puts on the realised-implied variance spread.
    """
    _d2 = d2(s, K, tau_minus_t, sigma_imp_val)
    return float(K * norm.pdf(_d2) / (sigma_imp_val * np.sqrt(tau_minus_t)))


def delta(s: float, K: float, tau_minus_t: float, sigma_imp_val: float) -> float:
    """
    Hedge delta, d/ds of `call_price` = Phi(d1).

    Evaluated at `sigma_imp`, NOT at the market maker's own `sigma`: the paper
    is explicit that "the computed delta is based on the implied volatility
    sigma_imp". Hedging on the implied delta is what makes the residual P&L
    the gamma-theta carry that `edge.phi_edge` prices.
    """
    return float(norm.cdf(d1(s, K, tau_minus_t, sigma_imp_val)))


# --- vectorized twins -------------------------------------------------------
# The scalar functions above are the reference implementations and mirror the
# paper's notation one-to-one. These take a whole vector of spots at one
# (K, tau-t, sigma_imp), for the simulation loop, where the per-call overhead
# of the scalar versions dominates everything else. Pinned to their scalar
# counterparts in the tests.

def call_price_array(
    spots: np.ndarray, K: float, tau_minus_t: float, sigma_imp_val: float
) -> np.ndarray:
    """`call_price` over a vector of spots."""
    spots = np.asarray(spots, dtype=float)
    _check_inputs(float(np.min(spots)), K, tau_minus_t, sigma_imp_val)
    sqrt_t = np.sqrt(tau_minus_t)
    _d1 = (np.log(spots / K) + 0.5 * sigma_imp_val ** 2 * tau_minus_t) / (sigma_imp_val * sqrt_t)
    return spots * norm.cdf(_d1) - K * norm.cdf(_d1 - sigma_imp_val * sqrt_t)


def delta_array(
    spots: np.ndarray, K: float, tau_minus_t: float, sigma_imp_val: float
) -> np.ndarray:
    """`delta` over a vector of spots."""
    spots = np.asarray(spots, dtype=float)
    _check_inputs(float(np.min(spots)), K, tau_minus_t, sigma_imp_val)
    sqrt_t = np.sqrt(tau_minus_t)
    _d1 = (np.log(spots / K) + 0.5 * sigma_imp_val ** 2 * tau_minus_t) / (sigma_imp_val * sqrt_t)
    return norm.cdf(_d1)


def dollar_gamma_array(
    spots: np.ndarray, K: float, tau_minus_t: float, sigma_imp_val: float
) -> np.ndarray:
    """`dollar_gamma` over a vector of spots."""
    spots = np.asarray(spots, dtype=float)
    _check_inputs(float(np.min(spots)), K, tau_minus_t, sigma_imp_val)
    sqrt_t = np.sqrt(tau_minus_t)
    _d2 = (np.log(spots / K) - 0.5 * sigma_imp_val ** 2 * tau_minus_t) / (sigma_imp_val * sqrt_t)
    return K * norm.pdf(_d2) / (sigma_imp_val * sqrt_t)

"""
Implied volatility solver.

Primary method is Newton-Raphson using the closed-form vega from `bs.py` as
the derivative (fast, few iterations near the solution). Falls back to
Brent's method (bracketed, guaranteed to converge if a root exists in the
bracket) whenever vega is too small to trust the Newton step, the step
overshoots into an invalid region, or Newton fails to converge within the
iteration budget.

Before solving, in-the-money quotes are converted to their out-of-the-money
equivalent via put-call parity (same implied vol, since parity is exact
under Black-Scholes). This is standard desk practice: an ITM price is mostly
intrinsic value, so the time-value signal that actually carries volatility
information gets lost to floating-point cancellation; solving on the OTM
side keeps that signal at full precision.
"""
import numpy as np
from scipy.optimize import brentq

import bs

SIGMA_LO = 1e-6
SIGMA_HI = 10.0
VEGA_FLOOR = 1e-10


def _initial_guess(price, S, K, T, r, q):
    """Brenner-Subrahmanyam style ATM approximation, clipped to a sane range."""
    with np.errstate(divide="ignore", invalid="ignore"):
        guess = np.sqrt(2 * np.pi / T) * price / S
    return np.clip(np.nan_to_num(guess, nan=0.5), 0.05, 3.0)


def _to_otm(price, S, K, T, r, q, option_type):
    """Return (price, option_type) for the equivalent out-of-the-money side."""
    is_call = str(option_type).strip().lower() in ("c", "call")
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    forward = S * np.exp((r - q) * T)
    if is_call and K < forward:
        return price - disc_S + disc_K, "P"
    if (not is_call) and K > forward:
        return price + disc_S - disc_K, "C"
    return price, "C" if is_call else "P"


def implied_vol(price, S, K, T, r, q, option_type, tol=1e-8, max_iter=100):
    """
    Solve for the implied volatility of a single option.

    Newton-Raphson first; falls back to `scipy.optimize.brentq` on
    [SIGMA_LO, SIGMA_HI] if Newton doesn't cleanly converge. Returns NaN if
    no root exists in that bracket (e.g. price outside no-arbitrage bounds).
    """
    price, S, K, T, r, q = (float(x) for x in (price, S, K, T, r, q))
    price, option_type = _to_otm(price, S, K, T, r, q, option_type)
    sigma = float(_initial_guess(price, S, K, T, r, q))

    for _ in range(max_iter):
        px = float(bs.price(S, K, T, r, q, sigma, option_type))
        diff = px - price
        # Tolerance scales with the *quote's own* magnitude, floored tiny
        # (not at 1.0): for a near-worthless OTM quote, flooring at 1.0
        # would call a wildly wrong sigma "converged" just because the
        # absolute price gap also happens to be tiny relative to 1.0.
        if abs(diff) < tol * max(abs(price), 1e-12):
            return sigma
        v = float(bs.vega(S, K, T, r, q, sigma))
        if v < VEGA_FLOOR:
            break
        sigma = sigma - diff / v
        if not (SIGMA_LO < sigma < SIGMA_HI):
            break

    def objective(s):
        return float(bs.price(S, K, T, r, q, s, option_type)) - price

    try:
        return brentq(objective, SIGMA_LO, SIGMA_HI, xtol=tol, maxiter=200)
    except ValueError:
        return np.nan


def implied_vol_batch(prices, S, K, T, r, q, option_type, tol=1e-8, max_iter=100):
    """
    Vectorized implied-vol solve for an entire chain at once.

    Runs Newton-Raphson simultaneously across all elements, then falls back
    to the scalar `implied_vol` (Newton->Brent, with the same OTM-parity
    conversion) element-by-element for any entries that didn't converge —
    keeping the batch result consistent with calling the scalar solver in a
    loop.
    """
    prices, S, K, T, r, q = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (prices, S, K, T, r, q))
    )
    option_type = np.broadcast_to(np.asarray(option_type, dtype=object), prices.shape)

    otm_price = np.empty_like(prices)
    otm_type = np.empty(prices.shape, dtype=object)
    for idx in np.ndindex(prices.shape):
        otm_price[idx], otm_type[idx] = _to_otm(
            prices[idx], S[idx], K[idx], T[idx], r[idx], q[idx], option_type[idx]
        )

    sigma = _initial_guess(otm_price, S, K, T, r, q)
    converged = np.zeros(prices.shape, dtype=bool)

    for _ in range(max_iter):
        px = bs.price(S, K, T, r, q, sigma, otm_type)
        diff = px - otm_price
        converged |= np.abs(diff) < tol * np.maximum(np.abs(otm_price), 1e-12)
        if converged.all():
            break
        v = bs.vega(S, K, T, r, q, sigma)
        active = ~converged & (v > VEGA_FLOOR)
        # np.where evaluates both branches eagerly, so elements with a
        # subnormal (but positive) vega can overflow the division even
        # though `active` will discard the result for them right after.
        with np.errstate(over="ignore", invalid="ignore"):
            step = np.where(active, diff / np.where(v > 0, v, 1.0), 0.0)
        sigma = np.where(active, sigma - step, sigma)
        sigma = np.clip(sigma, SIGMA_LO, SIGMA_HI)

    out = sigma.copy()
    for idx in np.argwhere(~converged):
        idx = tuple(idx)
        out[idx] = implied_vol(
            prices[idx], S[idx], K[idx], T[idx], r[idx], q[idx], option_type[idx],
            tol=tol, max_iter=max_iter,
        )
    return out

"""
Vectorized Black-Scholes pricer and Greeks.

Every function accepts scalars or numpy arrays (broadcast together), so a
full chain of strikes/expiries can be priced in one call. `option_type` is
'C'/'call' or 'P'/'put' (case-insensitive), scalar or array-like.

Conventions:
- S: spot, K: strike, T: time to expiry in years, r: risk-free rate,
  q: continuous dividend / funding yield, sigma: volatility (annualized).
- vega is dPrice/dSigma for one full unit of vol (i.e. divide by 100 to get
  the "per vol point" convention some desks quote).
"""
import numpy as np
from scipy.stats import norm


def _is_call_mask(option_type):
    """Boolean array: True where option_type denotes a call."""
    ot = np.asarray(option_type, dtype=object)
    flat = np.array([str(x).strip().lower() in ("c", "call") for x in ot.ravel()])
    return flat.reshape(np.shape(ot)) if np.ndim(ot) else bool(flat[0])


def _d1_d2(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def price(S, K, T, r, q, sigma, option_type):
    """Black-Scholes price for European call/put, vectorized."""
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    is_call = _is_call_mask(option_type)

    call_px = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_px = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return np.where(is_call, call_px, put_px)


def delta(S, K, T, r, q, sigma, option_type):
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    S, T, q = (np.asarray(x, dtype=float) for x in (S, T, q))
    is_call = _is_call_mask(option_type)
    call_delta = np.exp(-q * T) * norm.cdf(d1)
    put_delta = -np.exp(-q * T) * norm.cdf(-d1)
    return np.where(is_call, call_delta, put_delta)


def gamma(S, K, T, r, q, sigma, option_type=None):
    """Gamma is identical for calls and puts; option_type accepted for a uniform signature."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    S, T, q, sigma = (np.asarray(x, dtype=float) for x in (S, T, q, sigma))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, q, sigma, option_type=None):
    """Vega is identical for calls and puts; option_type accepted for a uniform signature."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    S, T, q = (np.asarray(x, dtype=float) for x in (S, T, q))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, q, sigma, option_type):
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    is_call = _is_call_mask(option_type)

    common = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    call_theta = common - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
    put_theta = common + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)
    return np.where(is_call, call_theta, put_theta)

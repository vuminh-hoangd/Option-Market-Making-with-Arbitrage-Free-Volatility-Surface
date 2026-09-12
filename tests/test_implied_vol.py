import numpy as np
import pytest

import bs
import implied_vol as iv


def _grid():
    """Deep ITM/OTM, short/long maturity grid used for round-trip tests.

    Crypto-realistic vols (30%-150%, per the Deribit use case in Step 3)
    combined with strikes out to 40%/250% of spot and maturities from 1 day
    to 2 years -- deep enough to stress the ITM-cancellation and near-zero
    vega paths without landing in the "zero optionality" regime (extreme
    OTM + short-dated + low-vol) where the true price underflows to a value
    no double-precision solver could recover a vol from.
    """
    S = 100.0
    strikes = [40.0, 70.0, 90.0, 100.0, 110.0, 140.0, 250.0]
    maturities = [1 / 365, 30 / 365, 0.25, 1.0, 2.0]
    sigmas = [0.3, 0.6, 1.2]
    r, q = 0.03, 0.01
    for K in strikes:
        for T in maturities:
            for sigma in sigmas:
                for otype in ("C", "P"):
                    yield S, K, T, r, q, sigma, otype


@pytest.mark.parametrize("S,K,T,r,q,sigma,otype", list(_grid()))
def test_round_trip_recovers_sigma(S, K, T, r, q, sigma, otype):
    price = bs.price(S, K, T, r, q, sigma, otype)
    # Skip combinations with no recoverable vol information: near-worthless
    # quotes, or vega so far underflowed that the price is insensitive to
    # sigma at double precision (genuine "zero optionality" cases, not a
    # solver failure).
    if price < 1e-8:
        pytest.skip("price too small to carry implied-vol information")
    if bs.vega(S, K, T, r, q, sigma) < 1e-6:
        pytest.skip("vega too small at double precision to recover sigma to 1e-6")
    recovered = iv.implied_vol(price, S, K, T, r, q, otype)
    assert recovered == pytest.approx(sigma, abs=1e-6)


def test_batch_matches_scalar_loop():
    rng = np.random.default_rng(0)
    n = 40
    S = np.full(n, 100.0)
    K = rng.uniform(60, 160, n)
    T = rng.uniform(7 / 365, 2.0, n)
    r = np.full(n, 0.03)
    q = np.full(n, 0.01)
    sigma_true = rng.uniform(0.1, 1.2, n)
    otype = np.array(["C" if i % 2 == 0 else "P" for i in range(n)])

    prices = bs.price(S, K, T, r, q, sigma_true, otype)

    batch_result = iv.implied_vol_batch(prices, S, K, T, r, q, otype)
    scalar_result = np.array([
        iv.implied_vol(prices[i], S[i], K[i], T[i], r[i], q[i], otype[i])
        for i in range(n)
    ])

    assert batch_result == pytest.approx(scalar_result, abs=1e-6)
    assert batch_result == pytest.approx(sigma_true, abs=1e-6)


def test_fallback_triggers_for_near_zero_vega():
    # Deep OTM, short-dated -> vega is small enough that Newton alone would
    # stall (but not so small that the price underflows past recoverability).
    S, K, T, r, q, sigma, otype = 100.0, 180.0, 7 / 365, 0.03, 0.0, 0.6, "C"
    price = bs.price(S, K, T, r, q, sigma, otype)
    assert 0 < price < 0.5
    assert bs.vega(S, K, T, r, q, sigma) < iv.VEGA_FLOOR * 1e4  # confirms this stresses the fallback path
    recovered = iv.implied_vol(price, S, K, T, r, q, otype)
    assert recovered == pytest.approx(sigma, abs=1e-4)


def test_invalid_price_returns_nan():
    # Price above the max possible (undiscounted forward) call value has no valid vol.
    S, K, T, r, q, otype = 100.0, 50.0, 1.0, 0.0, 0.0, "C"
    impossible_price = S + 100
    assert np.isnan(iv.implied_vol(impossible_price, S, K, T, r, q, otype))

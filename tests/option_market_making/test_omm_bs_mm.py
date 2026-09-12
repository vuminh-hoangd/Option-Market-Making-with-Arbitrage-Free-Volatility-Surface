import numpy as np
import pytest

import bs
from option_market_making import bs_mm


def test_sigma_imp_matches_the_frozen_snapshot(surface, contract):
    vol = bs_mm.sigma_imp(surface, contract["K"], contract["tau"])
    assert vol == pytest.approx(contract["sigma_imp"], rel=1e-12)


def test_sigma_imp_rejects_degenerate_contracts(surface, contract):
    with pytest.raises(ValueError):
        bs_mm.sigma_imp(surface, -1.0, contract["tau"])
    with pytest.raises(ValueError):
        bs_mm.sigma_imp(surface, contract["K"], 0.0)


def test_call_price_lies_between_intrinsic_and_spot(contract):
    s, K, tau, vol = contract["S0"], contract["K"], contract["tau"], contract["sigma_imp"]
    for moneyness in (0.7, 0.9, 1.0, 1.1, 1.4):
        spot = s * moneyness
        price = bs_mm.call_price(spot, K, tau, vol)
        assert max(spot - K, 0.0) <= price <= spot


def test_dollar_gamma_is_positive(contract):
    s, K, tau, vol = contract["S0"], contract["K"], contract["tau"], contract["sigma_imp"]
    for moneyness in (0.8, 1.0, 1.2):
        assert bs_mm.dollar_gamma(s * moneyness, K, tau, vol) > 0.0


def test_delta_is_a_probability_and_increases_with_spot(contract):
    K, tau, vol = contract["K"], contract["tau"], contract["sigma_imp"]
    spots = contract["S0"] * np.array([0.7, 0.9, 1.0, 1.1, 1.4])
    deltas = [bs_mm.delta(float(s), K, tau, vol) for s in spots]
    assert all(0.0 <= d <= 1.0 for d in deltas)
    assert np.all(np.diff(deltas) > 0.0)


def test_matches_the_existing_zero_rate_black_scholes_pricer(contract):
    """
    `bs_mm` reimplements BS with r = q = 0 rather than wrapping `vol_surface.bs`;
    this pins the two together so the duplication cannot silently drift.
    """
    s, K, tau, vol = contract["S0"], contract["K"], contract["tau"], contract["sigma_imp"]
    assert bs_mm.call_price(s, K, tau, vol) == pytest.approx(
        float(bs.price(s, K, tau, 0.0, 0.0, vol, "C"))
    )
    assert bs_mm.delta(s, K, tau, vol) == pytest.approx(
        float(bs.delta(s, K, tau, 0.0, 0.0, vol, "C"))
    )


def test_dollar_gamma_equals_s_squared_times_gamma(contract):
    """Gamma^$ = s^2 d_ss O, so it must agree with the standard gamma scaled by s^2."""
    s, K, tau, vol = contract["S0"], contract["K"], contract["tau"], contract["sigma_imp"]
    expected = s ** 2 * float(bs.gamma(s, K, tau, 0.0, 0.0, vol))
    assert bs_mm.dollar_gamma(s, K, tau, vol) == pytest.approx(expected, rel=1e-10)


def test_near_expiry_raises_rather_than_returning_nan(contract):
    s, K, vol = contract["S0"], contract["K"], contract["sigma_imp"]
    for bad_tau in (0.0, -1e-8):
        with pytest.raises(ValueError, match="time to expiry"):
            bs_mm.call_price(s, K, bad_tau, vol)
        with pytest.raises(ValueError, match="time to expiry"):
            bs_mm.dollar_gamma(s, K, bad_tau, vol)
        with pytest.raises(ValueError, match="time to expiry"):
            bs_mm.delta(s, K, bad_tau, vol)


def test_vectorized_twins_agree_with_the_scalar_references(contract):
    K, tau, vol = contract["K"], contract["tau"], contract["sigma_imp"]
    spots = contract["S0"] * np.linspace(0.7, 1.4, 25)

    assert bs_mm.call_price_array(spots, K, tau, vol) == pytest.approx(
        [bs_mm.call_price(float(s), K, tau, vol) for s in spots], rel=1e-12
    )
    assert bs_mm.delta_array(spots, K, tau, vol) == pytest.approx(
        [bs_mm.delta(float(s), K, tau, vol) for s in spots], rel=1e-12
    )


def test_vectorized_twins_guard_expiry_too(contract):
    spots = np.array([contract["S0"]])
    with pytest.raises(ValueError, match="time to expiry"):
        bs_mm.call_price_array(spots, contract["K"], 0.0, contract["sigma_imp"])
    with pytest.raises(ValueError, match="time to expiry"):
        bs_mm.delta_array(spots, contract["K"], 0.0, contract["sigma_imp"])

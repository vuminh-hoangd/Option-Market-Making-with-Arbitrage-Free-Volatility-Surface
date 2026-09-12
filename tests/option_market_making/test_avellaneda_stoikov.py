import numpy as np
import pytest

from option_market_making import avellaneda_stoikov as av


def test_params_reject_non_positive():
    with pytest.raises(ValueError):
        av.AvellanedaStoikovParams(gamma=0.0, k=1.0)
    with pytest.raises(ValueError):
        av.AvellanedaStoikovParams(gamma=1.0, k=-1.0)
    with pytest.raises(ValueError):
        av.AvellanedaStoikovParams(gamma=1.0, k=1.0, A=0.0)


def test_reservation_price_is_mid_at_zero_inventory():
    assert av.reservation_price(s=100.0, q=0.0, sigma=0.4, gamma=0.1, tau=0.5) == pytest.approx(100.0)


def test_reservation_price_moves_against_inventory():
    """Long inventory marks the price down; short marks it up."""
    long_r = av.reservation_price(s=100.0, q=5.0, sigma=0.4, gamma=0.1, tau=0.5)
    short_r = av.reservation_price(s=100.0, q=-5.0, sigma=0.4, gamma=0.1, tau=0.5)
    assert long_r < 100.0 < short_r
    assert long_r == pytest.approx(200.0 - short_r)  # symmetric about the mid


def test_reservation_price_linear_in_inventory():
    s, sigma, gamma, tau = 100.0, 0.4, 0.1, 0.5
    q = np.array([-3.0, -1.0, 0.0, 2.0, 4.0])
    r = av.reservation_price(s, q, sigma, gamma, tau)
    slope = np.diff(r) / np.diff(q)
    assert np.allclose(slope, -gamma * sigma ** 2 * tau)


def test_half_spread_is_positive():
    assert av.optimal_half_spread(sigma=0.4, gamma=0.1, k=1.5, tau=0.5) > 0.0


def test_half_spread_at_zero_tau_is_pure_liquidity_term():
    """No time left for inventory risk to accrue -- only the constant term survives."""
    gamma, k = 0.1, 1.5
    half = av.optimal_half_spread(sigma=0.4, gamma=gamma, k=k, tau=0.0)
    assert half == pytest.approx((1.0 / gamma) * np.log1p(gamma / k))


def test_half_spread_grows_with_remaining_time():
    half_short = av.optimal_half_spread(sigma=0.4, gamma=0.1, k=1.5, tau=0.1)
    half_long = av.optimal_half_spread(sigma=0.4, gamma=0.1, k=1.5, tau=1.0)
    assert half_long > half_short


def test_half_spread_shrinks_as_k_grows():
    """Faster-decaying intensity (a more competitive book) -> tighter optimal quote."""
    half_slow_decay = av.optimal_half_spread(sigma=0.4, gamma=0.1, k=0.5, tau=0.5)
    half_fast_decay = av.optimal_half_spread(sigma=0.4, gamma=0.1, k=5.0, tau=0.5)
    assert half_fast_decay < half_slow_decay


def test_quote_straddles_the_reservation_price():
    bid, ask = av.quote(s=100.0, q=2.0, sigma=0.4, gamma=0.1, k=1.5, tau=0.5)
    r = av.reservation_price(s=100.0, q=2.0, sigma=0.4, gamma=0.1, tau=0.5)
    assert bid < r < ask
    assert ask - r == pytest.approx(r - bid)  # symmetric around r


def test_quote_matches_the_formula_directly():
    s, q, sigma, gamma, k, tau = 100.0, 3.0, 0.4, 0.1, 1.5, 0.5
    bid, ask = av.quote(s, q, sigma, gamma, k, tau)
    r = s - q * gamma * sigma ** 2 * tau
    half = 0.5 * (gamma * sigma ** 2 * tau + (2.0 / gamma) * np.log1p(gamma / k))
    assert bid == pytest.approx(r - half)
    assert ask == pytest.approx(r + half)


def test_quote_rejects_negative_tau():
    with pytest.raises(ValueError):
        av.quote(s=100.0, q=0.0, sigma=0.4, gamma=0.1, k=1.5, tau=-0.01)


def test_quote_collapses_to_pure_liquidity_quote_at_tau_zero():
    """At t=T there's no more time for inventory risk: r=s, spread is the constant term only."""
    bid, ask = av.quote(s=100.0, q=5.0, sigma=0.4, gamma=0.1, k=1.5, tau=0.0)
    half = (1.0 / 0.1) * np.log1p(0.1 / 1.5)
    assert bid == pytest.approx(100.0 - half)
    assert ask == pytest.approx(100.0 + half)

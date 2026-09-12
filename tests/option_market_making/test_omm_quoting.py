from dataclasses import replace

import numpy as np
import pytest

from option_market_making import bs_mm, riccati
from option_market_making.psi1 import psi1
from option_market_making.quoting import optimal_spreads, quote, quote_at_implied_vol


def test_mmparams_rejects_non_positive_fields(params):
    for field in ("alpha", "beta", "lambda0_b", "lambda0_a", "kappa_b", "kappa_a", "sigma"):
        with pytest.raises(ValueError, match=field):
            replace(params, **{field: 0.0})
    with pytest.raises(ValueError):
        replace(params, sigma=float("nan"))


def test_quotes_straddle_the_mark(surface, params, horizon, contract):
    s, K, tau = contract["S0"], contract["K"], contract["tau"]
    for t in np.linspace(0.0, horizon, 5):
        bid, ask = quote(surface, float(t), s, 0.0, K, tau, horizon, params)
        mark = bs_mm.call_price(s, K, tau - t, contract["sigma_imp"])
        assert bid < mark < ask


def test_spreads_are_positive_for_reasonable_parameters(surface, params, horizon, contract):
    s, K, tau = contract["S0"], contract["K"], contract["tau"]
    for t in np.linspace(0.0, horizon, 5):
        for q in (-5.0, -1.0, 0.0, 1.0, 5.0):
            db, da = optimal_spreads(
                float(t), s, q, K, tau, horizon, contract["sigma_imp"], params
            )
            assert db > 0.0
            assert da > 0.0


def test_quote_agrees_with_the_frozen_implied_vol_path(surface, params, horizon, contract):
    s, K, tau = contract["S0"], contract["K"], contract["tau"]
    live = quote(surface, 0.0, s, 2.0, K, tau, horizon, params)
    frozen = quote_at_implied_vol(0.0, s, 2.0, K, tau, horizon, contract["sigma_imp"], params)
    assert live == pytest.approx(frozen)


def test_long_inventory_skews_the_book_lower(surface, params, horizon, contract):
    """
    psi2 < 0, so length must widen the bid and tighten the ask -- both quotes
    move down, pushing the inventory back toward flat.
    """
    s, K, tau = contract["S0"], contract["K"], contract["tau"]
    flat_bid, flat_ask = quote(surface, 0.0, s, 0.0, K, tau, horizon, params)
    long_bid, long_ask = quote(surface, 0.0, s, 5.0, K, tau, horizon, params)
    short_bid, short_ask = quote(surface, 0.0, s, -5.0, K, tau, horizon, params)

    assert long_bid < flat_bid < short_bid
    assert long_ask < flat_ask < short_ask


def test_inventory_skew_is_linear_and_symmetric(params, horizon, contract):
    """Both spreads are affine in q with slope -/+ 2 psi2 -- eq. 11 exactly."""
    s, K, tau, vol = contract["S0"], contract["K"], contract["tau"], contract["sigma_imp"]
    slope = 2.0 * riccati.psi2(0.0, horizon, params)

    db0, da0 = optimal_spreads(0.0, s, 0.0, K, tau, horizon, vol, params)
    db1, da1 = optimal_spreads(0.0, s, 1.0, K, tau, horizon, vol, params)
    assert db1 - db0 == pytest.approx(-slope)
    assert da1 - da0 == pytest.approx(slope)


def test_a_bigger_vol_edge_lifts_both_quotes(surface, params, horizon, contract):
    """
    A market maker who expects realised to beat implied wants to be long the
    option, so pays up: the bid rises AND the ask rises.
    """
    s, K, tau = contract["S0"], contract["K"], contract["tau"]
    modest = quote(surface, 0.0, s, 0.0, K, tau, horizon,
                   replace(params, sigma=contract["sigma_imp"] + 0.01))
    strong = quote(surface, 0.0, s, 0.0, K, tau, horizon,
                   replace(params, sigma=contract["sigma_imp"] + 0.15))
    assert strong[0] > modest[0]
    assert strong[1] > modest[1]


def test_spreads_reproduce_equation_11(params, horizon, contract):
    s, K, tau, vol = contract["S0"], contract["K"], contract["tau"], contract["sigma_imp"]
    q = 3.0
    p1 = psi1(0.0, s, K, tau, horizon, vol, params)
    p2 = riccati.psi2(0.0, horizon, params)

    db, da = optimal_spreads(0.0, s, q, K, tau, horizon, vol, params)
    assert db == pytest.approx(1.0 / params.kappa_b - p1 - (2 * q + 1) * p2)
    assert da == pytest.approx(1.0 / params.kappa_a + p1 + (2 * q - 1) * p2)


def test_at_the_horizon_the_spread_is_elasticity_plus_terminal_penalty(params, horizon, contract):
    """
    psi1(T) = 0 and psi2(T) = -alpha, so eq. 11 collapses to a pure
    liquidity-plus-terminal-penalty quote with no edge left in it.
    """
    s, K, tau, vol = contract["S0"], contract["K"], contract["tau"], contract["sigma_imp"]
    q = 2.0
    db, da = optimal_spreads(horizon, s, q, K, tau, horizon, vol, params)
    assert db == pytest.approx(1.0 / params.kappa_b + (2 * q + 1) * params.alpha)
    assert da == pytest.approx(1.0 / params.kappa_a - (2 * q - 1) * params.alpha)

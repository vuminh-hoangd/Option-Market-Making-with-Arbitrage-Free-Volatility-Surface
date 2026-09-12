import numpy as np
import pytest

import bs
from market_making import quoting


def test_linear_intensity_rejects_nonpositive_params():
    with pytest.raises(ValueError):
        quoting.LinearIntensity(C=0.0, D=1.0)
    with pytest.raises(ValueError):
        quoting.LinearIntensity(C=1.0, D=-1.0)


def test_linear_intensity_rate_shape():
    intensity = quoting.LinearIntensity(C=40.0, D=200.0)
    assert intensity.max_premium == pytest.approx(0.2)
    assert intensity.rate(0.0) == pytest.approx(40.0)
    assert intensity.rate(0.1) == pytest.approx(20.0)
    # zero beyond the cutoff, not negative
    assert intensity.rate(0.5) == 0.0


def test_optimal_premium_closed_form():
    intensity = quoting.LinearIntensity(C=40.0, D=200.0)
    eps = quoting.optimal_premium(intensity)
    assert eps == pytest.approx(40.0 / (2 * 200.0))
    # inside the domain where the intensity is actually positive
    assert 0.0 < eps < intensity.max_premium


@pytest.mark.parametrize("C,D", [(40.0, 200.0), (10.0, 50.0), (1.0, 1.0)])
def test_optimal_premium_matches_numeric_maximizer(C, D):
    intensity = quoting.LinearIntensity(C=C, D=D)
    closed_form = quoting.optimal_premium(intensity)
    numeric = quoting.optimal_premium_numeric(intensity.rate, eps_bounds=(0.0, intensity.max_premium))
    assert numeric == pytest.approx(closed_form, abs=1e-4)


def test_intensity_is_zero_outside_the_papers_support():
    """Paper eq (4): lambda(eps) = C - D*eps only on 0 <= eps < C/D, zero elsewhere.
    A negative eps must not extrapolate to a rate above C."""
    intensity = quoting.LinearIntensity(C=40.0, D=200.0)
    assert intensity.rate(-0.05) == 0.0
    assert intensity.rate(-1.0) == 0.0
    assert intensity.rate_prime(-0.05) == 0.0
    # boundaries: 0 is inside the support, C/D is not
    assert intensity.rate(0.0) == pytest.approx(intensity.C)
    assert intensity.rate(intensity.max_premium) == 0.0


def test_optimal_premium_numeric_raises_when_maximizer_is_truncated():
    """A search range too narrow for the premium scale would otherwise silently
    return the upper bound instead of the FOC's solution."""
    intensity = quoting.LinearIntensity(C=40.0, D=0.05)  # max_premium = 800, eps* = 400
    with pytest.raises(RuntimeError, match="upper bound"):
        quoting.optimal_premium_numeric(intensity.rate, eps_bounds=(0.0, 10.0))
    # with a wide enough range it recovers the closed form
    wide = quoting.optimal_premium_numeric(intensity.rate, eps_bounds=(0.0, intensity.max_premium))
    assert wide == pytest.approx(quoting.optimal_premium(intensity), rel=1e-4)


def test_optimal_premium_maximizes_revenue():
    """eps* should beat nearby eps at maximizing R(eps) = eps*lambda(eps)."""
    intensity = quoting.LinearIntensity(C=40.0, D=200.0)
    eps_star = quoting.optimal_premium(intensity)
    revenue_star = eps_star * intensity.rate(eps_star)
    for eps in (eps_star - 0.02, eps_star + 0.02):
        assert eps_star * intensity.rate(eps_star) >= eps * intensity.rate(eps)
    assert revenue_star > 0


def test_quote_option_brackets_mid():
    S, K, T, r, q, sigma = 30000.0, 30000.0, 0.25, 0.0, 0.0, 0.6
    intensity = quoting.LinearIntensity(C=40.0, D=200.0)
    quote = quoting.quote_option(S, K, T, r, q, sigma, "C", intensity)
    assert quote.bid_price < quote.mid < quote.ask_price
    assert quote.mid == pytest.approx(bs.price(S, K, T, r, q, sigma, "C"))
    assert quote.delta == pytest.approx(bs.delta(S, K, T, r, q, sigma, "C"))
    assert quote.gamma == pytest.approx(bs.gamma(S, K, T, r, q, sigma, "C"))
    assert quote.vega == pytest.approx(bs.vega(S, K, T, r, q, sigma, "C"))


def test_quote_option_symmetric_premiums_by_default():
    S, K, T, r, q, sigma = 30000.0, 30000.0, 0.25, 0.0, 0.0, 0.6
    intensity = quoting.LinearIntensity(C=40.0, D=200.0)
    quote = quoting.quote_option(S, K, T, r, q, sigma, "C", intensity)
    assert quote.bid_premium == pytest.approx(quote.ask_premium)
    assert quote.mid - quote.bid_price == pytest.approx(quote.ask_price - quote.mid)


def test_quote_option_asymmetric_intensity_gives_asymmetric_premiums():
    S, K, T, r, q, sigma = 30000.0, 30000.0, 0.25, 0.0, 0.0, 0.6
    bid_intensity = quoting.LinearIntensity(C=40.0, D=200.0)
    ask_intensity = quoting.LinearIntensity(C=40.0, D=400.0)  # thinner ask liquidity -> tighter premium
    quote = quoting.quote_option(S, K, T, r, q, sigma, "C", bid_intensity, ask_intensity=ask_intensity)
    assert quote.ask_premium != pytest.approx(quote.bid_premium)
    assert quote.ask_premium == pytest.approx(quoting.optimal_premium(ask_intensity))


def test_theorem1_premiums_are_inventory_independent():
    """Theorem 1's central claim: with continuous delta-hedging, bid/ask
    premiums must not move with inventory -- only the hedge should."""
    S, K, T, r, q, sigma = 30000.0, 31000.0, 0.5, 0.0, 0.0, 0.55
    intensity = quoting.LinearIntensity(C=40.0, D=200.0)

    quotes = [
        quoting.quote_option(S, K, T, r, q, sigma, "C", intensity, q_o=q_o)
        for q_o in (-50.0, -10.0, 0.0, 10.0, 50.0)
    ]

    bid_prices = {round(qt.bid_price, 10) for qt in quotes}
    ask_prices = {round(qt.ask_price, 10) for qt in quotes}
    assert len(bid_prices) == 1
    assert len(ask_prices) == 1


def test_hedge_scales_linearly_with_inventory():
    S, K, T, r, q, sigma = 30000.0, 31000.0, 0.5, 0.0, 0.0, 0.55
    intensity = quoting.LinearIntensity(C=40.0, D=200.0)
    delta = bs.delta(S, K, T, r, q, sigma, "C")

    for q_o in (-50.0, 0.0, 10.0, 50.0):
        quote = quoting.quote_option(S, K, T, r, q, sigma, "C", intensity, q_o=q_o)
        assert quote.hedge_shares == pytest.approx(-delta * q_o)
        assert quote.hedge_notional == pytest.approx(S * -delta * q_o)


def test_quote_from_surface_uses_surface_implied_vol():
    class StubSurface:
        def implied_vol(self, K, T):
            return 0.6

    S, K, T, r, q = 30000.0, 30000.0, 0.25, 0.0, 0.0
    intensity = quoting.LinearIntensity(C=40.0, D=200.0)

    from_surface = quoting.quote_from_surface(StubSurface(), S, K, T, r, q, "C", intensity)
    direct = quoting.quote_option(S, K, T, r, q, 0.6, "C", intensity)
    assert from_surface.mid == pytest.approx(direct.mid)


# --- Theorem 2: illiquid underlying -----------------------------------------

THEOREM2_BASE_KWARGS = dict(
    S=65000.0, K=65000.0, T_opt=0.25, r=0.0, q=0.0, sigma_bs=0.6, option_type="C",
    stock_intensity=quoting.LinearIntensity(C=40.0, D=2000.0),
    option_intensity=quoting.LinearIntensity(C=40.0, D=200.0),
    # gamma is tiny on purpose: the risk term scales with S^2, and S is a real
    # BTC price (~65000) rather than the paper's own toy example (S=100), so a
    # "reasonable" risk aversion here is many orders of magnitude smaller than
    # the paper's illustrative gamma=0.006/0.1.
    gamma=1e-9, sigma_risk=0.6, horizon=1 / 365,
)


def test_theorem2_risk_neutral_matches_revenue_maximizing_solution():
    """gamma=0 must collapse to the flat A/2B, C/2D solution regardless of inventory."""
    kwargs = dict(THEOREM2_BASE_KWARGS, gamma=0.0)
    for q_s, q_o in [(0.0, 0.0), (-50.0, 10.0), (100.0, -20.0)]:
        quote = quoting.quote_theorem2(**kwargs, q_s=q_s, q_o=q_o)
        assert quote.ask_stock_premium == pytest.approx(kwargs["stock_intensity"].max_premium / 2)
        assert quote.bid_stock_premium == pytest.approx(kwargs["stock_intensity"].max_premium / 2)
        assert quote.ask_option_premium == pytest.approx(kwargs["option_intensity"].max_premium / 2)
        assert quote.bid_option_premium == pytest.approx(kwargs["option_intensity"].max_premium / 2)


def test_theorem2_matches_hand_computed_formula():
    kwargs = dict(THEOREM2_BASE_KWARGS, q_s=5.0, q_o=-3.0)
    quote = quoting.quote_theorem2(**kwargs)

    d = bs.delta(kwargs["S"], kwargs["K"], kwargs["T_opt"], kwargs["r"], kwargs["q"],
                 kwargs["sigma_bs"], kwargs["option_type"])
    net_delta = kwargs["q_s"] + kwargs["q_o"] * d
    risk = kwargs["gamma"] * kwargs["sigma_risk"] ** 2 * kwargs["horizon"] * kwargs["S"] ** 2
    A_over_B = kwargs["stock_intensity"].max_premium
    C_over_D = kwargs["option_intensity"].max_premium

    expected_ask_stock = np.clip(A_over_B / 2 - risk * (net_delta - 0.5), 0.0, A_over_B)
    expected_bid_stock = np.clip(A_over_B / 2 + risk * (net_delta + 0.5), 0.0, A_over_B)
    expected_ask_option = np.clip(C_over_D / 2 - risk * d * (net_delta - 0.5 * d), 0.0, C_over_D)
    expected_bid_option = np.clip(C_over_D / 2 + risk * d * (net_delta + 0.5 * d), 0.0, C_over_D)

    assert quote.net_delta == pytest.approx(net_delta)
    assert quote.ask_stock_premium == pytest.approx(expected_ask_stock)
    assert quote.bid_stock_premium == pytest.approx(expected_bid_stock)
    assert quote.ask_option_premium == pytest.approx(expected_ask_option)
    assert quote.bid_option_premium == pytest.approx(expected_bid_option)


def test_theorem2_prices_are_mid_plus_minus_premium():
    quote = quoting.quote_theorem2(**THEOREM2_BASE_KWARGS, q_s=1.0, q_o=1.0)
    assert quote.ask_stock_price == pytest.approx(THEOREM2_BASE_KWARGS["S"] + quote.ask_stock_premium)
    assert quote.bid_stock_price == pytest.approx(THEOREM2_BASE_KWARGS["S"] - quote.bid_stock_premium)
    assert quote.ask_option_price == pytest.approx(quote.mid_option + quote.ask_option_premium)
    assert quote.bid_option_price == pytest.approx(quote.mid_option - quote.bid_option_premium)


def test_theorem2_tilts_toward_selling_when_long_net_delta():
    """Long inventory should tighten the ask (encourage selling) and widen the bid
    (discourage buying more) relative to flat, per the paper's tilting story."""
    flat = quoting.quote_theorem2(**THEOREM2_BASE_KWARGS, q_s=0.0, q_o=0.0)
    long_ = quoting.quote_theorem2(**THEOREM2_BASE_KWARGS, q_s=2.0, q_o=0.0)
    short = quoting.quote_theorem2(**THEOREM2_BASE_KWARGS, q_s=-2.0, q_o=0.0)

    assert long_.ask_stock_premium < flat.ask_stock_premium < short.ask_stock_premium
    assert short.bid_stock_premium < flat.bid_stock_premium < long_.bid_stock_premium


def test_theorem2_premiums_clip_to_bounds():
    kwargs = dict(THEOREM2_BASE_KWARGS, gamma=10.0)  # exaggerate risk aversion to force saturation
    quote = quoting.quote_theorem2(**kwargs, q_s=1_000_000.0, q_o=0.0)
    assert quote.ask_stock_premium == 0.0
    assert quote.bid_stock_premium == pytest.approx(kwargs["stock_intensity"].max_premium)
    assert quote.ask_option_premium >= 0.0
    assert quote.bid_option_premium <= kwargs["option_intensity"].max_premium


def test_theorem2_option_tilt_vanishes_for_near_zero_delta():
    """Deep OTM option (delta ~ 0): the option-side tilt term is proportional to
    delta, so its premiums should stay near the flat C/2D even under large inventory."""
    kwargs = dict(THEOREM2_BASE_KWARGS, K=200000.0)  # far OTM call, delta near 0
    quote = quoting.quote_theorem2(**kwargs, q_s=0.0, q_o=500.0)
    C_over_2D = kwargs["option_intensity"].max_premium / 2
    assert quote.ask_option_premium == pytest.approx(C_over_2D, abs=1e-3)
    assert quote.bid_option_premium == pytest.approx(C_over_2D, abs=1e-3)


# --- gamma calibration -------------------------------------------------------

def test_implied_gamma_from_spread_recovers_known_gamma():
    intensity = quoting.LinearIntensity(C=40.0, D=2000.0)  # max_premium = 0.02
    sigma_risk, S, horizon, leg_scale = 0.6, 65000.0, 1 / 365, 1.0
    gamma_true = 5e-8

    risk = gamma_true * sigma_risk ** 2 * horizon * S ** 2 * leg_scale ** 2
    real_spread = intensity.max_premium + risk

    gamma_hat = quoting.implied_gamma_from_spread(intensity, real_spread, sigma_risk, S, horizon, leg_scale)
    assert gamma_hat == pytest.approx(gamma_true, rel=1e-9)


def test_implied_gamma_from_spread_negative_when_model_already_wider():
    """If the risk-neutral (gamma=0) spread already exceeds the real market
    spread, no non-negative gamma reconciles them -- must return negative,
    not clip or raise (see the module's own real-data finding)."""
    intensity = quoting.LinearIntensity(C=40.0, D=2000.0)  # max_premium = 0.02
    tiny_real_spread = 0.001  # far tighter than max_premium=0.02
    gamma_hat = quoting.implied_gamma_from_spread(intensity, tiny_real_spread, 0.6, 65000.0, 1 / 365)
    assert gamma_hat < 0


def test_implied_gamma_from_risk_limit_is_always_positive():
    intensity = quoting.LinearIntensity(C=40.0, D=2000.0)
    for net_delta_limit in [1.0, 5.0, 50.0, 1000.0]:
        gamma_hat = quoting.implied_gamma_from_risk_limit(intensity, net_delta_limit, 0.6, 65000.0, 1 / 365)
        assert gamma_hat > 0


def test_implied_gamma_from_risk_limit_saturates_option_leg_at_stated_limit():
    """Regression: Theorem 2's option tilt is risk*Delta*(net_delta - Delta/2), so the
    saturation point depends on Delta both inside and outside the bracket -- treating it
    as risk*Delta^2*(net_delta - 1/2) saturates far too early."""
    kwargs = dict(THEOREM2_BASE_KWARGS)
    S, K, T_opt, r, q, sigma_bs, otype = (kwargs[k] for k in ("S", "K", "T_opt", "r", "q", "sigma_bs", "option_type"))
    delta = float(bs.delta(S, K, T_opt, r, q, sigma_bs, otype))
    net_delta_limit = 20.0

    gamma_hat = quoting.implied_gamma_from_risk_limit(
        kwargs["option_intensity"], net_delta_limit, kwargs["sigma_risk"], S,
        kwargs["horizon"], leg_scale=delta,
    )

    # saturation must happen exactly AT the limit: zero there, still strictly positive just
    # below it. The old (leg_scale**2) form saturated early, at net_delta_limit*Delta ~= 11
    # rather than 20, so it was already clipped to zero at 0.95*limit.
    at_limit = quoting.quote_theorem2(**dict(kwargs, gamma=gamma_hat), q_s=net_delta_limit, q_o=0.0)
    assert at_limit.net_delta == pytest.approx(net_delta_limit)
    assert at_limit.ask_option_premium == pytest.approx(0.0, abs=1e-9)

    just_below = quoting.quote_theorem2(**dict(kwargs, gamma=gamma_hat), q_s=0.95 * net_delta_limit, q_o=0.0)
    assert just_below.ask_option_premium > 0.0


def test_implied_gamma_from_risk_limit_rejects_unreachable_limit():
    intensity = quoting.LinearIntensity(C=40.0, D=2000.0)
    with pytest.raises(ValueError):
        quoting.implied_gamma_from_risk_limit(intensity, 0.5, 0.6, 65000.0, 1 / 365)
    with pytest.raises(ValueError):
        quoting.implied_gamma_from_risk_limit(intensity, 0.3, 0.6, 65000.0, 1 / 365)


def test_implied_gamma_from_risk_limit_saturates_at_the_stated_limit():
    """Plugging the resulting gamma back into quote_theorem2 at net_delta ==
    net_delta_limit should put the ask premium exactly at its floor (0)."""
    intensity = quoting.LinearIntensity(C=40.0, D=2000.0)
    sigma_risk, S, horizon = 0.6, 65000.0, 1 / 365
    net_delta_limit = 20.0
    gamma_hat = quoting.implied_gamma_from_risk_limit(intensity, net_delta_limit, sigma_risk, S, horizon)

    kwargs = dict(THEOREM2_BASE_KWARGS)
    kwargs["stock_intensity"] = intensity
    kwargs["gamma"] = gamma_hat
    kwargs["sigma_risk"] = sigma_risk
    kwargs["horizon"] = horizon
    quote = quoting.quote_theorem2(**kwargs, q_s=net_delta_limit, q_o=0.0)
    assert quote.ask_stock_premium == pytest.approx(0.0, abs=1e-9)


# --- Theorem 3: multi-period tilt recursion ----------------------------------

def theorem3_env():
    return dict(
        S=65000.0, K=65000.0, T_opt=0.25, r=0.0, q=0.0, sigma_bs=0.6, option_type="C",
        stock_intensity=quoting.LinearIntensity(C=40.0, D=2000.0),
        option_intensity=quoting.LinearIntensity(C=40.0, D=200.0),
        gamma=1e-9, sigma_risk=0.6,
    )


def test_tilt_slope_terminal_condition():
    env = theorem3_env()
    delta = bs.delta(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"])
    terminal_horizon = 1 / 365
    m = quoting.compute_tilt_slope_path(
        n_sessions=5, session_dt=1 / (365 * 24), terminal_horizon=terminal_horizon,
        gamma=env["gamma"], sigma_risk=env["sigma_risk"], S=env["S"], delta=delta,
        stock_intensity=env["stock_intensity"], option_intensity=env["option_intensity"],
    )
    expected_terminal = -env["gamma"] * env["sigma_risk"] ** 2 * env["S"] ** 2 * terminal_horizon
    assert len(m) == 6
    assert m[5] == pytest.approx(expected_terminal)


def test_tilt_slope_matches_hand_computed_two_steps():
    env = theorem3_env()
    delta = bs.delta(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"])
    terminal_horizon, session_dt = 1 / 365, 1 / (365 * 24)

    m = quoting.compute_tilt_slope_path(
        n_sessions=2, session_dt=session_dt, terminal_horizon=terminal_horizon,
        gamma=env["gamma"], sigma_risk=env["sigma_risk"], S=env["S"], delta=delta,
        stock_intensity=env["stock_intensity"], option_intensity=env["option_intensity"],
    )

    m2_expected = -env["gamma"] * env["sigma_risk"] ** 2 * env["S"] ** 2 * terminal_horizon
    B, D_ = env["stock_intensity"].D, env["option_intensity"].D
    coef = 2 * B + 2 * D_ * delta ** 2
    m1_expected = m2_expected + session_dt * coef * m2_expected ** 2
    m0_expected = m1_expected + session_dt * coef * m1_expected ** 2

    assert m[2] == pytest.approx(m2_expected)
    assert m[1] == pytest.approx(m1_expected)
    assert m[0] == pytest.approx(m0_expected)


def test_tilt_slope_magnitude_shrinks_going_backward():
    """Paper's claim: m_n >> m_0 in magnitude -- quotes are more sensitive to
    inventory near the end of the day than at the start."""
    env = theorem3_env()
    delta = bs.delta(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"])
    m = quoting.compute_tilt_slope_path(
        n_sessions=20, session_dt=1 / (365 * 24), terminal_horizon=1 / 365,
        gamma=env["gamma"], sigma_risk=env["sigma_risk"], S=env["S"], delta=delta,
        stock_intensity=env["stock_intensity"], option_intensity=env["option_intensity"],
    )
    assert np.all(np.diff(np.abs(m)) >= 0)  # |m| non-decreasing as i increases toward the terminal
    assert abs(m[0]) < abs(m[-1])


def test_tilt_slope_raises_on_numerical_divergence():
    """A large enough gamma/coefficient makes the explicit recursion overshoot past
    zero and blow up -- this must raise clearly rather than silently return inf/garbage."""
    env = theorem3_env()
    delta = bs.delta(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"])
    with pytest.raises(RuntimeError):
        quoting.compute_tilt_slope_path(
            n_sessions=100, session_dt=1.0, terminal_horizon=1.0,
            gamma=1e6, sigma_risk=env["sigma_risk"], S=env["S"], delta=delta,
            stock_intensity=env["stock_intensity"], option_intensity=env["option_intensity"],
        )


def test_tilt_slope_rejects_invalid_n_sessions():
    env = theorem3_env()
    with pytest.raises(ValueError):
        quoting.compute_tilt_slope_path(
            n_sessions=0, session_dt=1 / (365 * 24), terminal_horizon=1 / 365,
            gamma=env["gamma"], sigma_risk=env["sigma_risk"], S=env["S"], delta=0.5,
            stock_intensity=env["stock_intensity"], option_intensity=env["option_intensity"],
        )


def test_theorem3_reproduces_theorem2_at_single_session():
    """The paper's own claim: Theorem 2 is Theorem 3's last-session special case."""
    env = theorem3_env()
    horizon = 1 / 365
    delta = bs.delta(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"])

    m = quoting.compute_tilt_slope_path(
        n_sessions=1, session_dt=1 / (365 * 24), terminal_horizon=horizon,
        gamma=env["gamma"], sigma_risk=env["sigma_risk"], S=env["S"], delta=delta,
        stock_intensity=env["stock_intensity"], option_intensity=env["option_intensity"],
    )

    for q_s, q_o in [(0.0, 0.0), (5.0, -3.0), (-10.0, 20.0)]:
        q2 = quoting.quote_theorem2(
            env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"],
            env["stock_intensity"], env["option_intensity"],
            gamma=env["gamma"], sigma_risk=env["sigma_risk"], horizon=horizon, q_s=q_s, q_o=q_o,
        )
        q3 = quoting.quote_theorem3(
            env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"],
            env["stock_intensity"], env["option_intensity"],
            m_next=m[1], q_s=q_s, q_o=q_o,
        )
        assert q3.ask_stock_premium == pytest.approx(q2.ask_stock_premium)
        assert q3.bid_stock_premium == pytest.approx(q2.bid_stock_premium)
        assert q3.ask_option_premium == pytest.approx(q2.ask_option_premium)
        assert q3.bid_option_premium == pytest.approx(q2.bid_option_premium)


def test_theorem3_risk_neutral_matches_flat_solution():
    env = theorem3_env()
    env["gamma"] = 0.0
    delta = bs.delta(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"])
    m = quoting.compute_tilt_slope_path(
        n_sessions=10, session_dt=1 / (365 * 24), terminal_horizon=1 / 365,
        gamma=0.0, sigma_risk=env["sigma_risk"], S=env["S"], delta=delta,
        stock_intensity=env["stock_intensity"], option_intensity=env["option_intensity"],
    )
    assert np.allclose(m, 0.0)

    quote = quoting.quote_theorem3(
        env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"],
        env["stock_intensity"], env["option_intensity"], m_next=m[5], q_s=100.0, q_o=-50.0,
    )
    assert quote.ask_stock_premium == pytest.approx(env["stock_intensity"].max_premium / 2)
    assert quote.bid_stock_premium == pytest.approx(env["stock_intensity"].max_premium / 2)


def test_tilt_more_sensitive_near_terminal_session_than_early():
    """Using m_n (terminal) should produce a bigger inventory-driven price swing
    than using m_1 (near the start of the day) at the same inventory level."""
    env = theorem3_env()
    delta = bs.delta(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"])
    m = quoting.compute_tilt_slope_path(
        n_sessions=20, session_dt=1 / (365 * 24), terminal_horizon=1 / 365,
        gamma=env["gamma"], sigma_risk=env["sigma_risk"], S=env["S"], delta=delta,
        stock_intensity=env["stock_intensity"], option_intensity=env["option_intensity"],
    )

    def ask_stock_at(m_next, q_s):
        return quoting.quote_theorem3(
            env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"],
            env["stock_intensity"], env["option_intensity"], m_next=m_next, q_s=q_s, q_o=0.0
        ).ask_stock_premium

    late_swing = ask_stock_at(m[20], 0.0) - ask_stock_at(m[20], 5.0)   # using m_n
    early_swing = ask_stock_at(m[1], 0.0) - ask_stock_at(m[1], 5.0)    # using m_1
    assert late_swing > early_swing > 0


# --- Theorem 4/5: Section IV (Gamma/Vega risk under stochastic vol) --------

THEOREM4_BASE_KWARGS = dict(
    S=100.0, K=100.0, T_opt=1.0, r=0.0, q=0.0, sigma_bs=0.2, option_type="C",
    option_intensity=quoting.LinearIntensity(C=40.0, D=200.0),
    gamma=0.001, sigma_risk=0.2, alpha=0.05, horizon=0.1,
)


def test_gamma_vega_risk_k_matches_hand_computation():
    bs_gamma_greek, S, sigma_risk, alpha, horizon, T_opt = 0.02, 100.0, 0.2, 0.05, 0.1, 1.0
    k = quoting._gamma_vega_risk_k(bs_gamma_greek, S, sigma_risk, alpha, horizon, T_opt)
    expected = (0.5 * sigma_risk ** 2 * horizon + alpha ** 2 * T_opt ** 2) * bs_gamma_greek ** 2 * S ** 4 * sigma_risk ** 2 * horizon
    assert k == pytest.approx(expected)


def test_theorem4_risk_neutral_matches_flat_solution():
    kwargs = dict(THEOREM4_BASE_KWARGS, gamma=0.0)
    for q_o in [0.0, -50.0, 100.0]:
        quote = quoting.quote_theorem4(**kwargs, q_o=q_o)
        assert quote.ask_premium == pytest.approx(kwargs["option_intensity"].max_premium / 2)
        assert quote.bid_premium == pytest.approx(kwargs["option_intensity"].max_premium / 2)


def test_theorem4_matches_hand_computed_formula():
    kwargs = dict(THEOREM4_BASE_KWARGS, q_o=7.0)
    quote = quoting.quote_theorem4(**kwargs)

    S, K, T_opt, r, q, sigma_bs, otype = (kwargs[k] for k in ("S", "K", "T_opt", "r", "q", "sigma_bs", "option_type"))
    bs_gamma_greek = bs.gamma(S, K, T_opt, r, q, sigma_bs)
    k = quoting._gamma_vega_risk_k(bs_gamma_greek, S, kwargs["sigma_risk"], kwargs["alpha"], kwargs["horizon"], T_opt)
    m = -kwargs["gamma"] * k
    C_over_D = kwargs["option_intensity"].max_premium

    expected_ask = np.clip(C_over_D / 2 + m * (7.0 - 0.5), 0.0, C_over_D)
    expected_bid = np.clip(C_over_D / 2 - m * (7.0 + 0.5), 0.0, C_over_D)

    assert quote.m == pytest.approx(m)
    assert quote.ask_premium == pytest.approx(expected_ask)
    assert quote.bid_premium == pytest.approx(expected_bid)


def test_theorem4_tilts_toward_selling_when_long_inventory():
    flat = quoting.quote_theorem4(**THEOREM4_BASE_KWARGS, q_o=0.0)
    long_ = quoting.quote_theorem4(**THEOREM4_BASE_KWARGS, q_o=2.0)
    short = quoting.quote_theorem4(**THEOREM4_BASE_KWARGS, q_o=-2.0)

    assert long_.ask_premium < flat.ask_premium < short.ask_premium
    assert short.bid_premium < flat.bid_premium < long_.bid_premium


def test_theorem4_premiums_clip_to_bounds():
    kwargs = dict(THEOREM4_BASE_KWARGS, gamma=10.0)
    quote = quoting.quote_theorem4(**kwargs, q_o=1_000_000.0)
    assert quote.ask_premium == 0.0
    assert quote.bid_premium == pytest.approx(kwargs["option_intensity"].max_premium)


def test_implied_gamma_from_option_risk_limit_is_positive_and_saturates():
    kwargs = THEOREM4_BASE_KWARGS
    bs_gamma_greek = bs.gamma(kwargs["S"], kwargs["K"], kwargs["T_opt"], kwargs["r"], kwargs["q"], kwargs["sigma_bs"])
    q_o_limit = 5.0
    gamma_hat = quoting.implied_gamma_from_option_risk_limit(
        kwargs["option_intensity"], q_o_limit, bs_gamma_greek, kwargs["S"],
        kwargs["sigma_risk"], kwargs["alpha"], kwargs["horizon"], kwargs["T_opt"],
    )
    assert gamma_hat > 0

    quote = quoting.quote_theorem4(
        kwargs["S"], kwargs["K"], kwargs["T_opt"], kwargs["r"], kwargs["q"], kwargs["sigma_bs"], kwargs["option_type"],
        kwargs["option_intensity"], gamma=gamma_hat, sigma_risk=kwargs["sigma_risk"], alpha=kwargs["alpha"],
        horizon=kwargs["horizon"], q_o=q_o_limit,
    )
    assert quote.ask_premium == pytest.approx(0.0, abs=1e-9)


def test_implied_gamma_from_option_risk_limit_rejects_unreachable_limit():
    kwargs = THEOREM4_BASE_KWARGS
    bs_gamma_greek = bs.gamma(kwargs["S"], kwargs["K"], kwargs["T_opt"], kwargs["r"], kwargs["q"], kwargs["sigma_bs"])
    with pytest.raises(ValueError):
        quoting.implied_gamma_from_option_risk_limit(
            kwargs["option_intensity"], 0.5, bs_gamma_greek, kwargs["S"],
            kwargs["sigma_risk"], kwargs["alpha"], kwargs["horizon"], kwargs["T_opt"],
        )


def theorem5_env():
    return dict(THEOREM4_BASE_KWARGS)


def test_tilt_slope_theorem5_terminal_condition():
    env = theorem5_env()
    bs_gamma_greek = bs.gamma(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"])
    terminal_horizon = env["horizon"]
    m = quoting.compute_tilt_slope_path_theorem5(
        n_sessions=5, session_dt=terminal_horizon / 10, terminal_horizon=terminal_horizon,
        gamma=env["gamma"], sigma_risk=env["sigma_risk"], S=env["S"], bs_gamma_greek=bs_gamma_greek,
        alpha=env["alpha"], T_opt=env["T_opt"], option_intensity=env["option_intensity"],
    )
    expected_k = quoting._gamma_vega_risk_k(bs_gamma_greek, env["S"], env["sigma_risk"], env["alpha"],
                                             terminal_horizon, env["T_opt"])
    assert len(m) == 6
    assert m[5] == pytest.approx(-env["gamma"] * expected_k)


def test_tilt_slope_theorem5_matches_hand_computed_two_steps():
    env = theorem5_env()
    bs_gamma_greek = bs.gamma(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"])
    terminal_horizon = env["horizon"]
    session_dt = terminal_horizon / 2

    m = quoting.compute_tilt_slope_path_theorem5(
        n_sessions=2, session_dt=session_dt, terminal_horizon=terminal_horizon,
        gamma=env["gamma"], sigma_risk=env["sigma_risk"], S=env["S"], bs_gamma_greek=bs_gamma_greek,
        alpha=env["alpha"], T_opt=env["T_opt"], option_intensity=env["option_intensity"],
    )

    k = quoting._gamma_vega_risk_k(bs_gamma_greek, env["S"], env["sigma_risk"], env["alpha"], terminal_horizon, env["T_opt"])
    m2_expected = -env["gamma"] * k
    coef = 2 * env["option_intensity"].D
    m1_expected = m2_expected + session_dt * coef * m2_expected ** 2
    m0_expected = m1_expected + session_dt * coef * m1_expected ** 2

    assert m[2] == pytest.approx(m2_expected)
    assert m[1] == pytest.approx(m1_expected)
    assert m[0] == pytest.approx(m0_expected)


def test_theorem5_reproduces_theorem4_at_single_session():
    env = theorem5_env()
    bs_gamma_greek = bs.gamma(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"])

    m = quoting.compute_tilt_slope_path_theorem5(
        n_sessions=1, session_dt=env["horizon"] / 10, terminal_horizon=env["horizon"],
        gamma=env["gamma"], sigma_risk=env["sigma_risk"], S=env["S"], bs_gamma_greek=bs_gamma_greek,
        alpha=env["alpha"], T_opt=env["T_opt"], option_intensity=env["option_intensity"],
    )

    for q_o in [0.0, 5.0, -3.0]:
        q4 = quoting.quote_theorem4(
            env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"],
            env["option_intensity"], gamma=env["gamma"], sigma_risk=env["sigma_risk"], alpha=env["alpha"],
            horizon=env["horizon"], q_o=q_o,
        )
        q5 = quoting.quote_theorem5(
            env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"],
            env["option_intensity"], m_next=m[1], q_o=q_o,
        )
        assert q5.ask_premium == pytest.approx(q4.ask_premium)
        assert q5.bid_premium == pytest.approx(q4.bid_premium)


def test_theorem5_risk_neutral_matches_flat_solution():
    env = theorem5_env()
    bs_gamma_greek = bs.gamma(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"])
    m = quoting.compute_tilt_slope_path_theorem5(
        n_sessions=10, session_dt=env["horizon"] / 100, terminal_horizon=env["horizon"],
        gamma=0.0, sigma_risk=env["sigma_risk"], S=env["S"], bs_gamma_greek=bs_gamma_greek,
        alpha=env["alpha"], T_opt=env["T_opt"], option_intensity=env["option_intensity"],
    )
    assert np.allclose(m, 0.0)

    quote = quoting.quote_theorem5(
        env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"], env["option_type"],
        env["option_intensity"], m_next=m[5], q_o=200.0,
    )
    assert quote.ask_premium == pytest.approx(env["option_intensity"].max_premium / 2)
    assert quote.bid_premium == pytest.approx(env["option_intensity"].max_premium / 2)


def test_tilt_slope_theorem5_raises_on_numerical_divergence():
    env = theorem5_env()
    bs_gamma_greek = bs.gamma(env["S"], env["K"], env["T_opt"], env["r"], env["q"], env["sigma_bs"])
    with pytest.raises(RuntimeError):
        quoting.compute_tilt_slope_path_theorem5(
            n_sessions=100, session_dt=1.0, terminal_horizon=1.0,
            gamma=1e6, sigma_risk=env["sigma_risk"], S=env["S"], bs_gamma_greek=bs_gamma_greek,
            alpha=env["alpha"], T_opt=env["T_opt"], option_intensity=env["option_intensity"],
        )


def test_tilt_slope_theorem5_rejects_invalid_n_sessions():
    env = theorem5_env()
    with pytest.raises(ValueError):
        quoting.compute_tilt_slope_path_theorem5(
            n_sessions=0, session_dt=0.01, terminal_horizon=0.1,
            gamma=env["gamma"], sigma_risk=env["sigma_risk"], S=env["S"], bs_gamma_greek=0.02,
            alpha=env["alpha"], T_opt=env["T_opt"], option_intensity=env["option_intensity"],
        )

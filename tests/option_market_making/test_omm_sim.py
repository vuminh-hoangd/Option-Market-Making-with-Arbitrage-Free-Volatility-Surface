from dataclasses import replace

import numpy as np
import pytest

from option_market_making import bs_mm
from option_market_making.sim import simulate

N_STEPS = 400


@pytest.fixture(scope="module")
def run(request):
    """Cached simulation runs, keyed by whatever the test asks for."""
    cache: dict = {}

    def _run(surface, contract, params, horizon, n_paths=64, seed=7, n_steps=N_STEPS):
        # Keyed on the frozen params themselves, not id(): a temporary built by
        # `replace` can be collected and its id reused by the next one.
        key = (params, n_paths, seed, n_steps)
        if key not in cache:
            cache[key] = simulate(
                surface, contract["K"], contract["tau"], horizon, params,
                S0=contract["S0"], n_steps=n_steps, n_paths=n_paths, seed=seed,
            )
        return cache[key]

    return _run


def test_no_nan_or_inf_anywhere(surface, contract, params, horizon, run):
    result = run(surface, contract, params, horizon)
    for name in ("t", "S", "q", "V", "cash", "hedge", "option_price", "bid", "ask",
                 "delta_b", "delta_a"):
        array = getattr(result, name)
        assert np.all(np.isfinite(array)), f"non-finite value in {name}"


def test_shapes_are_aligned_to_the_time_grid(surface, contract, params, horizon, run):
    result = run(surface, contract, params, horizon)
    assert result.t.shape == (N_STEPS + 1,)
    for name in ("S", "q", "V", "cash", "hedge", "option_price", "bid", "ask"):
        assert getattr(result, name).shape == (64, N_STEPS + 1)
    assert result.t[0] == 0.0 and result.t[-1] == pytest.approx(horizon)


def test_inventory_stays_within_a_sane_bound(surface, contract, params, horizon, run):
    """
    Roughly 100 lots a day a side and a half-day horizon means order 50 fills
    a side; the inventory penalty keeps the net far below that. A run that
    blows through this bound means the skew has stopped controlling risk.
    """
    result = run(surface, contract, params, horizon)
    assert np.abs(result.q).max() <= 25.0
    assert result.n_bid_fills.sum() > 0 and result.n_ask_fills.sum() > 0


def test_inventory_only_moves_by_whole_lots(surface, contract, params, horizon, run):
    result = run(surface, contract, params, horizon)
    steps = np.diff(result.q, axis=1)
    assert np.all(steps == np.round(steps))


def test_a_tighter_penalty_holds_inventory_closer_to_flat(surface, contract, params, horizon, run):
    loose = run(surface, contract, replace(params, alpha=0.05, beta=100.0), horizon)
    tight = run(surface, contract, replace(params, alpha=20.0, beta=4.0e5), horizon)
    assert np.abs(tight.q).mean() < np.abs(loose.q).mean()


def test_portfolio_starts_flat_and_the_hedge_tracks_the_inventory(surface, contract, params, horizon, run):
    result = run(surface, contract, params, horizon)
    assert np.all(result.V[:, 0] == 0.0)
    assert np.all(result.q[:, 0] == 0.0)

    # Delta_t = q_t * BS delta, at the implied vol, on every path and step.
    tau = contract["tau"]
    for path in (0, 5, 17):
        for step in (0, N_STEPS // 3, N_STEPS):
            expected = result.q[path, step] * bs_mm.delta(
                float(result.S[path, step]), contract["K"],
                tau - float(result.t[step]), result.sigma_imp,
            )
            assert result.hedge[path, step] == pytest.approx(expected)


def test_portfolio_value_reconciles_with_cash_inventory_and_hedge(surface, contract, params, horizon, run):
    result = run(surface, contract, params, horizon)
    reconstructed = result.cash + result.q * result.option_price - result.hedge * result.S
    assert np.allclose(result.V, reconstructed)


def test_quotes_straddle_the_mark_on_every_path_and_step(surface, contract, params, horizon, run):
    result = run(surface, contract, params, horizon)
    assert np.all(result.bid < result.option_price)
    assert np.all(result.option_price < result.ask)


def test_spot_is_a_driftless_gbm_at_the_market_makers_own_vol(surface, contract, params, horizon):
    """
    Realised vol of the simulated log-returns must come back as params.sigma,
    and the spot itself must be a martingale -- the market maker has a view on
    volatility, never on direction.

    Zero drift is checked on E[S_T] rather than on the mean log-return: the
    log-drift is -sigma^2/2 dt per step, which over a half-day horizon is
    smaller than the standard error of the mean log-return even at a million
    samples, so that version of the test could not fail for the right reason.
    E[S_T] = S_0 has a standard error of S_0 sigma sqrt(T / n_paths), which at
    4000 paths is about 3 basis points.
    """
    n_paths = 4000
    result = simulate(
        surface, contract["K"], contract["tau"], horizon, params,
        S0=contract["S0"], n_steps=N_STEPS, n_paths=n_paths, seed=11,
    )
    dt = horizon / N_STEPS
    log_returns = np.diff(np.log(result.S), axis=1)

    realised = log_returns.std() / np.sqrt(dt)
    assert realised == pytest.approx(params.sigma, rel=0.02)

    standard_error = contract["S0"] * params.sigma * np.sqrt(horizon / n_paths)
    assert abs(result.S[:, -1].mean() - contract["S0"]) < 3.0 * standard_error


def test_the_strategy_captures_its_quoted_spread(surface, contract, params, horizon):
    """
    Over a half-day horizon the P&L is dominated by spread capture, not by the
    volatility view: roughly 38 fills a path at a ~38-dollar half-spread. The
    gamma-theta carry over the same horizon is worth single-digit dollars --
    see `test_being_wrong_about_vol_costs_money` for the term that isolates it,
    and the accompanying notebook for the horizon sweep that shows why.
    """
    result = simulate(
        surface, contract["K"], contract["tau"], horizon, params,
        S0=contract["S0"], n_steps=N_STEPS, n_paths=2000, seed=3,
    )
    assert result.terminal_value.mean() > 0.0


def test_simulate_validates_its_inputs(surface, contract, params, horizon):
    K, tau = contract["K"], contract["tau"]
    with pytest.raises(ValueError, match="n_steps"):
        simulate(surface, K, tau, horizon, params, S0=contract["S0"], n_steps=0)
    with pytest.raises(ValueError, match="initial spot"):
        simulate(surface, K, tau, horizon, params, S0=-1.0, n_steps=10)
    with pytest.raises(ValueError, match="strictly before expiry"):
        simulate(surface, K, tau, tau, params, S0=contract["S0"], n_steps=10)


def test_runs_are_reproducible_under_a_seed(surface, contract, params, horizon):
    kwargs = dict(S0=contract["S0"], n_steps=100, n_paths=8, seed=42)
    a = simulate(surface, contract["K"], contract["tau"], horizon, params, **kwargs)
    b = simulate(surface, contract["K"], contract["tau"], horizon, params, **kwargs)
    assert np.array_equal(a.S, b.S)
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.V, b.V)


def test_simulated_spreads_match_the_quoting_module(surface, contract, params, horizon, run):
    """
    `sim` inlines eq. 11 in vector form for speed; it must stay identical to
    `quoting.optimal_spreads`, which is the scalar statement of the same rule.
    """
    from option_market_making.quoting import optimal_spreads

    result = run(surface, contract, params, horizon)
    for path in (0, 11, 40):
        for step in (0, N_STEPS // 2, N_STEPS):
            db, da = optimal_spreads(
                float(result.t[step]), float(result.S[path, step]),
                float(result.q[path, step]), contract["K"], contract["tau"], horizon,
                result.sigma_imp, params,
            )
            assert result.delta_b[path, step] == pytest.approx(db, rel=1e-9)
            assert result.delta_a[path, step] == pytest.approx(da, rel=1e-9)


def test_constant_spreads_quote_a_fixed_width(surface, contract, params, horizon):
    """The paper's zero-intelligence benchmark: same width regardless of state."""
    widths = (30.0, 40.0)
    result = simulate(
        surface, contract["K"], contract["tau"], horizon, params, S0=contract["S0"],
        n_steps=100, n_paths=16, seed=5, constant_spreads=widths,
    )
    assert np.all(result.delta_b == widths[0])
    assert np.all(result.delta_a == widths[1])
    # Still the same accounting: quotes sit around the mark and V reconciles.
    assert np.all(result.bid < result.option_price)
    assert np.all(result.option_price < result.ask)
    reconstructed = result.cash + result.q * result.option_price - result.hedge * result.S
    assert np.allclose(result.V, reconstructed)


def test_constant_spreads_do_not_skew_with_inventory(surface, contract, params, horizon):
    """
    The whole point of the benchmark: it has no inventory control, so it should
    carry a wider inventory swing than the optimal rule at comparable turnover.
    """
    kwargs = dict(S0=contract["S0"], n_steps=N_STEPS, n_paths=256, seed=9)
    optimal = simulate(surface, contract["K"], contract["tau"], horizon, params, **kwargs)
    zero_iq = simulate(
        surface, contract["K"], contract["tau"], horizon, params,
        constant_spreads=(float(optimal.delta_b.mean()), float(optimal.delta_a.mean())),
        **kwargs,
    )
    assert np.abs(zero_iq.q[:, -1]).std() > np.abs(optimal.q[:, -1]).std()


def test_realised_sigma_defaults_to_the_makers_own_view(surface, contract, params, horizon):
    kwargs = dict(S0=contract["S0"], n_steps=100, n_paths=8, seed=21)
    implicit = simulate(surface, contract["K"], contract["tau"], horizon, params, **kwargs)
    explicit = simulate(
        surface, contract["K"], contract["tau"], horizon, params,
        realised_sigma=params.sigma, **kwargs,
    )
    assert np.array_equal(implicit.S, explicit.S)


def test_realised_sigma_drives_the_paths_not_the_quotes(surface, contract, params, horizon):
    """
    Doubling the vol the spot realises must widen the path distribution while
    leaving the t=0 quote -- which depends only on the believed sigma -- untouched.
    """
    kwargs = dict(S0=contract["S0"], n_steps=200, n_paths=512, seed=33)
    calm = simulate(surface, contract["K"], contract["tau"], horizon, params,
                    realised_sigma=0.20, **kwargs)
    wild = simulate(surface, contract["K"], contract["tau"], horizon, params,
                    realised_sigma=0.80, **kwargs)

    assert wild.S[:, -1].std() > 3.0 * calm.S[:, -1].std()
    assert calm.delta_b[0, 0] == pytest.approx(wild.delta_b[0, 0])
    assert calm.delta_a[0, 0] == pytest.approx(wild.delta_a[0, 0])


def test_being_wrong_about_vol_costs_money(surface, contract, params, horizon):
    """
    The model's edge is only real if the view is right. Hold the belief fixed
    (sigma_imp + 5 vol points, so the maker quotes to get long vol) and vary what
    the spot actually does: a market that realises far less than the maker
    believed must pay worse than one that vindicates them.
    """
    kwargs = dict(S0=contract["S0"], n_steps=N_STEPS, n_paths=1500, seed=77)
    vindicated = simulate(surface, contract["K"], contract["tau"], horizon, params,
                          realised_sigma=params.sigma, **kwargs)
    wrong = simulate(surface, contract["K"], contract["tau"], horizon, params,
                     realised_sigma=contract["sigma_imp"] - 0.15, **kwargs)
    assert wrong.terminal_value.mean() < vindicated.terminal_value.mean()


def test_objective_matches_its_definition_on_a_known_path(surface, contract, params, horizon, run):
    """Recompute eq. 2 the slow, explicit way and compare."""
    from option_market_making.sim import objective

    result = run(surface, contract, params, horizon)
    dt = float(result.t[1] - result.t[0])
    for path in (0, 3, 29):
        running = sum(result.q[path, i] ** 2 * dt for i in range(len(result.t) - 1))
        expected = (result.V[path, -1] - params.beta * running
                    - params.alpha * result.q[path, -1] ** 2)
        assert objective(result, params)[path] == pytest.approx(expected)


def test_objective_penalises_a_flat_book_least(surface, contract, params, horizon):
    """
    A strategy that never trades carries no inventory, so its objective equals
    its V_T exactly -- both penalty terms are zero.
    """
    from option_market_making.sim import objective

    # Spreads wide enough that essentially nothing fills.
    idle = simulate(
        surface, contract["K"], contract["tau"], horizon, params, S0=contract["S0"],
        n_steps=100, n_paths=8, seed=13, constant_spreads=(1e4, 1e4),
    )
    assert idle.n_bid_fills.sum() == 0 and idle.n_ask_fills.sum() == 0
    assert objective(idle, params) == pytest.approx(idle.terminal_value)


def test_spot_paths_are_identical_across_strategies(surface, contract, params, horizon):
    """
    The precondition for a paired comparison: changing the quoting rule must not
    change the spot paths. It would if fills and paths shared one RNG stream.
    """
    kwargs = dict(S0=contract["S0"], n_steps=200, n_paths=64, seed=101)
    a = simulate(surface, contract["K"], contract["tau"], horizon, params, **kwargs)
    b = simulate(surface, contract["K"], contract["tau"], horizon, params,
                 constant_spreads=(20.0, 20.0), **kwargs)
    assert np.array_equal(a.S, b.S)
    assert not np.array_equal(a.q, b.q)   # but the strategies really do differ


def test_the_optimal_rule_wins_on_the_objective_it_maximises(surface, contract, params, horizon):
    """
    The point of eq. 11. A constant-width quoter can beat it on mean V_T, but
    must not beat it on eq. 2 -- that is the quantity the rule optimises, and
    losing there would indict the second-order approximation itself.
    """
    from option_market_making.sim import objective

    kwargs = dict(S0=contract["S0"], n_steps=N_STEPS, n_paths=1500, seed=202)
    optimal = simulate(surface, contract["K"], contract["tau"], horizon, params, **kwargs)
    for width in (0.5, 1.0, 2.0):
        rival = simulate(
            surface, contract["K"], contract["tau"], horizon, params,
            constant_spreads=(width / params.kappa_b, width / params.kappa_a), **kwargs,
        )
        # Paired: identical spot paths, so the difference is the strategy alone.
        gap = objective(optimal, params) - objective(rival, params)
        assert gap.mean() > 3.0 * gap.std() / np.sqrt(len(gap)), (
            f"optimal rule failed to beat constant spread x{width} on eq. 2"
        )

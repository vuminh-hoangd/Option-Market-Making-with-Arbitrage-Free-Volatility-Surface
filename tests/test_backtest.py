import numpy as np
import pandas as pd
import pytest

from market_making import backtest, quoting


def make_tape(prices, directions, mark=None, amount=1.0, start_ts=1_700_000_000_000, step_ms=60_000):
    """
    Minimal USD-quoted tape (like BTC-PERPETUAL): `price`/`mark_price` already
    in dollars, so backtests can be run with price_in_underlying_units=False
    and the numbers stay hand-checkable.
    """
    n = len(prices)
    mark = mark if mark is not None else [100.0] * n
    return pd.DataFrame({
        "trade_id": [f"t{i}" for i in range(n)],
        "instrument_name": ["TEST"] * n,
        "timestamp": start_ts + np.arange(n) * step_ms,
        "price": np.asarray(prices, dtype=float),
        "mark_price": np.asarray(mark, dtype=float),
        "index_price": [1.0] * n,
        "direction": directions,
        "amount": [amount] * n,
    })


USD = dict(price_in_underlying_units=False)


def test_no_fills_when_quotes_are_never_crossed():
    # trades happen right at the mid; a wide quote should never be touched
    tape = make_tape([100.0] * 5, ["buy", "sell"] * 2 + ["buy"])
    result = backtest.simulate_fills(tape, backtest.constant_spread_policy(5.0), **USD)
    assert result.n_fills == 0
    assert result.total_pnl == pytest.approx(0.0)
    assert result.final_inventory == 0.0


def test_ask_fills_on_buy_aggressor_and_bid_on_sell():
    # trades at +/-2 around a mid of 100, with a half-spread of 1 -> both sides fill
    tape = make_tape([100.0, 102.0, 98.0], ["buy", "buy", "sell"])
    result = backtest.simulate_fills(tape, backtest.constant_spread_policy(1.0), **USD)

    assert result.n_ask_fills == 1   # the buy aggressor at 102 lifts our 101 ask
    assert result.n_bid_fills == 1   # the sell aggressor at 98 hits our 99 bid
    assert result.final_inventory == pytest.approx(0.0)
    # captured 1.0 of edge on each side, one unit each
    assert result.spread_pnl == pytest.approx(2.0)


def test_pnl_identity_holds():
    rng = np.random.default_rng(0)
    n = 200
    walk = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    prices = walk + rng.normal(0, 1.0, n)
    directions = rng.choice(["buy", "sell"], n)
    tape = make_tape(prices, directions, mark=walk)

    result = backtest.simulate_fills(
        tape, backtest.constant_spread_policy(0.5), fee_per_contract=0.01, **USD
    )
    assert result.n_fills > 0
    assert result.total_pnl == pytest.approx(
        result.spread_pnl + result.inventory_pnl - result.fees_paid, abs=1e-9
    )


def test_wider_spread_fills_less():
    rng = np.random.default_rng(1)
    n = 300
    prices = 100.0 + rng.normal(0, 2.0, n)
    directions = rng.choice(["buy", "sell"], n)
    tape = make_tape(prices, directions)

    counts = [
        backtest.simulate_fills(tape, backtest.constant_spread_policy(h), **USD).n_fills
        for h in (0.5, 1.5, 3.0)
    ]
    assert counts[0] > counts[1] > counts[2]


def test_queue_factor_scales_fills_down():
    tape = make_tape([102.0] * 10, ["buy"] * 10, amount=1.0)
    full = backtest.simulate_fills(tape, backtest.constant_spread_policy(1.0), queue_factor=1.0, **USD)
    quarter = backtest.simulate_fills(tape, backtest.constant_spread_policy(1.0), queue_factor=0.25, **USD)
    assert quarter.spread_pnl == pytest.approx(full.spread_pnl * 0.25)
    assert abs(quarter.final_inventory) == pytest.approx(abs(full.final_inventory) * 0.25)


def test_fees_reduce_total_pnl():
    tape = make_tape([102.0, 98.0] * 20, ["buy", "sell"] * 20)
    free = backtest.simulate_fills(tape, backtest.constant_spread_policy(1.0), **USD)
    charged = backtest.simulate_fills(
        tape, backtest.constant_spread_policy(1.0), fee_per_contract=0.1, **USD
    )
    assert charged.fees_paid > 0
    assert charged.total_pnl < free.total_pnl


def test_max_abs_position_caps_inventory():
    # one-sided flow: every trade is a sell aggressor, so the dealer only ever buys
    tape = make_tape([98.0] * 50, ["sell"] * 50)
    capped = backtest.simulate_fills(
        tape, backtest.constant_spread_policy(1.0), max_abs_position=3.0, **USD
    )
    assert capped.max_abs_inventory <= 3.0


def test_quotes_cannot_use_the_trade_that_fills_them():
    """The mid used to quote must be the previous trade's, never the current one --
    otherwise the policy sees the fill before deciding."""
    seen_mids = []

    def spy_policy(mid, inventory):
        seen_mids.append(mid)
        return backtest.Quote(bid=mid - 1.0, ask=mid + 1.0)

    marks = [100.0, 200.0, 300.0, 400.0]
    tape = make_tape([100.0] * 4, ["buy"] * 4, mark=marks)
    backtest.simulate_fills(tape, spy_policy, **USD)

    # 4 trades -> 3 quoting decisions, each using the PRIOR mark
    assert seen_mids == marks[:-1]


def test_inventory_tilt_controls_inventory_better_than_flat_quotes():
    """The economic claim the tilting exists for: under one-sided flow, tilting
    quotes with inventory should end up less exposed than a flat spread."""
    rng = np.random.default_rng(2)
    n = 400
    # flow skewed toward sell aggressors -> persistent pressure to accumulate longs
    directions = rng.choice(["buy", "sell"], n, p=[0.25, 0.75])
    prices = 100.0 + rng.normal(0, 1.5, n)
    tape = make_tape(prices, directions)

    intensity = quoting.LinearIntensity(C=40.0, D=20.0)  # max_premium = 2.0
    flat = backtest.constant_spread_policy(quoting.optimal_premium(intensity))

    def tilted(mid, inventory):
        # tilt linearly in inventory, saturating at the same bounds the theorems use
        m = 0.15
        ask = float(np.clip(intensity.max_premium / 2 - m * inventory, 0.0, intensity.max_premium))
        bid = float(np.clip(intensity.max_premium / 2 + m * inventory, 0.0, intensity.max_premium))
        return backtest.Quote(bid=mid - bid, ask=mid + ask)

    flat_result = backtest.simulate_fills(tape, flat, **USD)
    tilt_result = backtest.simulate_fills(tape, tilted, **USD)

    assert abs(tilt_result.final_inventory) < abs(flat_result.final_inventory)
    assert tilt_result.max_abs_inventory < flat_result.max_abs_inventory


def test_realized_fill_rate_and_objective_are_computable():
    tape = make_tape([102.0, 98.0] * 30, ["buy", "sell"] * 30)
    result = backtest.simulate_fills(tape, backtest.constant_spread_policy(1.0), **USD)
    assert result.realized_fill_rate > 0
    # a bigger gamma penalizes inventory variance more, so the objective can only fall
    assert result.mean_variance_objective(1.0) <= result.mean_variance_objective(0.0)


def test_compare_policies_tabulates_side_by_side():
    tape = make_tape([102.0, 98.0] * 30, ["buy", "sell"] * 30)
    table = backtest.compare_policies(
        tape,
        {"tight": backtest.constant_spread_policy(0.5),
         "wide": backtest.constant_spread_policy(1.5)},
        **USD,
    )
    assert list(table.index) == ["tight", "wide"]
    assert {"total_pnl", "spread_pnl", "inventory_pnl", "n_fills"} <= set(table.columns)
    assert table.loc["tight", "n_fills"] >= table.loc["wide", "n_fills"]


def test_simulate_fills_validates_arguments():
    tape = make_tape([100.0, 100.0], ["buy", "sell"])
    with pytest.raises(ValueError):
        backtest.simulate_fills(tape, backtest.constant_spread_policy(1.0), queue_factor=0.0, **USD)
    with pytest.raises(ValueError):
        backtest.simulate_fills(tape, backtest.constant_spread_policy(1.0), quote_size=-1.0, **USD)
    with pytest.raises(ValueError):
        backtest.simulate_fills(make_tape([100.0], ["buy"]), backtest.constant_spread_policy(1.0), **USD)

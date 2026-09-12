import numpy as np
import pandas as pd
import pytest

from market_making import calibration
from market_making.quoting import LinearIntensity


def test_survival_rate_counts_at_or_above_threshold():
    eps_values = np.array([0.0, 0.01, 0.02, 0.02, 0.05])
    grid = np.array([0.0, 0.02, 0.04])
    rate = calibration._survival_rate(eps_values, grid, window_seconds=10.0)
    # at eps=0: all 5 count; at eps=0.02: the three >= 0.02 count; at eps=0.04: only the 0.05
    assert rate.tolist() == pytest.approx([5 / 10.0, 3 / 10.0, 1 / 10.0])


def test_fit_relative_intensity_recovers_known_linear_intensity():
    """
    A linear survival curve lambda(eps) = C - D*eps has constant density
    -D over [0, C/D] -- i.e. fill events are uniformly distributed in eps on
    that interval. Generate synthetic fills that way and check the fit
    recovers the true C, D.
    """
    rng = np.random.default_rng(0)
    C_true, D_true = 0.01, 0.2
    window_seconds = 3600.0 * 24
    n_events = int(round(C_true * window_seconds))
    eps_values = rng.uniform(0.0, C_true / D_true, n_events)

    intensity, eps_grid, empirical_rate, fitted_rate = calibration._fit_relative_intensity(
        eps_values, window_seconds, n_bins=20, quantile_cutoff=0.9
    )

    assert intensity.C == pytest.approx(C_true, rel=0.15)
    assert intensity.D == pytest.approx(D_true, rel=0.15)
    assert len(eps_grid) == len(empirical_rate) == len(fitted_rate) == 20


def test_fit_relative_intensity_raises_on_degenerate_data():
    # all fills at eps=0: no decreasing curve to fit
    eps_values = np.zeros(50)
    with pytest.raises(calibration.CalibrationError):
        calibration._fit_relative_intensity(eps_values, window_seconds=100.0)


def test_relative_intensity_to_absolute_matches_hand_derivation():
    rel = calibration.RelativeIntensity(C=0.01, D=0.2)
    mid = 2000.0
    abs_intensity = rel.to_absolute(mid)
    assert isinstance(abs_intensity, LinearIntensity)
    assert abs_intensity.C == pytest.approx(0.01)
    assert abs_intensity.D == pytest.approx(0.2 / 2000.0)
    # sanity: relative premium at abs_intensity's optimal eps should match rel's optimal eta
    from market_making.quoting import optimal_premium
    eps_star = optimal_premium(abs_intensity)
    assert eps_star / mid == pytest.approx(rel.C / (2 * rel.D))


def test_relative_intensity_to_absolute_rejects_nonpositive_mid():
    rel = calibration.RelativeIntensity(C=0.01, D=0.2)
    with pytest.raises(ValueError):
        rel.to_absolute(0.0)


def _make_synthetic_trades(n, direction, base_ts=1_700_000_000_000, spread_seconds=3600, rng=None):
    rng = rng or np.random.default_rng(1)
    mark = 0.05  # BTC units, like a real Deribit option mark_price
    index = 65000.0
    # eps as a fraction of mark, uniform -> matches the linear-intensity assumption
    eta = rng.uniform(0.0, 0.05, n)
    eps_btc = eta * mark
    if direction == "buy":
        price = mark + eps_btc
    else:
        price = mark - eps_btc
    ts = base_ts + rng.uniform(0, spread_seconds * 1000, n).astype(np.int64)
    return pd.DataFrame({
        "instrument_name": "BTC-TEST-50000-C",
        "timestamp": ts,
        "price": price,
        "mark_price": mark,
        "index_price": index,
        "direction": direction,
        "amount": 1.0,
    })


def test_add_fill_distance_assigns_sides_and_clips_negative():
    trades = pd.DataFrame({
        "instrument_name": ["X", "X"],
        "timestamp": [1000, 2000],
        "price": [0.021, 0.020],       # buy above mark, sell below mark -> both "away" fills
        "mark_price": [0.020, 0.021],
        "index_price": [65000.0, 65000.0],
        "direction": ["buy", "sell"],
        "amount": [1.0, 1.0],
    })
    tagged = calibration.add_fill_distance(trades)
    assert list(tagged["side"]) == ["ask", "bid"]
    assert (tagged["eps_dollar"] >= 0).all()
    assert tagged.loc[0, "eps_dollar"] == pytest.approx((0.021 - 0.020) * 65000.0)
    assert tagged.loc[1, "eps_dollar"] == pytest.approx((0.021 - 0.020) * 65000.0)


def test_add_fill_distance_clips_wrong_side_fills_to_zero():
    # a 'buy' that printed *below* the mark (mark drift) shouldn't go negative
    trades = pd.DataFrame({
        "instrument_name": ["X"],
        "timestamp": [1000],
        "price": [0.019],
        "mark_price": [0.020],
        "index_price": [65000.0],
        "direction": ["buy"],
        "amount": [1.0],
    })
    tagged = calibration.add_fill_distance(trades)
    assert tagged.loc[0, "eps_dollar"] == 0.0


def test_calibrate_intensity_end_to_end_on_synthetic_trades():
    buys = _make_synthetic_trades(200, "buy", rng=np.random.default_rng(2))
    sells = _make_synthetic_trades(200, "sell", rng=np.random.default_rng(3))
    trades = pd.concat([buys, sells], ignore_index=True)

    result = calibration.calibrate_intensity(trades, min_trades_per_side=20, n_bins=15)

    assert result.bid.n_trades == 200
    assert result.ask.n_trades == 200
    assert result.bid.intensity.C > 0 and result.bid.intensity.D > 0
    assert result.ask.intensity.C > 0 and result.ask.intensity.D > 0

    mid = 2500.0
    bid_abs = result.bid.intensity.to_absolute(mid)
    ask_abs = result.ask.intensity.to_absolute(mid)
    assert bid_abs.C > 0 and bid_abs.D > 0
    assert ask_abs.C > 0 and ask_abs.D > 0


def test_calibrate_intensity_uses_one_shared_observation_window():
    """Regression: the rate denominator must be the full tape span, not each side's own
    first-to-last-trade span -- the latter is strictly shorter, inflates every fitted
    rate, and divides the two sides by different windows."""
    rng = np.random.default_rng(7)
    # asks span the full hour; bids are clustered into a narrow 6-minute burst inside it
    buys = _make_synthetic_trades(200, "buy", spread_seconds=3600, rng=rng)
    sells = _make_synthetic_trades(200, "sell", spread_seconds=360, rng=rng)
    trades = pd.concat([buys, sells], ignore_index=True)

    result = calibration.calibrate_intensity(trades, min_trades_per_side=20, n_bins=15)

    full_window = (trades["timestamp"].max() - trades["timestamp"].min()) / 1000.0
    assert result.bid.window_seconds == pytest.approx(full_window)
    assert result.ask.window_seconds == pytest.approx(full_window)


def test_calibrate_intensity_raises_when_too_sparse():
    buys = _make_synthetic_trades(5, "buy")
    sells = _make_synthetic_trades(5, "sell")
    trades = pd.concat([buys, sells], ignore_index=True)
    with pytest.raises(calibration.CalibrationError):
        calibration.calibrate_intensity(trades, min_trades_per_side=20)


@pytest.mark.network
def test_fetch_trades_live_smoke():
    """Loose smoke test against the real Deribit tape: shape/columns only, no value assertions."""
    import requests

    resp = requests.get(
        f"{calibration.DERIBIT_BASE_URL}/public/get_book_summary_by_currency",
        params={"currency": "BTC", "kind": "option"}, timeout=10,
    )
    resp.raise_for_status()
    instruments = sorted(resp.json()["result"], key=lambda x: -(x.get("volume") or 0))
    top_instrument = instruments[0]["instrument_name"]

    trades = calibration.fetch_trades(top_instrument, count=50)
    assert set(calibration.TRADE_COLUMNS) <= set(trades.columns)
    assert "iv" in trades.columns  # options report IV; this is always an option instrument


# --- add_fill_distance: USD-quoted instruments (e.g. BTC-PERPETUAL) --------

def test_add_fill_distance_usd_quoted_instrument_skips_index_conversion():
    trades = pd.DataFrame({
        "instrument_name": ["BTC-PERPETUAL"],
        "timestamp": [1000],
        "price": [65010.0],
        "mark_price": [65000.0],
        "index_price": [64995.0],  # deliberately different from mark, to prove it's unused here
        "direction": ["buy"],
        "amount": [1.0],
    })
    tagged = calibration.add_fill_distance(trades, price_in_underlying_units=False)
    assert tagged.loc[0, "eps_dollar"] == pytest.approx(10.0)
    assert tagged.loc[0, "eps_rel"] == pytest.approx(10.0 / 65000.0)
    assert tagged.loc[0, "side"] == "ask"


# --- realized_vol ------------------------------------------------------------

def test_realized_vol_recovers_known_volatility():
    rng = np.random.default_rng(4)
    sigma_true = 0.5
    n_minutes = 60 * 24 * 3  # 3 days of 1-minute bars
    dt = 1.0 / (365 * 24 * 60)
    log_returns = rng.normal(0.0, sigma_true * np.sqrt(dt), n_minutes)
    prices = 65000.0 * np.exp(np.cumsum(log_returns))
    timestamps = 1_700_000_000_000 + np.arange(n_minutes) * 60_000

    trades = pd.DataFrame({"timestamp": timestamps, "price": prices})
    sigma_hat = calibration.realized_vol(trades, freq="1min")

    assert sigma_hat == pytest.approx(sigma_true, rel=0.15)


def test_realized_vol_raises_on_too_little_data():
    trades = pd.DataFrame({"timestamp": [1000], "price": [65000.0]})
    with pytest.raises(calibration.CalibrationError):
        calibration.realized_vol(trades, freq="1min")


# --- implied_vol_of_vol -------------------------------------------------------

def test_implied_vol_of_vol_recovers_known_alpha():
    rng = np.random.default_rng(5)
    alpha_true = 0.05
    n_minutes = 60 * 24 * 3  # 3 days of 1-minute bars
    dt = 1.0 / (365 * 24 * 60)
    diffs = rng.normal(0.0, alpha_true * np.sqrt(dt), n_minutes)
    iv_path = 35.0 + np.cumsum(diffs) * 100.0  # percentage points, like Deribit's `iv` field
    timestamps = 1_700_000_000_000 + np.arange(n_minutes) * 60_000

    trades = pd.DataFrame({"timestamp": timestamps, "iv": iv_path})
    alpha_hat = calibration.implied_vol_of_vol(trades, freq="1min")

    assert alpha_hat == pytest.approx(alpha_true, rel=0.15)


def test_implied_vol_of_vol_raises_without_iv_column():
    trades = pd.DataFrame({"timestamp": [1000, 2000], "price": [100.0, 101.0]})
    with pytest.raises(calibration.CalibrationError):
        calibration.implied_vol_of_vol(trades)


def test_implied_vol_of_vol_raises_on_too_little_data():
    trades = pd.DataFrame({"timestamp": [1000], "iv": [35.0]})
    with pytest.raises(calibration.CalibrationError):
        calibration.implied_vol_of_vol(trades, freq="1min")

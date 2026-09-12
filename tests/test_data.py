from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import bs
import data


def _make_raw_row(instrument_name, bid_price, ask_price, underlying_price):
    return {
        "instrument_name": instrument_name,
        "bid_price": bid_price,
        "ask_price": ask_price,
        "mark_price": None if bid_price is None else 0.5 * (bid_price + ask_price),
        "underlying_price": underlying_price,
    }


def test_parse_instrument_name():
    expiry, strike, otype = data.parse_instrument_name("BTC-10AUG26-64000-P")
    assert strike == 64000.0
    assert otype == "P"
    assert expiry.year == 2026 and expiry.month == 8 and expiry.day == 10

    _, _, otype_c = data.parse_instrument_name("ETH-25DEC26-3000-C")
    assert otype_c == "C"


def test_years_to_expiry():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    expiry = now + timedelta(days=365)
    assert data.years_to_expiry(expiry, now) == pytest.approx(1.0, abs=1e-6)
    # expired instruments clip to zero, not negative
    assert data.years_to_expiry(now - timedelta(days=1), now) == 0.0


def _synthetic_raw(now, expiry_label, expiry_dt, forward, strikes, sigma, wide_strikes=()):
    """
    Build a raw book-summary DataFrame from BS-generated fair prices so a
    round-trip through build_chain should recover `sigma`. `wide_strikes`
    get an artificially blown-out bid-ask width to exercise the liquidity
    filter.
    """
    T = max(data.years_to_expiry(expiry_dt, now), 1e-6)  # avoid T=0 in the BS pricer used to build fixtures
    rows = []
    for K in strikes:
        otype = "C" if K >= forward else "P"
        fair = float(bs.price(forward, K, T, 0.0, 0.0, sigma, otype)) / forward
        width = 0.30 * fair if K in wide_strikes else 0.01 * fair
        bid = max(fair - width / 2, 1e-8)
        ask = fair + width / 2
        rows.append(_make_raw_row(f"BTC-{expiry_label}-{int(K)}-{otype}", bid, ask, forward))
    return pd.DataFrame(rows)


@pytest.fixture
def now():
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def expiry_label_and_dt(now):
    # Deribit expiries settle at 08:00 UTC; parse_instrument_name reconstructs
    # that hour from the label, so the fixture must match it exactly or the
    # round-trip T (and thus recovered vol) will be off by a few hours' worth.
    expiry_dt = (now + timedelta(days=30)).replace(hour=8)
    label = expiry_dt.strftime("%d%b%y").upper()
    return label, expiry_dt


def test_build_chain_round_trips_implied_vol(now, expiry_label_and_dt):
    label, expiry_dt = expiry_label_and_dt
    forward = 50000.0
    strikes = [40000, 45000, 48000, 50000, 52000, 55000, 60000]
    sigma = 0.65
    raw = _synthetic_raw(now, label, expiry_dt, forward, strikes, sigma)

    chain = data.build_chain(raw, max_width_frac=0.5, now=now)

    assert len(chain) == len(strikes)
    assert chain["mid_iv"].notna().all()
    assert (chain["mid_iv"] > 0).all()
    assert chain["mid_iv"].to_numpy() == pytest.approx(sigma, abs=1e-3)


def test_build_chain_filters_illiquid_quotes(now, expiry_label_and_dt):
    label, expiry_dt = expiry_label_and_dt
    forward = 50000.0
    strikes = [40000, 45000, 48000, 50000, 52000, 55000, 60000]
    wide = [40000, 60000]
    raw = _synthetic_raw(now, label, expiry_dt, forward, strikes, sigma=0.65, wide_strikes=wide)

    chain = data.build_chain(raw, max_width_frac=0.15, now=now)

    kept_strikes = set(chain["strike"])
    assert kept_strikes == set(strikes) - set(wide)


def test_build_chain_drops_one_sided_markets(now, expiry_label_and_dt):
    label, expiry_dt = expiry_label_and_dt
    forward = 50000.0
    raw = _synthetic_raw(now, label, expiry_dt, forward, [48000, 50000, 52000], sigma=0.65)
    # simulate a one-sided market (no bid) on one instrument
    raw.loc[0, "bid_price"] = None

    chain = data.build_chain(raw, max_width_frac=0.5, now=now)

    assert len(chain) == 2
    assert chain["mid_iv"].notna().all()


def test_build_chain_empty_input_returns_empty_frame():
    chain = data.build_chain(pd.DataFrame())
    assert chain.empty
    assert list(chain.columns) == data.CHAIN_COLUMNS


def test_build_chain_drops_expired_instruments(now):
    expiry_dt = now - timedelta(days=1)
    label = expiry_dt.strftime("%d%b%y").upper()
    raw = _synthetic_raw(now, label, expiry_dt, 50000.0, [48000, 50000], sigma=0.5)
    chain = data.build_chain(raw, max_width_frac=0.5, now=now)
    assert chain.empty


@pytest.mark.network
def test_live_deribit_smile_is_sane():
    """
    Step 3 acceptance criterion: implied vol vs. strike for one expiry
    should look like a real smile/skew, not noise -- checked here via
    convexity-ish sanity (vols stay in a plausible range and the curve
    isn't wildly non-monotonic point to point) rather than an exact shape
    assertion, since the live skew shape varies day to day.
    """
    try:
        raw = data.fetch_book_summary("BTC")
    except Exception as exc:
        pytest.skip(f"Deribit API unreachable: {exc}")

    chain = data.build_chain(raw)
    assert not chain.empty
    assert chain["mid_iv"].between(0.05, 3.0).all()

    biggest_expiry = chain["expiry"].value_counts().idxmax()
    slice_ = chain[chain["expiry"] == biggest_expiry].sort_values("strike")
    assert len(slice_) >= 5
    # no single strike-to-strike jump should dwarf the overall skew range
    step_changes = slice_["mid_iv"].diff().abs().dropna()
    assert step_changes.max() < 0.5 * (slice_["mid_iv"].max() - slice_["mid_iv"].min() + 0.05)

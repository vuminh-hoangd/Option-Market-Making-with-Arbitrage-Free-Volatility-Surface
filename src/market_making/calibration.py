"""
Fill-intensity and volatility calibration for `quoting.py`, all fit from
Deribit's public trade tape rather than guessed:

- `calibrate_intensity` -> the option's C, D (Theorems 1-5) and the stock's
  A, B (Theorems 2-3), via the fill-survival curve described below.
- `realized_vol` -> sigma_risk, the physical stock volatility driving
  inventory risk in Theorems 2-5.
- `implied_vol_of_vol` -> alpha, the volatility of the option's own implied
  vol (Schonbucher 1999), driving the Vega-risk term in Theorems 4-5.

The one parameter this module deliberately does NOT calibrate is gamma
(risk aversion): it is a preference, not a market observable, so there is
no curve to fit it against -- see `quoting.implied_gamma_from_risk_limit`
and `quoting.implied_gamma_from_option_risk_limit` for setting it from a
stated risk limit instead.

The approach, in one sentence: every historical trade reports both the
executed `price` and Deribit's `mark_price` (their fair-value mark) at that
instant, so `|price - mark_price|` is a direct per-trade estimate of how far
from fair value that fill happened -- and a marketable order that executes
at distance eps from the mark necessarily walked through, and filled, every
resting limit order between the mark and that price. So the empirical
*survival* rate (how many trades executed at distance >= eps, per unit time)
is exactly the fill rate a passive order resting at eps would have realized
-- lambda_o(eps), estimated straight from the tape, with no order-book
replay needed.

`direction` on each trade also splits the tape by side: a 'buy' aggressor
lifts a resting ask (informs the ask-side intensity), a 'sell' aggressor
hits a resting bid (informs the bid side) -- so this recovers separate
bid/ask intensities for `quoting.quote_option`'s `ask_intensity` parameter,
rather than assuming symmetric liquidity.

Why relative (not absolute-dollar) premiums: any single Deribit option
contract trades too rarely to fit a curve from (a liquid BTC option might
print a few dozen trades a day) -- `calibrate_intensity` is meant to be
called on trades pooled across many instruments. Pooling only makes sense if
premiums are on a comparable scale across strikes, so distances are
expressed as a fraction of the option's own mark price (`eps_rel = eps /
mark`) rather than raw dollars. `RelativeIntensity.to_absolute(mid)`
converts a fitted relative curve into the absolute-dollar
`quoting.LinearIntensity` a specific option's mid price needs. A single
highly-liquid instrument like BTC-PERPETUAL doesn't need pooling at all
(it alone prints far more trades per hour than the whole option chain
combined) -- pass a one-instrument trade tape straight to
`calibrate_intensity` in that case.

Caveat this doesn't address: option market activity is not a time-stationary
Poisson process (it clusters around news/funding events and varies by time
of day), and this calibrates a single lambda_o from whatever recent window
of trades the API returns -- treat it as a snapshot estimate, not a
constant.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from market_making.quoting import LinearIntensity

DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"

TRADE_COLUMNS = ["trade_id", "instrument_name", "timestamp", "price", "mark_price", "index_price",
                 "direction", "amount"]
# 'iv' is Deribit-reported implied vol at trade time (percentage points) -- only present for
# options, not futures/perpetuals -- so it's kept optional rather than a hard requirement.
OPTIONAL_TRADE_COLUMNS = ["iv"]


class CalibrationError(Exception):
    """Raised when there isn't enough (or well-behaved enough) trade data to fit an intensity."""


def fetch_trades(instrument_name, count=1000, include_old=True, timeout=10):
    """
    Pull up to `count` most recent trades for one instrument from Deribit's
    public tape. Includes `iv` (implied vol at trade time) when the venue
    reports it -- options do, futures/perpetuals don't.
    """
    resp = requests.get(
        f"{DERIBIT_BASE_URL}/public/get_last_trades_by_instrument",
        params={"instrument_name": instrument_name, "count": count, "include_old": include_old},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "result" not in payload:
        raise ValueError(f"unexpected Deribit response: {payload}")
    trades = payload["result"]["trades"]
    if not trades:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    df = pd.DataFrame(trades)
    cols = TRADE_COLUMNS + [c for c in OPTIONAL_TRADE_COLUMNS if c in df.columns]
    return df[cols]


def fetch_trades_multi(instrument_names, count_each=1000, include_old=True):
    """
    Pool trades across several instruments (Section: why relative premiums,
    above, explains why pooling is necessary in the first place). Instruments
    a request fails for are skipped with a warning rather than aborting the
    whole pool; instruments with zero trades are skipped silently (expected
    for illiquid strikes).
    """
    frames = []
    for inst in instrument_names:
        try:
            trades = fetch_trades(inst, count=count_each, include_old=include_old)
        except (requests.RequestException, ValueError) as exc:
            warnings.warn(f"skipping {inst}: {exc}")
            continue
        if not trades.empty:
            frames.append(trades)
    if not frames:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def add_fill_distance(trades, price_in_underlying_units=True):
    """
    Pure transform: given raw trades (as returned by `fetch_trades`/
    `fetch_trades_multi`), add `eps_dollar` (distance from mark at fill,
    dollar terms), `eps_rel` (`eps_dollar` as a fraction of the instrument's
    dollar mark price), and `side` ('ask' for a buy aggressor, 'bid' for a
    sell aggressor -- see module docstring). Distances are signed so that a
    fill on the "wrong" side of the mark (mark drifted between the last
    quote update and the match) clips to 0 rather than going negative.

    `price_in_underlying_units` controls the dollar conversion and must
    match how Deribit quotes the instrument: True (default) for options,
    where `price`/`mark_price` are fractions of the underlying and need
    multiplying by that trade's own `index_price` to get dollars; False for
    USD-quoted instruments like BTC-PERPETUAL, where `price`/`mark_price`
    are already in dollars and no conversion is needed.
    """
    out = trades.copy()
    if price_in_underlying_units:
        mid_dollar = out["mark_price"] * out["index_price"]
        price_dollar = out["price"] * out["index_price"]
    else:
        mid_dollar = out["mark_price"]
        price_dollar = out["price"]
    raw_dollar = price_dollar - mid_dollar
    signed = np.where(out["direction"] == "buy", raw_dollar, -raw_dollar)
    out["mid_dollar"] = mid_dollar
    out["price_dollar"] = price_dollar
    out["eps_dollar"] = np.maximum(signed, 0.0)
    out["eps_rel"] = out["eps_dollar"] / mid_dollar
    out["side"] = np.where(out["direction"] == "buy", "ask", "bid")
    return out


@dataclass(frozen=True)
class RelativeIntensity:
    """
    lambda_o(eta) = C - D*eta, where eta = eps/mark is the quoted premium as
    a *fraction* of the option's mark price -- the calibrated analogue of
    `quoting.LinearIntensity`, expressed in scale-free terms so it can be
    fit from trades pooled across strikes with very different price levels.
    """
    C: float
    D: float

    def to_absolute(self, mid: float) -> LinearIntensity:
        """
        Convert to an absolute-dollar `LinearIntensity` for an option
        currently priced at `mid`: eta = eps/mid, so
        lambda(eps) = C - D*(eps/mid) = C - (D/mid)*eps.
        """
        if mid <= 0:
            raise ValueError(f"mid must be positive, got {mid}")
        return LinearIntensity(C=self.C, D=self.D / mid)


def _survival_rate(eps_values, eps_grid, window_seconds):
    """empirical lambda_o(eps) at each grid point: count(eps_values >= eps) / window_seconds."""
    eps_values = np.asarray(eps_values, dtype=float)
    counts = np.array([(eps_values >= e).sum() for e in eps_grid], dtype=float)
    return counts / window_seconds


def _fit_relative_intensity(eps_values, window_seconds, n_bins=12, quantile_cutoff=0.9):
    """
    Build the empirical survival curve on `n_bins` points spanning
    [0, quantile(eps_values, quantile_cutoff)] -- the cutoff trims the far
    tail, where only a handful of trades ever land and the survival estimate
    is dominated by sampling noise -- then fits lambda(eps) = C - D*eps to
    that curve by ordinary least squares.
    """
    eps_values = np.asarray(eps_values, dtype=float)
    cutoff = np.quantile(eps_values, quantile_cutoff)
    if cutoff <= 0:
        raise CalibrationError(
            f"can't build a survival curve: {quantile_cutoff:.0%} of fills happened at eps=0 "
            "(need a wider quantile cutoff or more data)"
        )
    eps_grid = np.linspace(0.0, cutoff, n_bins)
    rate = _survival_rate(eps_values, eps_grid, window_seconds)

    design = np.column_stack([np.ones_like(eps_grid), -eps_grid])
    (C, D), *_ = np.linalg.lstsq(design, rate, rcond=None)

    if C <= 0 or D <= 0:
        raise CalibrationError(
            f"degenerate fit (C={C:.6g}, D={D:.6g}): the survival curve isn't "
            "decreasing enough to support a linear lambda_o -- likely too few trades"
        )

    fitted_rate = C - D * eps_grid
    return RelativeIntensity(C=float(C), D=float(D)), eps_grid, rate, fitted_rate


@dataclass(frozen=True)
class SideCalibration:
    """Diagnostics + fitted intensity for one side (bid or ask) of the tape."""
    intensity: RelativeIntensity
    eps_grid: np.ndarray
    empirical_rate: np.ndarray
    fitted_rate: np.ndarray
    n_trades: int
    window_seconds: float


@dataclass(frozen=True)
class IntensityCalibration:
    bid: SideCalibration
    ask: SideCalibration


def calibrate_intensity(trades, min_trades_per_side=20, n_bins=12, quantile_cutoff=0.9,
                         price_in_underlying_units=True) -> IntensityCalibration:
    """
    Fit separate bid- and ask-side `RelativeIntensity` curves from a pool of
    raw trades (typically `fetch_trades_multi`'s output). Raises
    `CalibrationError` if either side doesn't have `min_trades_per_side`
    fills -- a linear fit on a handful of trades is noise, not a policy.

    `price_in_underlying_units` is passed straight through to
    `add_fill_distance` -- set it False when calibrating a USD-quoted
    instrument (e.g. BTC-PERPETUAL) rather than an option.
    """
    tagged = add_fill_distance(trades, price_in_underlying_units=price_in_underlying_units)

    # One observation window shared by both sides -- the full span of the tape. Using each
    # side's OWN first-to-last-trade span instead would be a strictly shorter denominator
    # (that side's first and last prints sit inside the tape's span, not at its edges), which
    # biases every fitted rate upward and makes the two sides' rates non-comparable, since
    # each would be divided by a different window.
    window_seconds = float(tagged["timestamp"].max() - tagged["timestamp"].min()) / 1000.0
    if window_seconds <= 0:
        raise CalibrationError("all trades share one timestamp; can't estimate a rate")

    sides = {}
    for side in ("bid", "ask"):
        sub = tagged[tagged["side"] == side]
        if len(sub) < min_trades_per_side:
            raise CalibrationError(
                f"only {len(sub)} {side}-side trades (need >= {min_trades_per_side}); "
                "pool more instruments or widen the trade count"
            )

        intensity, eps_grid, empirical_rate, fitted_rate = _fit_relative_intensity(
            sub["eps_rel"].to_numpy(), window_seconds, n_bins=n_bins, quantile_cutoff=quantile_cutoff
        )
        sides[side] = SideCalibration(
            intensity=intensity, eps_grid=eps_grid, empirical_rate=empirical_rate,
            fitted_rate=fitted_rate, n_trades=len(sub), window_seconds=window_seconds,
        )

    return IntensityCalibration(bid=sides["bid"], ask=sides["ask"])


def realized_vol(trades, freq="1min", price_col="price", annualization_seconds=365 * 24 * 3600):
    """
    Annualized realized volatility from a trade tape's price series --
    `quoting.quote_theorem2`'s `sigma_risk`, the *physical* vol driving the
    dealer's inventory risk (distinct from an option's Black-Scholes implied
    vol). Trade-tick log-returns are dominated by bid-ask bounce noise, so
    prices are first resampled onto a regular `freq` grid (last trade per
    bin, forward-filled through empty bins) before taking log-returns --
    the standard fix for that microstructure bias.
    """
    ts = pd.to_datetime(trades["timestamp"].to_numpy(), unit="ms")
    px = pd.Series(trades[price_col].to_numpy(), index=ts).sort_index()
    bars = px.resample(freq).last().ffill()
    log_returns = np.log(bars / bars.shift(1)).dropna()
    if len(log_returns) < 2:
        raise CalibrationError(f"not enough {freq} bars ({len(log_returns)}) to estimate realized vol")

    bar_seconds = pd.Timedelta(freq).total_seconds()
    bars_per_year = annualization_seconds / bar_seconds
    return float(log_returns.std() * np.sqrt(bars_per_year))


def implied_vol_of_vol(trades, freq="1min", iv_col="iv", annualization_seconds=365 * 24 * 3600):
    """
    Estimate alpha -- Schonbucher (1999)'s `d(sigma_hat) = alpha * dW`, the
    volatility of the option's own implied volatility -- from real IV
    prints on an option trade tape (`iv`, only present on option trades;
    see `fetch_trades`). `quoting.quote_theorem4`/`quote_theorem5` need
    this for the Vega-risk term.

    Unlike `realized_vol` (log-returns of *price*, since price follows a
    geometric process), implied vol is modeled as a plain arithmetic
    Brownian motion here, so this differences the IV series directly rather
    than taking log differences -- consistent with `d(sigma_hat) =
    alpha*dW`, not a log-vol process. Deribit reports `iv` in percentage
    points (e.g. 33.07 meaning 33.07%), converted to a decimal fraction
    here. Same anti-microstructure-noise resampling as `realized_vol`: IV
    prints are first put on a regular `freq` grid before differencing.

    A single option contract's own IV prints are typically too sparse to
    fit this alone -- pool trades across several strikes at the *same*
    expiry first (via `fetch_trades_multi`), the same way
    `calibrate_intensity` pools for fill-intensity. Differences are always
    taken WITHIN each instrument's own IV series first (grouped by
    `instrument_name` when present), and only the resulting *differences*
    are pooled across instruments -- never differenced directly between two
    different strikes' raw IV levels. That distinction matters: two
    adjacent trades on different strikes mostly differ because of static
    skew (one strike is just priced at a structurally higher/lower vol than
    another), not because the vol *level* moved between them, and skew
    differences are typically an order of magnitude bigger than genuine
    tick-to-tick vol-of-vol -- pooling raw cross-instrument diffs was tried
    and inflated alpha by roughly 100x on live data before this fix.
    """
    if iv_col not in trades.columns:
        raise CalibrationError(f"trades has no {iv_col!r} column -- only option trades report IV")

    bar_seconds = pd.Timedelta(freq).total_seconds()
    groups = trades.groupby("instrument_name") if "instrument_name" in trades.columns else [(None, trades)]

    all_diffs = []
    for _, group in groups:
        ts = pd.to_datetime(group["timestamp"].to_numpy(), unit="ms")
        iv = pd.Series(group[iv_col].to_numpy(), index=ts).sort_index() / 100.0
        bars = iv.resample(freq).last().ffill()
        all_diffs.append(bars.diff().dropna())

    diffs = pd.concat(all_diffs) if all_diffs else pd.Series(dtype=float)
    if len(diffs) < 2:
        raise CalibrationError(f"not enough {freq} bars ({len(diffs)}) to estimate vol-of-vol")

    bars_per_year = annualization_seconds / bar_seconds
    return float(diffs.std() * np.sqrt(bars_per_year))

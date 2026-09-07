"""
Deribit option chain ingestion + smile extraction.

Pulls the live option book (best bid/ask, mark, underlying/forward price)
from Deribit's public REST API (no auth required), filters out illiquid
quotes, and converts survivors to implied vols -- producing the
`(strike, expiry, forward, mid_iv, weight)` table that every downstream
step (SVI fit, arbitrage checks, surface) consumes.

Deribit quotes options in units of the underlying (BTC/ETH), and
`underlying_price` on this endpoint is the forward price for that
instrument's expiry -- so pricing is done via Black-76 (S=forward, q=r),
under which the risk-free rate only sets the overall discount factor and
cancels out of the implied-vol solve. We default r=0, which is standard
practice for crypto vol surfaces since funding/carry is already embedded
in the forward.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import implied_vol as iv

DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"

CHAIN_COLUMNS = [
    "strike", "expiry", "T", "forward", "mid_iv", "weight",
    "option_type", "bid", "ask", "mid",
]


def fetch_book_summary(currency="BTC", timeout=10):
    """Pull the live book summary for every option on `currency` from Deribit."""
    resp = requests.get(
        f"{DERIBIT_BASE_URL}/public/get_book_summary_by_currency",
        params={"currency": currency, "kind": "option"},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "result" not in payload:
        raise ValueError(f"unexpected Deribit response: {payload}")
    return pd.DataFrame(payload["result"])


def parse_instrument_name(name):
    """'BTC-10AUG26-64000-P' -> (expiry_datetime, strike, option_type)."""
    _currency, expiry_str, strike_str, cp = name.split("-")
    expiry = datetime.strptime(expiry_str, "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
    return expiry, float(strike_str), "C" if cp == "C" else "P"


def years_to_expiry(expiry, now=None):
    now = now or datetime.now(timezone.utc)
    return max((expiry - now).total_seconds(), 0.0) / (365.0 * 24 * 3600)


def _empty_chain():
    return pd.DataFrame(columns=CHAIN_COLUMNS)


def build_chain(raw, max_width_frac=0.15, r=0.0, now=None):
    """
    Turn a raw Deribit book-summary DataFrame (as returned by
    `fetch_book_summary`) into the clean intermediate table:
    strike, expiry, T, forward, mid_iv, weight, plus bookkeeping columns.

    Liquidity filter: drops quotes without a two-sided market and quotes
    whose bid-ask width exceeds `max_width_frac` of the mid -- illiquid
    strikes shouldn't enter the fit.
    """
    if raw is None or raw.empty:
        return _empty_chain()

    now = now or datetime.now(timezone.utc)
    df = raw.copy()
    parsed = df["instrument_name"].apply(parse_instrument_name)
    df["expiry"] = parsed.apply(lambda t: t[0])
    df["strike"] = parsed.apply(lambda t: t[1])
    df["option_type"] = parsed.apply(lambda t: t[2])
    df["T"] = df["expiry"].apply(lambda e: years_to_expiry(e, now))
    df = df[df["T"] > 0]
    if df.empty:
        return _empty_chain()

    df["forward"] = df["underlying_price"].astype(float)
    df["bid"] = df["bid_price"].astype(float) * df["forward"]
    df["ask"] = df["ask_price"].astype(float) * df["forward"]
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["width"] = df["ask"] - df["bid"]

    has_two_sided_market = df["bid_price"].notna() & df["ask_price"].notna() & (df["bid"] > 0) & (df["ask"] > 0)
    tight_enough = df["width"] <= max_width_frac * df["mid"]
    df = df[has_two_sided_market & tight_enough].copy()
    if df.empty:
        return _empty_chain()

    df["mid_iv"] = iv.implied_vol_batch(
        df["mid"].to_numpy(),
        df["forward"].to_numpy(),
        df["strike"].to_numpy(),
        df["T"].to_numpy(),
        np.full(len(df), r),
        np.full(len(df), r),
        df["option_type"].to_numpy(),
    )

    # weight ~ 1/width^2: tighter markets carry more information about the smile
    df["weight"] = 1.0 / np.maximum(df["width"], 1e-8) ** 2

    df = df[np.isfinite(df["mid_iv"]) & (df["mid_iv"] > 0)]

    return (
        df[CHAIN_COLUMNS]
        .sort_values(["T", "strike"])
        .reset_index(drop=True)
    )


def fetch_chain(currency="BTC", max_width_frac=0.15, r=0.0):
    """Fetch the live chain from Deribit and run it through `build_chain`."""
    raw = fetch_book_summary(currency)
    return build_chain(raw, max_width_frac=max_width_frac, r=r)

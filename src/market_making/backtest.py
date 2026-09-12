"""
Passive-fill backtesting for the quoting policies in `quoting.py`.

Everything in `quoting.py` computes what to quote; nothing there says whether
quoting that way makes money. This module closes that loop: replay a recorded
trade tape, decide which of the dealer's resting quotes would have been hit,
and decompose the resulting P&L into exactly the two terms the paper's
objective is written over -- `Z_T` (spread capture from transactions) and
`I_T` (mark-to-market value of the inventory carried).

How fills are inferred. Deribit's public tape reports each trade's aggressor
`direction`, which is enough to test a resting quote without an order-book
replay:
- `direction == 'buy'`  -- the aggressor lifted an ask, so a resting ask at
  or below the trade price would have been taken. The dealer SELLS.
- `direction == 'sell'` -- the aggressor hit a bid, so a resting bid at or
  above the trade price would have been taken. The dealer BUYS.

Two things this cannot see, both of which make results OPTIMISTIC:
- Queue position. Being at a price that traded is necessary but not
  sufficient -- the depth resting ahead of you has to be consumed first.
  `queue_factor` scales every fill by a constant fraction to crudely stand in
  for that; it is a blunt instrument, not a queue model. Run any headline
  result at more than one `queue_factor` and report the range, because the
  gap between optimistic and pessimistic is the part of the P&L that is
  really just queue luck.
- Market impact / adverse selection beyond what the tape already shows: the
  dealer's own quotes are assumed not to change anyone else's behaviour.

Quotes are always set from the PREVIOUS trade's mark price, never the current
one, so a quote can never be informed by the trade that fills it. That
matters: Deribit's mark moves with the tape, so quoting off the current
trade's mark would leak the answer into the decision and quietly inflate
every result.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from market_making import calibration
from market_making import quoting


@dataclass(frozen=True)
class Quote:
    """A two-sided quote in absolute (dollar) prices."""
    bid: float
    ask: float


# A policy maps (mid, inventory) -> Quote. Anything richer (Greeks, time of day,
# calibrated intensities) is captured in the closure by the factories below.
Policy = Callable[[float, float], Quote]


@dataclass(frozen=True)
class BacktestResult:
    """
    Outcome of one replay. The P&L identity that always holds:

        total_pnl == spread_pnl + inventory_pnl - fees_paid

    `spread_pnl` is the paper's Z_T (edge captured versus the mid at the
    moment of each fill) and `inventory_pnl` is its I_T (what the carried
    position did afterwards). Reporting them separately is the whole point:
    a policy can look profitable on spread capture alone while bleeding it
    all back through inventory, and the paper's objective trades those two
    off explicitly.
    """
    fills: pd.DataFrame
    inventory_path: pd.DataFrame
    total_pnl: float
    spread_pnl: float
    inventory_pnl: float
    fees_paid: float
    n_bid_fills: int
    n_ask_fills: int
    final_inventory: float
    max_abs_inventory: float
    window_seconds: float
    n_trades_seen: int

    @property
    def n_fills(self) -> int:
        return self.n_bid_fills + self.n_ask_fills

    @property
    def realized_fill_rate(self) -> float:
        """
        Fills per second actually achieved -- directly comparable to the
        `LinearIntensity.rate(eps)` the calibration predicted at the same
        quoted distance. If these disagree badly, the intensity model is
        misspecified and any premium optimized on top of it is unreliable,
        regardless of what the P&L says.
        """
        return self.n_fills / self.window_seconds if self.window_seconds > 0 else float("nan")

    def mean_variance_objective(self, gamma: float) -> float:
        """
        The paper's own objective evaluated on realized data:
        E[Z_T] - gamma*Var[I_T]. `Var[I_T]` is taken as the variance of the
        marked inventory value along the replayed path -- the empirical
        analogue of the quantity Theorems 2-5 penalize.
        """
        inv_value = self.inventory_path["inventory_value"].to_numpy()
        return float(self.spread_pnl - gamma * np.var(inv_value))


def constant_spread_policy(half_spread: float) -> Policy:
    """
    Quote a fixed half-spread around the mid, ignoring inventory entirely.

    This is the honest baseline for every inventory-aware policy in this
    codebase: it is exactly Theorem 1's answer, and exactly the `gamma = 0`
    (risk-neutral) case of Theorems 2-5, which all collapse to a flat
    `C/2D` when the dealer stops caring about inventory variance. Beating
    it is the minimum bar for the tilting machinery earning its complexity.
    """
    def policy(mid: float, inventory: float) -> Quote:
        return Quote(bid=mid - half_spread, ask=mid + half_spread)
    return policy


def theorem4_policy(S, K, T_opt, r, q, sigma_bs, option_type,
                     option_intensity: quoting.LinearIntensity,
                     gamma: float, sigma_risk: float, alpha: float, horizon: float) -> Policy:
    """
    Section IV's inventory-tilting policy as a backtestable `Policy`.

    Theorems 4/5 are the natural fit for a single-instrument replay: they
    quote the option alone (Delta is assumed continuously hedged away), so
    unlike Theorems 2/3 there is no second stock leg whose fills would have
    to be simulated against a separate tape.

    The premiums come from `quoting.quote_theorem4`, but they are applied
    around the MARKET mid rather than the Black-Scholes mid the theorem
    prices internally -- the paper's `eps` are premiums around the observed
    mid price, and in a replay the observed mid is the tape's.
    """
    def policy(mid: float, inventory: float) -> Quote:
        quote = quoting.quote_theorem4(
            S, K, T_opt, r, q, sigma_bs, option_type, option_intensity,
            gamma=gamma, sigma_risk=sigma_risk, alpha=alpha, horizon=horizon, q_o=inventory,
        )
        return Quote(bid=mid - quote.bid_premium, ask=mid + quote.ask_premium)
    return policy


def simulate_fills(trades, policy: Policy, price_in_underlying_units=True, queue_factor=1.0,
                    quote_size=1.0, fee_per_contract=0.0,
                    max_abs_position: Optional[float] = None) -> BacktestResult:
    """
    Replay `trades` against `policy` and return the realized P&L decomposition.

    `queue_factor` in (0, 1] scales every fill to stand in for depth resting
    ahead of the dealer; 1.0 is the optimistic upper bound (see module
    docstring). `fee_per_contract` is charged on every filled unit -- pass
    the venue's real maker schedule, since at tight spreads fees can exceed
    the edge entirely. `max_abs_position`, if set, suppresses the side that
    would push inventory further past the limit (a hard risk stop the
    theorems don't model but any real desk has).
    """
    if not 0.0 < queue_factor <= 1.0:
        raise ValueError(f"queue_factor must be in (0, 1], got {queue_factor}")
    if quote_size <= 0:
        raise ValueError(f"quote_size must be positive, got {quote_size}")

    tagged = calibration.add_fill_distance(trades, price_in_underlying_units=price_in_underlying_units)
    tagged = tagged.sort_values("timestamp").reset_index(drop=True)
    if len(tagged) < 2:
        raise ValueError(f"need at least 2 trades to replay, got {len(tagged)}")

    ts = tagged["timestamp"].to_numpy(dtype=float)
    mid = tagged["mid_dollar"].to_numpy(dtype=float)
    px = tagged["price_dollar"].to_numpy(dtype=float)
    direction = tagged["direction"].to_numpy()
    amount = tagged["amount"].to_numpy(dtype=float)

    cash = 0.0
    inventory = 0.0
    spread_pnl = 0.0
    fees_paid = 0.0
    fills = []
    inv_path = []

    # Start at 1: each trade is tested against a quote set from the PREVIOUS trade's
    # mark, so no quote is ever informed by the trade that fills it.
    for i in range(1, len(tagged)):
        quote_mid = mid[i - 1]
        quote = policy(quote_mid, inventory)

        size = min(quote_size, amount[i] * queue_factor)
        filled_side = None

        if direction[i] == "buy" and quote.ask <= px[i]:
            # aggressor lifted an ask -> the dealer sells
            if max_abs_position is None or inventory - size >= -max_abs_position:
                cash += quote.ask * size
                inventory -= size
                spread_pnl += (quote.ask - quote_mid) * size
                fees_paid += fee_per_contract * size
                cash -= fee_per_contract * size
                filled_side = "ask"
        elif direction[i] == "sell" and quote.bid >= px[i]:
            # aggressor hit a bid -> the dealer buys
            if max_abs_position is None or inventory + size <= max_abs_position:
                cash -= quote.bid * size
                inventory += size
                spread_pnl += (quote_mid - quote.bid) * size
                fees_paid += fee_per_contract * size
                cash -= fee_per_contract * size
                filled_side = "bid"

        if filled_side is not None:
            fills.append({
                "timestamp": ts[i], "side": filled_side,
                "price": quote.ask if filled_side == "ask" else quote.bid,
                "size": size, "mid": quote_mid, "inventory_after": inventory,
            })

        inv_path.append({
            "timestamp": ts[i], "mid": mid[i], "inventory": inventory,
            "inventory_value": inventory * mid[i], "cash": cash,
            "equity": cash + inventory * mid[i],
        })

    inventory_path = pd.DataFrame(inv_path)
    fills_df = pd.DataFrame(fills) if fills else pd.DataFrame(
        columns=["timestamp", "side", "price", "size", "mid", "inventory_after"]
    )

    total_pnl = cash + inventory * mid[-1]
    # total = spread + inventory - fees, by construction (see BacktestResult)
    inventory_pnl = total_pnl - spread_pnl + fees_paid
    inv_abs = inventory_path["inventory"].abs()

    return BacktestResult(
        fills=fills_df,
        inventory_path=inventory_path,
        total_pnl=float(total_pnl),
        spread_pnl=float(spread_pnl),
        inventory_pnl=float(inventory_pnl),
        fees_paid=float(fees_paid),
        n_bid_fills=int((fills_df["side"] == "bid").sum()) if len(fills_df) else 0,
        n_ask_fills=int((fills_df["side"] == "ask").sum()) if len(fills_df) else 0,
        final_inventory=float(inventory),
        max_abs_inventory=float(inv_abs.max()) if len(inv_abs) else 0.0,
        window_seconds=float(ts[-1] - ts[0]) / 1000.0,
        n_trades_seen=len(tagged) - 1,
    )


def compare_policies(trades, policies: dict, **kwargs) -> pd.DataFrame:
    """
    Replay several named policies over the same tape and tabulate them
    side by side -- the actual benchmark. Pass at least one inventory-blind
    baseline (`constant_spread_policy`) alongside whatever is being
    evaluated, since "did the tilting earn its complexity" is only
    answerable against that control.
    """
    rows = []
    for name, policy in policies.items():
        result = simulate_fills(trades, policy, **kwargs)
        rows.append({
            "policy": name,
            "total_pnl": result.total_pnl,
            "spread_pnl": result.spread_pnl,
            "inventory_pnl": result.inventory_pnl,
            "fees": result.fees_paid,
            "n_fills": result.n_fills,
            "fill_rate_per_s": result.realized_fill_rate,
            "final_inventory": result.final_inventory,
            "max_abs_inventory": result.max_abs_inventory,
        })
    return pd.DataFrame(rows).set_index("policy")


def append_trade_log(path, instrument_names, count_each=1000):
    """
    Fetch the current tape for `instrument_names` and append it to a CSV at
    `path`, de-duplicated on `trade_id`.

    Deribit's public API only serves recent trades -- there is no queryable
    history -- so a backtest over anything longer than the last few hours
    requires recording the tape as it happens. Run this on a schedule (a
    cron job every few minutes is enough; the API returns far more than a
    few minutes' worth each call, so overlap plus de-duplication makes gaps
    unlikely) and the log accumulates the history the replay needs.

    Returns the number of genuinely new rows appended.
    """
    import os

    fresh = calibration.fetch_trades_multi(instrument_names, count_each=count_each)
    if fresh.empty:
        return 0

    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, fresh], ignore_index=True)
        n_before = len(existing)
    else:
        combined = fresh
        n_before = 0

    combined = combined.drop_duplicates(subset="trade_id").sort_values("timestamp")
    combined.to_csv(path, index=False)
    return len(combined) - n_before

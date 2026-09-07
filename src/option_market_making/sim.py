"""
Monte Carlo backtest of the quoting rule over the market-making horizon [0, T].

Discretization choices, all documented because they are the places where this
simulator is an approximation of the continuous-time model rather than a
transcription of it:

- Spot follows zero-drift GBM (dS_t/S_t = sigma dB_t): the model gives the
  market maker no view on drift, only on volatility, and the whole edge lives in
  the diffusion coefficient. By default it diffuses at the market maker's own
  `params.sigma` -- the model assumes the view is correct -- but `realised_sigma`
  decouples the two so a wrong view can be priced.
- Fills are drawn as POISSON COUNTS per step: n ~ Poisson(lambda dt) with
  lambda = lambda0 exp(-kappa delta*), lot size 1 (the paper's normalization).
  Because the quote -- and hence the intensity -- is held constant across a
  step, the Cox process restricted to that step is a homogeneous Poisson
  process, so this draw is exact given the discretized quote rather than an
  approximation of it. That is the one respect in which it beats a Bernoulli
  at-most-one-fill draw, which silently truncates multi-fill steps and so
  understates turnover whenever lambda dt is not small.
- Quotes are refreshed and the delta hedge is rebalanced on the same grid, at
  the start of each step, using the inventory carried in from the previous
  step. A fill landing inside a step is therefore hedged at the next grid
  point, not at the instant it happens -- which is what a real desk does.

Cash accounting follows the paper's cash process X_t directly. Its hedging
bracket [S_t Delta_t - S_0 Delta_0 - int Delta_u dS_u] differentiates to
S_t dDelta_t, so rebalancing enters the cash as `s * (Delta_new - Delta_prev)`:
increasing the short brings cash in. Portfolio value is then
V_t = X_t + q_t O_t - Delta_t S_t, and rebalancing leaves it unchanged, as it
must.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import bs_mm, psi1 as psi1_mod, riccati
from .bs_mm import VolSurfaceLike
from .edge import phi_edge_batch
from .quoting import MMParams


@dataclass(frozen=True)
class SimResult:
    """
    Full time series of a simulation.

    `t` has shape (n_steps+1,); every other array has shape
    (n_paths, n_steps+1) and is aligned to it. Quotes are recorded at every
    grid point including the terminal one (they are well defined there:
    psi1(T) = 0 and psi2(T) = -alpha), but fills are only simulated over the
    n_steps intervals, so the terminal quote never trades.

    `q`, `V`, `cash` and `hedge` are recorded AFTER the rebalance at that grid
    point and BEFORE that point's fills.
    """

    t: np.ndarray           # (n_steps+1,)   time grid, year-fractions
    S: np.ndarray           # (n_paths, n_steps+1) spot
    q: np.ndarray           # (n_paths, n_steps+1) option inventory, lots
    V: np.ndarray           # (n_paths, n_steps+1) portfolio value
    cash: np.ndarray        # (n_paths, n_steps+1) cash account X_t
    hedge: np.ndarray       # (n_paths, n_steps+1) shares shorted, Delta_t
    option_price: np.ndarray  # (n_paths, n_steps+1) mark-to-market O_t
    bid: np.ndarray         # (n_paths, n_steps+1)
    ask: np.ndarray         # (n_paths, n_steps+1)
    delta_b: np.ndarray     # (n_paths, n_steps+1) optimal bid spread
    delta_a: np.ndarray     # (n_paths, n_steps+1) optimal ask spread
    n_bid_fills: np.ndarray  # (n_paths,) lots bought over the horizon
    n_ask_fills: np.ndarray  # (n_paths,) lots sold over the horizon
    sigma_imp: float        # the frozen market-implied vol for this contract

    @property
    def terminal_value(self) -> np.ndarray:
        """V_T per path."""
        return self.V[:, -1]


def simulate(
    surface: VolSurfaceLike,
    K: float,
    tau: float,
    T: float,
    params: MMParams,
    S0: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int | None = None,
    constant_spreads: tuple[float, float] | None = None,
    realised_sigma: float | None = None,
    spread_fn: "Callable[[float, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None" = None,
) -> SimResult:
    """
    Run the quoting rule over [0, T] on `n_paths` simulated spot paths.

    `surface` is queried exactly once, for sigma_imp(K, tau), and the result is
    frozen for the whole run -- the model assumes the implied surface does not
    move over the market-making horizon.

    `constant_spreads=(delta_b, delta_a)` replaces eq. 11 with a fixed quote
    width, giving the paper's "zero-intelligence" benchmark: a market maker who
    quotes the same distance off the mark whatever their inventory or their
    volatility view. Everything else -- fills, hedging, cash -- runs through the
    identical accounting, so the two are comparable P&L for P&L. Leave it None
    for the optimal rule.

    `spread_fn(t, spots, inventory) -> (delta_b, delta_a)` replaces eq. 11 with an
    arbitrary quoting rule, so a policy that has no closed form can be scored on
    the same footing -- in particular the EXACT policy read off the finite-
    difference HJB solution (`pde.PDESolution.spreads_batch`), which is how the
    cost of the second-order approximation gets measured in objective terms
    rather than in spread error. Mutually exclusive with `constant_spreads`.

    `realised_sigma` sets the volatility the SPOT actually diffuses at, leaving
    `params.sigma` as the volatility the market maker BELIEVES and quotes on.
    They coincide by default, which is the model's own assumption -- the maker
    is right. Setting them apart is how you ask what the strategy costs when the
    view is wrong, which the model itself says nothing about.

    Cost note: psi1's edge term depends on the spot, so it is evaluated with
    `edge.phi_edge_batch` -- one shared Gauss-Legendre grid per time step,
    broadcast across every path -- rather than one adaptive quadrature per
    (path, step). Its order-flow corrections and psi2 depend only on t and are
    computed once per step. Runtime is therefore linear in n_steps and nearly
    free in n_paths.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths}")
    if S0 <= 0.0:
        raise ValueError(f"initial spot must be positive, got S0={S0}")
    if T <= 0.0:
        raise ValueError(f"horizon must be positive, got T={T}")
    if T >= tau:
        raise ValueError(
            f"the market-making horizon must end strictly before expiry, got T={T}, tau={tau}"
        )
    if constant_spreads is not None and len(constant_spreads) != 2:
        raise ValueError(
            f"constant_spreads must be a (delta_b, delta_a) pair, got {constant_spreads!r}"
        )
    if constant_spreads is not None and spread_fn is not None:
        raise ValueError("pass at most one of constant_spreads and spread_fn")

    path_sigma = params.sigma if realised_sigma is None else float(realised_sigma)
    if path_sigma <= 0.0:
        raise ValueError(f"realised_sigma must be positive, got {realised_sigma}")

    sigma_imp_val = bs_mm.sigma_imp(surface, K, tau)

    # Two INDEPENDENT streams. The spot path must not depend on the strategy:
    # `rng.poisson` draws a variable amount of underlying randomness depending on
    # its rate, so sharing one stream would silently desynchronise the paths
    # between two strategies at the same seed and destroy any paired comparison.
    path_rng, fill_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    dt = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1)

    shape = (n_paths, n_steps + 1)
    S = np.empty(shape)
    q = np.zeros(shape)
    V = np.zeros(shape)
    cash = np.zeros(shape)
    hedge = np.zeros(shape)
    option_price = np.zeros(shape)
    bid = np.zeros(shape)
    ask = np.zeros(shape)
    delta_b = np.zeros(shape)
    delta_a = np.zeros(shape)

    S[:, 0] = S0
    inventory = np.zeros(n_paths)
    cash_now = np.zeros(n_paths)
    hedge_prev = np.zeros(n_paths)
    n_bid_fills = np.zeros(n_paths, dtype=int)
    n_ask_fills = np.zeros(n_paths, dtype=int)

    # GBM increments: zero drift, at the vol the spot actually realises. Drawn
    # up front in one shot, so two strategies at the same seed run on identical
    # spot paths -- what a paired comparison of the objective needs.
    log_drift = -0.5 * path_sigma ** 2 * dt
    diffusion = path_sigma * np.sqrt(dt)
    increments = log_drift + diffusion * path_rng.standard_normal((n_paths, n_steps))
    S[:, 1:] = S0 * np.exp(np.cumsum(increments, axis=1))

    for i, t in enumerate(t_grid):
        spot = S[:, i]
        tau_minus_t = tau - t

        # Time-only coefficients, shared by every path at this step.
        psi2_val = riccati.psi2(float(t), T, params)
        corrections = psi1_mod.flow_corrections(float(t), T, params)

        prices = bs_mm.call_price_array(spot, K, tau_minus_t, sigma_imp_val)
        deltas = bs_mm.delta_array(spot, K, tau_minus_t, sigma_imp_val)

        # Rebalance the hedge to the inventory carried in from the last step.
        # Value-neutral at the instant it happens: cash in exactly offsets the
        # larger short position.
        hedge_now = inventory * deltas
        cash_now = cash_now + spot * (hedge_now - hedge_prev)
        hedge_prev = hedge_now

        option_price[:, i] = prices
        q[:, i] = inventory
        cash[:, i] = cash_now
        hedge[:, i] = hedge_now
        V[:, i] = cash_now + inventory * prices - hedge_now * spot

        # Quote. psi1 = spot-dependent edge + shared order-flow corrections.
        # Eq. 11 inlined in vector form; `quoting.spreads_from_coefficients` is
        # the scalar statement of the same two lines and the tests pin them together.
        if constant_spreads is None and spread_fn is None:
            psi1_vals = (
                phi_edge_batch(float(t), spot, K, tau, T, sigma_imp_val, params) + corrections
            )
            db = 1.0 / params.kappa_b - psi1_vals - (2.0 * inventory + 1.0) * psi2_val
            da = 1.0 / params.kappa_a + psi1_vals + (2.0 * inventory - 1.0) * psi2_val
        elif constant_spreads is not None:
            db = np.full(n_paths, float(constant_spreads[0]))
            da = np.full(n_paths, float(constant_spreads[1]))
        else:
            db, da = spread_fn(float(t), spot, inventory.astype(int))
            db = np.asarray(db, dtype=float)
            da = np.asarray(da, dtype=float)

        delta_b[:, i] = db
        delta_a[:, i] = da
        bid[:, i] = prices - db
        ask[:, i] = prices + da

        if i == n_steps:
            break  # terminal grid point: quotes recorded, but nothing trades

        # Fills over [t, t+dt), exact given the step-constant quote.
        # Spreads can go negative under extreme inventory or a large edge, which
        # drives the intensity above lambda0 -- that is the model's own
        # behaviour (quoting through the mark to shed risk), so it is not
        # clipped here.
        lambda_b = params.lambda0_b * np.exp(-params.kappa_b * db)
        lambda_a = params.lambda0_a * np.exp(-params.kappa_a * da)

        bought = fill_rng.poisson(lambda_b * dt)
        sold = fill_rng.poisson(lambda_a * dt)

        # Bid hit: the market maker buys lots at its bid. Ask lifted: sells at its ask.
        cash_now = cash_now - bought * bid[:, i] + sold * ask[:, i]
        inventory = inventory + bought - sold
        n_bid_fills += bought
        n_ask_fills += sold

    return SimResult(
        t=t_grid,
        S=S,
        q=q,
        V=V,
        cash=cash,
        hedge=hedge,
        option_price=option_price,
        bid=bid,
        ask=ask,
        delta_b=delta_b,
        delta_a=delta_a,
        n_bid_fills=n_bid_fills,
        n_ask_fills=n_ask_fills,
        sigma_imp=sigma_imp_val,
    )


def objective(result: SimResult, params: MMParams) -> np.ndarray:
    """
    The paper's objective (eq. 2), realised per path:

        V_T - beta * int_0^T Q_u^2 du - alpha * Q_T^2

    This -- not mean V_T -- is the quantity eq. 11 maximises. A rule that gives
    up P&L to hold inventory closer to flat is supposed to look worse on V_T
    alone; the objective is where that trade is priced.

    The running integral is EXACT for a simulated path, not a quadrature
    approximation: inventory only changes at grid points, so Q_u is piecewise
    constant at `q[:, i]` over each [t_i, t_{i+1}) and the left Riemann sum is
    the integral. `q[:, -1]` is Q_T and contributes only to the terminal term.

    Returns one value per path.
    """
    dt = float(result.t[1] - result.t[0])
    running = (result.q[:, :-1] ** 2).sum(axis=1) * dt
    return result.terminal_value - params.beta * running - params.alpha * result.q[:, -1] ** 2


def objective_summary(runs: dict[str, SimResult], params: MMParams) -> "pd.DataFrame":  # noqa: F821
    """
    Decompose `objective` into its three terms for several strategies at once.

    Requires pandas, which the rest of this package does not -- imported locally
    so `sim` stays importable without it.
    """
    import pandas as pd

    dt_of = lambda r: float(r.t[1] - r.t[0])
    return pd.DataFrame([
        {
            "strategy": name,
            "mean V_T": r.terminal_value.mean(),
            "running penalty/fee": params.beta * (r.q[:, :-1] ** 2).sum(axis=1).mean() * dt_of(r),
            "terminal penalty/fee": params.alpha * (r.q[:, -1] ** 2).mean(),
            "OBJECTIVE (value after fees)": objective(r, params).mean(),
            "objective sd": objective(r, params).std(),
        }
        for name, r in runs.items()
    ])

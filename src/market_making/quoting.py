"""
Optimal quoting policies -- Stoikov & Saglam (2009), all three of the paper's
quoting regimes:

- Section II, Theorem 1 (`quote_option`, `quote_from_surface`): complete
  market. The underlying trades continuously with no transaction costs, so
  the dealer removes all inventory risk by delta-hedging at every instant,
  and optimal bid/ask premiums depend only on the option's own liquidity --
  never on inventory.
- Section III, Theorems 2-3 (`quote_theorem2`, `quote_theorem3` +
  `compute_tilt_slope_path`): illiquid underlying. The dealer must also
  quote the stock, so both legs tilt with net Delta (`q_s + q_o*Delta`).
  Theorem 3 is the multi-period recursion for how that tilt sensitivity
  itself evolves through a trading day; Theorem 2 is its last-session case.
- Section IV, Theorems 4-5 (`quote_theorem4`, `quote_theorem5` +
  `compute_tilt_slope_path_theorem5`): stochastic volatility. The dealer
  delta-hedges continuously again (no stock quoting, like Theorem 1), but
  now faces residual Gamma risk (discrete hedging) and Vega risk (the
  option's own implied vol moving) -- so the option leg alone tilts with
  net *Gamma* exposure (`q_o`) instead of net Delta. Theorem 5 is the
  multi-period recursion; Theorem 4 is its last-session case.

Reads Greeks and vols straight off Phase 1 (`bs.py`, `surface.py`): nothing
here refits anything, it only calls `bs.price`/`bs.delta`/`bs.gamma`/
`bs.vega`/`bs.theta` and, for the live-surface path, a fitted surface's
`implied_vol(K, T)`. Liquidity (`LinearIntensity`) and volatility inputs are
calibrated in `calibration.py`; `gamma` (risk aversion) is a preference
parameter with no market curve to fit it against -- see
`implied_gamma_from_risk_limit`/`implied_gamma_from_option_risk_limit` for
the practical way to set it instead of guessing.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize_scalar

import bs


@dataclass(frozen=True)
class LinearIntensity:
    """
    Poisson order-arrival intensity, linear in the quoted premium (paper eq.
    4): lambda(eps) = C - D*eps for 0 <= eps < C/D, 0 otherwise.

    C is the fill-arrival rate right at the mid (eps=0); D controls how fast
    the fill rate decays as the quote moves away from mid. C/D is therefore
    the maximum survivable premium -- fill probability hits zero beyond it.

    The paper picks a decreasing linear lambda_o both because it matches the
    empirical fact that quotes closer to mid fill more often, and because it
    is the simplest function satisfying the second-order condition for a
    well-defined revenue maximizer (eps*lambda''(eps) + 2*lambda'(eps) <= 0).
    Fit C and D from real fill data with `calibration.calibrate_intensity`
    rather than guessing them; see `optimal_premium_numeric` for the path to
    a non-linear, empirically-fit lambda_o.
    """
    C: float
    D: float

    def __post_init__(self):
        if self.C <= 0 or self.D <= 0:
            raise ValueError(f"C and D must be positive, got C={self.C}, D={self.D}")

    @property
    def max_premium(self) -> float:
        """C/D: the premium beyond which fill probability is zero."""
        return self.C / self.D

    def _in_domain(self, eps):
        """Paper eq (4)'s support: 0 <= eps < C/D, zero intensity outside it."""
        return (eps >= 0.0) & (eps < self.max_premium)

    def rate(self, eps):
        """lambda_o(eps), vectorized."""
        eps = np.asarray(eps, dtype=float)
        return np.where(self._in_domain(eps), self.C - self.D * eps, 0.0)

    def rate_prime(self, eps):
        """lambda_o'(eps), vectorized."""
        eps = np.asarray(eps, dtype=float)
        return np.where(self._in_domain(eps), -self.D, 0.0)


def optimal_premium(intensity: LinearIntensity) -> float:
    """
    Theorem 1's optimal premium: the implicit equation
        eps* = -lambda_o(eps*) / lambda_o'(eps*)
    equivalently the first-order condition on the revenue function
    R(eps) = eps*lambda_o(eps) (paper's Appendix A: lambda_o(eps) +
    eps*lambda_o'(eps) = 0).

    For the linear intensity this has the closed form eps* = C/(2D) -- the
    revenue-maximizing premium, independent of inventory. This is exactly
    the gamma=0 (risk-neutral) case of the paper's Theorem 2/3 as well:
    with continuous hedging available, the dealer's mean-variance problem
    collapses to pure revenue maximization regardless of risk aversion
    gamma, because delta-hedging removes the variance term entirely.
    """
    return intensity.C / (2.0 * intensity.D)


def optimal_premium_numeric(lambda_fn: Callable[[float], float], eps_bounds=(0.0, 10.0)) -> float:
    """
    General-lambda_o version of `optimal_premium`: numerically maximizes the
    revenue function R(eps) = eps*lambda_fn(eps) over eps_bounds instead of
    assuming a linear intensity. Use this once lambda_o is fit from real
    fill/trade data rather than assumed linear -- `optimal_premium`'s
    closed form no longer applies, but Theorem 1's FOC still does.

    Pass `eps_bounds` explicitly: the default upper bound is arbitrary, and
    a premium scale bigger than it (a real BTC option's `max_premium` runs
    to hundreds of dollars) would otherwise put the true maximizer outside
    the search range. `(0.0, intensity.max_premium)` is the natural choice
    for a `LinearIntensity`. A maximizer landing on the upper bound means
    the revenue function was still increasing there, so the result would be
    a truncation artifact rather than the FOC's solution -- that raises
    instead of returning a misleading number.
    """
    result = minimize_scalar(
        lambda eps: -eps * lambda_fn(eps), bounds=eps_bounds, method="bounded"
    )
    if not result.success:
        raise RuntimeError(f"revenue maximization failed: {result.message}")

    lo, hi = eps_bounds
    if hi - float(result.x) <= 1e-6 * max(abs(hi), 1.0):
        raise RuntimeError(
            f"revenue maximizer hit the upper bound eps={hi}: R(eps) = eps*lambda(eps) is still "
            "increasing there, so this is a truncation artifact, not the first-order condition's "
            "solution -- widen eps_bounds (for a LinearIntensity, use (0.0, intensity.max_premium))"
        )
    return float(result.x)


@dataclass(frozen=True)
class OptionQuote:
    """One dealer quote on a single option lot, per Theorem 1."""
    mid: float
    bid_premium: float
    ask_premium: float
    bid_price: float
    ask_price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    hedge_shares: float    # shares of underlying to hold: -delta * q_o
    hedge_notional: float  # dollar amount: S * hedge_shares (Theorem 1's pi_t)


def quote_option(S, K, T, r, q, sigma, option_type, intensity: LinearIntensity,
                  ask_intensity: Optional[LinearIntensity] = None, q_o: float = 0.0) -> OptionQuote:
    """
    Theorem 1's optimal bid/ask premiums and delta-hedge for one option,
    given current inventory `q_o` (in option lots).

    `intensity` is lambda_o for the bid side; `ask_intensity` lets the ask
    side use a different lambda_o (real bid/ask liquidity is rarely
    perfectly symmetric) and defaults to the same `intensity` if omitted,
    matching the paper's base case of a single lambda_o for both sides.

    Because the underlying is perfectly liquid here, `bid_price`/`ask_price`
    do NOT depend on `q_o` -- that's Theorem 1's central claim. Only the
    hedge (`hedge_shares`, `hedge_notional` = pi_t = -S*Delta*q_o) responds
    to inventory. `q_o` is accepted purely to compute that hedge.
    """
    ask_intensity = ask_intensity or intensity

    mid = float(bs.price(S, K, T, r, q, sigma, option_type))
    d = float(bs.delta(S, K, T, r, q, sigma, option_type))
    g = float(bs.gamma(S, K, T, r, q, sigma, option_type))
    v = float(bs.vega(S, K, T, r, q, sigma, option_type))
    th = float(bs.theta(S, K, T, r, q, sigma, option_type))

    eps_bid = optimal_premium(intensity)
    eps_ask = optimal_premium(ask_intensity)

    hedge_shares = -d * q_o
    hedge_notional = S * hedge_shares

    return OptionQuote(
        mid=mid,
        bid_premium=eps_bid,
        ask_premium=eps_ask,
        bid_price=mid - eps_bid,
        ask_price=mid + eps_ask,
        delta=d, gamma=g, vega=v, theta=th,
        hedge_shares=hedge_shares,
        hedge_notional=hedge_notional,
    )


def quote_from_surface(surface, S, K, T, r, q, option_type, intensity: LinearIntensity,
                        ask_intensity: Optional[LinearIntensity] = None, q_o: float = 0.0) -> OptionQuote:
    """
    Same as `quote_option`, but reads sigma off a fitted Phase 1 surface
    (`surface.VolSurface` or `surface.SSVIVolSurface`) instead of taking it
    as an argument -- the Phase 1 -> Phase 2 hookup: implied vol comes from
    `surface.implied_vol(K, T)`, everything else is unchanged.
    """
    sigma = surface.implied_vol(K, T)
    return quote_option(S, K, T, r, q, sigma, option_type, intensity, ask_intensity, q_o)


@dataclass(frozen=True)
class Theorem2Quote:
    """One dealer quote on both the stock and the option, per Theorem 2."""
    net_delta: float
    mid_option: float
    delta: float
    ask_stock_premium: float
    bid_stock_premium: float
    ask_stock_price: float
    bid_stock_price: float
    ask_option_premium: float
    bid_option_premium: float
    ask_option_price: float
    bid_option_price: float


def _tilted_quote(S, mid, delta, stock_intensity: LinearIntensity, option_intensity: LinearIntensity,
                   m, q_s, q_o) -> Theorem2Quote:
    """
    Shared closed-form policy behind both `quote_theorem2` and
    `quote_theorem3`. `m` is Theorem 3's tilt-slope coefficient (paper's
    m_i), which generalizes Theorem 2's one-period `-risk` term: passing
    `m = -gamma * sigma_risk**2 * horizon * S**2` (Theorem 2's terminal
    condition, m_n) reproduces `quote_theorem2` exactly -- exactly the
    paper's own claim that Theorem 2 is Theorem 3's last-session case.
    """
    net_delta = q_s + q_o * delta

    ask_stock = float(np.clip(
        stock_intensity.max_premium / 2.0 + m * (net_delta - 0.5), 0.0, stock_intensity.max_premium
    ))
    bid_stock = float(np.clip(
        stock_intensity.max_premium / 2.0 - m * (net_delta + 0.5), 0.0, stock_intensity.max_premium
    ))
    ask_option = float(np.clip(
        option_intensity.max_premium / 2.0 + m * delta * (net_delta - 0.5 * delta),
        0.0, option_intensity.max_premium
    ))
    bid_option = float(np.clip(
        option_intensity.max_premium / 2.0 - m * delta * (net_delta + 0.5 * delta),
        0.0, option_intensity.max_premium
    ))

    return Theorem2Quote(
        net_delta=net_delta, mid_option=mid, delta=delta,
        ask_stock_premium=ask_stock, bid_stock_premium=bid_stock,
        ask_stock_price=S + ask_stock, bid_stock_price=S - bid_stock,
        ask_option_premium=ask_option, bid_option_premium=bid_option,
        ask_option_price=mid + ask_option, bid_option_price=mid - bid_option,
    )


def quote_theorem2(S, K, T_opt, r, q, sigma_bs, option_type,
                    stock_intensity: LinearIntensity, option_intensity: LinearIntensity,
                    gamma: float, sigma_risk: float, horizon: float,
                    q_s: float = 0.0, q_o: float = 0.0) -> Theorem2Quote:
    """
    Theorem 2 (paper Section III / Appendix B): the one-period optimal
    quoting policy once the underlying is illiquid enough that the dealer
    must quote the stock too, rather than delta-hedging costlessly like
    Theorem 1. Both stock and option premiums now tilt with the dealer's
    *net Delta* (`q_s + q_o*Delta`) instead of staying flat regardless of
    inventory -- the paper's central new result versus the complete-market
    case, and the reason this needs `q_s`/`q_o` as real inputs rather than
    just a hedge-notional afterthought.

    `sigma_risk` is the *physical* (not implied) volatility of the stock
    driving the risk term `gamma * sigma^2 * (T-t_n) * S^2` -- typically a
    realized vol estimated from recent trade history (see
    `calibration.realized_vol`), which need not equal `sigma_bs` (the
    implied vol used here only to price/Greek the option). `gamma` itself
    is a preference parameter with no fill-rate curve to fit it against --
    see `implied_gamma_from_risk_limit` for the practical way to set it.

    `horizon` is `T - t_n`: the remaining time (in years) until the
    terminal risk resolves. In the paper this is literally "until the
    market closes for the day, plus the overnight gap"; on a 24/7 venue
    there's no such close, so treat `horizon` as a chosen risk-review
    window (e.g. "re-quote and re-hedge every few hours", with `horizon`
    set to that window) rather than a literal session boundary.
    """
    mid = float(bs.price(S, K, T_opt, r, q, sigma_bs, option_type))
    delta = float(bs.delta(S, K, T_opt, r, q, sigma_bs, option_type))
    m = -gamma * sigma_risk ** 2 * horizon * S ** 2
    return _tilted_quote(S, mid, delta, stock_intensity, option_intensity, m, q_s, q_o)


def implied_gamma_from_spread(intensity: LinearIntensity, real_spread: float,
                               sigma_risk: float, S: float, horizon: float, leg_scale: float = 1.0) -> float:
    """
    Invert Theorem 2's flat-inventory (net_delta=0) spread formula against a
    REAL observed market spread to back out gamma, holding `horizon` fixed
    (gamma and horizon only ever appear as the product gamma*horizon in this
    model, so fixing one is required to identify the other from spread
    data alone). At flat inventory, this leg's total quoted spread is

        real_spread == intensity.max_premium + gamma*sigma_risk**2*horizon*S**2*leg_scale**2

    `leg_scale` is 1.0 for the stock leg or the option's Delta for the
    option leg (matching the extra Delta factor in Theorem 2's option-side
    risk term).

    CAN return a negative value -- and that's a real, meaningful result, not
    a bug: it means `intensity.max_premium` (the risk-*neutral*, gamma=0
    spread implied by the fitted fill-intensity curve alone) is already
    wider than the real observed spread, so no non-negative gamma can
    reconcile the two. In practice this happens because the fitted
    intensity is a single monopolistic dealer's revenue-maximizing spread,
    fit from trade prints that walk through real depth, while a real
    market's top-of-book reflects many competing dealers -- there's no
    reason those two numbers have to agree. Prefer
    `implied_gamma_from_risk_limit` when this happens.
    """
    return (real_spread - intensity.max_premium) / (sigma_risk ** 2 * horizon * S ** 2 * leg_scale ** 2)


def implied_gamma_from_risk_limit(intensity: LinearIntensity, net_delta_limit: float,
                                   sigma_risk: float, S: float, horizon: float, leg_scale: float = 1.0) -> float:
    """
    Back out gamma from a stated inventory risk limit instead of a real
    spread: the net-Delta level at which this leg's "wrong-side" premium
    saturates to its floor (0 -- fully aggressive, one-sided quoting).
    Unlike A/B/C/D, gamma is a genuine preference parameter with no
    fill-rate curve to fit it against, so the standard practical way to set
    it is to state a risk limit ("how much net Delta before I go maximally
    aggressive on one side") and solve for the gamma consistent with it,
    rather than guessing gamma directly.

    Unlike `implied_gamma_from_spread`, this can't hit the same sign problem
    for the stock leg, since it targets the model's own saturation point
    rather than an external competitive benchmark the model may never reach.

    `leg_scale` is 1.0 for the stock leg or the option's Delta for the
    option leg. Note this enters DIFFERENTLY here than in
    `implied_gamma_from_spread`: Theorem 2's tilt term has the form
    `risk * L * (net_delta - L/2)` (L=1 stock, L=Delta option), so the
    saturation point depends on `L` both inside and outside the bracket --
    it is NOT simply `risk * L^2 * (net_delta - 1/2)`. (At flat inventory
    the bracket collapses and the spread formula genuinely does reduce to an
    `L^2` scaling, which is why `implied_gamma_from_spread` differs.) For
    L=1 this reduces to `(max_premium/2)/(net_delta_limit - 0.5)`, the stock
    leg's usual form.
    """
    denominator = leg_scale * (net_delta_limit - 0.5 * leg_scale)
    if denominator <= 0:
        raise ValueError(
            f"unreachable risk limit: leg_scale*(net_delta_limit - leg_scale/2) must be positive, "
            f"got {denominator} for leg_scale={leg_scale}, net_delta_limit={net_delta_limit}. "
            "For a positive-Delta leg this means net_delta_limit must exceed leg_scale/2; a "
            "negative-Delta leg (e.g. a put) saturates its bid, not its ask, at positive net Delta."
        )
    risk = (intensity.max_premium / 2.0) / denominator
    return risk / (sigma_risk ** 2 * horizon * S ** 2)


def compute_tilt_slope_path(n_sessions, session_dt, terminal_horizon, gamma, sigma_risk, S, delta,
                             stock_intensity: LinearIntensity, option_intensity: LinearIntensity) -> np.ndarray:
    """
    Theorem 3's backward recursion (paper Section III.B / Appendix C) for
    the tilt-slope coefficient m_i, i = 0..n_sessions. `m[n_sessions]` is
    the terminal condition -- exactly Theorem 2's `m` (see `_tilted_quote`)
    for a one-period horizon of `terminal_horizon` -- and each earlier
    `m[i]` is computed backward via

        m_i = m_{i+1} + session_dt * (2*B + 2*D*Delta^2) * m_{i+1}^2

    where B, D are the stock/option intensities' `D` parameters. `delta`
    (the option's Delta) and `S` are held fixed across the whole recursion,
    matching the paper's own multi-period simplification (Appendix B: "the
    stock price does not move during the day... the only inventory risk
    comes from the possibility of an overnight move") -- sessions only
    differ through how much time remains before that overnight-equivalent
    terminal risk, not through any assumed intraday price move.

    The `2*B + 2*D*delta^2` coefficient assumes the paper's auxiliary
    indicators I_i, J_i (whether that session's own optimal premiums sit
    strictly inside `(0, max_premium)`, i.e. aren't saturated) equal 2 at
    every step. The paper's exact recursion lets I_i, J_i depend on the
    realized inventory path, which would require a full backward induction
    over an inventory grid (their own numerical procedure, Appendix C's
    Lemma 6) rather than this closed-form scalar recursion -- this is the
    same "assume normal, non-saturated operation" simplification used
    throughout this module; it holds away from extreme inventory and
    breaks down near it.

    Returns an array of length `n_sessions + 1`. `m[i]` is only ever
    consumed as the *next-step* coefficient for session `i-1`'s quote (see
    `quote_theorem3`) -- `m[0]` is computed but never itself used as an
    `m_next`.
    """
    if n_sessions < 1:
        raise ValueError(f"n_sessions must be >= 1, got {n_sessions}")

    B = stock_intensity.D
    D_ = option_intensity.D
    m = np.empty(n_sessions + 1)
    m[n_sessions] = -gamma * sigma_risk ** 2 * S ** 2 * terminal_horizon

    coef = 2.0 * B + 2.0 * D_ * delta ** 2
    for i in range(n_sessions - 1, -1, -1):
        m[i] = m[i + 1] + session_dt * coef * m[i + 1] ** 2
        # m must stay negative and shrink in magnitude going backward (m_i -> 0 monotonically);
        # this is an explicit-Euler discretization of a Riccati-type equation, which overshoots
        # past zero and blows up once session_dt*coef*|m_next| stops being well below 1 per step.
        if not np.isfinite(m[i]) or m[i] > 0.0:
            raise RuntimeError(
                f"tilt-slope recursion diverged at session {i} (m={m[i]!r}): the explicit "
                "backward recursion is only numerically stable while session_dt*coef*|m_next| "
                "stays well below 1 each step -- use more/smaller sessions (larger n_sessions "
                "over the same total horizon) or a smaller gamma/terminal_horizon"
            )
    return m


def quote_theorem3(S, K, T_opt, r, q, sigma_bs, option_type,
                    stock_intensity: LinearIntensity, option_intensity: LinearIntensity,
                    m_next: float, q_s: float = 0.0, q_o: float = 0.0) -> Theorem2Quote:
    """
    Theorem 3 (paper Section III.B / Appendix C): the multi-period optimal
    quoting policy at one session, given that session's own inventory and
    the tilt-slope coefficient `m_next` for the *following* session (i.e.
    `compute_tilt_slope_path(...)[i+1]` when quoting at session `i`).

    Returns the same `Theorem2Quote` shape as `quote_theorem2` -- by
    construction, calling this with `m_next` equal to
    `compute_tilt_slope_path(...)[-1]` (the terminal m_n, for a
    single-session path) reproduces `quote_theorem2` exactly, since
    Theorem 2 is this recursion's last-session special case. `delta` here
    should be computed the same way (same S, K, T_opt, sigma_bs) as
    whatever was passed into `compute_tilt_slope_path` to build `m_next` --
    the recursion assumes a fixed Delta across the whole session path (see
    that function's docstring).
    """
    mid = float(bs.price(S, K, T_opt, r, q, sigma_bs, option_type))
    delta = float(bs.delta(S, K, T_opt, r, q, sigma_bs, option_type))
    return _tilted_quote(S, mid, delta, stock_intensity, option_intensity, m_next, q_s, q_o)


def _gamma_vega_risk_k(bs_gamma_greek, S, sigma_risk, alpha, horizon, T_opt):
    """
    Theorem 4/5's combined Gamma+Vega risk coefficient (paper Section IV):

        k = (0.5*sigma_risk^2*horizon + alpha^2*T_opt^2) * bs_gamma_greek^2 * S^4 * sigma_risk^2 * horizon

    `horizon` is the short one-period risk window (`T - t_n`, same meaning
    as Theorem 2/3's `horizon`); `T_opt` doubles as the paper's
    `T_mat - t_n` -- the OPTION's own remaining time to maturity, a much
    longer, separate timescale from `horizon`. The first term inside the
    parentheses is discrete-hedging Gamma risk; the second is Vega risk
    from the implied vol itself moving. The paper's own finding (Section
    IV, Figures 4-5): the Vega term dominates for long-dated options, the
    Gamma term for short-dated ones, since it scales with `T_opt^2` while
    the Gamma term only scales with `horizon`.
    """
    return (
        (0.5 * sigma_risk ** 2 * horizon + alpha ** 2 * T_opt ** 2)
        * bs_gamma_greek ** 2 * S ** 4 * sigma_risk ** 2 * horizon
    )


@dataclass(frozen=True)
class Theorem4Quote:
    """One dealer quote on the option only, per Theorem 4/5 (Section IV: delta-hedged, residual Gamma/Vega risk)."""
    mid: float
    delta: float
    gamma_greek: float
    vega: float
    m: float
    ask_premium: float
    bid_premium: float
    ask_price: float
    bid_price: float


def _tilted_option_quote(mid, delta, bs_gamma_greek, vega, option_intensity: LinearIntensity,
                          m, q_o) -> Theorem4Quote:
    """
    Shared closed-form policy behind both `quote_theorem4` and
    `quote_theorem5`, mirroring `_tilted_quote`'s role for Theorems 2/3 --
    except there's only one leg here (no stock quoting in Section IV) and
    the tilt is in plain `q_o`, not `net_delta`: Delta risk is already
    hedged away, so only the option's own Gamma/Vega exposure remains, and
    that's already folded into `m` (via `_gamma_vega_risk_k`) rather than
    needing an extra Delta-weighting factor the way Theorem 2/3's option
    leg does.
    """
    ask = float(np.clip(
        option_intensity.max_premium / 2.0 + m * (q_o - 0.5), 0.0, option_intensity.max_premium
    ))
    bid = float(np.clip(
        option_intensity.max_premium / 2.0 - m * (q_o + 0.5), 0.0, option_intensity.max_premium
    ))
    return Theorem4Quote(
        mid=mid, delta=delta, gamma_greek=bs_gamma_greek, vega=vega, m=m,
        ask_premium=ask, bid_premium=bid, ask_price=mid + ask, bid_price=mid - bid,
    )


def quote_theorem4(S, K, T_opt, r, q, sigma_bs, option_type, option_intensity: LinearIntensity,
                    gamma: float, sigma_risk: float, alpha: float, horizon: float,
                    q_o: float = 0.0) -> Theorem4Quote:
    """
    Theorem 4 (paper Section IV, one-period model): the dealer delta-hedges
    continuously in the stock (like Theorem 1 -- no stock quoting here), so
    net Delta is fully hedged away, but still faces residual risk from
    discrete-hedging Gamma and from the option's own implied volatility
    moving (Vega risk, via `alpha`; Schonbucher 1999). Only the option is
    quoted, and its premiums tilt with net *Gamma* exposure (`q_o` alone --
    no Delta-weighting, since Delta risk is gone) instead of net Delta.

    `alpha` is the volatility of the option's own implied vol
    (`d(sigma_hat) = alpha*dW`) -- see `calibration.implied_vol_of_vol` for
    estimating it from real option-trade IV prints. `sigma_risk` is the
    physical stock vol, as in Theorem 2/3. `horizon` is `T - t_n`, the
    short one-period risk window; `T_opt` doubles as the option's own
    (much longer) remaining time to maturity -- see `_gamma_vega_risk_k`.
    """
    mid = float(bs.price(S, K, T_opt, r, q, sigma_bs, option_type))
    delta = float(bs.delta(S, K, T_opt, r, q, sigma_bs, option_type))
    bs_gamma_greek = float(bs.gamma(S, K, T_opt, r, q, sigma_bs))
    vega = float(bs.vega(S, K, T_opt, r, q, sigma_bs))

    k = _gamma_vega_risk_k(bs_gamma_greek, S, sigma_risk, alpha, horizon, T_opt)
    m = -gamma * k
    return _tilted_option_quote(mid, delta, bs_gamma_greek, vega, option_intensity, m, q_o)


def implied_gamma_from_option_risk_limit(option_intensity: LinearIntensity, q_o_limit: float,
                                          bs_gamma_greek: float, S: float, sigma_risk: float,
                                          alpha: float, horizon: float, T_opt: float) -> float:
    """
    Section IV analogue of `implied_gamma_from_risk_limit`: back out gamma
    from a stated option-inventory risk limit (net Gamma exposure via `q_o`
    alone -- there's no stock leg in this section) instead of guessing
    gamma directly. Solves for the gamma at which the ask premium reaches
    its floor (0) exactly when `q_o` reaches `q_o_limit`.
    """
    if q_o_limit <= 0.5:
        raise ValueError(f"q_o_limit must exceed 0.5, got {q_o_limit}")
    k = _gamma_vega_risk_k(bs_gamma_greek, S, sigma_risk, alpha, horizon, T_opt)
    risk = (option_intensity.max_premium / 2.0) / (q_o_limit - 0.5)
    return risk / k


def compute_tilt_slope_path_theorem5(n_sessions, session_dt, terminal_horizon, gamma, sigma_risk, S,
                                      bs_gamma_greek, alpha, T_opt, option_intensity: LinearIntensity) -> np.ndarray:
    """
    Theorem 5's backward recursion (paper Section IV multi-period model)
    for the tilt-slope m_i -- the Section IV analogue of
    `compute_tilt_slope_path`. Terminal condition `m[n_sessions] =
    -gamma*k` (see `_gamma_vega_risk_k`) is exactly Theorem 4's `m`, and
    each earlier `m[i]` is computed backward via

        m_i = m_{i+1} + session_dt * D * m_{i+1}^2 * J_i

    where `D` is the option intensity's `D` parameter -- there's only one
    leg's worth of `D` here, unlike Theorem 3's `2*B + 2*D*Delta^2` (no
    stock leg, no Delta-weighting). As in `compute_tilt_slope_path`, `J_i`
    (whether session i's own premiums sit strictly inside their bounds) is
    assumed to equal 2 at every step -- same "assume normal, non-saturated
    operation" simplification, same reason (the exact recursion needs a
    full inventory-grid backward induction otherwise). `S`, `bs_gamma_greek`,
    and both vols are held fixed across sessions, matching the paper's own
    "no intraday movement" multi-period simplification (as in Theorem 3).

    Raises `RuntimeError` on numerical divergence, same as
    `compute_tilt_slope_path` and for the same reason (explicit-Euler
    discretization of a Riccati-type equation).
    """
    if n_sessions < 1:
        raise ValueError(f"n_sessions must be >= 1, got {n_sessions}")

    D_ = option_intensity.D
    k = _gamma_vega_risk_k(bs_gamma_greek, S, sigma_risk, alpha, terminal_horizon, T_opt)
    m = np.empty(n_sessions + 1)
    m[n_sessions] = -gamma * k

    coef = 2.0 * D_
    for i in range(n_sessions - 1, -1, -1):
        m[i] = m[i + 1] + session_dt * coef * m[i + 1] ** 2
        if not np.isfinite(m[i]) or m[i] > 0.0:
            raise RuntimeError(
                f"tilt-slope recursion diverged at session {i} (m={m[i]!r}): the explicit "
                "backward recursion is only numerically stable while session_dt*coef*|m_next| "
                "stays well below 1 each step -- use more/smaller sessions (larger n_sessions "
                "over the same total horizon) or a smaller gamma/terminal_horizon"
            )
    return m


def quote_theorem5(S, K, T_opt, r, q, sigma_bs, option_type, option_intensity: LinearIntensity,
                    m_next: float, q_o: float = 0.0) -> Theorem4Quote:
    """
    Theorem 5 (paper Section IV multi-period model): the multi-period
    analogue of `quote_theorem4`, given the tilt-slope coefficient
    `m_next` for the *following* session (i.e.
    `compute_tilt_slope_path_theorem5(...)[i+1]` when quoting at session
    `i`). Returns the same `Theorem4Quote` shape as `quote_theorem4` -- by
    construction, calling this with `m_next` equal to
    `compute_tilt_slope_path_theorem5(...)[-1]` (the terminal m_n, for a
    single-session path) reproduces `quote_theorem4` exactly, since
    Theorem 4 is this recursion's last-session special case.
    """
    mid = float(bs.price(S, K, T_opt, r, q, sigma_bs, option_type))
    delta = float(bs.delta(S, K, T_opt, r, q, sigma_bs, option_type))
    bs_gamma_greek = float(bs.gamma(S, K, T_opt, r, q, sigma_bs))
    vega = float(bs.vega(S, K, T_opt, r, q, sigma_bs))
    return _tilted_option_quote(mid, delta, bs_gamma_greek, vega, option_intensity, m_next, q_o)

"""
Cross-model benchmark: classical Avellaneda & Stoikov (2008), pointed at this
package's own option and run through `sim.simulate` on the same footing as
its built-in `constant_spreads` fixed-width benchmark.

Adapter design. AS(2008) (`avellaneda_stoikov.py`, this package) quotes a single
asset directly, with no hedging -- there is no option, no Greeks, no implied
surface in the original paper at all. To point it at this package's option,
choices have to be made, made explicit here rather than folded silently into
a benchmark number.

**Deriving the asset's own volatility, by Ito's lemma.** The "asset" AS quotes
is taken to be the option's own mark `O(t, S_t)`, a deterministic function
solving the Black-Scholes PDE at `sigma_imp` (zero rates/dividends):

    d_t O + (1/2) sigma_imp^2 s^2 d_ss O = 0

Ito's lemma on the composite `O(t, S_t)`, for `S_t` following `dS_t = S_t *
sigma * dB_t` at some real-world volatility `sigma`, gives:

    dO = [d_t O + (1/2) sigma^2 S_t^2 d_ss O] dt + d_s O * S_t * sigma * dB_t

Substituting the PDE identity `d_t O = -(1/2) sigma_imp^2 S_t^2 d_ss O`:

    dO = (sigma^2 - sigma_imp^2)/2 * Gamma^$(t, S_t) dt  +  Delta_t * S_t * sigma * dB_t

using `Gamma^$(t,s) := s^2 d_ss O(t,s)` (`bs_mm.dollar_gamma`) and
`Delta_t = d_s O(t, S_t)`. AS's own asset has no drift term at all
(`ds = sigma dB`) -- its model has no place for the `dt` term above, which is
exactly `edge.phi_edge`'s own instantaneous integrand, the paper's
gamma-theta carry. That drift is *only* exactly zero -- matching AS's own
assumed `dO = sigma_O dB_t` with no hidden inconsistency -- when
`sigma == sigma_imp`. So this module always sets the diffusion coefficient
`sigma_O := |Delta_t| * S_t * sigma_imp`, using the market's own implied vol
rather than the maker's belief: the self-consistent reading of "apply AS's
no-drift assumption to an option" is that AS can only ever be a model for a
maker with no capacity to hold a volatility view at all. `sigma_imp` never
mattered in the alternative (real-volatility) reading either -- it only ever
entered through the PDE substitution into the (dropped) drift term -- so
using it here is not an approximation of that other reading, it is a
different, self-consistent one, chosen deliberately over it.

`sigma_O` is still re-evaluated fresh at every `(t,s)` the simulator visits
(`Delta_t` and `S_t` both move), so this remains a frozen-coefficient
heuristic even with `sigma_imp` fixed -- AS's own closed form is derived for
a CONSTANT `sigma`, and this does not re-solve the (harder,
non-constant-coefficient) HJB equation a genuinely time-varying `sigma_O`
would require.

**AS has no asymmetric-book concept**, so its single symmetric intensity
decay `k` is taken as the average of this package's own `kappa_b`, `kappa_a`.

**`gamma` (AS's risk aversion) has no natural translation** from this
package's own `(alpha, beta)` -- the two objectives are different shapes
entirely (quadratic running/terminal penalty vs. CARA utility).
`calibrate_gamma` picks it by matching a target average inventory instead of
guessing, so a benchmark run is risk-matched rather than comparing whichever
strategy happens to sit on less risk.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.optimize import brentq

from .avellaneda_stoikov import optimal_half_spread
from . import bs_mm
from .sim import SimResult, simulate

if TYPE_CHECKING:
    from .bs_mm import VolSurfaceLike
    from .quoting import MMParams

SpreadFn = Callable[[float, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def symmetric_kappa(params: "MMParams") -> float:
    """AS has no bid/ask asymmetry; the closest single decay is the average of the two."""
    return 0.5 * (params.kappa_b + params.kappa_a)


def avellaneda_stoikov_spread_fn(
    K: float, tau: float, T: float, sigma_imp_val: float, gamma: float, k: float,
) -> SpreadFn:
    """
    Build a `sim.simulate`-compatible `spread_fn` for classical AS(2008),
    applied to this option's own mark as derived in the module docstring.

    `sigma_imp_val` marks the option (and its delta), and is also what AS's
    own diffusion coefficient `sigma_O := |Delta_t| * S_t * sigma_imp_val`
    uses -- the unique choice making AS's own no-drift assumption hold
    exactly for `O(t,S_t)` (see the module docstring's derivation).

    Returned as `(delta_b, delta_a)` distances from the mark, matching every
    other `spread_fn`/`constant_spreads` policy `sim.simulate` accepts --
    not raw bid/ask prices.
    """
    if gamma <= 0.0 or not np.isfinite(gamma):
        raise ValueError(f"gamma must be finite and positive, got {gamma}")
    if k <= 0.0 or not np.isfinite(k):
        raise ValueError(f"k must be finite and positive, got {k}")

    def spread_fn(t: float, spots: np.ndarray, inventory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tau_minus_t = tau - t
        delta_t = bs_mm.delta_array(spots, K, tau_minus_t, sigma_imp_val)
        sigma_O = np.abs(delta_t) * spots * sigma_imp_val
        remaining = T - t
        half = optimal_half_spread(sigma_O, gamma, k, remaining)
        skew = inventory * gamma * sigma_O ** 2 * remaining
        return skew + half, half - skew

    return spread_fn


def calibrate_gamma(
    surface: "VolSurfaceLike",
    K: float,
    tau: float,
    T: float,
    params: "MMParams",
    target_mean_abs_q: float,
    *,
    S0: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    k: float | None = None,
    realised_sigma: float | None = None,
    log10_gamma_bracket: tuple[float, float] = (-7.0, -3.0),
    xtol: float = 1e-4,
) -> float:
    """
    Bisect for the `gamma` at which classical AS's own simulated average
    `|inventory|` matches `target_mean_abs_q` (typically the optimal rule's
    own `mean|q|` on the same contract), so a benchmark run compares the two
    rules at a matched risk level rather than whichever happens to sit on
    less of it.

    `realised_sigma` controls only the SIMULATED SPOT PATH's true diffusion
    (as in `sim.simulate` itself) -- leave it None (matching `params.sigma`)
    to match whatever the rest of a given comparison uses, so every strategy
    in it faces the identical simulated world. It does not affect AS's own
    `sigma_O`, which always uses `sigma_imp` (see the module docstring).

    `log10_gamma_bracket` is in log10(gamma), not gamma itself: sane values
    of `gamma` here sit many orders of magnitude below 1, because AS's
    formula multiplies `gamma` by the option's *dollar* variance
    (`sigma_O**2`, itself on the order of 1e8-1e9 at typical BTC option
    scales) rather than a stock's usual O(1) percentage variance. Widen the
    bracket and re-check its endpoints with `avellaneda_stoikov_spread_fn`
    directly before trusting `brentq` on a different contract or parameter
    regime -- a bracket that is too wide can send `sim.simulate`'s Poisson
    fill draws to `inf` (see the module's own tests for a worked example of
    what that failure looks like).
    """
    k = symmetric_kappa(params) if k is None else k
    sigma_imp_val = bs_mm.sigma_imp(surface, K, tau)
    sim_kwargs = dict(S0=S0, n_steps=n_steps, n_paths=n_paths, seed=seed, realised_sigma=realised_sigma)

    def gap(log10_gamma: float) -> float:
        gamma = 10.0 ** log10_gamma
        spread_fn = avellaneda_stoikov_spread_fn(K, tau, T, sigma_imp_val, gamma, k)
        result = simulate(surface, K, tau, T, params, spread_fn=spread_fn, **sim_kwargs)
        return float(np.abs(result.q).mean() - target_mean_abs_q)

    lo, hi = log10_gamma_bracket
    log10_gamma_star = brentq(gap, lo, hi, xtol=xtol)
    return 10.0 ** log10_gamma_star


def simulate_avellaneda_stoikov(
    surface: "VolSurfaceLike",
    K: float,
    tau: float,
    T: float,
    params: "MMParams",
    gamma: float,
    *,
    S0: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    k: float | None = None,
    realised_sigma: float | None = None,
) -> SimResult:
    """
    Run classical AS(2008) through the exact same `sim.simulate` machinery
    (fills, hedging, cash, objective) as every other strategy in this
    package, using the adapter above. `gamma` is a required input here --
    see `calibrate_gamma` to choose it by matched risk rather than guessing.

    `realised_sigma` is forwarded to `sim.simulate` itself -- pass the same
    value used for the optimal rule's own `simulate` call in the same
    comparison, so both strategies face the identical simulated world. It
    does not affect AS's own `sigma_O`, which always uses `sigma_imp`.
    """
    k = symmetric_kappa(params) if k is None else k
    sigma_imp_val = bs_mm.sigma_imp(surface, K, tau)
    spread_fn = avellaneda_stoikov_spread_fn(K, tau, T, sigma_imp_val, gamma, k)
    return simulate(surface, K, tau, T, params, spread_fn=spread_fn,
                    S0=S0, n_steps=n_steps, n_paths=n_paths, seed=seed, realised_sigma=realised_sigma)

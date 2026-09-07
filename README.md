# Volatility Surface Calibration Engine

Phase 1 of an option market-making system: pulls a live BTC/ETH option
chain from Deribit, fits a no-arbitrage volatility surface to it, and
exposes a single `surface.implied_vol(K, T)` query interface. Everything
here is stateless and side-effect-free by design — the next phase (an
Avellaneda-Stoikov quoting layer) reads Greeks and vols straight off this
surface without any of these modules needing to change.

## Layout

```
src/
  vol_surface/    bs.py, implied_vol.py, data.py, svi.py, arbitrage.py, ssvi.py, surface.py
  market_making/  quoting.py -- Avellaneda-Stoikov-style quoting layer (Phase 2)
  option_market_making/
                  Lucic & Tse (2025) single-option vol-arbitrage market maker:
                  bs_mm.py, riccati.py, edge.py, psi1.py, quoting.py, sim.py
                  pde.py -- exact-HJB finite-difference validation gate on the
                  second-order approximation (diagnostic only, not a quoting path)
notebooks/        vol_surface_pipeline.ipynb -- runs the Phase 1 pipeline against live data
                   optimal_quoting_theorem1.ipynb -- Phase 2, complete-market quoting (Theorem 1)
                   option_mm_vol_arbitrage.ipynb -- does Lucic & Tse's vol-arb edge
                     actually earn its keep? (spoiler: inventory control does the work)
notes/            pde_validation_report.html -- results of the exact-HJB gate
```

`pytest.ini` puts both `src` and `src/vol_surface` on `pythonpath`, so both
`pytest` (from the repo root) and the notebooks (via
`sys.path.insert(0, '../src')` + `sys.path.insert(0, '../src/vol_surface')`
in their first cell) import the vol-surface modules as plain top-level names
(`import bs`, `import surface`, ...) without needing the package installed --
exactly the flat-namespace convention the rest of the codebase already uses.
`market_making` is imported as a package (`from market_making import
quoting`) since only `src` needs to be on the path for that; its modules
reach the vol-surface modules the same flat way (`import bs`), which is why
`src/vol_surface` has to be on the path too.

## Pipeline

```
data.py            Deribit chain --> mid prices --> implied vols (liquidity-filtered)
      |
      v
svi.py             per-expiry raw SVI fit (weighted least squares on total variance)
      |
      v
arbitrage.py       butterfly check (gates every slice fit) + calendar check (gates the surface)
      |
      v
surface.py         term-structure interpolation in total variance -> implied_vol(K, T)
```

Run the whole thing end to end:

```python
import surface

surf = surface.run_pipeline("BTC")     # ingest -> fit -> validate -> ready to query
surf.implied_vol(K=65000, T=0.25)      # interpolated vol, on-grid or off
fig, ax = surface.plot_surface_3d(surf)
```

`notebooks/vol_surface_pipeline.ipynb` runs and visualizes every step
against live data — the fastest way to see it work.

## Modules

- **`bs.py`** — vectorized Black-Scholes price + Greeks (delta, gamma, vega, theta).
- **`implied_vol.py`** — implied vol solver: Newton-Raphson (using `bs.py`'s
  vega) with a Brent's-method fallback for near-zero-vega or non-converging
  cases. Before solving, in-the-money quotes are converted to their
  out-of-the-money equivalent via put-call parity — an ITM price is mostly
  intrinsic value, so the time-value signal that actually carries
  volatility information is lost to floating-point cancellation unless you
  solve on the OTM side.
- **`data.py`** — pulls `get_book_summary_by_currency` from Deribit's public
  API, filters quotes with no two-sided market or a bid-ask width above a
  threshold, and converts survivors to implied vols. Outputs
  `strike, expiry, T, forward, mid_iv, weight`.
- **`svi.py`** — fits the raw SVI parameterization
  `w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))` to one expiry's
  `(k, mid_iv, weight)` via weighted nonlinear least squares, where
  `k = ln(K/F)` and `w` is total variance (`iv^2 * T`).
- **`arbitrage.py`** — the no-arbitrage gate (see below).
- **`surface.py`** — loops the fit across every expiry, gates the whole term
  structure with the calendar check, and interpolates between expiries in
  total variance so every query point stays consistent with that check.

## No-arbitrage checks, and why they matter

A vol surface isn't just a curve-fit — it has to correspond to a valid
(non-negative) risk-neutral probability density at every expiry, and to
consistent forward variance between expiries. Two static-arbitrage
conditions are checked, both in total variance:

**Butterfly (per-slice).** The Gatheral-Jacquier function

```
g(k) = (1 - k*w'(k)/(2*w(k)))^2 - (w'(k)^2/4)*(1/w(k) + 1/4) + w''(k)/2
```

is proportional to the Black-Scholes implied risk-neutral density at
log-moneyness `k`. If `g(k) < 0` anywhere, the fitted smile implies a
*negative* probability density there — meaning a butterfly spread
centered at that strike (long one wing, short two ATM, long the other
wing) would cost less than zero but never pay off negative. That's free
money, i.e. arbitrage. `arbitrage.check_butterfly` evaluates `g(k)` on a
grid spanning realistic BTC/ETH moneyness (`k` in `[-1.5, 1.5]`, roughly
22%-450% of the forward) and is called from `svi.fit_svi_slice` on every
fit — a slice that fails is rejected before it ever reaches a price.

`g(k) >= 0` is necessary but not sufficient: Gatheral & Jacquier show a
slice is fully free of butterfly arbitrage only if `g(k) >= 0` *and*
`lim_{k->+inf} d+(k) = -inf`, where `d+(k) = -k/sqrt(w(k)) + sqrt(w(k))/2`
(`arbitrage.d_plus` evaluates this directly, so the limit can be inspected
or plotted at genuinely large `k` rather than taken on faith). Without
that second condition the right wing could grow steeply enough on strikes
*beyond* the check grid to imply infinite (or non-existent) moments of the
underlying's terminal price — a real gap, since `g(k)` is a pointwise
check and can't see that far out. For raw SVI this limit reduces to a
closed-form bound on the parameters: `b*(1+rho) < 2`, exactly Roger Lee's
moment-formula bound on the asymptotic slope of total variance.
`arbitrage.check_wing_condition` checks that bound directly rather than
literally evaluating a limit at infinity, and
`check_butterfly` runs it alongside the `g(k)` grid check.

**Calendar spread (cross-slice).** Total variance `w(k, T)` must be
non-decreasing in `T` at fixed log-moneyness. If it weren't, you could
sell the shorter-dated claim and buy the longer-dated one at the same
moneyness and lock in a profit regardless of where volatility goes next —
equivalently, the implied *forward variance* between the two expiries
would be negative, which a variance swap can never have. `arbitrage.check_calendar`
checks this between every pair of adjacent (by maturity) fitted slices,
and is run once by `surface.fit_surface` after every expiry has been
fitted (it needs neighboring slices, so it can't run inside a single-slice
fit the way the butterfly check does).

`check_calendar` evaluates `w2 - w1` on `DEFAULT_K_GRID`, which is fast
but can in principle miss a violation in a narrow gap between grid points.
`arbitrage.check_calendar_exact` is a slower, exact alternative: it finds
every real intersection of two raw SVI slices via the closed-form quartic
of Gatheral & Jacquier (2013, Lemma 3.3) (`find_slice_crossings`, with the
quartic's coefficients derived symbolically via `sympy` once at import
time), then checks the ordering right at each crossing rather than on a
fixed grid. A crossing alone isn't necessarily arbitrage — the curves
could touch tangentially and stay correctly ordered on both sides — so it
checks `w2 - w1` just either side of each crossing point. Use
`check_calendar` for cheap checks during iterative calibration and
`check_calendar_exact` for a final rigorous pass or to validate a
suspected narrow-crossing edge case.

**Why interpolate in total variance.** Linear interpolation between two
points always stays between them. Since the calendar check already
guarantees `w(k, T_hi) >= w(k, T_lo)` between adjacent fitted expiries,
interpolating `w` linearly in `T` — the way `surface.py` does — means every
off-grid query point automatically respects the calendar condition too.
Interpolating implied vol directly would not carry that guarantee.

**Handling real market noise.** `svi.fit_svi_slice` raises
`arbitrage.ArbitrageError` immediately when called directly (the strict
behavior above). `surface.run_pipeline`, which has to run on live,
sometimes-noisy data with no human in the loop, instead calls
`fit_surface(..., on_bad_slice="skip")`: an expiry whose fit fails the
butterfly gate is excluded from the surface and flagged via
`warnings.warn`, rather than aborting the entire build. The rejected slice
is never used for pricing — it's dropped, not silently passed through —
so this doesn't weaken the guarantee, it just keeps one illiquid wing on
one expiry from taking down an otherwise-good surface.

## Results

**Per-expiry SVI fit against the live market smile:**

![SVI fit vs. market smile](pics/SVI-fit.png)

**Global eSSVI surface fit across all expiries at once:**

![Global eSSVI surface](pics/eSSVI-raw.png)

**Vol-arb quoting policy: terminal PnL and inventory vs. fixed-width quoting baselines:**

![Terminal portfolio value and inventory](pics/PnL-and-inventory.png)

## Design notes for the next phase

- `bs.price`/Greeks and `surface.implied_vol` are pure functions of their
  inputs — no global state, no assumption that there's only one run or one
  expiry. That's intentional: the planned Avellaneda-Stoikov quoting layer
  will call these directly, per live inventory update, without any
  wrapping needed here.
- The SVI fit's initial guess is a from-scratch heuristic
  (`svi._initial_guess`); `fit_svi_slice` accepts an `x0` override so a
  live system can warm-start each fit from the previous tick's converged
  parameters instead — smile parameters move very little tick to tick, and
  that would both speed up and stabilize repeated fitting.

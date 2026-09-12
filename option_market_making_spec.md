# Build Spec: Single-Option Market-Making Engine (Constant σ)

## Source

Lucic, V. & Tse, A.S.L. (2025), "Option market-making and vol arbitrage,"
Risk.net, February 2025. Baseline single-option model (pp. 1–3 of the
article). All equation numbers below refer to this article unless stated
otherwise.

## Goal

Implement the closed-form Avellaneda-Stoikov / HJB optimal bid-ask quoting
engine for a single European call option, under the simplifying assumption
of a **constant real-world volatility** `sigma` (no stochastic-vol or
local-vol path for the trader's own view). Mark-to-market and Greeks are
sourced live from the existing eSSVI implied-volatility surface.

This is a scoped build. See "Out of scope" section — do not implement those
items in this pass.

---

## File structure

```
src/
├── vol_surface/                 (existing — eSSVI surface pipeline)
│   ├── ssvi.py
│   ├── surface.py
│   ├── arbitrage.py
│   └── run_pipeline
└── option_market_making/        (new — this build)
    ├── bs_mm.py
    ├── riccati.py
    ├── edge.py
    ├── psi1.py
    ├── quoting.py
    └── sim.py
```

**Before writing any code**, inspect `vol_surface/surface.py` to find the
actual function/class used to query `sigma_imp(K, tau)` from the calibrated
eSSVI surface. The exact import and call signature in `bs_mm.py` below must
be adapted to match whatever that module actually exposes — do not assume
a specific API name without checking.

---

## Global conventions

- Python 3, full type hints on every function signature.
- Parameter bundling via `@dataclass`, matching the existing `SSVIParams`
  pattern used in `vol_surface/ssvi.py`.
- All time variables (`t`, `tau`, `T`) are year-fractions, consistent with
  the conventions already used in `vol_surface`.
- No network calls anywhere inside `option_market_making`. `sigma_imp` is
  obtained via a single call into `vol_surface.surface` per (K, tau); no
  other module talks to the surface directly except `bs_mm.py`.
- `sigma`, `lambda0_b`, `lambda0_a`, `kappa_b`, `kappa_a`, `alpha`, `beta`
  are **inputs only**. No estimation logic for any of these belongs in this
  package. They are computed externally (by the user, in a notebook or
  elsewhere) and passed in via `MMParams`.

---

## MMParams (defined in `quoting.py`)

```python
@dataclass
class MMParams:
    alpha: float       # terminal inventory penalty, alpha > 0
    beta: float        # running inventory penalty,  beta  > 0
    lambda0_b: float   # bid-side base fill intensity, lambda0_b > 0
    lambda0_a: float   # ask-side base fill intensity, lambda0_a > 0
    kappa_b: float     # bid-side intensity decay,     kappa_b > 0
    kappa_a: float     # ask-side intensity decay,     kappa_a > 0
    sigma: float        # constant real-world vol estimate, sigma > 0
```

Order flow is the **general asymmetric case**: do not assume
`lambda0_b == lambda0_a` or `kappa_b == kappa_a` anywhere. `psi1.py` must
include the full correction terms (see Stage 4) — no symmetric-flow
shortcut.

---

## Stage 1 — `bs_mm.py`

Black-Scholes pricing/Greeks layer for a single European call, with
`sigma_imp` sourced live from the eSSVI surface.

Functions:

- `sigma_imp(K: float, tau: float) -> float`
  Thin wrapper around the `vol_surface.surface` query for a single (K, tau)
  point. Adapt to the real surface API (see note above).

- `d2(s: float, K: float, tau_minus_t: float, sigma_imp: float) -> float`

  ```
  d2 = (ln(s/K) - sigma_imp**2 * tau_minus_t / 2) / (sigma_imp * sqrt(tau_minus_t))
  ```

- `call_price(s, K, tau_minus_t, sigma_imp) -> float`
  Standard Black-Scholes call formula (zero rates/dividends, per the
  paper's assumptions).

- `dollar_gamma(s, K, tau_minus_t, sigma_imp) -> float`

  ```
  Gamma_dollar = K * phi_pdf(d2) / (sigma_imp * sqrt(tau_minus_t))
  ```

  where `phi_pdf` is the standard normal density.

- `delta(s, K, tau_minus_t, sigma_imp) -> float`
  Standard BS delta, `d/ds` of `call_price`, evaluated at `sigma_imp` (not
  the real-world `sigma`).

Edge cases: guard `tau_minus_t -> 0` (near expiry) against division by
zero — raise a clear `ValueError` rather than returning `nan`/`inf`.

---

## Stage 2 — `riccati.py`

Closed-form solution to the Riccati ODE for `psi2(t)`, and the discount
kernel `D(t,u)`. Pure algebra, no numerical ODE solver.

Functions:

- `upsilon(lambda0_b, kappa_b, lambda0_a, kappa_a) -> float`

  ```
  Upsilon = 1 / (2 * exp(-1) * (lambda0_a * kappa_a + lambda0_b * kappa_b))
  ```

- `eta(beta, upsilon_val) -> float`

  ```
  eta = sqrt(beta / upsilon_val)
  ```

- `zeta_pm(alpha, beta, upsilon_val) -> tuple[float, float]`

  ```
  zeta_plus  = sqrt(upsilon_val * beta) + alpha
  zeta_minus = sqrt(upsilon_val * beta) - alpha
  ```

- `psi2(t: float, T: float, params: MMParams) -> float`

  ```
  psi2(t) = sqrt(Upsilon * beta) *
      (zeta_minus * exp(-eta*(T-t)) - zeta_plus * exp(eta*(T-t))) /
      (zeta_minus * exp(-eta*(T-t)) + zeta_plus * exp(eta*(T-t)))
  ```

  Terminal condition to test: `psi2(T) == -alpha`.

- `discount_kernel(t: float, u: float, T: float, params: MMParams) -> float`

  ```
  D(t,u) = (zeta_minus * exp(-eta*(T-u)) + zeta_plus * exp(eta*(T-u))) /
           (zeta_minus * exp(-eta*(T-t)) + zeta_plus * exp(eta*(T-t)))
  ```

---

## Stage 3 — `edge.py`

The volatility-arbitrage edge term `phi_edge(t,s)`, closed form under
constant `sigma`. This is the only module requiring numerical quadrature.

Depends on: `riccati.discount_kernel`, standard normal pdf (`scipy.stats.norm.pdf`).

- `phi_edge(t, s, K, tau, T, sigma_imp_val, params: MMParams) -> float`

  ```
  phi_edge(t,s) = K * (sigma**2 - sigma_imp_val**2) / 2 *
      integral_{u=t}^{T} [
          D(t,u) / sqrt(sigma_imp_val**2*(tau-u) + sigma**2*(u-t))
          * phi_pdf( z(u) )
      ] du

  z(u) = ( ln(s/K) - 0.5*sigma**2*(u-t) - 0.5*sigma_imp_val**2*(tau-u) )
         / sqrt(sigma_imp_val**2*(tau-u) + sigma**2*(u-t))
  ```

  Implement via `scipy.integrate.quad` over `u in [t, T]`. `sigma_imp_val`
  is evaluated once per contract and held frozen over the integration (this
  matches the model's own assumption that the surface doesn't move within
  the market-making horizon).

  Boundary check for tests: at `t == T` the integration interval is empty,
  so `phi_edge(T, s) == 0`.

---

## Stage 4 — `psi1.py`

`psi1(t,s)`, the full asymmetric-flow form (no symmetric shortcut).

Depends on: `edge.phi_edge`, `riccati.discount_kernel`, `riccati.psi2`.

- `psi1(t, s, K, tau, T, sigma_imp_val, params: MMParams) -> float`

  ```
  psi1(t,s) = phi_edge(t,s)
      + 2*exp(-1)*(lambda0_b - lambda0_a) *
            integral_{u=t}^{T} D(t,u) * psi2(u) du
      + 2*exp(-1)*(lambda0_b*kappa_b - lambda0_a*kappa_a) *
            integral_{u=t}^{T} D(t,u) * psi2(u)**2 du
  ```

  Each correction integral via `scipy.integrate.quad`, integrand calling
  `riccati.psi2(u, T, params)` and `riccati.discount_kernel(t, u, T, params)`.

  Boundary check for tests: `psi1(T, s) == 0`.

---

## Stage 5 — `quoting.py`

Assembles optimal spreads and quotes. Hosts `MMParams`.

Depends on: `bs_mm`, `riccati`, `psi1`.

- `optimal_spreads(t, s, q, K, tau, T, params: MMParams) -> tuple[float, float]`
  Returns `(delta_b_star, delta_a_star)`.

  ```
  delta_b_star = 1/kappa_b - psi1(t,s) - (2*q+1) * psi2(t)
  delta_a_star = 1/kappa_a + psi1(t,s) + (2*q-1) * psi2(t)
  ```

- `quote(t, s, q, K, tau, T, params: MMParams) -> tuple[float, float]`
  Returns `(bid, ask)`.

  ```
  O = bs_mm.call_price(s, K, tau - t, sigma_imp(K, tau))
  bid = O - delta_b_star
  ask = O + delta_a_star
  ```

  Sanity property to test: `bid < O < ask` for reasonable parameter ranges.

---

## Stage 6 — `sim.py`

Backtest loop over `[0, T]`.

Depends on: `quoting`, `bs_mm`.

- `simulate(K, tau, T, params: MMParams, S0: float, n_steps: int, n_paths: int = 1, seed: int | None = None) -> SimResult`

  Steps:
  1. Discretize `[0, T]` into `n_steps` intervals of width `dt`.
  2. Simulate `S_u` as GBM with constant vol `params.sigma` and **zero
     drift** (the model assumes the market maker has no view on drift —
     see the paper's setup, `dS_t/S_t = sigma_t dB_t`).
  3. At each step: call `quoting.quote(t, s, q, K, tau, T, params)`.
  4. Simulate Cox-process fills over the interval: bid fills with
     intensity `lambda0_b * exp(-kappa_b * delta_b_star)`, ask fills with
     intensity `lambda0_a * exp(-kappa_a * delta_a_star)` (Poisson
     thinning or Bernoulli approximation over `dt` — either is acceptable,
     document the choice).
  5. Update inventory `q`, cash `X_t` per the cashflow equations in the
     paper (ask fills add `O + delta_a_star` to cash and decrement `q`;
     bid fills subtract `O - delta_b_star` from cash and increment `q`).
  6. Delta hedge: `Delta_t = q * bs_mm.delta(s, K, tau - t, sigma_imp(K,tau))`.
  7. Track portfolio value `V_t = X_t + q*O_t - Delta_t*S_t`.

  Return a dataclass or dict bundling the full time series
  (`t`, `S`, `q`, `V`, `bid`, `ask`) per path, for downstream analysis.

---

## Out of scope — do not implement this pass

Flag each of these as a `# TODO (follow-up):` comment in the relevant
module, but do not build them now:

1. **Non-constant real-world vol.** A stochastic or local-vol `sigma(t,S_t)`
   path would replace `edge.py`'s closed-form quadrature with a numerical
   expectation (Monte Carlo or PDE) — out of scope.
2. **Estimators for `sigma`, `lambda0_b/a`, `kappa_b/a`.** These are
   supplied externally as `MMParams` fields. No fitting/estimation logic
   belongs in this package.
3. **PDE finite-difference validation gate.** A follow-up task: solve the
   exact HJB equation (6) numerically on the discrete `q`-grid and compare
   against the second-order-approximation quotes from `quoting.py`, as an
   accuracy check before production use.
4. **Multi-option extension.** Vector/matrix `Theta2(t)`, `theta1(t)`,
   risk-factor matrices `A`, `B` — deferred until this single-option engine
   is validated.

---

## Testing

- `tests/option_market_making/`
- One **frozen real Deribit snapshot** as a fixture (not a live pull on
  every test run): a specific `(S0, K, tau)` plus the eSSVI-derived
  `sigma_imp` at that point, captured once and stored as a fixture file
  (JSON or a small constants module).
- Per-module tests:
  - `bs_mm`: call price within `[intrinsic, spot]`; dollar gamma positive;
    delta in `[0, 1]`.
  - `riccati`: `psi2(T) == -alpha`; `psi2` finite and bounded over `[0,T]`.
  - `edge`: `phi_edge(T, s) == 0`; sign of `phi_edge` matches sign of
    `(sigma**2 - sigma_imp**2)`.
  - `psi1`: `psi1(T, s) == 0`.
  - `quoting`: `bid < O < ask`; spreads positive for reasonable parameters.
  - `sim`: no NaN/inf over any simulated path; inventory `q` stays within a
    sane bound given the fill intensities used.

---

## Dependencies

`numpy`, `scipy` (`integrate.quad`, `stats.norm`), `dataclasses` (stdlib).

---

## Implementation order

1. `bs_mm.py` — independent; wire to `vol_surface.surface` first.
2. `riccati.py` — independent, pure algebra.
3. `edge.py` — depends on `riccati`, `bs_mm`.
4. `psi1.py` — depends on `edge`, `riccati`.
5. `quoting.py` — depends on `bs_mm`, `riccati`, `psi1`; defines `MMParams`.
6. `sim.py` — depends on `quoting`, `bs_mm`.

Write tests for each module immediately after implementing it — do not
defer all testing to the end.

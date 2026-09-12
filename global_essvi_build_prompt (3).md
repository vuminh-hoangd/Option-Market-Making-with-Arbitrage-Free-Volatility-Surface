# Build: Global eSSVI Calibrator — `global_essvi.py`

## Scope

New standalone module. Do not modify `ssvi.py`, `SSVIParams`, `arbitrage.py`, or
`surface.py`. Reuse `arbitrage.py`'s `g(k) >= 0` check for validation only.

End deliverable: a continuous, arbitrage-free implied-vol surface calibrated to live
Deribit option prices, queryable as `implied_vol(K, T)`.

---

## 1. eSSVI slice formula (globaleSSVI.pdf eq. 1)

Total variance at log-forward-moneyness `k = log(K / F0(T))`:

```
w(k, T) = (1/2) * [ θ(T) + ρ(T)ψ(T)k + sqrt( (ψ(T)k + θ(T)ρ(T))² + θ(T)²(1 − ρ(T)²) ) ]
```

where `ψ := θφ`. Implied vol: `σ(K,T) = sqrt(w(k,T) / T)`.

---

## 2. Butterfly bound `f(θ, |ρ|)` — two implementations, switchable

Default to GJ. Support MM as opt-in via `butterfly_bound: Literal["GJ", "MM"] = "GJ"`.

**GJ (sufficient only)** — globaleSSVI.pdf §2.2.1, from SVIcal.pdf Theorem 4.2:
```
f_GJ(θ, |ρ|) = 4θ / (1 + |ρ|)
```

**MM (necessary and sufficient)** — globaleSSVI.pdf §2.2.2, from Martini & Mingone
[9] Proposition 6.3:
```
f_MM(θ, |ρ|) = inf_{l > l₂(|ρ|)}  4θ√(1−ρ²) h²(l,|ρ|) / [ θ√(1−ρ²) g²(l,|ρ|) − g₂(l,|ρ|) ]
```
where (paper states "derivatives are taken with respect to l" — written out here for
implementation, differentiated from the paper's definition of `N`):
```
N(l,ρ)  = √(1−ρ²) + ρl + √(l²+1)
N'(l,ρ) = ρ + l/√(l²+1)
N''(l,ρ)= 1/(l²+1)^{3/2}
g(l,ρ)  = N'(l,ρ) / 4
h(l,ρ)  = 1 − (l − ρ/√(1−ρ²)) · N'(l,ρ) / (2N(l,ρ))
g₂(l,ρ) = N''(l,ρ) − N'(l,ρ)² / (2N(l,ρ))
l₂(|ρ|) = tan(arccos(−|ρ|) / 3)⁻¹       (i.e. cot(arccos(−|ρ|)/3), confirmed against
                                            primary source arXiv:2106.02418)
```

The paper (§2.3) notes: "the MM conditions are less strict than the GJ conditions...
However, in contrast with the latter ones, they are not explicit and require to use a
minimization algorithm to evaluate f_MM(θ,|ρ|), causing an increase in calibration time."

Use `scipy.optimize.minimize_scalar` for the infimum. Do a coarse grid scan first to
locate a good bracket, then polish.

Both functions share signature `(theta: float, abs_rho: float) -> float`.

---

## 3. No-arbitrage conditions (globaleSSVI.pdf §2.1, §2.3)

### Calendar spread (§2.1, Hendriks-Martini [7] Prop 3.5)

Given two maturities with eSSVI parameters `(ρ₁, θ₁, ψ₁)` and `(ρ₂, θ₂, ψ₂)`:

- **necessary**: `θ₂ > θ₁`; `ψ₂ > ψ₁ max( (1+ρ₁)/(1+ρ₂), (1−ρ₁)/(1−ρ₂) ) ≥ ψ₁`
- **sufficient** (added to the above): `ψ₂ ≤ (ψ₁/θ₁)·θ₂`

The paper (§2.1) notes the alternative sufficient condition
`ρ₁ − (ψ₂/ψ₁)ρ₂² ≤ (θ₂/θ₁ − 1)(ψ₂²θ₁/(ψ₁²θ₂) − 1)` but chooses the first "since
it is more tractable and a natural candidate for a global parametrization."

### Butterfly (§2.2.1–2.2.2)

Per-slice bounds using `f_GJ` or `f_MM` from §2:
- **necessary**: `ψ ≤ 4/(1+|ρ|)`
- **sufficient** (GJ): the above and `ψ² ≤ f_GJ(θ, |ρ|)`
- **necessary and sufficient** (MM): the above and `ψ² ≤ f_MM(θ, |ρ|)`

### Combined (§2.3, eq. 3)

All constraints for successive slices, butterfly and calendar together:
```
θ₂ > θ₁ > 0

ψ₁ ≤ min( 4/(1+|ρ₁|),  √f(θ₁,|ρ₁|) )

0 < ψ₁ max( (1+ρ₁)/(1+ρ₂), (1−ρ₁)/(1−ρ₂) )  <  ψ₂  ≤  min( (ψ₁/θ₁)·θ₂,  4/(1+|ρ₂|),  √f(θ₂,|ρ₂|) )
```
where `f` is either `f_GJ` or `f_MM` (§2). The paper notes (§2.3): "the function f can
be either from the MM model (f = f_MM) or from the GJ model (f = f_GJ)."

These are the conditions that the box reparametrization in §4 is designed to satisfy
automatically for any point in the open hyperrectangle.

---

## 4. Global reparametrization — `unbox()` (globaleSSVI.pdf §3, eq. 4-6, Prop 3.1)

**Free parameters** (eq. 4) — an open hyperrectangle:
```
ρ₁,...,ρ_N  ∈  ]−1, 1[
θ₁          ∈  ]0, ∞[
a₂,...,a_N  ∈  ]0, ∞[
c₁,...,c_N  ∈  ]0, 1[
```
Total: 3N parameters.

Since `least_squares` bounds are inclusive and Prop 3.1 requires the open box, use
epsilon-inset bounds: `a_i ∈ [ε, ∞)`, `θ₁ ∈ [ε, ∞)`, `c_i ∈ [ε, 1−ε]`,
`ρ_i ∈ [−1+margin, 1−margin]` with configurable `margin` (default 0.05, per §4.4's
recommendation of `]−0.95, 0.95[`).

**Recovery formulas** (eq. 5-6): implement as a pure function
`unbox(rho, theta_1, a, c, f_fn) -> (theta, rho, psi)`.

Forward pass — compute `θ` first:
```
p_i = max( (1+ρ_{i-1})/(1+ρ_i), (1−ρ_{i-1})/(1−ρ_i) )    for i > 1
θ_i = θ_{i-1} · p_i + a_i                                   for i > 1
```

Then `f_i` (needs all `θ_i` known):
```
f_i = min( 4/(1+|ρ_i|), sqrt(f(θ_i, |ρ_i|)) )              for all i
```

Then `ψ` — note `C_ψ_i` looks ahead to all future slices (eq. 5):
```
A_ψ₁ = 0
A_ψ_i = ψ_{i-1} · p_i                                      for i > 1

C_ψ₁ = min( f₁, f₂/p₂, f₃/(p₂·p₃), ..., f_N / ∏_{j=2}^{N} p_j )
C_ψ_i = min( (ψ_{i-1}/θ_{i-1})·θ_i,  f_i,  f_{i+1}/p_{i+1},  f_{i+2}/(p_{i+1}·p_{i+2}),  ...,  f_N / ∏_{j=i+1}^{N} p_j )    for i > 1

ψ_i = c_i · (C_ψ_i − A_ψ_i) + A_ψ_i
```

`ψ` must be computed sequentially (ψ₁ first, then ψ₂, etc.) because `A_ψ_i` and
`C_ψ_i` depend on `ψ_{i-1}`.

**Proposition 3.1:** any point in the open hyperrectangle maps to an arbitrage-free
surface — by construction.

---

## 5. Calibration (globaleSSVI.pdf §4)

**Objective** (§4.1, weighted squared price error):
```
min  Σ_{K,T}  ω(K,T) · ( C_market(K,T) − C_model(K,T) )²
```
Model prices via Black-Scholes with `σ = sqrt(w(k,T) / T)`.

Weights `ω(K,T)` are user-configurable. Default to uniform (`ω = 1`). The paper notes
weights can be chosen as inverse squared market Black-Scholes vegas for a first-order
implied-vol calibration.

Price each quote using its own option type (OTM calls and puts), matching SVIcal.pdf
§4.1's approach of selecting OTM options directly.

**Optimizer:** `scipy.optimize.least_squares` with epsilon-inset bounds (see §4).
Settings: `max_nfev=1000`, `ftol=1e-8` (paper's §4.1 values). `least_squares` takes a
residual vector — return `r(K,T) = sqrt(ω(K,T)) · (C_market(K,T) − C_model(K,T))`
elementwise.

**Initial guess** (separate function, improvable later):
- `ρ_i = 0`
- `θ₁` = ATM total variance from shortest-dated slice
- `a_i = max(θ_i^obs − θ_{i-1}^obs, ε)` (floor needed since observed ATM total variance
  may not be monotone across expiries)
- `c_i = 0.5`

---

## 6. Interpolation / extrapolation (globaleSSVI.pdf §5, taken from [3])

### ⚠ CRITICAL: interpolate `ρψ` (the product), NEVER `ρ` directly.

**Interpolation** (§5.1, `T_i ≤ t ≤ T_{i+1}`, `λ = (t − T_i) / (T_{i+1} − T_i)`):
```
θ_t       = (1−λ)θ_i     + λ θ_{i+1}
ψ_t       = (1−λ)ψ_i     + λ ψ_{i+1}
(ρψ)_t    = (1−λ)ρ_i ψ_i + λ ρ_{i+1} ψ_{i+1}
ρ_t       = (ρψ)_t / ψ_t
```

**Extrapolation before T₁** (§5.2.1, `λ = t / T₁`):
```
θ_t = λ θ₁ ;   ψ_t = λ ψ₁ ;   ρ_t = ρ₁
```

**Extrapolation after T_N** (§5.2.2, `λ = t / T_N`):
```
θ_t = λ θ_N ;   ψ_t = ψ_N ;   ρ_t = ρ_N
```

The paper proves all three preserve absence of arbitrage.

---

## 7. Data pipeline — Deribit

- Pull live option chain via public API (BTC/ETH options).
- Derive one `F₀(T)` per maturity: use Deribit's mark/index price if exposed, otherwise
  put-call parity regression on mid prices.
- Filter: drop zero-bid quotes, excessively wide bid-ask, and sub-tick prices. Pull
  tick size from the live contract spec rather than hardcoding.
- Output: clean `(K, T, C_market, option_type)` array per snapshot.

---

## 8. Validation

**(a) Runtime arbitrage check** (§4.4 style) — run after every calibration:
- Per-slice: `g(k) ≥ 0` butterfly check (reuse `arbitrage.py`), sampled densely in k.
- Pairwise: `w(k, T_{i+1}) ≥ w(k, T_i)` for adjacent pairs, sampled densely in k.
- The paper notes (§4.4) that numerical artifacts can arise near `|ρ| → 1`, avoided by
  the `ρ` clipping in §4.

**(b) Property-based tests on `unbox()`** — independent of market data:
- Sample `(ρ, θ₁, a, c)` from the open hyperrectangle (≥1000 draws, both small N=2-3
  and larger N=10-15). Use log-uniform for `θ₁` and `a_i` since they're unbounded above
  (e.g. `θ₁ ∈ (1e-4, 1.0)`, `a_i ∈ (1e-5, 0.5)`).
- Assert per draw: `θ` strictly increasing, `ψ` strictly increasing, `g(k) ≥ 0` per
  slice, `w(k, T_{i+1}) ≥ w(k, T_i)` pairwise.
- This directly tests Proposition 3.1. Should never fail for any sample in the open box.

---

## 9. Output interface

```python
def implied_vol(K: float, T: float) -> float:
    """
    Query the calibrated Global eSSVI surface.
    Converts K to log-forward-moneyness, interpolates/extrapolates (θ,ψ,ρψ),
    evaluates the eSSVI formula, returns σ_BS = sqrt(w/T).
    """
```

Greeks are out of scope for this task.

---

## 10. Codebase integration — verify against actual repo before implementing

- **Butterfly check reuse**: `arbitrage.py`'s `g(k) ≥ 0` check likely needs eSSVI → raw
  SVI conversion. The paper (globaleSSVI.pdf p.4) gives the map:
  `a=θ(1−ρ²)/2, b=ψ/2, m=−θρ/ψ, σ=θ√(1−ρ²)/ψ`. Confirm the actual function signature
  in `arbitrage.py` before writing an adapter.
- **Deribit pipeline**: check whether a data-fetching module already exists before
  building a new one.
- **Vega for weighting**: reuse if a Black-Scholes vega function already exists.
- **Import/path conventions**: match the rest of the project.

---

## 11. Deliverables

- [ ] `global_essvi.py`: `f_GJ()`, `f_MM()`, `unbox()`, calibration entry point,
      interpolation/extrapolation, `implied_vol(K,T)`
- [ ] Deribit data-pull + cleaning function
- [ ] Runtime arbitrage-check routine
- [ ] Property-based test suite for `unbox()` (Prop 3.1)

"""
Full volatility surface: fit every expiry, validate no-arbitrage across the
whole term structure, and expose a single `implied_vol(K, T)` query
interface -- the thing the pricer (and eventually the quoting layer)
actually calls.

Two ways to build a surface:
- `fit_surface` fits an independent raw-SVI slice per expiry (Steps 4-5),
  then interpolates across maturities via `VolSurface`, which blends
  undiscounted option prices between knots (Gatheral-Jacquier Lemma 5.1) --
  the only interpolation the paper proves is free of static arbitrage in
  the interpolated region. `check_interior_arbitrage` (wired into
  `fit_surface` via `check_interior=True`) gives a non-exhaustive numerical
  spot-check on top; since Lemma 5.1 already guarantees the result, this is
  a regression check, not a substantive gate.
- `fit_ssvi_surface` instead fits the single shared-rho SSVI
  parameterization (`ssvi.py`, Section 4) to the whole chain at once. The
  resulting `SSVIVolSurface` is free of static arbitrage EVERYWHERE (not
  just at knots, not just approximately between them) whenever
  `ssvi.check_ssvi_no_static_arbitrage` passes on the fitted
  (rho, phi, theta_t) -- Corollary 4.1 gives a surface-wide guarantee,
  which is the point of using SSVI at all.

Queries outside the fitted maturity range extrapolate flat in vol (constant
implied vol from the nearest fitted slice), which is equivalent to total
variance scaling linearly with T from the origin -- still consistent with
the calendar condition, for every mode above.
"""
import warnings

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

import arbitrage as arb
from svi import fit_svi_slice

MIN_POINTS_PER_SLICE = 5


def black_scholes_undiscounted_call(k, w):
    """
    Undiscounted Black-Scholes call price divided by the forward, as a
    function of log-moneyness k=ln(K/F) and total variance w (Gatheral's
    normalized form):
        c(k, w) = N(d+) - e^k * N(d-),   d+/- = -k/sqrt(w) +/- sqrt(w)/2
    No discount factor is needed since everything in this project works in
    forward/undiscounted terms. `d+` here is exactly `arbitrage.d_plus`,
    generalized to a raw (k, w) pair rather than an SVIParams slice, since
    Lemma 5.1's price-space interpolation needs to evaluate it at
    intermediate w values that have no SVIParams object of their own.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    sqrt_w = np.sqrt(np.maximum(w, 1e-300))
    d_plus = -k / sqrt_w + sqrt_w / 2.0
    d_minus = d_plus - sqrt_w
    return norm.cdf(d_plus) - np.exp(k) * norm.cdf(d_minus)


def black_scholes_undiscounted_put(k, w):
    """
    Undiscounted Black-Scholes put price divided by the forward:
        p(k, w) = e^k * N(-d-) - N(-d+)
    computed directly (not as call - parity), so it doesn't inherit the
    call formula's cancellation problem for k << 0 -- see
    `_normalized_price` for why that matters here.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    sqrt_w = np.sqrt(np.maximum(w, 1e-300))
    d_plus = -k / sqrt_w + sqrt_w / 2.0
    d_minus = d_plus - sqrt_w
    return np.exp(k) * norm.cdf(-d_minus) - norm.cdf(-d_plus)


def _normalized_price(k, w):
    """
    Undiscounted price divided by the forward, always taken from the
    out-of-the-money side: the put formula for k < 0, the call formula for
    k >= 0. For k << 0 (deep ITM call / deep OTM put -- exactly the
    short-dated far-wing case `check_interior_arbitrage` probes),
    `black_scholes_undiscounted_call` is almost entirely intrinsic value
    (~1 - e^k), so two different total variances can produce call prices
    that agree to 10+ decimal places -- the same floating-point
    cancellation problem `implied_vol.py` solves via put-call parity for
    the scalar solver. Pricing from the OTM side keeps the time-value
    signal (the only part that actually depends on w) at full precision,
    which both the interpolation and its brentq inversion depend on.
    """
    return black_scholes_undiscounted_put(k, w) if k < 0.0 else black_scholes_undiscounted_call(k, w)


def _invert_normalized_price_for_variance(k, target_price, w_bracket):
    """
    Solve _normalized_price(k, w) == target_price for w > 0, using
    whichever of call/put is well-conditioned at this k (see
    `_normalized_price`). `target_price` is a convex combination of the
    prices at w_bracket's two endpoints, and price is monotonically
    increasing in w at fixed k for both call and put, so w_bracket already
    brackets the root by construction; the widening below only guards
    against floating-point edge cases at the boundary.
    """
    lo, hi = w_bracket
    lo = max(lo, 1e-12)

    def objective(w):
        return _normalized_price(k, w) - target_price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo > 0:
        lo = 1e-12
        f_lo = objective(lo)
    if f_hi < 0:
        hi = max(hi * 4.0, 50.0)
        f_hi = objective(hi)
    if f_lo * f_hi > 0:
        hi = 200.0
    return brentq(objective, lo, hi, xtol=1e-13, maxiter=200)


def _log_linear_interp(x, xp, fp):
    """
    Interpolate fp (all > 0) log-linearly against xp at points x: linear
    interpolation of log(fp) in x, clamped to the boundary value outside
    [xp[0], xp[-1]] (flat extrapolation). `xp` must be sorted ascending.
    Used for the forward curve (price-space interpolation) and the ATM
    total-variance curve (`SSVIVolSurface`).
    """
    x = np.asarray(x, dtype=float)
    x_clamped = np.clip(x, xp[0], xp[-1])
    return np.exp(np.interp(x_clamped, xp, np.log(fp)))


class VolSurface:
    """
    Queryable surface built from a set of per-expiry raw-SVI knots.

    `implied_vol` blends between two neighboring knots T_lo < T < T_hi by
    blending undiscounted option prices, not total variance directly
    (Gatheral & Jacquier 2013, Lemma 5.1) -- the interpolation the paper
    proves is free of static arbitrage in the interpolated region. At a
    shared log-moneyness k (via a log-linearly interpolated reference
    forward for the query T), each knot's total variance is converted to
    an undiscounted Black-Scholes call price; the prices are blended with
    weight
        alpha_T = (sqrt(theta_hi) - sqrt(theta_T)) / (sqrt(theta_hi) - sqrt(theta_lo))
    where theta_lo/hi/T are the ATM total variances at T_lo/T_hi/T
    (theta_T interpolated linearly in T -- the lemma only requires
    monotonicity, not a specific shape); the blended price is inverted
    back to a total variance via `black_scholes_undiscounted_call`.

    (An earlier version of this class also offered a `linear_variance` mode
    that blended total variance directly at fixed strike -- simpler, but
    with no arbitrage-free proof, only the weaker property of staying
    between the two knots pointwise. Removed: Lemma 5.1's interpolation
    costs nothing extra to compute and is the only one of the two with a
    guarantee.)
    """
    def __init__(self, maturities, forwards, params, fit_results=None, strike_bounds=None):
        order = np.argsort(maturities)
        self.maturities = np.asarray(maturities, dtype=float)[order]
        self.forwards = np.asarray(forwards, dtype=float)[order]
        self.params = [params[i] for i in order]
        self.fit_results = [fit_results[i] for i in order] if fit_results is not None else None
        self.strike_bounds = strike_bounds

    def _slice_total_variance(self, i, K):
        k = np.log(np.asarray(K, dtype=float) / self.forwards[i])
        return self.params[i].total_variance(k)

    def _atm_total_variance(self, i):
        """theta_i = w(0) for knot i: the ATM total variance, per Lemma 5.1."""
        return float(self.params[i].total_variance(0.0))

    def implied_vol(self, K, T):
        """Interpolated (or extrapolated) implied vol at strike(s) K, maturity(ies) T."""
        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        scalar_input = K_arr.ndim == 0 and T_arr.ndim == 0
        K_arr, T_arr = np.broadcast_arrays(K_arr, T_arr)
        shape = K_arr.shape
        Kf, Tf = K_arr.ravel().astype(float), T_arr.ravel().astype(float)
        out = np.empty_like(Kf)

        Ts = self.maturities
        below = Tf <= Ts[0]
        above = Tf >= Ts[-1]
        inside = ~below & ~above

        if np.any(below):
            w0 = self._slice_total_variance(0, Kf[below])
            out[below] = np.sqrt(np.maximum(w0, 0.0) / Ts[0])

        if np.any(above):
            wL = self._slice_total_variance(len(Ts) - 1, Kf[above])
            out[above] = np.sqrt(np.maximum(wL, 0.0) / Ts[-1])

        if np.any(inside):
            Ti, Ki = Tf[inside], Kf[inside]
            idx_hi = np.clip(np.searchsorted(Ts, Ti, side="right"), 1, len(Ts) - 1)
            idx_lo = idx_hi - 1

            w = self._interpolate_price_space(idx_lo, idx_hi, Ti, Ki)

            out[inside] = np.sqrt(np.maximum(w, 0.0) / Ti)

        out = out.reshape(shape)
        return float(out) if scalar_input else out

    def _interpolate_price_space(self, idx_lo, idx_hi, Ti, Ki):
        """
        Lemma 5.1 interpolation -- see class docstring. Requires a scalar
        root-find per query point (Black-Scholes has no closed-form inverse
        in w), so this loops in Python; fine for typical pricing/plotting
        grid sizes (a few thousand points at most).
        """
        w = np.empty_like(Ki)
        for j in range(len(Ki)):
            i_lo, i_hi = int(idx_lo[j]), int(idx_hi[j])
            T_lo, T_hi = self.maturities[i_lo], self.maturities[i_hi]
            F_lo, F_hi = self.forwards[i_lo], self.forwards[i_hi]
            K, T = Ki[j], Ti[j]

            frac = (T - T_lo) / (T_hi - T_lo)
            F_T = np.exp(np.log(F_lo) + (np.log(F_hi) - np.log(F_lo)) * frac)
            k_shared = float(np.log(K / F_T))

            theta_lo = self._atm_total_variance(i_lo)
            theta_hi = self._atm_total_variance(i_hi)
            theta_T = theta_lo + (theta_hi - theta_lo) * frac

            sqrt_lo, sqrt_hi = np.sqrt(theta_lo), np.sqrt(theta_hi)
            if abs(sqrt_hi - sqrt_lo) < 1e-14:
                alpha_T = 0.5  # degenerate: knots have (near-)equal ATM level
            else:
                alpha_T = (sqrt_hi - np.sqrt(theta_T)) / (sqrt_hi - sqrt_lo)

            w_lo_k = float(self.params[i_lo].total_variance(k_shared))
            w_hi_k = float(self.params[i_hi].total_variance(k_shared))
            price_lo = _normalized_price(k_shared, w_lo_k)
            price_hi = _normalized_price(k_shared, w_hi_k)
            price_T = alpha_T * price_lo + (1.0 - alpha_T) * price_hi

            lo_b, hi_b = sorted((w_lo_k, w_hi_k))
            margin = 0.5 * max(lo_b, 1e-6)
            w[j] = _invert_normalized_price_for_variance(
                k_shared, float(price_T), (max(lo_b - margin, 1e-12), hi_b + margin)
            )
        return w


def _interior_maturities(T_lo, T_hi, n_points):
    """`n_points` maturities strictly between T_lo and T_hi (endpoints excluded)."""
    return np.linspace(T_lo, T_hi, n_points + 2)[1:-1]


def check_interior_arbitrage(surface, k_grid=arb.DEFAULT_K_GRID, n_points=4, h=1e-4):
    """
    Sample `n_points` maturities strictly between each pair of adjacent
    knots in `surface` and check, at each: (1) the interpolated smile's own
    butterfly condition g(k) >= 0, via `arbitrage.g_function`'s formula
    applied to w, w', w'' obtained by finite-differencing the interpolated
    w(k) in k (there's no closed-form SVIParams for an interpolated slice
    to hand to `arbitrage.g_function` directly); (2) the interpolated total
    variance sits between the two neighboring knots' own total variance at
    matched log-moneyness (a local calendar-monotonicity check).

    This is a finite-grid empirical check, not a proof. `VolSurface`'s
    interpolation is already guaranteed arbitrage-free by construction
    (Lemma 5.1), so this exists as a regression/sanity check on that
    guarantee rather than a substantive gate.

    Raises `arbitrage.ArbitrageError` on the first violation found.
    """
    for i in range(len(surface.maturities) - 1):
        T_lo, T_hi = surface.maturities[i], surface.maturities[i + 1]
        F_lo, F_hi = surface.forwards[i], surface.forwards[i + 1]

        for T_mid in _interior_maturities(T_lo, T_hi, n_points):
            frac = (T_mid - T_lo) / (T_hi - T_lo)
            F_mid = np.exp(np.log(F_lo) + (np.log(F_hi) - np.log(F_lo)) * frac)

            def w_at(k_shift, F_mid=F_mid, T_mid=T_mid):
                K_shift = F_mid * np.exp(k_grid + k_shift)
                iv = surface.implied_vol(K_shift, T_mid)
                return iv ** 2 * T_mid

            w = w_at(0.0)
            w_p = w_at(h)
            w_m = w_at(-h)
            dw = (w_p - w_m) / (2 * h)
            d2w = (w_p - 2 * w + w_m) / h ** 2

            g = (1.0 - k_grid * dw / (2.0 * w)) ** 2 - (dw ** 2 / 4.0) * (1.0 / w + 0.25) + d2w / 2.0
            if np.any(g < arb.BUTTERFLY_TOL):
                bad_k = k_grid[np.argmin(g)]
                raise arb.ArbitrageError(
                    f"interior butterfly arbitrage (numerical): g(k={bad_k:.4f}) = {g.min():.3e} < 0 at "
                    f"interpolated T={T_mid:.4f} (between knots T={T_lo:.4f} and T={T_hi:.4f})"
                )

            w_lo_curve = surface.params[i].total_variance(k_grid)
            w_hi_curve = surface.params[i + 1].total_variance(k_grid)

            diff_lo = w - w_lo_curve
            if np.any(diff_lo < arb.CALENDAR_TOL):
                bad_k = k_grid[np.argmin(diff_lo)]
                raise arb.ArbitrageError(
                    f"interior calendar arbitrage (numerical): interpolated total variance at T={T_mid:.4f} "
                    f"drops below the T={T_lo:.4f} knot at k={bad_k:.4f}"
                )
            diff_hi = w_hi_curve - w
            if np.any(diff_hi < arb.CALENDAR_TOL):
                bad_k = k_grid[np.argmin(diff_hi)]
                raise arb.ArbitrageError(
                    f"interior calendar arbitrage (numerical): interpolated total variance at T={T_mid:.4f} "
                    f"exceeds the T={T_hi:.4f} knot at k={bad_k:.4f}"
                )
    return True


def fit_surface(chain, validate=True, min_points=MIN_POINTS_PER_SLICE, k_grid=arb.DEFAULT_K_GRID,
                on_bad_slice="raise", check_interior=True):
    """
    Loop the SVI fit (Step 4) + butterfly validation (Step 5, via
    `fit_svi_slice`) across every expiry in `chain`, then validate the
    calendar condition across the resulting term structure. Expiries with
    fewer than `min_points` quotes are skipped outright (not enough data to
    fit 5 parameters).

    `check_interior`, when true, additionally runs `check_interior_arbitrage`
    on the fitted surface once it's built: a finite-grid, non-exhaustive
    numerical check of the interpolated region (not just the knots).
    Failures are gated by `on_bad_slice` the same way knot-level failures
    are, except there's no slice to "skip" here -- it's a validation-only
    check on interpolated points, so "skip" just warns.

    `on_bad_slice` controls what happens when a slice, the knot-level
    calendar check, or the interior check fails the no-arbitrage gate:
    - "raise" (default): propagate `arbitrage.ArbitrageError` immediately --
      the strict behavior Step 5 asks for, and what you want when calling
      this directly to find out a fit is bad.
    - "skip": for a bad slice, exclude the offending expiry (with a
      `warnings.warn`) instead of aborting the whole surface, so one
      noisy/illiquid expiry doesn't take down an otherwise-good surface.
      For the knot-level calendar check or the interior check, there's
      nothing to exclude -- just `warnings.warn` and keep going.
      `run_pipeline` uses this mode for exactly that reason.

    `chain` is the table produced by `data.build_chain`:
    columns strike, expiry, T, forward, mid_iv, weight.
    """
    if on_bad_slice not in ("raise", "skip"):
        raise ValueError(f"on_bad_slice must be 'raise' or 'skip', got {on_bad_slice!r}")

    maturities, forwards, params_list, fit_results = [], [], [], []

    for T, group in chain.groupby("T", sort=True):
        if len(group) < min_points:
            continue
        forward = float(group["forward"].iloc[0])
        k = np.log(group["strike"].to_numpy() / forward)
        try:
            result = fit_svi_slice(
                k, group["mid_iv"].to_numpy(), group["weight"].to_numpy(), T, validate=validate
            )
        except arb.ArbitrageError as exc:
            if on_bad_slice == "raise":
                raise
            warnings.warn(f"skipping expiry T={T:.4f}: {exc}")
            continue
        maturities.append(T)
        forwards.append(forward)
        params_list.append(result.params)
        fit_results.append(result)

    if not maturities:
        raise ValueError("no expiry in the chain had enough quotes to fit an SVI slice")

    if validate and len(maturities) > 1:
        try:
            arb.check_calendar(list(zip(maturities, params_list)), k_grid=k_grid)
        except arb.ArbitrageError as exc:
            if on_bad_slice == "raise":
                raise
            warnings.warn(f"calendar arbitrage across accepted slices: {exc}")

    strike_bounds = (float(chain["strike"].min()), float(chain["strike"].max()))
    surface = VolSurface(maturities, forwards, params_list, fit_results, strike_bounds)

    if check_interior and len(surface.maturities) > 1:
        try:
            check_interior_arbitrage(surface, k_grid=k_grid)
        except arb.ArbitrageError as exc:
            if on_bad_slice == "raise":
                raise
            warnings.warn(f"interior arbitrage check failed (validation-only, nothing to skip): {exc}")

    return surface


class SSVIVolSurface:
    """
    Query interface for a fitted SSVI surface (`ssvi.py`, Section 4):
    implied vol at any (K, T) is computed directly from the shared `rho`,
    `phi`, and an interpolated ATM total-variance curve theta(T) -- no
    per-knot SVIParams needed at all.

    Unlike `VolSurface` (independent per-expiry raw-SVI fits, interpolated
    after the fact), a surface built this way is free of static arbitrage
    EVERYWHERE, not just at the fitted knots, whenever
    `ssvi.check_ssvi_no_static_arbitrage` passed on
    `(maturities, theta_values, ssvi_params)` (Corollary 4.1) -- that
    surface-wide guarantee is the entire reason to build a surface this way
    instead of via `fit_surface`.

    theta(T) is interpolated linearly in T between fitted knots; the
    forward curve is interpolated log-linearly.

    Beyond the last knot t_n, `extrapolate` selects between:
    - "section_5_3" (default): the paper's Section 5.3 prescription --
      "fix a monotonic increasing extrapolation of theta_t (asymptotically
      linear in time would seem to be reasonable) and extrapolate the smile
      for t > t_n according to w(k, theta_t) = w(k, theta_{t_n}) + theta_t -
      theta_{t_n}", which is free of static arbitrage by Theorem 4.3
      whenever the final slice is free of butterfly arbitrage. The smile
      SHAPE is frozen at the final slice and only a constant is added -- it
      is Theorem 4.3's shift with alpha_t = theta_t - theta_{t_n}. theta_t
      is continued linearly at the slope of the final knot interval (the
      "asymptotically linear" choice; monotonic increasing because theta is
      non-decreasing across the knots). Pass `theta_extrapolation` to
      supply a different monotonic increasing t -> theta_t.
    - "flat_theta": hold theta constant past t_n, i.e. constant total
      variance. Calendar-admissible (d/dt w = 0) but not the paper's
      prescription; kept for comparison.
    """
    def __init__(self, maturities, forwards, ssvi_params, theta_values, strike_bounds=None,
                 extrapolate="section_5_3", theta_extrapolation=None):
        if extrapolate not in ("section_5_3", "flat_theta"):
            raise ValueError(
                f"extrapolate must be 'section_5_3' or 'flat_theta', got {extrapolate!r}"
            )
        order = np.argsort(maturities)
        self.maturities = np.asarray(maturities, dtype=float)[order]
        self.forwards = np.asarray(forwards, dtype=float)[order]
        self.theta_values = np.asarray(theta_values, dtype=float)[order]
        self.ssvi_params = ssvi_params
        self.strike_bounds = strike_bounds
        self.extrapolate = extrapolate
        self.theta_extrapolation = theta_extrapolation

    def _forward_at(self, T):
        return _log_linear_interp(T, self.maturities, self.forwards)

    def _theta_slope(self):
        """Slope of theta over the final knot interval; 0 if there's only one knot."""
        if len(self.maturities) < 2:
            return 0.0
        d_theta = self.theta_values[-1] - self.theta_values[-2]
        d_t = self.maturities[-1] - self.maturities[-2]
        return max(d_theta / d_t, 0.0)

    def _theta_at(self, T):
        """
        theta_t: linear between knots, clamped below the first knot, and
        beyond the last knot either continued (Section 5.3) or held flat.
        """
        T = np.asarray(T, dtype=float)
        T_clamped = np.clip(T, self.maturities[0], self.maturities[-1])
        theta = np.interp(T_clamped, self.maturities, self.theta_values)

        if self.extrapolate == "section_5_3":
            beyond = T > self.maturities[-1]
            if np.any(beyond):
                if self.theta_extrapolation is not None:
                    theta = np.where(beyond, self.theta_extrapolation(T), theta)
                else:
                    extra = self._theta_slope() * (T - self.maturities[-1])
                    theta = np.where(beyond, self.theta_values[-1] + extra, theta)
        return theta

    def implied_vol(self, K, T):
        import ssvi as ssvi_module

        K_arr = np.asarray(K, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        scalar_input = K_arr.ndim == 0 and T_arr.ndim == 0
        K_arr, T_arr = np.broadcast_arrays(K_arr, T_arr)
        shape = K_arr.shape
        Kf, Tf = K_arr.ravel().astype(float), T_arr.ravel().astype(float)

        F = self._forward_at(Tf)
        k = np.log(Kf / F)
        theta = self._theta_at(Tf)

        theta_n = self.theta_values[-1]
        beyond = (Tf > self.maturities[-1]) & (self.extrapolate == "section_5_3")

        # Section 5.3: w(k, theta_t) = w(k, theta_{t_n}) + theta_t - theta_{t_n}
        # past the final slice -- the shape stays frozen at theta_{t_n} and only
        # the constant alpha_t = theta_t - theta_{t_n} is added (Theorem 4.3).
        theta_for_shape = np.where(beyond, theta_n, theta)
        w = ssvi_module.ssvi_total_variance(k, theta_for_shape, self.ssvi_params)
        w = w + np.where(beyond, theta - theta_n, 0.0)

        out = np.sqrt(np.maximum(w, 0.0) / Tf).reshape(shape)
        return float(out) if scalar_input else out


def fit_ssvi_surface(chain, shape="power_law", validate=True, min_points=MIN_POINTS_PER_SLICE):
    """
    Build a surface from market data by fitting the shared `rho` and the
    ATM total-variance term structure theta_t implied by `chain`, using the
    SSVI parameterization (Definition 4.1) instead of independent
    per-maturity raw SVI. Unlike `fit_surface`, the resulting surface is
    free of static arbitrage EVERYWHERE (not just at fitted knots and not
    just approximately between them) whenever
    `ssvi.check_ssvi_no_static_arbitrage` passes on the fitted
    (rho, phi, theta_t) -- this is the point of using SSVI: Corollary 4.1
    gives a surface-wide guarantee, not just a slice-wide or knot-wide one.

    `shape` selects the shape function fit alongside rho: "power_law"
    (fits eta, gamma -- Example 4.2) or "heston_like" (fits lam --
    Example 4.1).

    Internally: pre-fits an independent raw-SVI slice per expiry (reusing
    `fit_surface(..., validate=False)` purely as a data-extraction step --
    the individual slices are discarded, not returned) to read off each
    knot's ATM total variance theta_t_i = w(0) and ATM skew w'(0), then
    fits (rho, shape params) by least squares against the closed-form SSVI
    ATM skew w'(0) = theta_t * rho * phi(theta_t).
    """
    import ssvi as ssvi_module

    knot_surface = fit_surface(
        chain, validate=False, min_points=min_points, on_bad_slice="skip", check_interior=False
    )
    Ts = knot_surface.maturities
    forwards = knot_surface.forwards
    thetas = np.array([p.total_variance(0.0) for p in knot_surface.params])
    observed_slopes = np.array([
        p.b * (p.rho - p.m / np.sqrt(p.m ** 2 + p.sigma ** 2)) for p in knot_surface.params
    ])

    if shape == "power_law":
        def phi_factory(x):
            return ssvi_module.phi_power_law(x[1], x[2])
        x0 = np.array([0.0, 1.0, 0.4])
        bounds = ([-0.999, 1e-6, 1e-6], [0.999, 20.0, 0.999])
    elif shape == "heston_like":
        def phi_factory(x):
            return ssvi_module.phi_heston_like(x[1])
        x0 = np.array([0.0, 1.0])
        bounds = ([-0.999, 1e-6], [0.999, 50.0])
    else:
        raise ValueError(f"unknown shape {shape!r}, expected 'power_law' or 'heston_like'")

    def residuals(x):
        rho = x[0]
        phi = phi_factory(x)
        predicted = thetas * rho * phi(thetas)
        return predicted - observed_slopes

    opt = least_squares(residuals, x0, bounds=bounds)
    rho_fit = float(opt.x[0])
    phi_fit = phi_factory(opt.x)
    ssvi_params = ssvi_module.SSVIParams(rho=rho_fit, phi=phi_fit)

    if validate:
        ssvi_module.check_ssvi_no_static_arbitrage(Ts, thetas, ssvi_params)

    strike_bounds = (float(chain["strike"].min()), float(chain["strike"].max()))
    return SSVIVolSurface(Ts, forwards, ssvi_params, thetas, strike_bounds=strike_bounds)


def plot_surface_3d(surface, n_strikes=40, n_maturities=40, strike_bounds=None, maturity_bounds=None, ax=None):
    """
    3D plot of strike x maturity x implied vol. Returns (fig, ax) without
    calling plt.show(), so it can be used headlessly or embedded in a
    notebook. Works with any surface exposing `.strike_bounds`,
    `.maturities`, and `.implied_vol(K, T)` -- both `VolSurface` and
    `SSVIVolSurface` qualify.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

    lo_k, hi_k = strike_bounds or surface.strike_bounds
    lo_t, hi_t = maturity_bounds or (surface.maturities[0], surface.maturities[-1])

    strikes = np.linspace(lo_k, hi_k, n_strikes)
    maturities = np.linspace(lo_t, hi_t, n_maturities)
    KK, TT = np.meshgrid(strikes, maturities)
    IV = surface.implied_vol(KK, TT)

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.plot_surface(KK, TT, IV, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied Vol")
    ax.set_title("Implied Volatility Surface")
    return fig, ax


def run_pipeline(currency="BTC", max_width_frac=0.15, validate=True, min_points=MIN_POINTS_PER_SLICE):
    """
    End-to-end, no-manual-intervention pipeline: ingest live Deribit data,
    fit every slice, validate no-arbitrage, and return the queryable
    surface (Step 6 acceptance criterion).

    Uses `on_bad_slice="skip"`: live, illiquid far-wing quotes can
    occasionally make a single expiry's unconstrained SVI fit fail the
    butterfly check even when the rest of the surface is clean, and a
    pipeline that's supposed to run with no manual intervention shouldn't
    abort the entire surface over one bad expiry. Any skip is still
    flagged via `warnings.warn`, not silently absorbed.

    Uses `fit_surface`'s default `check_interior=True`, so the returned
    surface also gets a regression spot-check on top of Lemma 5.1's
    provably arbitrage-free interpolation.
    """
    import data

    raw = data.fetch_book_summary(currency)
    chain = data.build_chain(raw, max_width_frac=max_width_frac)
    if chain.empty:
        raise ValueError(f"no liquid quotes returned for {currency}")
    return fit_surface(chain, validate=validate, min_points=min_points, on_bad_slice="skip")

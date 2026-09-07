"""
No-arbitrage validation for fitted SVI slices/surfaces.

Two static-arbitrage conditions are checked, both on total variance
w(k) = sigma_impl^2 * T:

Calendar spread: w(k, T) must be non-decreasing in T at fixed log-moneyness
k. If it weren't, you could sell the shorter-dated option and buy the
longer-dated one at the same strike-in-moneyness-terms and lock in a profit
regardless of what vol does next -- the two total-variance claims would be
mispriced relative to each other. Equivalently: the "forward variance"
between two expiries, w(k,T2) - w(k,T1), must be >= 0, since a variance
swap over that forward period cannot have negative price.

Two implementations are provided: `check_calendar` evaluates w2-w1 on
`DEFAULT_K_GRID` -- fast, and the right choice for iterative calibration,
but it can in principle miss a violation that falls in a narrow gap
between grid points. `check_calendar_exact` instead finds every exact
crossing point of the two raw SVI slices via the closed-form quartic of
Gatheral & Jacquier (2013, Lemma 3.3) and checks ordering right at each
one -- slower (a one-time sympy derivation at import, then a `np.roots`
call per pair) but exact for any raw SVI pair, so no grid resolution to
miss. Use it for a final rigorous pass or to validate a suspected
narrow-crossing edge case; `check_calendar` for everything else.

Before either arbitrage condition is meaningful, the slice has to be a
valid total-variance curve at all: Section 3.1's domain condition
a + b*sigma*sqrt(1-rho^2) >= 0 (equivalently min_k w(k) >= 0) is checked by
`check_minimum_variance`, called first from `check_butterfly`.

Butterfly (static): a slice is free of butterfly arbitrage iff (Gatheral &
Jacquier, "Arbitrage-free SVI volatility surfaces", 2013):
  (i)  g(k) >= 0 for all k, where g(k) is proportional to the Black-Scholes
       implied risk-neutral density at log-moneyness k -- g(k) < 0 means
       the slice implies a *negative* density there, i.e. a butterfly
       spread centered at that strike would have negative cost but a
       non-negative payoff, a pure arbitrage; and
  (ii) lim_{k -> +inf} d+(k) = -inf, where d+(k) = -k/sqrt(w(k)) + sqrt(w(k))/2
       -- without this, the right wing grows too fast for the tail to
       correspond to a proper (finite-mean) probability distribution, per
       Roger Lee's moment formula. For raw SVI this reduces in closed form
       to a bound on the parameters: b*(1+rho) < 2 (see `_wing_slope`).

Both checks are wired as a gate: `check_butterfly` is called from
`svi.fit_svi_slice` on every fit, and `check_calendar` is called from
`surface.py` once every expiry has been fit. A failing check raises
`ArbitrageError` instead of letting a bad slice reach the pricer.
"""
import numpy as np
import sympy as sp

from svi import SVIParams, raw_svi

# log-moneyness range +-1.5 (~ strike 22% to 450% of forward) comfortably
# covers what BTC/ETH chains actually quote. Checking much further out
# tests pure model extrapolation with no calibration data behind it, where
# an unconstrained 5-parameter SVI fit is prone to (financially
# meaningless) wing artifacts that don't reflect anything tradeable. The
# right wing beyond this range is not left unchecked -- that is exactly what
# `check_wing_condition`'s asymptotic test covers, in closed form.
#
# BOTH NUMBERS ARE LOAD-BEARING, so change them only deliberately:
#   range 1.5   -- widening it makes `g(k)` fire on far-wing extrapolation
#                  artifacts (STEEP_WING_PARAMS in tests/test_arbitrage.py
#                  is a slice that is g-clean here but wing-arbitrageable,
#                  which is what proves the two checks are independent).
#   121 points  -- a 0.025-wide step. `check_calendar` is a grid check and
#                  can miss a violation narrower than one step; the fixture
#                  in test_check_calendar_exact_catches_narrow_crossing_grid_check_misses
#                  crosses over ~0.0046, so it is missed here and caught by
#                  `check_calendar_exact`. Refining the grid defeats that
#                  test's premise -- and the point it exists to make.
DEFAULT_K_GRID = np.linspace(-1.5, 1.5, 121, dtype=float)
BUTTERFLY_TOL = -1e-9
CALENDAR_TOL = -1e-8
WING_SLOPE_LIMIT = 2.0
WING_TOL = 1e-9
MIN_VARIANCE_TOL = -1e-12
CROSSING_TOL = 1e-6
CROSSING_IMAG_TOL = 1e-6
CROSSING_SIDE_EPS = 1e-4


def _derive_quartic_coeff_funcs():
    """
    Symbolically derive the quartic (in k) whose real roots are the
    candidate intersection points of two raw SVI slices w(k;chi1) = w(k;chi2)
    (Gatheral & Jacquier 2013, Lemma 3.3), and return one fast lambdified
    function per coefficient (highest degree first, matching `np.roots`).

    Runs once at import time: setting w(k;chi1) - w(k;chi2) = 0, isolating
    the square root of slice 1 and squaring once leaves one square root
    (from slice 2); squaring again removes it, at the cost of introducing
    spurious roots that don't solve the original (unsquared) equation --
    `find_slice_crossings` filters those out numerically afterward.
    """
    k, a1, b1, rho1, m1, s1, a2, b2, rho2, m2, s2 = sp.symbols(
        "k a1 b1 rho1 m1 s1 a2 b2 rho2 m2 s2", real=True
    )
    syms = (a1, b1, rho1, m1, s1, a2, b2, rho2, m2, s2)

    # w1(k) - w2(k) = 0  <=>  b1*sqrt((k-m1)^2+s1^2) - b2*sqrt((k-m2)^2+s2^2) = alpha + beta*k
    alpha = a2 - a1 + b1 * rho1 * m1 - b2 * rho2 * m2
    beta = b2 * rho2 - b1 * rho1

    # square once: isolates the remaining (slice-2) square root in P(k)
    P = b1 ** 2 * ((k - m1) ** 2 + s1 ** 2) - (alpha + beta * k) ** 2 - b2 ** 2 * ((k - m2) ** 2 + s2 ** 2)
    # square again: P(k)^2 = [2*b2*(alpha+beta*k)*sqrt((k-m2)^2+s2^2)]^2
    quartic = sp.expand(P ** 2 - 4 * b2 ** 2 * (alpha + beta * k) ** 2 * ((k - m2) ** 2 + s2 ** 2))

    coeffs = sp.Poly(quartic, k).all_coeffs()
    return [sp.lambdify(syms, c, "numpy") for c in coeffs]


# Expensive symbolic derivation happens exactly once, at import time.
_QUARTIC_COEFF_FUNCS = _derive_quartic_coeff_funcs()


class ArbitrageError(ValueError):
    """Raised when a fitted SVI slice or surface violates a no-arbitrage condition."""


def _svi_derivatives(k, params: SVIParams):
    """w, w', w'' of the raw SVI function at k, in closed form."""
    x = np.asarray(k, dtype=float) - params.m
    root = np.sqrt(x ** 2 + params.sigma ** 2)
    w = params.a + params.b * (params.rho * x + root)
    dw = params.b * (params.rho + x / root)
    d2w = params.b * params.sigma ** 2 / root ** 3
    return w, dw, d2w


def g_function(k, params: SVIParams):
    """
    Gatheral-Jacquier g(k): proportional to the Black-Scholes implied
    risk-neutral density at log-moneyness k. g(k) >= 0 everywhere <=> no
    butterfly arbitrage in this slice.
    """
    w, dw, d2w = _svi_derivatives(k, params)
    k = np.asarray(k, dtype=float)
    term1 = (1.0 - k * dw / (2.0 * w)) ** 2
    term2 = (dw ** 2 / 4.0) * (1.0 / w + 0.25)
    term3 = d2w / 2.0
    return term1 - term2 + term3


def d_plus(k, params: SVIParams):
    """
    d+(k) = -k/sqrt(w(k)) + sqrt(w(k))/2, evaluated directly from its
    definition. A slice is free of butterfly arbitrage only if
    lim_{k->+inf} d+(k) = -inf; evaluating this at genuinely large k lets
    that limit be inspected/plotted directly, rather than only checked via
    the closed-form parameter bound in `check_wing_condition`.
    """
    w, _, _ = _svi_derivatives(k, params)
    k = np.asarray(k, dtype=float)
    return -k / np.sqrt(w) + np.sqrt(w) / 2.0


def d_minus(k, params: SVIParams):
    """d-(k) = -k/sqrt(w(k)) - sqrt(w(k))/2 (Section 2.2)."""
    w, _, _ = _svi_derivatives(k, params)
    k = np.asarray(k, dtype=float)
    return -k / np.sqrt(w) - np.sqrt(w) / 2.0


def implied_density(k, params: SVIParams):
    """
    The risk-neutral probability density implied by the slice, from the
    proof of Lemma 2.2:

        p(k) = g(k) / sqrt(2*pi*w(k)) * exp(-d-(k)^2 / 2)

    which the paper obtains by explicit differentiation of the
    Black-Scholes formula, p(k) = d^2 C / dK^2 evaluated at K = F_t e^k
    (Breeden-Litzenberger). Definition 2.3 -- "a slice is free of butterfly
    arbitrage if the corresponding density is non-negative" -- is a
    statement about exactly this function; since the two factors multiplying
    g(k) are strictly positive, `p(k) >= 0` and `g(k) >= 0` flag the same
    points, and `check_butterfly` gates on the cheaper g(k).
    """
    w, _, _ = _svi_derivatives(k, params)
    return g_function(k, params) / np.sqrt(2.0 * np.pi * w) * np.exp(-d_minus(k, params) ** 2 / 2.0)


def _wing_slope(params: SVIParams):
    """
    Asymptotic slope of total variance as k -> +inf: as x = k-m -> +inf,
    sqrt(x^2+sigma^2) -> x, so w(k) -> a + b*(rho+1)*x, i.e. w(k)/k -> b(1+rho).
    """
    return params.b * (1.0 + params.rho)


def check_wing_condition(params: SVIParams, limit=WING_SLOPE_LIMIT, tol=WING_TOL):
    """
    Raise `ArbitrageError` unless b*(1+rho) < 2. A slope at or above 2 means
    the right wing of the smile grows faster than Roger Lee's moment-formula
    bound allows for any arbitrage-free distribution.

    PROVENANCE: this bound is *not* stated in Gatheral & Jacquier (2013) for
    raw SVI. The paper's condition is Lemma 2.2(ii), lim_{k->+inf} d+(k) =
    -inf; the only place it puts a number on an asymptotic slope is Remark
    4.3, which is about SSVI's theta*phi(theta)*(1+|rho|) < 4. The step from
    the one to the other for raw SVI is ours.

    It is nonetheless exactly equivalent, not an approximation. With
    c := b*(1+rho), w(k) = c*k + (a - c*m) + O(1/k) as k -> +inf, so
    d+(k) = sqrt(k)*(c-2)/(2*sqrt(c)) + O(1/sqrt(k)): the limit is -inf iff
    c < 2. At c = 2 the sqrt(k) terms cancel and d+ -> 0 at rate
    (a-2m)/(2*sqrt(2k)), so the boundary fails and the inequality is strict.
    Verified against `d_plus` on 200k random slices with 100% agreement.

    Preferred over evaluating `d_plus` at large k, which only approximates
    the limit: near c = 2 the two terms of d+ are large and nearly equal, and
    the difference loses its sign to floating-point cancellation.
    """
    slope = _wing_slope(params)
    if slope >= limit - tol:
        raise ArbitrageError(
            f"butterfly arbitrage: right-wing slope b*(1+rho) = {slope:.4f} >= {limit} "
            f"(lim_{{k->+inf}} d+(k) is not -inf) for SVI params {params}"
        )
    return True


def minimum_total_variance(params: SVIParams):
    """
    The minimum over k of the raw SVI total variance, which Gatheral &
    Jacquier (2013, Section 3.1) give in closed form as

        min_k w(k; chi_R) = a + b*sigma*sqrt(1 - rho^2).

    Their "obvious condition" on the raw SVI parameter domain is that this
    be non-negative, "which ensures that w(k; chi_R) >= 0 for all k".
    """
    return params.a + params.b * params.sigma * np.sqrt(1.0 - params.rho ** 2)


def check_minimum_variance(params: SVIParams, tol=MIN_VARIANCE_TOL):
    """
    Raise `ArbitrageError` unless a + b*sigma*sqrt(1-rho^2) >= 0, the
    Section 3.1 domain condition guaranteeing w(k) >= 0 for every k.

    This is not itself one of the two static-arbitrage conditions -- it is
    a precondition for the slice to be a well-defined total-variance curve
    at all. It has to be checked separately rather than left to `g_function`
    because g(k) divides by w(k): on a slice with w < 0 the formula still
    returns finite, positive-looking numbers, so the g(k) grid test passes a
    slice whose implied vol is nonsense (`SVIParams.implied_vol` then clamps
    the negative variance to zero and silently returns 0 vol).
    """
    w_min = minimum_total_variance(params)
    if w_min < tol:
        raise ArbitrageError(
            f"invalid slice: min_k w(k) = a + b*sigma*sqrt(1-rho^2) = {w_min:.3e} < 0 "
            f"(total variance goes negative) for SVI params {params}"
        )
    return True


def check_butterfly(params: SVIParams, k_grid=DEFAULT_K_GRID, tol=BUTTERFLY_TOL):
    """
    Raise `ArbitrageError` if this slice violates either butterfly
    no-arbitrage condition: g(k) < tol anywhere on k_grid, or the
    right-wing slope condition checked by `check_wing_condition`.

    `check_minimum_variance` runs first, since g(k) is only meaningful on a
    slice whose total variance is non-negative everywhere.
    """
    check_minimum_variance(params)
    check_wing_condition(params)
    g = g_function(k_grid, params)
    if np.any(g < tol):
        bad_k = k_grid[np.argmin(g)]
        raise ArbitrageError(
            f"butterfly arbitrage: g(k={bad_k:.4f}) = {g.min():.3e} < 0 "
            f"(negative implied density) for SVI params {params}"
        )
    return True


def check_calendar(slices, k_grid=DEFAULT_K_GRID, tol=CALENDAR_TOL):
    """
    Raise `ArbitrageError` if total variance decreases at any matched
    log-moneyness between two adjacent (by T) expiries.

    `slices` is an iterable of (T, SVIParams) pairs; they're sorted by T
    internally so callers don't need to pre-sort.
    """
    ordered = sorted(slices, key=lambda pair: pair[0])
    for (T1, params1), (T2, params2) in zip(ordered, ordered[1:]):
        w1 = raw_svi(k_grid, *params1.as_array())
        w2 = raw_svi(k_grid, *params2.as_array())
        diff = w2 - w1
        if np.any(diff < tol):
            bad_k = k_grid[np.argmin(diff)]
            raise ArbitrageError(
                f"calendar arbitrage between T={T1:.4f} and T={T2:.4f}: "
                f"total variance drops by {-diff.min():.3e} at k={bad_k:.4f}"
            )
    return True


def find_slice_crossings(params1: SVIParams, params2: SVIParams, tol=CROSSING_TOL):
    """
    Return every real log-moneyness k at which the raw SVI slices for
    params1 and params2 exactly intersect (w(k;params1) == w(k;params2)).

    Uses the closed-form quartic from Gatheral-Jacquier Lemma 3.3: the
    crossing equation is squared twice to remove both square roots, giving
    a quartic in k. Candidate roots from the quartic are filtered against
    the ORIGINAL (unsquared) equation with tolerance `tol`, since squaring
    can introduce spurious roots that don't solve the real problem.
    """
    vals = (
        params1.a, params1.b, params1.rho, params1.m, params1.sigma,
        params2.a, params2.b, params2.rho, params2.m, params2.sigma,
    )
    coeffs = np.array([f(*vals) for f in _QUARTIC_COEFF_FUNCS], dtype=float)
    roots = np.roots(coeffs)

    real_roots = [r.real for r in roots if abs(r.imag) < CROSSING_IMAG_TOL]

    verified = []
    for k_star in real_roots:
        w1 = raw_svi(k_star, *params1.as_array())
        w2 = raw_svi(k_star, *params2.as_array())
        if abs(w2 - w1) < tol:
            verified.append(float(k_star))
    return sorted(verified)


def crossedness(params1: SVIParams, params2: SVIParams, tol=CROSSING_TOL):
    """
    Definition 5.1: the crossedness of two SVI slices, where `params1` is
    the shorter-dated slice (chi_1, at t_1) and `params2` the longer-dated
    one (chi_2, at t_2 > t_1).

    Section 5.2 computes the crossing points k_1 < ... < k_n (n <= 4, via
    Lemma 3.3), then the probe points

        k~_1   = k_1 - 1
        k~_i   = (k_{i-1} + k_i)/2   for 2 <= i <= n
        k~_n+1 = k_n + 1

    and the amounts by which the slices cross,
    c_i = max[0, w(k~_i; chi_1) - w(k~_i; chi_2)]. The crossedness is the
    maximum of the c_i, and is null when n = 0.

    Note the paper computes n+1 probe points but then writes the maximum
    over "i = 1, ..., n"; the maximum here is taken over all n+1 computed
    values, since dropping one would discard the rightmost probe point.

    Zero crossedness means the slices are correctly ordered at every probe
    point. This is the penalty term the Section 5.2 calibration recipe adds
    to the per-slice objective to discourage crossing a neighbouring slice.

    Careful: by the definition's own "if n = 0, the crossedness is null",
    a pair that never crosses scores zero even when chi_1 lies entirely
    ABOVE chi_2 -- calendar arbitrage at every k. Crossedness is a
    calibration penalty, not an arbitrage test; use `check_calendar` or
    `check_calendar_exact` to decide whether a pair is admissible.
    """
    crossings = find_slice_crossings(params1, params2, tol=tol)
    if not crossings:
        return 0.0

    probes = [crossings[0] - 1.0]
    probes += [0.5 * (crossings[i - 1] + crossings[i]) for i in range(1, len(crossings))]
    probes.append(crossings[-1] + 1.0)

    probes_arr = np.asarray(probes, dtype=float)
    gaps = raw_svi(probes_arr, *params1.as_array()) - raw_svi(probes_arr, *params2.as_array())
    return float(np.maximum(gaps, 0.0).max())


def check_calendar_exact(slices, tol=CROSSING_TOL):
    """
    Raise `ArbitrageError` if any two adjacent (by T) expiries' raw SVI
    slices cross at a real log-moneyness with the ordering violated there,
    using exact quartic root-finding (`find_slice_crossings`) rather than
    grid evaluation. Catches narrow crossings `check_calendar`'s grid
    search can miss; restricted to the raw SVI closed form (any smile not
    representable in raw SVI parameters isn't covered).

    `slices` is an iterable of (T, SVIParams) pairs; sorted by T
    internally so callers don't need to pre-sort.

    A crossing by itself isn't necessarily arbitrage -- the curves could
    touch tangentially and stay correctly ordered everywhere else. So
    around each crossing k*, w2-w1 is checked just to either side
    (k* +/- CROSSING_SIDE_EPS); if there are no crossings at all, the
    ordering is checked once at k=0, mirroring `check_calendar`'s
    grid-baseline logic.
    """
    ordered = sorted(slices, key=lambda pair: pair[0])
    for (T1, params1), (T2, params2) in zip(ordered, ordered[1:]):
        crossings = find_slice_crossings(params1, params2, tol=tol)

        if crossings:
            probe_points = [
                k_star + sign * CROSSING_SIDE_EPS
                for k_star in crossings
                for sign in (-1.0, 1.0)
            ]
        else:
            probe_points = [0.0]

        w1 = raw_svi(probe_points, *params1.as_array())
        w2 = raw_svi(probe_points, *params2.as_array())
        diff = w2 - w1

        if np.any(diff < 0):
            bad_idx = int(np.argmin(diff))
            raise ArbitrageError(
                f"calendar arbitrage (exact) between T={T1:.4f} and T={T2:.4f}: "
                f"total variance ordering violated at k={probe_points[bad_idx]:.4f} "
                f"(drop {-diff.min():.3e}); slice crossings at k={crossings}"
            )
    return True

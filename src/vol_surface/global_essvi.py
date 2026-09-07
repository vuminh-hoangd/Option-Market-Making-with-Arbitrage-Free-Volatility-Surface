"""
Global eSSVI: an arbitrage-free volatility surface by construction.

Implements Mingone, "No arbitrage global parametrization for the eSSVI
volatility surface" (arXiv:2204.00312), referred to throughout as "the
paper". Section references in this module are to that paper unless stated
otherwise.

The eSSVI surface extends SSVI by letting rho vary with maturity. Writing
psi := theta*phi, one slice's total variance is (eq. 1)

    w(k) = 1/2 * [theta + rho*psi*k + sqrt((psi*k + theta*rho)^2 + theta^2(1-rho^2))]

so a surface with N quoted expiries is 3N numbers: (theta_i, rho_i, psi_i).

What makes this parametrization worth having is that those 3N numbers are
*not* what gets optimized. The no-arbitrage conditions (eq. 3) couple every
slice to its neighbours through inequalities, so a naive optimizer over
(theta, rho, psi) either wanders into arbitrage or has to be fenced in with
penalty terms and post-hoc rejection. Instead, Section 3 reparametrizes the
admissible region as an open hyperrectangle

    rho_i in ]-1,1[,  theta_1 in ]0,inf[,  a_i in ]0,inf[,  c_i in ]0,1[

and gives an explicit map `unbox` from that box onto surfaces satisfying
eq. 3. Proposition 3.1: *every* point of the box maps to an arbitrage-free
surface, and every arbitrage-free eSSVI surface is hit. So the calibration
in `calibrate` is an ordinary box-constrained least-squares problem, and
absence of arbitrage is a property of the parametrization rather than
something checked and repaired afterward.

`check_no_arbitrage` still exists, but as a test of this implementation
against `arbitrage.py`'s independent numerical machinery -- not as a gate
the calibration relies on for correctness.

Relationship to the rest of the package: this is a standalone module. It
reuses `arbitrage.g_function`/`check_butterfly` (via `essvi_slice_to_raw_svi`,
which routes through `ssvi.ssvi_slice_to_raw_svi`), `bs` for pricing, and
`data` for the Deribit chain, but modifies none of them. In particular it
does not touch `ssvi.SSVIParams`, whose `rho` is a single scalar for the
whole surface -- the one thing eSSVI exists to relax.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

import arbitrage as arb
import data
import ssvi
from svi import SVIParams

# Prop 3.1 needs the *open* box, but `least_squares` bounds are inclusive,
# so the optimizer is given a box inset by these margins. See `param_bounds`.
DEFAULT_EPS = 1e-8
# Section 4.4 calibrates over ]-0.95, 0.95[: at |rho| -> 1 the slice
# degenerates (sqrt(1-rho^2) -> 0) and both butterfly bounds lose precision.
DEFAULT_RHO_MARGIN = 0.05

# `f_mm` searches l over (l2, l2 + MM_SEARCH_SPAN]; see `f_mm` for why
# truncating an infimum over an unbounded ray is safe here.
MM_SEARCH_SPAN = 50.0
MM_GRID_POINTS = 200

# Relative haircut applied to the necessary condition psi <= 4/(1+|rho|) to
# keep it strict. See `psi_bound`. Comfortably above `arbitrage.WING_TOL`
# (1e-9 absolute on a slope of 2) and far below quoting precision.
WING_STRICTNESS = 1e-8

# Plot colormap. "viridis" is matplotlib's default and what `surface.py`'s own
# plot_surface_3d uses, so every surface in the package looks the same. It is
# perceptually uniform and monotone in lightness, which is what a magnitude scale
# needs; unlike a true rainbow (jet/hsv) it does not invent bands the data has no
# and it stays readable under colour-vision deficiency.
#
# Pass any matplotlib colormap name, or a list of hex steps for a custom ramp
# (e.g. VOL_SURFACE_BLUES below), via the `colorscale` argument.
VOL_SURFACE_CMAP = "viridis"
VOL_SURFACE_BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
# One camera for every 3-D surface in the package. Two surfaces drawn from
# different angles cannot be compared by eye -- the viewer re-reads the shape
# instead of the difference -- so orientation is a shared constant, not a
# per-plot choice. Matplotlib takes degrees; the plotly eye vector below is the
# same viewpoint expressed as a camera position.
VOL_SURFACE_VIEW = {"elev": 24, "azim": -131}
VOL_SURFACE_CAMERA_EYE = {"x": -1.6, "y": -1.5, "z": 0.85}
# Half-width of the default log-moneyness window, in standard deviations of the
# LONGEST calibrated maturity. See `vol_surface_grid` for why this is derived
# from theta rather than hardcoded.
DEFAULT_PLOT_SIGMAS = 2.5


# ---------------------------------------------------------------------------
# The slice
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ESSVISlice:
    """
    One eSSVI expiry: theta (ATM total variance), rho (skew), psi (= theta*phi).

    Note psi -- not phi -- is the stored parameter. Every no-arbitrage
    condition in the paper is expressed in psi, and the calendar condition
    in particular (psi non-decreasing, Section 2.1) has no equally clean
    statement in phi.
    """
    theta: float
    rho: float
    psi: float

    def total_variance(self, k):
        return essvi_total_variance(k, self.theta, self.rho, self.psi)

    def implied_vol(self, k, T):
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / T)

    @property
    def phi(self):
        """The SSVI shape-function value phi = psi/theta for this slice."""
        return self.psi / self.theta


def essvi_total_variance(k, theta, rho, psi):
    """
    Equation (1): eSSVI total variance at log-forward-moneyness k, vectorized.

        w(k) = 1/2 [theta + rho*psi*k + sqrt((psi*k + theta*rho)^2 + theta^2(1-rho^2))]

    The discriminant is a sum of squares up to the (1-rho^2) factor, so it
    is non-negative for any |rho| <= 1 and w(k) > 0 for theta > 0 -- unlike
    raw SVI, an eSSVI slice cannot accidentally imply negative variance.
    """
    k = np.asarray(k, dtype=float)
    disc = (psi * k + theta * rho) ** 2 + theta ** 2 * (1.0 - rho ** 2)
    return 0.5 * (theta + rho * psi * k + np.sqrt(disc))


def essvi_slice_to_raw_svi(sl: ESSVISlice, alpha: float = 0.0) -> SVIParams:
    """
    Convert an eSSVI slice to raw SVI, so it can be handed to `arbitrage.py`'s
    g(k) machinery unchanged. The paper (p. 4) gives the map as

        a = theta(1-rho^2)/2,  b = psi/2,  m = -theta*rho/psi,
        sigma = theta*sqrt(1-rho^2)/psi

    which is exactly `ssvi.ssvi_slice_to_raw_svi` evaluated at phi = psi/theta
    (substitute zeta = phi into its Lemma 3.1 mapping and the two agree term
    by term), so that function is reused rather than the algebra repeated
    here. The constant-valued lambda is because `SSVIParams.phi` is a shape
    *function* theta -> phi(theta): eSSVI has no such global function, only a
    per-slice value, which is the whole point of the extension.
    """
    return ssvi.ssvi_slice_to_raw_svi(
        sl.theta,
        ssvi.SSVIParams(rho=sl.rho, phi=lambda _theta, _p=sl.phi: _p),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Butterfly bounds: f(theta, |rho|)
# ---------------------------------------------------------------------------
# Both bounds cap psi^2 from above; larger f = weaker constraint = more
# admissible surfaces. They enter the parametrization only through
# `psi_bound` below.

def l2(abs_rho):
    """
    The lower end l_2(|rho|) of the ray over which `f_mm` takes its infimum:

        l_2(|rho|) = cot(arccos(-|rho|) / 3)

    The paper writes this as tan(arccos(-|rho|)/3)^-1, which is ambiguous
    between a reciprocal and an arctangent; the reciprocal (cot) reading is
    the one that matches the primary source (Martini & Mingone,
    arXiv:2106.02418) and is confirmed numerically by `test_l2_is_root_of_g2`:
    l_2 is exactly the root of g_2(., |rho|), which is what the ray needs to
    start at for `f_mm`'s denominator to stay positive.

    Ranges over (1/sqrt(3), sqrt(3)] as |rho| goes 1 -> 0.
    """
    abs_rho = np.asarray(abs_rho, dtype=float)
    return 1.0 / np.tan(np.arccos(-abs_rho) / 3.0)


def _mm_terms(l, abs_rho):
    """
    N and the derived quantities of Section 2.2.2, all differentiated with
    respect to l (the paper defines N and then says "derivatives are taken
    with respect to l"; they are written out explicitly here):

        N     = sqrt(1-rho^2) + rho*l + sqrt(l^2+1)
        N'    = rho + l/sqrt(l^2+1)
        N''   = 1/(l^2+1)^(3/2)
        g     = N'/4
        h     = 1 - (l - rho/sqrt(1-rho^2)) * N' / (2N)
        g_2   = N'' - N'^2/(2N)

    Evaluated at rho = |rho| >= 0: the butterfly bound depends on rho only
    through its magnitude (the smile is mirrored, not reshaped, by rho -> -rho),
    and taking the positive branch is what makes the l -> inf limit come out
    to the Roger Lee bound -- see `f_mm`.
    """
    l = np.asarray(l, dtype=float)
    root = np.sqrt(1.0 - abs_rho ** 2)
    s = np.sqrt(l ** 2 + 1.0)

    N = root + abs_rho * l + s
    dN = abs_rho + l / s
    d2N = 1.0 / s ** 3

    g = dN / 4.0
    h = 1.0 - (l - abs_rho / root) * dN / (2.0 * N)
    g2 = d2N - dN ** 2 / (2.0 * N)
    return g, h, g2


def f_gj(theta, abs_rho):
    """
    The Gatheral-Jacquier butterfly bound (Section 2.2.1), sufficient but not
    necessary:

        f_GJ(theta, |rho|) = 4*theta / (1 + |rho|)

    This is Theorem 4.2 of Gatheral & Jacquier, "Arbitrage-free SVI volatility
    surfaces" (arXiv:1204.0646, in notes/ as `1204.0646v4.pdf`), restated in
    psi. That theorem's two conditions are theta*phi*(1+|rho|) < 4 and
    theta*phi^2*(1+|rho|) <= 4; substituting phi = psi/theta turns the first
    into psi < 4/(1+|rho|) -- the necessary condition, handled separately by
    `psi_bound` -- and the second into psi^2 <= 4*theta/(1+|rho|), this
    function.

    Closed form, and the default: `f_mm` is weaker (admits more surfaces) but
    costs a numerical minimization per slice per objective evaluation.
    """
    return 4.0 * float(theta) / (1.0 + float(abs_rho))


def f_mm(theta, abs_rho, span=MM_SEARCH_SPAN, grid_points=MM_GRID_POINTS):
    """
    The Martini-Mingone butterfly bound (Section 2.2.2), necessary *and*
    sufficient:

        f_MM(theta,|rho|) = inf_{l > l_2(|rho|)}
              4*theta*sqrt(1-rho^2)*h^2 / [theta*sqrt(1-rho^2)*g^2 - g_2]

    (Martini & Mingone, arXiv:2106.02418, Proposition 6.3.) Being necessary
    as well as sufficient it is never tighter than `f_gj`, so it admits
    strictly more surfaces -- at the cost the paper flags in Section 2.3:
    "they are not explicit and require to use a minimization algorithm to
    evaluate f_MM(theta,|rho|), causing an increase in calibration time."

    TRUNCATION. The infimum is over an unbounded ray but is searched on
    (l_2, l_2 + span]. The tail is not discarded, it is added back in closed
    form: as l -> inf, h -> 1/2, g -> (1+|rho|)/4, and g_2 -> 0 (N'' ~ l^-3
    while N'^2/2N ~ (1+|rho|)/2l, so g_2 -> 0 from below), giving

        limit = 4*theta*r*(1/4) / [theta*r*(1+|rho|)^2/16] = 16/(1+|rho|)^2

    with r = sqrt(1-rho^2) cancelling out. An infimum can never exceed a
    limit of its own integrand, so taking the min of the searched window
    against that limit is exact wherever the objective is still descending
    at the window edge -- which it is for large theta: at theta=5, |rho|=0
    the window alone returns 16.14 against a true infimum of 16.

    That limit is also exactly (4/(1+|rho|))^2, the square of the *necessary*
    condition psi <= 4/(1+|rho|). So for f = f_MM the first term of
    `psi_bound`'s min is never the binding one; it stays there for f = f_GJ,
    where it does bind (f_GJ has no such ceiling and exceeds it once
    theta > 4/(1+|rho|)).

    The denominator is positive throughout: g_2 < 0 strictly for l > l_2
    (l_2 being its root, see `l2`), so both terms of
    theta*sqrt(1-rho^2)*g^2 - g_2 are positive and no pole can occur inside
    the search window.
    """
    theta = float(theta)
    abs_rho = float(abs_rho)
    root = np.sqrt(1.0 - abs_rho ** 2)
    lo = float(l2(abs_rho))

    def objective(l):
        g, h, g2 = _mm_terms(l, abs_rho)
        return 4.0 * theta * root * h ** 2 / (theta * root * g ** 2 - g2)

    # Coarse scan first: the objective is not unimodal for every (theta,rho),
    # so handing the whole ray straight to a local method can converge to the
    # wrong basin. Log spacing concentrates points near l_2, where the
    # denominator is smallest and the objective moves fastest.
    grid = lo + np.geomspace(1e-6, span, grid_points)
    values = objective(grid)
    best = int(np.argmin(values))

    # Polish within the bracket straddling the best grid point.
    left = grid[best - 1] if best > 0 else lo + 1e-9
    right = grid[best + 1] if best < len(grid) - 1 else grid[-1]
    polished = minimize_scalar(objective, bounds=(left, right), method="bounded")

    tail_limit = 16.0 / (1.0 + abs_rho) ** 2
    return float(min(values[best], polished.fun, tail_limit))


BUTTERFLY_BOUNDS = {"GJ": f_gj, "MM": f_mm}


def resolve_bound(butterfly_bound):
    """Map the "GJ"/"MM" selector (or a callable) to an f(theta,|rho|)."""
    if callable(butterfly_bound):
        return butterfly_bound
    try:
        return BUTTERFLY_BOUNDS[butterfly_bound]
    except KeyError:
        raise ValueError(
            f"butterfly_bound must be one of {sorted(BUTTERFLY_BOUNDS)} or a "
            f"callable (theta, abs_rho) -> float; got {butterfly_bound!r}"
        ) from None


def psi_bound(theta, rho, f_fn=f_gj, strictness=WING_STRICTNESS):
    """
    The largest psi this slice can carry without butterfly arbitrage --
    the quantity the paper calls f_i in eq. (5):

        f_i = min( 4/(1+|rho_i|),  sqrt(f(theta_i, |rho_i|)) )

    combining the necessary condition (Section 2.2, first term) with the
    chosen sufficient or necessary-and-sufficient bound (second term).

    DELIBERATE DEVIATION: the first term carries a relative haircut of
    `strictness`, so the bound returned is 4(1-strictness)/(1+|rho|) rather
    than 4/(1+|rho|). Equation (3) writes this ceiling non-strictly
    (psi <= min(...)), but the condition it comes from -- Gatheral-Jacquier
    Theorem 4.2, theta*phi*(1+|rho|) < 4 -- is strict, and the endpoint is
    not merely a measure-zero curiosity: at psi = 4/(1+|rho|) exactly the
    raw-SVI right-wing slope b(1+rho) equals precisely 2, so d+(k) fails to
    tend to -infinity and Roger Lee's moment bound is violated. Without the
    haircut, `arbitrage.check_wing_condition` rejects such a slice, and it
    is reachable: `unbox` puts psi_i at c_i*(C_i - A_i) + A_i, so once
    C_i - A_i shrinks below float resolution -- which happens on surfaces
    with large theta and swinging rho -- psi_i lands on C_i exactly for any
    c_i < 1. Insetting c away from 1 cannot prevent that; only moving the
    ceiling can.

    The second term needs no such haircut: Theorem 4.2's other condition,
    theta*phi^2*(1+|rho|) <= 4, is genuinely non-strict.
    """
    abs_rho = abs(float(rho))
    necessary = 4.0 * (1.0 - strictness) / (1.0 + abs_rho)
    return min(necessary, np.sqrt(f_fn(theta, abs_rho)))


# ---------------------------------------------------------------------------
# The global reparametrization (Section 3)
# ---------------------------------------------------------------------------
def calendar_ratio(rho_prev, rho):
    """
    p_i = max( (1+rho_{i-1})/(1+rho_i),  (1-rho_{i-1})/(1-rho_i) )

    the factor by which psi must grow from slice i-1 to slice i to keep the
    calendar-spread condition of Section 2.1 (Hendriks & Martini, Prop 3.5).

    Always >= 1, with equality iff rho_i == rho_{i-1}: if rho increases one
    ratio falls below 1 but the other rises above it, and vice versa. That
    is what forces psi to be non-decreasing across maturities regardless of
    which way the skew moves.
    """
    return max((1.0 + rho_prev) / (1.0 + rho), (1.0 - rho_prev) / (1.0 - rho))


def unbox(rho, theta_1, a, c, f_fn=f_gj):
    """
    Map a point of the open hyperrectangle (eq. 4) to eSSVI slices, via the
    recovery formulas of eq. (5)-(6). This is the heart of the paper.

    Arguments (all plain floats/sequences, no market data involved):
        rho     -- length N, each in ]-1, 1[
        theta_1 -- scalar > 0, the first slice's ATM total variance
        a       -- length N-1, each > 0. NOTE the offset: a[j] is the
                   paper's a_{j+2}, i.e. these are a_2..a_N. There is no
                   a_1, because theta_1 is itself a free parameter.
        c       -- length N, each in ]0, 1[
        f_fn    -- butterfly bound, `f_gj` (default) or `f_mm`

    Returns a list of N `ESSVISlice`.

    Proposition 3.1 guarantees the result is free of both calendar and
    butterfly arbitrage for *any* input in the open box, and that every
    arbitrage-free eSSVI surface arises this way. So this function is a
    bijection onto the admissible set, and calibration reduces to
    unconstrained-in-spirit optimization over a box.

    Structure, in three passes:

    1. theta, forward: theta_i = theta_{i-1}*p_i + a_i. Since p_i >= 1 and
       a_i > 0, theta is strictly increasing -- the calendar condition on
       theta, satisfied by construction rather than checked.

    2. f_i = `psi_bound`(theta_i, rho_i): the butterfly ceiling on each psi.
       Needs every theta first, hence a separate pass.

    3. psi, forward again but with a backward-looking ceiling. Slice i's psi
       is confined to [A_i, C_i] and placed inside it by c_i:

           A_i = psi_{i-1} * p_i        (calendar floor; A_1 = 0)
           C_i = min( (psi_{i-1}/theta_{i-1})*theta_i,   <- calendar ceiling
                      f_i,                                <- own butterfly
                      f_{i+1}/p_{i+1}, ...,               <- see below
                      f_N / prod_{m=i+1..N} p_m )
           psi_i = c_i*(C_i - A_i) + A_i

    The look-ahead terms in C_i are the subtle part, and the reason unbox
    cannot be written as a simple forward recursion. Choosing psi_i also
    sets a floor on psi_{i+1} (namely psi_i*p_{i+1}), which must itself fit
    under f_{i+1}; iterating, psi_i * p_{i+1}...p_j <= f_j for every future
    j. Without those terms a greedy choice of psi_i could leave slice j
    with an empty interval [A_j, C_j] and no arbitrage-free continuation.
    Including them makes every future interval non-empty by construction --
    which is exactly what lets Prop 3.1 quantify over the whole open box.
    """
    rho = np.asarray(rho, dtype=float)
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    n = len(rho)

    if n < 1:
        raise ValueError("need at least one maturity")
    if len(a) != n - 1:
        raise ValueError(
            f"a holds a_2..a_N, so it must have len(rho)-1 = {n - 1} entries, got {len(a)}"
        )
    if len(c) != n:
        raise ValueError(f"c must have len(rho) = {n} entries, got {len(c)}")
    if np.any(np.abs(rho) >= 1.0):
        raise ValueError("every rho must lie strictly inside ]-1, 1[")

    # p_i, 1-indexed in the paper; p[i] here is p_{i+1}, defined for i >= 1.
    p = np.ones(n, dtype=float)
    for i in range(1, n):
        p[i] = calendar_ratio(rho[i - 1], rho[i])

    # Pass 1: theta.
    theta = np.empty(n, dtype=float)
    theta[0] = float(theta_1)
    for i in range(1, n):
        theta[i] = theta[i - 1] * p[i] + a[i - 1]

    # Pass 2: per-slice butterfly ceilings.
    f = np.array([psi_bound(theta[i], rho[i], f_fn) for i in range(n)])

    # Pass 3: psi.
    psi = np.empty(n, dtype=float)
    for i in range(n):
        if i == 0:
            floor = 0.0
            ceil = np.inf
        else:
            floor = psi[i - 1] * p[i]
            ceil = (psi[i - 1] / theta[i - 1]) * theta[i]

        # own butterfly bound, then each future slice's discounted by the
        # calendar growth required to reach it
        running = 1.0
        ceil = min(ceil, f[i])
        for j in range(i + 1, n):
            running *= p[j]
            ceil = min(ceil, f[j] / running)

        psi[i] = c[i] * (ceil - floor) + floor

    return [ESSVISlice(theta=float(theta[i]), rho=float(rho[i]), psi=float(psi[i]))
            for i in range(n)]


# ---------------------------------------------------------------------------
# Packing the box for the optimizer
# ---------------------------------------------------------------------------
def pack(rho, theta_1, a, c):
    """Flatten free parameters into the 3N vector `least_squares` optimizes."""
    return np.concatenate([np.asarray(rho, dtype=float),
                           [float(theta_1)],
                           np.asarray(a, dtype=float),
                           np.asarray(c, dtype=float)])


def unpack(x, n):
    """Inverse of `pack`: split the 3N vector into (rho, theta_1, a, c)."""
    x = np.asarray(x, dtype=float)
    if len(x) != 3 * n:
        raise ValueError(f"expected 3N = {3 * n} parameters for N = {n} maturities, got {len(x)}")
    return x[:n], float(x[n]), x[n + 1:2 * n], x[2 * n:]


def param_bounds(n, eps=DEFAULT_EPS, rho_margin=DEFAULT_RHO_MARGIN, theta_max=np.inf):
    """
    Box bounds for `least_squares`, inset from eq. (4)'s *open* hyperrectangle.

    The inset is not cosmetic. `least_squares` treats bounds as inclusive and
    will sit exactly on one when the data pull it there, but Prop 3.1 is
    stated on the open box, and the endpoints genuinely misbehave:

      - c_i = 0 collapses psi_i onto its floor A_i = psi_{i-1}*p_i, which
        makes the calendar inequality of eq. (3) non-strict; for i = 1 it
        sets psi_1 = 0 outright, a degenerate slice with no smile.
      - a_i = 0 allows theta_i = theta_{i-1}*p_i, which equals theta_{i-1}
        whenever rho is flat -- violating the strict theta_2 > theta_1.
      - |rho| = 1 makes sqrt(1-rho^2) vanish; Section 4.4 calibrates over
        ]-0.95, 0.95[, which `rho_margin` reproduces.
      - c_i = 1 puts psi_i exactly on its ceiling. Equation (3) writes that
        ceiling non-strictly (psi_2 <= min(...)), which suggests the endpoint
        is admissible, but it is not when the 4/(1+|rho|) term is the binding
        one: Gatheral-Jacquier's Theorem 4.2 states that condition strictly
        (theta*phi*(1+|rho|) < 4), and at equality the right-wing slope
        b(1+rho) hits exactly 2, so d+(k) no longer tends to -infinity and
        Roger Lee's moment bound is violated. `arbitrage.check_wing_condition`
        enforces the strict version and rejects such a slice.

        This only bites for theta > 4/(1+|rho|) >= 2 -- below that
        sqrt(f(theta,|rho|)) is the smaller term and the ceiling is strict
        anyway -- so it is unreachable for realistic total variances. Inset
        regardless, since eps costs nothing and the alternative is a
        calibration that fails validation only on extreme surfaces.
    """
    lower = np.concatenate([np.full(n, -1.0 + rho_margin), [eps], np.full(n - 1, eps), np.full(n, eps)])
    upper = np.concatenate([np.full(n, 1.0 - rho_margin), [theta_max], np.full(n - 1, np.inf), np.full(n, 1.0 - eps)])
    return lower, upper


# ---------------------------------------------------------------------------
# The surface: interpolation and extrapolation (Section 5)
# ---------------------------------------------------------------------------
class GlobalESSVISurface:
    """
    A calibrated Global eSSVI surface, queryable at any (K, T).

    Holds N calibrated slices at the quoted expiries plus the forward curve
    needed to turn strikes into log-forward-moneyness, and fills the
    continuum between and beyond them by Section 5's rules -- which are
    themselves arbitrage-preserving, so the surface is arbitrage-free
    everywhere, not just on the pillars.
    """

    def __init__(self, maturities, slices, forwards):
        maturities = np.asarray(maturities, dtype=float)
        forwards = np.asarray(forwards, dtype=float)
        if not (len(maturities) == len(slices) == len(forwards)):
            raise ValueError("maturities, slices and forwards must have equal length")
        if len(maturities) == 0:
            raise ValueError("need at least one calibrated maturity")
        if np.any(np.diff(maturities) <= 0):
            raise ValueError(f"maturities must be strictly increasing, got {maturities}")
        if np.any(maturities <= 0):
            raise ValueError("maturities must be positive")

        self.maturities = maturities
        self.slices = list(slices)
        self.forwards = forwards

    def __len__(self):
        return len(self.maturities)

    def __repr__(self):
        return (f"GlobalESSVISurface(N={len(self)}, "
                f"T={np.round(self.maturities, 4).tolist()})")

    # -- parameters at an arbitrary maturity -------------------------------
    def slice_at(self, t):
        """
        The eSSVI parameters at any t > 0, by Section 5.

        INTERPOLATION (5.1), for T_i <= t <= T_{i+1} with
        lam = (t - T_i)/(T_{i+1} - T_i): theta, psi and the *product* rho*psi
        are each linear in lam, and rho is recovered as (rho*psi)/psi.

        Interpolating rho directly instead would be the natural-looking
        mistake and it breaks the model: the calendar condition of Section 2.1
        is an inequality on psi and on the products rho*psi, and linear
        interpolation preserves an inequality between two endpoints only for
        the quantities that appear in it linearly. rho is not one of them.

        EXTRAPOLATION BEFORE T_1 (5.2.1), lam = t/T_1: theta and psi both
        scale by lam, rho is held. Both scaling together is what matters --
        it makes w(k~, t) = lam * w(k, T_1) under k~ = lam*k, an exact
        rescaling of an arbitrage-free slice. Holding psi fixed instead would
        break that identity and can introduce butterfly arbitrage on any
        slice calibrated near its bound at T_1.

        EXTRAPOLATION AFTER T_N (5.2.2), lam = t/T_N: theta scales by lam,
        psi and rho are held. DEVIATION: Section 5.2.2 continues theta along
        the slope of the last segment, theta_N + (theta_N - theta_{N-1}) /
        (T_N - T_{N-1}) * (t - T_N). The form used here is
        theta_N + (theta_N/T_N)*(t - T_N), i.e. the same construction with
        the terminal slope replaced by the average slope theta_N/T_N. The
        paper's proof asks only for a positive angular coefficient, which
        theta_N/T_N is, so absence of arbitrage carries over; the average
        slope is preferred because the terminal slope is estimated from a
        single pair of expiries and goes badly wrong when the last two are
        close together or the last is illiquid.
        """
        t = float(t)
        if t <= 0:
            raise ValueError(f"maturity must be positive, got {t}")

        T = self.maturities
        first, last = self.slices[0], self.slices[-1]

        if t < T[0]:
            lam = t / T[0]
            return ESSVISlice(theta=lam * first.theta, rho=first.rho, psi=lam * first.psi)

        if t > T[-1]:
            lam = t / T[-1]
            return ESSVISlice(theta=lam * last.theta, rho=last.rho, psi=last.psi)

        i = int(np.searchsorted(T, t, side="right")) - 1
        if i >= len(T) - 1:          # t == T[-1]
            return last
        lo, hi = self.slices[i], self.slices[i + 1]
        lam = (t - T[i]) / (T[i + 1] - T[i])

        theta = (1 - lam) * lo.theta + lam * hi.theta
        psi = (1 - lam) * lo.psi + lam * hi.psi
        rho_psi = (1 - lam) * lo.rho * lo.psi + lam * hi.rho * hi.psi
        return ESSVISlice(theta=theta, rho=rho_psi / psi, psi=psi)

    def forward_at(self, t):
        """
        F(t), log-linearly interpolated across the quoted expiries.

        Linear in log F means piecewise-constant instantaneous forward rate,
        the standard convention; beyond the pillars the nearest segment's
        rate is continued, so the basis term structure flattens rather than
        the forward itself. With one expiry there is nothing to interpolate
        and F is constant.
        """
        t = float(t)
        T = self.maturities
        if len(T) == 1:
            return float(self.forwards[0])

        log_f = np.log(self.forwards)
        if t < T[0]:
            slope = (log_f[1] - log_f[0]) / (T[1] - T[0])
            return float(np.exp(log_f[0] + slope * (t - T[0])))
        if t > T[-1]:
            slope = (log_f[-1] - log_f[-2]) / (T[-1] - T[-2])
            return float(np.exp(log_f[-1] + slope * (t - T[-1])))
        return float(np.exp(np.interp(t, T, log_f)))

    # -- queries -----------------------------------------------------------
    def total_variance(self, k, T):
        """Total variance at log-forward-moneyness k and maturity T."""
        return self.slice_at(T).total_variance(k)

    def implied_vol(self, K, T):
        """
        Black-Scholes implied volatility at strike K and maturity T.

        K may be scalar or array-like; T must be scalar (one slice per call).
        Returns sigma = sqrt(w(k,T)/T) with k = log(K/F(T)).
        """
        K = np.asarray(K, dtype=float)
        if np.any(K <= 0):
            raise ValueError("strikes must be positive")
        k = np.log(K / self.forward_at(T))
        w = self.slice_at(T).total_variance(k)
        vol = np.sqrt(np.maximum(w, 0.0) / float(T))
        return float(vol) if vol.ndim == 0 else vol


# ---------------------------------------------------------------------------
# Calibration (Section 4)
# ---------------------------------------------------------------------------
class CalibrationResult:
    """A fitted surface plus enough of the fit's diagnostics to judge it."""

    def __init__(self, surface, opt_result, quotes, weights, model_prices):
        self.surface = surface
        self.opt_result = opt_result
        self.success = bool(opt_result.success)
        self.n_quotes = len(quotes.price)
        self.price_residuals = model_prices - quotes.price
        self.weighted_rmse = float(
            np.sqrt(np.average(self.price_residuals ** 2, weights=weights))
        )
        self.max_abs_error = float(np.max(np.abs(self.price_residuals)))

    def __repr__(self):
        return (f"CalibrationResult(success={self.success}, n_quotes={self.n_quotes}, "
                f"weighted_rmse={self.weighted_rmse:.6g}, {self.surface!r})")


class _Quotes:
    """Validated, expiry-grouped quote set. Internal to `calibrate`."""

    def __init__(self, strike, T, price, option_type, forward, mid_iv=None):
        self.strike = np.asarray(strike, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.price = np.asarray(price, dtype=float)
        self.option_type = np.asarray(option_type, dtype=object)
        self.forward = np.asarray(forward, dtype=float)
        self.mid_iv = None if mid_iv is None else np.asarray(mid_iv, dtype=float)

        sizes = {len(self.strike), len(self.T), len(self.price),
                 len(self.option_type), len(self.forward)}
        if len(sizes) != 1:
            raise ValueError("all quote arrays must have the same length")
        if np.any(self.T <= 0):
            raise ValueError("all maturities must be positive")
        if np.any(self.strike <= 0):
            raise ValueError("all strikes must be positive")

        self.maturities = np.unique(self.T)
        # index of the pillar each quote belongs to
        self.expiry_index = np.searchsorted(self.maturities, self.T)
        self.k = np.log(self.strike / self.forward)

    @property
    def n_expiries(self):
        return len(self.maturities)

    def pillar_forwards(self, rtol=1e-3):
        """
        One forward per expiry, for the surface's forward curve.

        Deribit reports `underlying_price` per instrument, and it is the
        forward for that instrument's expiry, so every quote at one expiry
        should agree -- but only to within snapshot noise: the book summary is
        assembled instrument by instrument and the index moves between them, so
        a live BTC chain routinely shows a few parts per million of spread
        across one expiry. `rtol` is set well above that and well below
        anything that would matter, since a 1e-3 error in F moves k by 1e-3
        against a typical listed strike spacing of ~0.05. What it is really
        guarding against is a structurally broken snapshot -- expiries parsed
        into the wrong group, or a stale instrument -- which shows up as a
        spread orders of magnitude larger.

        Note this affects only the curve used by `implied_vol(K, T)` later; the
        calibration itself prices every quote against its own forward.
        """
        out = np.empty(self.n_expiries, dtype=float)
        for i in range(self.n_expiries):
            fwds = self.forward[self.expiry_index == i]
            out[i] = np.median(fwds)
            spread = np.ptp(fwds)
            if spread > rtol * out[i]:
                raise ValueError(
                    f"inconsistent forwards at T={self.maturities[i]:.6f}: "
                    f"spread {spread:.6g} over median {out[i]:.6g} exceeds {rtol:%}"
                )
        return out


def observed_atm_total_variance(quotes, r=0.0):
    """
    theta_i^obs: ATM total variance per expiry, read off the quote nearest
    k = 0. Used only to seed the optimizer, so the crudeness is deliberate --
    the nearest listed strike is typically within a percent or two of the
    forward, and `least_squares` moves off the seed immediately.
    """
    if quotes.mid_iv is not None:
        vols = quotes.mid_iv
    else:
        import implied_vol as iv
        vols = iv.implied_vol_batch(
            quotes.price, quotes.forward, quotes.strike, quotes.T,
            np.full(len(quotes.price), r), np.full(len(quotes.price), r),
            quotes.option_type,
        )

    theta = np.empty(quotes.n_expiries, dtype=float)
    for i in range(quotes.n_expiries):
        mask = quotes.expiry_index == i
        k_i, v_i, T_i = quotes.k[mask], vols[mask], quotes.maturities[i]
        finite = np.isfinite(v_i) & (v_i > 0)
        if not np.any(finite):
            raise ValueError(f"no usable implied vol at T={T_i:.6f} to seed theta")
        atm = np.argmin(np.abs(k_i[finite]))
        theta[i] = v_i[finite][atm] ** 2 * T_i
    return theta


def initial_guess(quotes, eps=DEFAULT_EPS, r=0.0):
    """
    Section 4.2's starting point, in box coordinates:
    rho_i = 0, theta_1 = observed ATM total variance at the front expiry,
    a_i = theta_i^obs - theta_{i-1}^obs, c_i = 0.5.

    With every rho_i = 0 all p_i are 1, so the recovery collapses to
    theta_i = theta_{i-1} + a_i and this seed reproduces the observed ATM
    term structure exactly -- the optimizer starts on the right level and
    only has to find skew and curvature.

    The max(., eps) on a_i is load-bearing: observed ATM total variance is
    not always monotone across expiries (a stale or wide front-month quote
    is enough to invert it), and a negative a_i is outside the box, so
    `least_squares` would clip it to the bound and start from a seed that is
    not the one intended.
    """
    theta_obs = observed_atm_total_variance(quotes, r=r)
    n = quotes.n_expiries
    rho0 = np.zeros(n)
    a0 = np.maximum(np.diff(theta_obs), eps)
    c0 = np.full(n, 0.5)
    return pack(rho0, max(theta_obs[0], eps), a0, c0)


def quote_weights(quotes, scheme="uniform", r=0.0):
    """
    Section 4.1's omega(K,T). "uniform" weights every quote equally in price
    space; "vega" uses 1/vega^2, which to first order turns the price
    objective into a vol objective (a price error of vega*dsigma contributes
    dsigma^2), so the fit stops being dominated by ATM options simply because
    their prices are largest.
    """
    if scheme == "uniform":
        return np.ones(len(quotes.price))
    if scheme == "vega":
        if quotes.mid_iv is None:
            raise ValueError("vega weighting needs mid_iv on the quotes")
        import bs
        v = bs.vega(quotes.forward, quotes.strike, quotes.T,
                    np.full(len(quotes.price), r), np.full(len(quotes.price), r),
                    quotes.mid_iv)
        return 1.0 / np.maximum(v, 1e-8) ** 2
    raise ValueError(f"unknown weighting scheme {scheme!r}; use 'uniform' or 'vega'")


def calibrate(strike, T, price, option_type, forward, mid_iv=None, weights="uniform",
              butterfly_bound="GJ", r=0.0, eps=DEFAULT_EPS,
              rho_margin=DEFAULT_RHO_MARGIN, max_nfev=1000, ftol=1e-8, x0=None):
    """
    Fit a Global eSSVI surface to option prices (Section 4.1).

        min  sum_{K,T} omega(K,T) * (C_market(K,T) - C_model(K,T))^2

    over the 3N box coordinates, so every trial surface the optimizer visits
    is arbitrage-free and the problem carries no arbitrage penalty term and
    needs no post-hoc repair.

    Each quote is priced with its own `option_type`, so a chain of OTM puts
    below the forward and OTM calls above it is fitted as quoted. Doing so
    keeps the objective in the liquid instrument at every strike; converting
    everything to calls by put-call parity would instead compare deep-ITM
    model prices against parity-implied ones, where the option's value is
    almost all intrinsic and the vol signal is swamped.

    Prices follow the Black-76 convention `data.py` already uses for Deribit:
    S = forward and q = r, so `r` only sets an overall discount factor.

    Returns a `CalibrationResult`. `butterfly_bound="MM"` swaps in the
    necessary-and-sufficient bound; it admits more surfaces but runs a scalar
    minimization per slice per objective evaluation, so expect it to be
    noticeably slower than the default "GJ".
    """
    import bs

    quotes = _Quotes(strike, T, price, option_type, forward, mid_iv)
    n = quotes.n_expiries
    if len(quotes.price) < 3 * n:
        raise ValueError(
            f"{len(quotes.price)} quotes cannot determine 3N = {3 * n} parameters "
            f"across {n} expiries"
        )

    f_fn = resolve_bound(butterfly_bound)
    omega = quote_weights(quotes, weights, r=r) if isinstance(weights, str) else np.asarray(weights, float)
    sqrt_omega = np.sqrt(omega)
    r_vec = np.full(len(quotes.price), r)
    pillar_forwards = quotes.pillar_forwards()

    def surface_from(x):
        rho, theta_1, a, c = unpack(x, n)
        return GlobalESSVISurface(quotes.maturities, unbox(rho, theta_1, a, c, f_fn), pillar_forwards)

    def model_prices(x):
        surf = surface_from(x)
        w = np.empty(len(quotes.price))
        for i in range(n):
            mask = quotes.expiry_index == i
            w[mask] = surf.slices[i].total_variance(quotes.k[mask])
        sigma = np.sqrt(np.maximum(w, 1e-12) / quotes.T)
        return bs.price(quotes.forward, quotes.strike, quotes.T, r_vec, r_vec,
                        sigma, quotes.option_type)

    def residuals(x):
        return sqrt_omega * (model_prices(x) - quotes.price)

    lower, upper = param_bounds(n, eps=eps, rho_margin=rho_margin)
    if x0 is None:
        x0 = initial_guess(quotes, eps=eps, r=r)
    x0 = np.clip(x0, lower, upper)

    opt_result = least_squares(residuals, x0, bounds=(lower, upper),
                               max_nfev=max_nfev, ftol=ftol)

    return CalibrationResult(surface_from(opt_result.x), opt_result, quotes,
                             omega, model_prices(opt_result.x))


# ---------------------------------------------------------------------------
# Runtime validation (Section 4.4)
# ---------------------------------------------------------------------------
def check_no_arbitrage(surface, k_grid=None, t_grid=None, calendar_tol=None):
    """
    Verify a calibrated surface is free of butterfly and calendar arbitrage,
    raising `arbitrage.ArbitrageError` on the first violation.

    This is a check on *this implementation*, not a safety net the calibration
    depends on: Prop 3.1 already guarantees the result, so a failure here means
    a bug in `unbox`, `slice_at`, or the bounds -- not a surface that needs
    repairing. It is cheap enough to run after every calibration and is the
    only thing that would catch a silent regression in the parametrization.

    Butterfly is delegated to `arbitrage.check_butterfly`, which is written for
    raw SVI and knows nothing about eSSVI -- the point of routing through
    `essvi_slice_to_raw_svi` is that the check stays independent of the code it
    is checking.

    `t_grid` defaults to the calibrated expiries. Passing a denser grid also
    exercises Section 5's interpolation and extrapolation, which is worth doing
    since those rules are what the surface uses everywhere between pillars.

    A NOTE ON WHAT IS *NOT* CHECKED: eq. (3) requires psi to be strictly
    increasing, and after T_N `slice_at` holds psi fixed, so a validator
    written directly from eq. (3) would flag every t > T_N. It would be wrong
    to. With psi and rho held and only theta growing,

        dw/dtheta = 1/2 [1 + (rho*psi*k + theta) / sqrt((psi*k+theta*rho)^2
                                                        + theta^2(1-rho^2))]

    and the discriminant exceeds (rho*psi*k + theta)^2 by (psi*k)^2(1-rho^2)
    >= 0, so the ratio is bounded below by -1 and dw/dtheta > 0 everywhere:
    total variance is strictly increasing in t regardless. The strictness in
    eq. (3) is what the open-box parametrization needs to stay a bijection,
    not a necessary condition for absence of arbitrage. Hence this function
    tests w directly rather than the parameter inequalities.
    """
    k_grid = arb.DEFAULT_K_GRID if k_grid is None else np.asarray(k_grid, dtype=float)
    calendar_tol = arb.CALENDAR_TOL if calendar_tol is None else calendar_tol
    t_grid = surface.maturities if t_grid is None else np.asarray(t_grid, dtype=float)
    t_grid = np.sort(np.asarray(t_grid, dtype=float))

    for t in t_grid:
        arb.check_butterfly(essvi_slice_to_raw_svi(surface.slice_at(t)), k_grid=k_grid)

    w_prev, t_prev = None, None
    for t in t_grid:
        w = surface.slice_at(t).total_variance(k_grid)
        if w_prev is not None:
            gap = w - w_prev
            if np.any(gap < calendar_tol):
                bad = k_grid[np.argmin(gap)]
                raise arb.ArbitrageError(
                    f"calendar arbitrage between T={t_prev:.6f} and T={t:.6f}: "
                    f"total variance drops by {-gap.min():.3e} at k={bad:.4f}"
                )
        w_prev, t_prev = w, t
    return True


# ---------------------------------------------------------------------------
# Deribit quotes (Section 7 of the build spec)
# ---------------------------------------------------------------------------
def fetch_tick_sizes(currency="BTC", timeout=10):
    """
    Map instrument name -> tick size, from Deribit's live contract spec.

    Read rather than hardcoded: Deribit has changed option tick sizes, and
    applies `tick_size_steps` so that cheap far-OTM instruments quote on a
    finer grid than expensive ones. The base `tick_size` is the coarsest and
    therefore the conservative choice for the sub-tick filter below.
    """
    import requests
    resp = requests.get(
        f"{data.DERIBIT_BASE_URL}/public/get_instruments",
        params={"currency": currency, "kind": "option", "expired": "false"},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "result" not in payload:
        raise ValueError(f"unexpected Deribit response: {payload}")
    return {row["instrument_name"]: float(row["tick_size"]) for row in payload["result"]}


def filter_sub_tick(raw, tick_sizes, min_ticks=2.0):
    """
    Drop quotes whose bid is within `min_ticks` of the minimum price increment.

    A one-tick option is not a price so much as a placeholder: its relative
    error is 100%, its implied vol is whatever the grid happens to allow, and
    weighting by 1/width^2 gives exactly those quotes the largest weight in
    the fit. Applied to the raw book summary, where prices are still in
    contract units (the same units as `tick_size`) and the instrument name is
    still present -- `data.build_chain` scales prices by the forward and drops
    the name.
    """
    if raw is None or raw.empty:
        return raw
    tick = raw["instrument_name"].map(tick_sizes)
    keep = tick.notna() & (raw["bid_price"].astype(float) >= min_ticks * tick)
    return raw[keep].copy()


def fetch_quotes(currency="BTC", max_width_frac=0.15, min_ticks=2.0, r=0.0,
                 min_T=1.0 / 365.0, max_T=None, min_quotes_per_expiry=5,
                 itm="fold", now=None):
    """
    Pull a live Deribit chain and clean it into something `calibrate` can fit.

    Reuses `data.fetch_book_summary`/`data.build_chain` for the parsing,
    forward extraction, liquidity filter and implied vols, and adds the two
    things Section 7 needs on top:

      - the sub-tick filter (`filter_sub_tick`), applied to the raw payload
        before `build_chain` scales prices out of contract units;
      - ITM handling (parity folding by default) and expiry-level filtering,
        both in `filter_chain`.

    Returns the chain DataFrame; pass it through `chain_to_calibration_inputs`
    to get `calibrate`'s arguments.
    """
    raw = data.fetch_book_summary(currency)
    raw = filter_sub_tick(raw, fetch_tick_sizes(currency), min_ticks=min_ticks)
    chain = data.build_chain(raw, max_width_frac=max_width_frac, r=r, now=now)
    return filter_chain(chain, min_T=min_T, max_T=max_T,
                        min_quotes_per_expiry=min_quotes_per_expiry, itm=itm, r=r)


def fold_parity_to_otm(chain, r=0.0, require_otm_leg=True):
    """
    Fold each in-the-money quote onto its out-of-the-money twin by put-call
    parity, so both sides of a strike inform the fit instead of half the board
    being thrown away.

    With `data.py`'s Black-76 convention (S = forward, q = r) parity reads
    `C - P = D*(F - K)`, `D = exp(-r*T)` -- an exact identity involving no
    volatility, so restating an ITM quote on the OTM side adds no model error
    and the converted price simply inherits the ITM quote's own bid-ask as its
    uncertainty. Where a strike has both legs the two observations of the OTM
    price are combined by inverse-variance weighting on that width, so the
    typically-wider ITM leg contributes in proportion to how tightly it is
    actually quoted. Measured on a live BTC board this is a small but real
    improvement over discarding the ITM half outright.

    WHAT PARITY DOES NOT FIX, and the reason for `require_otm_leg`. Converting
    an ITM quote does not improve its conditioning. An option's price is
    intrinsic plus time value and only the time value carries volatility;
    the median ITM quote is ~95% intrinsic and its bid-ask is ~58% as wide as
    the whole time value (against ~5% for OTM). Parity preserves the *absolute*
    uncertainty while moving it onto a much smaller price, so the poor
    signal-to-noise carries over intact -- it is not repaired.

    That is harmless where an OTM leg exists to outvote it, and harmful where
    one does not. So by default a strike quoted only on its ITM side is
    dropped rather than resurrected: on a live board those are ~120 strikes
    whose vol content is too noisy to help, and admitting them costs an order
    of magnitude of fit quality (mean |dvol| 0.27% -> 3.2%), which no
    weighting scheme recovers because the noise is in the data and not in the
    objective. Pass `require_otm_leg=False` to keep them anyway.

    Note the vega red herring: at a given strike the call and the put have
    *identical* vega, forced by parity itself, since `C - P` contains no sigma.
    The ITM leg is not less sensitive to volatility -- it just buries that
    identical sensitivity inside a price roughly eight times larger.

    Returns one row per (expiry, strike), always the OTM type, with `bid`/`ask`
    spanning the combined uncertainty, `weight` the summed inverse variance,
    `n_legs` how many quotes were combined, and `mid_iv` re-inverted from the
    combined price.
    """
    if chain is None or chain.empty:
        return chain

    df = chain.copy()
    F = df["forward"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    T = df["T"].to_numpy(dtype=float)
    is_call = df["option_type"].to_numpy() == "C"
    otm_type = np.where(K >= F, "C", "P")

    df["otm_type"] = otm_type
    df["is_otm_leg"] = (otm_type == "C") == is_call
    # this quote's price restated on the OTM side of the strike
    parity = np.exp(-r * T) * (F - K)                       # C - P
    mid = df["mid"].to_numpy(dtype=float)
    df["otm_price"] = np.where(
        df["is_otm_leg"].to_numpy(), mid,
        np.where(otm_type == "C", mid + parity, mid - parity),
    )
    df["w"] = 1.0 / np.maximum((df["ask"] - df["bid"]).to_numpy(dtype=float), 1e-8) ** 2

    if require_otm_leg:
        df = df[df.groupby(["expiry", "strike"])["is_otm_leg"].transform("any")]
        if df.empty:
            return df

    combined = df.groupby(["expiry", "strike"], sort=False).apply(
        lambda g: pd.Series({
            "T": g["T"].iloc[0],
            "forward": g["forward"].iloc[0],
            "option_type": g["otm_type"].iloc[0],
            "mid": float(np.average(g["otm_price"], weights=g["w"])),
            "weight": float(g["w"].sum()),
            "n_legs": int(len(g)),
        }),
        include_groups=False,
    ).reset_index()

    half = 0.5 / np.sqrt(combined["weight"].to_numpy())
    combined["bid"] = combined["mid"] - half
    combined["ask"] = combined["mid"] + half

    import implied_vol as iv
    combined["mid_iv"] = iv.implied_vol_batch(
        combined["mid"].to_numpy(), combined["forward"].to_numpy(),
        combined["strike"].to_numpy(), combined["T"].to_numpy(),
        np.full(len(combined), r), np.full(len(combined), r),
        combined["option_type"].to_numpy(),
    )
    combined = combined[np.isfinite(combined["mid_iv"]) & (combined["mid_iv"] > 0)]
    return combined.sort_values(["T", "strike"]).reset_index(drop=True)


def filter_chain(chain, min_T=1.0 / 365.0, max_T=None, min_quotes_per_expiry=5,
                 itm="fold", r=0.0):
    """
    Restrict a chain to what is worth calibrating to.

    `itm` decides what happens to the in-the-money half of the board, which
    Deribit quotes at every strike alongside the OTM half:

    - `"fold"` (default) restates each ITM quote on the OTM side of its strike
      by put-call parity and inverse-variance-combines it with the OTM quote
      there, so both sides of every usable strike inform the fit. Strikes
      quoted only on their ITM side are still dropped -- see
      `fold_parity_to_otm` for why parity does not make those usable.
    - `"drop"` keeps only the OTM leg (calls at k >= 0, puts at k < 0) and
      ignores the ITM quotes entirely.
    - `"keep"` leaves the board untouched. Not recommended: an ITM price is
      ~95% intrinsic, so a price-space objective spends its effort on the
      deterministic part.

    Measured on one live BTC board (mean |dvol| against the quoted mids):
    `fold` 0.27%, `drop` 0.28%, `keep` 3.3%. `fold` edges out `drop` because it
    adds a second observation at each strike; both are an order of magnitude
    better than fitting the raw board.

    EXPIRIES: maturities under `min_T` (by default under a day, where a few
    hours of clock error moves total variance more than the quotes do), over
    `max_T`, or with fewer than `min_quotes_per_expiry` survivors -- an expiry
    that thin cannot pin down its own three parameters.
    """
    if itm not in ("fold", "drop", "keep"):
        raise ValueError(f"itm must be 'fold', 'drop' or 'keep'; got {itm!r}")
    if chain is None or chain.empty:
        return chain

    out = chain[chain["T"] >= min_T]
    if max_T is not None:
        out = out[out["T"] <= max_T]
    if out.empty:
        return out

    if itm == "fold":
        out = fold_parity_to_otm(out, r=r)
    elif itm == "drop":
        k = np.log(out["strike"].to_numpy(dtype=float) / out["forward"].to_numpy(dtype=float))
        is_call = out["option_type"].to_numpy() == "C"
        out = out[np.where(is_call, k >= 0.0, k < 0.0)]
    if out.empty:
        return out

    counts = out.groupby("T")["strike"].transform("size")
    return out[counts >= min_quotes_per_expiry].reset_index(drop=True)


def chain_to_calibration_inputs(chain, price_column="mid"):
    """
    Turn a `data.build_chain` frame into `calibrate(**kwargs)`.

    `mid` is the default target. The chain's prices are already scaled by the
    forward into absolute terms, matching the Black-76 convention `calibrate`
    prices under.
    """
    return {
        "strike": chain["strike"].to_numpy(dtype=float),
        "T": chain["T"].to_numpy(dtype=float),
        "price": chain[price_column].to_numpy(dtype=float),
        "option_type": chain["option_type"].to_numpy(dtype=object),
        "forward": chain["forward"].to_numpy(dtype=float),
        "mid_iv": chain["mid_iv"].to_numpy(dtype=float),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def surface_forward_at(surface, t):
    """
    F(t) for any surface in this package. `GlobalESSVISurface` carries its own
    log-linear `forward_at`; `VolSurface`/`SSVIVolSurface` expose only the
    per-knot `forwards` array, so the same log-linear rule is applied here.
    Interpolating log F (not F) means a piecewise-constant forward rate, the
    standard convention.
    """
    if hasattr(surface, "forward_at"):
        return float(surface.forward_at(t))
    T, F = np.asarray(surface.maturities, float), np.asarray(surface.forwards, float)
    if len(T) == 1:
        return float(F[0])
    return float(np.exp(np.interp(float(t), T, np.log(F))))


def surface_theta_at(surface, t, forward=None):
    """
    ATM total variance theta(t) = sigma_ATM(t)^2 * t, read straight off the
    surface's own `implied_vol`.

    Deliberately computed from the public query interface rather than from any
    model's internal parameters, so it is defined identically for raw-SVI,
    SSVI and eSSVI surfaces. That is what lets `vol_surface_grid` put all three
    on a standardised-moneyness axis and have `z` mean the same thing on each.
    """
    forward = surface_forward_at(surface, t) if forward is None else forward
    return float(surface.implied_vol(forward, t)) ** 2 * float(t)


def vol_surface_grid(surface, x_axis="k", k_range=None, t_range=None, n_k=80, n_t=80):
    """
    Evaluate `surface.implied_vol` on a grid, returning `(x, t, vol)` with
    `vol` shaped `(n_t, n_k)` — row per maturity, ready for any 3-D renderer.

    Separated from the plotting so the numbers can be tested without a
    rendering backend, and so callers who want a different chart get the grid
    without re-deriving the ranges.

    `x_axis` selects the horizontal coordinate:

    - `"k"`   log-moneyness `ln(K/F(T))`, the default.
    - `"K"`   absolute strike. Note that with a term structure of forwards the
              same strike sits at a different moneyness on every row, so the
              smile appears to drift sideways with maturity — that drift is the
              forward curve, not the surface.
    - `"z"`   standardised moneyness `k / sqrt(theta(T))`.

    ON THE DEFAULT `k` WINDOW. A single log-moneyness window applied to every
    maturity is not maturity-neutral, and on a real chain it is badly skewed:
    the quoted strikes of a BTC board span roughly `k in [-0.11, +0.05]` at
    eleven days but `[-0.79, +0.82]` at ten months, because a given strike is
    many more standard deviations away when there is less time to travel. A
    window wide enough to show the long end is therefore ±15 sigma or worse at
    the front, where the model reports vols no quote has ever constrained and
    the resulting spike dominates both the shape and the colour scale.

    So `k_range=None` derives the window from the surface rather than
    hardcoding it: ±`DEFAULT_PLOT_SIGMAS * sqrt(theta)` at the *longest*
    maturity, which covers the back end properly and adapts to the asset's vol
    level (a hardcoded ±1 is meaningless on a 12-vol index and cramped on
    crypto). The short-dated wings inside that window are still extrapolation.
    Pass `x_axis="z"` to remove the distortion entirely — there the quoted
    region is a comparable band at every maturity, which is what makes the
    surface readable across the term structure.
    """
    if x_axis not in ("k", "K", "z"):
        raise ValueError(f"x_axis must be 'k', 'K' or 'z'; got {x_axis!r}")
    if n_k < 2 or n_t < 2:
        raise ValueError(f"need at least a 2x2 grid, got n_k={n_k}, n_t={n_t}")

    t_lo, t_hi = t_range if t_range is not None else (surface.maturities[0], surface.maturities[-1])
    if not 0 < t_lo <= t_hi:
        raise ValueError(f"t_range must satisfy 0 < lo <= hi, got {(t_lo, t_hi)}")
    t = np.linspace(float(t_lo), float(t_hi), n_t)

    forwards = np.array([surface_forward_at(surface, ti) for ti in t])
    sqrt_theta = np.sqrt([surface_theta_at(surface, ti, forwards[i])
                          for i, ti in enumerate(t)])

    if x_axis == "z":
        x = np.linspace(-DEFAULT_PLOT_SIGMAS, DEFAULT_PLOT_SIGMAS, n_k) if k_range is None \
            else np.linspace(float(k_range[0]), float(k_range[1]), n_k)
        # k = z*sqrt(theta(T)) differs per row, so strikes are built row by row
        k_rows = x[None, :] * sqrt_theta[:, None]
    else:
        if k_range is None:
            half = DEFAULT_PLOT_SIGMAS * float(sqrt_theta[-1])
            k_lo, k_hi = -half, half
        else:
            k_lo, k_hi = float(k_range[0]), float(k_range[1])
        if k_lo >= k_hi:
            raise ValueError(f"k_range must satisfy lo < hi, got {(k_lo, k_hi)}")
        k_rows = np.repeat(np.linspace(k_lo, k_hi, n_k)[None, :], n_t, axis=0)
        x = k_rows[0]

    vol = np.array([surface.implied_vol(forwards[i] * np.exp(k_rows[i]), t[i])
                    for i in range(n_t)])

    if x_axis == "K":
        # One shared strike axis: span the union of every row's strikes, then
        # re-evaluate on it. Re-using the per-row strikes would give a ragged
        # grid that no surface renderer accepts.
        strikes = np.linspace((forwards[:, None] * np.exp(k_rows)).min(),
                              (forwards[:, None] * np.exp(k_rows)).max(), n_k)
        vol = np.array([surface.implied_vol(strikes, t[i]) for i in range(n_t)])
        x = strikes

    return x, t, vol


def _axis_label(x_axis):
    return {"k": "log-moneyness  k = ln(K/F)",
            "K": "strike K",
            "z": "standardised moneyness  z = k/√θ"}[x_axis]


def plot_vol_surface(surface, x_axis="k", k_range=None, t_range=None, n_k=80, n_t=80,
                     backend="auto", html_path=None, title=None, colorscale=None,
                     include_plotlyjs=True, show=False):
    """
    Render a calibrated surface as a 3-D plot: x = moneyness (or strike),
    y = maturity, z = Black-Scholes implied vol.

    Returns the figure object — a `plotly.graph_objects.Figure` or a
    `matplotlib.figure.Figure` depending on the backend actually used — so the
    caller can restyle it before saving.

    `backend`:
    - `"auto"` (default) uses plotly when it imports and silently falls back to
      matplotlib when it does not. This is the headless/CI path: nothing here
      needs a display or a browser.
    - `"plotly"` / `"matplotlib"` force one and raise if it is unavailable.

    `html_path` writes a standalone interactive page. It requires plotly, so
    asking for HTML while forcing the matplotlib backend is an error rather
    than a silent no-op. With `include_plotlyjs=True` (the default) the library
    is inlined and the file opens offline at the cost of a few MB; pass
    `"cdn"` for a small file that needs a network connection to render.

    The matplotlib path deliberately avoids `pyplot`: it builds a bare `Figure`,
    which never touches pyplot's global backend registry and so cannot fail or
    block on a machine with no display. Call `fig.savefig(...)` to write a PNG.

    See `vol_surface_grid` for the grid, the `x_axis` choices, and why the
    default `k` window is derived from theta rather than hardcoded.
    """
    if backend not in ("auto", "plotly", "matplotlib"):
        raise ValueError(f"backend must be 'auto', 'plotly' or 'matplotlib'; got {backend!r}")

    x, t, vol = vol_surface_grid(surface, x_axis=x_axis, k_range=k_range,
                                 t_range=t_range, n_k=n_k, n_t=n_t)
    title = title or f"Global eSSVI surface — {len(surface)} expiries"
    colorscale = colorscale or VOL_SURFACE_CMAP

    use_plotly = backend in ("auto", "plotly")
    if use_plotly:
        try:
            import plotly.graph_objects as go  # noqa: F401
        except ImportError:
            if backend == "plotly":
                raise
            use_plotly = False

    if not use_plotly and html_path is not None:
        raise ValueError("html_path requires plotly; the matplotlib backend cannot write "
                         "an interactive page")

    if use_plotly:
        fig = _plot_surface_plotly(x, t, vol, x_axis, title, colorscale)
        if html_path is not None:
            fig.write_html(str(html_path), include_plotlyjs=include_plotlyjs)
        if show:
            fig.show()
        return fig

    return _plot_surface_mpl(x, t, vol, x_axis, title, colorscale)


def _mpl_cmap(colorscale):
    """
    Resolve `colorscale` to a matplotlib colormap. A string is a registered
    colormap name (the default, "viridis"); a sequence of colours is built into
    a ramp so a custom palette still works.
    """
    if isinstance(colorscale, str):
        import matplotlib
        return matplotlib.colormaps[colorscale]
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("vol_surface", list(colorscale))


def _plotly_colorscale(colorscale):
    """
    Same for plotly. Plotly ships the common matplotlib ramps under capitalised
    names ("Viridis"), so a name is passed through with its first letter raised;
    a list of colours becomes an explicit [position, colour] scale.
    """
    if isinstance(colorscale, str):
        return colorscale[:1].upper() + colorscale[1:]
    colors = list(colorscale)
    return [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]


def _plot_surface_plotly(x, t, vol, x_axis, title, colors):
    import plotly.graph_objects as go

    fig = go.Figure(go.Surface(
        x=x, y=t, z=vol, colorscale=_plotly_colorscale(colors),
        colorbar=dict(title="implied vol", tickformat=".0%", len=0.7),
        # Contours projected onto the surface give it readable level lines;
        # height alone on a rotatable 3-D plot is hard to judge.
        contours={"z": {"show": True, "usecolormap": False, "color": "rgba(255,255,255,0.35)",
                        "width": 1, "start": float(np.nanmin(vol)),
                        "end": float(np.nanmax(vol)), "size": max(float(np.ptp(vol)) / 12, 1e-4)}},
        hovertemplate=(f"{_axis_label(x_axis)}: %{{x:.4f}}<br>"
                       "T: %{y:.4f}y<br>implied vol: %{z:.2%}<extra></extra>"),
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=_axis_label(x_axis),
            yaxis_title="T (years)",
            zaxis=dict(title="implied vol", tickformat=".0%"),
            camera=dict(eye=dict(**VOL_SURFACE_CAMERA_EYE)),
            aspectratio=dict(x=1.4, y=1.1, z=0.8),
        ),
        margin=dict(l=0, r=0, t=44, b=0), template="plotly_white",
    )
    return fig


def _plot_surface_mpl(x, t, vol, x_axis, title, colors):
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

    fig = Figure(figsize=(9, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    X, T = np.meshgrid(x, t)
    art = ax.plot_surface(X, T, vol, cmap=_mpl_cmap(colors),
                          linewidth=0, antialiased=True, rcount=len(t), ccount=len(x))
    ax.view_init(**VOL_SURFACE_VIEW)
    ax.set_xlabel(_axis_label(x_axis), labelpad=10)
    ax.set_ylabel("T (years)", labelpad=10)
    ax.set_zlabel("implied vol", labelpad=10)
    ax.zaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(title, pad=0)
    ax.set_box_aspect((1.5, 1.2, 0.9), zoom=1.1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0)
        axis._axinfo["grid"].update(color="#d8d7d3", linewidth=0.5)
    fig.colorbar(art, ax=ax, shrink=0.55, aspect=18, pad=0.02, label="implied vol",
                 format=lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    return fig

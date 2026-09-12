"""
Tests for the Global eSSVI parametrization.

The centrepiece is `test_prop_3_1_*`: Proposition 3.1 claims every point of
the open hyperrectangle maps to an arbitrage-free surface, so it can be
tested by sampling the box and checking the output against `arbitrage.py`'s
numerical machinery -- which knows nothing about eSSVI and was written for
raw SVI. Agreement between a closed-form claim and an independent numerical
check is the whole point; these tests need no market data.
"""
import builtins

import numpy as np
import pytest

import arbitrage as arb
import global_essvi as ge
from svi import raw_svi

K_GRID = np.linspace(-1.5, 1.5, 301)
BOUNDS = ["GJ", "MM"]


# ---------------------------------------------------------------------------
# Sampling the open box
# ---------------------------------------------------------------------------
def sample_box(rng, n, rho_margin=ge.DEFAULT_RHO_MARGIN):
    """
    One draw from eq. (4)'s open hyperrectangle.

    theta_1 and a_i are unbounded above, so "uniform" is undefined for them;
    they are drawn log-uniformly over ranges that bracket what real option
    chains produce (theta_1 in 1e-4..1 is roughly 1% to 100% annualized vol
    at the front expiry).
    """
    rho = rng.uniform(-1.0 + rho_margin, 1.0 - rho_margin, size=n)
    theta_1 = float(np.exp(rng.uniform(np.log(1e-4), np.log(1.0))))
    a = np.exp(rng.uniform(np.log(1e-5), np.log(0.5), size=n - 1))
    c = rng.uniform(1e-6, 1.0 - 1e-6, size=n)
    return rho, theta_1, a, c


def assert_arbitrage_free(slices, k_grid=K_GRID):
    """Every Prop 3.1 guarantee, checked one slice / one adjacent pair at a time."""
    thetas = np.array([s.theta for s in slices])
    psis = np.array([s.psi for s in slices])

    assert np.all(np.diff(thetas) > 0), f"theta not strictly increasing: {thetas}"
    assert np.all(np.diff(psis) > 0), f"psi not strictly increasing: {psis}"

    # Butterfly, per slice, via arbitrage.py's g(k) -- raises ArbitrageError.
    for i, sl in enumerate(slices):
        arb.check_butterfly(ge.essvi_slice_to_raw_svi(sl), k_grid=k_grid)

    # Calendar, adjacent pairs, on total variance directly.
    for i in range(len(slices) - 1):
        w_lo = slices[i].total_variance(k_grid)
        w_hi = slices[i + 1].total_variance(k_grid)
        assert np.all(w_hi - w_lo >= arb.CALENDAR_TOL), (
            f"calendar arbitrage between slice {i} and {i + 1}: "
            f"min gap {np.min(w_hi - w_lo):.3e}"
        )


# ---------------------------------------------------------------------------
# Proposition 3.1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("n", [2, 3])
def test_prop_3_1_small_n(bound, n):
    """Dense sampling at the maturity counts where the look-ahead in C_psi is shortest."""
    rng = np.random.default_rng(20220401 + n)
    f_fn = ge.resolve_bound(bound)
    for _ in range(250):
        rho, theta_1, a, c = sample_box(rng, n)
        assert_arbitrage_free(ge.unbox(rho, theta_1, a, c, f_fn))


@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("n", [10, 15])
def test_prop_3_1_large_n(bound, n):
    """
    Longer chains, where C_psi_i's look-ahead over all future slices is what
    keeps later intervals [A_j, C_j] non-empty. A forward-only recursion
    (ceiling = min(calendar, f_i) with no look-ahead) fails here.
    """
    rng = np.random.default_rng(31415 + n)
    f_fn = ge.resolve_bound(bound)
    for _ in range(60):
        rho, theta_1, a, c = sample_box(rng, n)
        assert_arbitrage_free(ge.unbox(rho, theta_1, a, c, f_fn))


@pytest.mark.parametrize("bound", BOUNDS)
def test_prop_3_1_extreme_rho_swings(bound):
    """
    rho alternating between the ends of its allowed range on every step --
    the case that drives p_i up (p ~ 39 at rho = -+0.95) and so stresses the
    cumulative products inside C_psi_i.
    """
    f_fn = ge.resolve_bound(bound)
    n = 8
    rho = np.array([0.94 if i % 2 else -0.94 for i in range(n)])
    for theta_1 in [1e-4, 1e-2, 0.5]:
        for c_val in [1e-6, 0.5, 1.0 - 1e-6]:
            slices = ge.unbox(rho, theta_1, np.full(n - 1, 1e-3), np.full(n, c_val), f_fn)
            assert_arbitrage_free(slices)


@pytest.mark.parametrize("bound", BOUNDS)
def test_prop_3_1_at_box_edges(bound):
    """
    c at both extremes and rho flat. c -> 0 puts psi on its calendar floor,
    c -> 1 on its ceiling; both are where the inequalities of eq. (3) go
    tight, so this is the boundary `param_bounds` insets away from.
    """
    f_fn = ge.resolve_bound(bound)
    n = 5
    for c_val in [1e-9, 1.0 - 1e-9]:
        for rho_val in [-0.94, 0.0, 0.94]:
            slices = ge.unbox(
                np.full(n, rho_val), 0.02, np.full(n - 1, 0.01), np.full(n, c_val), f_fn
            )
            assert_arbitrage_free(slices)


def test_unbox_is_deterministic_and_pure():
    rng = np.random.default_rng(7)
    rho, theta_1, a, c = sample_box(rng, 4)
    first = ge.unbox(rho, theta_1, a, c)
    rho_copy, a_copy, c_copy = rho.copy(), a.copy(), c.copy()
    second = ge.unbox(rho, theta_1, a, c)
    assert first == second
    np.testing.assert_array_equal(rho, rho_copy)
    np.testing.assert_array_equal(a, a_copy)
    np.testing.assert_array_equal(c, c_copy)


# ---------------------------------------------------------------------------
# The pieces unbox is built from
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("abs_rho", [0.0, 0.1, 0.4, 0.7, 0.9, 0.95, 0.999])
def test_l2_is_root_of_g2(abs_rho):
    """
    l_2(|rho|) = cot(arccos(-|rho|)/3) is exactly the root of g_2, which is
    what disambiguates the paper's tan(.)^-1 notation: the reciprocal reading
    puts the ray's start where g_2 changes sign, the arctangent reading does
    not. g_2 < 0 above it is what keeps `f_mm`'s denominator positive.
    """
    root = float(ge.l2(abs_rho))
    _, _, g2_at = ge._mm_terms(root, abs_rho)
    assert abs(float(g2_at)) < 1e-12

    _, _, g2_above = ge._mm_terms(root * 1.05, abs_rho)
    _, _, g2_below = ge._mm_terms(root * 0.95, abs_rho)
    assert float(g2_above) < 0 < float(g2_below)


@pytest.mark.parametrize("abs_rho", [0.0, 0.3, 0.7, 0.95])
@pytest.mark.parametrize("theta", [1e-3, 0.05, 1.0])
def test_f_mm_denominator_positive_on_whole_ray(theta, abs_rho):
    """No pole inside the search window: both terms of the denominator are positive."""
    lo = float(ge.l2(abs_rho))
    l = lo + np.geomspace(1e-9, ge.MM_SEARCH_SPAN, 500)
    g, _, g2 = ge._mm_terms(l, abs_rho)
    denom = theta * np.sqrt(1 - abs_rho ** 2) * g ** 2 - g2
    assert np.all(denom > 0)


@pytest.mark.parametrize("abs_rho", [0.0, 0.2, 0.5, 0.8, 0.95])
def test_f_mm_tail_limit(abs_rho):
    """
    The l -> inf limit of f_MM's integrand is 16/(1+|rho|)^2 = (4/(1+|rho|))^2,
    the square of the necessary condition. `f_mm` relies on this to close off
    the truncated tail.
    """
    l_far = float(ge.l2(abs_rho)) + 1e8
    root = np.sqrt(1 - abs_rho ** 2)
    for theta in [0.01, 1.0]:
        g, h, g2 = ge._mm_terms(l_far, abs_rho)
        val = 4 * theta * root * h ** 2 / (theta * root * g ** 2 - g2)
        assert float(val) == pytest.approx(16.0 / (1 + abs_rho) ** 2, rel=1e-4)


@pytest.mark.parametrize("abs_rho", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
@pytest.mark.parametrize("theta", [1e-4, 1e-2, 0.2, 1.0, 5.0, 20.0])
def test_mm_bound_never_stricter_than_gj(theta, abs_rho):
    """
    Section 2.3: "the MM conditions are less strict than the GJ conditions".
    The claim holds for the bound actually applied, `psi_bound` -- not for f
    itself, where f_MM dips below f_GJ once theta > 4/(1+|rho|) (both are then
    clipped to the same necessary condition and the difference is invisible).
    """
    assert ge.psi_bound(theta, abs_rho, ge.f_mm) >= ge.psi_bound(theta, abs_rho, ge.f_gj) - 1e-12


@pytest.mark.parametrize("abs_rho", [0.0, 0.3, 0.7, 0.95])
@pytest.mark.parametrize("theta", [1e-3, 0.05, 1.0, 10.0])
def test_f_mm_never_exceeds_its_own_tail_limit(theta, abs_rho):
    """An infimum cannot exceed a limit of its own integrand."""
    assert ge.f_mm(theta, abs_rho) <= 16.0 / (1 + abs_rho) ** 2 + 1e-12


@pytest.mark.parametrize("rho_prev,rho", [(-0.9, 0.9), (0.9, -0.9), (0.0, 0.5), (0.5, 0.0), (0.3, 0.3)])
def test_calendar_ratio_at_least_one(rho_prev, rho):
    """p_i >= 1 always, with equality iff rho is unchanged."""
    p = ge.calendar_ratio(rho_prev, rho)
    assert p >= 1.0
    assert (p == pytest.approx(1.0)) == (rho_prev == rho)


# ---------------------------------------------------------------------------
# The bridge to arbitrage.py
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta,rho,psi", [
    (0.04, -0.3, 0.5), (0.01, 0.0, 0.2), (0.25, 0.7, 1.1), (0.5, -0.94, 0.3),
])
def test_raw_svi_conversion_reproduces_essvi(theta, rho, psi):
    """
    `essvi_slice_to_raw_svi` routes through `ssvi.ssvi_slice_to_raw_svi` at
    phi = psi/theta rather than repeating the paper's p.4 algebra. If the two
    parametrizations describe the same curve, total variance must agree at
    every k -- otherwise every butterfly check in this module is testing the
    wrong slice.
    """
    sl = ge.ESSVISlice(theta=theta, rho=rho, psi=psi)
    p = ge.essvi_slice_to_raw_svi(sl)
    np.testing.assert_allclose(raw_svi(K_GRID, *p.as_array()), sl.total_variance(K_GRID), rtol=1e-12)


@pytest.mark.parametrize("theta,rho,psi", [(0.04, -0.3, 0.5), (0.25, 0.7, 1.1)])
def test_raw_svi_conversion_matches_papers_closed_form(theta, rho, psi):
    """The p.4 map, written out, as an independent check on the reuse above."""
    p = ge.essvi_slice_to_raw_svi(ge.ESSVISlice(theta=theta, rho=rho, psi=psi))
    root = np.sqrt(1 - rho ** 2)
    assert p.a == pytest.approx(theta * (1 - rho ** 2) / 2)
    assert p.b == pytest.approx(psi / 2)
    assert p.rho == pytest.approx(rho)
    assert p.m == pytest.approx(-theta * rho / psi)
    assert p.sigma == pytest.approx(theta * root / psi)


def test_essvi_total_variance_is_positive_everywhere():
    """
    Unlike raw SVI, eSSVI cannot imply negative total variance for theta > 0:
    the discriminant is a sum of squares. `arbitrage.check_minimum_variance`
    should therefore never be the check that fires.
    """
    wide_k = np.linspace(-20, 20, 2001)
    rng = np.random.default_rng(11)
    for _ in range(200):
        theta = float(np.exp(rng.uniform(np.log(1e-5), np.log(5))))
        rho = float(rng.uniform(-0.999, 0.999))
        psi = float(np.exp(rng.uniform(np.log(1e-4), np.log(3))))
        assert np.all(ge.essvi_total_variance(wide_k, theta, rho, psi) > 0)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_pack_unpack_round_trip(n):
    rng = np.random.default_rng(n)
    rho, theta_1, a, c = sample_box(rng, n) if n > 1 else (
        rng.uniform(-0.9, 0.9, 1), 0.04, np.array([]), rng.uniform(0.1, 0.9, 1)
    )
    rho2, theta_1_2, a2, c2 = ge.unpack(ge.pack(rho, theta_1, a, c), n)
    np.testing.assert_allclose(rho2, rho)
    np.testing.assert_allclose(a2, a)
    np.testing.assert_allclose(c2, c)
    assert theta_1_2 == pytest.approx(theta_1)


@pytest.mark.parametrize("n", [2, 6])
def test_param_bounds_shape_and_openness(n):
    lo, hi = ge.param_bounds(n)
    assert len(lo) == len(hi) == 3 * n
    assert np.all(lo < hi)
    # every lower bound strictly inside the open box, never on its edge
    assert np.all(lo[:n] > -1.0) and np.all(hi[:n] < 1.0)      # rho
    assert lo[n] > 0.0                                          # theta_1
    assert np.all(lo[n + 1:2 * n] > 0.0)                        # a
    assert np.all(lo[2 * n:] > 0.0) and np.all(hi[2 * n:] < 1.0)  # c


def test_unbox_rejects_malformed_input():
    with pytest.raises(ValueError, match="a_2..a_N"):
        ge.unbox([0.0, 0.1], 0.04, [0.01, 0.02], [0.5, 0.5])
    with pytest.raises(ValueError, match="c must have"):
        ge.unbox([0.0, 0.1], 0.04, [0.01], [0.5])
    with pytest.raises(ValueError, match="strictly inside"):
        ge.unbox([0.0, 1.0], 0.04, [0.01], [0.5, 0.5])


def test_resolve_bound():
    assert ge.resolve_bound("GJ") is ge.f_gj
    assert ge.resolve_bound("MM") is ge.f_mm
    assert ge.resolve_bound(ge.f_gj) is ge.f_gj
    with pytest.raises(ValueError, match="butterfly_bound"):
        ge.resolve_bound("gj")


# ---------------------------------------------------------------------------
# The surface: interpolation and extrapolation (Section 5)
# ---------------------------------------------------------------------------
MATURITIES = np.array([0.02, 0.08, 0.25, 0.5, 1.0])


def build_surface(rng, n=len(MATURITIES), bound="GJ", maturities=MATURITIES):
    rho, theta_1, a, c = sample_box(rng, n)
    slices = ge.unbox(rho, theta_1, a, c, ge.resolve_bound(bound))
    forwards = 60000.0 * np.exp(0.05 * maturities[:n])
    return ge.GlobalESSVISurface(maturities[:n], slices, forwards)


@pytest.mark.parametrize("bound", BOUNDS)
def test_surface_is_arbitrage_free_off_the_pillars(bound):
    """
    Section 5's rules are claimed to preserve absence of arbitrage, so the
    continuum between and beyond the calibrated expiries must be as clean as
    the pillars themselves. Sampled densely across all three regimes:
    before T_1, interpolated, and after T_N.
    """
    rng = np.random.default_rng(2024)
    grid = np.concatenate([
        np.linspace(0.001, MATURITIES[0], 12),      # extrapolated back
        np.linspace(MATURITIES[0], MATURITIES[-1], 40),  # interpolated
        np.linspace(MATURITIES[-1], 4.0, 12),       # extrapolated forward
    ])
    for _ in range(25):
        surf = build_surface(rng, bound=bound)
        for t in grid:
            arb.check_butterfly(ge.essvi_slice_to_raw_svi(surf.slice_at(t)), k_grid=K_GRID)
        w = np.array([surf.slice_at(t).total_variance(K_GRID) for t in grid])
        gaps = np.diff(w, axis=0)
        assert np.all(gaps >= arb.CALENDAR_TOL), (
            f"calendar arbitrage off-pillar: min gap {gaps.min():.3e}"
        )


def test_slice_at_reproduces_pillars_exactly():
    rng = np.random.default_rng(5)
    surf = build_surface(rng)
    for T, sl in zip(surf.maturities, surf.slices):
        got = surf.slice_at(T)
        assert got.theta == pytest.approx(sl.theta)
        assert got.rho == pytest.approx(sl.rho)
        assert got.psi == pytest.approx(sl.psi)


def test_interpolation_is_linear_in_rho_psi_not_rho():
    """
    Section 5.1 interpolates the product rho*psi. Unless rho happens to be
    equal at the two pillars, the recovered rho is therefore *not* the linear
    interpolant of rho -- which is the distinction the CRITICAL note in the
    build spec is about, and the one that keeps the calendar condition intact.
    """
    slices = [ge.ESSVISlice(theta=0.02, rho=-0.6, psi=0.4),
              ge.ESSVISlice(theta=0.09, rho=0.1, psi=0.9)]
    surf = ge.GlobalESSVISurface([0.1, 0.5], slices, [100.0, 100.0])
    mid = surf.slice_at(0.3)

    assert mid.theta == pytest.approx(0.055)
    assert mid.psi == pytest.approx(0.65)
    assert mid.rho * mid.psi == pytest.approx(0.5 * (-0.6 * 0.4) + 0.5 * (0.1 * 0.9))
    assert mid.rho != pytest.approx(0.5 * (-0.6 + 0.1))


def test_extrapolation_before_first_maturity_scales_total_variance():
    """
    With theta and psi both scaled by lam and rho held, every term of eq. (1)
    carries a factor lam, so w(k, t) = lam * w(k, T_1) at every k. That exact
    proportionality is the content of Section 5.2.1; scaling only theta (and
    holding psi) would destroy it.
    """
    sl = ge.ESSVISlice(theta=0.04, rho=-0.4, psi=0.7)
    surf = ge.GlobalESSVISurface([0.25], [sl], [100.0])
    for lam in [0.05, 0.3, 0.8]:
        got = surf.slice_at(lam * 0.25).total_variance(K_GRID)
        np.testing.assert_allclose(got, lam * sl.total_variance(K_GRID), rtol=1e-12)


def test_extrapolation_after_last_maturity_holds_psi_and_rho():
    sl = ge.ESSVISlice(theta=0.04, rho=-0.4, psi=0.7)
    surf = ge.GlobalESSVISurface([0.25], [sl], [100.0])
    far = surf.slice_at(1.0)
    assert far.theta == pytest.approx(4.0 * 0.04)
    assert far.psi == pytest.approx(0.7)
    assert far.rho == pytest.approx(-0.4)


def test_forward_interpolation_is_log_linear():
    surf = ge.GlobalESSVISurface(
        [0.5, 1.0], [ge.ESSVISlice(0.04, -0.2, 0.5), ge.ESSVISlice(0.09, -0.1, 0.7)],
        [100.0, 110.0],
    )
    assert surf.forward_at(0.75) == pytest.approx(np.sqrt(100.0 * 110.0))
    assert surf.forward_at(0.5) == pytest.approx(100.0)
    assert surf.forward_at(1.0) == pytest.approx(110.0)
    # constant rate continued beyond the pillars
    assert surf.forward_at(1.5) == pytest.approx(110.0 * (110.0 / 100.0))


def test_implied_vol_round_trips_through_total_variance():
    rng = np.random.default_rng(9)
    surf = build_surface(rng)
    T = 0.3
    F = surf.forward_at(T)
    strikes = F * np.exp(np.linspace(-0.8, 0.8, 25))
    vols = surf.implied_vol(strikes, T)
    w = surf.total_variance(np.log(strikes / F), T)
    np.testing.assert_allclose(vols ** 2 * T, w, rtol=1e-12)
    assert np.all(vols > 0)
    assert isinstance(surf.implied_vol(float(F), T), float)


def test_surface_rejects_malformed_input():
    sl = ge.ESSVISlice(0.04, -0.2, 0.5)
    with pytest.raises(ValueError, match="equal length"):
        ge.GlobalESSVISurface([0.5, 1.0], [sl], [100.0, 110.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        ge.GlobalESSVISurface([1.0, 0.5], [sl, sl], [100.0, 110.0])
    with pytest.raises(ValueError, match="at least one"):
        ge.GlobalESSVISurface([], [], [])
    surf = ge.GlobalESSVISurface([0.5], [sl], [100.0])
    with pytest.raises(ValueError, match="must be positive"):
        surf.slice_at(0.0)
    with pytest.raises(ValueError, match="strikes must be positive"):
        surf.implied_vol(-1.0, 0.5)


# ---------------------------------------------------------------------------
# Calibration (Section 4)
# ---------------------------------------------------------------------------
import bs  # noqa: E402  (kept next to the tests that use it)

TRUE_RHO = np.array([-0.70, -0.60, -0.50, -0.42, -0.35])
TRUE_A = np.array([0.004, 0.012, 0.020, 0.030])
TRUE_THETA_1 = 0.0016


def synthetic_surface():
    """A plausible BTC-shaped surface: steep short-dated skew flattening out with T."""
    slices = ge.unbox(TRUE_RHO, TRUE_THETA_1, TRUE_A, np.full(len(TRUE_RHO), 0.45))
    forwards = 60000.0 * np.exp(0.05 * MATURITIES)
    return ge.GlobalESSVISurface(MATURITIES, slices, forwards)


def synthetic_quotes(surf, n_strikes=21, k_max=0.6, noise=0.0, rng=None):
    """
    An OTM chain priced off `surf`: puts below the forward, calls above,
    which is how the calibration is meant to be fed.
    """
    strike, T, forward, option_type = [], [], [], []
    for i, t in enumerate(surf.maturities):
        F = surf.forwards[i]
        for k in np.linspace(-k_max, k_max, n_strikes):
            strike.append(F * np.exp(k))
            T.append(t)
            forward.append(F)
            option_type.append("C" if k >= 0 else "P")

    strike = np.array(strike); T = np.array(T); forward = np.array(forward)
    option_type = np.array(option_type, dtype=object)
    vol = np.array([surf.implied_vol(k_, t_) for k_, t_ in zip(strike, T)])
    price = bs.price(forward, strike, T, 0.0, 0.0, vol, option_type)

    if noise:
        price = price * (1.0 + noise * rng.standard_normal(len(price)))
        price = np.maximum(price, 1e-8)
        import implied_vol as iv
        vol = iv.implied_vol_batch(price, forward, strike, T,
                                   np.zeros(len(price)), np.zeros(len(price)), option_type)
    return dict(strike=strike, T=T, price=price, option_type=option_type,
                forward=forward, mid_iv=vol)


def test_calibration_recovers_a_known_surface_exactly():
    """
    Noiseless prices generated from a surface inside the box must be fitted
    back to that surface. This is the end-to-end inverse of `unbox`: if the
    recovery formulas, the pricing convention and the packing order did not
    all agree, the fit could still converge but not to these parameters.
    """
    truth = synthetic_surface()
    result = ge.calibrate(**synthetic_quotes(truth))

    assert result.success
    assert result.weighted_rmse < 1e-6
    for fitted, expected in zip(result.surface.slices, truth.slices):
        assert fitted.theta == pytest.approx(expected.theta, rel=1e-6)
        assert fitted.rho == pytest.approx(expected.rho, rel=1e-5, abs=1e-7)
        assert fitted.psi == pytest.approx(expected.psi, rel=1e-6)


@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("noise", [0.02, 0.10])
def test_calibration_on_noisy_prices_is_still_arbitrage_free(bound, noise):
    """
    The claim the whole parametrization exists to make: the fit cannot produce
    arbitrage no matter what it is fitted to. Prices are perturbed hard enough
    (up to 10% relative, i.i.d. per quote, so the smile is genuinely ragged and
    not a shifted version of anything admissible) that an unconstrained SVI fit
    would routinely cross slices or go butterfly-negative.
    """
    rng = np.random.default_rng(hash((bound, noise)) % 2 ** 32)
    quotes = synthetic_quotes(synthetic_surface(), noise=noise, rng=rng)
    result = ge.calibrate(**quotes, butterfly_bound=bound)

    dense_t = np.concatenate([
        np.linspace(0.002, MATURITIES[0], 5),
        np.linspace(MATURITIES[0], MATURITIES[-1], 25),
        np.linspace(MATURITIES[-1], 3.0, 5),
    ])
    assert ge.check_no_arbitrage(result.surface, k_grid=K_GRID, t_grid=dense_t) is True


def test_calibration_fits_noisy_prices_at_least_as_well_as_the_truth():
    """
    The fit is measured against the noise floor rather than an absolute
    tolerance: prices in an OTM chain span orders of magnitude, so any fixed
    RMSE threshold is really a statement about the largest quotes. The
    generating surface's own RMSE against the perturbed prices is what a
    perfect fit would score, and least squares should match or beat it.
    """
    rng = np.random.default_rng(4242)
    truth = synthetic_surface()
    quotes = synthetic_quotes(truth, noise=0.02, rng=rng)
    result = ge.calibrate(**quotes)

    truth_vols = np.array([truth.implied_vol(k_, t_)
                           for k_, t_ in zip(quotes["strike"], quotes["T"])])
    truth_prices = bs.price(quotes["forward"], quotes["strike"], quotes["T"],
                            0.0, 0.0, truth_vols, quotes["option_type"])
    noise_floor = float(np.sqrt(np.mean((truth_prices - quotes["price"]) ** 2)))

    assert result.success
    assert result.weighted_rmse <= noise_floor


def test_vega_weighting_reweights_the_objective():
    """
    1/vega^2 weights turn a price objective into a vol one to first order.
    The two schemes should therefore disagree -- if they produced the same
    surface, the weights would not be reaching the residuals.
    """
    quotes = synthetic_quotes(synthetic_surface(), noise=0.03,
                              rng=np.random.default_rng(77))
    uniform = ge.calibrate(**quotes, weights="uniform")
    vega = ge.calibrate(**quotes, weights="vega")
    assert ge.check_no_arbitrage(vega.surface) is True
    assert any(abs(u.psi - v.psi) > 1e-6
               for u, v in zip(uniform.surface.slices, vega.surface.slices))


def test_initial_guess_reproduces_observed_atm_term_structure():
    """
    With every rho_i = 0 the recovery collapses to theta_i = theta_{i-1} + a_i,
    so the seed's thetas must equal the observed ATM total variances exactly.
    """
    truth = synthetic_surface()
    quotes = synthetic_quotes(truth)
    q = ge._Quotes(quotes["strike"], quotes["T"], quotes["price"],
                   quotes["option_type"], quotes["forward"], quotes["mid_iv"])
    theta_obs = ge.observed_atm_total_variance(q)
    rho0, theta_1, a0, c0 = ge.unpack(ge.initial_guess(q), q.n_expiries)

    np.testing.assert_allclose(rho0, 0.0)
    np.testing.assert_allclose(c0, 0.5)
    seeded = ge.unbox(rho0, theta_1, a0, c0)
    np.testing.assert_allclose([s.theta for s in seeded], theta_obs, rtol=1e-12)


def test_initial_guess_floors_non_monotone_atm_variance():
    """
    Observed ATM total variance need not be monotone -- one stale front quote
    inverts it -- and a negative a_i falls outside the box. The floor keeps the
    seed admissible; without it `least_squares` silently clips and starts
    somewhere else.
    """
    truth = synthetic_surface()
    quotes = synthetic_quotes(truth)
    # make the second expiry look cheaper than the first
    inverted = quotes["mid_iv"].copy()
    inverted[quotes["T"] == MATURITIES[1]] *= 0.2
    q = ge._Quotes(quotes["strike"], quotes["T"], quotes["price"],
                   quotes["option_type"], quotes["forward"], inverted)

    lower, upper = ge.param_bounds(q.n_expiries)
    x0 = ge.initial_guess(q)
    assert np.all(x0 >= lower) and np.all(x0 <= upper)
    assert ge.check_no_arbitrage(
        ge.GlobalESSVISurface(q.maturities, ge.unbox(*ge.unpack(x0, q.n_expiries)),
                              q.pillar_forwards())
    ) is True


def test_calibrate_rejects_underdetermined_problems():
    truth = synthetic_surface()
    quotes = synthetic_quotes(truth, n_strikes=2)
    with pytest.raises(ValueError, match="cannot determine"):
        ge.calibrate(**quotes)


def test_quotes_reject_inconsistent_forwards():
    quotes = synthetic_quotes(synthetic_surface())
    forwards = quotes["forward"].copy()
    forwards[0] *= 1.01
    q = ge._Quotes(quotes["strike"], quotes["T"], quotes["price"],
                   quotes["option_type"], forwards, quotes["mid_iv"])
    with pytest.raises(ValueError, match="inconsistent forwards"):
        q.pillar_forwards()


def test_check_no_arbitrage_catches_a_planted_violation():
    """
    The validator has to be capable of failing -- otherwise the tests above
    that assert it passes prove nothing.
    """
    # Calendar only: the later slice is the earlier one with total variance
    # halved (theta and psi both scaled), which leaves it perfectly valid on
    # its own -- so the butterfly pass must let it through and the calendar
    # pass must catch it.
    early = ge.ESSVISlice(theta=0.04, rho=-0.3, psi=0.30)
    late = ge.ESSVISlice(theta=0.02, rho=-0.3, psi=0.15)
    for sl in (early, late):
        arb.check_butterfly(ge.essvi_slice_to_raw_svi(sl), k_grid=K_GRID)
    surf = ge.GlobalESSVISurface([0.25, 0.5], [early, late], [100.0, 100.0])
    with pytest.raises(arb.ArbitrageError, match="calendar arbitrage"):
        ge.check_no_arbitrage(surf)

    butterfly_broken = ge.GlobalESSVISurface(
        [0.5], [ge.ESSVISlice(theta=0.04, rho=0.5, psi=3.0)], [100.0])
    with pytest.raises(arb.ArbitrageError):
        ge.check_no_arbitrage(butterfly_broken)


# ---------------------------------------------------------------------------
# Deribit plumbing
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402


def test_filter_sub_tick_drops_placeholder_quotes():
    raw = pd.DataFrame({
        "instrument_name": ["BTC-1JAN27-50000-C", "BTC-1JAN27-90000-C", "BTC-1JAN27-99000-C"],
        "bid_price": [0.05, 0.0005, 0.0002],
    })
    kept = ge.filter_sub_tick(raw, {n: 0.0005 for n in raw["instrument_name"]}, min_ticks=2.0)
    assert kept["instrument_name"].tolist() == ["BTC-1JAN27-50000-C"]


def test_filter_sub_tick_drops_instruments_missing_from_the_spec():
    raw = pd.DataFrame({"instrument_name": ["BTC-1JAN27-50000-C"], "bid_price": [0.05]})
    assert ge.filter_sub_tick(raw, {}).empty


def test_filter_chain_drops_short_and_thin_expiries():
    chain = pd.DataFrame({
        "T": [0.0005] * 8 + [0.1] * 8 + [0.4] * 2,
        "strike": list(range(18)),
        "forward": [10.0] * 18,
        "option_type": ["C"] * 18,
    })
    out = ge.filter_chain(chain, min_T=1 / 365, min_quotes_per_expiry=5, itm='keep')
    assert sorted(out["T"].unique()) == [0.1]


def test_filter_chain_keeps_only_otm():
    """
    itm='drop': calls above the forward, puts below. Deribit lists both types at
    every strike, so this halves the chain and removes exactly the ITM half.
    (The default itm='fold' keeps the same strikes but folds the ITM leg in by
    parity, so it is exercised separately.)
    """
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    chain = pd.DataFrame({
        "T": [0.5] * 10,
        "strike": np.concatenate([strikes, strikes]),
        "forward": [100.0] * 10,
        "option_type": ["C"] * 5 + ["P"] * 5,
    })
    out = ge.filter_chain(chain, min_quotes_per_expiry=1, itm="drop")
    kept = set(zip(out["strike"], out["option_type"]))
    assert kept == {(100.0, "C"), (110.0, "C"), (120.0, "C"), (80.0, "P"), (90.0, "P")}


def test_filter_chain_rejects_unknown_itm_mode():
    with pytest.raises(ValueError, match="itm must be"):
        ge.filter_chain(pd.DataFrame({"T": [0.5]}), itm="discard")


def test_parity_fold_combines_both_legs_exactly():
    """
    Parity is an identity, not a model: C - P = D*(F - K) contains no sigma, so
    restating an ITM quote on the OTM side must reproduce the OTM price exactly
    when the two quotes are mutually consistent.
    """
    F, K, T = 100.0, 120.0, 0.5           # K > F, so the call is the OTM leg
    call_mid, put_mid = 3.0, 3.0 + (K - F)  # consistent by parity at r = 0
    chain = pd.DataFrame({
        "expiry": ["X", "X"], "strike": [K, K], "T": [T, T], "forward": [F, F],
        "option_type": ["C", "P"], "mid": [call_mid, put_mid],
        "bid": [call_mid - 0.5, put_mid - 0.5], "ask": [call_mid + 0.5, put_mid + 0.5],
    })
    folded = ge.fold_parity_to_otm(chain)

    assert len(folded) == 1
    row = folded.iloc[0]
    assert row["option_type"] == "C"          # folded onto the OTM side
    assert row["n_legs"] == 2                 # both quotes used
    assert row["mid"] == pytest.approx(call_mid)


def test_parity_fold_weights_the_wider_leg_less():
    """Inverse-variance weighting: a leg quoted twice as wide gets a quarter the say."""
    F, K, T = 100.0, 120.0, 0.5
    chain = pd.DataFrame({
        "expiry": ["X", "X"], "strike": [K, K], "T": [T, T], "forward": [F, F],
        "option_type": ["C", "P"], "mid": [3.0, 5.0 + (K - F)],   # legs disagree by 2.0
        "bid": [3.0 - 0.5, 5.0 + (K - F) - 1.0],                  # OTM leg half as wide
        "ask": [3.0 + 0.5, 5.0 + (K - F) + 1.0],
    })
    row = ge.fold_parity_to_otm(chain).iloc[0]
    # weights 1/1^2 and 1/2^2 -> 4:1 toward the tighter OTM leg
    assert row["mid"] == pytest.approx((4 * 3.0 + 1 * 5.0) / 5)


def test_parity_fold_drops_strikes_quoted_only_on_the_itm_side():
    """
    Parity preserves a quote's absolute uncertainty while moving it onto a much
    smaller price, so it does not repair an ITM quote's conditioning. With no
    OTM leg to outvote it such a strike is dropped by default, and kept only on
    request.
    """
    F, K, T = 100.0, 120.0, 0.5
    itm_only = pd.DataFrame({
        "expiry": ["X"], "strike": [K], "T": [T], "forward": [F],
        "option_type": ["P"], "mid": [3.0 + (K - F)],
        "bid": [3.0 + (K - F) - 0.5], "ask": [3.0 + (K - F) + 0.5],
    })
    assert ge.fold_parity_to_otm(itm_only).empty
    kept = ge.fold_parity_to_otm(itm_only, require_otm_leg=False)
    assert len(kept) == 1 and kept.iloc[0]["option_type"] == "C"


def test_otm_filter_removes_deep_itm_quotes_that_wreck_the_fit():
    """
    Regression test for the failure this filter exists to prevent. A deep ITM
    put is almost all intrinsic: its vega is negligible, so the price carries
    no usable vol information, yet in a price-space objective it is worth
    multiples of an ATM option and dominates the sum of squares. Fitting the
    OTM chain plus its ITM twins must not degrade the fit -- and without the
    filter it degrades it by an order of magnitude on live data.
    """
    truth = synthetic_surface()
    otm = synthetic_quotes(truth)

    # the same strikes quoted as their ITM counterparts, priced consistently
    itm_type = np.where(otm["option_type"] == "C", "P", "C").astype(object)
    itm_price = bs.price(otm["forward"], otm["strike"], otm["T"], 0.0, 0.0,
                         otm["mid_iv"], itm_type)
    both = pd.DataFrame({
        "strike": np.concatenate([otm["strike"], otm["strike"]]),
        "T": np.concatenate([otm["T"], otm["T"]]),
        "forward": np.concatenate([otm["forward"], otm["forward"]]),
        "option_type": np.concatenate([otm["option_type"], itm_type]),
        "mid": np.concatenate([otm["price"], itm_price]),
        "mid_iv": np.concatenate([otm["mid_iv"], otm["mid_iv"]]),
    })

    # a tight, symmetric quoted width so parity folding has something to combine
    both["bid"] = both["mid"] * 0.999
    both["ask"] = both["mid"] * 1.001
    both["expiry"] = both["T"]

    for mode in ("drop", "fold"):
        filtered = ge.filter_chain(both, min_quotes_per_expiry=1, itm=mode)
        assert len(filtered) == len(otm["strike"]), mode

    filtered = ge.filter_chain(both, min_quotes_per_expiry=1, itm="drop")
    result = ge.calibrate(**ge.chain_to_calibration_inputs(filtered))
    assert result.success
    for fitted, expected in zip(result.surface.slices, truth.slices):
        assert fitted.psi == pytest.approx(expected.psi, rel=1e-5)


def test_chain_to_calibration_inputs_round_trips_into_calibrate():
    quotes = synthetic_quotes(synthetic_surface())
    chain = pd.DataFrame({
        "strike": quotes["strike"], "T": quotes["T"], "mid": quotes["price"],
        "option_type": quotes["option_type"], "forward": quotes["forward"],
        "mid_iv": quotes["mid_iv"],
    })
    result = ge.calibrate(**ge.chain_to_calibration_inputs(chain))
    assert result.success
    assert ge.check_no_arbitrage(result.surface) is True


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x_axis", ["k", "K", "z"])
def test_vol_surface_grid_shape_and_values(x_axis):
    surf = synthetic_surface()
    x, t, vol = ge.vol_surface_grid(surf, x_axis=x_axis, n_k=25, n_t=11)

    assert x.shape == (25,) and t.shape == (11,) and vol.shape == (11, 25)
    assert np.all(np.diff(x) > 0) and np.all(np.diff(t) > 0)
    assert np.all(np.isfinite(vol)) and np.all(vol > 0)
    assert t[0] == pytest.approx(surf.maturities[0])
    assert t[-1] == pytest.approx(surf.maturities[-1])


def test_vol_surface_grid_matches_implied_vol_pointwise():
    """The grid must be exactly what implied_vol returns, not a resampling of it."""
    surf = synthetic_surface()
    x, t, vol = ge.vol_surface_grid(surf, x_axis="k", n_k=9, n_t=7)
    for i, ti in enumerate(t):
        expected = surf.implied_vol(surf.forward_at(ti) * np.exp(x), ti)
        np.testing.assert_allclose(vol[i], expected, rtol=1e-12)


def test_vol_surface_grid_default_k_window_scales_with_vol_level():
    """
    The default window is +-2.5 sd of the longest maturity, not a hardcoded
    constant -- so a high-vol surface gets a proportionally wider one. A fixed
    window would be meaningless on a low-vol underlying and cramped on crypto.
    """
    quiet = ge.GlobalESSVISurface([0.5], [ge.ESSVISlice(0.01, -0.2, 0.1)], [100.0])
    wild = ge.GlobalESSVISurface([0.5], [ge.ESSVISlice(0.25, -0.2, 0.1)], [100.0])
    x_quiet, _, _ = ge.vol_surface_grid(quiet, n_k=5, n_t=3)
    x_wild, _, _ = ge.vol_surface_grid(wild, n_k=5, n_t=3)

    assert x_wild.max() > x_quiet.max()
    assert x_quiet.max() == pytest.approx(ge.DEFAULT_PLOT_SIGMAS * np.sqrt(0.01))
    assert x_wild.max() == pytest.approx(ge.DEFAULT_PLOT_SIGMAS * np.sqrt(0.25))


def test_vol_surface_grid_z_axis_is_maturity_neutral():
    """
    In z the window is a fixed number of standard deviations at *every*
    maturity, which is the whole reason to offer it: the same z maps to a much
    narrower strike range at the front than at the back.
    """
    surf = synthetic_surface()
    z, t, _ = ge.vol_surface_grid(surf, x_axis="z", n_k=21, n_t=5)
    assert z[0] == pytest.approx(-ge.DEFAULT_PLOT_SIGMAS)
    assert z[-1] == pytest.approx(ge.DEFAULT_PLOT_SIGMAS)

    front, back = surf.slice_at(t[0]), surf.slice_at(t[-1])
    assert z[-1] * np.sqrt(front.theta) < z[-1] * np.sqrt(back.theta)


def test_vol_surface_grid_rejects_bad_arguments():
    surf = synthetic_surface()
    with pytest.raises(ValueError, match="x_axis"):
        ge.vol_surface_grid(surf, x_axis="delta")
    with pytest.raises(ValueError, match="2x2"):
        ge.vol_surface_grid(surf, n_k=1)
    with pytest.raises(ValueError, match="k_range"):
        ge.vol_surface_grid(surf, k_range=(0.5, -0.5))
    with pytest.raises(ValueError, match="t_range"):
        ge.vol_surface_grid(surf, t_range=(0.0, 1.0))


def test_plot_vol_surface_plotly_and_html(tmp_path):
    surf = synthetic_surface()
    out = tmp_path / "surface.html"
    fig = ge.plot_vol_surface(surf, backend="plotly", html_path=out, n_k=20, n_t=12)

    assert type(fig).__module__.startswith("plotly")
    assert fig.data[0].type == "surface"
    assert np.asarray(fig.data[0].z).shape == (12, 20)
    assert out.exists() and out.stat().st_size > 1000
    assert "<div" in out.read_text()[:400000]


def test_plot_vol_surface_matplotlib_is_headless_and_saveable(tmp_path):
    """
    The fallback must not need a display. Building a bare `Figure` rather than
    going through pyplot keeps it off the global backend registry entirely, so
    this works on a CI box with no DISPLAY.
    """
    fig = ge.plot_vol_surface(synthetic_surface(), backend="matplotlib", n_k=20, n_t=12)
    assert type(fig).__module__.startswith("matplotlib")

    png = tmp_path / "surface.png"
    fig.savefig(png)
    assert png.stat().st_size > 5000


def test_plot_vol_surface_auto_falls_back_when_plotly_missing(monkeypatch, tmp_path):
    """`backend='auto'` degrades to matplotlib; `backend='plotly'` still raises."""
    real_import = builtins.__import__

    def no_plotly(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("plotly is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_plotly)
    surf = synthetic_surface()

    fig = ge.plot_vol_surface(surf, backend="auto", n_k=12, n_t=8)
    assert type(fig).__module__.startswith("matplotlib")

    with pytest.raises(ImportError):
        ge.plot_vol_surface(surf, backend="plotly", n_k=12, n_t=8)


def test_plot_vol_surface_rejects_html_without_plotly(tmp_path):
    """Asking for HTML from the static backend is an error, never a silent no-op."""
    with pytest.raises(ValueError, match="html_path requires plotly"):
        ge.plot_vol_surface(synthetic_surface(), backend="matplotlib",
                            html_path=tmp_path / "x.html", n_k=12, n_t=8)


def test_plot_vol_surface_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        ge.plot_vol_surface(synthetic_surface(), backend="bokeh")


class _ForeignSurface:
    """
    A surface exposing only `maturities`, `forwards` and `implied_vol(K, T)` --
    the interface `VolSurface`/`SSVIVolSurface` share and `GlobalESSVISurface`
    adds to. Nothing eSSVI-specific: no `slice_at`, no `forward_at`, no slices.
    """

    def __init__(self, inner):
        self.maturities = inner.maturities
        self.forwards = inner.forwards
        self._inner = inner

    def implied_vol(self, K, T):
        return self._inner.implied_vol(K, T)


def test_grid_works_on_any_surface_not_just_essvi():
    """
    The plotting path must drive `VolSurface` and `SSVIVolSurface` too, so all
    three models can be drawn on one axis convention. It reads theta and the
    forward through the public interface rather than eSSVI internals, so a
    surface with no `slice_at`/`forward_at` still works.
    """
    truth = synthetic_surface()
    foreign = _ForeignSurface(truth)

    for axis in ("k", "K", "z"):
        x_ref, t_ref, v_ref = ge.vol_surface_grid(truth, x_axis=axis, n_k=17, n_t=9)
        x, t, v = ge.vol_surface_grid(foreign, x_axis=axis, n_k=17, n_t=9)
        np.testing.assert_allclose(x, x_ref, rtol=1e-10)
        np.testing.assert_allclose(t, t_ref, rtol=1e-10)
        np.testing.assert_allclose(v, v_ref, rtol=1e-10)


def test_surface_theta_matches_the_slice_parameter():
    """
    `surface_theta_at` recovers theta from ATM implied vol. For an eSSVI
    surface that must agree with the slice's own theta -- otherwise the z axis
    would mean something different for eSSVI than for the other two models.
    """
    surf = synthetic_surface()
    for T in [surf.maturities[0], 0.3, surf.maturities[-1]]:
        assert ge.surface_theta_at(surf, T) == pytest.approx(surf.slice_at(T).theta, rel=1e-10)


def test_surface_forward_at_falls_back_to_log_linear():
    truth = synthetic_surface()
    foreign = _ForeignSurface(truth)
    for T in [truth.maturities[0], 0.3, truth.maturities[-1]]:
        assert ge.surface_forward_at(foreign, T) == pytest.approx(truth.forward_at(T), rel=1e-12)


def test_default_colormap_is_the_matplotlib_standard():
    """
    Every surface in the package is drawn with one colormap, and it is a
    registered matplotlib name rather than a bespoke ramp so `surface.py`'s own
    plot_surface_3d matches without changes.
    """
    import matplotlib
    assert ge.VOL_SURFACE_CMAP in matplotlib.colormaps
    assert ge._mpl_cmap(ge.VOL_SURFACE_CMAP).name == ge.VOL_SURFACE_CMAP


def test_colorscale_accepts_a_name_or_a_list_of_colours():
    """A name resolves to that colormap; a list still builds a custom ramp."""
    from matplotlib.colors import LinearSegmentedColormap
    assert isinstance(ge._mpl_cmap(ge.VOL_SURFACE_BLUES), LinearSegmentedColormap)

    # plotly wants the capitalised name, and an explicit scale for a list
    assert ge._plotly_colorscale("viridis") == "Viridis"
    scale = ge._plotly_colorscale(ge.VOL_SURFACE_BLUES)
    assert scale[0] == [0.0, ge.VOL_SURFACE_BLUES[0]]
    assert scale[-1] == [1.0, ge.VOL_SURFACE_BLUES[-1]]


@pytest.mark.parametrize("x_axis", ["k", "z"])
def test_k_and_z_plots_share_one_camera(x_axis):
    """
    The two axes are rendered as separate figures, so the shared camera has to
    come from the module constant rather than from sharing a figure.
    """
    fig = ge.plot_vol_surface(synthetic_surface(), x_axis=x_axis, n_k=20, n_t=14,
                              backend="matplotlib")
    ax = next(a for a in fig.axes if hasattr(a, "get_zlim"))
    assert (ax.elev, ax.azim) == (ge.VOL_SURFACE_VIEW["elev"], ge.VOL_SURFACE_VIEW["azim"])


@pytest.mark.network
def test_calibrates_against_the_live_deribit_chain():
    chain = ge.fetch_quotes("BTC", max_T=1.5)
    assert not chain.empty
    inputs = ge.chain_to_calibration_inputs(chain)
    result = ge.calibrate(**inputs)

    assert result.success
    # arbitrage-free well beyond the quoted expiries, not just on the pillars
    assert ge.check_no_arbitrage(
        result.surface, t_grid=np.linspace(0.002, 2.5, 200)) is True

    model_iv = np.array([result.surface.implied_vol(k_, t_)
                         for k_, t_ in zip(inputs["strike"], inputs["T"])])
    err = np.abs(model_iv - inputs["mid_iv"])
    # eSSVI carries three parameters per slice, so it cannot chase every quote;
    # these are loose enough to survive a quiet or a dislocated session, but
    # tight enough to fail if ITM quotes ever leak back into the chain (which
    # pushed the median above 6% and the max past 140%).
    assert np.median(err) < 0.02
    assert np.percentile(err, 90) < 0.05

    T0 = result.surface.maturities[0]
    assert 0.05 < result.surface.implied_vol(result.surface.forward_at(T0), T0) < 4.0

import numpy as np
import pytest

from svi import SVIParams, fit_svi_slice, raw_svi
import arbitrage as arb


GOOD_PARAMS = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
# High convexity / near-degenerate wing (large b, tiny sigma, rho near -1):
# a textbook butterfly-arbitrage shape -- see module docstring in arbitrage.py.
BAD_PARAMS = SVIParams(a=0.01, b=2.0, rho=-0.95, m=0.0, sigma=0.01)
# Right-wing slope b*(1+rho) = 1.5*1.5 = 2.25 >= 2 (violates Roger Lee's
# moment-formula bound / lim_{k->+inf} d+(k) = -inf), but g(k) >= 0 across
# the entire DEFAULT_K_GRID -- demonstrates that the g(k) grid check alone
# cannot catch this condition; it's asymptotic, not a pointwise property.
STEEP_WING_PARAMS = SVIParams(a=0.01, b=1.5, rho=0.5, m=0.0, sigma=1.2)


def test_butterfly_check_passes_for_valid_slice():
    assert arb.check_butterfly(GOOD_PARAMS) is True


def test_butterfly_check_rejects_bad_slice():
    with pytest.raises(arb.ArbitrageError):
        arb.check_butterfly(BAD_PARAMS)


def test_fit_svi_slice_raises_when_fit_is_arbitrageable():
    # Force the optimizer toward the known-bad region by starting there and
    # feeding it data generated from those exact (bad) parameters, so the
    # least-squares fit converges right back to an arbitrageable slice.
    T = 0.1
    k = np.linspace(-0.3, 0.3, 15)
    w = raw_svi(k, *BAD_PARAMS.as_array())
    mid_iv = np.sqrt(w / T)
    weight = np.ones_like(k)

    with pytest.raises(arb.ArbitrageError):
        fit_svi_slice(k, mid_iv, weight, T, x0=BAD_PARAMS.as_array())


def test_fit_svi_slice_succeeds_when_validate_disabled():
    T = 0.1
    k = np.linspace(-0.3, 0.3, 15)
    w = raw_svi(k, *BAD_PARAMS.as_array())
    mid_iv = np.sqrt(w / T)
    weight = np.ones_like(k)

    result = fit_svi_slice(k, mid_iv, weight, T, x0=BAD_PARAMS.as_array(), validate=False)
    assert result.params is not None


def test_calendar_check_passes_for_increasing_total_variance():
    early = (0.1, GOOD_PARAMS)
    late = (0.5, SVIParams(a=0.08, b=0.1, rho=-0.3, m=0.0, sigma=0.2))
    assert arb.check_calendar([early, late]) is True


def test_calendar_check_rejects_decreasing_total_variance():
    early = (0.1, GOOD_PARAMS)
    # same shape but lower level at a *longer* maturity -> total variance
    # drops moving from T=0.1 to T=0.5, which is a calendar arbitrage.
    late = (0.5, SVIParams(a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.2))
    with pytest.raises(arb.ArbitrageError):
        arb.check_calendar([early, late])


def test_calendar_check_sorts_input_by_maturity():
    early = (0.1, GOOD_PARAMS)
    late = (0.5, SVIParams(a=0.08, b=0.1, rho=-0.3, m=0.0, sigma=0.2))
    # pass out of order -- check_calendar must sort internally
    assert arb.check_calendar([late, early]) is True


def test_d_plus_diverges_to_minus_infinity_for_valid_slice():
    # actual asymptotic test, not just the closed-form b*(1+rho) < 2 shortcut:
    # d+(k) should trend to -inf (strictly decreasing, unboundedly negative)
    # as k grows for a slice that respects the wing condition.
    k_probe = np.array([1e2, 1e4, 1e6, 1e8])
    d_vals = arb.d_plus(k_probe, GOOD_PARAMS)
    assert np.all(np.diff(d_vals) < 0)
    assert d_vals[-1] < -1e4


def test_d_plus_diverges_to_plus_infinity_for_steep_wing_slice():
    # the same probe applied to STEEP_WING_PARAMS shows d+(k) heading to
    # +inf instead -- concretely, not just via the parameter inequality.
    k_probe = np.array([1e2, 1e4, 1e6, 1e8])
    d_vals = arb.d_plus(k_probe, STEEP_WING_PARAMS)
    assert np.all(np.diff(d_vals) > 0)
    assert d_vals[-1] > 1e2


def test_wing_condition_passes_for_valid_slice():
    assert arb.check_wing_condition(GOOD_PARAMS) is True


def test_wing_condition_rejects_steep_right_wing():
    with pytest.raises(arb.ArbitrageError):
        arb.check_wing_condition(STEEP_WING_PARAMS)


def test_wing_condition_boundary_is_exclusive():
    # b*(1+rho) exactly 2.0 must fail: the limit isn't -inf at equality,
    # only strictly below the bound.
    boundary = SVIParams(a=0.01, b=1.0, rho=1.0, m=0.0, sigma=0.3)
    assert arb._wing_slope(boundary) == pytest.approx(2.0)
    with pytest.raises(arb.ArbitrageError):
        arb.check_wing_condition(boundary)


def test_g_grid_check_alone_would_miss_the_steep_wing_case():
    # sanity check on the test fixture itself: g(k) >= 0 everywhere on the
    # check grid even though the slice is arbitrageable via the wing
    # condition -- proves the two checks are independent, not redundant.
    g = arb.g_function(arb.DEFAULT_K_GRID, STEEP_WING_PARAMS)
    assert g.min() >= 0


def test_butterfly_check_rejects_steep_wing_slice():
    with pytest.raises(arb.ArbitrageError, match="right-wing slope"):
        arb.check_butterfly(STEEP_WING_PARAMS)


def test_find_slice_crossings_matches_dense_grid_search():
    p1 = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    p2 = SVIParams(a=0.05, b=0.15, rho=0.2, m=0.1, sigma=0.3)

    crossings = arb.find_slice_crossings(p1, p2)

    kk = np.linspace(-5, 5, 200_001)
    diff = raw_svi(kk, *p2.as_array()) - raw_svi(kk, *p1.as_array())
    grid_crossing_idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    approx_roots = kk[grid_crossing_idx]

    assert len(crossings) == len(approx_roots) == 1
    assert crossings[0] == pytest.approx(approx_roots[0], abs=1e-3)


def test_find_slice_crossings_finds_multiple_roots():
    # slices whose difference changes sign twice, found via random search
    # and cross-checked against a dense grid.
    p1 = SVIParams(a=0.05000708815108327, b=0.13046734776898555, rho=0.11316003623963611,
                    m=-0.09725326469572004, sigma=0.25664760021126454)
    p2 = SVIParams(a=0.07231920464033546, b=0.10678939838334493, rho=0.14782457362325085,
                    m=-0.2495907938505691, sigma=0.4330576590613592)

    crossings = arb.find_slice_crossings(p1, p2)

    kk = np.linspace(-5, 5, 200_001)
    diff = raw_svi(kk, *p2.as_array()) - raw_svi(kk, *p1.as_array())
    approx_roots = kk[np.where(np.diff(np.sign(diff)) != 0)[0]]

    assert len(crossings) == 2 == len(approx_roots)
    assert crossings == pytest.approx(sorted(approx_roots), abs=1e-3)


def test_find_slice_crossings_empty_for_parallel_shift():
    # pure vertical shift in total variance (same shape, different level
    # everywhere) -> the slices never cross.
    p1 = GOOD_PARAMS
    p2 = SVIParams(a=GOOD_PARAMS.a + 0.05, b=GOOD_PARAMS.b, rho=GOOD_PARAMS.rho,
                    m=GOOD_PARAMS.m, sigma=GOOD_PARAMS.sigma)
    assert arb.find_slice_crossings(p1, p2) == []


def test_check_calendar_exact_passes_for_increasing_total_variance():
    early = (0.1, GOOD_PARAMS)
    late = (0.5, SVIParams(a=0.08, b=0.1, rho=-0.3, m=0.0, sigma=0.2))
    assert arb.check_calendar_exact([early, late]) is True


def test_check_calendar_exact_rejects_decreasing_total_variance():
    early = (0.1, GOOD_PARAMS)
    late = (0.5, SVIParams(a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.2))
    with pytest.raises(arb.ArbitrageError):
        arb.check_calendar_exact([early, late])


def test_check_calendar_exact_sorts_input_by_maturity():
    early = (0.1, GOOD_PARAMS)
    late = (0.5, SVIParams(a=0.08, b=0.1, rho=-0.3, m=0.0, sigma=0.2))
    assert arb.check_calendar_exact([late, early]) is True


def test_check_calendar_exact_catches_narrow_crossing_grid_check_misses():
    # These two slices cross twice just ~0.0046 apart in k -- narrower than
    # DEFAULT_K_GRID's ~0.025 spacing, so check_calendar's grid sampling
    # steps clean over the dip and reports no violation. check_calendar_exact
    # finds the crossings exactly (via the quartic) and correctly rejects.
    p1 = SVIParams(a=0.02236232183296754, b=0.13667759465470686, rho=0.03925944666914394,
                    m=-0.012900631505364302, sigma=0.10832690341116893)
    p2 = SVIParams(a=0.02081207476543416, b=0.15249346064086547, rho=0.07957366224968813,
                    m=-0.0013092781972601879, sigma=0.10924503147696951)

    assert arb.check_calendar([(0.1, p1), (0.5, p2)]) is True  # grid check misses it

    with pytest.raises(arb.ArbitrageError):
        arb.check_calendar_exact([(0.1, p1), (0.5, p2)])


def test_g_function_matches_manual_formula():
    k = np.array([-0.4, 0.0, 0.3])
    x = k - GOOD_PARAMS.m
    root = np.sqrt(x ** 2 + GOOD_PARAMS.sigma ** 2)
    w = GOOD_PARAMS.a + GOOD_PARAMS.b * (GOOD_PARAMS.rho * x + root)
    dw = GOOD_PARAMS.b * (GOOD_PARAMS.rho + x / root)
    d2w = GOOD_PARAMS.b * GOOD_PARAMS.sigma ** 2 / root ** 3
    expected = (1 - k * dw / (2 * w)) ** 2 - (dw ** 2 / 4) * (1 / w + 0.25) + d2w / 2

    got = arb.g_function(k, GOOD_PARAMS)
    assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Section 3.1 domain condition: "the obvious condition a + b sigma
# sqrt(1-rho^2) >= 0, which ensures that w(k;chi_R) >= 0 for all k". This is
# not one of the two arbitrage conditions -- it's a precondition for the
# slice to be a total-variance curve at all -- and it genuinely has to be
# checked separately: g(k) divides by w(k), so on a slice whose variance is
# negative everywhere the formula still returns finite positive numbers and
# the grid test passes a slice whose implied vol is nonsense.
# ---------------------------------------------------------------------------
NEGATIVE_VARIANCE_PARAMS = SVIParams(
    a=-0.671141933663538, b=0.024112738920300858, rho=0.2885273732383966,
    m=0.8796375340347722, sigma=4.179490390336368,
)


def test_minimum_total_variance_matches_the_curve_minimum():
    # closed form from Section 3.1 agrees with a dense numerical minimum
    for params in (GOOD_PARAMS, STEEP_WING_PARAMS, NEGATIVE_VARIANCE_PARAMS):
        dense = raw_svi(np.linspace(-60.0, 60.0, 400001), *params.as_array()).min()
        assert arb.minimum_total_variance(params) == pytest.approx(dense, abs=1e-6)


def test_check_minimum_variance_passes_for_valid_slice():
    assert arb.check_minimum_variance(GOOD_PARAMS) is True


def test_check_minimum_variance_rejects_negative_total_variance():
    with pytest.raises(arb.ArbitrageError, match="total variance goes negative"):
        arb.check_minimum_variance(NEGATIVE_VARIANCE_PARAMS)


def test_negative_variance_slice_is_invisible_to_the_g_function_alone():
    # the reason the condition needs its own check: w < 0 at every grid
    # point, yet g(k) stays comfortably positive and reports no violation
    w = NEGATIVE_VARIANCE_PARAMS.total_variance(arb.DEFAULT_K_GRID)
    assert (w < 0).all()
    assert arb.g_function(arb.DEFAULT_K_GRID, NEGATIVE_VARIANCE_PARAMS).min() > 0
    # ... and the implied vol silently clamps to zero rather than erroring
    assert NEGATIVE_VARIANCE_PARAMS.implied_vol(0.0, 1.0) == 0.0


def test_check_butterfly_gates_on_the_domain_condition():
    with pytest.raises(arb.ArbitrageError, match="total variance goes negative"):
        arb.check_butterfly(NEGATIVE_VARIANCE_PARAMS)


# ---------------------------------------------------------------------------
# Lemma 2.2 (proof): the implied risk-neutral density
#   p(k) = g(k)/sqrt(2 pi w(k)) * exp(-d_-(k)^2/2)
# ---------------------------------------------------------------------------
def test_implied_density_integrates_to_one():
    # p is a density in k, so it must integrate to 1 over the real line
    k = np.linspace(-8.0, 8.0, 320001)
    total = float(np.trapezoid(arb.implied_density(k, GOOD_PARAMS), k))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_implied_density_is_non_negative_for_an_arbitrage_free_slice():
    assert arb.check_butterfly(GOOD_PARAMS) is True
    assert arb.implied_density(np.linspace(-5.0, 5.0, 4001), GOOD_PARAMS).min() >= 0.0


def test_implied_density_goes_negative_exactly_where_g_does():
    # Definition 2.3 is about the density; check_butterfly gates on g(k).
    # The two factors multiplying g are strictly positive, so they agree.
    vogt = SVIParams(a=-0.0410, b=0.1331, m=0.3586, rho=0.3060, sigma=0.4153)
    k = np.linspace(-3.0, 3.0, 2001)
    density, g = arb.implied_density(k, vogt), arb.g_function(k, vogt)
    assert density.min() < 0
    assert np.array_equal(np.sign(density), np.sign(g))


def test_d_minus_matches_its_definition():
    k = np.linspace(-2.0, 2.0, 41)
    w = GOOD_PARAMS.total_variance(k)
    assert arb.d_minus(k, GOOD_PARAMS) == pytest.approx(-k / np.sqrt(w) - np.sqrt(w) / 2)
    # d+ - d- = sqrt(w)
    assert arb.d_plus(k, GOOD_PARAMS) - arb.d_minus(k, GOOD_PARAMS) == pytest.approx(np.sqrt(w))


# ---------------------------------------------------------------------------
# Definition 5.1: crossedness
# ---------------------------------------------------------------------------
def test_crossedness_is_null_when_slices_do_not_cross():
    lower = SVIParams(a=0.05, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    upper = SVIParams(a=0.10, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    assert arb.find_slice_crossings(lower, upper) == []
    assert arb.crossedness(lower, upper) == 0.0
    # "If n = 0, the crossedness is null" -- so an entirely-inverted pair
    # also scores zero. Crossedness is a calibration penalty, not a test.
    assert arb.crossedness(upper, lower) == 0.0
    with pytest.raises(arb.ArbitrageError):
        arb.check_calendar([(0.1, upper), (0.5, lower)])


def test_crossedness_is_positive_for_a_crossing_pair():
    p1 = SVIParams(a=0.02236232183296754, b=0.13667759465470686, rho=0.03925944666914394,
                   m=-0.012900631505364302, sigma=0.10832690341116893)
    p2 = SVIParams(a=0.02081207476543416, b=0.15249346064086547, rho=0.07957366224968813,
                   m=-0.0013092781972601879, sigma=0.10924503147696951)
    assert len(arb.find_slice_crossings(p1, p2)) == 2
    assert arb.crossedness(p1, p2) > 0


def test_crossedness_probe_points_follow_the_paper():
    # k~_1 = k_1 - 1, midpoints between consecutive crossings, k~_n+1 = k_n + 1
    p1 = SVIParams(a=0.02236232183296754, b=0.13667759465470686, rho=0.03925944666914394,
                   m=-0.012900631505364302, sigma=0.10832690341116893)
    p2 = SVIParams(a=0.02081207476543416, b=0.15249346064086547, rho=0.07957366224968813,
                   m=-0.0013092781972601879, sigma=0.10924503147696951)
    crossings = arb.find_slice_crossings(p1, p2)
    probes = [crossings[0] - 1.0, 0.5 * (crossings[0] + crossings[1]), crossings[-1] + 1.0]
    gaps = raw_svi(np.array(probes), *p1.as_array()) - raw_svi(np.array(probes), *p2.as_array())
    assert arb.crossedness(p1, p2) == pytest.approx(max(0.0, gaps.max()))

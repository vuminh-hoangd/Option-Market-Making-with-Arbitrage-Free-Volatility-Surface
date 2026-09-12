import numpy as np
import pytest

import arbitrage as arb
import ssvi
from svi import SVIParams


THETA_SMALL_TO_LARGE = np.geomspace(0.01, 20.0, 60)


# ---------------------------------------------------------------------------
# Example 4.1 (Heston-like phi): Theorem 4.1's calendar condition holds for
# every lam > 0 and every rho in (-1,1) -- that's the content of the example.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lam", [0.05, 0.2, 1.0, 3.0, 10.0])
@pytest.mark.parametrize("rho", [-0.9, -0.5, -0.1, 0.0, 0.3, 0.7, 0.95])
def test_heston_like_satisfies_calendar_condition_for_all_lam_and_rho(lam, rho):
    params = ssvi.SSVIParams(rho=rho, phi=ssvi.phi_heston_like(lam))
    assert ssvi.check_ssvi_calendar(THETA_SMALL_TO_LARGE, params) is True


# ---------------------------------------------------------------------------
# Example 4.2 (power law): the calendar condition holds unconditionally for
# any 0 < gamma < 1 and any rho in (-1,1) -- algebraically,
# d/dtheta(theta*phi(theta)) = eta*(1-gamma)*theta^-gamma is theta-independent
# relative to phi(theta), and (1-gamma) < 1 <= (1/rho^2)(1+sqrt(1-rho^2))
# for all rho in (-1,1), so the bound is never binding.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gamma", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("rho", [-0.95, -0.5, -0.1, 0.0, 0.1, 0.5, 0.95])
def test_power_law_satisfies_calendar_condition_for_all_gamma_and_rho(gamma, rho):
    params = ssvi.SSVIParams(rho=rho, phi=ssvi.phi_power_law(1.0, gamma))
    assert ssvi.check_ssvi_calendar(THETA_SMALL_TO_LARGE, params) is True


def test_power_law_gamma_half_gives_constant_jw_params():
    # Special case noted in the paper: with gamma=1/2, the JW parameters
    # psi_t, p_t, c_t are constant across theta (v_t and v_tilde_t are not).
    eta, gamma, rho = 0.9, 0.5, -0.35
    params = ssvi.SSVIParams(rho=rho, phi=ssvi.phi_power_law(eta, gamma))

    jw_by_theta = []
    for theta in [0.05, 0.3, 1.0, 4.0, 10.0]:
        raw = ssvi.ssvi_slice_to_raw_svi(theta, params)
        jw_by_theta.append(ssvi.svi_jw_params(raw, T=theta))

    for key, expected in [("psi_t", eta * rho / 2), ("p_t", eta * (1 - rho) / 2), ("c_t", eta * (1 + rho) / 2)]:
        values = [jw[key] for jw in jw_by_theta]
        assert values == pytest.approx([expected] * len(values), abs=1e-8)


def test_power_law_gamma_not_half_gives_theta_dependent_wing_slopes():
    # sanity check on the test above: away from gamma=1/2, c_t genuinely
    # varies with theta, confirming gamma=1/2 is a special case and not an
    # artifact of the test setup.
    params = ssvi.SSVIParams(rho=-0.3, phi=ssvi.phi_power_law(0.9, 0.3))
    c_t_values = [
        ssvi.svi_jw_params(ssvi.ssvi_slice_to_raw_svi(theta, params), T=theta)["c_t"]
        for theta in [0.05, 1.0, 10.0]
    ]
    assert len(set(np.round(c_t_values, 6))) > 1


# ---------------------------------------------------------------------------
# Theorem 4.2 boundary behavior (Remark 4.4): with eta=1, gamma=1/2, rho=0,
# condition 1 (theta*phi(theta)*(1+|rho|) < 4) reduces to sqrt(theta) < 4,
# i.e. theta < 16 -- and condition 2 is theta-independent (always 1 <= 4),
# so this isolates exactly where condition 1 starts failing.
# ---------------------------------------------------------------------------
def test_ssvi_butterfly_passes_before_crossing_and_fails_after():
    params = ssvi.SSVIParams(rho=0.0, phi=ssvi.phi_power_law(1.0, 0.5))
    crossing = 16.0

    assert ssvi.check_ssvi_butterfly(np.array([1.0, 4.0, 9.0, 15.9]), params) is True

    for theta_bad in (16.0, 20.0, 30.0):
        with pytest.raises(arb.ArbitrageError, match="condition 1"):
            ssvi.check_ssvi_butterfly(np.array([1.0, theta_bad]), params)

    # crossing itself is computable in closed form and matches the theorem
    cond1_at_crossing = crossing * params.phi(crossing) * (1 + abs(params.rho))
    assert cond1_at_crossing == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Cross-validation: closed-form check_ssvi_butterfly passing should imply
# the independent numerical arbitrage.check_butterfly also passes on the
# raw-SVI conversion, at several theta values -- the main integration test
# tying this module to arbitrage.py's existing machinery.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rho,eta,gamma", [
    (-0.3, 0.8, 0.3),
    (0.2, 0.5, 0.6),
    (-0.6, 1.0, 0.4),
    (0.0, 1.2, 0.5),
])
def test_ssvi_butterfly_pass_implies_raw_svi_butterfly_pass(rho, eta, gamma):
    params = ssvi.SSVIParams(rho=rho, phi=ssvi.phi_power_law(eta, gamma))
    thetas = [0.02, 0.1, 0.5, 1.0, 3.0]

    assert ssvi.check_ssvi_butterfly(np.array(thetas), params) is True

    for theta in thetas:
        raw_params = ssvi.ssvi_slice_to_raw_svi(theta, params)
        assert arb.check_butterfly(raw_params) is True


def test_check_ssvi_no_static_arbitrage_runs_cross_validation():
    params = ssvi.SSVIParams(rho=-0.3, phi=ssvi.phi_power_law(0.8, 0.4))
    t_grid = np.array([0.1, 0.5, 1.0, 2.0])
    theta_values = np.array([0.05, 0.2, 0.5, 1.1])  # non-decreasing, consistent with t_grid

    assert ssvi.check_ssvi_no_static_arbitrage(t_grid, theta_values, params) is True


def test_check_ssvi_no_static_arbitrage_rejects_non_monotonic_theta():
    params = ssvi.SSVIParams(rho=-0.3, phi=ssvi.phi_power_law(0.8, 0.4))
    t_grid = np.array([0.1, 0.5, 1.0])
    theta_values = np.array([0.2, 0.1, 0.5])  # decreases from t=0.1 to t=0.5

    with pytest.raises(arb.ArbitrageError):
        ssvi.check_ssvi_no_static_arbitrage(t_grid, theta_values, params)


# ---------------------------------------------------------------------------
# Theorem 4.3: a non-negative, non-decreasing alpha(theta) shift preserves
# no-arbitrage of an already arbitrage-free SSVI surface.
# ---------------------------------------------------------------------------
def test_theorem_4_3_shift_preserves_arbitrage_free_surface():
    params = ssvi.SSVIParams(rho=-0.3, phi=ssvi.phi_power_law(0.8, 0.4))
    theta_grid = np.array([0.05, 0.2, 0.5, 1.1, 2.0])
    assert ssvi.check_ssvi_no_static_arbitrage(theta_grid, theta_grid, params) is True

    alpha = lambda theta: 0.01 * theta
    w_alpha = ssvi.apply_theorem_4_3_shift(theta_grid, params, alpha)

    for theta in theta_grid:
        shifted_raw = ssvi.ssvi_slice_to_raw_svi(float(theta), params, alpha=alpha(theta))
        assert arb.check_butterfly(shifted_raw) is True
        # the closure should reproduce the same total variance as the shifted raw params
        k_probe = np.array([-0.3, 0.0, 0.3])
        from svi import raw_svi
        assert w_alpha(k_probe, theta) == pytest.approx(raw_svi(k_probe, *shifted_raw.as_array()))


def test_apply_theorem_4_3_shift_rejects_decreasing_alpha():
    params = ssvi.SSVIParams(rho=-0.3, phi=ssvi.phi_power_law(0.8, 0.4))
    theta_grid = np.array([0.1, 0.5, 1.0])
    with pytest.raises(arb.ArbitrageError):
        ssvi.apply_theorem_4_3_shift(theta_grid, params, alpha=lambda theta: -theta)


def test_apply_theorem_4_3_shift_rejects_negative_alpha():
    params = ssvi.SSVIParams(rho=-0.3, phi=ssvi.phi_power_law(0.8, 0.4))
    theta_grid = np.array([0.1, 0.5, 1.0])
    with pytest.raises(arb.ArbitrageError):
        ssvi.apply_theorem_4_3_shift(theta_grid, params, alpha=lambda theta: theta - 1.0)


# ---------------------------------------------------------------------------
# Negative control: Theorem 4.3 only preserves no-arbitrage of a base
# surface that *already* satisfies Corollary 4.1 -- it cannot rescue one
# that doesn't (in the spirit of Remark 5.1's counterexample regime, where
# an ATM-only vertical shift can't repair a smile that's already
# arbitrageable in its curvature/wings). `apply_theorem_4_3_shift` itself
# only validates alpha's hypotheses and never asserts the shifted surface
# is arbitrage-free, so it must not raise merely because the *base*
# surface is bad -- and the shifted slice must still fail check_butterfly.
# ---------------------------------------------------------------------------
def test_theorem_4_3_shift_does_not_rescue_an_arbitrageable_base_surface():
    # same configuration as the boundary test above: theta=20 is well past
    # the theta*phi(theta)*(1+|rho|) < 4 crossing at theta=16, so this base
    # slice is genuinely arbitrageable.
    params = ssvi.SSVIParams(rho=0.0, phi=ssvi.phi_power_law(1.0, 0.5))
    bad_theta = 20.0
    with pytest.raises(arb.ArbitrageError):
        ssvi.check_ssvi_butterfly(np.array([bad_theta]), params)

    # apply_theorem_4_3_shift does not check the base surface at all, so a
    # valid (non-negative, non-decreasing) alpha is accepted regardless
    alpha = lambda theta: 0.01 * theta
    w_alpha = ssvi.apply_theorem_4_3_shift(np.array([bad_theta]), params, alpha)
    assert callable(w_alpha)

    # but the shift does NOT fix the underlying arbitrage: the shifted
    # slice, converted to raw SVI, still fails the numerical butterfly check
    shifted_raw = ssvi.ssvi_slice_to_raw_svi(bad_theta, params, alpha=alpha(bad_theta))
    with pytest.raises(arb.ArbitrageError):
        arb.check_butterfly(shifted_raw)


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------
def test_ssvi_total_variance_at_the_money_equals_theta():
    params = ssvi.SSVIParams(rho=-0.4, phi=ssvi.phi_power_law(0.7, 0.35))
    for theta in [0.01, 0.5, 3.0]:
        assert ssvi.ssvi_total_variance(0.0, theta, params) == pytest.approx(theta)


def test_ssvi_slice_to_raw_svi_reproduces_same_total_variance():
    params = ssvi.SSVIParams(rho=0.25, phi=ssvi.phi_heston_like(0.8))
    theta = 0.6
    raw = ssvi.ssvi_slice_to_raw_svi(theta, params)

    from svi import raw_svi
    k = np.linspace(-2, 2, 21)
    direct = ssvi.ssvi_total_variance(k, theta, params)
    converted = raw_svi(k, *raw.as_array())
    assert converted == pytest.approx(direct, abs=1e-10)


def test_check_theta_monotonic_passes_and_rejects():
    t_grid = np.array([0.1, 0.3, 0.6])
    assert ssvi.check_theta_monotonic(t_grid, np.array([0.05, 0.1, 0.2])) is True
    with pytest.raises(arb.ArbitrageError):
        ssvi.check_theta_monotonic(t_grid, np.array([0.2, 0.1, 0.3]))


# ---------------------------------------------------------------------------
# Lemma 3.2: the inverse of the SVI-JW map (3.5). Tested as an exact
# round-trip against `svi_jw_params`, covering both branches of the lemma
# (m != 0, and the m = 0 case where alpha is unbounded).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("params", [
    SVIParams(a=-0.0410, b=0.1331, m=0.3586, rho=0.3060, sigma=0.4153),  # Vogt, Example 3.1
    SVIParams(a=0.04, b=0.15, rho=-0.35, m=0.05, sigma=0.25),
    SVIParams(a=0.02, b=0.30, rho=0.45, m=-0.40, sigma=0.60),
    SVIParams(a=0.01, b=0.08, rho=-0.80, m=0.90, sigma=0.15),
    SVIParams(a=0.03, b=0.20, rho=-0.25, m=0.0, sigma=0.35),   # m = 0 branch
    SVIParams(a=0.05, b=0.40, rho=0.62, m=0.0, sigma=0.22),    # m = 0 branch
])
@pytest.mark.parametrize("T", [0.25, 1.0, 2.5])
def test_svi_jw_to_raw_inverts_svi_jw_params(params, T):
    recovered = ssvi.svi_jw_to_raw(ssvi.svi_jw_params(params, T), T)
    assert recovered.as_array() == pytest.approx(params.as_array(), abs=1e-9)


def test_svi_jw_to_raw_rejects_beta_outside_unit_interval():
    # beta = rho - 2 psi_t sqrt(w_t)/b must lie in [-1,1] for the lemma to
    # apply; an outsized ATM skew relative to the wings pushes it out.
    jw = {"v_t": 0.04, "psi_t": -5.0, "p_t": 0.5, "c_t": 0.5, "v_tilde_t": 0.02}
    with pytest.raises(ValueError, match="beta"):
        ssvi.svi_jw_to_raw(jw, T=1.0)


def test_svi_jw_to_raw_rejects_degenerate_m_zero_branch_at_rho_zero():
    # rho = 0 with m = 0 puts the smile minimum exactly at the money, so
    # v_t == v_tilde_t and sigma carries no information in JW coordinates
    symmetric = SVIParams(a=0.03, b=0.20, rho=0.0, m=0.0, sigma=0.35)
    with pytest.raises(ValueError, match="degenerate"):
        ssvi.svi_jw_to_raw(ssvi.svi_jw_params(symmetric, 1.0), T=1.0)


# ---------------------------------------------------------------------------
# Section 5.1 + Example 5.1: eliminating butterfly arbitrage from the Vogt
# smile. The paper publishes both the original SVI-JW parameters and the
# repaired (c_t, v_tilde_t), so these are checked against its actual
# numbers rather than against a self-consistent recomputation.
# ---------------------------------------------------------------------------
VOGT_PARAMS = SVIParams(a=-0.0410, b=0.1331, m=0.3586, rho=0.3060, sigma=0.4153)
VOGT_T = 1.0
# Example 5.1: "(v_t, psi_t, p_t, c_t, v_tilde_t) = ..."
VOGT_JW = {
    "v_t": 0.01742625, "psi_t": -0.1752111, "p_t": 0.6997381,
    "c_t": 1.316798, "v_tilde_t": 0.0116249,
}
# Example 5.1: "choosing (c_t, v_tilde_t) = (c_t^o, v_tilde_t^o) := ...
# gives a smile free of butterfly arbitrage"
VOGT_REPAIRED_C_T = 0.3493158
VOGT_REPAIRED_V_TILDE_T = 0.01548182


def test_vogt_smile_jw_parameters_match_example_5_1():
    jw = ssvi.svi_jw_params(VOGT_PARAMS, VOGT_T)
    for key, expected in VOGT_JW.items():
        assert jw[key] == pytest.approx(expected, abs=5e-7)


def test_vogt_smile_has_butterfly_arbitrage():
    # Figure 1: g dips below zero even though the smile itself looks benign
    assert arb.g_function(arb.DEFAULT_K_GRID, VOGT_PARAMS).min() < 0
    with pytest.raises(arb.ArbitrageError):
        arb.check_butterfly(VOGT_PARAMS)


def test_eliminate_butterfly_arbitrage_reproduces_example_5_1_parameters():
    repaired_jw = ssvi.svi_jw_params(
        ssvi.eliminate_butterfly_arbitrage(VOGT_PARAMS, VOGT_T), VOGT_T
    )
    assert repaired_jw["c_t"] == pytest.approx(VOGT_REPAIRED_C_T, abs=5e-7)
    assert repaired_jw["v_tilde_t"] == pytest.approx(VOGT_REPAIRED_V_TILDE_T, abs=5e-8)


def test_eliminate_butterfly_arbitrage_holds_v_psi_and_p_fixed():
    # the recipe is defined as fixing v_t, psi_t, p_t and moving only the
    # call wing and the minimum variance
    repaired_jw = ssvi.svi_jw_params(
        ssvi.eliminate_butterfly_arbitrage(VOGT_PARAMS, VOGT_T), VOGT_T
    )
    for key in ("v_t", "psi_t", "p_t"):
        assert repaired_jw[key] == pytest.approx(VOGT_JW[key], abs=5e-7)


def test_eliminate_butterfly_arbitrage_produces_an_arbitrage_free_slice():
    repaired = ssvi.eliminate_butterfly_arbitrage(VOGT_PARAMS, VOGT_T)
    assert arb.check_butterfly(repaired) is True
    assert arb.g_function(arb.DEFAULT_K_GRID, repaired).min() >= 0


@pytest.mark.parametrize("params", [
    SVIParams(a=0.1570583563821652, b=0.29765545336774507, rho=0.5741850991791186,
              m=0.7346671036848293, sigma=0.1980736219120287),
    SVIParams(a=0.018478490480862322, b=0.09306587747517403, rho=0.18694500391567237,
              m=0.7915155034825632, sigma=0.08098492309240822),
    SVIParams(a=0.11896153280264453, b=0.8565843128604752, rho=0.8285769755067289,
              m=0.6945896679045498, sigma=0.4955910024244023),
])
def test_eliminate_butterfly_arbitrage_repairs_other_arbitrageable_slices(params):
    with pytest.raises(arb.ArbitrageError):
        arb.check_butterfly(params)
    assert arb.check_butterfly(ssvi.eliminate_butterfly_arbitrage(params, 1.0)) is True


def test_eliminate_butterfly_arbitrage_puts_the_slice_into_ssvi_form():
    # the recipe's two replacements are Lemma 4.1's identities, so the
    # repaired slice must satisfy them exactly: c_t = p_t + 2 psi_t and
    # v_tilde_t = v_t (1 - rho^2) with rho read off the wings
    repaired_jw = ssvi.svi_jw_params(
        ssvi.eliminate_butterfly_arbitrage(VOGT_PARAMS, VOGT_T), VOGT_T
    )
    assert repaired_jw["c_t"] == pytest.approx(repaired_jw["p_t"] + 2 * repaired_jw["psi_t"])
    _, _, rho = ssvi.svi_jw_to_ssvi_coordinates(repaired_jw, VOGT_T)
    assert repaired_jw["v_tilde_t"] == pytest.approx(repaired_jw["v_t"] * (1 - rho ** 2))


def test_eliminate_butterfly_arbitrage_rejects_when_theorem_4_2_still_fails():
    # SSVI form alone is not sufficient -- Theorem 4.2's conditions must
    # hold on the result too. A small-sigma smile gives a large phi, and
    # condition 2 (theta phi^2 (1+|rho|) <= 4) fails badly.
    params = SVIParams(a=0.01, b=0.5, rho=-0.95, m=0.0, sigma=0.01)
    with pytest.raises(arb.ArbitrageError, match="condition 2"):
        ssvi.eliminate_butterfly_arbitrage(params, 1.0)

    # validate=False returns the bare Section 5.1 output, which is in SSVI
    # form but genuinely still arbitrageable
    unchecked = ssvi.eliminate_butterfly_arbitrage(params, 1.0, validate=False)
    assert arb.g_function(arb.DEFAULT_K_GRID, unchecked).min() < 0


def test_eliminate_butterfly_arbitrage_rejects_the_excluded_sigma_zero_slice():
    # Section 3.1 excludes sigma = 0 ("which corresponds to a linear
    # smile"). That is exactly where the recipe degenerates: the assigned
    # call wing c_t' = p_t + 2 psi_t collapses to zero, leaving no wing to
    # put the repaired smile on. For every sigma > 0 the recipe is feasible.
    degenerate = SVIParams(a=0.02, b=0.9, rho=-0.99, m=2.5, sigma=0.0)
    jw = ssvi.svi_jw_params(degenerate, 1.0)
    assert jw["p_t"] + 2 * jw["psi_t"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="infeasible"):
        ssvi.eliminate_butterfly_arbitrage(degenerate, 1.0)


def test_remark_5_1_theorem_4_3_shift_cannot_help_the_vogt_smile():
    # "For alpha_t to help, we must have alpha_t > 0; it is straightforward
    # to verify that this translates to the condition v_t(1-rho^2) < v_tilde_t
    # which is violated in the Vogt case."
    assert ssvi.theorem_4_3_shift_can_help(VOGT_PARAMS, VOGT_T) is False
    jw = ssvi.svi_jw_params(VOGT_PARAMS, VOGT_T)
    assert jw["v_t"] * (1 - VOGT_PARAMS.rho ** 2) > jw["v_tilde_t"]


def test_remark_5_1_condition_holds_for_a_slice_with_a_deep_minimum():
    params = SVIParams(a=0.01, b=0.10, rho=-0.30, m=0.0, sigma=0.90)
    jw = ssvi.svi_jw_params(params, 1.0)
    assert jw["v_t"] * (1 - params.rho ** 2) < jw["v_tilde_t"]
    assert ssvi.theorem_4_3_shift_can_help(params, 1.0) is True


# ---------------------------------------------------------------------------
# Lemma 4.1: the SVI-JW parameters of an SSVI surface, in closed form.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta,T", [(0.05, 0.25), (0.3, 1.0), (1.2, 2.0)])
def test_ssvi_jw_params_match_conversion_through_raw_svi(theta, T):
    params = ssvi.SSVIParams(rho=-0.35, phi=ssvi.phi_power_law(0.9, 0.4))
    direct = ssvi.ssvi_jw_params(theta, T, params)
    via_raw = ssvi.svi_jw_params(ssvi.ssvi_slice_to_raw_svi(theta, params), T)
    for key in direct:
        assert direct[key] == pytest.approx(via_raw[key], abs=1e-12)


def test_lemma_4_1_v_tilde_is_v_times_one_minus_rho_squared():
    params = ssvi.SSVIParams(rho=0.4, phi=ssvi.phi_heston_like(0.7))
    jw = ssvi.ssvi_jw_params(0.6, 1.5, params)
    assert jw["v_tilde_t"] == pytest.approx(jw["v_t"] * (1 - params.rho ** 2))


# ---------------------------------------------------------------------------
# Equation (4.2): ATM volatility skew.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta,T", [(0.05, 0.25), (0.3, 1.0), (1.2, 2.0)])
def test_eq_4_2_matches_numerical_atm_volatility_skew(theta, T):
    params = ssvi.SSVIParams(rho=-0.35, phi=ssvi.phi_power_law(0.9, 0.4))
    h = 1e-6
    sigma = lambda k: np.sqrt(ssvi.ssvi_total_variance(k, theta, params) / T)
    numerical = (sigma(h) - sigma(-h)) / (2 * h)
    assert ssvi.ssvi_atm_volatility_skew(theta, T, params) == pytest.approx(numerical, abs=1e-8)


def test_example_4_2_atm_skew_is_rho_eta_over_two_sqrt_t_when_gamma_is_half():
    eta, rho, T = 0.9, -0.35, 1.0
    params = ssvi.SSVIParams(rho=rho, phi=ssvi.phi_power_law(eta, 0.5))
    for theta in (0.05, 0.3, 1.2):
        assert ssvi.ssvi_atm_volatility_skew(theta, T, params) == pytest.approx(
            rho * eta / (2 * np.sqrt(T))
        )


# ---------------------------------------------------------------------------
# Equation (4.3): the time-zero smile, w(k, theta_0) = phi_0 (rho k + |k|)/2.
# ---------------------------------------------------------------------------
def test_eq_4_3_zero_time_smile_is_v_shaped_and_vanishes_at_the_money():
    phi_0, rho = 0.8, -0.3
    k = np.linspace(-1.0, 1.0, 41)
    w0 = ssvi.ssvi_zero_time_smile(k, phi_0, rho)
    assert ssvi.ssvi_zero_time_smile(0.0, phi_0, rho) == 0.0
    assert (w0 >= 0).all()
    # V-shaped: linear either side of the money with slopes phi_0(rho+-1)/2
    assert w0[-1] == pytest.approx(0.5 * phi_0 * (rho + 1) * 1.0)
    assert w0[0] == pytest.approx(0.5 * phi_0 * (-rho + 1) * 1.0)


def test_phi_0_is_zero_for_both_worked_examples():
    # Example 4.1 is a stochastic-volatility-like case (phi_0 = 0); the
    # power law of Example 4.2 also has theta*phi(theta) -> 0 for gamma < 1.
    assert ssvi.ssvi_phi_0(ssvi.SSVIParams(rho=-0.3, phi=ssvi.phi_heston_like(0.8))) == 0.0
    assert ssvi.ssvi_phi_0(ssvi.SSVIParams(rho=-0.3, phi=ssvi.phi_power_law(0.9, 0.4))) == 0.0


# ---------------------------------------------------------------------------
# Remark 4.2: Theorem 4.2 restated in SVI-JW terms.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rho,eta,gamma,theta", [
    (-0.3, 0.8, 0.3, 0.5), (0.2, 0.5, 0.6, 1.0), (0.0, 6.0, 0.5, 3.0), (0.9, 4.0, 0.2, 2.0),
])
def test_remark_4_2_agrees_with_theorem_4_2(rho, eta, gamma, theta):
    params = ssvi.SSVIParams(rho=rho, phi=ssvi.phi_power_law(eta, gamma))
    grid = np.array([theta])

    def passes(check):
        try:
            check()
            return True
        except arb.ArbitrageError:
            return False

    assert passes(lambda: ssvi.check_ssvi_butterfly(grid, params)) == passes(
        lambda: ssvi.check_ssvi_butterfly_jw(grid, 1.0, params)
    )


# ---------------------------------------------------------------------------
# Remark 4.3: asymptotic behaviour of SSVI as |k| -> infinity.
# ---------------------------------------------------------------------------
def test_remark_4_3_asymptote_differs_from_exact_by_o_of_one():
    params = ssvi.SSVIParams(rho=-0.3, phi=ssvi.phi_power_law(0.8, 0.4))
    theta = 0.3
    residuals = []
    for k_mag in (50.0, 500.0, 5000.0, 50000.0):
        k = np.array([-k_mag, k_mag])
        residuals.append(
            ssvi.ssvi_total_variance(k, theta, params)
            - ssvi.ssvi_asymptotic_total_variance(k, theta, params)
        )
    residuals = np.array(residuals)
    # the O(1) remainder converges to a constant rather than growing with |k|
    assert np.all(np.abs(residuals) < 1.0)
    # successive changes shrink by ~10x for each 10x in |k| (an O(1/|k|) tail)
    steps = np.abs(np.diff(residuals, axis=0)).max(axis=1)
    assert np.all(np.diff(steps) < 0)
    assert steps[-1] < 1e-4


# ---------------------------------------------------------------------------
# Remark 4.4 / Equation (4.5) and Remark 4.5.
# ---------------------------------------------------------------------------
def test_eq_4_5_is_free_of_butterfly_arbitrage_for_all_theta_unlike_plain_power_law():
    rho, eta, gamma = -0.3, 1.4, 0.5
    assert eta * (1 + abs(rho)) <= 2  # the condition Remark 4.4 attaches to (4.5)
    assert ssvi.check_eq_4_5_eta_condition(eta, rho) is True

    theta_grid = np.geomspace(1e-3, 1e6, 60)
    bounded = ssvi.SSVIParams(rho=rho, phi=ssvi.phi_power_law_bounded(eta, gamma))
    assert ssvi.check_ssvi_butterfly(theta_grid, bounded) is True

    # Remark 4.4's point: the plain power law only works up to some maximum expiry
    plain = ssvi.SSVIParams(rho=rho, phi=ssvi.phi_power_law(eta, gamma))
    with pytest.raises(arb.ArbitrageError):
        ssvi.check_ssvi_butterfly(theta_grid, plain)


def test_eq_4_5_eta_condition_rejects_large_eta():
    with pytest.raises(arb.ArbitrageError, match=r"Equation \(4.5\)"):
        ssvi.check_eq_4_5_eta_condition(1.9, -0.3)


@pytest.mark.parametrize("rho", [-0.9, -0.3, 0.0, 0.5, 0.95])
def test_remark_4_5_lambda_bound_is_where_the_large_theta_limit_hits_four(rho):
    bound = ssvi.heston_like_lambda_lower_bound(rho)
    assert bound == pytest.approx((1 + abs(rho)) / 4)
    # at lam = bound the limit theta*phi(theta)*(1+|rho|) is exactly 4
    assert (1 + abs(rho)) / bound == pytest.approx(4.0)
    assert ssvi.check_heston_like_lambda(bound, rho) is True
    with pytest.raises(arb.ArbitrageError, match="Remark 4.5"):
        ssvi.check_heston_like_lambda(bound * 0.5, rho)


def test_remark_4_5_small_lambda_passes_on_a_short_grid_but_fails_asymptotically():
    rho, lam = -0.3, 0.05
    params = ssvi.SSVIParams(rho=rho, phi=ssvi.phi_heston_like(lam))
    assert ssvi.check_ssvi_butterfly(np.array([0.01, 0.1, 1.0]), params) is True
    with pytest.raises(arb.ArbitrageError):
        ssvi.check_ssvi_butterfly(np.array([1e9]), params)
    with pytest.raises(arb.ArbitrageError, match="Remark 4.5"):
        ssvi.check_heston_like_lambda(lam, rho)


# ---------------------------------------------------------------------------
# Examples 4.1 and 4.2 supply closed forms for d_theta(theta phi(theta)),
# which check_ssvi_calendar now uses in place of finite differences.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phi", [ssvi.phi_heston_like(0.8), ssvi.phi_power_law(0.9, 0.4)])
def test_shape_function_exact_derivatives_match_finite_differences(phi):
    theta = np.array([0.05, 0.3, 1.0, 3.0])
    h = 1e-7
    numerical = ((theta + h) * phi(theta + h) - (theta - h) * phi(theta - h)) / (2 * h)
    assert phi.d_theta_phi(theta) == pytest.approx(numerical, abs=1e-6)
    assert phi.d_theta_phi_over_phi(theta) == pytest.approx(phi.d_theta_phi(theta) / phi(theta))


def test_example_4_2_derivative_ratio_is_one_minus_gamma():
    for gamma in (0.1, 0.4, 0.75):
        phi = ssvi.phi_power_law(1.3, gamma)
        assert phi.d_theta_phi_over_phi(np.array([0.05, 1.0, 9.0])) == pytest.approx(1 - gamma)


def test_example_4_1_derivative_ratio_tends_to_one_as_theta_goes_to_zero():
    # "the map theta -> d_theta(theta phi(theta))/phi(theta) is strictly
    # decreasing on (0, inf) with limit as theta tends to zero equal to one"
    phi = ssvi.phi_heston_like(0.8)
    assert phi.d_theta_phi_over_phi(np.array([1e-10])) == pytest.approx(1.0, abs=1e-7)
    # Strict decrease is only numerically resolvable away from theta = 0,
    # where the map is flat at 1 and the true step falls below rounding.
    ratio = phi.d_theta_phi_over_phi(np.geomspace(1e-3, 50.0, 80))
    assert np.all(np.diff(ratio) < 0)
    assert ratio[0] < 1.0

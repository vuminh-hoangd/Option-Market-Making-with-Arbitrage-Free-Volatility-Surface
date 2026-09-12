"""
Tests for the classical Avellaneda-Stoikov (2008) benchmark adapter.

Reuses the same frozen Deribit fixture (`conftest.py`) and `params` as every
other test in this package, so a reader can compare its numbers directly
against `test_omm_sim.py`, `test_omm_pde.py`, etc.
"""
import numpy as np
import pytest

from option_market_making import bs_mm
from option_market_making.avellaneda_stoikov import optimal_half_spread
from option_market_making.benchmarks import (
    avellaneda_stoikov_spread_fn,
    calibrate_gamma,
    simulate_avellaneda_stoikov,
    symmetric_kappa,
)
from option_market_making.sim import objective, simulate

SHORT_T = 0.5 / 252
N_STEPS = 400


def test_symmetric_kappa_is_the_average(params):
    assert symmetric_kappa(params) == pytest.approx(0.5 * (params.kappa_b + params.kappa_a))


def test_spread_fn_rejects_non_positive_gamma_or_k(contract):
    with pytest.raises(ValueError, match="gamma"):
        avellaneda_stoikov_spread_fn(contract["K"], contract["tau"], SHORT_T,
                                     contract["sigma_imp"], gamma=0.0, k=0.025)
    with pytest.raises(ValueError, match="k must"):
        avellaneda_stoikov_spread_fn(contract["K"], contract["tau"], SHORT_T,
                                     contract["sigma_imp"], gamma=1e-5, k=-1.0)


def test_spread_fn_straddles_the_mark_at_zero_inventory(contract):
    """At q=0 there's no skew, so both offsets reduce to the same positive half-spread."""
    spread_fn = avellaneda_stoikov_spread_fn(
        contract["K"], contract["tau"], SHORT_T, contract["sigma_imp"], gamma=2e-5, k=0.0275,
    )
    spots = np.array([contract["S0"]])
    db, da = spread_fn(0.0, spots, np.array([0]))
    assert db[0] > 0.0
    assert db[0] == pytest.approx(da[0])


def test_spread_fn_skews_with_inventory_like_classical_AS(contract):
    """
    Long inventory should widen the bid distance and tighten the ask distance,
    same qualitative direction as Lucic-Tse's own psi_2 skew (§5), even though
    the two models are unrelated in derivation.
    """
    spread_fn = avellaneda_stoikov_spread_fn(
        contract["K"], contract["tau"], SHORT_T, contract["sigma_imp"], gamma=2e-5, k=0.0275,
    )
    spots = np.array([contract["S0"]] * 3)
    db, da = spread_fn(0.0, spots, np.array([-5, 0, 5]))
    assert db[0] < db[1] < db[2]
    assert da[0] > da[1] > da[2]


def test_spread_fn_uses_sigma_imp_for_the_diffusion_term(contract):
    """
    The Ito's-lemma derivation: AS's own sigma_O is |Delta_t|*S_t*sigma_imp --
    the unique choice making AS's assumed no-drift dO=sigma_O dB_t exactly
    true for O(t,S_t). Verified directly against the closed-form half-spread,
    computed independently here rather than re-deriving via the spread_fn.
    """
    gamma, k = 2e-5, 0.0275
    spot, K, tau = contract["S0"], contract["K"], contract["tau"]
    spread_fn = avellaneda_stoikov_spread_fn(K, tau, SHORT_T, contract["sigma_imp"], gamma, k)

    delta_t = bs_mm.delta(spot, K, tau, contract["sigma_imp"])
    sigma_O = abs(delta_t) * spot * contract["sigma_imp"]
    expected_half = optimal_half_spread(sigma_O, gamma, k, SHORT_T)

    db, da = spread_fn(0.0, np.array([spot]), np.array([0]))
    assert db[0] == pytest.approx(expected_half)
    assert da[0] == pytest.approx(expected_half)


def test_spread_fn_width_matches_the_closed_form_half_spread(contract):
    """The total width (db+da) has no inventory dependence, same as eq. 11's own invariant width (§5)."""
    gamma, k = 2e-5, 0.0275
    spread_fn = avellaneda_stoikov_spread_fn(contract["K"], contract["tau"], SHORT_T,
                                             contract["sigma_imp"], gamma, k)
    spot = contract["S0"]
    delta_t = bs_mm.delta(spot, contract["K"], contract["tau"], contract["sigma_imp"])
    sigma_O = abs(delta_t) * spot * contract["sigma_imp"]
    expected_width = 2.0 * optimal_half_spread(sigma_O, gamma, k, SHORT_T)

    for q in (-5, 0, 5):
        db, da = spread_fn(0.0, np.array([spot]), np.array([q]))
        assert (db[0] + da[0]) == pytest.approx(expected_width, rel=1e-9)


def test_calibration_hits_the_target_mean_abs_q(surface, contract, params, horizon):
    """
    The whole point of calibrate_gamma: verify the returned gamma actually
    reproduces the target within a sane tolerance, by re-simulating with it.
    """
    K, tau = contract["K"], contract["tau"]
    sim_kwargs = dict(S0=contract["S0"], n_steps=N_STEPS, n_paths=3000, seed=7)

    baseline = simulate(surface, K, tau, horizon, params, **sim_kwargs)
    target = float(np.abs(baseline.q).mean())

    gamma = calibrate_gamma(surface, K, tau, horizon, params, target,
                            log10_gamma_bracket=(-6.0, -4.0), **sim_kwargs)
    assert 1e-6 < gamma < 1e-4

    achieved = simulate_avellaneda_stoikov(surface, K, tau, horizon, params, gamma, **sim_kwargs)
    assert np.abs(achieved.q).mean() == pytest.approx(target, abs=0.05)


def test_calibration_bracket_too_wide_fails_loudly_not_silently(surface, contract, params, horizon):
    """
    Documented in calibrate_gamma's own docstring: too wide a bracket can push
    gamma high enough that AS's own skew term explodes fill intensities past
    what the Poisson sampler in `sim.simulate` can draw. This should surface
    as an error, not a silently wrong answer.
    """
    with pytest.raises((ValueError, RuntimeError)):
        calibrate_gamma(
            surface, contract["K"], contract["tau"], horizon, params,
            target_mean_abs_q=1.0, log10_gamma_bracket=(-7.0, 2.0),
            S0=contract["S0"], n_steps=N_STEPS, n_paths=500, seed=7,
        )


def test_realised_sigma_does_not_affect_AS_own_quote(surface, contract, params, horizon):
    """
    realised_sigma only ever drives the simulated spot path -- AS's own
    sigma_O always uses sigma_imp regardless, so two runs differing only in
    realised_sigma must produce different paths but identical quotes at a
    shared (t, spot, q).
    """
    K, tau = contract["K"], contract["tau"]
    sim_kwargs = dict(S0=contract["S0"], n_steps=N_STEPS, n_paths=2000, seed=11)
    gamma = 2e-5

    default_real_vol = simulate_avellaneda_stoikov(surface, K, tau, horizon, params, gamma, **sim_kwargs)
    other_real_vol = simulate_avellaneda_stoikov(surface, K, tau, horizon, params, gamma,
                                                 realised_sigma=params.sigma * 2, **sim_kwargs)

    assert not np.array_equal(default_real_vol.S, other_real_vol.S), \
        "realised_sigma should have changed the simulated path"
    assert np.array_equal(default_real_vol.delta_b[:, 0], other_real_vol.delta_b[:, 0]), \
        "AS's own t=0 quote (same starting spot, same q=0) must not depend on realised_sigma"


def test_optimal_beats_classical_AS_on_the_objective_at_matched_risk(surface, contract, params, horizon):
    """
    The main comparison result: at a risk-matched gamma, the exact eq. 11
    rule should still win on the objective it was built to maximise, since it
    additionally sees the vol-arb edge and the asymmetric order-flow
    correction that classical AS has no way to represent.
    """
    K, tau = contract["K"], contract["tau"]
    n_paths = 4000
    sim_kwargs = dict(S0=contract["S0"], n_steps=N_STEPS, n_paths=n_paths, seed=42)
    cal_kwargs = dict(S0=contract["S0"], n_steps=N_STEPS, n_paths=3000, seed=7)

    optimal = simulate(surface, K, tau, horizon, params, **sim_kwargs)
    target = float(np.abs(optimal.q).mean())
    gamma = calibrate_gamma(surface, K, tau, horizon, params, target,
                            log10_gamma_bracket=(-6.0, -4.0), **cal_kwargs)
    classical_as = simulate_avellaneda_stoikov(surface, K, tau, horizon, params, gamma, **sim_kwargs)

    assert np.array_equal(optimal.S, classical_as.S), "paired comparison requires identical spot paths"

    gap = objective(optimal, params) - objective(classical_as, params)
    se = gap.std() / np.sqrt(n_paths)
    assert gap.mean() > 3.0 * se, "optimal should beat calibrated classical AS by more than noise"

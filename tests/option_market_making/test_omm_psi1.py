import numpy as np
import pytest

from option_market_making.edge import phi_edge
from option_market_making.psi1 import flow_corrections, psi1


def test_psi1_vanishes_at_the_end_of_the_horizon(params, horizon, contract):
    value = psi1(
        horizon, contract["S0"], contract["K"], contract["tau"], horizon,
        contract["sigma_imp"], params,
    )
    assert value == 0.0


def test_flow_corrections_vanish_for_a_symmetric_book(symmetric_params, horizon):
    for t in np.linspace(0.0, horizon, 11):
        assert flow_corrections(float(t), horizon, symmetric_params) == 0.0


def test_psi1_is_pure_edge_for_a_symmetric_book(symmetric_params, horizon, contract):
    args = (0.0, contract["S0"], contract["K"], contract["tau"], horizon, contract["sigma_imp"])
    assert psi1(*args, symmetric_params) == pytest.approx(phi_edge(*args, symmetric_params))


def test_psi1_decomposes_into_edge_plus_corrections(params, horizon, contract):
    """The asymmetric corrections must actually be switched on for this book."""
    args = (0.0, contract["S0"], contract["K"], contract["tau"], horizon, contract["sigma_imp"])
    edge = phi_edge(*args, params)
    corrections = flow_corrections(0.0, horizon, params)

    assert corrections != 0.0
    assert psi1(*args, params) == pytest.approx(edge + corrections)


def test_flow_corrections_flip_sign_with_the_order_flow_imbalance(params, horizon):
    """
    psi2 < 0, so a bid-heavy book (lambda0_b > lambda0_a) makes the linear
    correction negative, and mirroring the imbalance mirrors the correction.
    """
    from dataclasses import replace

    bid_heavy = replace(params, lambda0_b=30000.0, lambda0_a=25000.0,
                        kappa_b=0.025, kappa_a=0.025)
    ask_heavy = replace(params, lambda0_b=25000.0, lambda0_a=30000.0,
                        kappa_b=0.025, kappa_a=0.025)

    assert flow_corrections(0.0, horizon, bid_heavy) < 0.0
    assert flow_corrections(0.0, horizon, ask_heavy) > 0.0
    assert flow_corrections(0.0, horizon, bid_heavy) == pytest.approx(
        -flow_corrections(0.0, horizon, ask_heavy)
    )


def test_correction_integrals_match_a_direct_quadrature(params, horizon):
    """
    Reproduce eq. 9's two correction integrals from `riccati` alone, on an
    independent Simpson grid.
    """
    from scipy.integrate import simpson

    from option_market_making import riccati

    t, T = 0.0, horizon
    u_grid = np.linspace(t, T, 401)
    kernel = np.array([riccati.discount_kernel(t, float(u), T, params) for u in u_grid])
    psi2_vals = np.array([riccati.psi2(float(u), T, params) for u in u_grid])

    two_over_e = 2.0 * np.exp(-1.0)
    expected = (
        two_over_e * (params.lambda0_b - params.lambda0_a) * simpson(kernel * psi2_vals, x=u_grid)
        + two_over_e
        * (params.lambda0_b * params.kappa_b - params.lambda0_a * params.kappa_a)
        * simpson(kernel * psi2_vals ** 2, x=u_grid)
    )
    assert flow_corrections(t, T, params) == pytest.approx(expected, rel=1e-8)

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import simpson

from option_market_making import bs_mm, riccati
from option_market_making.edge import phi_edge


def test_edge_vanishes_at_the_end_of_the_horizon(params, horizon, contract):
    value = phi_edge(
        horizon, contract["S0"], contract["K"], contract["tau"], horizon,
        contract["sigma_imp"], params,
    )
    assert value == 0.0


def test_edge_sign_tracks_the_variance_spread(params, horizon, contract):
    """
    Long vol (sigma > sigma_imp) is worth something; short vol against a market
    that is right is worth something negative; and a market maker who agrees
    with the market has no edge at all.
    """
    args = (0.0, contract["S0"], contract["K"], contract["tau"], horizon, contract["sigma_imp"])

    long_vol = phi_edge(*args, replace(params, sigma=contract["sigma_imp"] + 0.10))
    short_vol = phi_edge(*args, replace(params, sigma=contract["sigma_imp"] - 0.10))
    no_view = phi_edge(*args, replace(params, sigma=contract["sigma_imp"]))

    assert long_vol > 0.0
    assert short_vol < 0.0
    assert no_view == 0.0


def test_edge_grows_with_the_variance_spread(params, horizon, contract):
    args = (0.0, contract["S0"], contract["K"], contract["tau"], horizon, contract["sigma_imp"])
    edges = [
        phi_edge(*args, replace(params, sigma=contract["sigma_imp"] + bump))
        for bump in (0.01, 0.03, 0.05, 0.10)
    ]
    assert np.all(np.diff(edges) > 0.0)


def test_edge_matches_a_brute_force_evaluation_of_the_expectation(params, horizon, contract):
    """
    Cross-check eq. 10's closed form against the definition it collapses from:

        phi_edge = int_t^T D(t,u) (sigma^2 - sigma_imp^2)/2
                                 * E[ Gamma^$(u, S_u) | S_t = s ] du

    The inner expectation is taken here by Gauss-Hermite quadrature over the
    lognormal S_u and the outer integral by Simpson -- neither of which knows
    anything about the Gaussian collapse `edge.py` relies on, so this tests the
    analytic step rather than just re-running the same quadrature.
    """
    t, T = 0.0, horizon
    s, K, tau, vol = contract["S0"], contract["K"], contract["tau"], contract["sigma_imp"]

    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / weights.sum()  # E[g(Z)] for Z ~ N(0,1)

    def expected_dollar_gamma(u: float) -> float:
        dt = u - t
        if dt <= 0.0:
            return bs_mm.dollar_gamma(s, K, tau - u, vol)
        spots = s * np.exp(-0.5 * params.sigma ** 2 * dt + params.sigma * np.sqrt(dt) * nodes)
        gammas = np.array([bs_mm.dollar_gamma(float(x), K, tau - u, vol) for x in spots])
        return float(weights @ gammas)

    u_grid = np.linspace(t, T, 201)
    integrand = np.array([
        riccati.discount_kernel(t, float(u), T, params) * expected_dollar_gamma(float(u))
        for u in u_grid
    ])
    brute_force = (params.sigma ** 2 - vol ** 2) / 2.0 * simpson(integrand, x=u_grid)

    assert phi_edge(t, s, K, tau, T, vol, params) == pytest.approx(brute_force, rel=1e-6)


def test_edge_peaks_near_the_money(params, horizon, contract):
    """Gamma is largest at the money, so the edge is too."""
    K, tau, vol = contract["K"], contract["tau"], contract["sigma_imp"]
    spots = K * np.array([0.75, 0.9, 1.0, 1.1, 1.35])
    edges = [phi_edge(0.0, float(s), K, tau, horizon, vol, params) for s in spots]
    assert edges[2] == max(edges)
    assert edges[0] < edges[1] < edges[2] > edges[3] > edges[4]


def test_horizon_must_end_before_expiry(params, contract):
    with pytest.raises(ValueError, match="strictly before expiry"):
        phi_edge(
            0.0, contract["S0"], contract["K"], contract["tau"], contract["tau"],
            contract["sigma_imp"], params,
        )


def test_batch_edge_agrees_with_the_adaptive_reference(params, horizon, contract):
    """
    `phi_edge_batch` swaps the adaptive quadrature for a fixed Gauss-Legendre
    grid so `sim` can share nodes across paths; it must not cost accuracy.
    """
    from option_market_making.edge import phi_edge_batch

    K, tau, vol = contract["K"], contract["tau"], contract["sigma_imp"]
    spots = contract["S0"] * np.linspace(0.7, 1.4, 21)

    for t in (0.0, 0.37 * horizon, 0.99 * horizon):
        expected = [phi_edge(t, float(s), K, tau, horizon, vol, params) for s in spots]
        assert phi_edge_batch(t, spots, K, tau, horizon, vol, params) == pytest.approx(
            expected, rel=1e-10
        )


def test_batch_edge_is_zero_at_the_horizon_and_with_no_view(params, horizon, contract):
    from dataclasses import replace as dc_replace

    from option_market_making.edge import phi_edge_batch

    K, tau, vol = contract["K"], contract["tau"], contract["sigma_imp"]
    spots = contract["S0"] * np.array([0.9, 1.0, 1.1])

    assert np.all(phi_edge_batch(horizon, spots, K, tau, horizon, vol, params) == 0.0)
    no_view = dc_replace(params, sigma=vol)
    assert np.all(phi_edge_batch(0.0, spots, K, tau, horizon, vol, no_view) == 0.0)

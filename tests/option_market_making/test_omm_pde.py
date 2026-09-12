"""
The validation gate on the second-order approximation.

`pde.py` solves the EXACT HJB equation (eq. 6); `quoting.py` solves the paper's
second-order expansion of it (eq. 7). These tests first license the FD solution
as ground truth (grid convergence, plus two analytic checks on the diffusion
operator in isolation), then measure the approximation against it.
"""
from dataclasses import replace

import numpy as np
import pytest

from option_market_making.pde import _diffusion_solver, solve_hjb_pde
from option_market_making.quoting import optimal_spreads
from option_market_making.sim import objective, simulate

SHORT_T = 0.5 / 252
LONG_T = 10.0 / 252
GRID = dict(n_t=800, n_s=201, q_max=15, log_s_halfwidth=0.6)


@pytest.fixture(scope="module")
def solve(contract):
    """Cached PDE solves — each one is a second, and the matrix is reused across tests."""
    cache: dict = {}

    def _solve(params, T=SHORT_T, **overrides):
        key = (params, T, tuple(sorted(overrides.items())))
        if key not in cache:
            cache[key] = solve_hjb_pde(
                contract["K"], contract["tau"], T, params, contract["sigma_imp"],
                s_center=contract["S0"], **{**GRID, **overrides},
            )
        return cache[key]

    return _solve


# --- Step 1: the tridiagonal solve, in isolation, against analytic answers ---

def test_diffusion_operator_preserves_a_linear_payoff(contract):
    """
    h(T,s) = s has zero curvature and S is a driftless martingale, so the exact
    solution is h(t,s) = s. Any spurious numerical diffusion shows up here.
    """
    s0, sigma, n_t = contract["S0"], 0.4575, 400
    x = np.log(s0) + np.linspace(-0.6, 0.6, 201)
    s = np.exp(x)
    step = _diffusion_solver(x, sigma, SHORT_T / n_t)

    h = s[:, None].copy()
    for _ in range(n_t):
        h = step(h)
    interior = slice(20, -20)
    assert np.abs(h[interior, 0] / s[interior] - 1).max() < 1e-8


def test_diffusion_operator_matches_the_analytic_second_moment(contract):
    """E[S_T^2 | S_0=s] = s^2 exp(sigma^2 T) -- a real test of the curvature term."""
    s0, sigma, n_t = contract["S0"], 0.4575, 400
    x = np.log(s0) + np.linspace(-0.6, 0.6, 201)
    s = np.exp(x)
    step = _diffusion_solver(x, sigma, SHORT_T / n_t)

    h = (s[:, None] ** 2).copy()
    for _ in range(n_t):
        h = step(h)
    exact = s ** 2 * np.exp(sigma ** 2 * SHORT_T)
    interior = slice(20, -20)
    assert np.abs(h[interior, 0] / exact[interior] - 1).max() < 1e-6


# --- Step 2: grid convergence. This is what licenses the FD as ground truth. ---

def _probe_spreads(sol, S0, T):
    return np.array([
        sol.spreads_at(t, s, q)
        for t in (0.0, T / 2)
        for s in (0.9 * S0, S0, 1.1 * S0)
        for q in (-5, 0, 5)
    ])


@pytest.mark.parametrize("refinement", [
    pytest.param(dict(n_t=1600), id="halve-dt"),
    pytest.param(dict(n_s=401), id="halve-ds"),
    pytest.param(dict(q_max=20), id="widen-q_max"),
    pytest.param(dict(log_s_halfwidth=1.0), id="widen-s-grid"),
])
def test_grid_convergence(solve, params, contract, refinement):
    """
    Each grid parameter refined independently must move the spreads by well
    under 0.5%. Without this the FD solution is not ground truth for anything.
    """
    S0 = contract["S0"]
    base = _probe_spreads(solve(params), S0, SHORT_T)
    refined = _probe_spreads(solve(params, **refinement), S0, SHORT_T)
    assert np.abs(refined / base - 1).max() < 5e-3


def test_pde_value_function_agrees_with_the_simulated_objective(solve, surface, params, contract):
    """
    Independent cross-check of the whole PDE against the whole Monte Carlo:
    h(0, S0, 0) is the optimal expected value of eq. 2 starting flat, and the
    closed-form rule -- near-optimal -- should achieve statistically the same
    thing in `sim`. Two completely separate code paths.
    """
    n_paths = 4000
    h_fd = solve(params).value_at(0.0, contract["S0"])[GRID["q_max"]]
    run = simulate(surface, contract["K"], contract["tau"], SHORT_T, params,
                   S0=contract["S0"], n_steps=400, n_paths=n_paths, seed=1)
    achieved = objective(run, params)
    standard_error = achieved.std() / np.sqrt(n_paths)
    assert abs(h_fd - achieved.mean()) < 4.0 * standard_error


# --- Step 3: the approximation, measured against the licensed ground truth ---

def _max_relative_error(sol, params, contract, T):
    S0, K, tau, iv = contract["S0"], contract["K"], contract["tau"], contract["sigma_imp"]
    worst = 0.0
    for t in (0.0, T / 4, T / 2, 3 * T / 4):
        for s in (0.85 * S0, 0.95 * S0, S0, 1.05 * S0, 1.15 * S0):
            for q in range(-5, 6):
                fd = np.array(sol.spreads_at(t, s, q))
                cf = np.array(optimal_spreads(t, s, q, K, tau, T, iv, params))
                worst = max(worst, np.abs(fd / cf - 1).max())
    return worst


def test_short_horizon_spread_accuracy(solve, params, contract):
    """
    Over the paper's own short horizon the closed-form spreads must stay within
    a few percent of exact across the whole (t, s, q) test matrix.
    """
    error = _max_relative_error(solve(params), params, contract, SHORT_T)
    assert error < 0.03, f"short-horizon spread error {error:.2%} exceeds 3%"


def test_long_horizon_accuracy_is_reported_not_hidden(solve, params, contract, capsys):
    """
    Twenty times the horizon. Asserted only loosely on purpose -- the point is to
    print the number so degradation is visible rather than swallowed by a
    tolerance wide enough to pass either way.
    """
    sol = solve(params, T=LONG_T, n_t=8000)
    error = _max_relative_error(sol, params, contract, LONG_T)
    with capsys.disabled():
        print(f"\n  [long horizon T=10/252] max relative spread error: {error:.3%}")
    assert error < 0.10


def test_accuracy_tracks_beta_not_the_knife_edge(solve, params, contract, capsys):
    """
    THE KNIFE-EDGE QUESTION.

    beta = 4357 sits exactly on sqrt(Upsilon*beta) = alpha, i.e. zeta_minus = 0,
    which collapses psi2 to the constant -alpha. The worry was that this is an
    unusually easy case that flatters the approximation.

    It is not. Accuracy degrades MONOTONICALLY IN BETA and the knife-edge is
    simply the middle of that trend: beta/2 is MORE favorable to the
    approximation than the knife-edge, and beta*2 is LESS. The driver is the
    magnitude of beta -- which sets |psi2| and hence how far kappa*(delta - 1/kappa)
    strays from zero, that being the argument the exponential is expanded about --
    not proximity to zeta_minus = 0.
    """
    errors = {}
    for label, beta in (("beta/2", params.beta / 2), ("knife-edge", params.beta),
                        ("beta*2", params.beta * 2)):
        p = replace(params, beta=beta)
        errors[label] = _max_relative_error(solve(p), p, contract, SHORT_T)

    with capsys.disabled():
        print("\n  [knife-edge] max relative spread error: "
              + ", ".join(f"{k} {v:.3%}" for k, v in errors.items()))

    assert errors["beta/2"] < errors["knife-edge"] < errors["beta*2"], (
        f"expected error monotone in beta, got {errors}"
    )


def test_the_approximation_costs_almost_nothing_in_objective_terms(
    solve, surface, params, contract, capsys
):
    """
    The result that actually matters. Spread errors of a few percent are worth
    essentially nothing in eq. 2, because the objective is flat at its optimum --
    a first-order error in the control costs only second order in value.

    Checked OFF the knife-edge, at the least favorable beta, which is where the
    approximation has the most to lose.
    """
    n_paths = 3000
    p = replace(params, beta=params.beta * 2)
    sol = solve(p)
    shared = dict(S0=contract["S0"], n_steps=400, n_paths=n_paths, seed=1)

    exact = simulate(surface, contract["K"], contract["tau"], SHORT_T, p,
                     spread_fn=sol.spreads_batch, **shared)
    approx = simulate(surface, contract["K"], contract["tau"], SHORT_T, p, **shared)

    assert np.abs(exact.q).max() < GRID["q_max"], "simulation wandered outside the PDE grid"

    gap = objective(exact, p) - objective(approx, p)
    ci = 1.96 * gap.std() / np.sqrt(n_paths)
    with capsys.disabled():
        print(f"\n  [objective cost] exact - approx = {gap.mean():+.2f} +/- {ci:.2f} "
              f"on an objective of ~{objective(exact, p).mean():,.0f}")

    # Indistinguishable from zero, and in any case far below 1% of the objective.
    assert abs(gap.mean()) < max(ci, 0.01 * abs(objective(exact, p).mean()))

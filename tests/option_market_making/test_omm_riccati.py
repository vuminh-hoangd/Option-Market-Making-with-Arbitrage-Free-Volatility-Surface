import numpy as np
import pytest

from option_market_making import riccati


def test_psi2_hits_its_terminal_condition(params, horizon):
    assert riccati.psi2(horizon, horizon, params) == pytest.approx(-params.alpha, rel=1e-12)


def test_psi2_is_finite_negative_and_bounded_over_the_horizon(params, horizon):
    ups = riccati.upsilon(params.lambda0_b, params.kappa_b, params.lambda0_a, params.kappa_a)
    bound = max(params.alpha, np.sqrt(ups * params.beta))

    values = [riccati.psi2(float(t), horizon, params) for t in np.linspace(0.0, horizon, 101)]
    assert all(np.isfinite(v) for v in values)
    assert all(v < 0.0 for v in values)
    assert all(-bound - 1e-9 <= v for v in values)


def test_psi2_solves_its_riccati_ode(params, horizon):
    """
    psi2' + 2 e^{-1} (lambda0_b kappa_b + lambda0_a kappa_a) psi2^2 - beta = 0.

    Checked by central differences -- an independent test of the closed form,
    since nothing in `riccati` ever integrates the ODE.
    """
    coefficient = 2.0 * np.exp(-1.0) * (
        params.lambda0_b * params.kappa_b + params.lambda0_a * params.kappa_a
    )
    h = horizon * 1e-5
    for t in np.linspace(0.2 * horizon, 0.8 * horizon, 7):
        derivative = (
            riccati.psi2(float(t + h), horizon, params)
            - riccati.psi2(float(t - h), horizon, params)
        ) / (2.0 * h)
        residual = derivative + coefficient * riccati.psi2(float(t), horizon, params) ** 2 - params.beta
        assert residual == pytest.approx(0.0, abs=1e-4 * params.beta)


def test_discount_kernel_is_one_at_u_equals_t(params, horizon):
    for t in np.linspace(0.0, horizon, 11):
        assert riccati.discount_kernel(float(t), float(t), horizon, params) == pytest.approx(1.0)


def test_discount_kernel_decays_into_the_future(params, horizon):
    t = 0.1 * horizon
    grid = np.linspace(t, horizon, 25)
    values = [riccati.discount_kernel(t, float(u), horizon, params) for u in grid]
    assert all(0.0 < v <= 1.0 + 1e-12 for v in values)
    assert np.all(np.diff(values) < 0.0)


def test_upsilon_and_eta_reject_degenerate_inputs():
    with pytest.raises(ValueError):
        riccati.upsilon(0.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        riccati.upsilon(1.0, -1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        riccati.eta(0.0, 1.0)


def test_queries_past_the_horizon_are_rejected(params, horizon):
    with pytest.raises(ValueError):
        riccati.psi2(horizon * 1.01, horizon, params)
    with pytest.raises(ValueError):
        riccati.discount_kernel(0.0, horizon * 1.01, horizon, params)


def test_closed_form_survives_a_stiff_horizon(contract):
    """
    A large beta drives eta(T-t) into the tens, where the unscaled ratio of
    exponentials overflows. The rescaled form must stay finite and still land
    on -alpha at T.
    """
    from option_market_making.quoting import MMParams

    stiff = MMParams(
        alpha=2.0, beta=1e9, lambda0_b=25200.0, lambda0_a=25200.0,
        kappa_b=0.025, kappa_a=0.025, sigma=contract["sigma_imp"],
    )
    T = 0.5 / 252.0
    values = [riccati.psi2(float(t), T, stiff) for t in np.linspace(0.0, T, 51)]
    assert all(np.isfinite(v) for v in values)
    assert values[-1] == pytest.approx(-stiff.alpha, rel=1e-9)

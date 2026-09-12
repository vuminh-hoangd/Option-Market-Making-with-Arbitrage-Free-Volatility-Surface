import numpy as np
import pytest

import data
import svi
from svi import SVIParams, raw_svi, fit_svi_slice


GOOD_PARAMS = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)


def test_raw_svi_matches_definition():
    k = np.array([-0.5, 0.0, 0.5])
    expected = GOOD_PARAMS.a + GOOD_PARAMS.b * (
        GOOD_PARAMS.rho * (k - GOOD_PARAMS.m)
        + np.sqrt((k - GOOD_PARAMS.m) ** 2 + GOOD_PARAMS.sigma ** 2)
    )
    got = raw_svi(k, GOOD_PARAMS.a, GOOD_PARAMS.b, GOOD_PARAMS.rho, GOOD_PARAMS.m, GOOD_PARAMS.sigma)
    assert got == pytest.approx(expected)


def test_fit_recovers_known_synthetic_params():
    T = 0.25
    k = np.linspace(-1.2, 1.2, 25)
    w_true = raw_svi(k, *GOOD_PARAMS.as_array())
    mid_iv = np.sqrt(w_true / T)
    weight = np.ones_like(k)

    result = fit_svi_slice(k, mid_iv, weight, T)

    recovered = result.params.as_array()
    assert recovered == pytest.approx(GOOD_PARAMS.as_array(), abs=1e-3)
    assert result.weighted_rmse < 1e-6


def test_fit_recovers_known_params_with_noise():
    rng = np.random.default_rng(1)
    T = 0.5
    k = np.linspace(-1.5, 1.5, 40)
    w_true = raw_svi(k, *GOOD_PARAMS.as_array())
    noisy_iv = np.sqrt(w_true / T) + rng.normal(0, 1e-4, size=k.shape)
    weight = np.ones_like(k)

    result = fit_svi_slice(k, noisy_iv, weight, T)

    # fit residuals should be small relative to the noise injected
    assert result.weighted_rmse < 1e-4
    assert result.params.as_array() == pytest.approx(GOOD_PARAMS.as_array(), abs=0.05)


def test_fit_raises_on_too_few_points():
    k = np.array([-0.1, 0.0, 0.1])
    with pytest.raises(ValueError):
        fit_svi_slice(k, np.array([0.5, 0.5, 0.5]), np.ones(3), 0.25)


def test_fit_on_live_deribit_data_has_small_residuals():
    try:
        raw = data.fetch_book_summary("BTC")
    except Exception as exc:
        pytest.skip(f"Deribit API unreachable: {exc}")

    chain = data.build_chain(raw)
    if chain.empty:
        pytest.skip("no live liquid quotes returned")

    biggest_expiry = chain["expiry"].value_counts().idxmax()
    slice_ = chain[chain["expiry"] == biggest_expiry]
    if len(slice_) < 5:
        pytest.skip("largest live expiry doesn't have enough strikes to fit")

    T = float(slice_["T"].iloc[0])
    F = float(slice_["forward"].iloc[0])
    k = np.log(slice_["strike"].to_numpy() / F)

    result = fit_svi_slice(k, slice_["mid_iv"].to_numpy(), slice_["weight"].to_numpy(), T)

    # Judge fit quality the same way the optimizer does: weighted RMSE in
    # total-variance space. An unweighted vol-space RMSE would get dragged
    # around by deep-OTM strikes where the dollar bid-ask is a couple of
    # minimum ticks wide -- their mid_iv is noise, and the weighting
    # (weight ~ 1/width^2) is exactly what's supposed to discount them.
    assert result.weighted_rmse < 1e-3

    # Sanity check on the liquid core of the smile (the interior points a
    # market maker would actually trade off of), in interpretable vol terms.
    core = np.abs(k) < 0.5
    fitted_iv = result.params.implied_vol(k[core], T)
    core_rmse = float(np.sqrt(np.mean((fitted_iv - slice_["mid_iv"].to_numpy()[core]) ** 2)))
    assert core_rmse < 0.02


# ---------------------------------------------------------------------------
# Section 3.2 / Lemma 3.1: the natural SVI parameterization (3.2) and the
# raw <-> natural mappings (3.3) and (3.4).
# ---------------------------------------------------------------------------
import svi as svi_module
from svi import NaturalSVIParams


RAW_CASES = [
    SVIParams(a=-0.0410, b=0.1331, m=0.3586, rho=0.3060, sigma=0.4153),
    SVIParams(a=0.04, b=0.15, rho=-0.35, m=0.05, sigma=0.25),
    SVIParams(a=0.02, b=0.30, rho=0.45, m=-0.40, sigma=0.60),
    SVIParams(a=0.01, b=0.08, rho=-0.80, m=0.90, sigma=0.15),
]


@pytest.mark.parametrize("params", RAW_CASES)
def test_natural_svi_reproduces_the_same_total_variance_as_raw(params):
    # (3.2) and (3.1) describe the same curve once mapped by Lemma 3.1
    natural = svi_module.raw_to_natural(params)
    k = np.linspace(-2.0, 2.0, 81)
    assert natural.total_variance(k) == pytest.approx(params.total_variance(k), abs=1e-12)


@pytest.mark.parametrize("params", RAW_CASES)
def test_lemma_3_1_raw_to_natural_round_trips(params):
    back = svi_module.natural_to_raw(svi_module.raw_to_natural(params))
    assert back.as_array() == pytest.approx(params.as_array(), abs=1e-12)


@pytest.mark.parametrize("natural", [
    NaturalSVIParams(delta=0.01, mu=0.1, rho=-0.4, omega=0.3, zeta=1.2),
    NaturalSVIParams(delta=-0.05, mu=-0.7, rho=0.6, omega=1.1, zeta=0.4),
    NaturalSVIParams(delta=0.2, mu=0.0, rho=0.0, omega=0.05, zeta=3.0),
])
def test_lemma_3_1_natural_to_raw_round_trips(natural):
    back = svi_module.raw_to_natural(svi_module.natural_to_raw(natural))
    assert back.as_array() == pytest.approx(natural.as_array(), abs=1e-12)


def test_ssvi_slice_is_the_natural_parameterization_with_delta_and_mu_zero():
    # Section 4: "SSVI corresponds to the natural SVI volatility surface
    # parameterization (3.2) with chi_N = {0, 0, rho, theta_t, phi(theta_t)}"
    import ssvi
    params = ssvi.SSVIParams(rho=-0.35, phi=ssvi.phi_power_law(0.9, 0.4))
    for theta in (0.05, 0.5, 2.0):
        via_lemma_3_1 = svi_module.natural_to_raw(NaturalSVIParams(
            delta=0.0, mu=0.0, rho=params.rho, omega=theta, zeta=float(params.phi(theta)),
        ))
        assert via_lemma_3_1.as_array() == pytest.approx(
            ssvi.ssvi_slice_to_raw_svi(theta, params).as_array(), abs=1e-12
        )

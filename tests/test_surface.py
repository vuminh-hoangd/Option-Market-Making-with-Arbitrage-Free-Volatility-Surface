import numpy as np
import pandas as pd
import pytest

import arbitrage as arb
import data
import surface
from svi import SVIParams, raw_svi


def _synthetic_chain(forward_by_T, strikes, params_by_T):
    """
    Build a `data.build_chain`-shaped table directly from known SVI
    parameters per maturity, so round-trip / interpolation checks have a
    ground truth to compare against.
    """
    rows = []
    for T, forward in forward_by_T.items():
        params = params_by_T[T]
        k = np.log(np.asarray(strikes) / forward)
        w = raw_svi(k, *params.as_array())
        mid_iv = np.sqrt(w / T)
        for K, iv in zip(strikes, mid_iv):
            rows.append({
                "strike": float(K), "expiry": pd.Timestamp(0) + pd.Timedelta(days=T * 365),
                "T": T, "forward": forward, "mid_iv": iv, "weight": 1.0,
            })
    return pd.DataFrame(rows)


SHORT = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
LONG = SVIParams(a=0.05, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
STRIKES = np.linspace(60, 140, 15)


@pytest.fixture
def two_slice_chain():
    return _synthetic_chain(
        forward_by_T={0.1: 100.0, 0.5: 105.0},
        strikes=STRIKES,
        params_by_T={0.1: SHORT, 0.5: LONG},
    )


def test_fit_surface_on_grid_matches_fitted_slice(two_slice_chain):
    surf = surface.fit_surface(two_slice_chain)

    for T_i, F_i, params_i in zip(surf.maturities, surf.forwards, surf.params):
        K = F_i * np.exp(np.linspace(-0.3, 0.3, 7))
        expected = params_i.implied_vol(np.log(K / F_i), T_i)
        got = surf.implied_vol(K, T_i)
        assert got == pytest.approx(expected, abs=1e-8)


def test_implied_vol_interpolation_matches_lemma_5_1(two_slice_chain):
    surf = surface.fit_surface(two_slice_chain)

    T1, T2 = surf.maturities
    K = 100.0
    T_mid = 0.5 * (T1 + T2)
    frac = (T_mid - T1) / (T2 - T1)

    F1, F2 = surf.forwards
    F_mid = np.exp(np.log(F1) + (np.log(F2) - np.log(F1)) * frac)
    k_shared = np.log(K / F_mid)

    theta1 = surf.params[0].total_variance(0.0)
    theta2 = surf.params[1].total_variance(0.0)
    theta_mid = theta1 + (theta2 - theta1) * frac
    alpha_T = (np.sqrt(theta2) - np.sqrt(theta_mid)) / (np.sqrt(theta2) - np.sqrt(theta1))

    w1_k = surf.params[0].total_variance(k_shared)
    w2_k = surf.params[1].total_variance(k_shared)
    price1 = surface._normalized_price(k_shared, w1_k)
    price2 = surface._normalized_price(k_shared, w2_k)
    price_blend = alpha_T * price1 + (1 - alpha_T) * price2
    expected_w = surface._invert_normalized_price_for_variance(
        k_shared, price_blend, (min(w1_k, w2_k) * 0.5, max(w1_k, w2_k) * 2.0)
    )
    expected_iv = np.sqrt(expected_w / T_mid)

    got = surf.implied_vol(K, T_mid)
    assert got == pytest.approx(expected_iv, abs=1e-8)

    got_w = got ** 2 * T_mid
    assert min(w1_k, w2_k) <= got_w <= max(w1_k, w2_k)


def test_implied_vol_extrapolates_flat_outside_range(two_slice_chain):
    surf = surface.fit_surface(two_slice_chain)
    T1, T2 = surf.maturities
    K = 100.0

    iv_before = surf.implied_vol(K, T1 * 0.1)
    iv_at_T1 = surf.implied_vol(K, T1)
    assert iv_before == pytest.approx(iv_at_T1, abs=1e-10)

    iv_after = surf.implied_vol(K, T2 * 3.0)
    iv_at_T2 = surf.implied_vol(K, T2)
    assert iv_after == pytest.approx(iv_at_T2, abs=1e-10)


def test_implied_vol_accepts_arrays_and_scalars(two_slice_chain):
    surf = surface.fit_surface(two_slice_chain)
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([0.2, 0.3, 0.4])
    out = surf.implied_vol(K, T)
    assert out.shape == (3,)

    scalar_out = surf.implied_vol(100.0, 0.3)
    assert isinstance(scalar_out, float)


def test_fit_surface_skips_sparse_expiries():
    sparse_and_full = _synthetic_chain(
        forward_by_T={0.1: 100.0, 0.5: 105.0},
        strikes=STRIKES,
        params_by_T={0.1: SHORT, 0.5: LONG},
    )
    # truncate one expiry down to 3 points -- below the 5-point minimum
    mask_short = sparse_and_full["T"] == 0.1
    keep = sparse_and_full[mask_short].index[:3]
    drop = sparse_and_full[mask_short].index[3:]
    chain = sparse_and_full.drop(drop)

    surf = surface.fit_surface(chain, min_points=5)
    assert list(surf.maturities) == [0.5]


def test_fit_surface_raises_on_calendar_arbitrage():
    # total variance *decreases* from T=0.1 to T=0.5 -> calendar arbitrage
    bad_chain = _synthetic_chain(
        forward_by_T={0.1: 100.0, 0.5: 100.0},
        strikes=STRIKES,
        params_by_T={
            0.1: SVIParams(a=0.05, b=0.1, rho=-0.3, m=0.0, sigma=0.2),
            0.5: SVIParams(a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.2),
        },
    )
    with pytest.raises(arb.ArbitrageError):
        surface.fit_surface(bad_chain)


def test_fit_surface_raises_when_no_expiry_has_enough_points():
    tiny_chain = _synthetic_chain(
        forward_by_T={0.1: 100.0}, strikes=STRIKES[:3], params_by_T={0.1: SHORT},
    )
    with pytest.raises(ValueError):
        surface.fit_surface(tiny_chain)


def test_plot_surface_3d_runs_headlessly(two_slice_chain):
    import matplotlib
    matplotlib.use("Agg")
    surf = surface.fit_surface(two_slice_chain)
    fig, ax = surface.plot_surface_3d(surf, n_strikes=10, n_maturities=10)
    assert fig is not None
    assert ax is not None


# ---------------------------------------------------------------------------
# fit_ssvi_surface: Corollary 4.1 gives a surface-wide guarantee, not just a
# knot-wide one -- spot-check many OFF-KNOT (K, T) points, including beyond
# the last fitted expiry, and confirm none of them show a butterfly
# violation, the way an independent per-expiry raw-SVI surface's
# interpolated region could.
# ---------------------------------------------------------------------------
def _ssvi_synthetic_chain(ssvi_params, forward, Ts, thetas, strikes):
    import ssvi as ssvi_module

    rows = []
    for T, theta in zip(Ts, thetas):
        k = np.log(np.asarray(strikes) / forward)
        w = ssvi_module.ssvi_total_variance(k, theta, ssvi_params)
        mid_iv = np.sqrt(w / T)
        for K, iv in zip(strikes, mid_iv):
            rows.append({
                "strike": float(K), "expiry": pd.Timestamp(0) + pd.Timedelta(days=T * 365),
                "T": T, "forward": forward, "mid_iv": float(iv), "weight": 1.0,
            })
    return pd.DataFrame(rows)


def test_fit_ssvi_surface_has_no_interior_butterfly_violations():
    import ssvi as ssvi_module

    true_params = ssvi_module.SSVIParams(rho=-0.3, phi=ssvi_module.phi_power_law(0.8, 0.4))
    Ts = [0.05, 0.2, 0.5, 1.0, 2.0]
    thetas = [0.02, 0.07, 0.15, 0.28, 0.5]
    assert ssvi_module.check_ssvi_no_static_arbitrage(Ts, thetas, true_params) is True

    chain = _ssvi_synthetic_chain(true_params, forward=100.0, Ts=Ts, thetas=thetas,
                                   strikes=np.linspace(40, 250, 25))

    surf = surface.fit_ssvi_surface(chain, shape="power_law")
    assert isinstance(surf, surface.SSVIVolSurface)

    # off-knot maturities, including one beyond the last fitted expiry
    off_knot_Ts = [0.08, 0.12, 0.35, 0.7, 1.5, 3.0]
    k_grid = np.linspace(-1.0, 1.0, 41)
    h = 1e-4
    for T in off_knot_Ts:
        F = float(surf._forward_at(np.array([T]))[0])

        def w_at(shift, F=F, T=T):
            K = F * np.exp(k_grid + shift)
            iv = surf.implied_vol(K, T)
            return iv ** 2 * T

        w, w_p, w_m = w_at(0.0), w_at(h), w_at(-h)
        dw = (w_p - w_m) / (2 * h)
        d2w = (w_p - 2 * w + w_m) / h ** 2
        g = (1.0 - k_grid * dw / (2.0 * w)) ** 2 - (dw ** 2 / 4.0) * (1.0 / w + 0.25) + d2w / 2.0
        assert g.min() > -1e-6, f"butterfly violation at off-knot T={T}: min g={g.min()}"


def test_fit_ssvi_surface_rejects_unknown_shape():
    import ssvi as ssvi_module

    params = ssvi_module.SSVIParams(rho=-0.2, phi=ssvi_module.phi_power_law(0.5, 0.3))
    chain = _ssvi_synthetic_chain(
        params, forward=100.0, Ts=[0.1, 0.5], thetas=[0.03, 0.1], strikes=np.linspace(60, 160, 15),
    )
    with pytest.raises(ValueError):
        surface.fit_ssvi_surface(chain, shape="not-a-real-shape")


# ---------------------------------------------------------------------------
# run_pipeline: these additions must not change its declared defaults, nor
# its knot-level on_bad_slice="skip" behavior -- it should just transparently
# inherit fit_surface's new defaults (price_space, check_interior=True).
# ---------------------------------------------------------------------------
def test_run_pipeline_signature_unchanged():
    import inspect
    sig = inspect.signature(surface.run_pipeline)
    assert list(sig.parameters) == ["currency", "max_width_frac", "validate", "min_points"]
    assert sig.parameters["currency"].default == "BTC"
    assert sig.parameters["max_width_frac"].default == 0.15
    assert sig.parameters["validate"].default is True
    assert sig.parameters["min_points"].default == surface.MIN_POINTS_PER_SLICE


def test_run_pipeline_calls_fit_surface_with_on_bad_slice_skip(monkeypatch):
    captured = {}

    def fake_fit_surface(chain, **kwargs):
        captured.update(kwargs)
        return "sentinel-surface"

    fake_chain = pd.DataFrame({"strike": [1.0]})
    monkeypatch.setattr(surface, "fit_surface", fake_fit_surface)
    monkeypatch.setattr(data, "fetch_book_summary", lambda currency: None)
    monkeypatch.setattr(data, "build_chain", lambda raw, max_width_frac: fake_chain)

    result = surface.run_pipeline("BTC")

    assert result == "sentinel-surface"
    assert captured["on_bad_slice"] == "skip"
    # run_pipeline doesn't override these -- it relies on fit_surface's own
    # default check_interior=True unchanged
    assert "interp_method" not in captured
    assert "check_interior" not in captured


@pytest.mark.network
def test_run_pipeline_end_to_end_on_live_data():
    try:
        surf = surface.run_pipeline("BTC")
    except Exception as exc:
        pytest.skip(f"Deribit pipeline unavailable: {exc}")

    assert len(surf.maturities) >= 1
    F = surf.forwards[0]
    iv = surf.implied_vol(F, surf.maturities[0])
    assert 0.05 < iv < 3.0


# ---------------------------------------------------------------------------
# Section 5.3 extrapolation beyond the final slice:
#   w(k, theta_t) = w(k, theta_{t_n}) + theta_t - theta_{t_n}
# "which is free of static arbitrage if w(k, theta_{t_n}) is free of
# butterfly arbitrage by Theorem 4.3."
# ---------------------------------------------------------------------------
import ssvi as ssvi_module


def _ssvi_surface(extrapolate="section_5_3", **kwargs):
    params = ssvi_module.SSVIParams(rho=-0.3, phi=ssvi_module.phi_power_law(0.8, 0.4))
    Ts = np.array([0.05, 0.2, 0.5, 1.0, 2.0])
    thetas = np.array([0.02, 0.07, 0.15, 0.28, 0.5])
    forwards = np.full(len(Ts), 100.0)
    return surface.SSVIVolSurface(
        Ts, forwards, params, thetas, extrapolate=extrapolate, **kwargs
    ), params, thetas


def test_section_5_3_extrapolation_matches_the_stated_formula():
    surf, params, thetas = _ssvi_surface()
    k = np.linspace(-1.5, 1.5, 13)
    for T in (2.5, 4.0, 10.0):
        w_actual = surf.implied_vol(100.0 * np.exp(k), T) ** 2 * T
        theta_T = float(surf._theta_at(np.array(T)))
        w_expected = (
            ssvi_module.ssvi_total_variance(k, thetas[-1], params) + theta_T - thetas[-1]
        )
        assert w_actual == pytest.approx(w_expected, abs=1e-12)


def test_section_5_3_extrapolation_is_continuous_at_the_final_knot():
    surf, _, _ = _ssvi_surface()
    at_knot = surf.implied_vol(100.0, 2.0)
    just_after = surf.implied_vol(100.0, 2.0 + 1e-9)
    assert just_after == pytest.approx(at_knot, abs=1e-8)


def test_section_5_3_extrapolation_is_calendar_increasing_unlike_flat_theta():
    s53, _, _ = _ssvi_surface("section_5_3")
    flat, _, _ = _ssvi_surface("flat_theta")
    Ts = np.linspace(2.0, 20.0, 300)
    for k in (-1.5, 0.0, 1.5):
        K = 100.0 * np.exp(k)
        w_s53 = s53.implied_vol(K, Ts) ** 2 * Ts
        w_flat = flat.implied_vol(K, Ts) ** 2 * Ts
        assert np.diff(w_s53).min() > 0            # strictly increasing
        assert np.diff(w_flat).max() < 1e-12       # flat theta: constant total variance


def test_section_5_3_extrapolation_stays_butterfly_free(k_grid=arb.DEFAULT_K_GRID):
    # Theorem 4.3: adding a non-negative, non-decreasing alpha_t preserves it
    surf, _, _ = _ssvi_surface()
    h = 1e-4
    for T in (2.5, 5.0, 20.0):
        def w_at(shift, T=T):
            return surf.implied_vol(100.0 * np.exp(k_grid + shift), T) ** 2 * T
        w, w_p, w_m = w_at(0.0), w_at(h), w_at(-h)
        dw, d2w = (w_p - w_m) / (2 * h), (w_p - 2 * w + w_m) / h ** 2
        g = (1 - k_grid * dw / (2 * w)) ** 2 - (dw ** 2 / 4) * (1 / w + 0.25) + d2w / 2
        assert g.min() > 0


def test_section_5_3_accepts_a_custom_monotonic_theta_extrapolation():
    surf, params, thetas = _ssvi_surface(theta_extrapolation=lambda T: thetas_ref[-1] + 0.1 * (T - 2.0))
    theta_T = float(surf._theta_at(np.array(6.0)))
    assert theta_T == pytest.approx(thetas[-1] + 0.1 * 4.0)


thetas_ref = np.array([0.02, 0.07, 0.15, 0.28, 0.5])


def test_ssvi_vol_surface_rejects_unknown_extrapolation_mode():
    with pytest.raises(ValueError, match="extrapolate"):
        _ssvi_surface("nonsense")

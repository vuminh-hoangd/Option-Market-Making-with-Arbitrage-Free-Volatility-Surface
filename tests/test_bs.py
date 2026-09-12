import numpy as np
import pytest

import bs

# Textbook reference cases: (S, K, T, r, q, sigma, type, expected_price)
# Values cross-checked against Hull's examples / standard BS calculators.
REFERENCE_CASES = [
    (100, 100, 1.0, 0.05, 0.0, 0.2, "C", 10.4506),
    (100, 100, 1.0, 0.05, 0.0, 0.2, "P", 5.5735),
    (60, 65, 0.25, 0.08, 0.0, 0.3, "C", 2.1334),
    (42, 40, 0.5, 0.10, 0.0, 0.2, "C", 4.7594),
    (42, 40, 0.5, 0.10, 0.0, 0.2, "P", 0.8086),
]


@pytest.mark.parametrize("S,K,T,r,q,sigma,otype,expected", REFERENCE_CASES)
def test_price_matches_reference(S, K, T, r, q, sigma, otype, expected):
    got = bs.price(S, K, T, r, q, sigma, otype)
    assert got == pytest.approx(expected, abs=1e-3)


def test_put_call_parity():
    S, K, T, r, q, sigma = 100.0, 95.0, 0.75, 0.03, 0.01, 0.25
    C = bs.price(S, K, T, r, q, sigma, "C")
    P = bs.price(S, K, T, r, q, sigma, "P")
    assert C - P == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_vega_identity():
    S, K, T, r, q, sigma = 100.0, 110.0, 1.5, 0.02, 0.0, 0.35
    d1, d2 = bs._d1_d2(S, K, T, r, q, sigma)
    from scipy.stats import norm

    lhs = S * np.exp(-q * T) * norm.pdf(d1)
    rhs = K * np.exp(-r * T) * norm.pdf(d2)
    assert lhs == pytest.approx(rhs, rel=1e-8)


def test_price_monotonic_in_sigma():
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    sigmas = np.linspace(0.05, 1.5, 50)
    for otype in ("C", "P"):
        prices = np.array([bs.price(S, K, T, r, q, s, otype) for s in sigmas])
        assert np.all(np.diff(prices) > 0)


def test_vectorized_over_arrays():
    S = np.array([100.0, 100.0, 100.0])
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([1.0, 1.0, 1.0])
    r = np.array([0.05, 0.05, 0.05])
    q = np.array([0.0, 0.0, 0.0])
    sigma = np.array([0.2, 0.2, 0.2])
    otype = np.array(["C", "C", "P"])

    prices = bs.price(S, K, T, r, q, sigma, otype)
    assert prices.shape == (3,)
    # cross-check each element against the scalar call
    for i in range(3):
        expected = bs.price(S[i], K[i], T[i], r[i], q[i], sigma[i], otype[i])
        assert prices[i] == pytest.approx(expected)


def test_greeks_call_put_signs():
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
    call_delta = bs.delta(S, K, T, r, q, sigma, "C")
    put_delta = bs.delta(S, K, T, r, q, sigma, "P")
    assert 0 < call_delta < 1
    assert -1 < put_delta < 0
    # delta_call - delta_put == exp(-qT)
    assert call_delta - put_delta == pytest.approx(np.exp(-q * T), abs=1e-10)


def test_gamma_vega_positive_and_shared_across_type():
    S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.03, 0.01, 0.3
    g_c = bs.gamma(S, K, T, r, q, sigma, "C")
    g_p = bs.gamma(S, K, T, r, q, sigma, "P")
    v_c = bs.vega(S, K, T, r, q, sigma, "C")
    v_p = bs.vega(S, K, T, r, q, sigma, "P")
    assert g_c == pytest.approx(g_p)
    assert v_c == pytest.approx(v_p)
    assert g_c > 0
    assert v_c > 0


def test_delta_via_finite_difference():
    S, K, T, r, q, sigma = 100.0, 95.0, 0.8, 0.04, 0.02, 0.22
    h = 1e-4
    for otype in ("C", "P"):
        analytic = bs.delta(S, K, T, r, q, sigma, otype)
        fd = (bs.price(S + h, K, T, r, q, sigma, otype) - bs.price(S - h, K, T, r, q, sigma, otype)) / (2 * h)
        assert analytic == pytest.approx(fd, abs=1e-4)


def test_vega_via_finite_difference():
    S, K, T, r, q, sigma = 100.0, 95.0, 0.8, 0.04, 0.02, 0.22
    h = 1e-5
    analytic = bs.vega(S, K, T, r, q, sigma, "C")
    fd = (bs.price(S, K, T, r, q, sigma + h, "C") - bs.price(S, K, T, r, q, sigma - h, "C")) / (2 * h)
    assert analytic == pytest.approx(fd, abs=1e-3)

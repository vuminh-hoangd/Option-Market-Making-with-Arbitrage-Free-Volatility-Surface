"""
Closed-form solution of the Riccati ODE for psi2(t), and the discount kernel
D(t,u) built from it (paper eq. 8-9).

psi2 is the coefficient of q^2 in the ansatz
h(t,s,q) = psi0(t,s) + psi1(t,s) q + psi2(t) q^2 for the value function, so
it is the pure inventory-risk term: it depends on the penalty parameters
(alpha, beta) and the order-flow parameters, and on nothing about the option
or the spot. It satisfies

    psi2' + 2 e^{-1} (lambda0_b kappa_b + lambda0_a kappa_a) psi2^2 - beta = 0,
    psi2(T) = -alpha,

which is a constant-coefficient Riccati equation and so is solved in closed
form -- no numerical ODE solver anywhere in this module.

D(t,u) is the kernel that discounts future edge back to now in `psi1`; it is
the Green's function of the linear parabolic PDE psi2 induces, and satisfies
D(t,t) = 1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # import for typing only -- `quoting` imports this module
    from .quoting import MMParams


def upsilon(lambda0_b: float, kappa_b: float, lambda0_a: float, kappa_a: float) -> float:
    """
    Upsilon = 1 / (2 e^{-1} (lambda0_a kappa_a + lambda0_b kappa_b)).

    The aggregate two-sided liquidity of the option: large when the book is
    thin or fills are insensitive to the spread, small when flow is plentiful
    and spread-elastic.
    """
    if lambda0_b <= 0.0 or lambda0_a <= 0.0:
        raise ValueError(
            f"base intensities must be positive, got lambda0_b={lambda0_b}, "
            f"lambda0_a={lambda0_a}"
        )
    if kappa_b <= 0.0 or kappa_a <= 0.0:
        raise ValueError(
            f"intensity decays must be positive, got kappa_b={kappa_b}, kappa_a={kappa_a}"
        )
    return 1.0 / (2.0 * np.exp(-1.0) * (lambda0_a * kappa_a + lambda0_b * kappa_b))


def eta(beta: float, upsilon_val: float) -> float:
    """eta = sqrt(beta / Upsilon) -- the rate at which psi2 relaxes off its terminal value."""
    if beta <= 0.0:
        raise ValueError(
            f"running penalty must be positive, got beta={beta}; beta=0 degenerates "
            f"eta to zero and this closed form with it"
        )
    if upsilon_val <= 0.0:
        raise ValueError(f"Upsilon must be positive, got {upsilon_val}")
    return float(np.sqrt(beta / upsilon_val))


def zeta_pm(alpha: float, beta: float, upsilon_val: float) -> tuple[float, float]:
    """
    (zeta_plus, zeta_minus) = (sqrt(Upsilon beta) + alpha, sqrt(Upsilon beta) - alpha).

    zeta_minus goes negative once alpha > sqrt(Upsilon beta) (a terminal
    penalty dominating the running one). That is admissible and does not
    break anything: see `_denominator` for why the solution stays regular on
    [0, T] either way.
    """
    if alpha <= 0.0:
        raise ValueError(f"terminal penalty must be positive, got alpha={alpha}")
    root = np.sqrt(upsilon_val * beta)
    return float(root + alpha), float(root - alpha)


def _coefficients(params: "MMParams") -> tuple[float, float, float, float]:
    """(sqrt(Upsilon beta), eta, zeta_plus, zeta_minus) for a parameter set."""
    ups = upsilon(params.lambda0_b, params.kappa_b, params.lambda0_a, params.kappa_a)
    eta_val = eta(params.beta, ups)
    zeta_p, zeta_m = zeta_pm(params.alpha, params.beta, ups)
    return float(np.sqrt(ups * params.beta)), eta_val, zeta_p, zeta_m


def _denominator(zeta_p: float, zeta_m: float, x: float) -> float:
    """
    The common denominator zeta_minus e^{-eta(T-t)} + zeta_plus e^{eta(T-t)},
    rescaled by e^{-eta(T-t)} to `zeta_minus e^{-2x} + zeta_plus` with
    x = eta (T-t) >= 0.

    Rescaling matters: eta(T-t) can be tens for a short horizon with a large
    beta, and the raw form overflows to inf/inf where this one does not.

    It never vanishes for t <= T. zeta_plus > 0 always, so a zero needs
    zeta_minus < 0 and e^{-2x} = zeta_plus/|zeta_minus| > 1, i.e. x < 0, i.e.
    t > T. Every query in this model has t <= T, so psi2 and D are regular on
    the whole horizon.
    """
    den = zeta_m * np.exp(-2.0 * x) + zeta_p
    if den == 0.0:
        raise ValueError(
            f"degenerate Riccati denominator at eta*(T-t)={x}; this should be "
            f"unreachable for t <= T"
        )
    return float(den)


def _check_horizon(t: float, T: float) -> None:
    if not np.isfinite(t) or not np.isfinite(T):
        raise ValueError(f"t and T must be finite, got t={t}, T={T}")
    if t > T:
        raise ValueError(f"t must not exceed the horizon T, got t={t}, T={T}")


def psi2(t: float, T: float, params: "MMParams") -> float:
    """
    psi2(t), solution of the Riccati ODE with psi2(T) = -alpha.

        psi2(t) = sqrt(Upsilon beta)
                  * (zeta_- e^{-eta(T-t)} - zeta_+ e^{eta(T-t)})
                  / (zeta_- e^{-eta(T-t)} + zeta_+ e^{eta(T-t)})

    Negative throughout [0, T] (it penalises inventory), decreasing in
    magnitude away from expiry, and bounded below by -sqrt(Upsilon beta) or
    -alpha, whichever is larger.
    """
    _check_horizon(t, T)
    root_ub, eta_val, zeta_p, zeta_m = _coefficients(params)

    x = eta_val * (T - t)
    scaled = zeta_m * np.exp(-2.0 * x)
    return float(root_ub * (scaled - zeta_p) / _denominator(zeta_p, zeta_m, x))


def discount_kernel(t: float, u: float, T: float, params: "MMParams") -> float:
    """
    D(t,u), the kernel discounting edge earned at future time u back to t:

        D(t,u) = (zeta_- e^{-eta(T-u)} + zeta_+ e^{eta(T-u)})
                 / (zeta_- e^{-eta(T-t)} + zeta_+ e^{eta(T-t)})

    D(t,t) = 1, and D decays as u moves toward T -- edge you can only harvest
    later is worth less now, because the inventory you must carry to harvest
    it is itself penalised. Requires t <= u <= T.
    """
    _check_horizon(t, T)
    if u < t or u > T:
        raise ValueError(f"u must lie in [t, T], got t={t}, u={u}, T={T}")

    _, eta_val, zeta_p, zeta_m = _coefficients(params)

    # Both exponents below are <= 0 after rescaling by e^{-eta(T-t)}, since
    # t <= u <= T; see `_denominator` on why the raw form is not used.
    x = eta_val * (T - t)
    y = eta_val * (T - u)
    numerator = zeta_m * np.exp(-y - x) + zeta_p * np.exp(y - x)
    return float(numerator / _denominator(zeta_p, zeta_m, x))

"""
SSVI (Surface SVI): Gatheral & Jacquier, "Arbitrage-free SVI volatility
surfaces" (2013), Section 4.

SSVI parameterizes the *entire* implied-vol surface with a single
correlation `rho` and a shape function `phi`, rather than fitting each
expiry's 5 raw-SVI parameters independently (Steps 4-5 of the rest of this
project). This module implements Definition 4.1, the closed-form
no-arbitrage conditions of Theorems 4.1-4.3 and Corollary 4.1, and a
bridge back to `arbitrage.py`'s raw-SVI machinery (via the natural-SVI
correspondence of Lemma 3.1) so the closed-form conditions here can be
cross-checked against the existing numerical `g(k)` grid test rather than
trusted blindly.

Definition 4.1: for every t > 0, with theta_t the ATM total variance,
    w(k, theta_t) = (theta_t/2) * {1 + rho*phi(theta_t)*k
                                      + sqrt((phi(theta_t)*k+rho)^2 + (1-rho^2))}
This is exactly the natural-SVI slice (Delta=0, mu=0, omega=theta_t,
zeta=phi(theta_t)) at every t, so `ssvi_slice_to_raw_svi` converts an SSVI
slice to `arbitrage.SVIParams` via Lemma 3.1's natural -> raw mapping.

Two flavors of no-arbitrage check are provided, mirroring the
grid-vs-closed-form split in `arbitrage.py`:
- `check_ssvi_calendar` / `check_ssvi_butterfly` / `check_ssvi_no_static_arbitrage`
  implement the closed-form SSVI conditions (Theorems 4.1, 4.2, Corollary 4.1)
  directly on (rho, phi) -- cheap, and exact for any (rho, phi), not just
  ones representable pointwise.
- `check_ssvi_no_static_arbitrage(..., cross_validate=True)` additionally
  converts each checked slice to raw SVI and re-checks it with
  `arbitrage.check_butterfly`, as an independent numerical sanity check
  (Theorem 4.2's conditions are sufficient, not necessary except at the
  condition-1 boundary, so this should never disagree in the direction of
  "closed-form says fine, numerical says arbitrage" -- if it does, that's
  treated as an implementation bug, not a real edge case: see
  `SSVIConsistencyError`).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from arbitrage import ArbitrageError, DEFAULT_K_GRID, SVIParams, check_butterfly

CALENDAR_TOL = 1e-9
BUTTERFLY_TOL = 1e-9
THETA_MONOTONIC_TOL = -1e-9
BETA_ZERO_TOL = 1e-14


class SSVIConsistencyError(RuntimeError):
    """
    Raised when the closed-form SSVI butterfly conditions (Theorem 4.2)
    say a slice is arbitrage-free but the independent raw-SVI numerical
    check (`arbitrage.check_butterfly`) disagrees. Theorem 4.2's
    conditions are sufficient (and, for condition (i), also necessary --
    Lemma 4.2), so this combination should never happen; if it does, it
    points at a bug in this module's formulas or the conversion, not a
    genuine arbitrage edge case.
    """


@dataclass
class SSVIParams:
    """
    SSVI surface: w(k, theta_t) = theta_t/2 * {1 + rho*phi(theta_t)*k
                                                + sqrt((phi(theta_t)*k+rho)^2 + (1-rho^2))}
    rho:  correlation, constant across all maturities, |rho| < 1
    phi:  shape function theta -> phi(theta), phi(theta) > 0 for theta > 0
    """
    rho: float
    phi: Callable[[float], float]


def phi_heston_like(lam: float) -> Callable[[float], float]:
    """
    Example 4.1: phi(theta) = 1/(lam*theta) * (1 - (1-exp(-lam*theta))/(lam*theta)).

    Theorem 4.1's calendar condition holds for *every* lam > 0 and every
    rho in (-1,1) with this shape function (that's the point of the
    example) -- see `tests/test_ssvi.py`.

    Uses `np.expm1` for `1 - exp(-u)` to avoid catastrophic cancellation
    as theta -> 0+, where phi is finite (-> 1/2) but the naive formula
    subtracts two nearly-equal terms.
    """
    def phi(theta):
        theta = np.asarray(theta, dtype=float)
        u = lam * theta
        return (1.0 / u) * (1.0 - (-np.expm1(-u)) / u)

    def d_theta_phi(theta):
        # Example 4.1's closed form:
        #   d/dtheta(theta*phi(theta)) = e^{-lam theta}(e^{lam theta} - 1 - lam theta)
        #                                / (lam^2 theta^2)
        theta = np.asarray(theta, dtype=float)
        u = lam * theta
        return np.exp(-u) * (np.expm1(u) - u) / u ** 2

    def d_theta_phi_over_phi(theta):
        # Example 4.1: d_theta(theta phi(theta))/phi(theta)
        #            = (1 - (1 + lam theta) e^{-lam theta}) / (e^{-lam theta} + lam theta - 1)
        #
        # Both numerator and denominator vanish like u^2/2 as u -> 0, so the
        # expression as written loses almost all precision for small theta
        # (1 - x with x -> 1, and e^-u + u - 1 likewise). Written with expm1
        # -- the same rearrangement `phi` itself uses -- the subtractions
        # happen between accurately-computed small quantities instead:
        #   numerator   = -expm1(-u) - u*e^{-u}
        #   denominator =  expm1(-u) + u
        theta = np.asarray(theta, dtype=float)
        u = lam * theta
        exp_minus_u = np.exp(-u)
        return (-np.expm1(-u) - u * exp_minus_u) / (np.expm1(-u) + u)

    phi.d_theta_phi = d_theta_phi
    phi.d_theta_phi_over_phi = d_theta_phi_over_phi
    phi.phi_0 = 0.0  # lim_{theta->0} theta*phi(theta); Example 4.1 has phi_0 = 0
    return phi


def phi_power_law(eta: float, gamma: float) -> Callable[[float], float]:
    """
    Example 4.2: phi(theta) = eta * theta**(-gamma), eta > 0, 0 < gamma < 1.

    The example also records d_theta(theta*phi(theta))/phi(theta) = 1-gamma
    in (0,1) for all theta > 0, which is why Theorem 4.1's conditions hold
    unconditionally for this family.

    Remark 4.4 warns that such surfaces "can be free of static arbitrage
    only up to some maximum expiry", because Theorem 4.2's conditions
    eventually fail as theta grows -- see `phi_power_law_bounded` (Equation
    (4.5)) for the modification the paper gives to fix that.
    """
    def phi(theta):
        theta = np.asarray(theta, dtype=float)
        return eta * theta ** (-gamma)

    def d_theta_phi(theta):
        theta = np.asarray(theta, dtype=float)
        return eta * (1.0 - gamma) * theta ** (-gamma)

    def d_theta_phi_over_phi(theta):
        theta = np.asarray(theta, dtype=float)
        return np.full_like(theta, 1.0 - gamma)

    phi.d_theta_phi = d_theta_phi
    phi.d_theta_phi_over_phi = d_theta_phi_over_phi
    # lim_{theta->0} theta*phi(theta) = eta*lim theta^{1-gamma} = 0 for gamma < 1
    phi.phi_0 = 0.0
    return phi


def phi_power_law_bounded(eta: float, gamma: float) -> Callable[[float], float]:
    """
    Equation (4.5), the modified power law of Remark 4.4:

        phi(theta) = eta / (theta**gamma * (1 + theta)**(1 - gamma))

    which "gives a surface that is completely free of static arbitrage
    provided that eta*(1 + |rho|) <= 2" -- i.e. free of static arbitrage for
    ALL maturities, not merely up to some t*, which is what plain
    `phi_power_law` can only offer (Remark 4.4).

    Use `check_eq_4_5_eta_condition` to test eta*(1 + |rho|) <= 2.
    """
    def phi(theta):
        theta = np.asarray(theta, dtype=float)
        return eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))

    # theta*phi(theta) = eta*theta^{1-gamma}/(1+theta)^{1-gamma} -> 0 as theta -> 0
    phi.phi_0 = 0.0
    return phi


def check_eq_4_5_eta_condition(eta: float, rho: float):
    """
    Raise ArbitrageError unless eta*(1 + |rho|) <= 2, the condition under
    which Remark 4.4's Equation (4.5) shape function gives a surface
    "completely free of static arbitrage".
    """
    value = eta * (1.0 + abs(rho))
    if value > 2.0:
        raise ArbitrageError(
            f"Equation (4.5) condition violated: eta*(1+|rho|) = {value:.4f} > 2 "
            f"(eta={eta}, rho={rho}); the surface is not guaranteed free of static arbitrage"
        )
    return True


def heston_like_lambda_lower_bound(rho: float):
    """
    Remark 4.5: for the Heston-like shape function of Example 4.1,
    lim_{theta->+inf} theta*phi(theta)*(1+|rho|) = (1+|rho|)/lam, so
    Condition 3 of Corollary 4.1 imposes lam >= (1+|rho|)/4. Returns that
    lower bound.
    """
    return (1.0 + abs(rho)) / 4.0


def check_heston_like_lambda(lam: float, rho: float):
    """
    Raise ArbitrageError unless lam >= (1+|rho|)/4 (Remark 4.5).

    `check_ssvi_butterfly` only tests the theta values it is handed, so a
    lam below this bound can pass on a short-dated grid and still violate
    Corollary 4.1 Condition 3 in the large-theta limit. This checks the
    limit itself.
    """
    bound = heston_like_lambda_lower_bound(rho)
    if lam < bound:
        raise ArbitrageError(
            f"Remark 4.5: lam = {lam:.6f} < (1+|rho|)/4 = {bound:.6f} (rho={rho}), so "
            f"theta*phi(theta)*(1+|rho|) -> {(1 + abs(rho)) / lam:.4f} >= 4 as theta -> +inf, "
            f"violating Condition 3 of Corollary 4.1"
        )
    return True


def ssvi_total_variance(k, theta: float, params: SSVIParams):
    """Evaluate w(k, theta) directly from Definition 4.1."""
    k = np.asarray(k, dtype=float)
    ph = params.phi(theta)
    return (theta / 2.0) * (
        1.0 + params.rho * ph * k + np.sqrt((ph * k + params.rho) ** 2 + (1.0 - params.rho ** 2))
    )


def ssvi_jw_params(theta: float, T: float, params: SSVIParams):
    """
    Lemma 4.1: the SVI-JW parameters of the SSVI surface (4.1), in closed
    form directly from (theta_t, phi, rho) -- no conversion to raw SVI:

        v_t       = theta_t / t
        psi_t     = (1/2) * rho * sqrt(theta_t) * phi(theta_t)
        p_t       = (1/2) * sqrt(theta_t) * phi(theta_t) * (1 - rho)
        c_t       = (1/2) * sqrt(theta_t) * phi(theta_t) * (1 + rho)
        v_tilde_t = (theta_t / t) * (1 - rho^2)
    """
    rho = params.rho
    root_theta_phi = np.sqrt(theta) * params.phi(theta)
    return {
        "v_t": theta / T,
        "psi_t": 0.5 * rho * root_theta_phi,
        "p_t": 0.5 * root_theta_phi * (1.0 - rho),
        "c_t": 0.5 * root_theta_phi * (1.0 + rho),
        "v_tilde_t": (theta / T) * (1.0 - rho ** 2),
    }


def ssvi_atm_volatility_skew(theta: float, T: float, params: SSVIParams):
    """
    Equation (4.2): the ATM volatility skew of the SSVI surface,

        d/dk sigma_BS(k, t)|_{k=0} = rho * sqrt(theta_t) * phi(theta_t) / (2*sqrt(t)).

    Example 4.2 notes that with the power-law shape function this reduces
    to rho*eta/(2*sqrt(t)) when gamma = 1/2.
    """
    return params.rho * np.sqrt(theta) * params.phi(theta) / (2.0 * np.sqrt(T))


def ssvi_zero_time_smile(k, phi_0: float, rho: float):
    """
    Equation (4.3): the time-zero smile of an SSVI surface. Since
    theta_0 = 0,

        w(k, theta_0) = (1/2) * phi_0 * (rho*k + |k|),   for any k in R,

    where phi_0 := lim_{theta->0} theta*phi(theta) (see `ssvi_phi_0`).
    The paper notes phi_0 = 0 is characteristic of stochastic volatility
    models (Example 4.1), while phi_0 > 0 (Example 4.2) gives the V-shaped
    time-zero smile characteristic of models with jumps.
    """
    k = np.asarray(k, dtype=float)
    return 0.5 * phi_0 * (rho * k + np.abs(k))


def ssvi_phi_0(params: SSVIParams, theta_small: float = 1e-12):
    """
    phi_0 := lim_{theta -> 0} theta*phi(theta), the quantity Equation (4.3)
    is written in terms of. Definition 4.1 requires this limit to exist in
    R. Evaluated numerically at a small theta unless the shape function
    supplies an exact value via a `phi_0` attribute (both `phi_heston_like`
    and `phi_power_law` do).
    """
    exact = getattr(params.phi, "phi_0", None)
    if exact is not None:
        return float(exact)
    return float(theta_small * params.phi(theta_small))


def ssvi_asymptotic_total_variance(k, theta: float, params: SSVIParams):
    """
    Remark 4.3: the asymptotic behaviour of SSVI (4.1) as |k| -> infinity,

        w(k, theta_t) = ((1 +/- rho)*theta_t/2) * phi(theta_t)*|k| + O(1),

    with the + branch for k -> +infinity and the - branch for k -> -infinity.
    Returns the leading term only (the O(1) remainder is dropped), so it is
    a large-|k| approximation to `ssvi_total_variance`, not a replacement.
    """
    k = np.asarray(k, dtype=float)
    sign = np.where(k >= 0.0, 1.0, -1.0)
    return (1.0 + sign * params.rho) * theta / 2.0 * params.phi(theta) * np.abs(k)


def ssvi_slice_to_raw_svi(theta: float, params: SSVIParams, alpha: float = 0.0) -> SVIParams:
    """
    Convert one SSVI slice (fixed theta_t) to raw SVI parameters, via the
    natural-SVI correspondence: chi_N = {0, 0, rho, theta, phi(theta)},
    then Lemma 3.1's natural -> raw mapping:
        a = Delta + omega/2*(1-rho^2),  b = omega*zeta/2,
        m = mu - rho/zeta,  sigma = sqrt(1-rho^2)/zeta,
    with Delta=0, mu=0, omega=theta, zeta=phi(theta).

    `alpha` implements the Theorem 4.3 shift: adding a constant alpha to
    total variance at every k is exactly a shift of the natural-SVI
    `Delta` parameter by alpha, which (Delta being purely additive in the
    raw-SVI mapping too) means simply `a += alpha` -- the other four raw
    parameters are untouched. See `apply_theorem_4_3_shift`.

    Returns an SVIParams usable directly with arbitrage.py's g_function,
    check_butterfly, etc. -- this is the bridge for cross-validating the
    closed-form SSVI conditions against the existing numerical machinery.
    """
    rho = params.rho
    zeta = params.phi(theta)
    omega = theta
    a = omega / 2.0 * (1.0 - rho ** 2) + alpha
    b = omega * zeta / 2.0
    m = -rho / zeta
    sigma = np.sqrt(1.0 - rho ** 2) / zeta
    return SVIParams(a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sigma))


def svi_jw_params(params: SVIParams, T: float):
    """
    Convert raw SVI parameters (for a slice at maturity T) to the
    SVI-Jump-Wings (JW) parametrization (Gatheral 2004; Lemma 4.1 in
    Gatheral & Jacquier 2013): ATM variance v_t, ATM skew psi_t, left/put
    wing slope p_t, right/call wing slope c_t, and minimum variance
    v_tilde_t. Returned as a dict.

    Used here to demonstrate a special case noted in the paper: with the
    power-law shape function (`phi_power_law`) and gamma=1/2, psi_t, p_t,
    and c_t come out *constant* across theta (see `tests/test_ssvi.py`).
    """
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    w_t = a + b * (-rho * m + np.sqrt(m ** 2 + sigma ** 2))  # ATM total variance
    sqrt_w = np.sqrt(w_t)
    return {
        "v_t": w_t / T,
        "psi_t": b * (rho - m / np.sqrt(m ** 2 + sigma ** 2)) / (2.0 * sqrt_w),
        "p_t": b * (1.0 - rho) / sqrt_w,
        "c_t": b * (1.0 + rho) / sqrt_w,
        "v_tilde_t": (a + b * sigma * np.sqrt(1.0 - rho ** 2)) / T,
    }


def svi_jw_to_raw(jw: dict, T: float) -> SVIParams:
    """
    Invert `svi_jw_params`: recover raw SVI parameters from the SVI-JW
    parameters at maturity T. This is Lemma 3.2 of Gatheral & Jacquier
    (2013), implemented exactly as stated there.

    With w_t = v_t*T, the lemma defines

        beta := rho - 2*psi_t*sqrt(w_t)/b,   alpha := sign(beta)*sqrt(1/beta^2 - 1)

    (the lemma assumes beta in [-1,1], equivalently -p_t <= 2*psi_t <= c_t)
    and then

        b     = sqrt(w_t)/2 * (c_t + p_t)
        rho   = 1 - p_t*sqrt(w_t)/b
        m     = (v_t - v_tilde_t)*T
                / ( b*{-rho + sign(alpha)*sqrt(1+alpha^2) - alpha*sqrt(1-rho^2)} )
        sigma = alpha*m
        a     = v_tilde_t*T - b*sigma*sqrt(1-rho^2)

    The lemma's m != 0 branch is the general case. The m = 0 branch is
    signalled by beta = 0 (alpha is then unbounded, so `sigma = alpha*m` is
    the indeterminate 0*inf); there the lemma says b, rho and a keep their
    formulae above but sigma = (v_t*T - a)/b. Those last two are a pair of
    linear equations in the two unknowns (a, sigma), solved here as stated.

    Raises ValueError if beta falls outside [-1,1], where the lemma's
    hypotheses -- and hence the inversion -- do not hold, or if the m = 0
    branch is additionally degenerate (rho = 0, where v_t and v_tilde_t
    coincide and sigma is not recoverable from the JW parameters).

    Numerical note: the lemma's m formula carries the difference
    (v_t - v_tilde_t) in its numerator, so it loses relative precision as
    the smile's minimum approaches the money (v_t -> v_tilde_t, i.e. the
    approach to the m = 0 branch). Round-trip error is ~1e-15 typically and
    ~1e-6 in the worst ~0.04% of randomly sampled slices; that is the
    conditioning of the stated formula, not an extra approximation here.
    """
    v_t, psi_t = float(jw["v_t"]), float(jw["psi_t"])
    p_t, c_t, v_tilde_t = float(jw["p_t"]), float(jw["c_t"]), float(jw["v_tilde_t"])

    w_t = v_t * T
    sqrt_w = np.sqrt(w_t)

    b = sqrt_w / 2.0 * (c_t + p_t)
    rho = 1.0 - p_t * sqrt_w / b
    beta = rho - 2.0 * psi_t * sqrt_w / b

    if not -1.0 <= beta <= 1.0:
        raise ValueError(
            f"Lemma 3.2 requires beta in [-1,1] (equivalently -p_t <= 2*psi_t <= c_t); "
            f"got beta={beta:.6f} from v_t={v_t}, psi_t={psi_t}, p_t={p_t}, c_t={c_t}"
        )

    root_rho = np.sqrt(1.0 - rho ** 2)

    if abs(beta) < BETA_ZERO_TOL:
        # m = 0 branch: solve  a = v_tilde_t*T - b*sigma*root_rho  jointly
        # with  sigma = (v_t*T - a)/b.
        if abs(1.0 - root_rho) < 1e-15:
            raise ValueError(
                "Lemma 3.2 m=0 branch is degenerate at rho=0: v_t and v_tilde_t coincide "
                "and sigma cannot be recovered from the SVI-JW parameters"
            )
        sigma = (v_t - v_tilde_t) * T / (b * (1.0 - root_rho))
        a = v_tilde_t * T - b * sigma * root_rho
        return SVIParams(a=float(a), b=float(b), rho=float(rho), m=0.0, sigma=float(sigma))

    numerator = (v_t - v_tilde_t) * T
    alpha = np.sign(beta) * np.sqrt(1.0 / beta ** 2 - 1.0)
    denominator = b * (
        -rho + np.sign(alpha) * np.sqrt(1.0 + alpha ** 2) - alpha * root_rho
    )
    m = numerator / denominator
    sigma = alpha * m
    a = v_tilde_t * T - b * sigma * root_rho
    return SVIParams(a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sigma))


def svi_jw_to_ssvi_coordinates(jw: dict, T: float):
    """
    Recover (theta_t, phi(theta_t), rho) from SVI-JW parameters that belong
    to an SSVI slice, by reading Lemma 4.1 backwards. The lemma states

        p_t = (1/2)*sqrt(theta_t)*phi(theta_t)*(1-rho)
        c_t = (1/2)*sqrt(theta_t)*phi(theta_t)*(1+rho)
        v_t = theta_t/t

    so p_t + c_t = sqrt(theta_t)*phi(theta_t) and c_t - p_t = that times rho.

    Only meaningful for JW parameters that really are an SSVI slice's --
    i.e. ones satisfying Lemma 4.1's remaining two identities
    c_t = p_t + 2*psi_t and v_tilde_t = v_t*(1-rho^2). That is exactly what
    `eliminate_butterfly_arbitrage` imposes, which is why the two are used
    together.
    """
    theta = jw["v_t"] * T
    total = jw["p_t"] + jw["c_t"]
    return theta, total / np.sqrt(theta), (jw["c_t"] - jw["p_t"]) / total


def eliminate_butterfly_arbitrage(params: SVIParams, T: float, validate: bool = True) -> SVIParams:
    """
    Return a butterfly-arbitrage-free raw SVI slice built from `params` by
    the recipe of Gatheral & Jacquier (2013) Section 5.1: hold the SVI-JW
    parameters v_t, psi_t and p_t fixed and replace the remaining two with

        c_t'       = p_t + 2*psi_t
        v_tilde_t' = v_t * 4*p_t*c_t' / (p_t + c_t')^2

    "In other words, given a smile defined in terms of its SVI-JW
    parameters, we are guaranteed to be able to eliminate butterfly
    arbitrage by changing the call wing c_t and the minimum variance
    v_tilde_t."

    This is the paper's *repair* step, the counterpart to `check_butterfly`'s
    detection: rather than discarding an arbitrageable expiry, it moves the
    slice onto the no-arbitrage boundary while preserving ATM level, ATM
    skew and the put wing -- the three features calibrated most reliably
    from quotes (Section 5.1 notes c_t and v_tilde_t are "both parameters
    that are hard to calibrate with available quotes in equity options
    markets").

    Section 5.1 derives the recipe "in view of Lemma 4.1": both
    replacements are precisely Lemma 4.1's identities (c_t = p_t + 2*psi_t,
    and v_tilde_t = v_t*(1-rho^2) rewritten via rho = (c_t-p_t)/(c_t+p_t)),
    so what the recipe actually does is force the slice into SSVI form. It
    is therefore the *most extreme* admissible smile, not a best fit;
    Example 5.1 goes on to search between the original and repaired
    parameters for a closer fit that still passes, and that search is a
    calibration choice left to the caller.

    Being in SSVI form is not on its own sufficient: Theorem 4.2's two
    conditions must also hold on the resulting (theta_t, phi, rho). They do
    so comfortably for the Vogt smile of Example 5.1, but not universally --
    across ~13000 randomly generated arbitrageable slices, ~72% of repaired
    outputs violated one or both (condition 2, theta*phi^2*(1+|rho|) <= 4,
    being the one that binds almost always; it fails for smiles with a small
    sigma, whose phi is correspondingly large). `validate=True` therefore
    checks Theorem 4.2 on the result via `check_ssvi_butterfly` and raises
    `ArbitrageError` rather than returning a slice that is still
    arbitrageable. Pass `validate=False` for the bare Section 5.1 output.
    """
    jw = svi_jw_params(params, T)
    c_new = jw["p_t"] + 2.0 * jw["psi_t"]
    if c_new <= 0.0:
        # Only reachable at sigma = 0, the linear-smile case Section 3.1
        # excludes: there the assigned call wing collapses to zero.
        raise ValueError(
            f"Section 5.1 repair is infeasible: c_t' = p_t + 2*psi_t = {c_new:.6e} <= 0, "
            f"leaving no call wing to assign (sigma=0 is excluded by Section 3.1) "
            f"for SVI params {params}"
        )
    v_tilde_new = jw["v_t"] * 4.0 * jw["p_t"] * c_new / (jw["p_t"] + c_new) ** 2
    repaired = dict(jw, c_t=c_new, v_tilde_t=v_tilde_new)

    if validate:
        theta, phi_value, rho = svi_jw_to_ssvi_coordinates(repaired, T)
        check_ssvi_butterfly(
            np.array([theta]), SSVIParams(rho=rho, phi=lambda _theta, _p=phi_value: _p)
        )

    return svi_jw_to_raw(repaired, T)


def theorem_4_3_shift_can_help(params: SVIParams, T: float) -> bool:
    """
    Remark 5.1: the extra freedom of Theorem 4.3's shift alpha_t is only
    usable when alpha_t > 0, which "translates to the condition
    v_t*(1-rho^2) < v_tilde_t".

    Returns True when that holds for this slice. The remark's point is the
    negative case: it is violated for the Vogt smile of Example 5.1, so no
    Theorem 4.3 shift can repair that smile and the Section 5.1 wing/minimum
    -variance adjustment is the tool that has to be used instead.
    """
    jw = svi_jw_params(params, T)
    return bool(jw["v_t"] * (1.0 - params.rho ** 2) < jw["v_tilde_t"])


def check_theta_monotonic(t_grid, theta_values, tol=THETA_MONOTONIC_TOL):
    """
    Raise ArbitrageError if theta_values is not non-decreasing in t_grid.
    This is Theorem 4.1 condition (1): d(theta_t)/dt >= 0.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    theta_values = np.asarray(theta_values, dtype=float)
    order = np.argsort(t_grid)
    t_sorted, theta_sorted = t_grid[order], theta_values[order]

    diffs = np.diff(theta_sorted)
    if np.any(diffs < tol):
        bad = int(np.argmin(diffs))
        raise ArbitrageError(
            f"SSVI calendar arbitrage: theta_t decreases from {theta_sorted[bad]:.6f} "
            f"at t={t_sorted[bad]:.4f} to {theta_sorted[bad + 1]:.6f} at t={t_sorted[bad + 1]:.4f}"
        )
    return True


def check_ssvi_calendar(theta_grid, params: SSVIParams, tol=CALENDAR_TOL, d_theta_phi=None):
    """
    Raise ArbitrageError unless, for every theta in theta_grid (theta > 0):
        0 <= d/dtheta(theta*phi(theta)) <= (1/rho^2)*(1+sqrt(1-rho^2))*phi(theta)
    (upper bound is +inf when rho == 0).

    Computes d/dtheta(theta*phi(theta)) exactly when the shape function
    carries a `d_theta_phi` attribute -- both `phi_heston_like` and
    `phi_power_law` attach the closed forms the paper gives in Examples 4.1
    and 4.2 -- and by central finite differences on the (sorted) theta_grid
    otherwise. Pass `d_theta_phi` explicitly to override either.

    This is Theorem 4.1 condition (2). Condition (1), theta_t monotonic in
    t, must be checked separately by the caller against the actual term
    structure -- see `check_theta_monotonic`.
    """
    theta_grid = np.sort(np.asarray(theta_grid, dtype=float))
    rho = params.rho

    if d_theta_phi is None:
        d_theta_phi = getattr(params.phi, "d_theta_phi", None)

    if d_theta_phi is not None:
        d = np.asarray([d_theta_phi(theta) for theta in theta_grid], dtype=float)
    else:
        theta_phi = theta_grid * params.phi(theta_grid)
        d = np.gradient(theta_phi, theta_grid)

    if np.any(d < -tol):
        bad = int(np.argmin(d))
        raise ArbitrageError(
            f"SSVI calendar arbitrage: d/dtheta(theta*phi(theta)) = {d[bad]:.4e} < 0 "
            f"at theta={theta_grid[bad]:.4f}"
        )

    if abs(rho) > 1e-12:
        upper = (1.0 / rho ** 2) * (1.0 + np.sqrt(1.0 - rho ** 2)) * params.phi(theta_grid)
        violation = d > upper + tol
        if np.any(violation):
            bad = int(np.argmax(d - upper))
            raise ArbitrageError(
                f"SSVI calendar arbitrage: d/dtheta(theta*phi(theta)) = {d[bad]:.4e} exceeds "
                f"bound {upper[bad]:.4e} at theta={theta_grid[bad]:.4f} (rho={rho})"
            )
    return True


def check_ssvi_butterfly(theta_grid, params: SSVIParams, tol=BUTTERFLY_TOL):
    """
    Raise ArbitrageError unless, for every theta in theta_grid (theta > 0):
        1. theta * phi(theta) * (1 + |rho|) < 4
        2. theta * phi(theta)^2 * (1 + |rho|) <= 4
    These are Theorem 4.2's sufficient conditions for the SSVI slice at
    that theta to be free of butterfly arbitrage. Condition 1 is also
    necessary (Lemma 4.2) and equivalent to Lee's moment-formula bound on
    wing slope (Remark 4.3).
    """
    theta_grid = np.asarray(theta_grid, dtype=float)
    rho_term = 1.0 + abs(params.rho)
    phi_vals = params.phi(theta_grid)

    cond1 = theta_grid * phi_vals * rho_term
    if np.any(cond1 >= 4.0 - tol):
        bad = int(np.argmax(cond1))
        raise ArbitrageError(
            f"SSVI butterfly arbitrage: theta*phi(theta)*(1+|rho|) = {cond1[bad]:.4f} >= 4 "
            f"at theta={theta_grid[bad]:.4f} (Theorem 4.2 condition 1 / Lemma 4.2)"
        )

    cond2 = theta_grid * phi_vals ** 2 * rho_term
    if np.any(cond2 > 4.0 + tol):
        bad = int(np.argmax(cond2))
        raise ArbitrageError(
            f"SSVI butterfly arbitrage: theta*phi(theta)^2*(1+|rho|) = {cond2[bad]:.4f} > 4 "
            f"at theta={theta_grid[bad]:.4f} (Theorem 4.2 condition 2)"
        )
    return True


def check_ssvi_butterfly_jw(theta_grid, T, params: SSVIParams, tol=BUTTERFLY_TOL):
    """
    Remark 4.2: Theorem 4.2's two conditions re-expressed in SVI-JW terms.
    A SSVI volatility surface is free of butterfly arbitrage if

        sqrt(v_t * t) * max(p_t, c_t) < 2,  and  (p_t + c_t) * max(p_t, c_t) <= 2

    hold for all t > 0. Equivalent to `check_ssvi_butterfly` (the remark
    derives one from the other via Lemma 4.1); provided because these are
    the coordinates a trading desk reads the smile in.
    """
    theta_grid = np.asarray(theta_grid, dtype=float)
    for theta in theta_grid:
        jw = ssvi_jw_params(float(theta), T, params)
        wing = max(jw["p_t"], jw["c_t"])
        cond1 = np.sqrt(jw["v_t"] * T) * wing
        if cond1 >= 2.0 - tol:
            raise ArbitrageError(
                f"SSVI butterfly arbitrage (Remark 4.2): sqrt(v_t*t)*max(p_t,c_t) = "
                f"{cond1:.4f} >= 2 at theta={theta:.4f}"
            )
        cond2 = (jw["p_t"] + jw["c_t"]) * wing
        if cond2 > 2.0 + tol:
            raise ArbitrageError(
                f"SSVI butterfly arbitrage (Remark 4.2): (p_t+c_t)*max(p_t,c_t) = "
                f"{cond2:.4f} > 2 at theta={theta:.4f}"
            )
    return True


def check_ssvi_no_static_arbitrage(
    t_grid, theta_values, params: SSVIParams, k_grid=DEFAULT_K_GRID, cross_validate=True
):
    """
    Full Corollary 4.1 check: runs check_theta_monotonic,
    check_ssvi_calendar, and check_ssvi_butterfly over theta_values.

    If cross_validate=True, additionally converts the SSVI slice at each
    theta to raw SVI via ssvi_slice_to_raw_svi and runs arbitrage.py's
    existing check_butterfly (grid-based g(k) check) as an independent
    numerical sanity check. Since Theorem 4.2's conditions are only
    sufficient (not necessary, except at the condition-1 boundary per
    Lemma 4.2), the closed-form check passing should always imply the
    numerical check also passes -- if they disagree in that direction,
    raises `SSVIConsistencyError` since it indicates a bug rather than a
    real edge case.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    theta_values = np.asarray(theta_values, dtype=float)
    order = np.argsort(t_grid)
    theta_sorted = theta_values[order]

    check_theta_monotonic(t_grid, theta_values)
    check_ssvi_calendar(theta_sorted, params)
    check_ssvi_butterfly(theta_sorted, params)

    if cross_validate:
        for theta in theta_sorted:
            raw_params = ssvi_slice_to_raw_svi(float(theta), params)
            try:
                check_butterfly(raw_params, k_grid=k_grid)
            except ArbitrageError as exc:
                raise SSVIConsistencyError(
                    f"closed-form SSVI check passed at theta={theta:.6f} but the numerical "
                    f"raw-SVI check_butterfly disagreed: {exc}"
                ) from exc

    return True


def apply_theorem_4_3_shift(theta_grid, params: SSVIParams, alpha: Callable[[float], float]):
    """
    Verify alpha is non-negative and non-decreasing on (sorted) theta_grid
    (raise ArbitrageError if not -- these are Theorem 4.3's hypotheses on
    alpha as a function of the same index used to order theta_grid).

    Does not itself modify SSVIParams (alpha only shifts total variance,
    not the SSVI parameterization) -- instead returns a small closure
        w_alpha(k, theta_t) = ssvi_total_variance(k, theta_t, params) + alpha(theta_t)
    that callers can evaluate directly.

    IMPORTANT: this function does NOT check that the base surface
    satisfies Corollary 4.1, and does not itself assert the shifted
    surface is arbitrage-free -- per Theorem 4.3, that conclusion only
    holds *if* the base SSVI surface already satisfies Corollary 4.1.
    Shifting an already-arbitrageable base surface up by a non-negative,
    non-decreasing alpha does not make it arbitrage-free; callers who need
    that guarantee must check the base surface themselves (e.g. via
    `check_ssvi_no_static_arbitrage`) before relying on this shift.
    """
    theta_grid = np.sort(np.asarray(theta_grid, dtype=float))
    alpha_vals = np.asarray([alpha(theta) for theta in theta_grid], dtype=float)

    if np.any(alpha_vals < 0):
        bad = int(np.argmin(alpha_vals))
        raise ArbitrageError(
            f"Theorem 4.3 hypothesis violated: alpha({theta_grid[bad]:.4f}) = "
            f"{alpha_vals[bad]:.4e} < 0"
        )
    if np.any(np.diff(alpha_vals) < 0):
        bad = int(np.argmin(np.diff(alpha_vals)))
        raise ArbitrageError(
            f"Theorem 4.3 hypothesis violated: alpha decreases from "
            f"{alpha_vals[bad]:.4e} at theta={theta_grid[bad]:.4f} to "
            f"{alpha_vals[bad + 1]:.4e} at theta={theta_grid[bad + 1]:.4f}"
        )

    def w_alpha(k, theta_t):
        return ssvi_total_variance(k, theta_t, params) + alpha(theta_t)

    return w_alpha

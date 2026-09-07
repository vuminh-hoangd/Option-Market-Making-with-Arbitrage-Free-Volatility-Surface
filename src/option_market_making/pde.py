"""
Finite-difference solution of the EXACT HJB equation (eq. 6), as a validation
gate on the closed-form quotes.

Everything else in this package solves eq. 7 -- the paper's second-order Taylor
expansion of the two exponential sup-terms in eq. 6 -- which is what makes the
closed form possible at all. Footnote 5 of the paper concedes those quotes are
"not theoretically optimal, due to the second-order approximation used" and
defers the check against the exact PDE to an extended preprint we do not have.
This module runs that check ourselves.

The equation solved here, in this package's names:

    d_t h + (sigma^2 - sigma_imp^2)/2 * Gamma_dollar(t,s) * q - beta q^2
          + sigma^2/2 * s^2 * d_ss h
          + (lambda0_a e^-1 / kappa_a) exp(-kappa_a (h(t,s,q) - h(t,s,q-1)))
          + (lambda0_b e^-1 / kappa_b) exp(-kappa_b (h(t,s,q) - h(t,s,q+1))) = 0

    h(T, s, q) = -alpha q^2

Coupling across q is algebraic, not differential -- no q-derivatives appear, so
inventory is a set of coupled 1-D problems in s rather than a second spatial
dimension.

This is a DIAGNOSTIC tool. `quoting.py` and `sim.py` keep using the closed form
whatever this says; the outcome informs how far to trust it, not which code path
runs. See `# TODO (follow-up)` in `quoting.py`, which this module discharges.

Scheme
------
Backward march from the terminal condition, with the diffusion implicit and the
reaction/coupling terms explicit at the later time level, so each step is a
linear tridiagonal solve per q rather than a nonlinear system.

Two deliberate departures from a naive reading of the scheme:

1. **Time-derivative sign.** With t_N = T and t increasing in n,
   d_t h ~ (h^{n+1} - h^n)/dt. Writing it the other way round flips the sign of
   the diffusion term and makes the march anti-diffusive and violently unstable.
   The step below is therefore
       (I - dt D) h^n = h^{n+1} + dt f^{n+1},
   which is backward Euler for a backward-parabolic equation and unconditionally
   stable in the diffusion (see `_diffusion_solver` on the M-matrix property).

2. **Log-spot grid.** In x = ln s the diffusion operator sigma^2/2 s^2 d_ss
   becomes the constant-coefficient sigma^2/2 (d_xx - d_x), so the tridiagonal
   matrix is the same at every node and every time step and can be factorized
   exactly once for the whole march. It also puts uniform resolution where the
   dollar gamma actually lives, around the money.

Stability of the EXPLICIT coupling is a real constraint and is not
unconditional: the reaction Jacobian is about
lambda0_a e^-1 + lambda0_b e^-1 ~ 2e4/yr at the solution, and it grows like
exp(kappa |psi2| 2 q_max) at the edge of the inventory grid. Keep q_max modest
(15-20 at these parameters) and dt below ~1e-4 yr. `solve_hjb_pde` raises rather
than returning garbage if the march diverges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import csc_matrix, diags, identity
from scipy.sparse.linalg import splu

from . import bs_mm
from .quoting import MMParams


@dataclass(frozen=True)
class PDESolution:
    """
    The value function on the grid, plus everything needed to query it.

    `h` has shape (n_snapshots, n_s, n_q). Only a subset of time levels is
    retained -- a full march at the step counts this scheme needs would be
    hundreds of megabytes and nothing downstream wants it.
    """

    t: np.ndarray        # (n_snapshots,) ascending, t[0] = 0, t[-1] = T
    s: np.ndarray        # (n_s,) log-uniform spot grid
    q: np.ndarray        # (n_q,) consecutive integers, -q_max .. q_max
    h: np.ndarray        # (n_snapshots, n_s, n_q)
    K: float
    tau: float
    T: float
    sigma_imp: float
    params: MMParams

    @property
    def q_max(self) -> int:
        return int(self.q[-1])

    def value_at(self, t: float, s: float) -> np.ndarray:
        """h(t, s, .) over the whole q grid, linear in t and in log s."""
        return _bilinear(self.t, np.log(self.s), self.h, float(t), float(np.log(s)))

    def spreads_at(self, t: float, s: float, q: int) -> tuple[float, float]:
        """
        (delta_b, delta_a) read off the value function via eq. 5:

            delta_b = 1/kappa_b + h(t,s,q) - h(t,s,q+1)
            delta_a = 1/kappa_a + h(t,s,q) - h(t,s,q-1)

        Needs q+1 and q-1 on the grid, so |q| must be strictly inside q_max.
        """
        q = int(q)
        if abs(q) >= self.q_max:
            raise ValueError(
                f"q={q} needs its neighbours q+-1 on the grid, but q_max={self.q_max}; "
                f"widen q_max or query a smaller |q|"
            )
        h_q = self.value_at(t, s)
        i = q + self.q_max
        return (
            1.0 / self.params.kappa_b + h_q[i] - h_q[i + 1],
            1.0 / self.params.kappa_a + h_q[i] - h_q[i - 1],
        )

    def spreads_batch(self, t: float, spots: np.ndarray, qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        `spreads_at` for a whole vector of (spot, inventory) states at one time.

        Exists so the exact policy can actually be simulated: `sim.simulate`
        needs a quote for every path at every step, and the scalar path would be
        millions of interpolations. Inventory outside the grid interior is
        clipped to it -- a diagnostic run that wanders past q_max is telling you
        the grid is too narrow, and `n_clipped` on the returned arrays is not
        tracked, so check the simulated |q| against q_max yourself.
        """
        spots = np.asarray(spots, dtype=float)
        qs = np.asarray(qs, dtype=int)

        x_grid = np.log(self.s)
        x = np.clip(np.log(spots), x_grid[0], x_grid[-1])

        # Interpolate the time slice once, then every spot at once.
        j = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        wt = (t - self.t[j]) / (self.t[j + 1] - self.t[j])
        slab = self.h[j] * (1.0 - wt) + self.h[j + 1] * wt      # (n_s, n_q)

        i = np.clip(np.searchsorted(x_grid, x) - 1, 0, len(x_grid) - 2)
        wx = (x - x_grid[i]) / (x_grid[i + 1] - x_grid[i])
        h_here = slab[i] * (1.0 - wx)[:, None] + slab[i + 1] * wx[:, None]   # (n, n_q)

        idx = np.clip(qs + self.q_max, 1, 2 * self.q_max - 1)
        rows = np.arange(len(spots))
        h_q = h_here[rows, idx]
        return (
            1.0 / self.params.kappa_b + h_q - h_here[rows, idx + 1],
            1.0 / self.params.kappa_a + h_q - h_here[rows, idx - 1],
        )



def _bilinear(t_grid: np.ndarray, x_grid: np.ndarray, values: np.ndarray,
              t: float, x: float) -> np.ndarray:
    """Linear interpolation in the first two axes of `values`, leaving the rest."""
    if not (t_grid[0] - 1e-12 <= t <= t_grid[-1] + 1e-12):
        raise ValueError(f"t={t} outside the solved horizon [{t_grid[0]}, {t_grid[-1]}]")
    if not (x_grid[0] <= x <= x_grid[-1]):
        raise ValueError(
            f"log-spot {x:.4f} outside the grid [{x_grid[0]:.4f}, {x_grid[-1]:.4f}]; "
            f"widen log_s_halfwidth"
        )

    def weights(grid: np.ndarray, v: float) -> tuple[int, float]:
        j = int(np.clip(np.searchsorted(grid, v) - 1, 0, len(grid) - 2))
        return j, (v - grid[j]) / (grid[j + 1] - grid[j])

    j, wt = weights(t_grid, t)
    i, wx = weights(x_grid, x)
    lo = values[j, i] * (1 - wx) + values[j, i + 1] * wx
    hi = values[j + 1, i] * (1 - wx) + values[j + 1, i + 1] * wx
    return lo * (1 - wt) + hi * wt


def _diffusion_solver(x: np.ndarray, sigma: float, dt: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factorize (I - dt D) once, where D is sigma^2/2 (d_xx - d_x), the log-space
    form of sigma^2/2 s^2 d_ss.

    Returns a callable taking a right-hand side of shape (n_s, n_q) and solving
    every inventory column at once -- the matrix does not depend on q or on t,
    so one factorization serves the entire backward march.

    The boundary rows are the identity (diffusion switched off at the two ends,
    i.e. d_ss h = 0 there, the Neumann-type condition). The interior rows are
    diagonally dominant with positive diagonal for any dx < 2, so the matrix is
    an M-matrix and the implicit step is unconditionally stable.
    """
    dx = float(x[1] - x[0])
    n = len(x)
    half_var = 0.5 * sigma ** 2

    sub = half_var * (1.0 / dx ** 2 + 1.0 / (2.0 * dx))
    diag = half_var * (-2.0 / dx ** 2)
    sup = half_var * (1.0 / dx ** 2 - 1.0 / (2.0 * dx))

    D = diags(
        [np.full(n - 1, sub), np.full(n, diag), np.full(n - 1, sup)],
        offsets=[-1, 0, 1], format="lil",
    )
    D[0, :] = 0.0          # d_ss h = 0 at both ends
    D[-1, :] = 0.0

    matrix = csc_matrix(identity(n, format="csc") - dt * D)
    factorized = splu(matrix)
    return factorized.solve


def _pad_inventory(h: np.ndarray) -> np.ndarray:
    """
    Extend h by one inventory level at each end by linear extrapolation in q,
    so the coupling terms have h(q-1) and h(q+1) everywhere on the grid.

    Linear (not quadratic) on purpose: quadratic extrapolation would be exact
    for the closed form's own ansatz and would quietly build the answer into the
    boundary. The `q_max`-widening convergence check is what licenses the grid.
    """
    left = 2.0 * h[:, :1] - h[:, 1:2]
    right = 2.0 * h[:, -1:] - h[:, -2:-1]
    return np.concatenate([left, h, right], axis=1)


def solve_hjb_pde(
    K: float,
    tau: float,
    T: float,
    params: MMParams,
    sigma_imp_val: float,
    *,
    s_center: float,
    n_t: int = 800,
    n_s: int = 201,
    q_max: int = 15,
    log_s_halfwidth: float = 0.6,
    n_snapshots: int = 41,
) -> PDESolution:
    """
    Solve eq. 6 backward from h(T,s,q) = -alpha q^2 to t = 0.

    `s_center` is placed exactly on a grid node (n_s is forced odd), so queries
    at the contract's spot need no interpolation in s.

    Raises if the march produces a non-finite value -- with the coupling handled
    explicitly that means dt is too large or q_max too wide, and a silently
    returned NaN grid would be worse than a stack trace.
    """
    if n_t < 1 or n_s < 5 or q_max < 2:
        raise ValueError(f"grid too small: n_t={n_t}, n_s={n_s}, q_max={q_max}")
    if T <= 0.0 or T >= tau:
        raise ValueError(f"need 0 < T < tau, got T={T}, tau={tau}")
    if s_center <= 0.0:
        raise ValueError(f"s_center must be positive, got {s_center}")

    n_s = n_s if n_s % 2 == 1 else n_s + 1          # keep s_center on a node
    x = np.log(s_center) + np.linspace(-log_s_halfwidth, log_s_halfwidth, n_s)
    s = np.exp(x)
    q = np.arange(-q_max, q_max + 1)
    dt = T / n_t

    solve = _diffusion_solver(x, params.sigma, dt)

    # Terminal condition, and the time levels we keep.
    h = -params.alpha * np.repeat(q[None, :] ** 2.0, n_s, axis=0)
    store_levels = np.unique(np.linspace(0, n_t, n_snapshots).astype(int))
    snapshots = {int(n_t): h.copy()}

    variance_spread = 0.5 * (params.sigma ** 2 - sigma_imp_val ** 2)
    coeff_a = params.lambda0_a * np.exp(-1.0) / params.kappa_a
    coeff_b = params.lambda0_b * np.exp(-1.0) / params.kappa_b
    running_penalty = params.beta * q[None, :] ** 2.0

    for n in range(n_t - 1, -1, -1):
        t_next = (n + 1) * dt

        # Source, all evaluated at the later (known) time level.
        gamma_dollar = bs_mm.dollar_gamma_array(s, K, tau - t_next, sigma_imp_val)
        drift = variance_spread * gamma_dollar[:, None] * q[None, :]

        padded = _pad_inventory(h)
        coupling = (
            coeff_a * np.exp(-params.kappa_a * (h - padded[:, :-2]))
            + coeff_b * np.exp(-params.kappa_b * (h - padded[:, 2:]))
        )

        h = solve(h + dt * (drift - running_penalty + coupling))

        if not np.all(np.isfinite(h)):
            raise ValueError(
                f"HJB march diverged at time level {n} (t={n * dt:.3e}). The coupling "
                f"terms are explicit, so this means dt={dt:.2e} is too large or "
                f"q_max={q_max} too wide -- the reaction Jacobian grows like "
                f"exp(kappa |psi2| 2 q_max). Raise n_t or lower q_max."
            )
        if n in store_levels:
            snapshots[n] = h.copy()

    levels = sorted(snapshots)
    return PDESolution(
        t=np.array(levels, dtype=float) * dt,
        s=s, q=q,
        h=np.stack([snapshots[n] for n in levels]),
        K=K, tau=tau, T=T, sigma_imp=sigma_imp_val, params=params,
    )
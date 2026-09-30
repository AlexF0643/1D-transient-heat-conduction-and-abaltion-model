"""1D transient heat conduction solver.

Uniform-grid, theta-method (generalized trapezoidal) finite-difference
solver for

    rho*cp * dT/dt = d/dx( k * dT/dx )

with constant material properties, so alpha = k/(rho*cp) and

    dT/dt = alpha * d2T/dx2

theta = 1.0   -> fully implicit / backward Euler (unconditionally
                 stable, first-order accurate in time)
theta = 0.5   -> Crank-Nicolson (unconditionally stable, second-order
                 accurate in time)
theta = 0.0   -> forward Euler / fully explicit (conditionally stable;
                 supported for completeness, not recommended)

Each end of the domain takes a DirichletBC, FluxBC, or
ConvectiveRadiativeBC (see boundary.py). Nonlinear (Robin) boundaries
are closed with a Newton iteration on the surface temperature at each
time step: the boundary flux is linearized about the current surface
temperature (using the BC's analytic dq/dTs), which keeps the
iteration convergent on grids and heating levels where a plain
fixed-point (Picard) iteration diverges. The converged solution is the
same nonlinear-implicit solution either way.

Spatial discretization error is O(dx^2); a boundary closed with a
FluxBC or ConvectiveRadiativeBC uses a ghost-node central difference
so it remains second order in space, matching the interior scheme.
"""
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .boundary import ConvectiveRadiativeBC, DirichletBC, FluxBC
from .material import Material


def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system Ax = d given sub/main/super diagonals.

    a[0] and c[-1] are unused (kept for indexing convenience, must be 0).
    """
    n = len(d)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


@dataclass
class HeatConduction1D:
    material: Material
    length: float
    n_nodes: int
    left_bc: object
    right_bc: object
    theta: float = 1.0
    # Number of leading backward-Euler steps before switching to `theta`
    # (Rannacher startup). Restores Crank-Nicolson's 2nd-order accuracy
    # when the initial/boundary data are non-smooth; ignored when
    # theta == 1.0.
    rannacher_steps: int = 0

    def __post_init__(self):
        self.dx = self.length / (self.n_nodes - 1)
        self.x = np.linspace(0.0, self.length, self.n_nodes)

    # -- boundary row assembly -------------------------------------------------
    def _left_row(self, r, t_np1, t_n, T_n, T0_guess):
        theta = self.theta
        k = self.material.k
        dx = self.dx
        bc = self.left_bc
        if isinstance(bc, DirichletBC):
            # b0*T0 = rhs, no coupling to T1
            return 1.0, 0.0, bc.value(t_np1), None
        if isinstance(bc, FluxBC):
            q_n = bc.value(t_n)
            q_np1 = bc.value(t_np1)
        elif isinstance(bc, ConvectiveRadiativeBC):
            q_n = bc.net_flux(t_n, T_n[0])
            q_np1 = bc.net_flux(t_np1, T0_guess)
            dq = bc.dflux_dTs(t_np1, T0_guess)
        else:
            raise TypeError(f"Unsupported left BC type: {type(bc)}")
        b0 = 1.0 + 2.0 * r * theta
        c0 = -2.0 * r * theta
        rhs = (
            T_n[0]
            + 2.0 * r * (1.0 - theta) * (T_n[1] - T_n[0])
            + 2.0 * r * dx / k * ((1.0 - theta) * q_n + theta * q_np1)
        )
        if isinstance(bc, ConvectiveRadiativeBC):
            # Newton: q(T0) ~ q(T0_guess) + dq*(T0 - T0_guess); move the
            # unknown-T0 part of the flux term to the left-hand side.
            g = 2.0 * r * dx / k * theta * dq
            b0 -= g
            rhs -= g * T0_guess
        return b0, c0, rhs, q_np1

    def _right_row(self, r, t_np1, t_n, T_n, TN_guess):
        theta = self.theta
        k = self.material.k
        dx = self.dx
        bc = self.right_bc
        N = self.n_nodes
        if isinstance(bc, DirichletBC):
            # returned as (sub-diagonal, diagonal, rhs) to match _assemble's
            # a[-1], b[-1], d[-1] = ... convention
            return 0.0, 1.0, bc.value(t_np1), None
        if isinstance(bc, FluxBC):
            q_n = bc.value(t_n)
            q_np1 = bc.value(t_np1)
        elif isinstance(bc, ConvectiveRadiativeBC):
            q_n = bc.net_flux(t_n, T_n[-1])
            q_np1 = bc.net_flux(t_np1, TN_guess)
            dq = bc.dflux_dTs(t_np1, TN_guess)
        else:
            raise TypeError(f"Unsupported right BC type: {type(bc)}")
        bN = 1.0 + 2.0 * r * theta
        aN = -2.0 * r * theta
        rhs = (
            T_n[-1]
            + 2.0 * r * (1.0 - theta) * (T_n[-2] - T_n[-1])
            + 2.0 * r * dx / k * ((1.0 - theta) * q_n + theta * q_np1)
        )
        if isinstance(bc, ConvectiveRadiativeBC):
            g = 2.0 * r * dx / k * theta * dq
            bN -= g
            rhs -= g * TN_guess
        return aN, bN, rhs, q_np1

    def _assemble(self, dt, t_np1, t_n, T_n, T0_guess, TN_guess):
        N = self.n_nodes
        alpha = self.material.alpha
        r = alpha * dt / self.dx**2
        theta = self.theta

        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        d = np.zeros(N)

        # interior nodes
        a[1:-1] = -r * theta
        b[1:-1] = 1.0 + 2.0 * r * theta
        c[1:-1] = -r * theta
        d[1:-1] = T_n[1:-1] + r * (1.0 - theta) * (T_n[:-2] - 2.0 * T_n[1:-1] + T_n[2:])

        b0, c0, rhs0, qL = self._left_row(r, t_np1, t_n, T_n, T0_guess)
        b[0], c[0], d[0] = b0, c0, rhs0

        aN, bN, rhsN, qR = self._right_row(r, t_np1, t_n, T_n, TN_guess)
        a[-1], b[-1], d[-1] = aN, bN, rhsN

        return a, b, c, d

    def _is_nonlinear(self) -> bool:
        return isinstance(self.left_bc, ConvectiveRadiativeBC) or isinstance(
            self.right_bc, ConvectiveRadiativeBC
        )

    def step(self, T_n: np.ndarray, t_n: float, dt: float, max_iter: int = 30, tol: float = 1e-8) -> np.ndarray:
        """Advance one time step, returning T at t_n + dt."""
        t_np1 = t_n + dt
        T0_guess = T_n[0]
        TN_guess = T_n[-1]

        if not self._is_nonlinear():
            a, b, c, d = self._assemble(dt, t_np1, t_n, T_n, T0_guess, TN_guess)
            return thomas_solve(a, b, c, d)

        T_np1 = T_n.copy()
        for _ in range(max_iter):
            a, b, c, d = self._assemble(dt, t_np1, t_n, T_n, T0_guess, TN_guess)
            T_np1 = thomas_solve(a, b, c, d)
            new_T0, new_TN = T_np1[0], T_np1[-1]
            err = max(abs(new_T0 - T0_guess), abs(new_TN - TN_guess))
            T0_guess, TN_guess = new_T0, new_TN
            if err < tol:
                break
        else:
            warnings.warn(
                f"Newton iteration on the surface temperature did not converge in {max_iter} "
                f"iterations (last change {err:.2e} K) at t = {t_np1:.6g} s; the step result "
                "may be inaccurate. Try a smaller time step or finer grid.",
                RuntimeWarning,
                stacklevel=2,
            )
        return T_np1

    def solve(self, T_initial: np.ndarray, dt: float, t_end: float, max_iter: int = 30, tol: float = 1e-8):
        """March from t=0 to t_end with fixed step dt.

        Returns (t, T) with t of shape (n_steps+1,) and T of shape
        (n_steps+1, n_nodes).
        """
        n_steps = int(round(t_end / dt))
        t = np.linspace(0.0, n_steps * dt, n_steps + 1)
        T = np.empty((n_steps + 1, self.n_nodes))
        T[0] = T_initial
        theta_main = self.theta
        try:
            for n in range(n_steps):
                self.theta = 1.0 if n < self.rannacher_steps else theta_main
                T[n + 1] = self.step(T[n], t[n], dt, max_iter=max_iter, tol=tol)
        finally:
            self.theta = theta_main
        return t, T

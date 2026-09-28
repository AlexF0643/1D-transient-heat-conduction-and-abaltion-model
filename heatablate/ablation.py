"""1D conduction with a receding ablative front.

Models a slab whose exposed surface starts at x=0 and recedes into the
material as x=s(t) once the surface reaches `material.ablation_temperature`.
The remaining (unablated) material occupies the physical domain
[s(t), length]; the back face at x=length is fixed.

Numerics: at each time step, the conduction sub-problem is solved on the
CURRENT grid [s^n, length] with `HeatConduction1D` (reused unmodified —
the diffusion PDE is translation-invariant in x, so only the domain
*length* `length - s^n` matters, not the absolute offset). The front
recession rate is then set by a surface energy balance (the classical
Stefan condition): once ablating, the surface is held at
`ablation_temperature`, and any incident flux in excess of what
conducts into the remaining material converts to recession, absorbing
`heat_of_ablation`:

    ds/dt = max(0, q_incident(t, T_ablation) - q_conducted_into_solid)
            / (rho * heat_of_ablation)

The grid is then rebuilt over the new (shrunk) domain [s^{n+1}, length]
and the temperature field is linearly interpolated onto it (a simple
moving-mesh / ALE scheme; the interpolation is always strictly within
the old domain since the domain only shrinks).

This is deliberately simpler than a full boundary-immobilized
(Landau-transformed) PDE with a mesh-velocity advection term — a valid
alternative, but more numerically involved for the accuracy this project
scope needs. See docs/PROJECT_PLAN.md Phase 3.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .boundary import ConvectiveRadiativeBC, DirichletBC, FluxBC
from .material import Material
from .solver import HeatConduction1D


@dataclass
class AblationFront1D:
    material: Material
    length: float
    n_nodes: int
    surface_bc: object  # FluxBC or ConvectiveRadiativeBC; drives pre-ablation heating and (evaluated at T_ablation) the post-onset recession
    back_bc: Optional[object] = None
    theta: float = 1.0

    def __post_init__(self):
        if self.back_bc is None:
            self.back_bc = FluxBC(0.0)
        if not isinstance(self.surface_bc, (FluxBC, ConvectiveRadiativeBC)):
            raise TypeError(
                "AblationFront1D.surface_bc must be a FluxBC or ConvectiveRadiativeBC "
                "(a Dirichlet surface has no well-defined incident flux to drive recession)"
            )

    def _incident_flux(self, t, Ts):
        if isinstance(self.surface_bc, FluxBC):
            return self.surface_bc.value(t)
        return self.surface_bc.net_flux(t, Ts)

    def solve(self, T_initial: np.ndarray, dt: float, t_end: float, t_start: float = 0.0, s_initial: float = 0.0):
        """March from t_start to t_end. Returns a dict with time series
        for front position and back-face temperature, plus the final
        profile.
        """
        n = self.n_nodes
        Tm = self.material.ablation_temperature
        rho = self.material.rho
        Lh = self.material.heat_of_ablation

        t = t_start
        s = s_initial
        T = np.asarray(T_initial, dtype=float).copy()

        t_hist = [t]
        s_hist = [s]
        Tback_hist = [T[-1]]

        n_steps = int(round((t_end - t_start) / dt))
        for _ in range(n_steps):
            t_np1 = t + dt
            domain_len = self.length - s
            dx = domain_len / (n - 1)
            ablating = T[0] >= Tm

            if ablating:
                left_bc = DirichletBC(Tm)
            else:
                left_bc = self.surface_bc

            solver = HeatConduction1D(self.material, domain_len, n, left_bc, self.back_bc, theta=self.theta)
            T_new = solver.step(T, t, dt)

            if ablating:
                # one-sided 2nd-order gradient at the (moving) surface node
                dTdx = (-3.0 * T_new[0] + 4.0 * T_new[1] - T_new[2]) / (2.0 * dx)
                q_cond = -self.material.k * dTdx
                q_in = self._incident_flux(t_np1, Tm)
                ds_dt = max(0.0, q_in - q_cond) / (rho * Lh)
                s_new = s + ds_dt * dt

                x_old = s + solver.x
                x_new = np.linspace(s_new, self.length, n)
                T_new = np.interp(x_new, x_old, T_new)
                s = s_new
            else:
                if T_new[0] >= Tm:
                    T_new[0] = Tm  # clamp on the step that first reaches ablation temperature

            T = T_new
            t = t_np1
            t_hist.append(t)
            s_hist.append(s)
            Tback_hist.append(T[-1])

        x_final = s + np.linspace(0.0, self.length - s, n)
        return {
            "t": np.array(t_hist),
            "s": np.array(s_hist),
            "T_back": np.array(Tback_hist),
            "x_final": x_final,
            "T_final": T,
        }

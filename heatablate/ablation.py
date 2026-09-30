"""1D conduction with a receding ablative front.

Models a slab whose exposed surface starts at x=0 and recedes into the
material as x=s(t) once the surface reaches `material.ablation_temperature`.
The remaining (unablated) material occupies the physical domain
[s(t), length]; the back face at x=length is fixed.

Numerics: at each time step, the conduction sub-problem is solved on the
CURRENT grid [s^n, length] with `HeatConduction1D` (reused unmodified —
the diffusion PDE is translation-invariant in x, so only the domain
*length* `length - s^n` matters, not the absolute offset). The front
recession rate is set by a surface energy balance (the classical Stefan
condition): while ablating, the surface is held at `ablation_temperature`
(Dirichlet), and any incident flux in excess of what conducts into the
remaining material converts to recession, absorbing `heat_of_ablation`:

    ds/dt = (q_incident(t, T_ablation) - q_conducted_into_solid)
            / (rho * heat_of_ablation)

Each step this is checked for sustainability: the Dirichlet(Tm) trial
solve above is only *kept* if q_incident >= q_conducted (i.e. ds/dt would
be >= 0). If the incident flux can no longer sustain the ablation
temperature — typically once a heating pulse fades — the step is
re-solved with the ordinary (unclamped) surface_bc instead, letting the
surface cool below Tm; ablation genuinely stops until/unless the surface
reheats back up to Tm. This is deliberately a per-step *physical*
sustainability check rather than a bare `T[0] >= Tm` re-check: after
remeshing, linear interpolation can leave the surface a hair below Tm
even while comfortably sustained, which a naive re-check would
misinterpret as "stop ablating," chattering between the two branches
every other step (a real bug caught via the energy-balance diagnostic
below — see tests/test_ablation.py).

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

from ._compat import trapezoid
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
        for front position and back-face temperature, the final profile,
        and an `energy_balance` diagnostic (see module docstring / README
        for what each term means and why a small residual is expected).
        """
        n = self.n_nodes
        Tm = self.material.ablation_temperature
        rho = self.material.rho
        cp = self.material.cp
        Lh = self.material.heat_of_ablation
        T_ref = float(np.asarray(T_initial, dtype=float)[0])  # reference temp for enthalpy bookkeeping; exact for uniform IC

        t = t_start
        s = s_initial
        T = np.asarray(T_initial, dtype=float).copy()

        t_hist = [t]
        s_hist = [s]
        Tback_hist = [T[-1]]

        E_in = 0.0  # cumulative incident energy at the surface [J/m^2]
        E_ablated_latent = 0.0  # cumulative latent heat carried away by ablated mass
        E_ablated_sensible = 0.0  # cumulative sensible enthalpy (rel. T_ref) carried away by ablated mass

        # `ablating` is decided fresh each step from a physical
        # sustainability check, not just re-derived from raw T[0] >= Tm:
        # once ablating, the trial Dirichlet(Tm) solve is only *kept* if
        # the incident flux can actually sustain it (q_in >= q_cond); once
        # a fading heat pulse can no longer sustain the ablation
        # temperature, q_in < q_cond and the step is re-solved with the
        # ordinary (unclamped) surface_bc instead, letting the surface
        # cool below Tm — ablation genuinely stops. A naive re-check of
        # T[0] >= Tm each step is NOT equivalent to this: after remeshing,
        # linear interpolation can leave the surface a hair below Tm even
        # while still comfortably sustained, which would incorrectly drop
        # back to the unclamped branch and chatter between the two every
        # other step (silently skipping about half the recession physics
        # — this was a real, caught bug; see tests/test_ablation.py).
        ablating = bool(T[0] >= Tm)

        n_steps = int(round((t_end - t_start) / dt))
        for _ in range(n_steps):
            t_np1 = t + dt
            domain_len = self.length - s

            if ablating:
                solver = HeatConduction1D(self.material, domain_len, n, DirichletBC(Tm), self.back_bc, theta=self.theta)
                T_new = solver.step(T, t, dt)
                x_old = s + solver.x
                # Energy-consistent conducted flux: the enthalpy the current
                # (pre-removal) domain gained this step, divided by dt. A raw
                # one-sided FD gradient at the Dirichlet-clamped surface node
                # is a poor estimator here (>50% off right after the surface
                # first clamps, where the near-surface gradient is steep and
                # under-resolved) — this is exact by construction instead,
                # since it's literally what the implicit solve moved into the
                # domain, matching the same energy-balance identity verified
                # for the plain (non-ablating) solver.
                q_cond = rho * cp * trapezoid(T_new - T, x_old) / dt
                q_in = self._incident_flux(t_np1, Tm)
                ablating = q_in >= q_cond

            if not ablating:
                # either never started, or the trial Dirichlet solve above
                # (if any) couldn't be sustained by the incident flux —
                # solve this step with the real (unclamped) surface_bc.
                solver = HeatConduction1D(self.material, domain_len, n, self.surface_bc, self.back_bc, theta=self.theta)
                T_new = solver.step(T, t, dt)
                E_in += self._incident_flux(t_np1, T_new[0]) * dt
                if T_new[0] >= Tm:
                    T_new[0] = Tm  # clamp on the step that first reaches ablation temperature
                    ablating = True  # next step tries the Dirichlet/recession branch
            else:
                ds_dt = (q_in - q_cond) / (rho * Lh)
                s_new = s + ds_dt * dt

                T_at_s_new = np.interp(s_new, x_old, T_new)
                E_in += q_in * dt
                E_ablated_latent += rho * Lh * (s_new - s)
                E_ablated_sensible += rho * cp * 0.5 * ((T_new[0] - T_ref) + (T_at_s_new - T_ref)) * (s_new - s)

                x_new = np.linspace(s_new, self.length, n)
                T_new = np.interp(x_new, x_old, T_new)
                s = s_new

            T = T_new
            t = t_np1
            t_hist.append(t)
            s_hist.append(s)
            Tback_hist.append(T[-1])

        x_final = s + np.linspace(0.0, self.length - s, n)
        E_remaining_sensible = rho * cp * trapezoid(T - T_ref, x_final)
        residual = E_in - (E_remaining_sensible + E_ablated_latent + E_ablated_sensible)

        return {
            "t": np.array(t_hist),
            "s": np.array(s_hist),
            "T_back": np.array(Tback_hist),
            "x_final": x_final,
            "T_final": T,
            "energy_balance": {
                "E_in": E_in,
                "E_remaining_sensible": E_remaining_sensible,
                "E_ablated_latent": E_ablated_latent,
                "E_ablated_sensible": E_ablated_sensible,
                "residual": residual,
                "residual_pct": 100.0 * residual / E_in if E_in != 0 else 0.0,
            },
        }

"""Boundary condition models for the 1D slab.

Each BC exposes what the solver needs to build a ghost-node closure at
its end of the domain:

  - DirichletBC:           prescribed surface temperature T(t)
  - FluxBC:                prescribed flux into the domain q(t) [W/m^2]
  - ConvectiveRadiativeBC: aero-heating with radiative reradiation,
                            net flux is a nonlinear function of the
                            (unknown) surface temperature

Sign convention: flux q is positive when it flows INTO the slab
(increasing x at the left face, decreasing x at the right face).
"""

from __future__ import annotations  # `X | Y` annotations on Python 3.9
from dataclasses import dataclass, field
from typing import Callable, Optional

SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m^2 K^4)]


def _as_time_fn(value) -> Callable[[float], float]:
    if callable(value):
        return value
    return lambda t: value


@dataclass
class DirichletBC:
    """Prescribed surface temperature T(t)."""

    temperature: Callable[[float], float] | float

    def __post_init__(self):
        self._fn = _as_time_fn(self.temperature)

    kind = "dirichlet"

    def value(self, t: float) -> float:
        return self._fn(t)


@dataclass
class FluxBC:
    """Prescribed heat flux into the domain, q(t) [W/m^2]."""

    flux: Callable[[float], float] | float

    def __post_init__(self):
        self._fn = _as_time_fn(self.flux)

    kind = "flux"

    def value(self, t: float) -> float:
        return self._fn(t)


@dataclass
class ConvectiveRadiativeBC:
    """Aerodynamic convective heating with radiative reradiation.

    q_net(Ts, t) = h(t) * (T_aw(t) - Ts) - emissivity * sigma * (Ts^4 - T_inf^4)

    h: convective heat transfer coefficient [W/(m^2 K)]
    T_aw: adiabatic wall (recovery) temperature driving convection [K]
    emissivity: surface total hemispherical emissivity [-]
    T_inf: effective radiative background temperature [K]
    """

    h: Callable[[float], float] | float
    T_aw: Callable[[float], float] | float
    emissivity: float
    T_inf: float = 0.0
    sigma: float = SIGMA_SB

    def __post_init__(self):
        self._h_fn = _as_time_fn(self.h)
        self._Taw_fn = _as_time_fn(self.T_aw)

    kind = "robin"

    def net_flux(self, t: float, Ts: float) -> float:
        h = self._h_fn(t)
        T_aw = self._Taw_fn(t)
        q_conv = h * (T_aw - Ts)
        q_rad = self.emissivity * self.sigma * (Ts**4 - self.T_inf**4)
        return q_conv - q_rad

    def dflux_dTs(self, t: float, Ts: float) -> float:
        """Analytic d(q_net)/dTs, used for the Newton linearization."""
        h = self._h_fn(t)
        return -h - 4.0 * self.emissivity * self.sigma * Ts**3

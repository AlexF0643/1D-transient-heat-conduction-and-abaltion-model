from dataclasses import dataclass


@dataclass
class Material:
    """Constant thermophysical properties of a slab.

    k: thermal conductivity [W/(m K)]
    rho: density [kg/m^3]
    cp: specific heat [J/(kg K)]
    emissivity: total hemispherical emissivity [-], used by radiative BCs
    heat_of_ablation: effective heat of ablation [J/kg], used by the
        ablation front-recession model (Phase 3)
    ablation_temperature: temperature at which the surface begins to
        recede [K] (Phase 3)
    """

    k: float
    rho: float
    cp: float
    emissivity: float = 0.0
    heat_of_ablation: float = float("inf")
    ablation_temperature: float = float("inf")

    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m^2/s]."""
        return self.k / (self.rho * self.cp)

    @classmethod
    def from_diffusivity(cls, alpha: float, k: float, **kwargs) -> "Material":
        """Build a Material from a measured diffusivity and conductivity.

        Useful when alpha and k are direct lab measurements: rho*cp is
        recovered as k/alpha rather than specified independently.
        """
        rho_cp = k / alpha
        return cls(k=k, rho=1.0, cp=rho_cp, **kwargs)

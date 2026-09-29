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
    def from_diffusivity(cls, alpha: float, k: float, rho: float = 1.0, **kwargs) -> "Material":
        """Build a Material from a measured diffusivity and conductivity.

        Useful when alpha and k are direct lab measurements: rho*cp is
        recovered as k/alpha rather than specified independently.

        Pass the real density as `rho` (cp is then k/(alpha*rho)) whenever
        the ablation model is used: the recession rate divides by
        rho*heat_of_ablation, so the default rho=1 (fine for pure
        conduction, where only rho*cp matters) would badly overstate
        recession.
        """
        rho_cp = k / alpha
        return cls(k=k, rho=rho, cp=rho_cp / rho, **kwargs)

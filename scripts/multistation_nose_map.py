"""Phase 7: multi-station nose thermal map.

A true 2D/3D coupled ablation solve (curved mesh, level-set front
tracking) is out of scope for this project (see docs/PROJECT_PLAN.md
Phase 7) — and isn't how heat-shield sizing is actually done in
practice. Tools like NASA's FIAT are themselves 1D, run independently
at multiple body stations along the vehicle with locally-varying
heating input; this is standard because at hypersonic heating rates the
boundary layer is thin and through-thickness conduction dominates, so
surface-tangential conduction is usually negligible.

This script reproduces that approach: a blunted (spherical-cap) nose is
parametrized by the angle phi from the stagnation point, local heating
is scaled off the Phase 4 stagnation-point pulse by the standard
"cosine law" Newtonian-flow approximation

    h(phi, t) = h_stagnation(t) * cos(phi)

(a first-order engineering approximation for heat flux distribution
over a spherical cap — the more accurate Lees correlation adds a
boundary-layer-thinning correction near the stagnation point, but
cos(phi) captures the right qualitative falloff and is standard in
introductory hypersonic aerothermal analysis). The existing 1D
AblationFront1D solver is run independently at each station — no
coupling is needed between stations under the thin-boundary-layer
assumption, so this trivially parallelizes (run sequentially here for
simplicity).
"""
import numpy as np

from heatablate import Material, ConvectiveRadiativeBC, AblationFront1D
from heatablate.plotting import savefig

# Same material and heat pulse as scripts/mission_ablation.py (duplicated
# rather than imported — see scripts/convergence_study_ablation.py for why).
K = 0.5  # W/(m K)
RHO = 1400.0  # kg/m^3
CP = 1200.0  # J/(kg K)
EMISSIVITY = 0.85
HEAT_OF_ABLATION = 8.0e6  # J/kg
ABLATION_TEMPERATURE = 2200.0  # K
LENGTH = 0.018  # m
T0 = 300.0  # K
H_PEAK = 800.0  # W/(m^2 K), at the stagnation point (phi=0)
T_PEAK = 30.0  # s
T_AW = 6000.0  # K, treated as uniform across the nose cap (a simplification —
                # the dominant spatial variation is in h, not T_aw)
T_END = 120.0  # s
N_NODES = 201
DT = 0.02  # s, per the Phase 5 convergence study (0.24% error at N=201)

PHI_DEG = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])


def h_stagnation(t):
    tau = np.asarray(t, dtype=float) / T_PEAK
    return H_PEAK * tau * np.exp(1.0 - tau)


def run_station(phi_deg):
    scale = np.cos(np.deg2rad(phi_deg))

    def h_local(t):
        return scale * h_stagnation(t)

    mat = Material(k=K, rho=RHO, cp=CP, emissivity=EMISSIVITY,
                    heat_of_ablation=HEAT_OF_ABLATION, ablation_temperature=ABLATION_TEMPERATURE)
    bc = ConvectiveRadiativeBC(h=h_local, T_aw=T_AW, emissivity=EMISSIVITY, T_inf=0.0)
    model = AblationFront1D(mat, LENGTH, N_NODES, bc, theta=1.0)
    result = model.solve(np.full(N_NODES, T0), DT, T_END)
    return result


if __name__ == "__main__":
    print("Phase 7: multi-station nose thermal map")
    print(f"  {len(PHI_DEG)} stations, phi = {list(PHI_DEG)} deg from stagnation point")
    print(f"  h(phi,t) = h_stag(t) * cos(phi), h_stag peak = {H_PEAK} W/m2K at t={T_PEAK}s")
    print()

    Tback = []
    recession = []
    for phi in PHI_DEG:
        result = run_station(phi)
        Tb = result["T_back"][-1]
        s = result["s"][-1]
        Tback.append(Tb)
        recession.append(s)
        ablated = "ablated" if s > 0 else "no ablation"
        print(f"  phi={phi:5.1f} deg  cos(phi)={np.cos(np.deg2rad(phi)):.3f}  "
              f"T_back(t_end)={Tb:7.2f} K  recession={s*1e3:6.3f} mm  ({ablated})")

    Tback = np.array(Tback)
    recession = np.array(recession)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    phi_fine = np.linspace(0, 90, 200)
    ax.plot(phi_fine, np.cos(np.deg2rad(phi_fine)), color="tab:orange")
    ax.plot(PHI_DEG, np.cos(np.deg2rad(PHI_DEG)), "o", color="tab:orange")
    ax.set_xlabel("body station phi [deg from stagnation point]")
    ax.set_ylabel("h(phi) / h_stagnation")
    ax.set_title("Assumed heating distribution (cosine law)")

    ax = axes[1]
    ax.plot(PHI_DEG, recession * 1e3, "o-", color="tab:red")
    ax.set_xlabel("body station phi [deg]")
    ax.set_ylabel("total recession [mm]")
    ax.set_title("Recession vs. body station")

    ax = axes[2]
    ax.plot(PHI_DEG, Tback, "o-", color="tab:blue")
    ax.set_xlabel("body station phi [deg]")
    ax.set_ylabel("back-face T at t_end [K]")
    ax.set_title("Back-face temperature vs. body station")

    fig.suptitle("Phase 7: multi-station nose thermal map")
    fig.tight_layout()
    savefig(fig, "phase7_nose_map.png")

"""Phase 4: full coupled ablation model — aero heating + reradiation +
front recession, driven by a representative reentry heat pulse.

This is the actual "ablation model" the project is named for: it couples
ConvectiveRadiativeBC (aerodynamic convective heating with radiative
reradiation) to AblationFront1D's front-recession physics (Phase 3),
and tracks back-face temperature as the primary structural survival
metric.

Material properties (k, rho, cp, emissivity, heat_of_ablation,
ablation_temperature) below are representative order-of-magnitude values
for a carbon-phenolic-style ablator, not measurements of a specific real
material — this project's own measured diffusivity/emissivity would
replace these once available (see README/PROJECT_PLAN.md). The heating
pulse (h(t), T_aw) is a simple analytic single-hump shape representative
of a reentry heat pulse; replacing it with real h(t)/T_aw(t) from a
trajectory simulator is Phase 6 (stretch goal).
"""
import numpy as np

from heatablate import Material, ConvectiveRadiativeBC, AblationFront1D
from heatablate.plotting import savefig

# --- representative ablator material (order-of-magnitude, not a specific measured material) ---
K = 0.5  # W/(m K)
RHO = 1400.0  # kg/m^3
CP = 1200.0  # J/(kg K)
EMISSIVITY = 0.85
HEAT_OF_ABLATION = 8.0e6  # J/kg
ABLATION_TEMPERATURE = 2200.0  # K

# --- domain ---
LENGTH = 0.018  # m
N_NODES = 201
T0 = 300.0  # K, initial uniform temperature

# --- representative single-hump reentry heat pulse ---
H_PEAK = 800.0  # W/(m^2 K)
T_PEAK = 30.0  # s
T_AW = 6000.0  # K, adiabatic wall (recovery) temperature, held ~constant
T_END = 120.0  # s
DT = 0.02  # s


def h_profile(t):
    # classic single-hump heat-pulse shape: h(t) = h_peak*(t/t_peak)*exp(1-t/t_peak)
    tau = np.asarray(t, dtype=float) / T_PEAK
    return H_PEAK * tau * np.exp(1.0 - tau)


def run():
    mat = Material(
        k=K, rho=RHO, cp=CP, emissivity=EMISSIVITY,
        heat_of_ablation=HEAT_OF_ABLATION, ablation_temperature=ABLATION_TEMPERATURE,
    )
    bc = ConvectiveRadiativeBC(h=h_profile, T_aw=T_AW, emissivity=EMISSIVITY, T_inf=0.0)
    model = AblationFront1D(mat, LENGTH, N_NODES, bc, theta=1.0)
    T_init = np.full(N_NODES, T0)
    result = model.solve(T_init, DT, T_END)
    return result


if __name__ == "__main__":
    result = run()
    t, s, Tback = result["t"], result["s"], result["T_back"]
    eb = result["energy_balance"]

    onset_idx = int(np.argmax(s > 0))
    onset_time = t[onset_idx] if s[-1] > 0 else None
    q_peak = H_PEAK * (T_AW - ABLATION_TEMPERATURE)  # rough peak net-flux scale, ignoring reradiation

    print("Phase 4: coupled aero-heating + ablation model")
    print(f"  material: k={K} W/mK, rho={RHO} kg/m3, cp={CP} J/kgK, eps={EMISSIVITY}")
    print(f"  heat_of_ablation={HEAT_OF_ABLATION:.2e} J/kg, ablation_temperature={ABLATION_TEMPERATURE:.0f} K")
    print(f"  heat pulse: h_peak={H_PEAK} W/m2K at t={T_PEAK}s, T_aw={T_AW:.0f} K, mission={T_END:.0f}s")
    if onset_time is not None:
        print(f"  ablation onset at t = {onset_time:.2f} s")
    else:
        print("  surface never reached ablation temperature")
    print(f"  total recession at t_end: {s[-1]*1e3:.4f} mm")
    print(f"  back-face temperature: initial={Tback[0]:.1f} K, final={Tback[-1]:.1f} K, peak={Tback.max():.1f} K")
    print(f"  energy balance residual: {eb['residual_pct']:.3f}% of total incident energy "
          f"(E_in={eb['E_in']:.3e} J/m^2)")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(t, h_profile(t), color="tab:orange")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("h(t) [W/(m$^2$K)]")
    ax.set_title("Convective heating pulse")

    ax = axes[1]
    ax.plot(t, s * 1e3, color="tab:red")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("recession s(t) [mm]")
    ax.set_title(f"Front recession (total {s[-1]*1e3:.3f} mm)")

    ax = axes[2]
    ax.plot(t, Tback, color="tab:blue")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("back-face temperature [K]")
    ax.set_title(f"Back-face temperature (final {Tback[-1]:.1f} K)")

    fig.suptitle("Phase 4: coupled aero-heating + ablation model")
    fig.tight_layout()
    savefig(fig, "phase4_mission_ablation.png")

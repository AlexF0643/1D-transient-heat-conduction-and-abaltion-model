"""Phase 5 (repeat): grid and time-step convergence for the full coupled
ablation model (Phase 4).

There's no analytic solution for this problem (unlike Validations 1-3),
so this is a self-convergence study: back-face temperature and total
recession at t_end are tracked as node count / time step are refined,
and convergence is reported relative to the finest run in each sweep
(Richardson-style), rather than against a closed form.

Uses the same mission_ablation.py scenario (single-hump reentry heat
pulse through a carbon-phenolic-style ablator) as the reference problem.
"""
import time

import numpy as np

from heatablate import Material, ConvectiveRadiativeBC, AblationFront1D
from heatablate.plotting import savefig

# Same scenario as scripts/mission_ablation.py (duplicated rather than
# imported: scripts/ is a collection of standalone entry points, not a
# package, and re-importing across sibling scripts is fragile depending
# on how each is invoked).
K = 0.5  # W/(m K)
RHO = 1400.0  # kg/m^3
CP = 1200.0  # J/(kg K)
EMISSIVITY = 0.85
HEAT_OF_ABLATION = 8.0e6  # J/kg
ABLATION_TEMPERATURE = 2200.0  # K
LENGTH = 0.018  # m
T0 = 300.0  # K
H_PEAK = 800.0  # W/(m^2 K)
T_PEAK = 30.0  # s
T_AW = 6000.0  # K
T_END = 120.0  # s


def h_profile(t):
    tau = np.asarray(t, dtype=float) / T_PEAK
    return H_PEAK * tau * np.exp(1.0 - tau)


def run_case(n_nodes, dt):
    mat = Material(k=K, rho=RHO, cp=CP, emissivity=EMISSIVITY,
                    heat_of_ablation=HEAT_OF_ABLATION, ablation_temperature=ABLATION_TEMPERATURE)
    bc = ConvectiveRadiativeBC(h=h_profile, T_aw=T_AW, emissivity=EMISSIVITY, T_inf=0.0)
    model = AblationFront1D(mat, LENGTH, n_nodes, bc, theta=1.0)
    t0 = time.time()
    result = model.solve(np.full(n_nodes, T0), dt, T_END)
    elapsed = time.time() - t0
    return result["T_back"][-1], result["s"][-1], elapsed


if __name__ == "__main__":
    print("=== Spatial convergence (dt = 0.01 s) ===")
    node_counts = [51, 101, 201, 401]
    Tback_by_n = []
    s_by_n = []
    for n in node_counts:
        Tback, s_final, elapsed = run_case(n, dt=0.01)
        Tback_by_n.append(Tback)
        s_by_n.append(s_final)
        print(f"  N={n:4d}  T_back(t_end)={Tback:8.3f} K  s(t_end)={s_final*1e3:7.4f} mm  ({elapsed:.1f}s)")

    Tback_ref = Tback_by_n[-1]
    print(f"  relative error vs. finest (N={node_counts[-1]}):")
    for n, Tb in zip(node_counts, Tback_by_n):
        print(f"    N={n:4d}  {abs(Tb-Tback_ref)/Tback_ref*100:.4f} %")

    print()
    print("=== Temporal convergence (N = 201 nodes) ===")
    dts = [0.04, 0.02, 0.01, 0.005]
    Tback_by_dt = []
    s_by_dt = []
    for dt in dts:
        Tback, s_final, elapsed = run_case(201, dt)
        Tback_by_dt.append(Tback)
        s_by_dt.append(s_final)
        print(f"  dt={dt:6.3f} s  T_back(t_end)={Tback:8.3f} K  s(t_end)={s_final*1e3:7.4f} mm  ({elapsed:.1f}s)")

    Tback_ref_dt = Tback_by_dt[-1]
    print(f"  relative error vs. finest (dt={dts[-1]}):")
    for dt, Tb in zip(dts, Tback_by_dt):
        print(f"    dt={dt:6.3f}  {abs(Tb-Tback_ref_dt)/Tback_ref_dt*100:.4f} %")

    finest_n_err = abs(Tback_by_n[-2] - Tback_ref) / Tback_ref * 100
    finest_dt_err = abs(Tback_by_dt[-2] - Tback_ref_dt) / Tback_ref_dt * 100
    print()
    print(f"Summary: back-face temperature converged to within {finest_n_err:.3f}% "
          f"at N={node_counts[-2]} nodes (vs. N={node_counts[-1]}), and within "
          f"{finest_dt_err:.3f}% at dt={dts[-2]}s (vs. dt={dts[-1]}s).")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax1.plot(node_counts, Tback_by_n, "o-", color="tab:blue")
    ax1.axhline(Tback_ref, color="gray", ls="--", lw=1, alpha=0.7, label=f"finest (N={node_counts[-1]})")
    ax1.set_xlabel("node count N")
    ax1.set_ylabel("back-face T at t_end [K]")
    ax1.set_title("Spatial convergence (dt=0.01s)")
    ax1.legend()

    ax2.plot(dts, Tback_by_dt, "s-", color="tab:red")
    ax2.axhline(Tback_ref_dt, color="gray", ls="--", lw=1, alpha=0.7, label=f"finest (dt={dts[-1]}s)")
    ax2.set_xlabel("time step dt [s]")
    ax2.set_ylabel("back-face T at t_end [K]")
    ax2.set_title("Temporal convergence (N=201)")
    ax2.invert_xaxis()
    ax2.legend()

    fig.suptitle("Phase 5: convergence study on the coupled ablation model")
    fig.tight_layout()
    savefig(fig, "phase5_ablation_convergence.png")

"""Validation 2: semi-infinite solid, constant applied surface heat flux.

This exercises the FluxBC ghost-node closure rather than the plain
diffusion scheme, since the boundary condition itself is a Neumann
condition rather than a prescribed temperature.
"""
import numpy as np

from heatablate import Material, FluxBC, HeatConduction1D
from heatablate.analytic import semi_infinite_constant_flux_temperature
from heatablate.plotting import savefig

ALPHA = 6.0e-6  # m^2/s
K = 1.5  # W/(m K)
T0 = 300.0  # K
Q0 = 5.0e5  # W/m^2
L = 0.08  # m
T_END = 2.0  # s


def run(n_nodes=201, dt=2e-3):
    mat = Material(k=K, rho=1.0, cp=K / ALPHA)
    solver = HeatConduction1D(mat, L, n_nodes, FluxBC(Q0), FluxBC(0.0), theta=1.0)
    T_init = np.full(n_nodes, T0)
    t, T = solver.solve(T_init, dt, T_END)
    T_analytic = semi_infinite_constant_flux_temperature(solver.x, t[-1], ALPHA, K, Q0, T0)
    err = T[-1] - T_analytic
    return solver.x, T[-1], T_analytic, err


if __name__ == "__main__":
    x, T_num, T_an, err = run()
    rise = T_an[0] - T0
    max_abs = np.max(np.abs(err))
    max_rel = max_abs / rise * 100

    print("Validation 2: semi-infinite constant surface heat flux (Neumann/FluxBC)")
    print(f"  q0 = {Q0:.2e} W/m^2, domain L = {L*1000:.1f} mm, t_end = {T_END:.2f} s")
    print(f"  numeric surface T  : {T_num[0]:.3f} K")
    print(f"  analytic surface T : {T_an[0]:.3f} K")
    print(f"  max abs error      : {max_abs:.4f} K")
    print(f"  max rel error      : {max_rel:.4f} % of surface rise ({rise:.0f} K)")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x_mm = x * 1e3
    ax1.plot(x_mm, T_an, "-", color="tab:gray", lw=3, alpha=0.6, label="analytic")
    ax1.plot(x_mm, T_num, "--", color="tab:red", lw=1.5, label="numeric (FD, N=201)")
    ax1.set_xlabel("depth x [mm]")
    ax1.set_ylabel("temperature [K]")
    ax1.set_title(f"Temperature profile at t = {T_END:.1f} s")
    ax1.legend()
    ax1.set_xlim(0, x_mm[-1])

    ax2.plot(x_mm, err, color="tab:blue")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("depth x [mm]")
    ax2.set_ylabel("numeric - analytic [K]")
    ax2.set_title(f"Error (max {max_rel:.3f}% of surface rise)")
    ax2.set_xlim(0, x_mm[-1])

    fig.suptitle("Validation 2: semi-infinite constant surface heat flux")
    fig.tight_layout()
    savefig(fig, "validation2_constant_flux.png")

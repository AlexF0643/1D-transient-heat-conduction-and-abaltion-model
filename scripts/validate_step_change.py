"""Validation 1: semi-infinite solid, step change in surface temperature.

Compares the implicit FD solver against the analytic erf solution and
reports the peak error as a percentage of the applied temperature rise.
"""
import numpy as np

from heatablate import Material, DirichletBC, FluxBC, HeatConduction1D
from heatablate.analytic import semi_infinite_step_temperature

ALPHA = 6.0e-6  # m^2/s
K = 1.5  # W/(m K)
T0 = 300.0  # K
TS = 1200.0  # K
L = 0.08  # m, long enough that the back face never sees the front
T_END = 2.0  # s


def run(n_nodes=201, dt=2e-3):
    mat = Material(k=K, rho=1.0, cp=K / ALPHA)
    solver = HeatConduction1D(mat, L, n_nodes, DirichletBC(TS), FluxBC(0.0), theta=1.0)
    T_init = np.full(n_nodes, T0)
    t, T = solver.solve(T_init, dt, T_END)
    T_analytic = semi_infinite_step_temperature(solver.x, t[-1], ALPHA, T0, TS)
    err = T[-1] - T_analytic
    return solver.x, T[-1], T_analytic, err


if __name__ == "__main__":
    x, T_num, T_an, err = run()
    rise = TS - T0
    max_abs = np.max(np.abs(err))
    max_rel = max_abs / rise * 100

    print("Validation 1: semi-infinite step surface temperature (Dirichlet BC)")
    print(f"  alpha = {ALPHA:.3e} m^2/s, domain L = {L*1000:.1f} mm, t_end = {T_END:.2f} s")
    print(f"  penetration depth sqrt(alpha*t_end) = {np.sqrt(ALPHA*T_END)*1000:.3f} mm (<< L, semi-infinite OK)")
    print(f"  max abs error   : {max_abs:.4f} K")
    print(f"  max rel error   : {max_rel:.4f} % of applied rise ({rise:.0f} K)")
    print(f"  back face rise  : {T_num[-1]-T0:.2e} K (should be ~0)")

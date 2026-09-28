"""Validation 3: the classical one-phase Stefan (moving-boundary) problem.

This validates the ablation front-recession/remeshing machinery in
AblationFront1D specifically, decoupled from the diffusion scheme
(Validations 1-2) and from the nonlinear convective/radiative BC used in
the full model (Phase 4).

Setup: the material starts uniformly at the ablation temperature Tm (the
classical one-phase idealization — no sensible pre-heating ahead of the
front). The front is then driven by the exact analytic Stefan flux
`stefan_front_flux(t)`, which is what a fixed hot boundary at Ts would
deliver through the (here: already-removed) material ahead of the front.
Feeding the solver this closed-form flux, rather than a real physical
BC, isolates one question: given the theoretically correct driving flux,
does the front-tracking/remeshing algorithm reproduce the analytic front
position s(t) = 2*lambda*sqrt(alpha*t)?

The run starts at a small t0 > 0 (bootstrapped from the analytic front
position at t0) rather than t=0, since the self-similar solution has an
integrable but numerically inconvenient 1/sqrt(t) flux singularity at
t=0 — standard practice for this class of validation.
"""
import numpy as np

from heatablate import Material, FluxBC, AblationFront1D
from heatablate.analytic import stefan_lambda, stefan_front_position, stefan_front_flux
from heatablate.plotting import savefig

ALPHA = 6.0e-6  # m^2/s
K = 1.5  # W/(m K)
RHO = 1500.0  # kg/m^3
TM = 800.0  # K, ablation temperature
TS = 2000.0  # K, driving temperature (sets the analytic reference flux)
LH = 1.5e6  # J/kg, heat of ablation
L = 0.05  # m
T0_BOOTSTRAP = 2.0  # s
T_END = 60.0  # s
N_NODES = 201
DT = 0.01  # s


def run(n_nodes=N_NODES, dt=DT):
    cp = K / (ALPHA * RHO)
    mat = Material(k=K, rho=RHO, cp=cp, heat_of_ablation=LH, ablation_temperature=TM)
    Ste = cp * (TS - TM) / LH
    lam = stefan_lambda(Ste)

    s0 = stefan_front_position(T0_BOOTSTRAP, ALPHA, lam)
    T_init = np.full(n_nodes, TM)
    surface_bc = FluxBC(lambda t: stefan_front_flux(t, ALPHA, K, TS, TM, lam))

    ablation = AblationFront1D(mat, L, n_nodes, surface_bc, theta=1.0)
    result = ablation.solve(T_init, dt, T_END, t_start=T0_BOOTSTRAP, s_initial=s0)
    return result, lam, Ste


if __name__ == "__main__":
    result, lam, Ste = run()
    t, s_num = result["t"], result["s"]
    s_an = stefan_front_position(t, ALPHA, lam)
    rel_err = np.abs(s_num[1:] - s_an[1:]) / s_an[1:]

    print("Validation 3: classical one-phase Stefan problem (moving front)")
    print(f"  Stefan number Ste = {Ste:.4f}, lambda = {lam:.6f}")
    print(f"  bootstrap t0 = {T0_BOOTSTRAP:.2f} s, s(t0) = {s_num[0]*1e3:.4f} mm")
    print(f"  front position at t_end={T_END:.1f}s: numeric={s_num[-1]*1e3:.4f} mm, analytic={s_an[-1]*1e3:.4f} mm")
    print(f"  max relative error in s(t) over run: {np.max(rel_err)*100:.4f} %")
    print(f"  remaining-material temperature range at final step: "
          f"[{result['T_final'].min():.3f}, {result['T_final'].max():.3f}] K "
          f"(should be pinned at Tm={TM} K, the one-phase idealization)")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(t, s_an * 1e3, "-", color="tab:gray", lw=3, alpha=0.6, label="analytic")
    ax1.plot(t, s_num * 1e3, "--", color="tab:red", lw=1.5, label="numeric (AblationFront1D)")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("front position s(t) [mm]")
    ax1.set_title("Front position vs. time")
    ax1.legend()

    ax2.plot(t[1:], rel_err * 100, color="tab:blue")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("relative error [%]")
    ax2.set_title(f"Front position error (max {np.max(rel_err)*100:.3f}%)")

    fig.suptitle("Validation 3: one-phase Stefan problem")
    fig.tight_layout()
    savefig(fig, "validation3_stefan.png")

import numpy as np
import pytest

from heatablate import Material, DirichletBC, FluxBC, ConvectiveRadiativeBC, AblationFront1D
from heatablate.analytic import stefan_lambda, stefan_front_position, stefan_front_flux

ALPHA = 6.0e-6
K = 1.5
RHO = 1500.0
TM = 800.0
TS = 2000.0
LH = 1.5e6
L = 0.05


def make_material(heat_of_ablation=LH):
    cp = K / (ALPHA * RHO)
    return Material(k=K, rho=RHO, cp=cp, heat_of_ablation=heat_of_ablation, ablation_temperature=TM)


def test_front_position_matches_stefan_analytic():
    mat = make_material()
    cp = K / (ALPHA * RHO)
    Ste = cp * (TS - TM) / LH
    lam = stefan_lambda(Ste)

    t0 = 2.0
    t_end = 60.0
    n = 201
    s0 = stefan_front_position(t0, ALPHA, lam)
    T_init = np.full(n, TM)
    surface_bc = FluxBC(lambda t: stefan_front_flux(t, ALPHA, K, TS, TM, lam))

    ablation = AblationFront1D(mat, L, n, surface_bc, theta=1.0)
    result = ablation.solve(T_init, dt=0.01, t_end=t_end, t_start=t0, s_initial=s0)

    s_an = stefan_front_position(result["t"], ALPHA, lam)
    rel_err = np.abs(result["s"][1:] - s_an[1:]) / s_an[1:]
    assert np.max(rel_err) < 1e-3  # < 0.1%


def test_remaining_material_stays_at_ablation_temperature_in_one_phase_case():
    """The classical one-phase idealization (initial T = Tm everywhere)
    should keep the whole remaining domain pinned at Tm throughout,
    since there is no mechanism to raise it further."""
    mat = make_material()
    cp = K / (ALPHA * RHO)
    Ste = cp * (TS - TM) / LH
    lam = stefan_lambda(Ste)
    t0 = 2.0
    n = 101
    s0 = stefan_front_position(t0, ALPHA, lam)
    T_init = np.full(n, TM)
    surface_bc = FluxBC(lambda t: stefan_front_flux(t, ALPHA, K, TS, TM, lam))

    ablation = AblationFront1D(mat, L, n, surface_bc, theta=1.0)
    result = ablation.solve(T_init, dt=0.02, t_end=30.0, t_start=t0, s_initial=s0)

    assert np.max(np.abs(result["T_final"] - TM)) < 1e-6


def test_larger_heat_of_ablation_recedes_more_slowly():
    """Doubling the heat of ablation, with everything else fixed, must
    at least halve the recession rate (more energy needed per unit
    recession)."""
    n = 101
    T_init = np.full(n, 300.0)
    bc = ConvectiveRadiativeBC(h=400.0, T_aw=3000.0, emissivity=0.85, T_inf=0.0)

    mat_low = Material(k=1.5, rho=1500.0, cp=1200.0, emissivity=0.85, heat_of_ablation=1.0e6, ablation_temperature=1000.0)
    mat_high = Material(k=1.5, rho=1500.0, cp=1200.0, emissivity=0.85, heat_of_ablation=4.0e6, ablation_temperature=1000.0)

    s_low = AblationFront1D(mat_low, 0.03, n, bc).solve(T_init, dt=0.02, t_end=40.0)["s"][-1]
    s_high = AblationFront1D(mat_high, 0.03, n, bc).solve(T_init, dt=0.02, t_end=40.0)["s"][-1]

    assert s_low > 0
    assert s_high < s_low


def test_surface_pinned_at_ablation_temperature_once_receding():
    mat = Material(k=1.5, rho=1500.0, cp=1200.0, emissivity=0.85, heat_of_ablation=2.0e6, ablation_temperature=1500.0)
    bc = ConvectiveRadiativeBC(h=400.0, T_aw=3000.0, emissivity=0.85, T_inf=0.0)
    n = 121
    T_init = np.full(n, 300.0)
    result = AblationFront1D(mat, 0.03, n, bc).solve(T_init, dt=0.02, t_end=60.0)

    assert result["s"][-1] > 0  # ablation actually started
    assert abs(result["T_final"][0] - 1500.0) < 1e-6


def test_dirichlet_surface_bc_rejected():
    mat = make_material()
    with pytest.raises(TypeError):
        AblationFront1D(mat, L, 101, DirichletBC(TS))


def test_zero_incident_flux_never_ablates():
    mat = Material(k=1.5, rho=1500.0, cp=1200.0, heat_of_ablation=1.0e6, ablation_temperature=350.0)
    bc = FluxBC(0.0)
    n = 51
    T_init = np.full(n, 300.0)
    result = AblationFront1D(mat, 0.02, n, bc).solve(T_init, dt=1.0, t_end=200.0)
    assert result["s"][-1] == 0.0
    assert np.allclose(result["T_final"], 300.0, atol=1e-8)

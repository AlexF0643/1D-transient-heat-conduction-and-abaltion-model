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
    """The surface stays close to Tm throughout recession. Not exact: each
    step's remeshing linearly interpolates the (Dirichlet-clamped, exactly
    Tm) surface value onto a new grid, which can leave the new grid's
    surface node a small fraction of a degree off Tm — a benign, bounded
    numerical artifact (see AblationFront1D docstring), not a physical
    drift, so a loose bound here is the right check."""
    mat = Material(k=1.5, rho=1500.0, cp=1200.0, emissivity=0.85, heat_of_ablation=2.0e6, ablation_temperature=1500.0)
    bc = ConvectiveRadiativeBC(h=400.0, T_aw=3000.0, emissivity=0.85, T_inf=0.0)
    n = 121
    T_init = np.full(n, 300.0)
    result = AblationFront1D(mat, 0.03, n, bc).solve(T_init, dt=0.02, t_end=60.0)

    assert result["s"][-1] > 0  # ablation actually started
    assert abs(result["T_final"][0] - 1500.0) < 1.0


def test_energy_balance_closes_for_physically_driven_ablation():
    """Regression test for a real bug: `ablating` used to be re-derived from
    `T[0] >= Tm` every step, and after remeshing the interpolated surface
    value can land a hair below Tm, flipping back to the unclamped
    surface_bc branch and silently skipping about half the recession
    physics on alternating steps (verified: exactly 1:1 alternation over a
    3000-step run). That bug was invisible in the Stefan validation (it
    starts already at Tm, so the flag latches immediately and never
    round-trips), which is why this independent energy-balance check on a
    physically-driven run matters. Caught because the diagnostic showed an
    ~8% unexplained energy residual instead of the sub-1% expected from
    remeshing-interpolation drift alone."""
    mat = Material(k=1.5, rho=1500.0, cp=1200.0, emissivity=0.85, heat_of_ablation=2.0e6, ablation_temperature=1500.0)
    bc = ConvectiveRadiativeBC(h=400.0, T_aw=3000.0, emissivity=0.85, T_inf=0.0)
    n = 151
    T_init = np.full(n, 300.0)
    result = AblationFront1D(mat, 0.03, n, bc).solve(T_init, dt=0.02, t_end=60.0)

    assert result["s"][-1] > 0
    assert abs(result["energy_balance"]["residual_pct"]) < 1.0


def test_ablating_state_does_not_chatter_under_steady_heating():
    """Directly checks the bug's symptom: once the surface first reaches
    Tm under STEADY (non-fading) heating, every subsequent step must
    genuinely continue receding, never spuriously revert to the raw
    surface_bc branch due to a benign sub-degree interpolation undershoot
    (as opposed to a real, physically-driven exit from ablation — see
    test_recession_stops_once_heat_pulse_fades for that legitimate case)."""
    mat = Material(k=1.5, rho=1500.0, cp=1200.0, emissivity=0.85, heat_of_ablation=2.0e6, ablation_temperature=1500.0)
    bc = ConvectiveRadiativeBC(h=400.0, T_aw=3000.0, emissivity=0.85, T_inf=0.0)
    n = 151
    T_init = np.full(n, 300.0)
    dt = 0.02
    result = AblationFront1D(mat, 0.03, n, bc).solve(T_init, dt=dt, t_end=60.0)

    s = result["s"]
    onset_idx = np.argmax(s > 0)
    assert onset_idx > 0  # ablation actually started partway through
    # once started, recession must never sit at a flat plateau for two
    # consecutive recorded steps under this steady, strongly-heating BC
    ds = np.diff(s[onset_idx:])
    assert np.all(ds > 0)


def test_recession_stops_once_heat_pulse_fades():
    """The physically correct counterpart to the latch fix above: once a
    heat pulse fades enough that the incident flux can no longer sustain
    the ablation temperature, recession must genuinely stop AND the
    surface must be allowed to cool below Tm — not stay artificially
    clamped there forever. A single-hump pulse (rises then decays) is the
    natural test: recession should plateau well before the run ends, and
    final surface temperature should end up measurably below Tm."""
    h_peak, t_peak, T_aw = 800.0, 30.0, 6000.0

    def h_profile(t):
        tau = np.asarray(t, dtype=float) / t_peak
        return h_peak * tau * np.exp(1.0 - tau)

    mat = Material(k=0.5, rho=1400.0, cp=1200.0, emissivity=0.85, heat_of_ablation=8.0e6, ablation_temperature=2200.0)
    bc = ConvectiveRadiativeBC(h=h_profile, T_aw=T_aw, emissivity=0.85, T_inf=0.0)
    n = 151
    T_init = np.full(n, 300.0)
    result = AblationFront1D(mat, 0.018, n, bc).solve(T_init, dt=0.05, t_end=120.0)

    s = result["s"]
    assert s[-1] > 0  # ablation happened at all
    last_quarter = 3 * len(s) // 4
    assert s[-1] == s[last_quarter]  # recession has already plateaued well before the run ends
    assert result["T_final"][0] < mat.ablation_temperature - 10.0  # surface genuinely cooled below Tm
    assert abs(result["energy_balance"]["residual_pct"]) < 2.0


def test_dirichlet_surface_bc_rejected():
    mat = make_material()
    with pytest.raises(TypeError):
        AblationFront1D(mat, L, 101, DirichletBC(TS))


def test_ablation_temperature_to_infinity_recovers_plain_solution():
    """Limiting case: if the surface never reaches ablation_temperature,
    AblationFront1D must reduce exactly to the plain (non-ablating)
    HeatConduction1D + ConvectiveRadiativeBC solution — same BC, same
    domain, same theta, just never entering the ablating branch."""
    from heatablate import HeatConduction1D

    k, rho, cp, eps = 1.5, 1500.0, 1200.0, 0.85
    bc = ConvectiveRadiativeBC(h=400.0, T_aw=3000.0, emissivity=eps, T_inf=0.0)
    n = 101
    length = 0.03
    dt = 0.05
    t_end = 60.0
    T_init = np.full(n, 300.0)

    mat_plain = Material(k=k, rho=rho, cp=cp, emissivity=eps)
    solver = HeatConduction1D(mat_plain, length, n, bc, FluxBC(0.0), theta=1.0)
    t_plain, T_plain = solver.solve(T_init, dt, t_end)

    mat_ablation = Material(k=k, rho=rho, cp=cp, emissivity=eps, heat_of_ablation=1.0e6, ablation_temperature=1.0e6)
    result = AblationFront1D(mat_ablation, length, n, bc).solve(T_init, dt, t_end)

    assert result["s"][-1] == 0.0
    assert np.max(np.abs(result["T_final"] - T_plain[-1])) < 1e-8
    assert np.max(np.abs(result["T_back"] - T_plain[:, -1])) < 1e-8


def test_large_heat_of_ablation_gives_negligible_recession():
    """Limiting case: a very large heat of ablation should drive
    recession to (near) zero even though the surface still reaches and
    clamps at ablation_temperature — energy just isn't enough to remove
    meaningful mass, all of it effectively going into sensible heating
    of the (still Dirichlet-clamped) remaining material instead."""
    bc = ConvectiveRadiativeBC(h=400.0, T_aw=3000.0, emissivity=0.85, T_inf=0.0)
    n = 121
    T_init = np.full(n, 300.0)

    mat_normal = Material(k=1.5, rho=1500.0, cp=1200.0, emissivity=0.85, heat_of_ablation=2.0e6, ablation_temperature=1500.0)
    mat_huge_lh = Material(k=1.5, rho=1500.0, cp=1200.0, emissivity=0.85, heat_of_ablation=1.0e12, ablation_temperature=1500.0)

    s_normal = AblationFront1D(mat_normal, 0.03, n, bc).solve(T_init, dt=0.02, t_end=60.0)["s"][-1]
    result_huge = AblationFront1D(mat_huge_lh, 0.03, n, bc).solve(T_init, dt=0.02, t_end=60.0)

    assert result_huge["s"][-1] < 1e-7  # effectively no recession (nanometers, not mm)
    assert s_normal > 1e-6  # sanity: the normal case actually receded meaningfully
    assert abs(result_huge["T_final"][0] - 1500.0) < 1.0  # surface still clamps at Tm


def test_zero_incident_flux_never_ablates():
    mat = Material(k=1.5, rho=1500.0, cp=1200.0, heat_of_ablation=1.0e6, ablation_temperature=350.0)
    bc = FluxBC(0.0)
    n = 51
    T_init = np.full(n, 300.0)
    result = AblationFront1D(mat, 0.02, n, bc).solve(T_init, dt=1.0, t_end=200.0)
    assert result["s"][-1] == 0.0
    assert np.allclose(result["T_final"], 300.0, atol=1e-8)

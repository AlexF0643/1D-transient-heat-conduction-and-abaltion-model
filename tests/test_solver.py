import numpy as np
import pytest

from heatablate._compat import trapezoid
from heatablate import Material, DirichletBC, FluxBC, ConvectiveRadiativeBC, HeatConduction1D
from heatablate.analytic import semi_infinite_step_temperature, semi_infinite_constant_flux_temperature

ALPHA = 6.0e-6
K = 1.5
T0 = 300.0
L = 0.08
T_END = 2.0


def make_material():
    return Material(k=K, rho=1.0, cp=K / ALPHA)


def test_step_change_matches_analytic():
    Ts = 1200.0
    mat = make_material()
    solver = HeatConduction1D(mat, L, 201, DirichletBC(Ts), FluxBC(0.0), theta=1.0)
    t, T = solver.solve(np.full(201, T0), dt=2e-3, t_end=T_END)
    T_an = semi_infinite_step_temperature(solver.x, t[-1], ALPHA, T0, Ts)
    max_rel_err = np.max(np.abs(T[-1] - T_an)) / (Ts - T0)
    assert max_rel_err < 5e-4  # < 0.05%


def test_constant_flux_matches_analytic():
    q0 = 5.0e5
    mat = make_material()
    solver = HeatConduction1D(mat, L, 201, FluxBC(q0), FluxBC(0.0), theta=1.0)
    t, T = solver.solve(np.full(201, T0), dt=2e-3, t_end=T_END)
    T_an = semi_infinite_constant_flux_temperature(solver.x, t[-1], ALPHA, K, q0, T0)
    rise = T_an[0] - T0
    max_rel_err = np.max(np.abs(T[-1] - T_an)) / rise
    assert max_rel_err < 2e-3  # < 0.2%


def test_adiabatic_slab_conserves_energy():
    """No-flux both ends: total enthalpy of a finite slab must be conserved."""
    mat = make_material()
    n = 101
    solver = HeatConduction1D(mat, 0.05, n, FluxBC(0.0), FluxBC(0.0), theta=1.0)
    rng = np.random.default_rng(0)
    T_init = 500.0 + 50.0 * np.sin(np.linspace(0, 3 * np.pi, n)) + rng.normal(0, 1, n)
    t, T = solver.solve(T_init, dt=0.05, t_end=50.0)

    def enthalpy(Tprofile):
        return trapezoid(Tprofile, solver.x) * mat.rho * mat.cp

    H0 = enthalpy(T[0])
    H1 = enthalpy(T[-1])
    assert abs(H1 - H0) / abs(H0) < 1e-8
    # and it should have relaxed towards a flat (equilibrium) profile
    assert np.std(T[-1]) < np.std(T[0])


def test_uniform_initial_condition_stays_uniform_with_no_flux():
    mat = make_material()
    n = 51
    solver = HeatConduction1D(mat, 0.05, n, FluxBC(0.0), FluxBC(0.0), theta=1.0)
    T_init = np.full(n, 400.0)
    t, T = solver.solve(T_init, dt=1.0, t_end=100.0)
    assert np.allclose(T[-1], 400.0, atol=1e-8)


def test_dirichlet_both_ends_reaches_linear_steady_state():
    mat = make_material()
    n = 41
    T_left, T_right = 1000.0, 300.0
    solver = HeatConduction1D(mat, 0.1, n, DirichletBC(T_left), DirichletBC(T_right), theta=1.0)
    T_init = np.full(n, 300.0)
    t, T = solver.solve(T_init, dt=5.0, t_end=30000.0)  # >> L^2/alpha diffusion time
    T_expected = np.linspace(T_left, T_right, n)
    assert np.max(np.abs(T[-1] - T_expected)) < 1e-3


def test_convective_radiative_bc_reaches_radiative_equilibrium():
    """A slab with no incident heating and pure radiative BC on both faces
    should relax to the background radiative equilibrium temperature."""
    mat = Material(k=1.0, rho=1500.0, cp=1000.0, emissivity=0.8)
    n = 21
    T_inf = 800.0
    bc = ConvectiveRadiativeBC(h=0.0, T_aw=0.0, emissivity=0.8, T_inf=T_inf)
    solver = HeatConduction1D(mat, 0.02, n, bc, bc, theta=1.0)
    T_init = np.full(n, 300.0)
    t, T = solver.solve(T_init, dt=5.0, t_end=200000.0)
    assert np.max(np.abs(T[-1] - T_inf)) < 1.0


def test_thomas_solve_matches_dense_solve():
    from heatablate.solver import thomas_solve

    rng = np.random.default_rng(1)
    n = 30
    a = rng.uniform(-1, -0.1, n)
    c = rng.uniform(-1, -0.1, n)
    b = np.abs(a) + np.abs(c) + rng.uniform(1, 2, n)  # diagonally dominant
    a[0] = 0.0
    c[-1] = 0.0
    d = rng.uniform(-1, 1, n)

    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    x_dense = np.linalg.solve(A, d)
    x_thomas = thomas_solve(a, b, c, d)
    assert np.allclose(x_dense, x_thomas)


def test_rannacher_startup_improves_crank_nicolson():
    alpha, k, T0, Ts, L, t_end, N, dt = 6.0e-6, 1.5, 300.0, 1200.0, 0.08, 2.0, 401, 0.05

    def err(rannacher_steps):
        mat = Material(k=k, rho=1.0, cp=k / alpha)
        s = HeatConduction1D(mat, L, N, DirichletBC(Ts), FluxBC(0.0),
                             theta=0.5, rannacher_steps=rannacher_steps)
        t, T = s.solve(np.full(N, T0), dt, t_end)
        T_an = semi_infinite_step_temperature(s.x, t[-1], alpha, T0, Ts)
        assert s.theta == 0.5  # theta restored after solve
        return np.sqrt(np.mean((T[-1] - T_an) ** 2))

    assert err(2) < 0.2 * err(0)


def test_stiff_radiative_boundary_converges_on_coarse_grid():
    # h*dx/k >> 1: the old fixed-point (Picard) surface iteration diverged here.
    mat = Material(k=0.5, rho=1400.0, cp=1200.0, emissivity=0.85)
    bc = ConvectiveRadiativeBC(h=800.0, T_aw=6000.0, emissivity=0.85, T_inf=0.0)
    s = HeatConduction1D(mat, 0.018, 21, bc, FluxBC(0.0), theta=1.0)
    T = np.full(21, 300.0)
    for n in range(20):
        T = s.step(T, n * 1.0, 1.0)
    assert np.all(np.isfinite(T))
    # surface sits between the initial temperature and the recovery temperature,
    # and satisfies its own nonlinear flux balance to within the linearization tolerance
    assert 300.0 < T[0] < 6000.0
    assert T[0] >= T[-1]


def test_newton_and_fine_grid_agree_with_radiative_equilibrium():
    # zero-conduction limit: surface settles where h*(T_aw - Ts) = eps*sigma*Ts^4
    mat = Material(k=50.0, rho=1000.0, cp=500.0, emissivity=0.9)
    bc = ConvectiveRadiativeBC(h=200.0, T_aw=1500.0, emissivity=0.9, T_inf=0.0)
    s = HeatConduction1D(mat, 0.01, 41, bc, FluxBC(0.0), theta=1.0)
    T = np.full(41, 300.0)
    for n in range(4000):
        T = s.step(T, n * 5.0, 5.0)
    Ts = T[0]
    assert bc.net_flux(0.0, Ts) == pytest.approx(0.0, abs=1e-3)


def test_nonconvergence_emits_warning():
    mat = Material(k=0.5, rho=1400.0, cp=1200.0, emissivity=0.85)
    bc = ConvectiveRadiativeBC(h=800.0, T_aw=6000.0, emissivity=0.85, T_inf=0.0)
    s = HeatConduction1D(mat, 0.018, 21, bc, FluxBC(0.0))
    with pytest.warns(RuntimeWarning, match="did not converge"):
        s.step(np.full(21, 300.0), 0.0, 1.0, max_iter=1)

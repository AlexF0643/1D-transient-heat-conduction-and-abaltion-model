# 1D Transient Heat Conduction and Ablation Model

A 1D transient conduction solver for a hypersonic vehicle nose/heat
shield, built around aerodynamic heating, radiative reradiation, and
(in progress) ablative front recession. See
[`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full phased plan,
including the stretch goal of coupling to a 3-DOF trajectory simulator
and running Monte Carlo failure-probability analysis.

## Status

- Core implicit (theta-method) finite-difference solver: **done**
- Dirichlet, flux, and convective+radiative boundary conditions: **done**
- Validation against semi-infinite step-change analytic solution: **done, 0.021% max error**
- Validation against semi-infinite constant-flux analytic solution: **done, 0.096% max error**
- Grid/time-step convergence study: **done** (see below)
- Stefan problem (moving-boundary ablation) validation: **done, 0.031% max error in front position**
- Full coupled ablation model + back-face survival tracking: **planned, Phase 4**
- Trajectory coupling + Monte Carlo: **stretch, Phase 6**
- Multi-station nose thermal map: **stretch, Phase 7**

## Install

```bash
pip install -e .
```

## Run

```bash
python3 scripts/validate_step_change.py
python3 scripts/validate_constant_flux.py
python3 scripts/validate_stefan.py
python3 scripts/convergence_study.py
pytest tests/
```

## Physics and numerics

Solves `rho*cp * dT/dt = d/dx(k * dT/dx)` on a uniform 1D grid with a
generalized theta-method in time (`theta=1`: backward Euler, `theta=0.5`:
Crank-Nicolson), both unconditionally stable, solved with a hand-written
Thomas (tridiagonal) algorithm. Boundary conditions are pluggable:

- `DirichletBC(T)` — prescribed surface temperature
- `FluxBC(q)` — prescribed heat flux into the domain (used for the
  adiabatic back face and the constant-flux validation)
- `ConvectiveRadiativeBC(h, T_aw, emissivity, T_inf)` — aerodynamic
  convective heating with radiative reradiation,
  `q = h*(T_aw - Ts) - eps*sigma*(Ts^4 - T_inf^4)`, nonlinear in the
  unknown surface temperature `Ts` and closed each step with a Picard
  iteration.

All boundary closures use a ghost-node central difference, so they stay
second-order accurate in space like the interior scheme.

`Material.from_diffusivity(alpha, k, ...)` lets you feed in a *measured*
thermal diffusivity and conductivity directly (rather than re-deriving
`rho*cp` from tabulated density/specific-heat values) — the intended use
case being lab-measured diffusivity and emissivity as direct model
inputs.

`AblationFront1D` (`heatablate/ablation.py`) adds a receding surface on
top of `HeatConduction1D`: once the surface reaches
`material.ablation_temperature`, it's held there and the excess incident
flux (beyond what conducts into the remaining material) drives recession,
absorbing `material.heat_of_ablation`. Implemented as a moving-mesh (ALE)
scheme — the conduction sub-problem is re-solved each step on the
current, shrinking domain, with the temperature field linearly
interpolated onto the new grid after each recession increment.

## Validation results

| Check | Metric | Result |
|---|---|---|
| Semi-infinite step surface temperature (erf solution) | max error vs. 900 K applied step | 0.19 K (0.021%) |
| Semi-infinite constant surface flux | max error vs. surface temperature rise | 1.25 K (0.096%) |
| One-phase Stefan problem (moving front) | max relative error in front position s(t) | 0.031% |
| Spatial convergence | observed order (central difference) | ~1.9–2.0 |
| Temporal convergence, backward Euler | observed order | ~1.0 (as expected) |
| Temporal convergence, Crank-Nicolson | observed order | ~1.0, not the textbook 2nd order — see note below |

![Validation 1: step change](figures/validation1_step_change.png)
![Validation 2: constant flux](figures/validation2_constant_flux.png)
![Validation 3: Stefan problem](figures/validation3_stefan.png)
![Convergence study](figures/convergence_study.png)

Each validation script regenerates its figure under `figures/` when run
(`python3 scripts/validate_step_change.py`, etc.) — the PNGs above are
committed so they render directly on GitHub without needing to run
anything.

**Note on the Crank-Nicolson result:** a step-change Dirichlet boundary
condition is non-smooth ("rough") data at `t=0`, a case where
Crank-Nicolson's temporal order is known to degrade (Rannacher, 1984).
This was confirmed directly: prefixing the time march with two
backward-Euler steps before switching to Crank-Nicolson (a standard
"Rannacher startup") drops the error by ~20x at the largest time step
tested. The solver defaults to `theta=1` (backward Euler), which is
unaffected by this and is what all validation runs above use.

**Note on the Stefan validation:** the classical one-phase Stefan
solution assumes the material starts uniformly at the ablation
temperature (no sensible pre-heating ahead of the front), which makes
the remaining-material temperature field trivially uniform by
construction — the meaningful, non-trivial check is the front position
`s(t)` itself, driven by the exact analytic front flux fed in as a
boundary condition. This isolates the front-recession/remeshing
algorithm from the (separately validated) diffusion scheme. See
`docs/PROJECT_PLAN.md` Phase 3 for the full reasoning.

## Tests

`tests/test_solver.py`: agreement with both analytic solutions above,
energy conservation on an adiabatic slab, correctness of the
hand-written Thomas solve against a dense linear solve, and steady-state
limits for both Dirichlet and radiative-equilibrium boundary conditions.

`tests/test_ablation.py`: agreement with the Stefan analytic front
position, the remaining material staying pinned at the ablation
temperature in the one-phase idealization, recession rate scaling
inversely with heat of ablation, surface temperature staying pinned at
`ablation_temperature` once receding, and a physically-driven run
(convective+radiative heating, non-idealized initial condition) showing
sensible pre-heat → onset → recession behavior.

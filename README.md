# 1D Transient Heat Conduction and Ablation Model

A 1D transient conduction and ablation solver for a hypersonic vehicle
nose/heat shield, coupling aerodynamic heating, radiative reradiation,
and ablative front recession, with back-face temperature tracked as the
structural survival metric. See
[`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full phased plan.

## Status

- Core implicit (theta-method) finite-difference solver: **done**
- Dirichlet, flux, and convective+radiative boundary conditions: **done**
- Validation against semi-infinite step-change analytic solution: **done, 0.021% max error**
- Validation against semi-infinite constant-flux analytic solution: **done, 0.096% max error**
- Grid/time-step convergence study (Validations 1-2): **done** (see below)
- Stefan problem (moving-boundary ablation) validation: **done, 0.031% max error in front position**
- Full coupled ablation model + back-face survival tracking: **done** (see below)
- Grid/time-step convergence study on the coupled ablation model: **done, converged to within 0.24% at 201 nodes**
- Multi-station nose thermal map: **done** (see below)
- Trajectory coupling + Monte Carlo: **blocked** — checked the actual 3-DOF
  intercept simulator; its flight regime (Mach ~1-1.3, low-altitude
  terminal intercept) is physically incompatible with hypersonic
  stagnation heating (~2,000-5,000x too slow). See `docs/PROJECT_PLAN.md`
  Phase 6 for the investigation and what would unblock it.

**The core validated model (Phases 0-5) and one of the two stretch
goals (Phase 7) are complete.**

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
python3 scripts/mission_ablation.py
python3 scripts/convergence_study_ablation.py   # ~2-3 min, self-convergence sweep
python3 scripts/multistation_nose_map.py        # ~2 min, 8 stations
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
absorbing `material.heat_of_ablation` — but only while the incident flux
can actually sustain it (`q_incident >= q_conducted`, checked every
step); once a fading heat pulse can no longer sustain the ablation
temperature, the surface is released and allowed to cool. Implemented as
a moving-mesh (ALE) scheme — the conduction sub-problem is re-solved each
step on the current, shrinking domain, with the temperature field
linearly interpolated onto the new grid after each recession increment.

## Phase 4: coupled ablation model

`scripts/mission_ablation.py` couples `ConvectiveRadiativeBC` to
`AblationFront1D` and runs a representative single-hump reentry heat
pulse through a carbon-phenolic-style ablator (order-of-magnitude
material properties — not a specific measured material; this project's
own measured diffusivity/emissivity would replace these once available):

![Phase 4: coupled ablation model](figures/phase4_mission_ablation.png)

Ablation onset at t=7.8s, 6.32mm total recession, back-face temperature
rising from 300K to 573K over a 120s mission with ~12mm of material
margin remaining. Recession correctly tracks the heating pulse and
plateaus once the pulse fades past what it can sustain — the surface
then cools back below the ablation temperature rather than staying
artificially clamped there.

**`AblationFront1D.solve()` returns an `energy_balance` diagnostic**
(cumulative incident energy vs. sensible + latent enthalpy accounted
for) specifically to catch silent bugs in the front-recession logic that
"does it run and look plausible" wouldn't. It caught two real ones while
building this phase:

1. A **state-chattering bug**: the ablating/not-ablating decision was
   originally re-derived from a bare `T[0] >= Tm` check every step, but
   post-remeshing interpolation can leave the surface a hair below Tm
   even while ablation is fully sustained — which flipped the algorithm
   back to the unclamped boundary condition on *exactly half* the steps
   in a 3000-step test run, silently skipping the recession update each
   time. Symptom: an ~8% energy-balance residual against an expected
   sub-1%.
2. A **missing exit-from-ablation physics bug**: the first fix (a
   one-way latch) traded the chattering for a different error — under a
   fading heat pulse, it kept forcing the surface to stay at the
   ablation temperature forever, even once incident flux could no longer
   sustain it. Symptom: a **-12.8%** residual on the full mission-pulse
   scenario.

Both are fixed by deciding the ablating state from a genuine per-step
physical sustainability check rather than a temperature re-check (see
`heatablate/ablation.py` module docstring and `docs/PROJECT_PLAN.md`
Phase 4 for the full derivation). After both fixes, the mission-pulse
residual is **-0.23%**, consistent with the remeshing-interpolation
drift alone. Regression tests for both bugs are in
`tests/test_ablation.py`.

**Limiting cases** (see `tests/test_ablation.py`): `ablation_temperature
→ infinity` (never reached) recovers the plain, non-ablating
`HeatConduction1D` + `ConvectiveRadiativeBC` solution exactly. A very
large `heat_of_ablation` instead drives recession to near-zero while the
surface still clamps at the ablation temperature — a different, and
correct, limit (the original plan had this backwards; see
`docs/PROJECT_PLAN.md`).

## Phase 5: convergence on the coupled ablation model

There's no analytic solution for the full coupled model, so
`scripts/convergence_study_ablation.py` runs a self-convergence study on
the Phase 4 mission scenario: back-face temperature at t_end is tracked
across node-count and time-step refinement, and error is measured
against the finest run in each sweep (Richardson-style).

![Phase 5: ablation convergence](figures/phase5_ablation_convergence.png)

**Back-face temperature converged to within 0.24% at 201 nodes** (vs.
401 nodes: 1.63% → 0.71% → 0.24% → reference — roughly halving each
doubling, consistent with the ALE remeshing's linear interpolation being
the dominant error source, 1st order, rather than the interior
diffusion scheme, 2nd order as shown in Validations 1-2). Time-step
error is negligible by comparison: **within 0.003% at dt=0.01s**, so the
grid — not the time step — is what to refine if tighter accuracy is ever
needed.

## Phase 6: trajectory coupling — investigated, blocked

Before wiring the ablation model to a trajectory source, I pulled and
checked the actual 3-DOF intercept simulator this stretch goal named.
Every one of its 9 preset scenarios launches at 1000m altitude, missile
speed 60 m/s off the rail, target at 250 m/s, closing speeds topping out
around 300-450 m/s (~Mach 1-1.3, per the simulator's own README) — a
low-altitude terminal intercept engagement, not a hypersonic reentry
trajectory. Stagnation heating correlations scale roughly as velocity³,
so feeding this simulator's real output into the model would show
essentially zero heating and no ablation: technically "using the real
trajectory," but a physically empty demonstration.

Rather than silently produce that non-result, or quietly substitute a
synthetic trajectory in its place, this was flagged and deferred —
see `docs/PROJECT_PLAN.md` Phase 6 for the full reasoning and what would
unblock it (a genuinely hypersonic trajectory source, real or a
purpose-built synthetic reentry integrator labeled as such).

## Phase 7: multi-station nose thermal map

`scripts/multistation_nose_map.py` parametrizes a blunted (spherical-cap)
nose by the angle `phi` from the stagnation point and scales the Phase 4
heating pulse at each of 8 stations (0°-70°) by the standard cosine-law
Newtonian-flow approximation `h(phi,t) = h_stagnation(t) * cos(phi)`.
`AblationFront1D` runs independently at each station — no coupling
needed between stations under the thin-boundary-layer assumption that
justifies this whole approach (see `docs/PROJECT_PLAN.md` Phase 7 for
why this, not a full 2D/3D solve, is how heat-shield sizing tools like
NASA's FIAT actually work).

![Phase 7: multi-station nose map](figures/phase7_nose_map.png)

Recession falls from 6.32mm at the stagnation point (phi=0°, exactly
reproducing the Phase 4/5 reference case as a consistency check) to
0.32mm at phi=60°, and stops entirely by phi=70° — the heating there
never reaches the ablation temperature at all. Back-face temperature
falls similarly, 573.5K → 399.2K across the mapped stations. A genuine
spatial thermal map of the nose.

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
position; the remaining material staying pinned at the ablation
temperature in the one-phase idealization; recession rate scaling
inversely with heat of ablation; surface temperature staying near
`ablation_temperature` once receding; a physically-driven run
(convective+radiative heating, non-idealized initial condition) showing
sensible pre-heat → onset → recession behavior; the energy balance
closing to within 1% on that physically-driven run; no state-chattering
under steady heating; recession correctly stopping (and the surface
cooling back below Tm) once a heat pulse fades; and both limiting cases
(`ablation_temperature → infinity` recovers the plain solution exactly;
large `heat_of_ablation` drives recession to near-zero).

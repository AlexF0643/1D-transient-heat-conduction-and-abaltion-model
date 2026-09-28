# Project plan: 1D transient heat conduction and ablation model for a hypersonic vehicle nose

## Why this project

A 1D transient conduction solver is coursework. The same solver, framed
around re-entry/hypersonic heat-shield ablation, with a nose material
whose emissivity and thermal diffusivity were measured in the lab rather
than looked up, validated against three independent analytic solutions,
and driven by a trajectory from an existing 3-DOF intercept simulator, is
an aerospace project. It also pairs cleanly with an existing
estimation/control portfolio (thermofluids + GN&C = the standard aerospace
split) and is directly relevant to sustained hypersonic flight, where
thermal management is the limiting problem.

## Scope and phases (~30 hours)

### Phase 0 — Core solver (done this session, ~4 hrs)
- 1D uniform-grid finite-difference solver for `rho*cp dT/dt = d/dx(k dT/dx)`.
- Generalized theta-method in time (`theta=1`: backward Euler,
  `theta=0.5`: Crank-Nicolson), unconditionally stable, tridiagonal solve
  via a hand-written Thomas algorithm (`heatablate/solver.py`).
- Boundary conditions as pluggable objects (`heatablate/boundary.py`):
  - `DirichletBC` — prescribed surface temperature
  - `FluxBC` — prescribed heat flux (used for adiabatic back face and
    constant-flux validation)
  - `ConvectiveRadiativeBC` — `q = h*(T_aw - Ts) - eps*sigma*(Ts^4 - T_inf^4)`,
    the aero-heating + reradiation boundary condition, nonlinear in `Ts`
    and closed each time step with a Picard (fixed-point) iteration.
- All boundary closures use a ghost-node central difference, so they stay
  second-order accurate in space, matching the interior scheme.
- `Material` dataclass carries `k, rho, cp, emissivity`, plus a
  `from_diffusivity()` constructor so a *measured* diffusivity and
  conductivity (the lab data) can be fed in directly instead of
  re-deriving `rho*cp`.

**Status: implemented, unit-tested (7 passing tests:
analytic agreement, energy conservation, steady-state limits for
Dirichlet/Robin BCs, and the Thomas solve against a dense linear solve).**

### Phase 1 — Validation 1: step change in surface temperature (done, ~2 hrs)
Semi-infinite solid, surface suddenly held at `Ts`:

    T(x,t) = Ts + (T0 - Ts) * erf(x / (2*sqrt(alpha*t)))

`scripts/validate_step_change.py` runs the Dirichlet-BC solver on a slab
long enough that the back face never sees the thermal front
(`sqrt(alpha*t_end) << L`) and compares to this closed form.

**Result: max error 0.19 K, i.e. 0.021% of the 900 K applied step
(N=201, dt=2 ms).**

### Phase 2 — Validation 2: constant surface heat flux (done, ~1.5 hrs)
Semi-infinite solid, constant flux `q0` applied at `t=0`:

    T(x,t) - T0 = (q0/k) * [2*sqrt(alpha*t/pi)*exp(-x^2/(4*alpha*t)) - x*erfc(x/(2*sqrt(alpha*t)))]

This exercises the `FluxBC` ghost-node closure specifically, since the
prescribed quantity is a derivative, not the field itself.

**Result: max error 1.25 K, i.e. 0.096% of the surface temperature rise
(N=201, dt=2 ms).**

### Phase 3 — Validation 3: Stefan problem (moving boundary) (done, ~4 hrs)
The classical one-phase Stefan problem: a semi-infinite solid at the melt
temperature has its surface held above `Tm`; a phase-change front recedes
from the surface as

    s(t) = 2*lambda*sqrt(alpha*t)

with `lambda` the root of `lambda*exp(lambda^2)*erf(lambda) = Ste/sqrt(pi)`
(`Ste` = Stefan number = sensible/latent heat ratio).

**Solver (`heatablate/ablation.py`, `AblationFront1D`):** at each step,
the conduction sub-problem is solved on the current domain
`[s(t), length]` with the existing `HeatConduction1D` (reused unmodified
— the diffusion PDE is translation-invariant in x, so only the domain
*length* matters). Once the surface reaches `ablation_temperature`, it's
held there (Dirichlet-clamped) and the recession rate is set by a
surface energy balance: incident flux in excess of what conducts into
the remaining material converts to recession, absorbing
`heat_of_ablation` — `ds/dt = max(0, q_incident - q_conducted) /
(rho*heat_of_ablation)`. The grid is then rebuilt over the shrunk domain
and the temperature field is linearly interpolated onto it (a
moving-mesh/ALE scheme, always strict interpolation since the domain
only shrinks — no extrapolation risk). This was chosen over a
boundary-immobilized (Landau-transformed) PDE with a mesh-velocity
advection term — a valid, more standard alternative, but more numerically
involved than this scope needs; that alternative was worked out
analytically as a check but not implemented (see commit history/code
comments in `ablation.py`).

**Validation strategy:** to isolate the front-tracking/remeshing
algorithm from the (already-validated) diffusion scheme and from the
nonlinear convective/radiative BC, `scripts/validate_stefan.py`
initializes the material uniformly at `Tm` (the classical one-phase
idealization) and drives the front with the *exact* analytic Stefan flux
(`stefan_front_flux` — derived from the Stefan energy balance and
verified independently to reproduce `ds/dt = lambda*sqrt(alpha/t)`
exactly by construction). The run bootstraps from a small `t0 > 0`
(using the analytic front position at `t0`) to avoid the 1/sqrt(t) flux
singularity at t=0.

**Result: max relative error in front position 0.031% over a 60 s run**
(N=201, dt=0.01 s); the remaining-material temperature field stays
pinned at `Tm` throughout, matching the one-phase idealization exactly.
6 unit tests in `tests/test_ablation.py`, including a physically-driven
(non-idealized) run with `ConvectiveRadiativeBC` showing sensible
pre-heat → onset → recession → surface-pinned-at-Tm behavior.

### Phase 4 — Full ablation model, ~6 hrs
- Couple `ConvectiveRadiativeBC` (aero heating + reradiation) with the
  Phase 3 front-recession model, so the surface both heats radiatively/
  convectively *and* recedes once it reaches the ablation temperature.
- Track back-face temperature as the primary structural survival metric.
- Sanity checks: energy balance (incident − reradiated − absorbed by
  ablation − sensible heating = 0 to numerical tolerance) and limiting
  cases (heat of ablation → infinity recovers the non-ablating
  `ConvectiveRadiativeBC` solution).

### Phase 5 — Convergence study (done for Phases 1-2, repeat for Phase 3-4), ~2 hrs
`scripts/convergence_study.py` already demonstrates, on the Phase 1
problem:
- Spatial: ~2nd-order convergence (matches the central-difference scheme)
  until the fixed dt=5e-4s temporal error floor is reached at fine grids.
- Temporal: backward Euler measures ~1st order, as expected.
- Temporal: Crank-Nicolson measures only ~1st order in this problem (not
  the textbook 2nd order) because the step-change Dirichlet BC is
  non-smooth ("rough") data at t=0 — a known CN pathology (Rannacher,
  1984). Confirmed the diagnosis directly: a 2-step backward-Euler
  startup before switching to CN recovers ~20x lower error at the
  largest dt tested. The solver defaults to `theta=1` (backward Euler),
  which is unaffected and is what all validations above use.

Once ablation is added (Phase 3-4), repeat the grid/dt sweep on the full
back-face-temperature problem and report the "converged to within X% at N
nodes" statement.

### Phase 6 — Stretch: trajectory coupling and Monte Carlo, ~8-10 hrs
- Feed velocity/altitude vs. time from the existing 3-DOF intercept
  simulator into a stagnation-point heating correlation (e.g.
  Fay-Riddell or Sutton-Graves) to generate `h(t)` and `T_aw(t)` for the
  `ConvectiveRadiativeBC`.
- Run the existing Monte Carlo / sensitivity-analysis machinery over the
  thermal model: sample material properties (measured diffusivity and
  emissivity plus their lab uncertainty) and trajectory dispersion.
- Report probability of back-face temperature exceeding a structural
  limit — a proper P(failure) statement, not just a single deterministic
  run.

### Phase 7 — Stretch: multi-station nose mapping (not full 2D/3D), ~4-6 hrs
A true 2D/3D coupled ablation solve (curved mesh, sparse or ADI implicit
solve, level-set/front-tracking for a moving *surface* rather than a
moving *point*) is a much larger undertaking than this project's scope —
and isn't actually how heat-shield sizing is done in practice. Tools like
NASA's FIAT are themselves 1D, run independently at multiple body
stations along the vehicle with locally-varying heating input; this is
standard because at hypersonic heating rates the boundary layer is thin
and through-thickness conduction dominates, so surface-tangential
conduction is usually negligible.

This phase reproduces that approach instead of a full 2D/3D rewrite:
- Parametrize the nose profile by arc length / body station `sigma`
  (e.g. a simple cone or a Sutton-Graves-style blunted nose radius
  profile).
- At each station, get a local stagnation-point-equivalent heating input
  by scaling the trajectory-driven `h(t)`/`T_aw(t)` from Phase 6 with a
  station-dependent factor (e.g. the standard `cos(theta)` or
  local-radius scaling used in engineering heating correlations).
- Run the existing 1D `AblationFront1D` solver independently at each
  station (this parallelizes trivially — no coupling between stations
  needed under the thin-boundary-layer assumption).
- Assemble the per-station back-face-temperature and recession results
  into a single spatial plot: back-face temperature and total recession
  vs. body station — a genuine 2D thermal map of the nose, at a fraction
  of the cost/risk of a coupled 2D/3D solver, and consistent with actual
  industry practice for this exact problem.

## Repository layout

```
heatablate/            core package
  material.py          Material dataclass (k, rho, cp, emissivity, ablation params)
  boundary.py           DirichletBC, FluxBC, ConvectiveRadiativeBC
  solver.py             HeatConduction1D (theta-method FD solver, Thomas algorithm)
  ablation.py            AblationFront1D (moving-front recession, ALE remeshing)
  analytic.py           closed-form validation solutions (step, flux, Stefan)
  plotting.py            shared matplotlib figure-saving helper
scripts/
  validate_step_change.py
  validate_constant_flux.py
  validate_stefan.py
  convergence_study.py
tests/
  test_solver.py         unit tests (analytic agreement, conservation, steady states)
  test_ablation.py       unit tests (Stefan agreement, onset/pinning, ablation-rate scaling)
docs/
  PROJECT_PLAN.md         this file
figures/                  committed PNGs from each validation/convergence script
```

## Time log so far
- Phase 0 (core solver + BC framework): ~4 hrs
- Phase 1 (step-change validation): ~2 hrs
- Phase 2 (constant-flux validation): ~1.5 hrs
- Phase 5 (convergence study on Phases 1-2, including the CN/Rannacher
  investigation): ~2 hrs
- Static validation/convergence figures: ~1 hr
- Phase 3 (Stefan problem / ablation front-recession): ~4 hrs
- **Total so far: ~14.5 hrs of the ~30 hr budget.**
- Remaining: Phase 4 (full coupled ablation model with
  ConvectiveRadiativeBC + back-face survival tracking) ~6 hrs, Phase 5
  repeat on the coupled model ~1 hr, Phase 6 (stretch: trajectory +
  Monte Carlo) ~8-10 hrs, Phase 7 (stretch: multi-station nose map)
  ~8-10 hrs.

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
    and closed each time step with a Newton iteration (analytic dq/dTs; an
    earlier Picard fixed-point version diverged on coarse grids).
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

### Phase 4 — Full ablation model (done, ~7 hrs incl. two real bugs found via the energy-balance diagnostic)
Couples `ConvectiveRadiativeBC` (aero heating + reradiation) with the
Phase 3 front-recession model in `AblationFront1D`, so the surface both
heats radiatively/convectively *and* recedes once it reaches the
ablation temperature, with back-face temperature tracked as the primary
structural survival metric. `scripts/mission_ablation.py` runs a
representative single-hump reentry heat pulse through a carbon-phenolic-
style ablator (order-of-magnitude material properties, not a specific
measured material — this project's own measured diffusivity/emissivity
would replace these): ablation onset at t=7.8s, 6.32mm total recession,
back-face temperature rising from 300K to 573K over a 120s mission with
~12mm of margin remaining — see `figures/phase4_mission_ablation.png`.

**Energy-balance diagnostic:** `AblationFront1D.solve()` tracks cumulative
incident energy, sensible enthalpy of the remaining material, and the
sensible + latent enthalpy carried away by ablated mass, returning a
`residual_pct` (energy in vs. energy accounted for). Building this
diagnostic caught two real bugs that a "does it run and look plausible"
check would have missed:

1. **State-chattering bug.** The ablating/not-ablating branch was
   originally re-derived from a bare `T[0] >= Tm` check every step. After
   remeshing, linear interpolation can leave the surface a hair below Tm
   even while ablation is fully sustained, flipping back to the unclamped
   surface BC for that step — which, empirically, happened on exactly
   half the steps in a 3000-step run (1141 ablating / 1141 chattered
   back), each time silently skipping the recession update. Symptom: an
   ~8% energy-balance residual against an expected sub-1% (matching
   independently-measured remeshing-interpolation drift). Fixed by
   deciding the ablating state from a physical sustainability check
   (`q_incident >= q_conducted`, both evaluated from the same
   energy-consistent flux the diagnostic itself uses) rather than a raw
   temperature re-check.
2. **Missing exit-from-ablation physics.** The first fix for (1) used a
   one-way latch (once ablating, always ablating), which fixed the
   chattering but broke a genuine physical case: under a fading heat
   pulse, once incident flux drops below what's needed to sustain the
   ablation temperature, the surface must be allowed to cool back below
   Tm and recession must stop — a permanent latch instead kept forcing
   the surface to Tm indefinitely (symptom: a **-12.8%** residual on the
   full mission-pulse scenario). Fixed by making the sustainability check
   per-step rather than one-way: the Dirichlet(Tm) trial solve is kept
   only while `q_incident >= q_conducted`; once that fails, the step is
   re-solved with the ordinary surface BC and the surface cools.
   `scripts/mission_ablation.py` now shows exactly this: recession tracks
   the rising pulse, plateaus once the pulse fades past what it can
   sustain, and the surface visibly drops below Tm afterward (verified in
   `tests/test_ablation.py::test_recession_stops_once_heat_pulse_fades`).

Both bugs would have shipped invisibly without the energy-balance check —
the Stefan validation (Phase 3) couldn't have caught either of them,
since it starts already at Tm and never fades, so the state latches
immediately and never round-trips or needs to exit. After both fixes,
the mission-pulse residual is **-0.23%**, consistent with the
remeshing-interpolation drift alone (same order of magnitude as measured
independently: ~17,000 J/m² drift on a smaller run, scaling with total
recession distance).

**Limiting cases tested** (corrected from the original plan, which had
the wrong parameter): `ablation_temperature → infinity` (never reached)
exactly recovers the plain `HeatConduction1D` + `ConvectiveRadiativeBC`
solution — not `heat_of_ablation → infinity`, which still clamps the
surface at Tm once reached and so does *not* recover the unclamped
solution; that limit instead correctly drives recession to (near) zero
while the surface still pins at Tm. Both are covered in
`tests/test_ablation.py`.

### Phase 5 — Convergence study (done, ~3.5 hrs total)
`scripts/convergence_study.py` demonstrates, on the Phase 1
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

**Repeated on the full coupled ablation model (done, ~1.5 hrs):**
`scripts/convergence_study_ablation.py` runs the Phase 4 mission scenario
at increasing resolution and tracks back-face temperature at t_end (no
analytic solution exists for this problem, so convergence is measured
Richardson-style against the finest run in each sweep, not against a
closed form):

- Spatial (dt=0.01s fixed): N=51 → 1.63%, N=101 → 0.71%, N=201 → 0.24%,
  N=401 → reference. Roughly halving each doubling — consistent with the
  ALE remeshing's linear interpolation being the dominant error source
  (1st order), rather than the interior diffusion scheme (2nd order, as
  shown in Validations 1-2 above).
- Temporal (N=201 fixed): dt=0.04s → 0.019%, dt=0.02s → 0.008%,
  dt=0.01s → 0.003%, dt=0.005s → reference. Time-step error is
  negligible next to spatial error at any of these resolutions —
  confirms the grid (not the time step) is what should be refined if
  tighter accuracy is ever needed.

**Headline: back-face temperature converged to within 0.24% at 201
nodes** (vs. 401 nodes), and within 0.003% at dt=0.01s (vs. 0.005s).
See `figures/phase5_ablation_convergence.png`.

### Phase 6 — Stretch: trajectory coupling and Monte Carlo — investigated, blocked, deferred
Checked the actual `missile-intercept-tracker` repo before wiring anything
up. Every one of its 9 preset scenarios launches at 1000m altitude,
missile speed 60 m/s off the rail, target at 250 m/s, closing speeds
topping out around 300-450 m/s (~Mach 1-1.3 per the simulator's own
README) — a low-altitude, subsonic-to-transonic terminal intercept
engagement, not a hypersonic reentry trajectory. Stagnation-point heating
correlations (Sutton-Graves, Fay-Riddell) scale roughly as velocity^3;
at these speeds vs. a genuine reentry speed (~5,000-7,000 m/s), heating
would be ~2,000-5,000x smaller — the ablation model would show no
ablation and barely any surface heating at all. Feeding this simulator's
real output through the heating BC would be technically "using the real
trajectory" but a physically empty demonstration.

Rather than silently coupling a mismatched trajectory (misleading) or
quietly substituting a synthetic one (contradicts the point of "coupling
to your own simulator"), this was flagged and deferred by choice — Phase
7 was picked up instead since it doesn't depend on an external trajectory
source. Revisit if a genuinely hypersonic trajectory source becomes
available (a different simulator, or a purpose-built synthetic
reentry integrator built and labeled as such).

### Phase 7 — Stretch: multi-station nose mapping (not full 2D/3D) (done, ~2 hrs)
A true 2D/3D coupled ablation solve (curved mesh, sparse or ADI implicit
solve, level-set/front-tracking for a moving *surface* rather than a
moving *point*) is a much larger undertaking than this project's scope —
and isn't actually how heat-shield sizing is done in practice. Tools like
NASA's FIAT are themselves 1D, run independently at multiple body
stations along the vehicle with locally-varying heating input; this is
standard because at hypersonic heating rates the boundary layer is thin
and through-thickness conduction dominates, so surface-tangential
conduction is usually negligible.

`scripts/multistation_nose_map.py` reproduces that approach: a blunted
(spherical-cap) nose is parametrized by the angle `phi` from the
stagnation point (0° to 70°, 8 stations), with local heating scaled off
the Phase 4 stagnation-point pulse by the standard cosine-law Newtonian-
flow approximation `h(phi,t) = h_stagnation(t) * cos(phi)` — a first-order
engineering approximation (the more accurate Lees correlation adds a
boundary-layer-thinning correction near the stagnation point; cosine
captures the right qualitative falloff and is standard in introductory
hypersonic aerothermal analysis). `AblationFront1D` is run independently
at each station — no coupling needed between stations under the
thin-boundary-layer assumption.

**Result** (see `figures/phase7_nose_map.png`): recession falls from
6.32mm at the stagnation point (phi=0°, exactly reproducing the Phase 4/5
reference case as a consistency check) to 0.32mm at phi=60°, and ablation
stops entirely by phi=70° — the heating there simply never reaches the
ablation temperature. Back-face temperature falls similarly, 573.5K →
399.2K across the mapped stations. A genuine spatial thermal map of the
nose, at a fraction of the cost/risk of a coupled 2D/3D solver.

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
  mission_ablation.py     Phase 4: coupled aero-heating + ablation mission run
  convergence_study_ablation.py   Phase 5 repeat: convergence on the coupled model
  multistation_nose_map.py   Phase 7: back-face T / recession vs. body station
tests/
  test_solver.py         unit tests (analytic agreement, conservation, steady states)
  test_ablation.py       unit tests (Stefan agreement, energy balance, limiting cases, exit-from-ablation)
docs/
  PROJECT_PLAN.md         this file
figures/                  committed PNGs from each validation/convergence/mission script
```

## Time log so far
- Phase 0 (core solver + BC framework): ~4 hrs
- Phase 1 (step-change validation): ~2 hrs
- Phase 2 (constant-flux validation): ~1.5 hrs
- Phase 5 (convergence study on Phases 1-2, including the CN/Rannacher
  investigation): ~2 hrs
- Static validation/convergence figures: ~1 hr
- Phase 3 (Stefan problem / ablation front-recession): ~4 hrs
- Phase 4 (full coupled ablation model, including diagnosing and fixing
  the two energy-balance bugs above): ~7 hrs
- Phase 5 repeat (grid/dt convergence on the coupled ablation model): ~1.5 hrs
- Phase 6 investigation (cloning and checking the intercept simulator's
  actual flight regime, concluding it's a mismatch): ~0.5 hr
- Phase 7 (multi-station nose thermal map): ~2 hrs
- **Total so far: ~25.5 hrs of the ~30 hr budget.**
- The core validated model (Phases 0-5) and one of the two stretch goals
  (Phase 7) are complete. Phase 6 remains blocked on a genuine hypersonic
  trajectory source (see above) — ~4.5 hrs of budget left if picked up.

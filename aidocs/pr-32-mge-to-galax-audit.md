# PR #32 audit: Implement `to_galax()` for MGE composite potentials

Audit date: 2026-08-26

Pull request: #32, `mge-to-galax` into `main`

## Overall judgment

PR #32 needs focused corrections before merge, rather than an architectural
revision. The MGE-to-`galax` conversion mathematics is sound, the exact locked
`galax` implementation agrees with TNT's parameter mapping and normalization,
and the full Linux test, lint, and documentation suites pass. The principal
unresolved concern is how TNT handles deprojections that are numerically finite
but violate its documented intrinsic-axis convention.

TNT is a new package. Renaming the MGE potential types is therefore not treated
as a compatibility problem, and no old-name/new-name compatibility tests are
recommended.

## Architectural summary

PR #32 makes the triaxial MGE potentials operational:

- `build_mges()` reads angular MGEs and converts them to physical coordinates
  using the configured system distance.
- `TriaxialLightMGEPotential` multiplies a light MGE by its `ml`
  mass-to-light parameter.
- `TriaxialMassMGEPotential` multiplies a mass MGE by the dimensionless
  `mge_mass_scale` parameter.
- The physical mass MGE is deprojected using the global `theta`, `phi`, and
  `psi` viewing angles.
- Each intrinsic Gaussian becomes a
  `galax.potential.TriaxialGaussianPotential` with `r_s = sigma`, `q1 = p =
  B/A`, `q2 = q = C/A`, and
  `m_tot = I p q (2 pi)^(3/2) sigma^3`.
- The Gaussian potentials are combined in a `galax` `CompositePotential`.

That mass formula and the `p -> q1`, `q -> q2` mapping agree with the exact
locked `galax` implementation.

Configuration declarations retain their original units. Runtime `Quantity`
construction occurs in the parameter generator, while the MGE
angular-to-physical conversion occurs when runtime MGEs are built.

## Findings

### High: Physical deprojection validity and axis convention are not enforced

`AbstractMGE.deproject_triaxial()` in `tnt/mge.py` documents `p = B/A` and
`q = C/A`, implying the ordered-axis convention `0 < q <= p <= 1`. The
implementation produces square-root `NaN`s for some invalid combinations, but
other viewing geometries produce finite values outside that convention.
`_triaxial_mge_potential()` in `tnt/potential/triaxial_mge.py` passes those
values directly to `galax` without checking them.

The committed NGC6278 light MGE was evaluated at its configured angles,
`theta = 1`, `phi = 0.5`, and `psi = 0`. It produced approximately:

```text
p = [1, 1, 1, 1, 1, 1]
q = [1.16, 1.36, 1.00, 2.06, 1.00, 2.06]
```

These ellipsoids are mathematically finite and correctly reproduce the
projected MGE, but `q > 1` contradicts TNT's stated `C/A` short-to-long-axis
convention. This matters if later orbit machinery assumes the intrinsic x, y,
and z axes are ordered long/intermediate/short.

The configured search ranges in
`tests/integration_tests/configuration.yaml` also include singular endpoints
such as `theta = 0` and `phi = 0`, which can produce genuine `NaN`s.

Issue #30 covers general potential-domain validation, but its description left
the final MGE schema to the viewing-geometry work. PR #32 should therefore
either enforce or explicitly represent MGE deprojection validity, or document
that unordered intrinsic axes are accepted and confirm that downstream orbit
code supports them.

A plain eager Python exception may not be sufficient because viewing angles
will eventually be traced by JAX. A JAX-compatible validity result or an
explicit failed-model path may be preferable.

Actionable locations:

- `tnt/mge.py`, `AbstractMGE.deproject_triaxial()`
- `tnt/potential/triaxial_mge.py`, `_triaxial_mge_potential()`
- `tests/integration_tests/configuration.yaml`, the `stars` viewing-angle
  ranges

### Medium: Runtime MGE and projected-binning coordinates become inconsistent

`build_mges()` now always returns physical MGEs. Meanwhile,
`ModelIterator.from_configuration()` constructs spatial binnings in angular
coordinates and passes both objects into kinematics.

This does not fail today because the kinematics `design_matrix()` methods are
still scaffolding. Once they use `mge.get_projected_mass(binning)`, a physical
MGE and angular binning will be dimensionally incompatible.

Before implementing those projections, TNT needs one consistent policy:

- convert both MGEs and binnings to physical coordinates;
- keep the shared observational MGE angular and create a physical copy for
  potentials; or
- make each relevant consumer perform the conversion explicitly.

This need not force an architectural rewrite of PR #32, but it should be an
explicit decision or tracked follow-up.

Actionable locations:

- `tnt/mge.py`, `build_mges()`
- `tnt/model_iterator.py`, `ModelIterator.from_configuration()`
- `tnt/kinematics/base.py`, the future `design_matrix()` boundary

### Medium: The advertised subclass registration is only partially dynamic

`tnt/potential/components.py` says a newly imported subclass participates
automatically. In practice:

- only direct subclasses returned by
  `AbstractPotentialComponent.__subclasses__()` are discovered;
- duplicate `_type` values silently overwrite one another;
- configuration validation still requires editing `_MGE_POTENTIAL_TYPES`;
- parameter dimensions still require editing `_MGE_RAW_DIMENSIONS`; and
- the implementation module must still be imported from `tnt.potential`.

PR #33 already expands these parallel registries for axisymmetric MGEs. TNT
should either use an explicit registry with duplicate detection, as the
kinematics layer does, or place the type identifier, parameter dimensions, and
schema metadata on each component class and derive one registry from it.

This is not a current runtime bug for the two direct triaxial subclasses, but
the extensibility claim should be narrowed or the implementation consolidated.

Actionable locations:

- `tnt/potential/components.py`, `AbstractPotentialComponent.resolve()`
- `tnt/potential/registry.py`, `_MGE_RAW_DIMENSIONS`
- `tnt/configuration/validation.py`, `_MGE_POTENTIAL_TYPES`

### Low: One documented example is no longer executable

The `Constructing kinematics` example in `docs/source/configuration.md` calls
the new four-argument `build_mges()` with only three arguments. It needs to
resolve and pass `system_attributes.distance`, as the earlier example in the
same document already does.

Actionable location:

- `docs/source/configuration.md`, `Constructing kinematics`

### Low: Documentation overstates two behaviors

- `AbstractMGE.deproject_triaxial()` says an unsolvable deprojection produces
  `NaN`. Some out-of-convention solutions are finite instead.
- `build_mges()` says every consumer needs a physical MGE. Projected
  observational consumers naturally operate in angular coordinates unless
  their binnings are converted too.

Actionable locations:

- `tnt/mge.py`, the `deproject_triaxial()` and `build_mges()` docstrings
- `docs/source/configuration.md`, the MGE and spatial-binning construction
  sections

## Missing or weak tests

The existing `to_galax()` tests deliberately use circular Gaussians. They
verify normalization, light-to-mass conversion, mass scaling, and component
summation, but cannot detect:

- swapped `p`/`q` or `q1`/`q2`;
- incorrect density falloff along the intrinsic y and z axes;
- handling of genuinely triaxial deprojections; or
- singular or convention-violating viewing geometries.

Recommended additions:

1. Use one known valid non-spherical forward-projected Gaussian and compare
   the resulting `galax` density along all three principal axes with the
   analytic intrinsic Gaussian.
2. Exercise `Potential.to_galax()` using the realistic integration
   configuration rather than stopping before it, as the current model-search
   test does.
3. Define and test expected behavior for singular angles and finite `p`/`q`
   values outside the chosen convention.
4. Add a JIT regression around MGE conversion and evaluation.

A direct manual `jax.jit` of MGE `to_galax()` plus potential evaluation
succeeded during this audit. Vectorization was not separately tested.

## Checks run

The following checks used Colima and the documented Linux/amd64 Docker
environment with the exact updated lockfile:

- Docker image rebuild: passed.
- Full test suite: 283 passed, with three existing JAX/multiprocessing or
  dependency warnings.
- `ruff check .`: passed.
- Sphinx build with warnings treated as errors: passed.
- Direct locked-`galax` source and formula inspection: completed.
- Direct JAX-jitted MGE potential evaluation: passed.
- Committed NGC6278 deprojection inspection: finite, but returned `q > 1` as
  described above.
- `git diff --check`: passed.

Not run:

- Apple Silicon execution.
- GitHub CI, because PR #32 had no reported checks at audit time.
- Vectorized MGE-potential construction or evaluation.

## Interaction with active work

- PR #33 (`axisym-mge`) is stacked on PR #32 and should be rebased after PR
  #32 is finalized.
- PR #33 extends the same scattered type, validation, and dimension registries,
  so the registry decision is cheaper before the axisymmetric work is merged.
- Issue #30 is the natural home for general exact-key and physical-domain
  validation, but it should be updated explicitly with the MGE
  coupled-viewing-angle problem.

## Recommended review and correction sequence

1. Ask Prash to confirm whether TNT requires ordered intrinsic axes
   `0 < q <= p <= 1`, or permits arbitrary positive axis scales with
   potentially relabelled principal axes.
2. Decide how invalid viewing-angle search points become failed models without
   contaminating orbit integration with `NaN`s.
3. Add one genuinely flattened/triaxial analytic density test and one realistic
   end-to-end `to_galax()` test.
4. Correct the broken documentation example and the two overstated claims.
5. Decide or record the MGE-versus-binning coordinate policy before kinematic
   projections are implemented.
6. Optionally consolidate the component registry before rebasing PR #33.
7. Rerun the Linux test, lint, and documentation checks.

## Questions requiring an astrophysical or design decision

1. Must intrinsic axes always be ordered so that `A >= B >= C`, or are `p`
   and `q` merely y/x and z/x scale ratios that may exceed one?
2. Should invalid deprojection points be rejected before orbit integration,
   represented as failed models, or permitted to propagate `NaN`s through
   JAX?
3. Should the shared runtime MGE registry be angular, physical, or provide
   both representations?
4. Should viewing angles remain the public search parameters, or should the
   eventual `(p, q, u)` parameterization be the preferred way to guarantee
   valid geometries?

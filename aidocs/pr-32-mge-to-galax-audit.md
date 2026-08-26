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

## Response

Independently re-verified every finding against the code before acting on
it, including reproducing the NGC6278 numbers (`q` up to 2.06 under the
committed viewing angles) directly. Addressed the full recommended sequence:

1. **Axis convention: `0 < q <= p <= 1` is required.** Both
   `AbstractMGE.deproject_triaxial` and `deproject_axisymmetric`
   (`tnt/mge.py`) now raise a new `MGEDeprojectionError` -- unifying the old
   "unsolvable produces `nan`" case with the new "finite but out-of-convention"
   case under one check, since `nan` already fails the same comparison. No
   shared `InvalidPotentialError` base was introduced: `MGEDeprojectionError`
   is currently the only way a `Potential` can fail to build, so there's
   nothing yet to unify it with (module-local `ValueError` subclass, same
   pattern as `ModelSearchStateError`/`ConfigurationCompatibilityError`).
2. **Invalid points become failed models, before orbit integration is
   attempted.** Deprojection was moved out of `to_galax()` and into the two
   MGE composite components' own construction
   (`AbstractPotentialComponent._build`, `tnt/potential/triaxial_mge.py`) --
   so an invalid viewing geometry now fails at the point a proposed
   parameter set becomes a `Potential`, not later when something tries to
   integrate orbits in it. `to_galax()` is now a pure, non-raising
   conversion of an already-validated, cached `deprojected` field.
   `Model.potential` (`tnt/model.py`) became `Potential | None` to
   accommodate this -- `None` means construction itself failed -- and a new
   `valid_potential: bool` field (backed by a new `AllModels` column)
   distinguishes that case from a failed orbit integration or weight solve.
   `ModelIterator._evaluate` wraps `build_potential` in its own
   `except MGEDeprojectionError`, separate from and preceding the existing
   bare `except Exception` around `generate_orbit_library`.
   `rescale()` on the two MGE composite types was updated to scale the
   cached `deprojected.I` directly (exact, since a mass rescale never
   changes the deprojection's shape) rather than re-deriving -- and
   therefore never re-validating -- the deprojection on every rescale.
3. **New tests added**, per the two recommendations here: a genuinely
   triaxial (`p != q != 1`) `to_galax()` test comparing `galax`'s `.density()`
   independently along all three intrinsic axes against the closed-form
   Gaussian (`test_triaxial_light_mge_to_galax_density_matches_analytic_along_each_axis`,
   `tests/unit_tests/test_potential.py`) -- the existing tests deliberately
   used circular MGEs and couldn't have caught a `p`/`q1` <-> `q`/`q2` swap;
   and a real end-to-end `Potential.to_galax()` call against the full
   resolved example configuration
   (`test_potential_to_galax_succeeds_against_the_resolved_example_configuration`,
   `tests/integration_tests/test_model_search.py`), since every other test
   there fakes `generate_orbit_library` and so never actually reaches
   `to_galax()`. Also added: build-time-vs-`to_galax()` error-timing
   coverage, an `_evaluate`-level test for the new invalid-potential path,
   and a triaxial (not just axisymmetric) `MGEDeprojectionError` case.
4. **Documentation corrected.** The broken `build_mges()` call in
   "Constructing kinematics" (`docs/source/configuration.md`) now passes
   `system_attributes.distance`. Both overstated claims are fixed: the
   deprojection docstrings no longer say an unsolvable point produces `nan`
   (it now raises), and `build_mges`/`build_spatial_binnings`'s docstrings
   now state precisely that both are physical by construction -- true as of
   item 5 below, so no longer an overstatement.
5. **MGE-versus-binning coordinate policy: converted both to physical.**
   `build_spatial_binnings` (`tnt/spatial_binnings.py`) now takes a
   `distance` argument and converts every returned `ProjectedBinning` via
   its existing `angular_to_physical`, exactly mirroring `build_mges`'s
   existing pattern. `ModelIterator.from_configuration` threads the
   already-computed `distance` through. MGEs and spatial binnings loaded
   this way are now always dimensionally consistent for a future consumer
   needing both (e.g. `AbstractMGE.get_projected_mass`).
6. **Registry consolidation: deferred, not done here.** Prash would rather
   the eventual fix follow this section's other suggested direction --
   metadata-on-class, with one registry derived from it -- which is a real
   redesign, not something to fold into this correction pass. Only did the
   cheap part: narrowed `tnt/potential/components.py`'s docstring and
   `AbstractPotentialComponent.resolve`'s comment, which overstated the
   dynamic-registration guarantee (it's non-recursive and has no
   duplicate-`_type` detection, unlike `tnt.kinematics`'s equivalent
   registry). Tracked as a follow-up to land before PR #33 (`axisym-mge`,
   stacked on this branch) extends the same scattered registries further.
7. **Rechecked locally**: full `pytest` (291 passed), `ruff check .`
   (clean), and a Sphinx build with warnings treated as errors (clean). Did
   not re-run the Colima/Linux Docker parity setup the audit itself used --
   happy to if that's wanted before merge.

Answers to the design questions:

1. **Ordered axes are required**: `0 < q <= p <= 1` always, per item 1 above.
2. **Invalid points become failed models**: enforced at build time (item 2).
3. **Both MGEs and spatial binnings are now physical** (item 5) -- not
   "provide both representations"; a consumer needing angular coordinates
   would need its own explicit path, but nothing currently needs that.
4. **Viewing angles stay the native/required parameters.** Every
   parameterization has to resolve to `(theta, phi, psi)` before
   deprojection can happen at all, so they're the right thing to require
   now; the `(p, q, u)` parameterization may still become the *preferred*
   way to guarantee valid geometries once its conversion formula is
   confirmed, but that's additive, not a replacement.

### Follow-up: JAX-compatibility of the eager validity check

The audit's caveat on this point ("a plain eager Python exception may not
be sufficient because viewing angles will eventually be traced by JAX...")
is correct and deliberately not addressed by the above. `MGEDeprojectionError`
(`tnt/mge.py`) now carries an explicit docstring note on this. Two points
worth recording here:

- It's not just `deproject_triaxial`/`deproject_axisymmetric` that are
  non-traceable now -- the validity check itself (`bool(jnp.all(...))`,
  `.nonzero()`) means neither function can be called under `jax.jit`/
  `jax.vmap` even in isolation, regardless of anything above them in the
  call stack.
- Fixing only that, however, would not make `ModelIterator._evaluate`
  jittable: `generate_orbit_library`/weight solving are still
  `NotImplementedError`, and `_evaluate`'s own failure handling around them
  (`except Exception`, returning a variable-length `list[Model]`) is
  exactly as eager and non-traceable as the MGE check is. The MGE piece is
  a small fraction of what a jit-compatible `_evaluate` would actually
  require.

Deliberately not building a JAX-compatible validity mechanism now -- doing
so ahead of the orbit-integration implementation would mean guessing at a
design (per-model jit? `vmap` over a batch of proposed points, with failure
becoming a masked/NaN-flagged array rather than a shorter list?) before
that work forces the real decision. Revisit this -- MGE deprojection
validity and `_evaluate`'s failure handling together -- once orbit
integration/weight solving are implemented and a jit/vmap strategy for the
search loop is chosen.

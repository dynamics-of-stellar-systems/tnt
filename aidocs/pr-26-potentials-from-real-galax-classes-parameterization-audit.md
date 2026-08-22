# PR #26 Audit: Potentials from Real Galax Classes with Parameterization Support

## Overall judgment

PR #26 is **not ready to merge**. Its broad architecture is usable, but it needs focused corrections plus a small redesign of the potential-type/parameterization registry. The most serious problem is a silent mismatch between logarithmic search coordinates and physical runtime parameters, which can produce astrophysically wrong potentials while all tests pass.

I found:

- 1 critical issue
- 3 high-priority issues
- 5 medium/low issues
- a definite stacking conflict with PR #24
- a failing documentation build

The repository remained on clean `main`; I did not check out another branch, modify repository files, change either PR, or post comments.

## 1. Architectural summary

The intended flow is:

```text
resolved configuration
  → parameter generator proposes raw/search coordinates
  → ModelIterator overlays proposed values
  → build_potential resolves each component
  → optional parameterization converts raw → native galax parameters
  → Potential composes the runtime components
  → orbit library / weight solver evaluate it
  → optional mass rescaling modifies native parameters
  → inverse parameterization converts native → configured representation
  → Model.raw_parameters → AllModels columns
```

What PR #26 adds:

- `potential.<name>.type` can name a `galax.potential` class such as `PlummerPotential` or `NFWPotential`.
- Native parameter dimensions are read from galax `ParameterField` metadata.
- Native components are represented as immutable-looking Equinox modules containing unxt quantities.
- `concentration_m200` converts NFW `(c, M_200c)` into galax-native `(m, r_s)`:
  - \(\rho_\mathrm{crit}=3H_0^2/(8\pi G)\)
  - \(r_{200}=(3M_{200}/(4\pi\,200\,\rho_\mathrm{crit}))^{1/3}\)
  - \(r_s=r_{200}/c\)
  - \(m=M_{200}/[\ln(1+c)-c/(1+c)]\)
- The inverse solves \(c^3/g(c)\) numerically and reconstructs `M_200`.
- `Potential.rescale(s)` applies:
  - mass → \(s\)
  - speed/amplitude → \(\sqrt{s}\)
  - length, angle and dimensionless shape parameters → unchanged
- `Model.raw_parameters` and `AllModels` report configured names such as `dh.c` and `dh.M_200`, rather than native `dh.m` and `dh.r_s`.

Parameter meanings:

| Parameter category | Runtime unit/transformation | Mass rescaling |
|---|---|---|
| Native mass | Internal mass | × `mass_scale` |
| Native length | Internal length | unchanged |
| Native angle | Internal angle | unchanged |
| Native dimensionless | no unit | unchanged |
| Native speed | Internal speed | × √`mass_scale` |
| Frequency/time/wavenumber | Derived from unit system | rescaling deliberately unsupported |
| NFW `c` | dimensionless concentration \(r_{200}/r_s\) | recomputed after native rescaling |
| NFW `M_200` | mass enclosed at 200× critical density | recomputed after native rescaling |
| Light-MGE `ml` | mass/power | × `mass_scale` |
| Mass-MGE `mge_mass_scale` | dimensionless normalization | × `mass_scale` by special rule |
| MGE `q`, `p`, `u` | dimensionless geometry/search values | currently only carried as scaffold |

Fully implemented:

- Native Plummer/NFW and similar scalar galax potential construction.
- NFW `concentration_m200` forward and inverse conversion.
- Selected-dimension rescaling.
- Composition of implemented galax components.
- Model/AllModels parameter reporting.
- Parameter plumbing through `ModelIterator`.

Still scaffolding:

- Light- and mass-MGE `to_galax()`.
- The MGE viewing-geometry transformation.
- Orbit sampling, dithering, integration and orbit-library rescaling.
- Real weight solving.
- Grid-search parameter generation.
- Consequently, the integration test does not integrate a real orbit or evaluate the MGE potential.

## 2. Prioritized findings

### Critical — Logarithmic coordinates are passed directly as physical values

TNT’s documented unit logic treats a logarithmic parameter value as a base-10 search coordinate: unit conversion adds a logarithmic offset. PR #26 then wraps that coordinate directly in a physical `Quantity` without exponentiating it.

A Linux probe confirmed:

```text
configured logarithmic m_tot value = 5
runtime Plummer mass = 5 Msun
```

Under the existing logarithmic contract, that should be \(10^5\,M_\odot\).

Worse, the integration test explicitly expects `5 Msun`, locking in the incorrect behavior. Its NFW example marks `M_200` logarithmic but supplies `1e12`, mixing physical and log-space conventions in the same configuration.

References:

- [`tnt/units.py:229`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/units.py#L229)
- [`tnt/potential.py:405`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L405)
- [`tests/integration_tests/test_model_search.py:210`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tests/integration_tests/test_model_search.py#L210)
- [`tests/integration_tests/configuration.yaml:28`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tests/integration_tests/configuration.yaml#L28)

This must be resolved before judging the astrophysics produced by the PR.

### High — “Any galax potential class” is not a valid configuration contract

The dispatcher accepts any subclass of `AbstractPotential`, but the configuration representation only supports scalar numeric parameters converted to `Quantity`.

The pinned galax release also exports:

- composite and precomposed potentials;
- transformed/translated/decorator potentials requiring nested runtime objects;
- multipole potentials requiring array coefficients and integer `l_max`;
- classes with Boolean hyperparameters;
- parameterless potentials;
- time-dependent fields.

These cannot be faithfully represented by the current scalar `parameters.<name>.value` schema. Plain hyperparameters are incorrectly wrapped as dimensionless quantities. Some abstract/composite classes are also accepted by the subclass check.

References:

- [`tnt/potential.py:73`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L73)
- [`tnt/potential.py:377`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L377)
- [`tnt/potential.py:405`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L405)
- [`tnt/potential.py:498`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L498)
- [`docs/source/potential.md:121`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/docs/source/potential.md#L121)

Recommendation: define an explicit registry of TNT-configurable galax classes, or introduce per-class adapters capable of declaring scalar, array, Boolean, nested-object and static constructor fields.

### High — Missing parameter-schema and physical-domain validation allows silent invalid potentials

Validation checks generic numeric shape but not:

- the exact expected parameter names;
- missing native or parameterized fields;
- unknown fields;
- positivity of masses, radii, concentrations and scale factors;
- axis-ratio constraints;
- finiteness of converted native results.

Focused probes showed:

- `c = 0` produces infinite `m` and `r_s`;
- negative `M_200` produces negative `m` and `nan r_s`;
- an extra NFW raw parameter is silently discarded;
- missing fields surface as raw `KeyError`s.

References:

- [`tnt/configuration_validation.py:345`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/configuration_validation.py#L345)
- [`tnt/potential.py:214`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L214)
- [`tnt/potential.py:405`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L405)

The registered type/parameterization adapter should own expected keys, dimensions and physical constraints, and produce path-specific configuration errors.

### High — PR #24 cannot be combined mechanically

Both PRs alter `tnt/units.py` and `tnt/model_iterator.py`; a three-way merge predicts direct conflicts in both.

The semantic conflict is more important:

- PR #24 preserves `H0` as `{value, unit}` until runtime.
- PR #26 calls `float(cosmological_parameters["H0"])`.
- After PR #24, that will receive a mapping and fail.
- PR #24 creates one normalized runtime copy of potential settings for both generator and potential construction.
- PR #26 currently normalizes potential declarations during configuration preparation.

References:

- [PR #24 `tnt/units.py:90`](https://github.com/dynamics-of-stellar-systems/tnt/blob/a3eddcd0b21ef59eaefa0acf746ab36bddaf0d75/tnt/units.py#L90)
- [PR #24 `tnt/model_iterator.py:144`](https://github.com/dynamics-of-stellar-systems/tnt/blob/a3eddcd0b21ef59eaefa0acf746ab36bddaf0d75/tnt/model_iterator.py#L144)
- [PR #26 `tnt/potential.py:237`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L237)
- [PR #26 `tnt/units.py:103`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/units.py#L103)

Merge PR #24 first, then rebase PR #26 and adapt it to PR #24’s runtime conversion boundary. Runtime cosmology should be normalized explicitly alongside potential settings.

### Medium — `galax_type` is a dynamic string PyTree leaf

`GalaxPotentialComponent.galax_type` is not marked static. Consequently:

- `eqx.filter_jit` works;
- the resulting galax potential works under `jax.jit` and `vmap`;
- direct `jax.jit` over the TNT component fails because JAX sees the string leaf.

Reference:

- [`tnt/potential.py:481`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L481)

The structural type identifier should probably be an Equinox static field, and JIT expectations should be tested and documented.

### Medium — Parameterization metadata is split across registries

`Parameterization` bundles forward and inverse functions, but raw dimensions live in a separate mapping. Expected keys, constraints and rescaling policy live elsewhere again. These can drift as more parameterizations are added.

References:

- [`tnt/potential.py:59`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L59)
- [`tnt/potential.py:170`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L170)
- [`tnt/potential.py:311`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L311)

Use one descriptor containing raw fields, native fields, dimensions, validation, conversion, inversion and any per-field rescaling rules.

### Medium — The dimension-based rescaling design remains unsafe for future fields

The code correctly recognizes that physical dimension alone does not determine rescaling behavior—`Omega` and amplitude velocities differ—but still uses one global exponent for every parameter classified as `speed`.

References:

- [`tnt/potential.py:115`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L115)
- [`tnt/potential.py:504`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L504)

This works for the tested logarithmic-potential amplitudes, but future fields with the same dimension could be silently mis-scaled. Rescaling should be declared per native field or adapter.

### Medium — MGE behavior needs an explicit design decision

Light-MGE `ml` has the correct mass/power dimension. Mass-MGE now requires a dimensionless `mge_mass_scale`, even though the base documentation says the MGE is already mass-calibrated. That may be useful for uniform rescaling, but it changes the scientific schema and should be intentional.

Neither MGE type can yet produce a gravitational potential.

References:

- [`tnt/configuration_validation.py:326`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/configuration_validation.py#L326)
- [`tnt/potential.py:537`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L537)
- [`tnt/potential.py:578`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/potential.py#L578)

### Low/medium — Configuration preparation now imports the full JAX potential stack

`tnt.units` imports potential metadata from `tnt.potential`, which imports galax, JAX, MGE and orbit modules. This weakens the lightweight configuration/runtime boundary and caused the Linux suite to warn about forking after JAX initialized.

Reference:

- [`tnt/units.py:17`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/units.py#L17)

A lightweight potential-schema module would avoid the dependency inversion.

## 3. Missing or weak tests

Important additions:

- Logarithmic raw coordinate → physical runtime value for mass, length, `ml`, `c` and `M_200`.
- Explicit decision/test for whether `AllModels` stores physical values or log coordinates.
- Exact missing/unknown parameter keys for every registered parameterization.
- `c <= 0`, `M_200 <= 0`, `r_s <= 0`, nonfinite values and invalid axis ratios.
- NFW inverse against an independent high-precision root solver; current round-trip tests reuse the same formulas.
- Use `log1p(c)` and test small concentrations; `_nfw_g` currently suffers avoidable cancellation.
- Independent cosmology/unit-system cases, including SI-like and galactic internal units.
- A support matrix across every advertised galax type.
- Boolean, array and static hyperparameter behavior if those classes are intended to be supported.
- Direct `jax.jit`, `eqx.filter_jit`, `vmap` and PyTree-serialization tests.
- Mass-MGE construction/type mismatch/rescaling tests.
- A real `Potential.to_galax()` integration test without an MGE component.
- Eventually, an analytic force/acceleration test as well as potential-value tests.
- The current “end-to-end” test fakes orbit generation and never calls the MGE `to_galax()` path.

## 4. Documentation issues

The strict documentation build fails on three broken internal links:

- [`docs/source/potential.md:8`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/docs/source/potential.md#L8)
- [`docs/source/potential.md:22`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/docs/source/potential.md#L22)
- [`docs/source/potential.md:52`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/docs/source/potential.md#L52)

Additional inaccuracies:

- [`docs/source/configuration.md:154`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/docs/source/configuration.md#L154) still lists the old `nfw`/`plummer` identifiers.
- [`docs/source/units.md:85`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/docs/source/units.md#L85) still documents Plummer `m`/`a`, not galax-native `m_tot`/`r_s`.
- `potential.md` overstates support for every galax class.
- [`docs/source/potential.md:152`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/docs/source/potential.md#L152) says `q_min`, `p_min`, `u`, while the example configuration uses `q`, `p`, `u`.
- [`tnt/model_iterator.py:11`](https://github.com/dynamics-of-stellar-systems/tnt/blob/3ae71b35ddb1df43da2da4c8ad7b7c394c2235de/tnt/model_iterator.py#L11) still says `build_potential` is unimplemented.

## 5. PR #24 and other active work

Recommended order:

1. Finish and merge PR #24.
2. Rebase PR #26 onto the merged unit-preservation boundary.
3. Normalize one runtime potential-settings copy and one runtime cosmology representation.
4. Resolve logarithmic-coordinate semantics there.
5. Make the parameter generator and potential builder consume the same representation.
6. Re-run compatibility tests for physically equivalent unit declarations.
7. Then address issue #25’s cross-component prior/mass-ratio concept.

PR #26’s removal of the cross-component concentration/mass-ratio parameterization is correct: that relation belongs above per-component construction, likely in the parameter-space/prior layer.

Changing old type identifiers (`plummer`, `nfw`) to galax class names will also make old configurations and historical model searches incompatible. That should be an explicit migration decision and documented.

## 6. Checks run

Against an immutable archive of PR head `3ae71b3`, using Colima Linux `x86_64` and the exact locked dependencies:

- `pytest -q`: **270 passed**, 3 warnings.
- `ruff check .`: **passed**.
- `sphinx-build -E -b html -W`: **failed**, 3 broken-link warnings.
- `git diff --check`: **passed**.
- Focused numerical probes: logarithmic handling, invalid NFW inputs, inverse vectorization.
- Focused JAX probes:
  - galax potential `jax.jit`: passed.
  - galax potential `vmap`: passed.
  - TNT component `eqx.filter_jit`: passed.
  - direct `jax.jit` over TNT component: failed on dynamic `galax_type` string.

Not run:

- Ruff format check.
- Apple Silicon execution.
- Native Intel macOS tests, because the locked JAX version has no wheel there.
- Real orbit integration, MGE potential evaluation, grid search or weight solving, because those layers remain scaffolding.
- GitHub CI: the PR currently has no reported checks.

GitHub still reports no reviews, inline comments or CI runs. The PR is mergeable but blocked by policy/review status.

## 7. Practical human review sequence

1. Decide and document logarithmic parameter semantics.
2. Decide whether AllModels stores physical values or search coordinates.
3. Restrict the supported galax class set or design richer adapters.
4. Review NFW definitions and accepted physical domains.
5. Decide the mass-MGE normalization contract and MGE geometry names.
6. Merge PR #24 and rebase PR #26.
7. Review the unified runtime unit/cosmology boundary.
8. Review rescaling semantics per native field.
9. Review JAX static fields and compilation expectations.
10. Add missing numerical and validation tests.
11. Correct all affected docs and require the `-W` build to pass.
12. Only then review the remaining implementation details line by line.

## 8. Decisions needed from Prash or you

- Does `logarithmic: true` mean values are stored as \(\log_{10}\) coordinates? Existing unit conversion says yes.
- Should AllModels report physical values, log coordinates, or both?
- Is TNT intentionally supporting every galax class, or only a curated astrophysical subset?
- Is `M_200` specifically present-day `M_200c`, or must redshift/full cosmology be represented?
- Is uniform rescaling defined as scaling native NFW `m` at fixed `r_s`, thereby changing `c` and `M_200`?
- Must a mass MGE always have `mge_mass_scale`, or should its file-defined mass be sufficient until optional rescaling?
- Are the intended triaxial search coordinates `q,p,u`, `q_min,p_min,u`, or viewing angles `theta,phi,psi`?
- Is float32 acceptable for production halo conversion and orbit construction, or should TNT enable/use float64 for this boundary?
- Should time-dependent, transformed, multipole and composite galax classes ever be configurable?

The best path is focused correction rather than a wholesale rewrite: preserve the overall runtime-object layering, but make parameter metadata a coherent registry, settle log/physical coordinates, and rebase onto PR #24 before proceeding.

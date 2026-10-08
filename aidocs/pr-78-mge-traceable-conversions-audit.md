# PR #78 audit: Add traced MGE shape helpers and preserve coordinate units

Branch: `codex/mge-traceable-conversions`. Base: `main` (already includes #77).

## Questions from Prash

None outstanding — Thomas's own open question (whether to trim the
extreme-value test cases in `test_mge_parameterized_traceable.py` for runtime
cost) appears addressed by the final commit, which raises the documented
Colima memory allocation from 2GB to 4GB rather than removing tests.

## Scope

Extends #77's JAX-traceable native MGE construction to the three standalone
shape conversions (`q_min`, `pqu`, `T_maj_min`): each gets a `with_validity`
counterpart sharing the diagnostic path's numerical predicates and zero
placeholders on rejection. Two substantive changes ride along:

- Shape-geometry validity (`_oblate_geometry`, the triaxial/T-parameterization
  equivalents) is now checked independently of the unscaled MGE template's
  integrated luminosity/mass, so an extreme intensity value can't mask a
  shape error or vice versa (`_oblate_candidate` now composes
  `_oblate_geometry` + `_deprojected_valid` rather than inlining both checks
  together).
- `ParameterConstraint.unit`: `""` now means "compare as a unit-free ratio"
  (distinct from `unit=None`, "use the value's own unit"), so a scaled
  dimensionless declaration compares and round-trips correctly, and inverse
  reporting (`_inclination_to_qmin`, `_tpp_to_pqu`, `_tpp_to_tmajmin`) now
  preserves the proposal's declared unit via `.to(declared_units.get(...))`
  instead of always returning bare `""`.

## Verification performed

- Diffed `main...codex/mge-traceable-conversions` directly (1705 lines, 20
  files) rather than `gh pr diff`'s raw patch series, which replays
  already-merged #77 commits and triples the apparent size.
- Ran the new tests locally: `test_mge_conversions_traceable.py` (17),
  `test_mge_parameterized_traceable.py` (44) — all pass, x32 and x64.
- Ran the full affected surface: `test_mge.py`, `test_potential.py`,
  `test_configuration.py`, `test_model_iterator.py`, `test_all_models.py`,
  `test_model_search.py` — 471 tests, all pass.
- `ruff check` clean on all changed files.
- `sphinx-build -W` (warnings-as-errors) succeeds.
- Grepped every existing `ParameterConstraint(...)` call site to confirm none
  relied on the old `self.unit or value.unit` falsy-`""` fallback before the
  explicit `is None` check replaced it — the unit semantics change is safe.
- Confirmed `.to(declared_units[...])` reuses `Quantity`'s existing method
  (already used for NFW's `M_200` reporting before this PR), not a new API.

## Finding

**P3 — "percent" shape-unit support is not reachable through the config
schema.** The PR's docs/`KNOWLEDGE.md` additions describe `60 percent` and
`0.6` as interchangeable for a shape parameter's domain and reporting, and
the new unit-preservation machinery is real and tested at the
`Potential.build_with_validity`/`_raw_parameters_valid` level. But
`tnt.configuration.validation._validate_parameter_units` raises
`ValueError` on any `unit` key at all for a parameter whose dimension is
`"dimensionless"` (which every shape parameter — `q`, `p`, `u`, `T`,
`T_maj`, `T_min`, `q_min` — is), before `declared_quantity` or the registry
ever sees it. So nothing reachable from a YAML config can exercise this path
today; it's exercised only by tests and direct in-process calls. Not a
defect — the lower-level machinery is correct and the generality is cheap —
but the PR description and docs read as though this is a usable
end-to-end feature rather than a currently-dead config path. Worth a
one-line caveat if `_validate_parameter_units` isn't being relaxed in the
same change.

## Verdict

No correctness, structure, or documentation issues beyond the P3 above.
Ready to merge.

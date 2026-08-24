# Review: #24 `codex/issue-14-unit-handling` ("Preserve configuration units until runtime")

Reviewed against `main` at `6f93b23`.

## Verdict

#24's core goal -- preserving declared `{value, unit}` in the resolved
config instead of stripping units at read time, and moving dimensional
validation to each runtime object's own construction -- is sound and
correctly implemented. **Merge it.**

Finding 1 is a real bug, but it's resolved by design once
[issue #27](https://github.com/dynamics-of-stellar-systems/tnt/issues/27)
lands (redefining "run" so only a fully-constructed, already-validated
configuration ever gets archived) -- not something to patch locally in
#24, since #27 removes the pathway that makes it possible in the first
place. Finding 2 is tech debt worth tracking, not a correctness bug or a
merge blocker.

## Finding 1: bare exception breaks `ConfigurationCompatibilityError`'s contract

`_critical_configuration` (`tnt/configuration_compatibility.py`) gains a
new call into `tnt.units`:

```python
unit_systems = build_unit_systems(_mapping(config, "units"))
config = normalize_configuration_quantities(config, unit_systems)
```

Both raise bare `TypeError`/`ValueError` on malformed input -- before this
branch, `_critical_configuration` never called into `tnt.units` at all, it
only deepcopied fields. This breaks the module's documented invariant that
every error path raises `ConfigurationCompatibilityError`.

**Reproduced.** #24's new `validate_configuration_quantities` (the
write-time check) never validates `spatial_binnings.*` (by design -- see
Finding 2). So `spatial_binnings.kinset1_binning.min_x: 5.0` (a bare float
instead of `{value, unit}`) passes `validate_resolved_configuration`
cleanly and gets archived. Resuming a search against that run calls
`ensure_resume_compatible` -> `_critical_configuration` ->
`normalize_configuration_quantities`, which *does* process
`spatial_binnings` and raises `TypeError: spatial_binnings.kinset1_binning.min_x
must be a mapping containing value and unit...` instead of
`ConfigurationCompatibilityError`. A normal write-then-resume workflow
reaches this -- no corruption or version skew needed.

**Why #27 subsumes this, not a local patch:** under #27's redesign,
archiving only happens after `ModelIterator.from_configuration()` has
fully succeeded -- every MGE, kinematics data set, and spatial binning
already validated by real construction. A malformed field gets caught
there, as a clear error, before anything is ever archived. The failure
path above can't occur once only proven-constructible configs get
archived.

## Finding 2: three inconsistent validate/convert patterns, all introduced in this diff

#24 introduces a validate-at-prep / convert-at-construction split (per
issue #14's policy), but implements it three different, mutually
inconsistent ways within its own diff -- not inherited drift:

1. **`spatial_binnings`**: zero prep-time check; validated+converted only
   by `ProjectedBinning.from_settings` at construction. Matches the
   stated policy.
2. **`kinematic_data.histogram.{width,center}` /
   `observational_errors.systematic_uncertainties.{v,sigma}`**: checked
   *twice* -- a new prep-time check in `validate_configuration_quantities`,
   *and* an independent construction-time validate+convert added in the
   same diff (`_explicit_histogram`/`_gauss_hermite_systematics`, both
   previously took already-bare numbers, pre-#14).
3. **`potential.<name>.parameters.*`**: prep-time check plus a
   newly-extracted `normalize_potential_settings`, called once, early, in
   `ModelIterator.from_configuration()` -- before `Potential.from_settings`
   runs. Deliberate, per
   [issue #14's design-refinement comment](https://github.com/dynamics-of-stellar-systems/tnt/issues/14#issuecomment-5353302829):
   the parameter generator and potential construction need one shared
   representation. Still a third, distinct timing.

Since all three were written in this same PR, this isn't legacy drift --
it's how #24 implements its own stated policy, inconsistently. It's also
the direct cause of Finding 1: `spatial_binnings` got the
single-consumer treatment, but this same PR adds a second consumer
(`_critical_configuration`) that was never accounted for.

## Discussion: is early conversion to internal units even necessary?

`unxt`/`astropy` are unit-aware enough that a value could stay in
whatever unit its config declared, converting lazily wherever it's
finally consumed. Four candidate reasons this codebase might still need
eager, config-level conversion -- none hold up:

1. **Comparing declared quantities across two runs**
   (`_critical_configuration`'s actual job): doesn't need it.
   `Quantity(1.0, "kpc") == Quantity(1000.0, "pc")` is `True` --
   `unxt.Quantity` compares across compatible units natively. A direct
   `Quantity`-to-`Quantity` comparison would sidestep Finding 1 entirely,
   without the full `normalize_configuration_quantities` traversal or any
   exception-wrapping.
2. **`AllModels`' `QTable` columns wanting one fixed unit**: doesn't need
   it. `astropy.table.QTable` fixes a `Quantity` column's unit from the
   *first* inserted row and auto-converts every later row to match
   (verified: `QTable(rows=[{"r_s": 1.0*u.kpc}])`, then
   `.add_row({"r_s": 1000.0*u.pc})`, stores `1.0 kpc`). No config-level
   pre-conversion required.
3. **Float32 numerical stability** for JAX-based computation: also
   doesn't need it -- backwards, if anything. A `Quantity` becomes a raw
   JAX float wherever something calls `.ustrip()`/`.to()`, e.g. inside
   `galax.potential`'s own construction. That conversion happens there
   regardless of the config's original unit, since even an
   already-internal-unit float still has to become a JAX array at that
   point. Whatever stability comes from consistent magnitudes is
   delivered by *that* conversion -- the same "validate/convert at
   construction" pattern #24 is implementing -- not an earlier,
   config-level step.
4. **Shared representation for the parameter generator** (issue #14's
   design-refinement comment: potential construction and parameter
   generation need to interpret unit-bearing settings identically):
   doesn't need it either. Checked directly against
   `AbstractPotentialComponent.from_settings` (`build-potential`,
   unmerged) -- it never reads a parameter's declared `unit`, only
   `value`, wrapped using the *internal* unit system regardless of what
   the config declared. Potential construction was never a second,
   independent interpreter needing to stay in sync with the generator's;
   only the generator needs to interpret declared units, bounds, and
   steps. The right boundary is `Quantity`-typed, not a bare float: the
   generator produces unit-ful `Quantity`s, and potential construction
   requires and converts them generically (`.to()`) rather than trusting
   an untyped number by convention. (Current `build-potential` code
   doesn't do this yet -- a real, separate gap, tracked for when that
   branch is active again.)

No candidate reason survives -- `galax.potential` construction already
accepts a `Quantity` in any compatible unit and converts via its own
`units=` argument. There's no strong technical case found here for eager,
config-level conversion at all.

## Root cause: archiving happens before validation completes

Both findings trace back to one property: `Configuration.read()` archives
a run's `resolved_config.yaml` *before* anything downstream has validated
it by actually constructing the runtime objects it describes. Under #14's
own design (validation moves to each object's own construction),
`Configuration.read()` structurally cannot know a configuration is fully
valid -- only `ModelIterator.from_configuration()` succeeding proves that.

Not fixable as a local patch -- it needs redefining what "run" means and
when archiving happens. Proposed in #27: redefine "run" as exactly one
`ModelIterator.run()` call, and move run-identity allocation/archiving
there, so only a configuration already proven constructible ever gets
archived.

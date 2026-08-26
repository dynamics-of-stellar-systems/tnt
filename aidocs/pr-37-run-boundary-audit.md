# PR #37 audit: redefine run lifecycle and defer configuration archiving

Audit date: 2026-08-26

Pull request: #37, `codex/issue-27-run-boundary` into `main`

## Overall judgment

PR #37 is a clean, faithful implementation of issue #27's proposal, and one
genuine improvement over it. Run-identity allocation and archiving moved
correctly from `Configuration.read()` to `ModelIterator.run()`, gated on both
successful runtime construction (`from_configuration()`) and the existing
resume-compatibility preflight -- verified directly against the code, not
just the diff. I did not find a functional bug in the mechanism itself: the
ordering guarantees (archive only after preflight passes; preflight compares
against a genuinely earlier run, never the not-yet-allocated current one) all
hold, and I found no stale reference anywhere in the repository to the
removed `Configuration.run_id`/`resolved_path`/`run_manifest_path` fields or
the renamed `_preserve_run`/`_resolve_and_write`.

The one real gap is test coverage, not correctness: moving the archiving call
site dropped the only tests that checked `_build_run_manifest`'s actual
output shape (Git/TNT provenance, dependency versions, execution/host
context, random-seed status), and they were never restored at the new call
site. `_build_run_manifest` itself is untouched by this PR, so the risk is
low, but it is now effectively unverified by any test in the suite.

PR #37 also makes one deliberate, well-documented departure from issue #27's
own proposal: the issue explicitly said calling `ModelIterator.run()` more
than once on one iterator would no longer be supported ("extending a search
means a new process/resume attempt... not a second `run()` call"). The first
commit (`6f98be9`) implemented exactly that; the second commit (`96d1f3d`,
"Allow repeated model iterator runs") reversed it. This looks like the right
call -- it doesn't reopen the bug issue #27 was fixing (a run is still only
archived after construction+preflight succeed, every time) -- but it's worth
flagging explicitly since it contradicts prose still visible in the issue.

## Architectural summary

- `Configuration.read()`/`_resolve` (renamed from `_resolve_and_write`) no
  longer allocates a run ID or writes anything to disk. `Configuration` lost
  `run_id`/`resolved_path`/`run_manifest_path` entirely, gaining only
  `logfile_path`. Confirmed empirically: after `Configuration.read()`,
  `<output_directory>/config_repository/` does not exist
  (`test_read_resolves_defaults_without_allocating_a_run`,
  `test_configuration_session_logs_preparation`).
- `preserve_run` (was private `_preserve_run`, `tnt/configuration/core.py:495`)
  is unchanged in body, only exposed publicly and relocated in call site.
- `ModelIterator.from_configuration()` (`tnt/model_iterator.py:110-195`) now
  takes the `Configuration` object itself, guards with a `RuntimeError` if it
  hasn't been `read()` (checked via `data`/`unit_systems`/`workspace_root`),
  and retains `runtime_configuration`/`portable_configuration`/
  `workspace_root`/`logfile_path`/`configuration_repository` as new fields so
  a later `run()` call can archive exactly the configuration whose runtime
  objects were actually built. No archiving happens here.
- `ModelIterator.run()` (`tnt/model_iterator.py:197-354`) now: runs the
  existing `len(run_config_log) != models.n_iterations()` and
  `ensure_resume_compatible` preflight checks first (both unchanged in
  logic), *then* calls the new `_archive_run()` (`model_iterator.py:356-372`),
  which calls `preserve_run` and stamps the resulting `run_manifest`/`run_id`
  onto `self`. `run_id`/`run_manifest` are now `int | None`/
  `RunManifestReference | None`, defaulting to `None` until the first `run()`
  call.
- `RunConfigLog.baseline_run_reference` (`tnt/run_config_log.py:197-205`)
  dropped its `current_run_id` kwarg and the self-comparison it enabled --
  correctly, since it's now called before the current run's ID exists, so a
  returned baseline can never be the current run.
- Docs (`docs/source/configuration.md`'s "Run identity" section,
  `docs/source/model_search.md`, `aidocs/KNOWLEDGE.md`) were rewritten to
  match, and read as accurate against the actual code on this branch, not
  just aspirational.

## Findings

### Moderate: detailed run-manifest content is no longer tested anywhere

`test_configuration.py::test_read_resolves_defaults_and_writes_run_bundle`
(renamed to `test_read_resolves_defaults_without_allocating_a_run`) used to
assert `_build_run_manifest`'s full output shape: `manifest["tnt"]` (version,
`git_commit`, `git_working_tree_dirty`), `manifest["dependencies"]`,
`manifest["execution"]["workspace_root"]`,
`manifest["configuration"]["input_directory"]`/`"output_directory"`, and
`manifest["randomness"]` (`configured_orbit_library_seed`,
`effective_orbit_library_seed`, `status`). All of that was deleted along with
the archiving behavior it was testing, since `Configuration.read()` no longer
archives.

The replacement coverage at the new call site
(`test_model_iterator_runs_against_the_resolved_example_configuration`,
`tests/integration_tests/test_model_search.py:164-170`) only checks
`manifest["manifest_version"]`, `manifest["run_id"]`,
`manifest["configuration"]["logfile"]`, and
`manifest["configuration"]["resolved"]` -- four of the roughly fifteen fields
the old test verified. I grepped the full test tree for
`"git_commit"`/`"randomness"`/`manifest["dependencies"]`/
`manifest["execution"]`/`"configured_orbit_library_seed"`: no hits anywhere.

`_build_run_manifest` itself is untouched by this PR, so nothing is
*currently* broken -- but a future change to it (e.g. a provenance field
rename, a broken `_git_provenance()` call) would now go undetected until
someone reads an actual manifest file by hand.

Actionable location: `tests/integration_tests/test_model_search.py`, the
manifest assertions in
`test_model_iterator_runs_against_the_resolved_example_configuration` (or a
new dedicated test) -- restore the fuller shape check the deleted
`test_configuration.py` test had.

### Low: the new `from_configuration` guard clause is untested

```python
if (
    not configuration.data
    or configuration.unit_systems is None
    or configuration.workspace_root is None
):
    raise RuntimeError("Configuration must be read before construction.")
```

(`tnt/model_iterator.py:133-138`, new in this PR) has no test anywhere --
grepped the full tree for the exact message, one hit, the raise itself. Cheap
to add: `ModelIterator.from_configuration(Configuration())` should raise.

### Low: `_archive_run`'s self-check is unreachable

```python
run_manifest = RunManifestReference.from_run_manifest(manifest_path)
if run_manifest.run_id != run_id:
    raise RuntimeError(
        "Published run manifest does not match its allocated run ID."
    )
```

(`tnt/model_iterator.py:365-369`) can't actually fail: `preserve_run` writes
the same `run_id` value into both the run directory's name and the
manifest's own `run_id` field (`tnt/configuration/core.py:507,573`), and
`RunManifestReference.from_run_manifest` *already* raises if those two ever
disagree, independently, before it returns
(`tnt/run_config_log.py:59-64`). This isn't wrong, just redundant -- reads as
a real invariant check but can never trigger. Low value either to fix or
leave; noting it so it isn't mistaken for meaningful defense-in-depth later.

## Missing or weak tests

Beyond the manifest-content regression above (the main gap), coverage is
otherwise thorough and well-targeted -- the new tests are genuinely testing
the boundary the PR is about, not just plumbing:

- `test_invalid_runtime_owned_configuration_is_not_archived`: a config that
  fails `from_configuration()` never creates `config_repository/` at all.
- `test_incompatible_resume_is_rejected_before_allocating_a_run`: directly
  inspects `runs/` on disk to prove a rejected resume leaves only the
  original run directory -- not just that the correct exception is raised.
- `test_each_call_on_the_same_iterator_gets_a_new_run_id` /
  `test_each_zero_iteration_call_is_archived_and_reported_as_a_run`: confirm
  the repeated-`run()` reversal end to end, including a zero-iteration call
  still incrementing `total_runs`/`run_ids_without_iterations`.

I'd only add the `from_configuration` guard-clause test and the manifest
fuller-shape check above; nothing else stood out as thin.

## Checks run

All run directly on this branch, locally (no Colima/Linux Docker parity
check performed):

- Full `pytest`: 282 passed.
- `ruff check .`: passed.
- Sphinx build with warnings treated as errors: passed.
- Manual trace of every `Configuration.run_id`/`resolved_path`/
  `run_manifest_path` and `_preserve_run`/`_resolve_and_write` reference
  across the whole repository (not just the diff): no stale usages found.
- Manual trace of `_archive_run`'s ordering against
  `ensure_resume_compatible`/the models-vs-log-length check: archiving
  provably happens after both, matching what the tests claim.

## Recommended review and correction sequence

1. Restore manifest-content-shape coverage (git/tnt/dependency provenance,
   execution context, randomness/seed status) at the new archiving call
   site -- either extend
   `test_model_iterator_runs_against_the_resolved_example_configuration` or
   add a focused new test.
2. Add one test for the `from_configuration` "not yet read" `RuntimeError`
   guard.
3. Optional: drop or repurpose `_archive_run`'s unreachable self-check, or
   leave it with a one-line comment noting it's defense against a
   `preserve_run`/`RunManifestReference` contract that already enforces the
   same thing independently -- either is fine, just don't read it as an
   independent guarantee when reasoning about this code later.
4. No design questions to resolve -- issue #27's proposal is fully
   implemented, and the one deviation from it (allowing repeated `run()`
   calls) is already made and documented consistently across the code and
   both `docs/source/*.md` files.

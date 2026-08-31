# PR 54 audit: safe multiprocessing context for worker logging

Pull request: #54, `Use a safe multiprocessing context for worker logging`

Branch: `codex/safe-multiprocessing-context`

Issue: #53, `Use a safe multiprocessing context for worker logging`

Base/head reviewed: `main` at `a1b3a27` / PR head at `f5578a7`

Note: the reviewer added `f5578a7` (a one-line `aidocs/KNOWLEDGE.md` pointer)
directly to the branch, per the shared edit-docs-on-each-others-PRs workflow.
Everything else is `31f54bd` by the PR author.

## Overall judgment

PR 54 is correct, minimal, and consistent with the issue's proposed solution.
No critical, high, or medium-severity defects. It should be **squash-merged**
after the two low-severity notes below are considered; neither blocks merge.

## Architectural summary

`configure_logging()` previously created its cross-process log queue with the
platform-default multiprocessing context (`fork` on Linux) and the
worker-logging regression started its process with the same default. Starting
a `fork` child after JAX and the logging listener have created threads is the
documented deadlock hazard behind the two suite warnings in issue #53.

The change threads a single explicit `spawn` context through the session:

- `configure_logging()` builds the queue from
  `multiprocessing.get_context("spawn")` and stores that context on the new
  required `LoggingSession.__init__(worker_context=...)` keyword, exposed as
  the public `LoggingSession.worker_context`.
- The worker-logging test creates its process from `session.worker_context`,
  asserts the start method is `spawn`, and gains `try/finally` teardown
  (`terminate` + `close`) with the join timeout raised from 10 s to 30 s.
- `docs/source/logging.md` and `aidocs/KNOWLEDGE.md` record the policy: use
  `worker_context` for every worker and `ProcessPoolExecutor(mp_context=...)`
  for future pools; spawn targets must be importable and entry points guarded
  by `if __name__ == "__main__"`; TNT deliberately never calls
  `multiprocessing.set_start_method()` because that process-global policy
  belongs to the embedding application.

`spawn` is the right choice over `fork` (deadlock-safe, already the macOS and
Windows default) and over `set_start_method()` (an explicit context object is
the library-correct mechanism). Queue and process now share one context, so
their synchronization primitives are compatible. `LoggingSession(...)` is
constructed only inside `configure_logging`, so the new required keyword
breaks no other caller.

## Findings

### Critical

None.

### High

None.

### Medium

None.

### Low 1: the 30 s worker-join timeout is timing-fragile

Reference: `tests/unit_tests/test_logging.py:122`

A `spawn` worker re-imports the full stack before running its target;
`import tnt.logging` pulls in `jax`, `galax`, and `tnt.potential` (confirmed
locally). The 30 s join is comfortable on an unloaded machine but is the same
margin the PR 52 audit already flagged as fragile for the existing
`jax_enable_x64` subprocess test, which timed out once at 30 s under load.
Two independent 30 s-margin subprocess tests raise the chance of an
occasional CI timeout that looks like a real failure.

Options, in order of preference: raise this join to 60 s; or mark both
subprocess tests `slow` / give them a shared longer-timeout helper. Not a
blocker -- the `try/finally` teardown already prevents a hung process from
leaking.

### Low 2: `BaseContext` is imported from a non-public submodule path

Reference: `tnt/logging.py:13`

`from multiprocessing.context import BaseContext` reaches into
`multiprocessing.context`, which is not part of `multiprocessing`'s
documented public surface, purely for the `worker_context` type hint.
`BaseContext` itself is the documented base class of context objects and this
is the idiom most code uses, so this is a note rather than a defect. If a
fully-public annotation is preferred, `multiprocessing.context.BaseContext`
via `import multiprocessing.context` reads no better; `Any` would lose useful
information. Leaving it as written is reasonable.

## Observation (not a finding): compilation cost for future worker pools

`spawn` also gives each worker a cold JAX JIT cache, so a future production
worker pool would have every worker re-trace and re-compile every jitted
function. The mitigation is a shared `jax_compilation_cache_dir` (via
`jax.config`), which lets workers load compiled XLA executables from disk.
This is out of scope for PR 54 -- it belongs with the production-execution
work -- and is now flagged in `aidocs/KNOWLEDGE.md` next to the multiprocessing
policy (`f5578a7`). `forkserver` was considered as an alternative to `spawn`
for a future pool; it would save the import cost but still start each fork
with a cold JIT cache unless the server pre-compiled, which risks
re-introducing the fork-after-threads hazard. `spawn` plus a persistent
compilation cache is the cleaner combination.

## Test coverage assessment

`test_worker_records_are_written_by_parent_listener` now asserts the start
method is `spawn` and exercises the real parent-listener / worker-queue path
end to end, including robust teardown. That is adequate for what the PR
changes. The `ProcessPoolExecutor(mp_context=...)` path documented in the
policy is not tested, correctly -- production worker scheduling is still
scaffolding and there is nothing to exercise yet.

## Documentation assessment

The multiprocessing-context policy is stated consistently in
`docs/source/logging.md` (with a corrected, `spawn`-compatible worker example
using a `main()` function and `__name__` guard) and `aidocs/KNOWLEDGE.md`. The
"TNT does not call `set_start_method()`" rationale is worth keeping -- it is a
tempting wrong fix for a future spawn/pool problem. The compilation-cache
pointer is developer-facing and is in `KNOWLEDGE.md` only, deliberately not in
`logging.md`, since it concerns the JAX runtime rather than logging.

## Checks run

From the PR head on macOS (Apple Silicon):

- full suite: **332 passed**, one warning (the pre-existing, dependency-owned
  TensorFlow Probability / JAX `pytype_aval_mappings` deprecation);
- focused `tests/unit_tests/test_logging.py`: 5 passed, same single warning --
  the two fork/JAX warnings from issue #53 are gone;
- `ruff check .`: passed;
- strict `sphinx-build -E -b html -W`: passed.

macOS's default multiprocessing context is already `spawn`, so locally this
change is behaviourally a no-op and the run only confirms no breakage. The PR
author verified the warning removal on the Linux container (332 passed, three
warnings down to one).

## Recommended merge path

1. Decide whether to raise the worker-join timeout to 60 s or mark the
   subprocess tests `slow` (Low 1). Optional.
2. Re-run the Linux suite, Ruff, and strict Sphinx if (1) is taken.
3. Obtain the required GitHub review approval and squash-merge PR 54.
4. Remove this audit document in its own commit before merge.

## Decision required

None material. The only open question -- `spawn` vs `forkserver` for a future
worker pool, and when to add `jax_compilation_cache_dir` -- is deferred to the
production-execution work and does not affect this PR.

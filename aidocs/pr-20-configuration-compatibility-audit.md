# Pull Request 20 Configuration-Compatibility Audit

Status: temporary working document for reviewing and fixing pull request #20
(`configuration-compatibility` onto `main`). Update this document as findings
are resolved and delete it before the changes are merged into `main`.

Audit date: 2026-08-19

Audited branch: `configuration-compatibility` (diff against `main`)

An 8-angle automated review (`/code-review main high`) covered
`tnt/configuration.py`, `tnt/configuration_compatibility.py`,
`tnt/run_config_log.py`, `tnt/model_search_state.py`, and
`tnt/model_iterator.py`. Findings below are ranked most-severe first. Two
correctness bugs were empirically reproduced with standalone scripts, not just
inferred from reading.

## What held up

Most of the diff checked out. Cross-file wiring between `Configuration`,
`RunManifestReference`, and `ModelIterator` was traced symbol-by-symbol with
no stale references found; the zero-iteration-run handling, the
`os.replace`→`os.link` atomic-write change, and `ModelSearchState` being the
single path that pairs `AllModels`/`RunConfigLog` writes all matched their
documented contracts, with the full unit suite (109+ tests) passing. The
findings below are what's left after that pass, not a sign the design is
broadly unsound.

## Before merge

Both bugs below stem from the same gap — nothing in this PR assumes,
enforces, or documents a single writer — so they likely share one cheap fix
(see open question 1) rather than needing independent, more involved
concurrency handling.

### 1. Concurrent writers can create duplicate configuration snapshots

`tnt/configuration.py:539` (`_preserve_resolved_configuration`)

The semantic-match dedup scan runs once, before the create loop; the
`FileExistsError` retry only advances to a new numeric index without
re-scanning for a match. Because the snapshot directory name embeds a content
hash, two processes preparing the same semantic configuration at nearly the
same time (e.g. an HPC job array) never collide on `mkdir()` and each create
their own snapshot directory for identical content — reproduced directly,
producing e.g. `configurations/0000-<hash>/` and `configurations/0001-<hash>/`
holding byte-identical `resolved_config.yaml`. This violates the documented
"one snapshot per distinct configuration" guarantee. Contrast with the sibling
`build_and_preserve_compatibility_signature`, which handles the identical race
correctly via write-then-reconcile-on-collision.

### 2. `ModelSearchState.write()` can silently lose a concurrent checkpoint

`tnt/model_search_state.py:85` (`write`)

`write()` never reads or compares against whatever is currently published
before replacing it — only internal self-consistency checks run. Two callers
that each read a 5-iteration checkpoint and independently extend it (A to 8
iterations, B to 6) can race: if A writes first and B writes second, B's write
raises nothing and permanently overwrites A's 3 extra iterations of model
results, with no error, warning, or way to detect the loss afterward —
reproduced directly (final published state showed 6 iterations, not 8). This
contradicts the module's own "coordinated persistence and recovery" framing:
the coordination only covers the crash window between `write()`'s own two
`os.replace` calls, not two valid writers racing on the same checkpoint pair.

## Design discussion

Not merge blockers — a longer conversation about whether this subsystem's
complexity (~2450 lines across five files) matches what it needs to do.

### 3. NaN in a critical-configuration field breaks resuming an unedited run

`tnt/configuration_compatibility.py:397` (`_different_paths`)

The leaf comparison is plain `==`. A critical-configuration field that
legitimately resolves to NaN (an unset-sentinel, or a computed default) makes
a signature compare unequal to itself, because `ensure_resume_compatible`'s
`semantic_sha256`-equal fast path diffs the identical dict against itself and
the NaN leaf reports as different — raising `ConfigurationCompatibilityError`
on a resume that changed nothing.

### 4. Resume compatibility is re-derived from disk on every `run()` call

`tnt/configuration_compatibility.py:172` (`ensure_resume_compatible`), called
from `tnt/model_iterator.py:234`

Every distinct historical snapshot's compatibility signature is re-read and
re-hashed on every call, with no per-process caching, even though the result
is immutable once established. The docs describe an execution layer calling
`run()` once per checkpointed iteration; under that pattern, a search with N
distinct historical snapshots run for many iterations pays O(iterations × N)
redundant file reads and SHA-256 recomputations for a fact that never changes
within the process.

### 5. Run-log metadata refresh and checkpoint validation are O(M) and doubled

`tnt/run_config_log.py:327` (`_refresh_run_metadata`)

Globs and re-validates every manifest (re-hashing each one's resolved config)
on every `RunConfigLog.read()`/`write()`. `ModelSearchState.write()` then
repeats that same O(K) validation a second time via its own `_validate_table`
call before re-validating the just-written temp file. A search checkpointed
after every iteration, resumed many times, accumulates M historical runs;
every checkpoint write re-reads and re-hashes all M manifests' resolved
configs twice in a row, so checkpoint latency grows linearly with total run
count and doubles unnecessarily on top of that.

### 6. Configuration resolution re-hashes every archived snapshot on every read

`tnt/configuration.py:546` (`_preserve_resolved_configuration` dedup scan)

A project that has accumulated N archived configuration snapshots pays O(N)
YAML-parse-and-hash work on every single TNT invocation just to resolve its
configuration, growing linearly as project history accumulates.

### 7. Scientific input files are hashed even when a signature already exists

`tnt/configuration_compatibility.py:140`
(`build_and_preserve_compatibility_signature`)

`_scientific_input_hashes` (chunked SHA-256 over every MGE/kinematic/
population file) is computed unconditionally before checking whether
`compatibility_signature_path` already exists. Every resumed run re-reads and
re-hashes every scientific input file from disk before discovering the
signature already exists and discarding the freshly computed hashes — for
multi-hundred-MB datacubes, a full-file read+hash pass added to every run's
startup for no new information.

### 8. Duplicated logic across the new modules (maintainability)

Several patterns are reimplemented 2-3 times with subtly different behavior
each time, rather than shared:

- `_canonical_sha256` (`configuration_compatibility.py:421`) re-implements
  `_semantic_configuration_sha256` (`configuration.py:528`) exactly; a future
  change to canonicalization applied to one silently won't apply to the other.
- `_validate_run_id`/`_validate_snapshot_id` (`run_config_log.py:345`) are
  structurally identical nonnegative-integer checks.
- `_required_string` (`run_config_log.py:359`) duplicates
  `tnt.config_parsing._required_string` but raises `TypeError` instead of
  `ValueError` — risks a silent behavior change if a future import
  accidentally shadows the local one.
- `_mapping`/`_require_mapping` (`configuration_compatibility.py:432`)
  reimplement `tnt.config_parsing`'s mapping-shape check instead of importing
  it, risking drift from the rules used elsewhere in config preparation.
- `_snapshot_id` (`run_config_log.py:402`) duplicates `_leading_index`
  (`configuration.py`) — both parse a numeric directory-name prefix.
- `_temporary_path` (`model_search_state.py:164`) is a third independent
  "reserve a temp file for atomic replace" implementation, alongside
  `_write_bytes_immutably` (`configuration.py`) and `_write_table_atomically`
  (`run_config_log.py`).
- `_validate_pair` (`model_search_state.py:37`) runs in both `__post_init__`
  and again as the first line of `write()` on a frozen dataclass — dead code,
  since the fields can't have changed in between.
- `RunConfigLog.write()` and `ModelSearchState.write()`
  (`run_config_log.py:225`) are two independently-implemented "atomically
  write the run-config-log ECSV" paths that have already drifted: only the
  latter reads back and validates the temp file before publishing.

### 9. Resolved-config integrity check runs uncached from four call sites

`tnt/run_config_log.py:104` (`RunManifestReference.from_run_manifest`)

The `resolved_config_sha256` check (read the resolved config back off disk,
re-hash it, compare against the value recorded in the manifest at write time)
lives inside this one primitive, but nothing caches its result. It's invoked
separately from each of four call sites: `ModelIterator.from_configuration`
(`model_iterator.py:124`, once per construction, before any compatibility
checking happens), `RunConfigLog.snapshot_references`
(`run_config_log.py:285`, once per historical `run_id` per resume),
`_refresh_run_metadata` (`run_config_log.py:324,336`, on every
`RunConfigLog.read()`/`write()` — see finding 5), and `RunConfigLog.path_for`
(`run_config_log.py:222`, even when only a file path is needed). Each call
site re-reads and re-hashes the same immutable file rather than reusing an
already-verified `RunManifestReference`.

Separately from the efficiency angle: it's worth questioning what this check
is actually defending against. If it's meant to catch on-disk corruption
(e.g. bit rot) months after a run, the design is inconsistent — it protects
only the resolved config, which is cheap to regenerate from the user's
original YAML, while `all_models.ecsv` (the actual model-search output,
often representing days of compute, checked directly with `grep` to confirm
neither `model_search_state.py` nor `all_models.py` hash or verify it at
all) has no integrity protection whatsoever. That asymmetry suggests the
check was more likely intended as a wiring-correctness check ("does this
manifest point at the file it claims to") than a deliberate corruption
defense — worth confirming with Thomas rather than assuming either way.

## Open questions for Thomas

1. **Findings 1 & 2 — assume a single writer.** Both races only matter if
   `ModelSearchState` writes and snapshot creation ever run concurrently for
   the same repository. If TNT only ever needs one writer per run (likely?),
   the fix is just documenting that as a caller contract — no locking or
   reconciliation logic needed. Open question: is documentation enough, or
   does something actually enforce it (e.g. a lock file, PID/hostname check
   on the manifest)?

2. **Finding 7 — do we need `_scientific_input_hashes` at all?** Beyond its
   efficiency cost (re-hashing every scientific input file on every resumed
   run), it's worth asking whether the check earns its place in the first
   place — can we trust users not to change their input data mid-modelling,
   rather than defending against it?

3. **Findings 1, 6, 8 — is snapshot deduplication worth its complexity?** The
   content-addressed dedup scheme is the source of the TOCTOU race, the O(N)
   re-hash-on-every-read cost, and several of the duplicated-logic findings.
   Alternative: just copy the resolved config into each run's own manifest
   directory with no dedup and no shared `configurations/` scan — trading
   some redundant storage for removing an entire subsystem (index
   allocation, dedup scan, race handling).

4. **Finding 4 — should `ensure_resume_compatible` check only the
   immediately preceding run, not full history?** Rather than walking every
   historical snapshot recorded in the log, compare only the most recent
   one. Would remove the O(N) re-derivation cost and most of
   `snapshot_references`'s need to resolve every historical `run_id`.

5. **Finding 9 — remove all hashes?** Three separate SHA-256 jobs exist
   (`semantic_sha256` for dedup, `signature_sha256` for the compatibility
   contract, `resolved_config_sha256` for integrity) and this has never been
   something we've deliberately worried about before this PR. If points 2-4
   above land — no scientific-input hashing, no snapshot dedup, only the
   immediately preceding run checked directly instead of via a hashed
   signature — the case for hashing anything at all is worth revisiting from
   scratch, rather than keeping a lighter version of the current scheme by
   default.

## Codex response to the audit

Assessment against the current implementation (2026-08-20):

| Finding | Assessment | Recommended treatment |
| --- | --- | --- |
| 1. Concurrent snapshot creation | Agree. This is a real correctness issue because two writers can choose the same next snapshot ID. | Resolve before merging. Reconcile after a concurrent create conflict, and add a concurrency regression test. |
| 2. Concurrent checkpoint overwrite | Agree. This is a real data-loss risk if two `ModelSearchState` instances write the same repository. | Resolve before merging. Enforce a single active writer or reject stale state; documentation alone is insufficient. |
| 3. NaN in critical fields | Disagree that this is currently an exposed bug. Configuration validation rejects non-finite numeric values, and spatial-binning construction rejects NaN values before signature generation. Runtime model or MGE values are outside the critical configuration signature. | Record the validation coverage; no code change is currently required. |
| 4. Repeated compatibility reads | The cost is overstated. `ensure_resume_compatible()` runs once per `ModelIterator.run()`, not once per internal search iteration. | Keep full-history checking. Consider caching only if measurement later shows a material cost. |
| 5. Repeated manifest validation | Agree that this is avoidable repeated work, but it is not a correctness problem. | Treat as a later cleanup or caching optimization. |
| 6. Linear snapshot scan | Agree. Snapshot lookup and deduplication scale linearly with the number of snapshots. | Keep the current simple implementation for now; optimize only when repository size or profiling justifies an index. |
| 7. Scientific-input hashing | The behavior is intentional. Hashing detects scientific input files whose bytes change while their paths remain unchanged. | Retain it unless the compatibility contract is deliberately weakened. |
| 8. Duplicated logic | Partly agree. The exact duplicate canonical-hash implementation and small parsing/validation helpers are candidates for consolidation. The write-time `_validate_pair()` is not dead because frozen dataclasses can still contain mutable tables, and the temporary writers have different immutable, mutable, and paired-file semantics. | Make only focused, behavior-preserving consolidations; do not merge helpers solely because they look similar. |
| 9. Repeated resolved-config integrity checks | Agree that caching or consolidation may help; disagree with removing the integrity hash. The resolved snapshot is authoritative and cannot necessarily be recreated after the submitted YAML or defaults change. | Preserve the distinct semantic, compatibility-signature, and integrity hashes. Optimize repeated verification separately if needed. |

### Recommended path forward

1. Fix findings 1 and 2 before merging PR #20, with regression tests for the
   chosen concurrency guarantees.
2. Preserve snapshot deduplication, full-history compatibility checks,
   scientific-input hashing, and the distinct hashes because they protect
   different invariants.
3. Treat findings 5, 6, and the performance portion of finding 9 as later,
   measurement-driven optimization work.
4. Apply only the clearly useful consolidation from finding 8 after the
   correctness fixes, so cleanup does not obscure their review.
5. Close findings 3, 4, and 7 by documenting the existing validation,
   call frequency, and intended compatibility behavior.

For the audit's open questions, TNT should not merely assume a single writer:
the repository layer should enforce or detect that condition. The current
content-addressed snapshot repository and full-history comparison are worth
retaining because they provide deduplication and prevent a later configuration
from silently becoming incompatible with an earlier run. Likewise, removing
all hashes would conflate three separate purposes: semantic identity,
compatibility identity, and on-disk integrity.

## Final disposition after the lean-repository decision

The project subsequently chose a different trade-off from the preceding Codex
response. TNT now assumes exactly one coordinating process writes a given
output directory, and users are responsible for not modifying scientific input
files in place while a model set may be resumed. Parallel model calculation is
still supported conceptually: workers return results to the single coordinator,
which alone updates shared state.

The implementation archives `run_manifest.yaml` and `resolved_config.yaml`
together under `config_repository/runs/<run_id>/`. It has no configuration
snapshots, snapshot deduplication, compatibility-signature files, or
configuration/scientific-input SHA-256 hashes. Resume compatibility is derived
directly from the current configuration and the archived resolved configuration
of the earliest run that contributed an iteration. Runs without iterations are
recorded for provenance but do not establish the model set's compatibility
baseline.

| Finding | Disposition after simplification |
| --- | --- |
| 1. Concurrent snapshot creation | Obsolete. The shared snapshot store and deduplication algorithm no longer exist. Concurrent coordinators are explicitly unsupported. |
| 2. Concurrent checkpoint overwrite | Accepted as outside the supported execution contract. One coordinator owns `AllModels` and `RunConfigLog`; parallel workers do not write them. Crash-safe paired checkpoint publication remains. |
| 3. NaN comparison | Not an exposed configuration bug: configuration validation rejects non-finite numeric values. The former identical-signature self-comparison also no longer exists. |
| 4. Repeated compatibility reads | Substantially obsolete. There are no historical signatures or hashes to traverse. The check currently runs once at the start of each `ModelIterator.run()` invocation, outside its internal iteration loop, and reads only the earliest model-contributing run's resolved YAML. The future coordinating execution layer should move this check before runtime-object construction and perform it exactly once per TNT run after loading `AllModels` and `RunConfigLog`. |
| 5. Repeated manifest validation | Partly addressed. Hashing and the duplicate manifest-validation pass were removed. The single O(M) manifest scan remains because it derives `total_runs` and `run_ids_without_iterations`. This is an accepted simple implementation unless profiling shows material checkpoint cost; the first optimization should be a lightweight numeric run-directory scan, not a persistent index. |
| 6. Linear snapshot scan | Obsolete. Configuration preparation creates the next numbered run directory without scanning or hashing archived configurations for deduplication. |
| 7. Scientific-input hashing | Obsolete by policy. Scientific files are not hashed; unchanged contents at a stable configured path are a user responsibility. |
| 8. Duplicated logic | Addressed as far as useful without obscuring behavior. Both hash implementations, snapshot-ID parsing/validation, signature persistence, and related paths disappeared. The remaining local manifest `_required_string` duplicate was replaced with `tnt.config_parsing._required_string`. Compatibility-specific mapping errors, immutable versus replaceable temporary writers, standalone versus paired checkpoint writers, and write-time pair validation remain intentionally distinct. |
| 9. Repeated resolved-config integrity checks | Obsolete. Run manifests no longer contain or verify a resolved-configuration content hash. They validate the fixed per-run path and require the YAML file to exist and be readable. |

The earlier recommendation to retain content-addressed snapshots, full-history
signature comparison, and three distinct hashes is therefore superseded by
this section. The scientific compatibility rules themselves remain: TNT still
rejects changes to compatibility-critical resolved settings, validates the
selected chi-square against successful historical models, and requires the
expected potential-parameter columns in `AllModels`. Configured scientific file
references are now critical because file contents are no longer hashed.

Follow-up test cleanup removed the test that modified TNT's own archived
`resolved_config.yaml` merely to prove that no integrity hash was checked. Such
modification is not a supported contract: per-run archives remain immutable
even though TNT does not hash them. The formerly redundant changed-config test
now verifies instead that each run preserves its own resolved configuration,
including the historical value after the submitted user file is edited for a
later run. A focused regression test also confirms that the shared
`_required_string` helper rejects whitespace-only resolved-config paths in run
manifests.

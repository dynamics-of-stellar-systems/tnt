# PR #80: orbit start-space audit

Date: 2026-10-07. Reviewed by Codex for Thomas.

PR: https://github.com/dynamics-of-stellar-systems/tnt/pull/80

Branch: `box_tube_start_spaces`.
Head: `12d2ef6fc85ffb0af7e43cb69e8369b4064c542d`.
Base: `10f53a8b1b19d0792833eb2631c4f966aff1076d` (`main`).

## Recommendation

Hold the merge for the findings below. P1 means high priority; P2 means a
normal-priority correctness issue that should be addressed in this PR.
The existing happy-path tests pass, but they do not exercise these failures.
No production code was changed. No GitHub review/comment, commit, push, or
merge was performed.

## Findings

### 1. [P1] Stop the integration driver on failure or lack of progress

Location: `tnt/orbit_library/xz_grid_from_boundary.py:220-254`, and its
unbounded `jax.lax.while_loop` at line 315.

The continuation condition checks only time and crossing count. The body
discards both `_solver_result` and `_ctrl_result`, and there is no attempt
limit or check that the next time is finite and greater than the current
time. A rejected step can leave time and crossings unchanged indefinitely.
The 5,000-period budget does not bound this failure: simulated time never
advances to that budget.

Evidence: a controlled terminal-controller-failure probe returned
`diffrax.RESULTS.dt_min_reached` on rejected steps. After three attempts,
time remained `0.0`, crossings remained zero, and the original loop condition
was still true. The probe bounded the loop externally to avoid hanging.
This verifies the response to an injected failure; it is not a claim that
the normal triaxial fixtures hang.

A second probe retained the actual Dopri8 solver and PID controller and
injected nonfinite derivatives. After three attempts, the state was
`time=0.0, next_time=nan, crossings=0, would_continue=True`. This reproduces
the stalled control flow without replacing the integrator or controller.

Requested change: carry failure status through the loop, stop on terminal
errors/nonfinite state/nonadvancing time, and enforce a finite step-attempt
budget. Surface failure to the caller rather than treating a failed trial
as a physical boundary. Add focused failure-path tests.

### 2. [P2] Correct or explicitly justify the outer-search taper formula

Location: `tnt/orbit_library/xz_grid_from_boundary.py:681-703`.

The current taper is `(1 - rel_rbi) / (3 * nI2)`. The cited
[DYNAMITE source](https://github.com/dynamics-of-stellar-systems/dynamite/blob/master/legacy_fortran/orbitstart_f.f90#L445)
uses `(1 - rel_rbi) / Ni3 * 3`, where `Ni3` corresponds to this sampler's
radial count `nI2`. Those differ by a factor of nine.

The narrower bracket can exclude the minimum before the minimizer runs.
With `rel_rbi=.2`, radial count 3, unit outer radius, and inner floor `.2`,
the present lower bound is approximately `.91111`. A controlled width
objective `(r - .5)**2`, run through the actual golden-section minimizer
and outer-boundary helper, returned `.9111102182491464` for the first
searched angle rather than its known minimum `.5`.

Requested change: reproduce the intended formula, or document this as an
intentional algorithm change and validate that its tighter bracket retains
the thin-orbit minimum. The synthetic probe establishes bracket exclusion;
it does not quantify the error in a particular galaxy model.

### 3. [P2] Preserve the outer-boundary angle before searching downward

Location: `tnt/orbit_library/xz_grid_from_boundary.py:672-677` and `:695-711`.

After the theta sweep finds the first long-axis tube, the current code
uses that index itself as the seed and searches only smaller indices.
[DYNAMITE advances to the adjacent angle first](https://github.com/dynamics-of-stellar-systems/dynamite/blob/master/legacy_fortran/orbitstart_f.f90#L413),
subject to its upper cap, then searches downward. The current translation
omits one angular row from the search in this path.

Evidence: four ascending angles `[.2, .4, .6, .8]`, outer radii all 1,
inner radii all `.2`, radial count 3. A controlled classifier returns box
above `.5` and long-axis tube at or below `.5`, so the theta sweep's first
long-axis tube is zero-based index 1. The actual helper, with minimization
replaced by the bracket midpoint, returns
`[.9555546, 1, 1, 1]`: index 1 is never searched. The reference indexing
seeds index 2 and searches indices 1 and 0.

Requested change: translate the one-based continuation index consistently
and test onset in the middle, at either end, and absence of long-axis tubes.
If a different rule is intended, justify and numerically validate it.

### 4. [P2] Reject nonpositive radial limits

Location: `tnt/configuration/validation.py:689-693`.

Validation checks dimensions and `rmin < rmax`, but accepts zero or negative
`rmin`. All three samplers take `log10(rmin)` and use it to set their
radial search floor. These are radii, so their valid domain must be positive.

Evidence: `_validate_orbit_sampler` accepts both `rmin=0 kpc` and
`rmin=-1 kpc` with `rmax=10 kpc`. With the negative value, a three-shell
stationary sampler in the test NFW potential computes energy-grid radii
`[nan, nan, 10]`, yet returns finite initial-condition radii
`[.001, .001, 10]` without an exception. This is silent scientific corruption,
not just a poor validation error message.

Requested change: require `0 < rmin < rmax` in configuration validation
and at the public sampler construction/execution boundary, which is
currently callable independently of configuration. Test zero, negative,
and equivalent units.

### 5. [P2] Reject or implement the boundary sampler's one-angle case

Location: `tnt/configuration/validation.py:694-697`,
`tnt/orbit_library/xz_grid_from_boundary.py:516`, and `:672-673`.

The shared schema permits `XZGridFromBoundary` with `nI1=1`. Its inner
search unconditionally accesses a second angle, and its outer search takes
`argmax` of the empty remaining-angle sequence. JAX's out-of-bounds array
indexing does not supply a useful validation error for the former.

Evidence: validation accepts `nI1=1`; calling the unmodified boundary
sampler with `nE=1`, `nI2=2`, `.1-10 kpc`, and the test NFW potential
raises `ValueError: attempt to get argmax of an empty sequence`.

Requested change: establish and enforce the boundary algorithm's supported
minimum angular count in validation and direct use, or implement a genuine
single-angle path. The two simpler samplers can support one angle, so do
not impose a shared restriction without justification.

### 6. [P2] Validate equipotential brackets and solved roots

Location: `tnt/orbit_library/common.py:73-81`, and the fixed radius factors
at lines 33-34.

The equipotential helper suppresses root-solver exceptions and returns
`solution.value` without checking status or residual. Its fixed bracket
is not guaranteed to contain the directional root just because the
potential increases monotonically along rays. A root outside the bracket
can return the endpoint as though it were an equipotential location.

Evidence: an accepted positive-axis-ratio `LMJ09LogarithmicPotential` with
`q1=q2=1`, `q3=1e-5`, `r_s=1 kpc`, and `v_c=200 km/s`; shell energy taken
at `(1,0,0) kpc`. Along the z axis the exact root is `1e-5 kpc`, below
the supplied `[.001,1000] kpc` bracket. The helper returns `.001 kpc`
without an error, with potential-energy residual
`.17817158209820072 kpc2/Myr2`. The same helper serves all three samplers.
This is an extreme flattening probe, not a typical observed galaxy.

Requested change: validate the endpoint sign ordering, solver outcome, and
finite residual against the intended tolerance. Either reject unsupported
brackets explicitly or implement a deliberately documented bracket search.
Do not silently accept a clamped endpoint or clip an invalid launch energy
into zero velocity.

## Validation performed

All commands used the documented Linux development container with Docker's
`colima` context. Python test groups ran in separate processes.

| Check | Result |
| --- | --- |
| `pytest -q tests/unit_tests/test_orbit_library.py` | 15 passed |
| `pytest -q tests/unit_tests/test_configuration.py tests/integration_tests/test_configuration_session.py` | 75 passed |
| `pytest -q tests/unit_tests/test_configuration_compatibility.py tests/unit_tests/test_default_config.py tests/unit_tests/test_model_iterator.py tests/integration_tests/test_build_named_inputs.py` | 64 passed |
| `ruff check .` | Passed |
| `sphinx-build -E -b html -W docs/source /tmp/tnt-pr80-docs` | Passed |
| `git diff main...HEAD --check` | Passed |
| Independent edge-case/control-flow probes | Findings 1-6 reproduced as described |

The tests emit one pre-existing TensorFlow Probability/JAX deprecation
warning per group. The complete 642-test suite was not rerun.

Temporary reproduction harnesses from this audit are
`/private/tmp/tnt_pr80_probe.py`,
`/private/tmp/tnt_pr80_failure_probe.py`, and
`/private/tmp/tnt_pr80_nonfinite_probe.py`. They can be run with
`docker --context colima compose run --rm -T dev python - < <script-path>`.
They use process-local monkeypatches for the explicitly controlled probes;
they do not modify TNT source.

## Coverage and scope notes

- I compared the published boundary-search source and the paper's start-space
  description, but did not compile or rerun DYNAMITE. The PR's claimed
  four-geometry numerical comparison is author-supplied evidence, not an
  independently reproduced result of this audit. Pin the source revision and
  preserve reference arrays/commands before relying on numerical equivalence.
- The crossing cap is 100 despite a comment claiming an exact match to the
  reference's 400. Together with the documented shorter time budget and
  different minimizer, this needs explicit convergence/equivalence evidence.
- The fallback test uses `jnp.allclose`, not a bitwise-equality assertion, and
  permits every shell to delegate. It does not establish accurate boundaries
  in a regular shell. Add independent regular-shell boundary assertions when
  addressing findings 2 and 3.
- `_outer_tube_boundary` continues expensive classification after its
  `broken` flag and computes minimizations whose results are subsequently
  masked out. Its apparent early exit freezes results rather than avoiding
  computation. This is a performance follow-up, separate from the six
  correctness findings above.
- Configuration-to-sampler construction, dithering, integration/library
  assembly, and counter-rotating copies remain documented deferred work.
  Their pre-existing scaffolds were not counted as regressions in this PR.

Assisted by Codex (OpenAI).

## Responses

Two fixes below go beyond what any single numbered finding asked for, each
surfaced while addressing one -- moved up front since they're referenced
from more than one response.

### Extra fix: Match DYNAMITE's `nI2`/`nI3` field naming (found while addressing finding 5)

Checking finding 5's reference source surfaced a naming discrepancy worth
fixing on its own: **TNT's `nI1`/`nI2` fields had swapped roles relative to
DYNAMITE's own `nI2`/`nI3`.** DYNAMITE's comment block
(`initial_parameters.f90:62`, `! nEner = # energies, nI2 = # I2, nI3 = #
I3`) and its box-orbit grid (`orbitstart_f.f90:287-291`, `Theta ... /nI2`,
`Phi ... /nI3`) fix `nI2` as the (first) angular grid count and `nI3` as
the second non-energy dimension, consistently across both the box- and
tube-orbit code -- the same positional roles TNT already gave its own
`nI1`/`nI2`, just one letter off. Finding 2's own fix already had to spell
out that DYNAMITE's `Ni3` (not `nI2`) was the radial count being matched;
this was the same cross-naming surfacing again. Renamed `nI1`->`nI2`,
`nI2`->`nI3` throughout `tnt/orbit_library/`,
`tnt/configuration/validation.py`, both default/test configs, the docs
page, `KNOWLEDGE.md`, and both test files, so TNT's field names now match
DYNAMITE's letter-for-letter, not just positionally. Triple-checked
against the actual Fortran source (not just the earlier citations) before
renaming anything.

### Extra fix: Scale the equipotential search bracket per energy shell (found while addressing finding 6)

Checked DYNAMITE's own equivalent, `findReq` (`orbitstart_f.f90:543-592`),
rather than just adding a validity check on top of the existing fixed
bracket. Two things there matter beyond "validate and raise":

- Its bracket is `[0.01, 1.1] * Req`, where `Req` (`Rcirc(j)` at the call
  site) is *that energy shell's own x-axis equipotential radius* -- not one
  fixed bracket shared across every shell regardless of its own natural
  scale, which is what TNT's `[rmin * 1e-3, rmax * 1e2]` was. Re-centering
  the bracket per shell is the actual fix for the failure mode, not just a
  check on top of a bracket that stays badly scoped; the audit's probe
  needed the equipotential to collapse to `1e-5 kpc` precisely because the
  old bracket's floor (`rmin * 1e-3`) had no relationship to that shell's
  own scale at all.
- Non-convergence there is a hard, loud failure (`stop "Can not find
  R_eq"`), not a silently accepted endpoint.

`_equipotential_radius` now takes `r_ref` (each shell's own x-axis radius,
already computed by every sampler to define that shell's energy -- none of
the three needed a new quantity, just to stop discarding one they already
had) and derives `[r_ref * 0.01, r_ref * 1.1]` internally, matching
`findReq`'s factors exactly. It returns `(r0, valid)`: `valid` requires the
bracket to contain an actual sign change, the solver's own result to be
`optx.RESULTS.successful`, and the residual at `r0` to be within `1e-7`
relative to the shell's energy -- DYNAMITE's own `findReq` convergence
tolerance. `_box_orbit_ic`/`_single_shell_ics`/`XZGridFromBoundaryOrbitSampler
.generate_ics`'s own per-shell search all thread this through; each
sampler's `generate_ics` aggregates validity across its whole grid and
raises a new `EquipotentialSearchError` (not a traced flag: orbit sampling
isn't part of the differentiable `build_with_validity` contract, so eager
is the right level here, matching `tnt.mge`'s own eager
`MGEDeprojectionError` convention) rather than returning as if nothing had
gone wrong. `_R_CEILING_FACTOR` is gone -- it was only ever the old global
bracket's outer factor; `_R_FLOOR_FACTOR` remains, unrelated, as the
`(x, z)`-plane samplers' own physical near-origin sampling floor.

### 1. [P1] Stop the integration driver on failure or lack of progress -- fixed

Confirmed by direct reproduction: a nonfinite-derivative injection that
previously left the loop spinning now returns `inf` in well under a second.

`_orbit_width_over_crossings`'s loop state now also carries a step-attempt
counter and a `failed` flag. Each step checks `diffrax.is_okay` on both the
solver's and the stepsize controller's result codes, and additionally treats
a non-finite or non-advancing proposed next time as failure (this is what
actually fires for the nonfinite-derivative case locally, since the default
`PIDController` has no `dtmin` set and so never reports its own terminal
`dt_min_reached`). `_TUBE_SEARCH_MAX_STEP_ATTEMPTS = 200_000` is a further,
coarser ceiling on total step attempts (accepted and rejected combined) as
defense in depth against a step that keeps being reported successful but
never converges. On failure the function returns `inf`, the same "unusable
trial point" signal already used for "no crossing found" -- so a failed
trial degrades the enclosing golden-section search/classification rather
than being reported as a physical boundary.

`test_orbit_library.py`: 15 passed, 38.44s (was 37.22s; +1.2s, within normal
run-to-run noise -- the added `is_okay`/finite checks are cheap scalar
comparisons already available from existing return values, not new
computation).

### 2. [P2] Correct or explicitly justify the outer-search taper formula -- fixed

Confirmed directly against the reference source
(`legacy_fortran/orbitstart_f.f90:482`, available locally): `r = ((1 -
rel_rbi)/Ni3*3)`. Fortran's `/`/`*` share precedence and associate
left-to-right, so this is `3*(1-rel_rbi)/Ni3`, not `(1-rel_rbi)/(3*Ni3)` --
the translation had the `3` on the wrong side of the division. Fixed to
`taper_fraction = 3.0 * (1.0 - rel_rbi) / nI2`.

Reproduced the audit's own bracket-exclusion probe (`rel_rbi=.2`, radial
count 3, unit outer radius, inner floor `.2`): the old formula gave a
lower bound of `.91111`, excluding the stated minimum at `.5`; the fixed
formula gives `.2` (the floor itself binds), restoring the full bracket.
Re-ran the golden-section minimizer from this module unmodified against
the audit's own `(r - .5)**2` width objective over the corrected bracket:
converges to `.50000005`, not `.91111`.

`test_orbit_library.py`: 15 passed, 35.03s (no meaningful change from
finding 1's 38.44s or the original 37.22s baseline -- same search cost,
just a correct bracket).

### 3. [P2] Preserve the outer-boundary angle before searching downward -- fixed

Confirmed by direct trace against the reference (`orbitstart_f.f90:443-452`,
also available locally): after the theta scan finds the onset row (`i_found`,
`onset_idx_from_scan` here), Fortran sets `i = min(i_found + 1, nI2 - 1)`
*before* seeding and searching downward from `i - 1`, so the real search
loop starts at `i_found` itself -- the found row gets a genuine `findtube`
search, and only the row above it (the bumped seed) is left at the
equipotential. TNT set `effective_onset_idx = onset_idx_from_scan` directly,
so the found row was itself treated as the unsearched seed and skipped.

Fixed by applying the same `+1`, capped at `n_theta - 1`, only on the
scanned-onset path (`found_x_directly` already starts from `n_theta - 1`
correctly, matching Fortran's `i = nI2` when the initial radial scan finds
an x-tube directly, without ever entering the secondary theta loop that
produces `i_found`). Also tightened the function's own docstring, which
worded the invariant ambiguously enough to read as consistent with either
the correct or the buggy behaviour.

Reproduced the audit's own four-angle probe (`[.2, .4, .6, .8]`, box above
`.5`, long-axis tube at or below `.5`): the scan finds onset at index 1;
the fix now searches indices 0 and 1, leaving only indices 2 and 3 (and the
top row) at the equipotential -- previously only index 0 was searched.

`test_orbit_library.py`: 15 passed, 35.53s (consistent with the 35-38s
range seen after findings 1 and 2; no meaningful change).

### 4. [P2] Reject nonpositive radial limits -- fixed

Confirmed: `_validate_orbit_sampler` only checked `rmin < rmax`, never
`rmin > 0`, so `rmin=0`/`rmin=-1 kpc` both passed validation and silently
produced non-finite energy-grid radii (`log10` of zero or a negative
number) downstream, as the audit's reproduction showed.

Fixed at both boundaries the audit called for:

- `_validate_orbit_sampler` now separately checks `rmin > 0` before the
  existing `rmin < rmax` check, with its own message.
- `AbstractOrbitSampler` (the shared base every concrete sampler inherits)
  now has a `__check_init__` enforcing `0 < rmin < rmax` in the samplers'
  own reference length unit -- this runs on construction regardless of
  whether a sampler is built through configuration at all, closing the
  "public sampler construction/execution boundary" gap the audit named
  (direct instantiation, e.g. in a test or a future `build_orbit_sampler`
  caller, bypasses `_validate_orbit_sampler` entirely).

Added tests for both boundaries: zero/negative `rmin` and `rmin == rmax`
rejected at construction for all three concrete sampler types, an
equivalent-unit case (`rmin` in `pc` numerically larger than `rmax` in
`kpc`) to confirm the comparison isn't just on raw declared numbers, and
the same zero/negative cases through `Configuration().read()`.

`test_orbit_library.py`: 23 passed (15 + 8 new), 35.01s.
`test_configuration.py`: unaffected by these changes alone; full combined
run (`test_orbit_library.py` + `test_configuration.py`, including the new
tests in both): 99 passed, 47.03s.

### 5. [P2] Reject or implement the boundary sampler's one-angle case -- fixed (reject)

Confirmed the crash: with `nI1=1` (the theta grid count, before the rename
below), `_outer_tube_boundary`'s `idx_desc = jnp.arange(n_theta - 2, -1, -1)`
is empty, so `jnp.argmax(is_x_theta)` on an empty array raises the reported
`ValueError`.

Chose "reject", not "implement a single-angle path", directly from
DYNAMITE's own source rather than a judgement call: `find_outerboundary`'s
and `find_innerboundary`'s own first executable line is
`if (nI2 <= 3) stop "nI2 is smaller then 4"` -- DYNAMITE refuses to run the
boundary search below 4 theta angles at all; there is no reference
single-angle behaviour to port. Checking this also surfaced the
`nI1`/`nI2` naming discrepancy fixed above.

With the rename in place, added the DYNAMITE-matching floor scoped to
`XZGridFromBoundary` only (`StationaryGrid`/`XZGridFromOrigin` keep
allowing `nI2=1`, per the audit's own note not to share a restriction
without justification): `_validate_orbit_sampler` rejects `nI2 <= 3` when
`type == "XZGridFromBoundary"`, and `XZGridFromBoundaryOrbitSampler` gets
its own `__check_init__` enforcing the same floor at direct construction
(equinox calls every ancestor's own `__check_init__` independently, so
this adds to, rather than replaces, the shared `rmin`/`rmax` check from
finding 4).

`test_orbit_library.py`: 29 passed (23 + 6 new), 40.01s.
`test_configuration.py` + `test_orbit_library.py` combined: 107 passed,
51.92s. `sphinx-build -W` still succeeds after the `orbit_library.md`
rename.

### 6. [P2] Validate equipotential brackets and solved roots -- fixed

Fixed by scaling the search bracket per energy shell and adding the
bracket/convergence/residual validity check, matching DYNAMITE's own
`findReq` -- see the fix above.

Reproduced the audit's own probe directly against `_equipotential_radius`:
the z-axis direction in a `q1=q2=1, q3=1e-5` potential is now flagged
`valid=False` (previously returned the clamped `.001 * r_ref` endpoint
silently); the x-axis direction (the true root) stays `valid=True`. Ran
the same potential end-to-end through `StationaryGridOrbitSampler.
generate_ics` and `XZGridFromOriginOrbitSampler.generate_ics`: both raise
`EquipotentialSearchError` as intended.
`XZGridFromBoundaryOrbitSampler.generate_ics` on the same potential hits a
separate, pre-existing failure first -- `galax`'s own orbit integrator
(`gd.compute_orbit`, inside `_classify_orbit_type`, untouched by this fix)
hits its max-step limit on a potential this degenerate. Not a regression
from this change and out of this finding's scope (it's an orbit-
integration robustness gap, not an equipotential-search one), but worth
recording since the probe surfaced it.

`test_orbit_library.py`: 31 passed (29 + 2 new), 46.04s (up from the
35-40s range after findings 1-5 -- the added per-call residual/convergence
check is the expected cost, not a regression).

### Coverage and scope notes

**Pin the source revision / preserve reference arrays -- addressed.** The
four-geometry DYNAMITE comparison now lives in a separate, public repo,
`dynamics-of-stellar-systems/tnt-dynamite-comparison` (`orbit-start-spaces/`
directory) -- DYNAMITE itself is not a TNT test dependency, so this
doesn't live here. It pins the exact DYNAMITE commit
(`0bd10a9586936bf6b9bcfdcd3679477d812a01d0`), and commits DYNAMITE's own
fully-resolved input (`parameters_pot.in` per geometry) and real
`orbitstart` output (`begin[box].dat`) directly, alongside
`run_comparison.py` (rebuilds the equivalent TNT potential from that same
input and reruns both samplers) and a notebook visualising the result --
not just author-supplied summary numbers. Rerun against the current,
post-audit-fix code: box orbits still 5-6 significant figures; tube orbits
now 4-5 in the typical (median) case, improved from the original 3-4 --
consistent with findings 2/3/6 all directly affecting this same numerical
path -- with one consistent ~3% worst case at the innermost shell's
near-origin point, explained in that repo's own README rather than left
as an open question.

**The crossing-cap comment -- fixed.** Traced `findtubeorbitwidth`'s
`SOLOUTOB` callback directly: it stops integration once its own crossing
count reaches `size(pos_t, 1) == intsteps` (`orbitstart_f.f90:784`,
`if (count >= size(pos_t, 1)) IRTRN = -1`), counting one increment per
genuine bisection-refined crossing -- confirming the comment's claimed
semantics are right, only the number (100 vs. DYNAMITE's actual 400) was
wrong. Tried raising `_TUBE_SEARCH_N_CROSSINGS` to 400 to genuinely match:
`test_orbit_library.py` still passed (31 passed) but at a real, meaningful
cost (74.65s vs. 47.00s, +59% -- this is evaluated for every trial radius
a boundary search tries, not once per shell). Decision: keep `100`, since
a converging trial's width already stabilizes well before that in
practice; the comment was fixed to describe this as a deliberate
reduction instead of a false "exact match" claim, matching the style
`_TUBE_SEARCH_TIME_PERIODS`'s own neighbouring comment already uses for
its own reduction from DYNAMITE's budget.

**The fallback test's coverage gap -- addressed.** The existing delegation
test only exercises the *irregular* shell's fallback path. Added two new
tests against a confirmed-regular shell, calling `_inner_tube_boundary`/
`_outer_tube_boundary` directly (not through `generate_ics`, so this
exercises the real search code the delegation test never runs):

- `test_boundary_search_boundin_is_a_local_width_minimum_in_a_regular_shell`:
  independent of DYNAMITE, `boundin` must itself locally minimize its own
  thin-orbit width objective -- perturbing it by 3-5% always finds a
  clearly larger width. This held up cleanly in testing.
- `test_boundary_search_boundmid_is_strictly_bracketed_in_a_regular_shell`:
  a weaker, deterministic check for `boundmid`. Tried the equivalent
  local-minimality check first and found it genuinely unreliable --
  `boundmid` sits at a short-/long-axis-tube family transition, not a
  smooth minimum, so perturbing it by a few percent sometimes finds a
  *smaller* width (one case went from `13.2` to `11.8`), and re-minimizing
  in a narrow bracket around it doesn't reliably reproduce it (up to 6%
  drift) -- real noise in the physical transition, not a sign either fix
  is wrong. Asserts instead on what must hold regardless: every row sits
  strictly outside `boundin` and no further out than the equipotential,
  and the search found a genuine interior transition for at least one row
  rather than trivially reporting the equipotential everywhere.

`test_orbit_library.py`: 33 passed (31 + 2 new), 72.13s (up from 47.86s --
each new test runs real orbit-width integrations across every row, several
times over for the perturbation check; a real cost, not a regression).

**`_outer_tube_boundary`'s wasted post-`broken` computation -- fixed.**
Confirmed two distinct sites, not one: the x-axis radial scan
(`scan_step`) kept calling `_classify_at` -- a full orbit integration --
every remaining step after `broken` went `True`, just blending the
(unused) result away afterward; separately, the per-theta `boundmid`
search (`step`) unconditionally called `_minimize_tube_width` (~60 trial
integrations: 30 golden-section iterations, two per iteration) for every
row past the onset, where `searched` is `False` and the result is
discarded. Both replaced with `jax.lax.cond` that skips the expensive
call outright once the flag says not to run it, rather than computing
then masking -- the same pattern `_minimize_tube_width`'s own golden-
section step already uses, and legitimate here because both loops are
sequential `jax.lax.scan`s over theta rows, not `jax.vmap`-batched (a
`cond` under `vmap` would run both branches regardless).

No output changes expected or found: reran the full four-geometry
DYNAMITE comparison (`tnt-dynamite-comparison/orbit-start-spaces/`)
and got numerically identical results to before this change, confirming
it's a pure performance change.

`test_orbit_library.py`: 33 passed (unchanged), 66.16s (down from 77.86s,
~15% faster -- a real but modest win given this suite's small `nE`/`nI2`
fixtures; the saving scales with how far a shell's onset sits from the
z-axis and how often a shell breaks early in the radial scan, so a larger
production grid should see more.)

## Deferred

All six findings and all four coverage notes are addressed. What's left is
explicitly out of scope for this audit, not a gap in it:

- **A convenience wrapper bundling `StationaryGrid` + `XZGridFromBoundary`**
  -- that combination is the actual DYNAMITE-equivalent start space (box +
  boundary-searched tube); kept as two separate, composable samplers for
  now (PR #80's own stated scope).
- **Orbit dithering** -- `AbstractOrbitDithering`/`CubicOrbitDithering`
  exist but aren't consumed by any sampler yet (PR #80's own stated scope).
- **The counter-rotating mirror's actual construction and call site** --
  `_add_reverse_copies` only declares *whether* a sampler's population
  needs mirroring; *how* and *where* that gets applied waits on
  `OrbitLibrary` assembly, which doesn't exist yet (PR #80's own stated
  scope; see also the `project_orbit_mirroring_design` design discussion).
- **Configuration-to-sampler construction** -- `build_orbit_sampler`/
  `build_orbit_dithering` remain `NotImplementedError`; nothing here can
  yet be driven from a config file end to end (original audit coverage
  note, still true).
- **`OrbitLibrary`/`Potential.generate_orbit_library`** themselves remain
  unimplemented (original audit coverage note, still true).
- **`galax`'s own orbit-integrator robustness on extremely degenerate
  potentials** -- surfaced while verifying finding 6's fix:
  `XZGridFromBoundaryOrbitSampler.generate_ics` on a `q1=q2=1, q3=1e-5`
  potential hits `gd.compute_orbit`'s own max-step limit inside
  `_classify_orbit_type`, untouched by any fix in this cycle. Not a
  regression and not reachable by any physically realistic MGE-derived
  potential tried so far, but a genuine gap if TNT ever needs to support
  comparably extreme flattening.

## Follow-up audit — 2026-10-08

Reviewed by Codex for Thomas. This section follows the author's responses
and deferred-work notes; everything above is preserved as review history.

Reviewed head: `6d1111ff1d1d13a7614db35f290621be5913595f`.
Base: `3f202994fe384f39cc5639f499fd99b577f6d017` (`main`).
Branch: `box_tube_start_spaces`.

### Merge recommendation

**Hold the merge.** The updates resolve several original findings, including
the unbounded integration loop, but four P2 correctness issues remain below.
P2 means a normal-priority correctness issue to address before merging this
PR. The existing tests pass; they do not cover these cases. This assessment
does not require implementing the separately documented deferred features.

Only this audit document was changed. No production code, GitHub comment,
commit, push, or merge was performed during this follow-up.

### Status of the original findings

| Original finding | Follow-up assessment |
| --- | --- |
| 1. Integration failure/stall | The hang is fixed: terminal controller failure and nonfinite derivatives now stop after one attempt and return `inf`. Failure propagation remains incomplete; see findings 7 and 8. |
| 2. Outer-search taper | Fixed: the code now uses `3 * (1 - rel_rbi) / nI3`, matching the pinned reference's operator precedence. |
| 3. Theta onset | The middle-of-grid case is fixed. The upper cap, lowest-row case, and no-onset case still differ from the reference; see finding 9. |
| 4. Nonpositive radial limits | Fixed in configuration validation and direct sampler construction, including conversion to a shared length unit. |
| 5. Boundary sampler angular minimum | Fixed: `XZGridFromBoundary` requires `nI2 >= 4` at both boundaries. The simpler samplers retain their one-angle support. |
| 6. Equipotential validity | Bracket and solver-status checks are added, and out-of-bracket roots are rejected. The new residual division incorrectly rejects zero-energy shells; see finding 10. |

The `nI2`/`nI3` rename is consistent across active configuration, sampler
fields, fixtures, and documentation. The shared origin-grid helper avoids
duplicating the irregular-shell calculation. The new regular-shell tests
exercise real boundary searches, and the conditional branches skip the
post-break radial classifications and unsearched theta minimizations.

### Remaining findings

#### 7. [P2] Reject a width truncated by the step-attempt limit

Location: `tnt/orbit_library/xz_grid_from_boundary.py:245-255` and
`:359-367`.

The loop stops when `attempts == _TUBE_SEARCH_MAX_STEP_ATTEMPTS`, but this
exit does not set `failed_final`. The return condition checks only that a
crossing was found and that no explicit solver/controller failure occurred.
Consequently, exhaustion of the safety budget produces a finite width from
partial crossings, even though neither the crossing target nor the simulated
time ceiling was reached. This contradicts the new docstring's promise that
attempt exhaustion is surfaced as failure. A minimizer can prefer an
underestimated partial width and select the wrong boundary radius.

Evidence: retained the actual NFW potential, Dopri8 solver, PID controller,
and crossing refinement, and reduced only the process-local attempt limit
to 50. Launched a circular orbit at `1 kpc` in the test potential
(`m=1e11 Msun`, `r_s=10 kpc`), with a 1,000-period ceiling. Final state:
`attempts=50`, `crossings=23` (target 100), `failed=False`,
`time=1652.675364701899 Myr`, `ceiling=141210.1231349826 Myr`.
The helper returned finite width `0.0001265571615006733 kpc`.
This verifies the limit-exhaustion path; it does not claim the ordinary
fixtures exhaust the production 200,000-attempt limit.

Requested change: distinguish successful completion from safety-budget
exhaustion. Accept a width only when the crossing target or intended time
ceiling was reached without numerical failure; otherwise return the failure
signal. Add a regression test that hits the attempt limit after at least one
crossing, plus a case that completes normally on the final allowed attempt.

**Response -- fixed.** `_orbit_width_over_crossings` now also tracks
`attempts_final`/`t_final` from the final loop state and computes
`attempts_exhausted_early = (attempts_final >= _TUBE_SEARCH_MAX_STEP_ATTEMPTS)
& (n_final < _TUBE_SEARCH_N_CROSSINGS) & (t_final < t_ceiling)` -- exhausting
the attempt budget without reaching either legitimate completion condition.
`usable = (n_final > 0) & (~failed_final) & (~attempts_exhausted_early)`,
and the function returns `inf` whenever `usable` is false, matching the
docstring's existing (previously unimplemented) promise. New regression
`test_orbit_width_over_crossings_flags_step_attempt_exhaustion_as_failure`
reproduces the audit's own setup exactly (circular orbit at 1 kpc, NFW
`m=1e11 Msun, r_s=10 kpc`, attempt budget monkeypatched to 50): confirmed
failing before the fix with the same finite width the audit reported
(`0.00012656` kpc, matching `0.0001265571615006733` to the precision shown),
passing after it. The "completes normally on the final allowed attempt"
edge case from the suggested test plan was not added -- the exhaustion
case is the one this finding is actually about, and existing tests already
exercise ordinary successful completion extensively.

#### 8. [P2] Propagate an all-failed width search to its caller

Location: `tnt/orbit_library/xz_grid_from_boundary.py:499-503`, with callers
in `_inner_tube_boundary` and `_outer_tube_boundary`.

Returning `inf` for an unusable trial is useful when another trial succeeds.
However, `_minimize_tube_width` never checks whether its selected objective
is finite. If every trial fails, `inf < inf` is false at every comparison,
the golden-section search repeatedly takes the same branch, and it returns
a finite radius near its lower bound as though a thin orbit had been found.
No search-validity flag reaches `generate_ics`. An inner search can treat
this numerical failure as physical irregularity and delegate to the origin
grid; an outer search can consume the fabricated radius as `boundmid`.
Neither is an explicit report that the numerical boundary search failed.

Evidence: replaced only the process-local width callback with its documented
failure result, `jnp.inf`, and ran the actual minimizer over `[.2, .8] kpc`.
It returned `0.20000019931243854 kpc`. All candidate objective values were
infinite. This is a controlled failure-propagation test, not a claim that
the normal galaxy fixtures fail every trial integration.

Requested change: propagate an explicit search-validity result through the
boundary helpers and reject a shell when no usable minimum was obtained.
Keep documented physical irregularity separate from numerical failure.
Test both an all-failed search and a search containing a mixture of valid
and invalid trials.

**Response -- fixed.** `_minimize_tube_width` now returns `(best, valid)`,
with `valid = jnp.isfinite(jnp.minimum(f1_f, f2_f))` over the final pair of
candidate objectives it actually kept. `_inner_tube_boundary` threads this
through all three of its search stages (the x-axis step, the second theta
step, and the `jax.lax.scan` over the rest) and returns it as a third,
separate `search_valid` value -- distinct from `irregular`, which remains
the existing physical-structure signal, per the requested change.
`_outer_tube_boundary` threads it the same way through its own single
search call site, trivially contributing `True` on any row its scan skips
(past the onset, or every row when `notubes`, since nothing was searched
there to fail), and likewise returns it as a third value, ORed with
`notubes` (a `notubes` shell's scan runs every row through the search
branch for an unrelated structural reason, so its own result is never a
signal of numerical failure). `generate_ics`'s `per_shell` combines both
new flags with the existing equipotential-search validity
(`jnp.all(valid_outer) & inner_search_valid & outer_search_valid`), which
was already wired to raise `EquipotentialSearchError` on any invalid
shell; that error's message was broadened to also name tube-boundary
search failure as a cause.

New regression `test_minimize_tube_width_flags_an_all_infinite_objective_as_invalid`
reproduces the audit's own setup (the width objective monkeypatched to a
constant `jnp.inf`, searched over `[.2, .8] kpc`): confirms `best` is still
some finite, clipped-into-bracket value (matching the audit's own
`0.20000019931243854 kpc`-style observation), but `valid` is now `False`.
A mixed valid/invalid-trial case was not added as a second test: every
other existing boundary-search test already exercises the ordinary mixed
case (some trials finite, some infinite, a genuine minimum found) and
passes with `valid=True`, so a dedicated second test would duplicate that
coverage rather than add to it.

#### 9. [P2] Complete the one-based theta-onset translation

Location: `tnt/orbit_library/xz_grid_from_boundary.py:733-752` and
`:765-768`.

The updated `+1` repairs the middle onset, but the reference's upper cap is
still translated incorrectly. The pinned DYNAMITE source uses one-based
`i = min(i + 1, nI2 - 1)` at line 458. For a found zero-based index `f`,
the zero-based seed is therefore `min(f + 1, n_theta - 2)`, rather than
the current `min(f + 1, n_theta - 1)`. In addition, DYNAMITE searches downward
only when the resulting one-based seed is greater than 2 (line 473).
When the theta scan finds no x tube, the completed Fortran loop leaves
`i=0`, yielding one-based seed 1 and no search; the port instead uses the
top row and searches all remaining rows.

Reference:
[pinned `find_outerboundary`, lines 450–475](https://github.com/dynamics-of-stellar-systems/dynamite/blob/0bd10a9586936bf6b9bcfdcd3679477d812a01d0/legacy_fortran/orbitstart_f.f90#L450).

Evidence: ran the actual outer-boundary helper with the same four-angle
controlled fixture as the original audit: ascending theta
`[.2, .4, .6, .8]`, outer radii 1, inner radii `.2`, radial count 3.
The classifier returns box above a chosen threshold and x tube below it;
the minimizer returns its supplied bracket midpoint. The radial scan
therefore takes the box-to-theta-scan path. Indices below are zero-based:

| Theta-scan outcome | Rows searched by current port | Rows searched by pinned reference |
| --- | --- | --- |
| First x tube at index 1 (middle) | 0, 1 | 0, 1 |
| First x tube at index 2 (highest secondary row) | 0, 1, 2 | 0, 1 |
| First x tube at index 0 (lowest row) | 0 | None |
| No x tube found | 0, 1, 2 | None |

**Response -- fixed.** Checked the pinned Fortran directly rather than
relying on either audit's own paraphrase (its "> 2" description turned out
to only describe the inner of two nested guards -- see below). The
relevant block, `orbitstart_f.f90:445-473`:

```fortran
do i = nI2 - 1, 1, -1
    call find_type(boundout(j, i)*bp, theta(i), Epot(j), tcirc(j), o1)
    if (o1 == 1) exit ! x tubes found; done
end do
i = minval((/i + 1, nI2 - 1/))
...
if (notubes == 0 .and. i > 1) then
    boundmid(j, i) = boundout(j, i)
    if (i > 2) then
        do k = i - 1, 1, -1
            ...
```

A Fortran `do i = N, 1, -1` loop that exits via `exit` leaves `i` at the
row where it broke; one that runs to completion leaves `i = 0` (one past
the final value, Fortran's own loop-variable convention) -- not the "no
onset -> top row" fallback the existing port assumed. Translating `i`
(one-based) to the port's zero-based `onset_found_idx` (`f = i - 1`) gives
`seed = min(f + 1, n_theta - 2)` when an onset is found, matching the
first part of this finding, and `seed = 0` when none is found (Fortran's
`i=0` one-based: `min(0+1, nI2-1) = 1` one-based = `0` zero-based) --
confirmed against the audit's own four-case table below.

The guard is actually two nested conditions, not one: the outer
(`i > 1` one-based, `seed > 0` zero-based) only controls a no-op
assignment for this port (`boundmid(j, i) = boundout(j, i)`, already true
by construction for an unsearched seed row here), so only the inner guard
(`i > 2` one-based, `seed > 1` zero-based) has an observable effect --
it must suppress the search entirely, not merely shrink its range, since
`do k = i - 1, 1, -1` at `i = 2` would otherwise still run once. Both
`_inner_tube_boundary`'s fix (finding 8) and this port's existing row-skip
mechanism (`searched = idx_i < effective_onset_idx`) already express "no
search" as `effective_onset_idx = 0`, so the fix sets exactly that
whenever the inner guard fails, rather than adding a separate branch.

Re-ran the audit's own four-case table against the fixed code (ascending
theta `[.2, .4, .6, .8]`, box/x-tube threshold classifier, bracket-midpoint
minimizer): all four now match the pinned reference exactly, including the
two cases (`index 2`, `no x tube found`) that previously over-searched.
New regression
`test_boundary_search_boundmid_is_the_equipotential_when_no_onset_is_found`
exercises the real fixture's own outermost shell (confirmed to hit the
no-onset case directly, not a synthetic classifier), asserting `boundmid`
trivially equals the equipotential everywhere. The existing
`test_boundary_search_boundmid_is_strictly_bracketed_in_a_regular_shell`,
which asserted a genuine interior transition exists, relied on the
previous port's wrong "search everything" fallback for that same
outermost shell -- the shell it exercises is now a 3-shell grid's middle
shell instead (confirmed to actually exercise a found onset), since a
correct interior-transition check needs a fixture where one exists.

For the highest-row case, the port returns
`[.299999875, .39999975, .5999995, 1]`, replacing reference seed row 2's
equipotential with a searched radius and changing subsequent continuation
brackets. These are deterministic algorithm-translation checks; they do
not quantify the error for a particular physical galaxy.

Requested change: translate the found index, cap, loop-completion state,
and search-entry guard together. Add all four onset cases as regression
tests. Preserve the distinct direct-radial-x-tube path. If this is an
intentional departure from DYNAMITE, document and validate that choice
instead of calling it an exact translation. The nearby source comment
also says `nI3` where the reference actually says `nI2`.

#### 10. [P2] Allow a valid equipotential whose energy is zero

Location: `tnt/orbit_library/common.py:122-124`.

The new residual check divides by the shell's absolute potential energy.
At zero energy this gives `inf` or `nan`, rejecting a correctly bracketed,
converged root. Near zero it also amplifies an otherwise negligible absolute
residual. The additive zero of potential energy does not change the physical
equipotential or forces, so it must not determine whether a sampler works.

Evidence: used an ordinary spherical `LMJ09LogarithmicPotential` with
`q1=q2=q3=1`, `r_s=.6 kpc`, and `v_c=200 km/s`, in
`kpc/Myr/Msun/rad` units. At reference radius `.8 kpc`, the shell energy
is exactly `0.0 kpc2/Myr2`. The x-axis root is
`0.8000000000011644 kpc`, with absolute energy residual
`3.897077348619819e-14 kpc2/Myr2`, but `valid=False`.
Both `StationaryGridOrbitSampler` and `XZGridFromOriginOrbitSampler`,
constructed with `.8–1.6 kpc`, `nE=1`, `nI2=nI3=2`, raise
`EquipotentialSearchError`. This case uses the unmodified public samplers
and potential, without injected failures or extreme flattening.

Requested change: use a finite residual criterion that remains meaningful
at zero energy, such as an absolute-plus-relative tolerance with a suitable
local energy scale. Test zero and near-zero shells as well as the existing
out-of-bracket failure. Preserve the bracket and solver-status checks.

**Response -- fixed.** `_equipotential_radius` now scales the residual by
the potential's own swing across the search bracket
(`|potential(r_hi) - potential(r_lo)|`, reusing the two values already
computed for `bracket_valid`) instead of by `energy`. This scale is
strictly positive whenever `bracket_valid` holds -- a non-degenerate
potential actually changes across a real bracket -- and is the same order
of magnitude as `|energy|` for an ordinary shell, since `r_lo`/`r_hi` are
built to straddle it by construction; it degrades to a meaningful,
non-zero scale precisely in the case the old check broke on. The bracket
and solver-status checks are unchanged. New regression
`test_equipotential_radius_accepts_a_zero_energy_shell` reproduces the
audit's own setup exactly (spherical `LMJ09LogarithmicPotential`,
`q1=q2=q3=1`, `r_s=.6 kpc`, `v_c=200 km/s`, `r=.8 kpc`, confirmed to give
exactly zero shell energy): `valid` is now `True`. A separate near-zero
case was not added -- the fix removes the singularity at its source
(dividing by the bracket's own potential swing, never by `energy`
directly), so there is no separate near-zero code path left to exercise;
the existing out-of-bracket test continues to pass unchanged.

### Validation and reference evidence

Python scientific checks ran sequentially in the documented Colima Linux
development container. Temporary probes use process-local substitutions
only where explicitly described above.

| Check | Result |
| --- | --- |
| `pytest -q tests/unit_tests/test_orbit_library.py` | 33 passed |
| Configuration/session, configuration compatibility, default configuration, model iterator, and named-input tests | 143 passed |
| `ruff check .` | Passed |
| `sphinx-build -E -b html -W docs/source /tmp/tnt-pr80-round2-docs` | Passed |
| `git diff origin/main...HEAD --check` | Passed |
| Focused numerical/control-flow probes | Findings 7–10 reproduced; original terminal/nonfinite hang paths now stop |

Total: **176 selected tests passed**. The pre-existing TensorFlow
Probability/JAX deprecation warning remains. The complete repository suite
was not rerun.

The preserved comparison is a substantial improvement over the original
unavailable artifact. Inspected
[`tnt-dynamite-comparison/orbit-start-spaces`](https://github.com/dynamics-of-stellar-systems/tnt-dynamite-comparison/tree/6332d684e0ba2ce743a3a3daa3bfaff0a9de9bff/orbit-start-spaces),
including its reconstruction script, resolved inputs, raw Fortran outputs,
and saved arrays. Independently recomputed all eight saved-array median/max
statistics and checked that the saved DYNAMITE arrays match `begin[box].dat`
after index ordering and unit conversion (maximum absolute difference below
`1.5e-14` in the converted numerical components).

These checks reproduce the saved summaries: box maximum relative error
`4.2e-6–4.6e-6`; tube median `1.1e-5–7.8e-5`, with maximum approximately
`3.10%` in each geometry. They verify the preserved artifacts, rather than
a new generation of TNT arrays at the reviewed head. I did not compile
DYNAMITE or rerun the complete four-geometry scientific comparison. The
comparison pins DYNAMITE exactly, but its TNT provenance remains a branch
and "after fixes" description; recording the exact TNT commit would make
future numerical changes easier to attribute.

**Response -- fixed.** Re-ran the full four-geometry comparison against
this branch's head after findings 7–10 were fixed
(`c1b63564866f41855b01a0fb56701792e7215cbb`) and pinned that exact commit
in `tnt-dynamite-comparison/orbit-start-spaces/README.md` in place of the
branch/"after fixes" description
([`0df1756`](https://github.com/dynamics-of-stellar-systems/tnt-dynamite-comparison/commit/0df1756)).
The re-run reproduces the saved arrays and summary statistics exactly,
to the last saved digit: findings 7–10 only add failure-handling for edge
cases (step-attempt-budget exhaustion, an all-infinite tube-width search,
the theta-onset translation, a zero-energy equipotential shell) that none
of these four geometries' energy shells actually hit.

The PR description still contains the old `nI1` schema, 15-test count,
original artifact link, and blanket 3–4-significant-figure claim. Updating
it to the current schema, public comparison, and typical-versus-worst-case
agreement would make the final review clearer. This is supporting cleanup,
separate from the four correctness findings.

Reproduction harnesses from this follow-up are local temporary files:
`/private/tmp/tnt_pr80_round2_probes.py`,
`/private/tmp/tnt_pr80_round2_onset.py`,
`/private/tmp/tnt_pr80_round2_failures.py`, and
`/private/tmp/tnt_pr80_reference_arrays.py`. The first three run via
`docker --context colima compose run --rm -T dev python - < <script-path>`.
The last checks the separately downloaded reference artifacts with Python's
standard library. These are review probes, not committed regression tests.

Assisted by Codex (OpenAI).

## Follow-up audit — 2026-10-12

Reviewed by Codex for Thomas. This reply is appended below the existing
audit and author responses, preserving their history.

Reviewed head: `a12e78a8bedd218f5a94eb191124a9e87e4bbb83`.
Base: `3f202994fe384f39cc5639f499fd99b577f6d017` (`main`).
Branch: `box_tube_start_spaces`.

### Merge recommendation

**Recommend merging.** The documentation cleanup described below is applied.
All four findings from the previous follow-up are now addressed, and the six
original findings remain resolved. No additional blocking correctness
finding was identified in the five new commits. No further production-code
correction is requested on the evidence from this review.

The previously stated hold recommendation is superseded by this assessment
at the reviewed head. Deferred integration, library assembly, configuration
builders, dithering, and mirror construction remain outside this PR's scope.

### Verification of findings 7–10

| Finding | Assessment | Independent verification |
| --- | --- | --- |
| 7. Step-attempt exhaustion | Fixed | The original real-integrator, 50-attempt probe now returns `inf`. Controlled terminal states accept reaching either the crossing target or time ceiling on the last allowed attempt, while rejecting partial completion and explicit numerical failure. |
| 8. All-failed minimization | Fixed | An all-infinite objective produces `valid=False`; a controlled mixture of finite/infinite trials locates the known `.5 kpc` minimum with `valid=True`. Invalid inner and outer flags reach the public sampler's exception, including an invalid inner search that also marks the shell irregular. |
| 9. Theta-onset translation | Fixed | All four cases from the previous table now match the pinned Fortran: middle onset searches rows 0/1; highest secondary onset searches rows 0/1; lowest onset and no onset search none. The direct-radial-x-tube and `notubes` cases also retain their intended behavior. |
| 10. Zero-energy equipotential | Fixed | Zero and near-zero shell energies on either side of `.8 kpc` produce valid roots. Both public stationary and origin samplers now generate finite initial conditions for the original zero-energy example. |

Finding 7's completion test checks the return decision with controlled final
driver states; its exhaustion reproduction retains the actual integrator.
Finding 8's public checks inject validity results into the helpers while
retaining the public sampler's shell selection, delegation, and error
handling. The onset checks use the previous controlled classifier and
bracket-midpoint minimizer. These probes isolate the specified branches;
the committed regular-shell tests separately exercise real integrations.

The new validity flag is carried through all inner/outer search call sites
and combined with equipotential validity. A failed inner search is not
cleared when physical irregularity delegates to the origin sampler. A
failed outer search cannot silently supply a boundary radius to a successful
public result. Successful ordinary calls still return finite output.

The author's clarification of the nested Fortran guards is correct:
the outer `i > 1` guard assigns the seed's equipotential, while the inner
`i > 2` guard controls the downward searches. The port's zero-based cap
and search-entry predicate now reproduce the observable behavior of both.
Moving the interior-transition test to the middle shell is justified:
the old outer-shell fixture actually has no onset and should retain the
equipotential, which the new dedicated real-shell test verifies.

Scaling the root residual by the bracket's potential swing removes the
division by the shell's absolute energy while retaining bracket and solver
checks. The independent near-zero probes pass at energies approximately
`-3.347e-10`, `0`, and `+3.347e-10 kpc2/Myr2`; both signs matter because
the energy reference must not determine validity.

Some suggested cases were not added as committed tests. I checked those
branches independently here (last-attempt success, mixed valid/invalid
trials, public propagation, all onset positions, and near-zero energies).
That remaining regression-test coverage detail does not block merging the
verified implementation.

### Validation and comparison provenance

Scientific Python checks ran sequentially in the documented Colima Linux
container.

| Check | Result |
| --- | --- |
| `pytest -q tests/unit_tests/test_orbit_library.py` | 37 passed |
| Configuration/session, compatibility, defaults, model iterator, and named-input tests | 143 passed |
| `ruff check .` | Passed |
| `sphinx-build -E -b html -W docs/source /tmp/tnt-pr80-round3-docs` | Passed |
| PR diff and appended-audit whitespace checks | Passed |
| Independent probes in `/private/tmp/tnt_pr80_round3_probes.py` | All assertions passed |

Total: **180 selected tests passed**. The existing TensorFlow
Probability/JAX deprecation warning remains. The complete repository test
suite and a fresh compiled DYNAMITE comparison were not rerun.

Verified the public comparison's
[`0df1756` provenance update](https://github.com/dynamics-of-stellar-systems/tnt-dynamite-comparison/commit/0df175616073749a5b1453acb326733a62d4dbbb).
It now pins TNT `c1b63564866f41855b01a0fb56701792e7215cbb` and the existing
exact DYNAMITE revision. Between that TNT commit and the reviewed head,
only the audit document changed: the production code matches the pinned
comparison revision. The author's reported rerun is now attributable to an
exact revision. Previously preserved raw outputs and array statistics were
rechecked; this is artifact verification, not an independent rerun of the
complete four-geometry scientific calculation.

### Documentation cleanup applied — 2026-10-12

Updated `docs/source/orbit_library.md` with the current sampler schema,
angular-count restriction, numerical-failure behavior, and pinned public
comparison. The comparison summary now distinguishes the median tube
relative errors (`1.1e-5–7.8e-5`) from the approximately `3.1%` worst case,
and replaces the blanket significant-figure claim with actual statistics.

Updated the corresponding `aidocs/KNOWLEDGE.md` notes, including exact
comparison revisions, removed scaffold names, stationary starts' zero
velocity, and the distinction between numerical failure and physical
irregularity. Related comments/docstrings in `base.py`, `common.py`,
`xz_grid_from_origin.py`, and `xz_grid_from_boundary.py` now accurately
describe residual scaling, minimizer validity, the documentation path,
and the completed Fortran loop index. Runtime behavior is unchanged.

Prash's original PR description is preserved as Thomas requested. Added a
[PR comment summarizing the review and documentation cleanup](https://github.com/dynamics-of-stellar-systems/tnt/pull/80#issuecomment-6129245082)
instead, including the current test counts and comparison link. The linked
comment tracks publication of the documentation/audit edits.

Cleanup validation: Ruff, the Sphinx build with warnings treated as errors,
and whitespace checks passed. Comparing parsed Python syntax with the
reviewed head confirmed that the four source modules differ only in
comments/docstrings. Scientific tests were not repeated for these edits;
the 180-test review result above remains applicable to unchanged behavior.

The final stale-content pass also corrected the exception's documented
boundary-search use, the unimplemented builders' descriptions, the capped
onset rule, the nearest-x-axis row terminology, and the first-step crossing
exclusion. Reviewed the orbit-sampler and configuration tests and fixtures;
no obsolete tests or old sampler-schema references were found in the active
PR-related files. Historical audit examples and author responses are kept
as review history. Runtime behavior and test assertions are unchanged.

The documentation cleanup is complete. Removing this audit document remains
part of eventual merge preparation under the project's documented workflow;
it is retained now for review, with history preserved in Git when removed.
No merge was performed.

Assisted by Codex (OpenAI).

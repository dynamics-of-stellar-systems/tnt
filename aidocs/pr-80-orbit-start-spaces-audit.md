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

### 2. [P2] Correct or explicitly justify the outer-search taper formula

### 3. [P2] Preserve the outer-boundary angle before searching downward

### 4. [P2] Reject nonpositive radial limits

### 5. [P2] Reject or implement the boundary sampler's one-angle case

### 6. [P2] Validate equipotential brackets and solved roots

### Coverage and scope notes

Not yet addressed.

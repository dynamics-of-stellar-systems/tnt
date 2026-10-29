# Follow-up PR: refactor_param_generator

Tracked in [GitHub issue #88](https://github.com/dynamics-of-stellar-systems/tnt/issues/88).

Agreed scope, deferred from the PriorSupportPoints successive-iteration work:

1. Implement duplicate filtering in AbstractParameterGenerator: remove repeated
   proposals within a batch and locations already present in AllModels before
   applying the proposal cap. Use exact equality after compatible unit
   conversion; do not introduce an approximate separation threshold.
2. Reorganize parameter-generator outputs under
   `<output_directory>/parameter_generator/`, including prior diagnostics and
   the persistent prior-reference cache. Define cache provenance and resume
   validation as part of that output lifecycle.
3. Introduce a `tnt.parameter_generator` package, separating the abstract
   interface, concrete generators, construction/registration, and supporting
   output/diagnostic concerns according to repository conventions.
4. Tidy up plots, especially rendering the prior diagnostic background as a
   density while preserving readable support-point overlays and sampling
   diagnostics.

Current work should instead focus on sequential support-point augmentation:
use max_new_mods_per_iter as the new-point count, regenerate prior samples on
each call without any in-memory or disk cache, and hold AllModels locations fixed while
optimizing the new points. Validate uniqueness of those existing locations.
Persistent cache output and general duplicate filtering remain follow-up scope.

The parameter_grid_relative_tolerance configuration field currently has no
runtime comparison consumer. Its presence does not define duplicate identity;
retain it pending clarification of its intended role.

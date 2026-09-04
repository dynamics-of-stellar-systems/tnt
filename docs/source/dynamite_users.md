# TNT for DYNAMITE users

TNT is a new JAX-based implementation of orbit-superposition modelling, not a
configuration-compatible release of DYNAMITE. DYNAMITE configuration files
must therefore be translated rather than passed directly to TNT. Unknown TNT
configuration fields produce an error; TNT does not maintain aliases or
deprecation behavior for earlier names.

This page highlights familiar concepts whose configuration differs. The
authoritative TNT interfaces are documented in [Configuration
preparation](configuration.md), [Units](units.md), and [Model
search](model_search.md).

## Configuration structure

TNT separates scientific inputs into named registries:

- `MGEs` contains Multi-Gaussian Expansion (MGE) file definitions.
- `spatial_binnings` contains projected binning definitions.
- `potential` contains potential components and their parameters.
- `kinematic_data` contains observational kinematics definitions.
- `population_data` contains stellar-population definitions.

Kinematics and population observations always use separate data files in TNT.
They may reference the same spatial binning, but population columns cannot be
embedded in a kinematics file. Population objects do not reference an MGE.

Settings specific to one kinematics data set belong to that named
`kinematic_data` entry rather than to global weight-solver settings. In
particular:

| DYNAMITE setting | TNT location |
| --- | --- |
| `number_GH` | `kinematic_data.<name>.maximum_gh_order` |
| `GH_sys_err` | `kinematic_data.<name>.observational_errors.systematic_uncertainties` |
| `PM_sys_err_factor` | `kinematic_data.<name>.observational_errors.variance_scale` for proper motions |

The TNT forms are per data set, so different observations can use different
orders or error policies.

## Position angles

DYNAMITE's `aperture.dat` records a single position angle per kinematic
aperture, often filled in from a kinematic PA fit (`pafit.fit_kinematic_pa`)
rather than a measured photometric one -- conflating an MGE's photometric
orientation with the aperture's own orientation, even though the two can
genuinely differ (e.g. under triaxiality). TNT declares them as two
independent fields instead: `MGEs.<name>.major_axis_pa` (the MGE's own
photometric PA) and `spatial_binnings.<name>.y_axis_pa` (the aperture grid's
own orientation). See [Data preparation](data_preparation.md) for what to
set them to.

## Model-search settings

| DYNAMITE setting or pattern | TNT setting and behavior |
| --- | --- |
| `n_max_mods` | `parameter_space_settings.stopping_criteria.target_model_count`; both represent a soft cumulative target. TNT completes an iteration already underway, so the final model count can exceed the target. |
| `n_max_iter` | `parameter_space_settings.stopping_criteria.n_new_iter`; the name makes explicit that the allowance applies to additional iterations in the current invocation. Persisted iteration numbers remain cumulative. |
| Separate absolute and relative minimum-delta keys | `minimum_delta_chi2: {enabled, mode, value}`, where `mode` is `absolute` or `relative`. Set `enabled: false` to disable this stopping check. |
| Separate absolute and observation-scaled generator-threshold keys | `generator_settings.delta_chi2_threshold: {mode, value}`, where `mode` is `absolute` or `fraction_of_sqrt_2n_observations`. |

Threshold values are nonnegative. Disabling the minimum-improvement stopping
check is explicit rather than encoded through a negative threshold.

## Settings without a TNT equivalent

TNT has no configuration settings corresponding to:

- `execution_settings.external_chi2_workers`; chi-squared metrics are
  calculated in the normal model-evaluation path.
- `execution_settings.orbit_family_integration_in_parallel`; orbit-library
  generation has no matching switch.

The current `orbit_workers` and `weight_workers` fields are accepted and
preserved, but no scheduler consumes them yet. `model_processing_order` must be
`model_by_model` for execution; `stage_by_stage` raises `NotImplementedError`.
`weight_solver_settings.reattempt_failures` must currently be `false`.

## Observational file formats

Some familiar data layouts remain useful during migration:

- Bayesian line-of-sight velocity distribution (LOSVD) ECSV files use
  `bin_flux` and paired `losvd_N`/`dlosvd_N` columns. Rename the DYNAMITE
  `binID_dynamite` column to TNT's `bin_id`; TNT additionally requires the
  documented velocity metadata.
- Proper-motion NPZ files use `PM_2dhist`, `PM_2dhist_sigma`,
  `nstarbin`, `vxrange`, and `vyrange`. Rename the `binID_dynamite` array to
  `bin_id`; TNT additionally requires a scalar `velocity_unit`.

Rename `vbin_id` to `bin_id` in Gauss-Hermite and population inputs as well.
Every TNT observational file must cover all positive IDs in its referenced
spatial binning exactly once; see
[Configuration preparation](configuration.md).

## Migration checklist

1. Start from TNT's packaged default configuration rather than editing a
   DYNAMITE configuration in place.
2. Split MGE, binning, potential, kinematics, and population definitions into
   their TNT registries.
3. Move observational policies from global solver settings into each named
   kinematics data set.
4. Translate model-search limits and thresholds to the TNT structures above.
5. Remove settings without a TNT equivalent.
6. Run configuration preparation and resolve every unknown-field error before
   constructing runtime objects.

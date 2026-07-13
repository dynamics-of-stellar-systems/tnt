# Core Team Meeting Topics

This document collects open design questions for discussion and decision by
the TNT core team.

## Ownership of kinematics systematic-error settings

### Background

The current default configuration places `GH_sys_err` and
`PM_sys_err_factor` under `weight_solver_settings`. These settings affect the
weight-solving objective, but they describe how TNT interprets or adjusts the
uncertainties of Gaussian-Hermite and proper-motion observations.

### Proposed direction

Move these settings conceptually into the corresponding kinematics
configuration. The kinematics layer would resolve the observations and their
uncertainties, and the weight solver would consume those resolved values
without deciding how the uncertainties were constructed.

A possible structure is:

```yaml
kinematics_type_defaults:
  gauss_hermite:
    observational_errors:
      systematic: [0.0, 0.0, 0.0, 0.0]

  proper_motions:
    observational_errors:
      variance_scale: 1.0
```

Each dynamically named kinematics data set could override its type default.
This would allow data sets from different instruments to use different
systematic-error assumptions.

Genuinely solver-specific choices, such as the number of Gaussian-Hermite
coefficients fitted by the solver, would remain under
`weight_solver_settings`.

### Questions for the team

1. Are systematic errors properties of each observational data set?
2. Could two Gaussian-Hermite data sets require different systematic errors?
3. Should changing the weight-solver implementation ever change these errors?

If the expected answers are *yes*, *yes*, and *no*, moving the settings under
kinematics is the cleaner design.

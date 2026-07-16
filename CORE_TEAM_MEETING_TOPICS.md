# Core Team Meeting Topics

This document collects design questions and resulting decisions from the TNT
core team.

## Resolved: ownership of kinematics observational settings

### Decision

Systematic uncertainties, proper-motion variance scaling, and the maximum
Gauss-Hermite order can differ between observational data sets. They therefore
belong to each dynamically named kinematics set rather than to global weight
solver settings. Type-specific defaults provide neutral values, and each data
set may override them independently.

The adopted structure is:

```yaml
kinematics_type_defaults:
  gauss_hermite:
    maximum_gh_order: 4
    observational_errors:
      systematic_uncertainties:
        v: 0.0
        sigma: 0.0
        h3: 0.0
        h4: 0.0

  proper_motions:
    observational_errors:
      variance_scale: 1.0
```

`maximum_gh_order` replaces the former `number_GH` name. Named systematic
uncertainties replace the positional space-separated `GH_sys_err` string.
`observational_errors.variance_scale` replaces `PM_sys_err_factor` and still
multiplies proper-motion error variances, so uncertainties scale with its
square root. The former global keys are not accepted under
`weight_solver_settings`.

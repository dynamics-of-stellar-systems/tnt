# Configuration preparation

TNT handles configuration in two separate phases. The preparation phase reads
and resolves configuration data. A later execution phase will construct the
physical system and other runtime objects from that resolved data.

## Preparing a configuration

```python
import tnt

config = tnt.Configuration()
config.read("user_config.yaml")
```

`Configuration.read()` performs the following operations:

1. Loads the packaged `default_config.yaml` profile.
2. Recursively merges the user profile over the packaged profile.
3. Applies common defaults to every dynamically named component, parameter,
   and kinematics data set.
4. Applies defaults selected by each kinematics data set's `type`.
5. Writes the fully resolved configuration to
   `<output_directory>/config_repository/resolved_config.yaml`.

Mapping values are merged recursively. A user value replaces a default scalar
or list. User values always take precedence over applicable defaults.

The schema-only `dynamic_object_defaults` and `kinematics_type_defaults`
sections are applied during preparation and omitted from the resolved file.
Consequently, the generated YAML is a self-contained runtime configuration.

If a kinematics data set explicitly supplies complete histogram metadata
(`width`, `center`, and `bins`), that metadata replaces the type's histogram
derivation policy.

## Paths and side effects

The user profile must define non-empty `io_settings.input_directory` and
`io_settings.output_directory` strings. Relative paths are interpreted from
the process working directory and stored as absolute paths in the resolved
configuration, so the execution phase does not depend on a later working
directory.

Configuration preparation creates the output directory and its
`config_repository` subdirectory when necessary. It atomically replaces
`resolved_config.yaml` on each successful read. It does not instantiate
components, load observational data, or execute modelling code.

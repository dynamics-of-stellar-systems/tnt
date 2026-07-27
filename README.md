# TNT

TNT is software for orbit-superposition modelling.

## Development environment

Install the project and its development dependencies with:

```shell
uv sync --group dev --group docs
```

Dependency markers select a compatibility stack based on the host
architecture. Intel macOS uses the last mutually compatible JAX 0.4, unxt
1.1, and related package releases for which Intel wheels are available. Apple
Silicon and other supported platforms continue to use the modern dependency
stack.

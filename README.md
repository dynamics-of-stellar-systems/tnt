# TNT

TNT is software for orbit-superposition modelling.

**TNT is under development.**

## Development environment

### Native environment

Install the project and its development dependencies with:

```shell
uv sync --group dev --group docs
```

The native scientific stack supports Linux and Apple Silicon macOS. Intel
macOS is not supported because current JAX releases do not provide `jaxlib`
wheels for that platform; use the Linux container below instead.

### Linux container

Intel macOS developers should use the Linux development container instead of a
native TNT environment. Docker runs TNT as Linux `x86_64`, matching the primary
cluster architecture, while the repository remains on the host and is
immediately editable by an editor or Codex.

Codex continues to work with the host checkout and invokes the commands below;
it does not need to be installed inside the container.

Build the image after cloning the repository or changing dependencies:

```shell
docker compose build
```

Run the tests, lint checks, or documentation build:

```shell
docker compose run --rm dev pytest -q
docker compose run --rm dev ruff check .
docker compose run --rm dev sphinx-build -E -b html -W docs/source docs/build/html
```

Open an interactive Linux shell with the TNT environment active:

```shell
docker compose run --rm dev
```

Source edits do not require rebuilding the image. Rebuild it when
`pyproject.toml`, `uv.lock`, `.python-version`, or the Dockerfile changes.

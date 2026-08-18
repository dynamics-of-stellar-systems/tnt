# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.11.26@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5 AS uv
FROM python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

COPY --from=uv /uv /uvx /bin/

ENV PATH="/opt/tnt-venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/tnt-venv

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Cache third-party dependencies separately from the changing source tree.
COPY pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --frozen --group dev --group docs --no-install-project

COPY tnt ./tnt
RUN uv sync --frozen --group dev --group docs

CMD ["/bin/bash"]

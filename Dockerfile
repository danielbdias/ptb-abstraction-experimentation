ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.4.16
ARG DEBIAN_VERSION=bookworm

FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-${DEBIAN_VERSION}

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
COPY pyRDDLGym-jax /app/pyRDDLGym-jax

RUN uv sync
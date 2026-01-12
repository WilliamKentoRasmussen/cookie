#FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base
# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

#RUN uv sync --frozen --no-install-project

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#COPY src src/
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY data/ data/

#RUN uv sync --frozen
WORKDIR /

#Mount my local uv cache to the docker image
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync
#RUN uv sync --locked --no-cache --no-install-project



#ENTRYPOINT ["uv", "run", "src/my_project/train.py"]
ENTRYPOINT ["uv", "run", "src/my_project/train.py"]

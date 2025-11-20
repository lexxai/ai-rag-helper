# docker compose build --build-arg REPO_BUILDER=rocm/pytorch:latest --build-arg EXTRA=docker
# docker compose build --build-arg EXTRA=cu128

ARG PYTHON_VERSION=3.14
ARG UV_PYTHON=${UV_PYTHON:-}

ARG PYTHON_BUILDER=python:${PYTHON_VERSION}
ARG PYTHON_BUILDER_SLIM=python:${PYTHON_VERSION}-slim

ARG PYTHON_REPO=${PYTHON_APP:-$PYTHON_BUILDER_SLIM}

ARG REPO_BUILD=${REPO_BUILDER:-$PYTHON_BUILDER}
ARG REPO=${PYTHON_REPO:-$REPO_BUILD}

ARG EXTRA=${EXTRA:-cpu}

FROM ${REPO_BUILD} AS builder

ARG UV_PYTHON
ARG EXTRA

# Install uv
# RUN pip install uv
# Use an official image to get the uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set work directory
WORKDIR /app

# Install Rust and build dependencies only if UV_PYTHON is set
RUN if [ -n "${UV_PYTHON}" ]; then \
        apt-get update -y && \
        apt-get install -y curl build-essential && \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Add Rust to PATH (harmless if not installed)
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies
COPY "pyproject.toml" "uv.lock" .
ENV UV_PYTHON=${UV_PYTHON}
#--locked
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${EXTRA}" = "docker" ]; then \
        uv sync --locked  --no-group dev --no-group default_extras --extra monitoring --no-install-package torch --no-install-package pytorch-triton-rocm; \
    else \
        uv sync --locked --no-group dev --extra monitoring --extra ${EXTRA}; \
    fi

FROM ${REPO}

ARG _USER=appuser
ARG _GROUP=appgroup
ARG _MEDIA_DIR=/app/staticfiles

# Set work directory
WORKDIR /app

# Copy uv binary (needed for proper venv operation)
COPY --from=builder /bin/uv /bin/uvx /bin/

# Copy virtual environment from builder
# IMPORTANT: ${REPO} must be compatible with ${REPO_BUILD} (same base OS/Python version)
# Otherwise compiled binaries won't work due to missing shared libraries
COPY --from=builder /app/.venv /app/.venv

# Verify Python and venv work before proceeding
# RUN /app/.venv/bin/python --version && \
#     /app/.venv/bin/python -c "import sys; print(f'Python: {sys.executable}')"

RUN groupadd ${_GROUP} && useradd --no-log-init -r --no-create-home -G ${_GROUP} ${_USER} && \
    mkdir -p ${_MEDIA_DIR} && chown -R ${_USER}:${_GROUP} ${_MEDIA_DIR}

# Copy project
COPY --chmod=+x ./dockers/*.sh .
COPY --chown=${_GROUP}:${_USER} ./src ./src

# Set environment variables
# VIRTUAL_ENV tells Python where the venv is, allowing it to find site-packages automatically
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin/:$PATH \
    VIRTUAL_ENV=/app/.venv \
    PYTHONPATH="/app/src:${PYTHONPATH:-}"

USER ${_USER}

ENTRYPOINT ["bash", "-c", "/app/entrypoint.sh"]


ARG PYTHON_VER=3.14
ARG PYTHON_BUILDER=python:${PYTHON_VER}
ARG PYTHON_APP=${PYTHON_BUILDER}-slim
ARG REPO_BUILDER=${REPO-$PYTHON_BUILDER}
ARG REPO=${REPO_BUILDER-$PYTHON_APP}
ARG EXTRA=${EXTRA-cpu}

FROM ${REPO_BUILDER} AS builder

ARG PYTHON_VER
ARG REPO
ARG EXTRA

# Install uv
# RUN pip install uv
# Use an official image to get the uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set work directory
WORKDIR /app

# Install dependencies
COPY "pyproject.toml" "uv.lock" .
#--locked
RUN uv sync --locked --no-group dev  --all-groups --extra ${EXTRA-cpu}

FROM ${REPO}

ARG _USER=appuser
ARG _GROUP=appgroup
ARG _MEDIA_DIR=/app/staticfiles

# Set work directory
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

RUN groupadd ${_GROUP} && useradd --no-log-init -r --no-create-home -G ${_GROUP} ${_USER} && \
    mkdir -p ${_MEDIA_DIR} && chown -R ${_USER}:${_GROUP} ${_MEDIA_DIR}

# Copy project
COPY --chmod=+x ./dockers/*.sh .
COPY --chown=${_GROUP}:${_USER} ./src ./src

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/app/.venv/bin/:$PATH

USER ${_USER}

ENTRYPOINT ["/app/entrypoint.sh"]


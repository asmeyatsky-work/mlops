# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS build

WORKDIR /app
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
COPY mlops_orchestrator/ mlops_orchestrator/

RUN pip install --no-cache-dir .

FROM python:3.12-slim

# Non-root user for runtime (UID/GID 10001 — outside common host ranges).
RUN groupadd --system --gid 10001 mlops \
    && useradd --system --uid 10001 --gid mlops --no-create-home --shell /usr/sbin/nologin mlops

WORKDIR /app
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/mlops-orchestrator /usr/local/bin/mlops-orchestrator

ENV MLOPS_USE_STUBS=false \
    MLOPS_TRANSPORT=sse \
    MLOPS_ENVIRONMENT=production \
    PYTHONUNBUFFERED=1

USER mlops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(3); s.connect(('127.0.0.1',8000))" || exit 1

ENTRYPOINT ["mlops-orchestrator"]

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY mlops_orchestrator/ mlops_orchestrator/

RUN pip install --no-cache-dir .

ENV MLOPS_USE_STUBS=false
ENV MLOPS_TRANSPORT=sse

EXPOSE 8000

ENTRYPOINT ["mlops-orchestrator"]

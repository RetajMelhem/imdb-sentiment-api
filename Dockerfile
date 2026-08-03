# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.13.0+cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --system app && \
    useradd --system --gid app --create-home --home-dir /home/app app

COPY pyproject.toml README.md model_metadata.json ./
COPY src ./src
COPY app ./app

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" && \
    python -m pip install . && \
    mkdir -p "${HF_HOME}" && \
    chown -R root:root /app && \
    chmod -R a-w /app && \
    chown -R app:app /home/app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

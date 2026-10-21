# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS model-artifacts

WORKDIR /download
COPY model/manifest.json ./model/manifest.json
COPY scripts/download_model.py ./scripts/download_model.py
RUN python scripts/download_model.py --output /opt/model && \
    chmod -R a-w /opt/model

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS runtime

LABEL org.opencontainers.image.source="https://github.com/RetajMelhem/imdb-sentiment-api" \
    org.opencontainers.image.version="1.1.0"

ARG TORCH_INDEX_URL=https://pypi.org/simple
ARG TORCH_VERSION=2.13.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_PATH=/opt/model \
    DEVICE=auto \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --system app && \
    useradd --system --gid app --create-home --home-dir /home/app app

COPY requirements-docker.txt requirements-cuda.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip==26.2.2 && \
    python -m pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" -c requirements-docker.txt -c requirements-cuda.txt "torch==${TORCH_VERSION}" && \
    python -m pip install -r requirements-docker.txt

COPY pyproject.toml README.md model_metadata.json ./
COPY src ./src
COPY app ./app
COPY scripts/download_model.py ./scripts/download_model.py
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps . && \
    mkdir -p "${HF_HOME}" && \
    chown -R root:root /app && \
    chmod -R a-w /app && \
    chown -R app:app /home/app

COPY --from=model-artifacts /opt/model /opt/model

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

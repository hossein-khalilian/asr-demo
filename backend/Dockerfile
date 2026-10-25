# syntax=docker/dockerfile:1.7
#
# Backend image: FastAPI + NVIDIA NeMo.
#
# Stage map:
#   base  – slim Python + ffmpeg/libsndfile, venv on PATH, model cache paths
#   deps  – build tools + PyTorch + NeMo installed into /opt/venv
#   dev   – deps' venv + test tools; source is bind-mounted for hot reload
#   prod  – deps' venv + code, non-root, no build tools
#
# PyTorch comes from TORCH_INDEX_URL. The default CPU wheels keep the image
# small and run anywhere; docker-compose.gpu.yml swaps in the CUDA wheels
# (which bundle the CUDA runtime, so no CUDA base image is needed).
#
#   docker build --target prod -t asr-backend .
#   docker build --target prod --build-arg GPU=1 \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 -t asr-backend:gpu .

ARG REGISTRY=docker.arvancloud.ir/

# ---- base -----------------------------------------------------------------------
FROM ${REGISTRY}python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # Checkpoints download here; compose mounts a volume so they survive rebuilds.
    HF_HOME=/cache/huggingface \
    NEMO_CACHE_DIR=/cache/nemo \
    # The Xet transfer backend stalls on restricted networks.
    HF_HUB_DISABLE_XET=1
WORKDIR /app
# ffmpeg: decodes uploads (webm/opus, mp3, …) to 16 kHz WAV. libsndfile1: NeMo audio IO.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1

# ---- deps -----------------------------------------------------------------------
FROM base AS deps
# Some NeMo dependencies compile C/C++ extensions when no wheel matches.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends build-essential
# PyPI mirror, e.g. https://package-mirror.liara.ir/repository/pypi/simple
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG GPU=0
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN python -m venv /opt/venv
# PyTorch first, in its own layer, so editing requirements does not refetch it
# and NeMo reuses this build instead of pulling a different one.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --index-url ${TORCH_INDEX_URL} torch
COPY requirements.txt requirements-asr.txt requirements-gpu.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -r requirements-asr.txt \
    && if [ "${GPU}" = "1" ]; then pip install -r requirements-gpu.txt; fi

# ---- dev ------------------------------------------------------------------------
FROM base AS dev
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} appuser \
    && useradd --uid ${UID} --gid ${GID} --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /cache && chown appuser:appuser /app /cache
COPY --from=deps /opt/venv /opt/venv
COPY requirements.txt requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -r requirements-dev.txt
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/app"]

# ---- prod -----------------------------------------------------------------------
FROM base AS prod
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /cache /app/models && chown appuser:appuser /cache /app/models
COPY --from=deps /opt/venv /opt/venv
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser config ./config
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
# One worker: each process holds its own copy of the model and its own queue.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

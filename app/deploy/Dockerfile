# PertEMA backend - CPU-only reliability scoring service (N7 deployability).
#
# Scope: this image serves the synchronous scoring path only. That path is
# numpy plus a frozen xgboost estimator on CPU. There is no GPU, no CUDA, and no
# torch in this image. Build context is app/ (small) so the repo-root container
# images and datasets are never sent to the Docker daemon.
#
# Docker is not runnable on the research host, so this file is validated by
# review only. See app/deploy/OPERATIONS.md. The versions below are pinned to
# the tested pixi environment because reproducibility is a deliverable.

FROM python:3.11-slim

# Fail fast, no .pyc files, unbuffered logs, quiet pip.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Bound the CPU thread pools so numpy or xgboost do not oversubscribe a
# many-core host. Overridable at run time from the compose environment block.
ENV OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2

# xgboost needs libgomp (OpenMP) at run time. Nothing else is required for the
# CPU scoring path.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Web and scientific dependencies, pinned to the tested versions. pydantic is
# load bearing (the /score request model) so it is pinned explicitly next to
# fastapi rather than left to transitive resolution.
RUN pip install \
    fastapi==0.139.0 \
    uvicorn==0.51.0 \
    pydantic==2.13.4 \
    xgboost==3.2.0 \
    numpy==1.26.4 \
    scipy==1.17.1 \
    scikit-learn==1.8.0 \
    pandas==3.0.1

WORKDIR /srv

# Application package plus the frozen inference artifact. scoring.py lives under
# backend/, so copying backend covers the scoring core. The model is copied as
# app/model/pertema_model_v0.1.0, the exact path resolved by
# scoring.default_model(). To swap in another frozen model without rebuilding,
# mount over /srv/app/model (see docker-compose.yml and OPERATIONS.md).
COPY backend  /srv/app/backend
COPY frontend /srv/app/frontend
COPY examples /srv/app/examples
COPY model    /srv/app/model

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

# Liveness and readiness. /health returns ok only after the frozen model has
# loaded at import time, so this doubles as a readiness signal. It uses stdlib
# urllib, so no curl and no extra package are needed in the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# Default standalone command. docker-compose.yml overrides this to make the
# worker count and log level env-driven.
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

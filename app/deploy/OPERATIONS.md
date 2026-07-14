# PertEMA backend - deployment and operations

CPU-only reliability scoring service for single-cell perturbation predictors.
This directory holds a self-hostable Docker deployment of the backend in
`app/backend/`. The scoring path is numpy plus a frozen xgboost estimator, so
no GPU, no CUDA, and no torch are required.

## Validation status (read this first)

Docker is not runnable on the research host used to author this project. These
files are therefore provided and validated by review only. They have not been
built or started here. The invariants they encode were checked against the
backend source (`app/backend/main.py`, `app/backend/scoring.py`) and the frozen
model layout (`app/model/pertema_model_v0.1.0/`). Where a value could not be
verified it is called out plainly rather than guessed.

## What ships in the image

- `app/backend/` the FastAPI service and the framework-agnostic scoring core.
- `app/frontend/` the two static pages served at `/` and `/methods`.
- `app/examples/` the bundled in-sample example used by `/example`.
- `app/model/pertema_model_v0.1.0/` the frozen inference artifact, copied read
  only. The estimator is never retrained on user input, so the container has no
  write path back into the model.

The `/benchmark` route reads `results/benchmark/benchmark_reliability.csv`,
which is not baked into the image. That route returns 404 unless the results
directory is mounted (see Pointing at the frozen model below). Every other
route works from the image alone.

## Prerequisites

- Docker Engine with the Compose plugin (`docker compose`, v2).
- No GPU and no NVIDIA runtime are needed.
- Roughly 1 to 2 GB of disk for the built image (the Debian slim base plus the
  scientific wheels).

## One-command bring-up

Run from the repository root.

```
docker compose -f app/deploy/docker-compose.yml up
```

Add `-d` to run detached. The first invocation builds the image, which takes a
few minutes while the scientific wheels install. Subsequent runs reuse the
cached layers.

Stop and remove the stack with:

```
docker compose -f app/deploy/docker-compose.yml down
```

## Manual build and run (without Compose)

The build context must be `app/`, not the repo root, because the repo root
holds multi-GB container images and datasets that must not be sent to the
Docker daemon.

```
docker build -f app/deploy/Dockerfile -t pertema-backend:0.1.0 app/
docker run --rm -p 8000:8000 pertema-backend:0.1.0
```

## Endpoints and health checks

Once the service is up on the published port (default 8000):

- `GET /health` returns `{"status": "ok", "model_version": "0.1.0"}`. Because
  the frozen model loads at import time, a healthy response also means the model
  is ready to score.
- `GET /version` returns the full model provenance record.
- `POST /score` scores a feature matrix. The body is JSON `{"features": [[...]]}`
  with each row of length 64. Rows of a different width are rejected with 422.
- `GET /example` scores the bundled example. Its scores are in-sample and
  optimistic by construction, and the response says so.
- `GET /` and `GET /methods` serve the two static pages.

Quick manual check from the host:

```
curl -s http://localhost:8000/health
```

The container-level healthcheck (declared in both the Dockerfile and the
compose service) probes `/health` with stdlib urllib every 30 seconds after a
30 second start period. Inspect it with:

```
docker inspect --format '{{json .State.Health}}' pertema-backend
```

## Environment variables

All variables are optional and have safe defaults. Set them in the shell before
`docker compose up`, or in a `.env` file next to the compose file.

| Variable              | Default | Effect                                                    |
|-----------------------|---------|-----------------------------------------------------------|
| HOST_PORT             | 8000    | Host port that maps to the fixed in-container port 8000.  |
| UVICORN_WORKERS       | 1       | Number of uvicorn worker processes.                       |
| LOG_LEVEL             | info    | uvicorn log level (debug, info, warning, error).          |
| OMP_NUM_THREADS       | 2       | OpenMP thread cap for xgboost and numpy.                  |
| OPENBLAS_NUM_THREADS  | 2       | OpenBLAS thread cap.                                       |
| MKL_NUM_THREADS       | 2       | MKL thread cap.                                            |
| NUMEXPR_NUM_THREADS   | 2       | NumExpr thread cap.                                        |

The thread caps matter. The host this project runs on has 128 cores, and an
unbounded BLAS or OpenMP pool oversubscribes threads and slows the CPU scoring
path. The defaults keep each worker modest. Raise them if you give the
container more cores.

The in-container port is deliberately fixed at 8000 so that EXPOSE, the
healthcheck, and the uvicorn command all agree. Remap only the host side with
HOST_PORT.

## Pointing at the frozen model

By default the image bakes in `app/model/pertema_model_v0.1.0/` and the code
resolves that exact path through `scoring.default_model()`. Nothing further is
needed to serve version 0.1.0.

There is no environment variable in the shipped code for the model path, so
this deployment does not invent one. To serve a different frozen model without
rebuilding the image, mount your model directory over the baked path. The
subdirectory name must stay `pertema_model_v0.1.0` because the code resolves
that literal path. Uncomment the volume in `docker-compose.yml`:

```
volumes:
  - ${MODEL_DIR:-./model}:/srv/app/model:ro
```

The mount is read only, which matches the no-retrain-on-user-truth invariant. A
mounted model directory must contain the same files as the baked one:
`estimator.json`, `calibration.npz`, `feature_spec.json`, and
`provenance.json`.

To enable the `/benchmark` route, also mount the results directory:

```
volumes:
  - ${RESULTS_DIR:-../results}:/srv/results:ro
```

## The fuller stack (Redis, worker, nginx)

`docker-compose.yml` carries commented stubs for a Redis queue, an async
worker, and an nginx reverse proxy with TLS. They are illustrative only. The
current backend serves the synchronous scoring path and needs none of them. The
worker module is not implemented in this image, so its stub command is left
inert on purpose. Adopt the fuller stack by merging those services into the
active services block and supplying the referenced nginx config and TLS files.

## Troubleshooting

- Build sends a huge context or is slow: confirm the build context is `app/`
  and not the repo root. The compose file already sets this.
- `import xgboost` fails on a missing `libgomp.so.1`: the Dockerfile installs
  `libgomp1` for this reason. Confirm that apt step succeeded during the build.
- `/health` never becomes healthy: check the logs with
  `docker compose -f app/deploy/docker-compose.yml logs backend`. A model load
  failure at import time surfaces there.
- `/benchmark` returns 404: expected unless the results directory is mounted.

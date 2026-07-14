#!/usr/bin/env bash
# Run a command inside the PRESCRIBE glibc-2.35 container (torch 2.4.1 cu121), GPU 0, MPS bypassed,
# with the bind-mounted PYTHONUSERBASE that holds PRESCRIBE's deps. Rootless, no sudo.
# Usage: baselines/run_in_container.sh python Step1_preprocess.py
set -euo pipefail
ROOT=.
mkdir -p /tmp/pertema_nomps
exec singularity exec --nv \
  --env PYTHONUSERBASE="$ROOT/containers/pyuser" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env CUDA_MPS_PIPE_DIRECTORY=/tmp/pertema_nomps \
  --env OMP_NUM_THREADS=8 \
  --pwd "$ROOT/baselines/PRESCRIBE" \
  "$ROOT/containers/pytorch241.sif" "$@"

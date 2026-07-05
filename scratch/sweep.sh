#!/bin/bash
# Fast-iteration A/B sweep launcher. One sbatch job per variant, same seed, same net,
# ONE variable changed each. Winners compound into defaults; losers deleted (keep-only-
# winners rule). Usage:  bash scratch/sweep.sh <wave-tag> "<name>::<extra args>" ...
# e.g.  bash scratch/sweep.sh w3 "ctrl::" "cut8::--aug-cut 8" "jit6::--aug-jitter 0.6"
cd "$(dirname "$0")/.." || exit 1
WAVE="$1"; shift
# CAPS: sbuehrer may use at most 18 GPUs total across all sessions, and one sweep may
# submit at most 10 so parallel sessions keep headroom. budget = min(18 - in_use, 10).
GPU_CAP=18
SWEEP_CAP=10
in_use=$(squeue -u "$USER" -h -o "%b" 2>/dev/null | grep -c gpu)
budget=$(( GPU_CAP - in_use ))
(( budget > SWEEP_CAP )) && budget=$SWEEP_CAP
if (( budget <= 0 )); then
  echo "GPU cap reached: $in_use/$GPU_CAP GPUs already in use. Cancel a job first." >&2
  squeue -u "$USER" -o "%.10i %.12j %.8T %R"; exit 1
fi
if (( $# > budget )); then
  echo "WARNING: $# variants requested but only $budget GPU slot(s) in budget ($in_use/$GPU_CAP used, per-sweep cap $SWEEP_CAP)." >&2
  echo "Submitting the first $budget; run the rest in a later sweep." >&2
  set -- "${@:1:budget}"
fi
# Shared config = wave-2 winners (scratch/base.env; also used by backfill.sh).
source scratch/base.env
for spec in "$@"; do
  name="${spec%%::*}"; extra="${spec#*::}"
  tag="${WAVE}_${name}"
  sbatch --partition=disco.med --gres=gpu:1 --cpus-per-task=4 --mem=32G \
    --time=04:00:00 --job-name="$tag" --output="scratch/${tag}.out" \
    --wrap="bash scratch/cd.sh --ckpt scratch/${tag}.pt $BASE $extra"
done
squeue -u "$USER" -o "%.10i %.14j %.8T %.10M %R"

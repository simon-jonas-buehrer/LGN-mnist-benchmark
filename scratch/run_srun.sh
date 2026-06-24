#!/bin/bash
# Interactive single-GPU run: grabs one GPU and runs in the foreground so you watch the
# train/val/test accuracy table live. Ctrl-C kills it.
#
#   bash scratch/run_srun.sh                       # defaults (full 50k)
#   bash scratch/run_srun.sh --train-size 5000     # quick look on a subset
#
cd "$(dirname "$0")/.." || exit 1
srun --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=1:00:00 --pty \
  bash -c "export PYTHONUNBUFFERED=1; .venv/bin/python -u scratch/grow_lut.py --device cuda --download $*"

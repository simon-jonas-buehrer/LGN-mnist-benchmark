#!/bin/bash
# Batch single-GPU run of the grown LUT network. Output streams to scratch/grow.log.
#
#   sbatch scratch/run.sbatch                      # defaults (full 50k)
#   sbatch scratch/run.sbatch --train-size 5000    # extra args pass through to grow_lut.py
#
#SBATCH --job-name=grow-lut
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=scratch/grow.log

cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONUNBUFFERED=1
.venv/bin/python -u scratch/grow_lut.py --device cuda --download "$@"

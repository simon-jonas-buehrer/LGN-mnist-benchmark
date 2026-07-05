#!/bin/bash
# Run on a cluster GPU:
#   sbatch --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=48:00:00 --job-name=cd --output=scratch/cd.out --wrap="bash scratch/cd.sh"
# Extra args pass through, e.g.:  ... --wrap="bash scratch/cd.sh --channels 10240 --resume"
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -u scratch/cd.py --device cuda --ckpt scratch/cd.pt "$@"

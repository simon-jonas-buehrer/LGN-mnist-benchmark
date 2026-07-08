#!/bin/bash
# Double-descent sweep, FAST edition: signal-major fused Triton NAND kernel (4-9x forward speedup)
# + fixed GA (margin fitness, mut-count 1). One array task per hidden width, log-spaced so total
# gates (=16*W+100) span ~1e2 .. 1e8, DEPTH 16. FIXED batch 512 for every width (fair) with
# --low-mem so even the widest net fits an 11GB card -> runs on ANY GPU (gpu:1), all 16 busy.
# Long runtime + resume-by-default so each width actually converges. Plot with dd_plot.py.
#
#   mkdir -p .local/logs scratch_genetic/runs && sbatch scratch_genetic/dd_sweep.sbatch
#
#SBATCH --job-name=nand_dd
#SBATCH --array=0-15%16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --output=.local/logs/%x_%a.out

cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/itet-stor/sbuehrer/net_scratch/.triton_cache
PY=/itet-stor/sbuehrer/net_scratch/lut-tutorial/.venv-gpu/bin/python   # persistent (survives session)

# 16 hidden widths (depth 16, thin output R=100 -> total gates = 16*W + 100), ~1e2 .. 1e8 gates
WIDTHS=(1 10 30 100 300 1000 3000 10000 30000 100000 300000 1000000 2000000 3000000 4500000 6250000)
W=${WIDTHS[$SLURM_ARRAY_TASK_ID]}
GATES=$((16 * W + 100))

$PY scratch_genetic/nand_ga.py --device cuda --depth 16 --width "$W" --out-width 100 --low-mem \
  --k 16 --mut-count 1 --delta-bias 3 --batch 512 --gens 1000000000 --max-minutes 2760 \
  --eval-every 500 --out "scratch_genetic/runs/dd_g${GATES}"

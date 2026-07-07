#!/bin/bash
# Optimizer benchmark task array: {bp,cd,mab,rs} x {conn fixed, learnable} x 2 seeds = 16
# runs on the identical fixed LUT net (shared DEPTH/WIDTH/fan-in, monarch wiring, full
# 50k CIFAR-10). One GPU each (any type/node), all 16 in parallel; every run trains the
# full 2-day walltime cap, then finishes up (final eval incl. test + checkpoint).
# Plot progress any time (jsonl written incrementally): .venv/bin/python scratch/plot.py
#
#   mkdir -p .local/logs scratch/runs && sbatch .local/optbench.sbatch
#
#SBATCH --job-name=optbench
#SBATCH --array=0-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=22G
#SBATCH --time=2-00:00:00
#SBATCH --output=.local/logs/%x_%a.out

cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEPTH=2          # shared across ALL methods (fair)
WIDTH=64000
MINUTES=2865     # ~2 days walltime (gpu.normal cap 2-00:00:00), margin for final eval

METHODS=(bp cd rs mab)
cfg=$((SLURM_ARRAY_TASK_ID % 8))         # 0-7: method x connection mode
seed=$((SLURM_ARRAY_TASK_ID / 8))        # 0, 1
method=${METHODS[$((cfg % 4))]}
lc=$((cfg / 4))                          # 0 = frozen monarch wiring, 1 = learnable
tag="${method}_conn${lc}_s${seed}"

# per-method hyperparameters tuned so each clears 30% val at the shared depth
case "$method" in
  bp)  extra="" ;;
  cd)  extra="--chunk 512" ;;
  rs)  extra="--res-init 0 --rs-tt 512" ;;
  mab) extra="--res-init 0 --mab-rollouts 8 --lr-mab 4" ;;
esac

# Pause & continue is automatic: `scancel` the array, then just resubmit -- each run picks
# up from its saved scratch/runs/<tag>.pt weights and appends to its .jsonl. To start over,
# delete scratch/runs/*.pt first.
.venv/bin/python -u scratch/opt.py --method "$method" --learn-conn "$lc" --seed "$seed" \
  --depth "$DEPTH" --width "$WIDTH" --max-minutes "$MINUTES" --max-samples 1e10 \
  --out "scratch/runs/${tag}" $extra "$@"

#!/bin/bash
# THE SCALING SWEEP: 379 runs = gate budget (1e2..1e8) x depth (1..32) x mutation size (1..128).
# Every run is the same plain NAND net and the same search (pick gates uniformly from the whole
# pool, rewire connections, keep the best of k); only the SHAPE and the STEP SIZE change.
#
#   %1 => exactly ONE task on a GPU at any time: the whole sweep uses a single GPU, serially.
#   Every run checkpoints (.pkl every --ckpt-every gens AND every eval) and RESUMES BY DEFAULT, so
#   a preempted or requeued task picks up where it left off instead of starting over.
#   Every run stops on CONVERGENCE (no val gain for --patience evals) or its per-budget time cap,
#   whichever comes first. A run that already CONVERGED exits immediately if resubmitted; a run that
#   hit the TIME CAP resumes and keeps searching -- so resubmitting the array extends the sweep.
#
#   sbatch scratch_genetic/scale.sbatch                    # the whole sweep, cheap budgets first
#   sbatch --array=300-378%1 scratch_genetic/scale.sbatch  # only the big budgets
#   python scratch_genetic/scale.py --plot                 # curves + table, any time
#
# Budget: 22.2 h worst case (every run hits its cap); early stopping returns the rest. Under 24 h.
#
#SBATCH --job-name=nand_scale
#SBATCH --array=0-378%1
#SBATCH --nodes=1
#SBATCH --gres=gpu:geforce_rtx_3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=.local/logs/%x_%a.out

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p .local/logs scratch_genetic/runs/scale
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/itet-stor/sbuehrer/net_scratch/.triton_cache
PY=/itet-stor/sbuehrer/net_scratch/conda_envs/myenv/bin/python

# scale.py owns the grid: width/depth/mut-count/time-cap/eval cadence for this array index.
FLAGS=$("$PY" scratch_genetic/scale.py --task "$SLURM_ARRAY_TASK_ID")
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $FLAGS"

# shellcheck disable=SC2086  # FLAGS is a deliberate word-split argument list
"$PY" scratch_genetic/nand_ga.py --device cuda --patience 20 $FLAGS

#!/bin/bash
# DATA-PARALLEL launcher: train ONE big NAND net across N GPUs on a single node via torchrun. Each
# rank holds an identical genome (int32) and proposes identical mutations; each scores a different
# data shard; only fitness scalars are all-reduced -> effective batch = per-rank batch * N with
# negligible comms. Combined with --low-mem this trains a 1e9-gate net on 24GB cards.
#
#   sbatch scratch_genetic/dp_run.sbatch           # defaults to a 1e9-gate net on 4x rtx_3090
#
#SBATCH --job-name=nand_dp
#SBATCH --nodes=1
#SBATCH --gres=gpu:geforce_rtx_3090:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=.local/logs/%x.out

cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/itet-stor/sbuehrer/net_scratch/.triton_cache
PY=/itet-stor/sbuehrer/net_scratch/lut-tutorial/.venv-gpu
NPROC=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | grep -c .)   # GPUs Slurm gave this task

# 1e9 gates = depth 16 * width 62.5M + 100. --low-mem + int32 genome fit each rank in ~18GB (24GB).
"$PY/bin/torchrun" --standalone --nproc_per_node="$NPROC" scratch_genetic/nand_ga.py \
  --device cuda --width 62500000 --depth 16 --out-width 100 --low-mem \
  --k 16 --mut-count 256 --delta-bias 3 --batch 64 --gens 1000000000 --max-minutes 1380 \
  --eval-every 500 --out scratch_genetic/runs/dp_1e9 "$@"

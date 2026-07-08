#!/bin/bash
# Committed LONG run: depth 16 + locality 2 + UNIFORM mutation (delta-bias 0, every layer incl. the
# early low-SNR ones) + not-prob 0.25. Trained for days (resume-by-default). Task 0 = CE fitness,
# task 1 = bits fitness. The bet: forcing signals through the depth + mutating all layers lets the
# deep net keep climbing given enough time.
#SBATCH --job-name=nand_long
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --output=.local/logs/%x_%a.out
cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/itet-stor/sbuehrer/net_scratch/.triton_cache
PY=/itet-stor/sbuehrer/net_scratch/lut-tutorial/.venv-gpu/bin/python
FIT=(ce bits)
$PY scratch_genetic/nand_ga.py --device cuda --depth 16 --width 65536 --out-width 2560 \
  --k 16 --mut-count 1 --not-prob 0.25 --locality 2 --delta-bias 0 --fitness "${FIT[$SLURM_ARRAY_TASK_ID]}" \
  --batch 8192 --low-mem --gens 1000000000 --max-minutes 2760 --eval-every 1000 \
  --out "scratch_genetic/runs/long_${FIT[$SLURM_ARRAY_TASK_ID]}"

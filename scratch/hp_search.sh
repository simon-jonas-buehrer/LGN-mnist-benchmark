#!/bin/bash
# Coarse HP search: build batch size x gates-per-phase x flips-per-phase, rebuild regime.
export PYTHONUNBUFFERED=1
CONFIGS=(
  "64 2000 0.0"   "64 2000 0.25"   "64 16000 0.0"   "64 16000 0.25"
  "512 2000 0.0"  "512 2000 0.25"  "512 16000 0.0"  "512 16000 0.25"
  "4096 2000 0.0" "4096 2000 0.25" "4096 16000 0.0" "4096 16000 0.25"
)
for c in "${CONFIGS[@]}"; do
  set -- $c; BB=$1; BPP=$2; CF=$3
  echo "############ build-batch=$BB  build-per-phase=$BPP  cd-fraction=$CF ############"
  .venv/bin/python -u scratch/grow_lut.py --device cuda --train-size 0 --num-bits 5 \
    --window-factor 4 --max-gates 150000 --rebuild --build-per-phase $BPP --build-batch $BB \
    --cd-flips 256 --cd-batch 8192 --cd-fraction $CF --eval-every 6 --max-feats 16384 \
    2>&1 | grep -E "BEST|FINAL|depth:"
done

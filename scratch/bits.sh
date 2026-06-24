#!/bin/bash
export PYTHONUNBUFFERED=1
for B in 5 10 20; do
  echo "############ num-bits=$B  (N=3072*B) ############"
  .venv/bin/python -u scratch/grow_lut.py --device cuda --train-size 0 --num-bits $B \
    --window-factor 4 --max-gates 300000 --build-per-phase 4000 --build-batch 512 \
    --cd-fraction 0.5 --cd-flips 1024 --cd-batch 8192 --depth-penalty 2.0 --eval-every 20 \
    2>&1 | grep -E "N=C|^   p[0-9]|BEST|depth\("
done

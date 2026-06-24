#!/bin/bash
export PYTHONUNBUFFERED=1
for G in and or; do
  echo "############ gate=$G ############"
  .venv/bin/python -u scratch/grow_lut.py --device cuda --train-size 0 --num-bits 5 \
    --window-factor 4 --max-gates 150000 --rebuild --build-per-phase 8000 --build-batch 64 \
    --cd-flips 256 --cd-batch 8192 --cd-fraction 0.25 --gate $G --eval-every 6 --max-feats 16384 \
    2>&1 | grep -E "BEST|FINAL|depth:"
done

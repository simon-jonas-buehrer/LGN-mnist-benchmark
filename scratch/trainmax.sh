#!/bin/bash
export PYTHONUNBUFFERED=1
for CF in 1.0 2.0 4.0; do
  echo "############ f=8 cd-fraction=$CF (train-max) ############"
  .venv/bin/python -u scratch/grow_lut.py --device cuda --train-size 0 --num-bits 5 \
    --window-factor 8 --max-gates 400000 --build-per-phase 2000 --build-batch 64 \
    --cd-fraction $CF --cd-flips 4096 --cd-batch 8192 --depth-penalty 2.0 \
    --final-cd-flips 4000000 --eval-every 25 2>&1 | grep -E "^   p[0-9]|FINAL|depth\("
done

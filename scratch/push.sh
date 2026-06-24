#!/bin/bash
export PYTHONUNBUFFERED=1
.venv/bin/python -u scratch/grow_lut.py --device cuda --train-size 0 --num-bits 5 \
  --window-factor 8 --max-gates 1000000 --rebuild --build-per-phase 16000 --build-batch 64 \
  --cd-flips 512 --cd-batch 8192 --cd-fraction 0.25 --eval-every 4 --max-feats 16384

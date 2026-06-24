#!/bin/bash
export PYTHONUNBUFFERED=1
.venv/bin/python -u scratch/fit_train.py --device cuda --window-factor 32 --cd-batch 45000 \
  --cd-flips 2048 --target-train 95 --max-flips 400000000 --report-flips 4000000 --val-every 80000000

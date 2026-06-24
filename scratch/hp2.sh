#!/bin/bash
export PYTHONUNBUFFERED=1
run () {  # $1=label, rest=args
  local label="$1"; shift
  echo "############ $label ############"
  .venv/bin/python -u scratch/grow_lut.py --device cuda --train-size 0 --num-bits 5 \
    --max-gates 200000 --cd-flips 1024 --cd-batch 8192 --depth-penalty 2.0 --eval-every 20 \
    "$@" 2>&1 | grep -E "BEST|depth\("
}
# base: f=4, build-batch=512, build-per-phase=2000, cd-fraction=0.5, gate=and
BASE="--window-factor 4 --build-batch 512 --build-per-phase 2000 --cd-fraction 0.5 --gate and"
# A) window-factor f
for F in 2 4 8 16; do run "f=$F"            --window-factor $F --build-batch 512 --build-per-phase 2000 --cd-fraction 0.5 --gate and; done
# B) build batch
for BB in 64 256 1024 4096; do run "bb=$BB" --window-factor 4 --build-batch $BB --build-per-phase 2000 --cd-fraction 0.5 --gate and; done
# C) build-per-phase (build amount/phase)
for BPP in 1000 2000 8000; do run "bpp=$BPP" --window-factor 4 --build-batch 512 --build-per-phase $BPP --cd-fraction 0.5 --gate and; done
# D) cd-fraction (cd amount/phase)
for CF in 0.0 0.25 0.5 1.0; do run "cf=$CF" --window-factor 4 --build-batch 512 --build-per-phase 2000 --cd-fraction $CF --gate and; done
# E) gate
for G in and or; do run "gate=$G" --window-factor 4 --build-batch 512 --build-per-phase 2000 --cd-fraction 0.5 --gate $G; done

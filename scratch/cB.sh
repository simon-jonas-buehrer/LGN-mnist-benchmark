#!/bin/bash
export PYTHONUNBUFFERED=1
run(){ echo "############ $1 ############"; .venv/bin/python -u scratch/grow_lut.py --device cuda --train-size 0 --num-bits 5 --window-factor 8 --max-gates 150000 --build-start 1000 --build-end 1000 --cd-start 60000 --cd-end 60000 --cd-flips 1024 --cd-batch 8192 --eval-every 15 ${@:2} 2>&1 | grep -E "P\(bits|BEST|FINAL|OutOfMemory|Error"|tail -4; }
run "conj K64"  --gate-type conj --fan-in 64
run "conj K128" --gate-type conj --fan-in 128
run "lut K4 (ref)"  --gate-type lut --fan-in 4
run "lut K6 (ref)"  --gate-type lut --fan-in 6

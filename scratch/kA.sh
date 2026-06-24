#!/bin/bash
export PYTHONUNBUFFERED=1
run(){ echo "############ $1 ############"; .venv/bin/python -u scratch/grow_lut.py --device cuda --train-size 0 --num-bits 5 --window-factor 8 --max-gates 150000 --build-start 2000 --build-end 2000 --cd-start 60000 --cd-end 60000 --cd-flips 1024 --cd-batch 8192 --eval-every 10 ${@:2} 2>&1 | grep -E "K=|^   p[0-9]|FINAL|BEST|OutOfMemory|Error"|tail -8; }
run "K5 (2^K=32)"  --fan-in 5
run "K6 (2^K=64)"  --fan-in 6

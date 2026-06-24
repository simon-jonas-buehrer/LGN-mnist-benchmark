#!/bin/bash
export PYTHONUNBUFFERED=1
run(){ echo "############ $1 ############"; .venv/bin/python -u scratch/grow_lut.py --device cuda --train-size 0 --num-bits 5 --window-factor 8 --max-gates 150000 --build-start 2000 --build-end 2000 --cd-start 60000 --cd-end 60000 --cd-flips 2048 --cd-batch 8192 --eval-every 10 ${@:2} 2>&1 | grep -E "M\(coeffs\)|^   p[0-9]|FINAL|BEST|OutOfMemory|Error"|tail -6; }
run "K4 d2"  --fan-in 4 --degree 2
run "K32 d2" --fan-in 32 --degree 2
run "WIDTH f16 K4d4" --fan-in 4 --degree 4 --window-factor 16
run "WIDTH f32 K4d4" --fan-in 4 --degree 4 --window-factor 32

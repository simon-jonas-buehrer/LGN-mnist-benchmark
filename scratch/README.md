# scratch: a backprop-free LUT network grown by correlation + coordinate descent

An experiment that builds the same kind of boolean-gate circuit as the main tutorial, but with
**no gradients at all**. Everything is binary and bitpacked, so building and inference are a
handful of GPU matmuls and bit ops. See the long docstring in [`grow_lut.py`](grow_lut.py) for
the full description; the short version:

## The window

One fixed window of `WIN = f * N` signal slots (`f = 2` by default):

```
slot 0 .. N-1        the N thermometer input bits (frozen)
slot N .. WIN-1      start at 0, filled by gates as we build
```

Every slot is also a head output bit. The **GroupSum head reads the whole window**: a slot's
class is `slot % C` (round-robin, so the input bits spread evenly over the classes), each class
owns `H = WIN / C` slots, and a class score is the popcount of its slots over `tau = sqrt(H)`.
Because `H * C = WIN = f * N`, the `(B, N) -> (B, H*C)` map is just "fill the window". Pool,
output bits and depth wiring are all the same array.

The dimensions are made to line up exactly: `N = C_img * H_img * W_img * b = 3*32*32*b = 3072b`,
and `WIN = f*N` must divide by `C = 10`, i.e. `b*f` must be a multiple of 5. The defaults
`b = 5, f = 2` give `N = 15360`, `WIN = 30720`, `H = 3072` exactly.

## Building (no backprop)

Each build sweep, on a random batch:

1. **residual per slot** in `{-1, 0, +1}`: a class-`c` slot wants to fire more where class `c`
   should go up (multiclass-hinge subgradient) but the slot is 0, and fire less where class `c`
   should go down but the slot is 1. Every slot chases its own current mistakes.
2. **correlate** every filled slot against every empty slot's residual: a `(filled x empty)`
   `~ f*N^2` covariance matrix, one matmul (chunked over targets).
3. for the strongest empty slots, take the **top-2** correlating signals and wire an **AND/OR**
   gate with **NOTs** on the negatively correlated inputs (`--gate and|or`).

Build only ever **fills empty slots** (each is a fresh boosting vote); gates can read any filled
slot, including earlier gates, which gives depth. Filling many gates from one residual snapshot
overshoots, so each sweep commits a modest number and the residual is recomputed next sweep.

## Coordinate descent

Greedy wiring overfits the batch it saw, so we periodically run CD: on a batch, for a sample of
gate slots, score flipping each of the 4 truth-table bits and **commit the top-K flips by gain**
(`--cd-apply`). Because the head sums the window, flipping one slot's gate only changes that
slot's class score, so the loss delta is exact and computed in closed form for all candidate
gates and all 4 flips at once. CD explores all 16 two-input functions, not just the AND/OR it
started from. Committing only a few flips per pass keeps each step close to true coordinate
descent (committing thousands at once overshoots, since the gates share class scores).

## The ramp

Over the run we anneal from **lots of build, little CD** to **little build, lots of CD**
(`--build-start/--build-end`, `--cd-start/--cd-end`), then a final CD-only phase
(`--final-cd`). Once the window is full, building stops on its own and the schedule hands
everything to CD.

## Run

```bash
# interactive single GPU, watch the live train/val/test table
bash scratch/run_srun.sh

# batch single GPU, output to scratch/grow.log
sbatch scratch/run.sbatch

# direct
.venv/bin/python scratch/grow_lut.py --device cuda --train-size 0

# tiny/fast CPU sanity check (deliberately weak: 1 thermometer bit, small window)
.venv/bin/python scratch/grow_lut.py --device cpu --train-size 2000 --num-bits 1 \
    --window-factor 2 --rounds 60 --build-start 120 --build-end 20 \
    --cd-start 2 --cd-end 10 --final-cd 150 --cd-batch 1800
```

## Status

Validated on CPU smoke runs only (no GPU was available in the dev session). On the weak smoke
config above (1 thermometer bit, ~1800 images) it goes from 10% chance to ~21% test; the proper
GPU config (`b=5`, full 50k, larger window, more CD) is expected to do considerably better and
is the intended way to run it. Everything here is a scratch experiment, separate from the main
tutorial code.

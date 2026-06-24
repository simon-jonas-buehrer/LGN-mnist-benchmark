# scratch: a backprop-free LUT network grown by correlation + coordinate descent

An experiment that builds the same kind of boolean-gate circuit as the main tutorial
(`../model.py`), but with **no gradients at all**. There is no autograd, no optimizer, no
straight-through estimator. The network is grown by greedy correlation and then polished by
randomized coordinate descent. Everything is binary and bitpacked into `int64` words, so both
building and inference are a handful of GPU matmuls and bit ops.

Single file: [`grow_lut.py`](grow_lut.py). Slurm helpers: [`run.sbatch`](run.sbatch),
[`run_srun.sh`](run_srun.sh).

---

## 1. The data path

Identical front end to the tutorial:

```
image (3x32x32, uint8)
  -> Thermometer encoder      each pixel-channel -> b threshold bits (quantile thresholds)
  -> flatten + transpose      -> X : (N, D)   N input bits per image, D images
```

`N = C_img * H_img * W_img * b = 3 * 32 * 32 * b = 3072 b`. The thermometer thresholds are fit on
a sample of the training set only (no leakage). `X` is stored once as a packed bit matrix.

## 2. The window: one array that is pool, depth and head at once

The whole model state is a single fixed **window** of `WIN = f * N` signal slots:

```
slot 0 .. N-1        the N thermometer input bits          (frozen, never overwritten)
slot N .. WIN-1      gate outputs; start at 0, filled by build
```

A "signal" is one bit per image, packed into `ceil(D/64)` `int64` words. The window is
`(WIN, ceil(D/64))` `int64`.

Three roles collapse into this one array:

- **Pool.** Any filled slot can be an input to a new gate. Because gate outputs are written back
  into the window, later gates can read earlier gates: that is where circuit **depth** comes
  from.
- **Output bits / head.** The GroupSum head reads the **entire window**. A slot's class is
  `slot % C` (round-robin, so the `N` input bits are spread evenly over the `C` classes, not
  dumped onto the first few). Each class owns `H = WIN / C` slots. The class score is the
  popcount of its slots over `tau = sqrt(H)`:

  ```
  score[c] = (sum of window bits with class==c) / sqrt(H)
  prediction = argmax_c score[c]
  ```

  `sqrt(H)` is the same variance scaling as the tutorial's GroupSum: a class score is a sum of
  ~`H` near-Bernoulli bits, so dividing by `sqrt(H)` keeps the logit scale constant as the
  window grows. It is monotone, so it does not change the argmax.

Because the head reads the whole window, `H * C = WIN = f * N`, so the `(B, N) -> (B, H*C)` map
the idea started from is literally **"fill the window"**.

### Making the dimensions line up exactly

`WIN = f*N` must be divisible by `C = 10` for `H` to be integral. `N = 3072 b`, and
`3072 b f mod 10 = 2 b f mod 10`, so we need **`b * f` to be a multiple of 5**. The defaults
`b = 5, f = 2` give `N = 15360`, `WIN = 30720`, `H = 3072` exactly. (`run` prints `exact=True`
when it works out; a tiny safety trim rounds `WIN` down to a multiple of `C` otherwise.)

## 3. Supervision: a per-slot residual, no backprop

There is no loss to differentiate. Instead, on a batch we compute what each slot *should* be
doing, in `{-1, 0, +1}`, from the **multiclass-hinge subgradient**.

First, a per-class direction `d[c, i]` for image `i` (`class_direction`):

- `d = -1` for any **wrong** class whose logit is within margin 1 of the true class (a margin
  violator that should be pushed **down**);
- `d = +1` for the **true** class, unless it already wins by margin 1 (then `0`, push **up**);
- `0` otherwise.

Then a residual per **slot** (every slot is a class-`c` output bit):

```
r[slot, i] = +1   if d[class(slot), i] > 0  and the slot is currently 0   (it could help by firing)
           = -1   if d[class(slot), i] < 0  and the slot is currently 1   (it hurts by firing)
           =  0   otherwise
```

So every slot chases **its own current mistakes**, not a single shared class target. This is the
"error vector per output bit": a binary-ish target for each of the `H*C` bits.

## 4. Build: greedy correlation, exploratory

Each **build phase** fills empty slots. For a sweep that fills `n` slots (`build_sweep`):

1. Pick `n` empty slots at random (the targets to fill now).
2. Take the candidate input signals = all filled slots, **capped** to `max_feats` by sampling
   (Gumbel-top-k over the depth/usage bias below). This is what makes it scale: we never form the
   full `WIN x WIN` matrix.
3. **Correlate**: `cov = Fb_centered @ R_centered^T`, a `(max_feats x n)` covariance of every
   candidate signal against every target slot's residual. (At small scale, with `max_feats >=`
   pool and `n = ` all empties, this is the full `~ f*N^2` correlation matrix; at large scale it
   is a sampled slice of it.)
4. For each target slot, pick its **two input signals**. Not strictly the two strongest: we take
   the top-2 of a **Gumbel-perturbed score**

   ```
   key = log|cov| + feat_bias + explore_temp * Gumbel
   feat_bias = -depth_penalty * depth(slot) - usage_penalty * log(1 + usage(slot))
   ```

   - `explore_temp` injects randomness so we don't always grab the same few high-correlation
     signals (exploration);
   - `depth_penalty` prefers **shallow** slots, so we don't stack too many gates on top of each
     other (`depth(slot) = 1 + max(depth of its two inputs)`);
   - `usage_penalty` prefers **lightly-used** slots, spreading gates across the pool
     (`usage(slot) = #times the slot has been used as a gate input`).
5. Build the gate: an **AND** (or **OR**, `--gate`) of the two inputs, with a **NOT** on an input
   whose signed correlation was negative. Stored as a 4-bit truth table `[f00,f01,f10,f11]`
   indexed by `a_bit*2 + b_bit`. AND of two optionally-negated inputs is a single one-hot truth
   table; OR is its complement; coordinate descent later reaches all 16 functions.
6. Write the gate's output over the whole dataset into the slot (one packed bit op), update the
   class score, the `depth`/`usage` counters, and append the op to the replay history.

Build **only fills empty slots** — it never re-wires a filled slot. Re-wiring from fresh batches
just oscillates; refinement is CD's job.

## 5. Coordinate descent: randomized, no sorting

CD is deliberately **exploratory and unsorted** (`cd_pass`):

1. Pick `n_flip` **random** gate slots.
2. On each, pick **one random** truth-table bit (of the 4).
3. Score *only that one flip* against the current batch hinge loss, for all candidates at once
   (closed form: flipping a slot's gate only moves its own class score, so the loss delta is
   exact in isolation).
4. **Keep** the flips that lower the loss, drop the rest. No scoring of the other 3 bits, no
   top-K, no picking the single best — just random candidate flips, accepted iff they help.

This keeps CD from greedily collapsing onto the locally-best move and explores the 16-function
space. Flipping a random bit can turn an AND into an OR, XOR, NAND, pass-through, constant, etc.

CD uses a **large batch** (`--cd-batch`) on purpose: a flip is kept only if it helps on a big,
representative sample, so CD prunes gates that don't generalize instead of overfitting a tiny
batch. (A small CD batch silences good gates and collapses the model to chance — observed.)

## 6. The schedule: alternating build / CD phases

The run alternates phases; each phase is **build then CD**:

```
repeat:
    build_sweep:   add  build_per_phase  gates (fills empty slots; 0 once the window is full)
    cd:            run  round(cd_fraction * total_gates_built)  random bitflips,
                   in chunks of  cd_flips  attempts, each on a fresh cd_batch
until the window is full, then extra_cd_phases more CD-only phases.
```

Because build adds a constant chunk per phase while CD is a **fraction of all gates so far**, CD
automatically grows as the window fills — early phases are build-heavy, late phases are CD-heavy,
with no hand-tuned ramp. The first build covers every class (round-robin), so every GroupSum has
gates before the first CD.

Example (the kind of setting suggested for a real run): `--build-per-phase 10000
--cd-fraction 0.25 --cd-flips 8192` → 10k new gates then ~25%-of-total random bitflips each phase.

## 7. Inference

The op history is a list of vectorized **op-batches** `(slots, a, b, tt)`. Every gate in a build
sweep or CD pass reads only signals from *earlier* batches, so replay is exact and fully
vectorized: initialize a fresh window (inputs packed, rest 0), apply each op-batch in order
(`win[slots] = gate(win[a], win[b], tt)`), then GroupSum. Validation and test accuracy come from
replaying on their images.

## 8. Hyperparameters

| flag | meaning | default |
|------|---------|---------|
| `--num-bits b` | thermometer bits per channel; `N = 3072 b` | 5 |
| `--window-factor f` | `WIN = f*N` slots (so `H*C = f*N`); pick `b*f` divisible by 5 | 2.0 |
| `--max-gates` | hard upper bound on gates built | 200000 |
| `--max-feats` | pool signals correlated per build sweep (bounds the matmul) | 16384 |
| `--gate {and,or}` | initial gate family before CD | and |
| `--explore-temp` | Gumbel temperature for input selection (0 = strict top-2) | 0.7 |
| `--depth-penalty` | bias against deep slots as inputs (keeps circuits shallow) | 0.5 |
| `--usage-penalty` | bias against reusing heavily-used slots | 0.3 |
| `--build-batch` | images per build correlation | 8192 |
| `--cd-batch` | images per CD pass (large = less overfit) | 16384 |
| `--cd-flips` | random flips attempted per CD call | 8192 |
| `--build-per-phase` | empty slots filled per build phase | 10000 |
| `--cd-fraction` | CD bitflips per phase as a fraction of all gates so far | 0.25 |
| `--extra-cd-phases` | CD-only phases after the window is full | 30 |
| `--train-size` | train+val pool (0 = full 50k) | 0 |

## 9. Scaling

To grow the network, raise `--window-factor` (and `--max-gates`): `WIN = f*N` is the number of
gates. With `N = 15360` (`b=5`), `f=8` gives ~123k gates, `f≈64` gives ~1M. Cost is roughly
linear in `WIN` because the build matmul is capped at `max_feats x build_per_phase` and CD is
`O(flips)`; memory is dominated by the window, `WIN * ceil(D/64) * 8` bytes (≈5.6 GB at 1M gates
on the full 50k set), which fits a 24 GB RTX 3090. Larger windows want a larger `--max-feats` so
gates can find good inputs, and more `--extra-cd-phases`.

## 10. Run

```bash
# interactive single GPU, watch the live phase table
bash scratch/run_srun.sh

# batch single GPU, output to scratch/grow.log
sbatch scratch/run.sbatch                       # defaults
sbatch scratch/run.sbatch --window-factor 8 --max-gates 130000   # bigger

# direct
.venv/bin/python scratch/grow_lut.py --device cuda --train-size 0

# fast CPU sanity check (deliberately weak: 1 thermometer bit, small window)
.venv/bin/python scratch/grow_lut.py --device cpu --train-size 2500 --num-bits 1 \
    --window-factor 2 --build-per-phase 600 --cd-flips 512 --extra-cd-phases 8 \
    --build-batch 1024 --cd-batch 2250 --max-feats 4096
```

The live table prints, per phase, the gate count, gates built / bitflips kept, and
train/val/test accuracy.

## 11. Status

Validated on CPU smoke runs (no GPU in the dev session). On the deliberately weak smoke above
(1 thermometer bit, ~2250 images) it goes from 10% chance to ~27% test, with CD adding a few
points over build alone and depth staying shallow (max ~4). The intended way to run it is on a
single RTX 3090 with `b=5`, the full 50k set, a larger window and many CD phases; pushing test
accuracy up is an open tuning loop over window size, batch sizes, gate family and the
build/CD ratio. This is a scratch experiment, separate from the main tutorial code.

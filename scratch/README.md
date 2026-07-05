# scratch: pure coordinate descent on a stack of random LUT layers

One file: [`cd.py`](cd.py). No building, no growing, no backprop — a stack of fixed 3D
windows `(C_l, 32, 32)` of random K-input LUT gates (`--channels "c0,c1,..."`; one number =
the original flat depth-0 model) over the thermometer-encoded image, and **coordinate
descent learns everything** through three levers per gate:

1. **Truth table** (2^K bits). Each (copy, sample) lands on exactly one LUT cell, so the
   cells partition the data: one visit proposes *all* 2^K bits at once from the exact
   direct-vote deltas (block-CD for the price of one evaluation).
2. **Connections** (K absolute source coords). A gate in layer *l* may tap the input grid
   or any lower layer (skip connections included) — rank is explicit, so no cycles. One
   random slot, n candidates (half local), keep the best.
3. **Weight sharing** (a copies-count per window dimension, powers of 2) plus a learned
   **input stride** (`step` per dim). A gate with copies (nc, nh, nw) occupies slots of its
   layer strided dim/n from its base — the output tiling is fixed — but copy (i, j, k)
   reads its taps shifted (i·step_c, j·step_h, k·step_w) within each tap's own source grid:
   step = dim/n is plain conv, smaller overlaps, larger dilates, 0 ties copies exactly.
   Sharing *is* convolution — kernel shape (the taps), stride, dilation and tying all
   per-gate and learned. Splits (net2net-style clones, all dims incl. channels) unshare
   output-neutrally.

Head: every slot of every layer votes; slot class = channel % 10; score = per-class
popcount / sqrt(S/10); loss = Crammer–Singer hinge.

**Depth pays one cost — the cascade.** Changing a gate changes stored output bits that feed
higher layers, so an accept recomputes every (transitively) dirty reader, adds its vote
delta, and XOR-reverts everything if the exact hinge does not improve (`--casc-cap` bounds
runaway cascades). Top-layer moves cost the same as depth 0. An epsilon-greedy bandit
allocates work units across (operator × layer) arms by measured hinge-decrease-per-second.

**Every accept is exact on the full (freshly augmented) train set.** A per-round
random-batch mode was tried and measured strictly worse — accepts overfit the batch and
val crawled — so per the keep-only-winners rule it is gone; re-rolled augmentation is the
stochasticity. Losers removed the same way: static mirror-doubling, cutout (hurts while
the model underfits), and dead knobs.

## Run

```bash
# 8-layer deep run on a cluster GPU
sbatch --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=48:00:00 --job-name=cdd8 \
    --output=scratch/cdd8.out --wrap="bash scratch/cd.sh --ckpt scratch/cdd8.pt \
    --channels 320,160,80,40,20,20,10,10 --batch 1024 --init-deg 0,2,2 --aug full"

# resume a killed/expired run from the checkpoint (append output, keep history)
sbatch ... --open-mode=append --wrap="bash scratch/cd.sh --ckpt scratch/cdd8.pt --resume ..."

# CPU smoke test with exactness checks (score + every stored output row must be 0.0)
.venv/bin/python scratch/cd.py --device cpu --channels 20,10,10 --train-size 800 \
    --batch 256 --rounds 2 --n-cand 4 --share-moves 10 --pass-rows 512 --check
```

Knobs: `--channels` (per-layer sizes = depth), `--fan-in`, `--num-bits`, `--casc-cap`,
`--rewire-frac` / `--n-cand` / `--share-moves` / `--rs-*` / `--splits` (lever mix),
`--explore` (bandit exploration schedule), `--rs-temp` (annealed uphill acceptance),
`--init-deg` (conv-tiled start), `--aug*` (re-rolled flip/crop/jitter), `--pass-rows`,
`--ckpt/--resume`.

With `--ckpt` set, per-round distributions (truth-table bit stats, sharing histogram,
per-layer gate counts, bandit arm stats) are appended to `<ckpt>.jsonl`. Render plots /
animation any time, also mid-run:

```bash
.venv/bin/python scratch/plot.py scratch/cdd8 --gif   # -> cdd8_plots.png, cdd8_anim.gif
```

Reference points: flat depth-0 CD run with re-rolled flip aug reached val 55.0 / test 54.3
(SGD on the same flat architecture: 53.6; old grow+CD approach: val ~44). Depth exists to
break the shallow ceiling: a depth-0 model is a weighted bag of K-pixel patterns and tops
out in the 60s even with perfect conv structure.

Shape race (measured, 5h head-to-head at code parity): 8x32² flat-spatial BEAT a CNN-style
pyramid (32→4 grids, 640→5120 ch) on val-per-wall-clock — pyramid was stronger per round
but its 2x round cost never amortized (gap 1.9 → 1.3 → 1.5 at close). `--spatial` stays
for the scale path (coarse layers cut the slots x D stored-output memory quadratically),
but the default shape is flat spatial.

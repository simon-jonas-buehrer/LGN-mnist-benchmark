# Optimizer benchmark on a fixed LUT network

One architecture, four optimizers. The network is frozen for every run: Thermometer
encoder `(B, C*b, H, W)` -> 8 LUT layers of 64,000 fan-in-4 nodes wired in the same
deterministic **monarch** pattern -> GroupSum head `(B, h*C)` -> cross-entropy.
Light augmentation (flip + crop-4). The only thing that differs is HOW the truth tables
(and optionally the connections, via k=8 fixed candidate sources per tap) are learned:

| method | truth tables                    | connections (`--learn-conn 1`)          |
|--------|---------------------------------|-----------------------------------------|
| `bp`   | Adam on sin straight-through, Gaussian init | straight-through softmax over k |
| `cd`   | block coordinate descent, Bernoulli init, exact batch accepts | k candidates scored independently, best kept |
| `rs`   | (1+1)-ES joint mutations, Bernoulli init | tap re-draws jointly with tt flips |
| `mab`  | per-bit 2-armed bandit (REINFORCE), Gaussian init | per-tap k-armed bandit policy |

Every learning curve is logged against **samples seen** (train/val loss, accuracy,
perplexity) to `scratch/runs/<method>_conn<0|1>_s<seed>.jsonl`, with checkpoints in the
matching `.pt` (test metrics at the final eval).

```
mkdir -p .local/logs scratch/runs && sbatch .local/optbench.sbatch   # 24 runs (8 cfg x 3 seeds)
.venv/bin/python scratch/opt.py --method bp --learn-conn 1 --out scratch/runs/bp_conn1_s0
.venv/bin/python scratch/plot.py                                     # -> scratch/runs/curves.png
```

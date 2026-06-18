# LUT-Network Tutorial: boolean logic gates trained by gradient descent


*Hi Branton this Tutorial is for you!*

In bad fashion I made this tutorial with Claude, but i personall reviewed all of it and its fine.

A tiny, self-contained introduction to Look-Up Table (LUT) networks: neural networks whose
neurons are learnable boolean logic gates instead of weighted sums. Everything is pure
PyTorch, there are no custom CUDA kernels, and the model is about 200 lines you can read in
one sitting.

We train it on CIFAR-10 with no regularization and no data augmentation, so you can watch it
overfit: training accuracy keeps pulling ahead while validation and test accuracy stall.

## What is a LUT network?

A normal neuron computes `sigma(w*x + b)`, a weighted sum of all of its inputs. A LUT neuron
does something different:

1. it reads a small, fixed number of input bits (here the fan-in is 2), and
2. it applies a learned 2-input boolean function to them, stored as a 4-entry truth table.

A 2-input truth table has `2^2 = 4` entries `f(0,0), f(0,1), f(1,0), f(1,1)`. Four free bits
are enough to express all 16 boolean functions of two inputs (AND, OR, XOR, NAND, "pass A",
"constant 0", and so on), so each neuron learns which logic gate it should be. Stack a few
layers and you get a deep combinational logic circuit. It trains by SGD and maps directly onto
the LUTs inside an FPGA or ASIC at inference time. This is the idea behind
[DiffLogic](https://arxiv.org/abs/2210.08277) and related work.

The full pipeline, where every signal between layers is a single bit:

```
image (3x32x32 pixels)
  -> Thermometer encoder    real pixel -> a few threshold bits
  -> Flatten
  -> N x LUTLayer           each neuron = a learned 2-input gate (fan-in 2)
  -> GroupSum head          popcount the final bits into 10 class scores
  -> logits
```

## How do you backprop through a boolean gate?

You cannot differentiate `1[sin(z) > 0]`. The fix is a straight-through estimator, built on
the light parametrization with a `sin` activation:

```python
hard = (sin(z) > 0)            # exact 0/1, used in the forward pass
soft = 0.5 + 0.5*sin(z)        # smooth, used for the gradient
bit  = hard + (soft - soft.detach())   # forward = hard, backward = d(soft)
```

Each of a neuron's 4 truth-table bits is stored as a real latent `z`. The forward pass is an
exact boolean circuit, so `train` and `eval` agree bit-for-bit, while the backward pass flows
a smooth `sin` gradient that nudges each latent toward 0 or 1. Because `sin` is periodic the
latent never saturates and always has a gradient, unlike a sigmoid. The gate itself is
evaluated with the multilinear form of a 2-input LUT (see `LUTLayer.forward` in
[`model.py`](model.py)).

## Files

| file          | what it is                                                        |
|---------------|-------------------------------------------------------------------|
| `model.py`    | the whole network: sin activation, `LUTLayer`, thermometer, head  |
| `train.py`    | CIFAR-10 loader (no torchvision) plus the training loop and plots |
| `load.py`     | reload the saved checkpoint, evaluate it, inspect a learned gate  |
| `results/`    | the committed run: `train.log`, `metrics.csv`, `curves.png`, `lut_cifar10.pt` |

## Setup (uv)

```bash
# install uv once: https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh

cd lut-tutorial
uv sync          # creates .venv with torch + numpy + matplotlib (CPU or CUDA, picked automatically)
```

## Run

```bash
# Train on the full CIFAR-10 training set (downloads the data on first run). A GPU is
# recommended; on a laptop CPU use e.g. --train-size 2000 --width 4000 --layers 2 --epochs 30.
uv run python train.py --download

# Reload the trained checkpoint, check test accuracy, peek at a learned gate
uv run python load.py
```

Data is split the conventional way. The training set is split 90/10 into train and validation,
and the standard 10k CIFAR-10 test set is kept separate. The thermometer thresholds are fit on
training data only, so there is no leakage.

Every epoch prints train, validation and test loss, accuracy and perplexity, and the run
writes `results/train.log`, `results/metrics.csv`, `results/curves.png` and the checkpoint
`results/lut_cifar10.pt`. The committed `results/` come from training a large model (width
200000, 6 layers, about 1.2M trainable tables) on the full set for 100 epochs with no
regularization, on a single RTX 3090:

```bash
uv run python train.py --device cuda --train-size 0 --num-bits 3 --width 200000 --layers 6 --batch-size 1024 --lr 2e-2 --epochs 100
```

![learning curves](results/curves.png)

Train accuracy and perplexity keep improving while validation and test stall. That gap is
overfitting, which is the whole point of training with no regularization and no augmentation.
A larger model has more capacity, fits the training set harder, and widens the gap.

## Knobs to play with

See `train.py --help`. The main ones are `--num-bits` (thermometer resolution), `--width` and
`--layers` (capacity; wider overfits harder, and a handful of layers is usually enough), plus
`--train-size`, `--lr` and `--epochs`.

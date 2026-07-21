"""Phase-1 driver for the tao prototype: does a binary net of decision trees learn?

    python records/sbuehrer/tao/proto.py --selfcheck
    python records/sbuehrer/tao/proto.py --flipcheck
    python records/sbuehrer/tao/proto.py --widths 512,320 --bits 3
    python records/sbuehrer/tao/proto.py --stack-ablate

Trees are depth 2 from here on: capacity comes from more nodes, in width and in stack depth, not
from bigger trees. A node's area grows like 2^D, while at matched node count the accuracy gain per
depth step was already shrinking (81.05 -> 86.12 -> 88.40, so +5.1 then +2.3) against a steady
~1.9x in area each step.

Reads train_* and val_* only. The test set is the harness's, and is never touched here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for `mnistbench`
sys.path.insert(0, str(Path(__file__).resolve().parent))      # this dir, for `tao`

from mnistbench.data import load  # noqa: E402
from tao import TaoNet, estimate_gates, fit, predict_numpy, route_numpy  # noqa: E402


def make(a: argparse.Namespace, device: str, widths: str | None = None) -> TaoNet:
    w = tuple(int(v) for v in (widths or a.widths).split(","))
    return TaoNet(a.bits, w, a.depth, seed=a.seed).to(device)


def report(net: TaoNet, acc: float, tag: str = "") -> dict:
    e = estimate_gates(net)
    print(f"\n=== {tag or 'result'}")
    print(f"  val accuracy      {acc:.2f}%")
    print(f"  estimated GE      {e['ge_est']:,}  "
          f"(logic {e['logic']:,} + encoder {e['encoder']:,} + head {e['head']:,})")
    print(f"  live split slots  {e['live_slots']:,}   encoder bits used {e['enc_bits_used']:,}")
    for i, l in enumerate(e["layers"]):
        print(f"  layer {i}: {l['width']:5d} nodes  ~{l['ge']:6,} GE  "
              f"{l['live_slots']:6,} live slots  {l['free_nodes']:5,} free")
    print("  (pre-ABC estimate: prices each node alone, cannot see the sharing ABC finds between\n"
          "   them. Order of magnitude only.)")
    return e


def selfcheck(a: argparse.Namespace, device: str) -> None:
    """The torch path and a pure-numpy router must be the same function, bit for bit. This is the
    local stand-in for the harness's predict()-vs-netlist check."""
    data = load()
    net = make(a, device)
    pix = torch.from_numpy(np.ascontiguousarray(data.val_x[:512])).to(device)
    acts = net.activations(pix)

    h = acts[0].cpu().numpy().astype(np.uint8)
    thr = np.asarray(net.thresholds, np.int16)
    ref = (data.val_x[:512].astype(np.int16)[:, :, None] > thr).reshape(512, -1).astype(np.uint8)
    assert (h == ref).all(), "encoder: torch and numpy disagree"
    print(f"[ok] encoder            {h.shape[1]:,} bits match")

    for li, lay in enumerate(net.layers):
        t = acts[li + 1].cpu().numpy()
        assert np.isin(t, (0.0, 1.0)).all(), f"layer {li}: activation is not exactly 0.0/1.0"
        rep = lay.path_repeats()
        assert rep == 0, f"layer {li}: {rep} paths test a feature twice"
        r = route_numpy(h, lay.feat.cpu().numpy(), lay.leaf.cpu().numpy().astype(np.uint8))
        bad = int((t.astype(np.uint8) != r).sum())
        assert bad == 0, f"layer {li}: {bad} bits differ"
        print(f"[ok] layer {li}             {t.shape[1]:,} bits match, all exactly 0/1")
        h = r

    ct = net.predict(pix).cpu().numpy()
    cn = predict_numpy(net, data.val_x[:512])
    assert (ct == cn).all(), f"class differs on {int((ct != cn).sum())}/512"
    print("[ok] class              512/512 match\n\ntorch routing == numpy routing, bit for bit.")


def flipcheck(a: argparse.Namespace, device: str) -> None:
    """The downward message must be an EXACT counterfactual: a +1 vote has to mean the node really
    would be fixed by flipping that bit, and votes may only land on the path actually taken. This
    is the load-bearing claim of the whole design, so it is tested directly rather than assumed."""
    from tao import TreeLayer, flip_votes

    g = torch.Generator().manual_seed(a.seed)
    F, M, B, D = 64, 24, 12, a.depth
    lay = TreeLayer(F, M, D, g).to(device)
    x = (torch.rand(B, F, device=device) > 0.5).float()
    tgt = (torch.rand(B, M, device=device) > 0.5).float()

    votes = flip_votes(lay, x, tgt)
    print(f"[ok] sparsity           <= {int((votes != 0).sum(1).max())} of {F} bits voted on "
          f"by {M} depth-{D} nodes")

    hit = (lay(x) == tgt)
    checked = 0
    for b, f in zip(*[t.tolist() for t in (votes != 0).nonzero(as_tuple=True)]):
        xf = x.clone()
        xf[b, f] = 1.0 - xf[b, f]
        want = float(((lay(xf)[b] == tgt[b]).float() - hit[b].float()).sum())
        assert abs(want - float(votes[b, f])) < 1e-5, \
            f"b={b} f={f}: vote {float(votes[b, f])} but flipping really gives {want}"
        checked += 1
    print(f"[ok] exactness          {checked} votes equal the real effect of flipping the bit")
    print("\nthe downward message is an exact counterfactual, and lives only on the path taken.")


def _train(a: argparse.Namespace, device: str, data, *, widths: str | None = None,
           tag: str = "") -> tuple[float, dict]:
    print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}", flush=True)
    net = make(a, device, widths)
    acc = fit(net, data, device=device, seed=a.seed, epochs=a.epochs, steps=a.steps, rows=a.rows,
              topk=a.topk, mtry=a.mtry, chunk=a.chunk, pick=a.pick, patience=a.patience,
              log_every=a.log_every)
    return acc, report(net, acc, tag)


def stack_ablate(a: argparse.Namespace, device: str) -> None:
    """At depth-2 trees and matched node count, is a taller stack better than a wider one? This is
    the question that replaces tree depth: capacity has to come from somewhere."""
    data = load()
    runs = ["1024,320", "512,512,320", "448,320,320,256"]
    out = [(w, *_train(a, device, data, widths=w, tag=f"stack {w}")) for w in runs]
    print(f"\n{'=' * 78}\nstack shape, depth-2 trees, ~matched node count\n{'=' * 78}")
    for w, acc, e in out:
        print(f"  {acc:6.2f}%  ~{e['ge_est']:>8,} GE   {w}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bits", type=int, default=3, help="thermometer bits per pixel")
    p.add_argument("--widths", default="512,320", help="nodes per layer; last %% 10 == 0")
    p.add_argument("--depth", type=int, default=2, help="tree depth per node")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps", type=int, default=20, help="update steps per epoch")
    p.add_argument("--rows", type=int, default=512, help="batch size for one update step")
    p.add_argument("--topk", type=int, default=1,
                   help="decisions a node may change per step, of its 2^depth-1. The step size")
    p.add_argument("--pick", default="cycle", choices=("cycle", "random", "best"),
                   help="which decision a node changes per step: round-robin by (node + step) "
                        "mod slots, a random slot, or the slot whose change helps most (greedy, "
                        "and biased toward the root)")
    p.add_argument("--mtry", type=int, default=0,
                   help="candidate bits per node-chunk, 0 = all of them (the default: a node may "
                        "rewire any decision to ANY bit of the previous layer). Subsetting is a "
                        "speed shortcut and costs accuracy -- 70.33%% at 256 vs 72.17%% at 0")
    p.add_argument("--chunk", type=int, default=256, help="nodes per counting pass")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--selfcheck", action="store_true", help="torch routing == numpy routing")
    p.add_argument("--flipcheck", action="store_true", help="votes are exact counterfactuals")
    p.add_argument("--stack-ablate", action="store_true", help="taller vs wider, matched nodes")
    a = p.parse_args()

    dev = a.device
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[dev ] {dev}", flush=True)

    if a.selfcheck:
        return selfcheck(a, dev)
    if a.flipcheck:
        return flipcheck(a, dev)
    if a.stack_ablate:
        return stack_ablate(a, dev)

    data = load()
    net = make(a, dev)
    acc = fit(net, data, device=dev, seed=a.seed, epochs=a.epochs, steps=a.steps, rows=a.rows,
              topk=a.topk, mtry=a.mtry, chunk=a.chunk, pick=a.pick, patience=a.patience,
              log_every=a.log_every)
    report(net, acc)
    ct = net.predict(torch.from_numpy(np.ascontiguousarray(data.val_x[:512])).to(dev)).cpu().numpy()
    bad = int((ct != predict_numpy(net, data.val_x[:512])).sum())
    print(f"  torch vs numpy path: {512 - bad}/512 agree" + ("" if not bad else "  <-- BUG"))


if __name__ == "__main__":
    main()

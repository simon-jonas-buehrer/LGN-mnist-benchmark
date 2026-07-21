"""Phase-1 driver for the tao prototype: does a deep net of decision trees learn?

    python records/sbuehrer/tao/proto.py --selfcheck
    python records/sbuehrer/tao/proto.py --gradcheck
    python records/sbuehrer/tao/proto.py --widths 1024,512,320 --bits 3 --depth 3
    python records/sbuehrer/tao/proto.py --signal-ablate
    python records/sbuehrer/tao/proto.py --ablate
    python records/sbuehrer/tao/proto.py --depth-ablate

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

from mnistbench.data import N_CLASSES, load  # noqa: E402
from tao import TaoNet, estimate_gates, fit, predict_numpy, route_numpy  # noqa: E402


def make(a: argparse.Namespace, device: str) -> TaoNet:
    widths = tuple(int(w) for w in a.widths.split(","))
    return TaoNet(a.bits, widths, a.depth, seed=a.seed).to(device)


def report(net: TaoNet, acc: float, tag: str = "") -> dict:
    e = estimate_gates(net)
    print(f"\n=== {tag or 'result'}")
    print(f"  val accuracy      {acc:.2f}%")
    print(f"  estimated GE      {e['ge_est']:,}  "
          f"(logic {e['logic']:,} + encoder {e['encoder']:,} + head {e['head']:,})")
    print(f"  live split slots  {e['live_slots']:,}   encoder bits used {e['enc_bits_used']:,}")
    for i, l in enumerate(e["layers"]):
        print(f"  layer {i}: {l['width']:5d} nodes  ~{l['ge']:6,} GE  "
              f"{l['live_slots']:6,} live slots  {l['free_nodes']:5,} free "
              f"(collapsed to a constant or a bare literal)")
    print("  (the GE figure is a pre-ABC estimate: it prices each node alone and cannot see the\n"
          "   sharing ABC finds between them. Order of magnitude only.)")
    return e


# ==========================================================================================
# Check 1: the torch multilinear forward and the reference router are the same function.
# ==========================================================================================
def selfcheck(a: argparse.Namespace, device: str) -> None:
    data = load()
    net = make(a, device)
    # a trained-looking net is a weaker test than a random one for structure bugs, but leaves
    # must be exercised in both directions, so randomise the latents first
    with torch.no_grad():
        for lay in net.layers:
            lay.leaf.normal_()

    pix = torch.from_numpy(np.ascontiguousarray(data.val_x[:512])).to(device)
    with torch.no_grad():
        acts = net.activations(pix)

    h = acts[0].cpu().numpy().astype(np.uint8)
    thr = np.asarray(net.thresholds, np.int16)
    enc_ref = (data.val_x[:512].astype(np.int16)[:, :, None] > thr).reshape(512, -1).astype(np.uint8)
    assert (h == enc_ref).all(), "encoder: torch and numpy disagree"
    print(f"[ok] encoder            {h.shape[1]:,} bits match")

    for li, lay in enumerate(net.layers):
        t = acts[li + 1].cpu().numpy()
        assert np.isin(t, (0.0, 1.0)).all(), f"layer {li}: activation is not exactly 0.0/1.0"
        rep = lay.path_repeats()
        assert rep == 0, f"layer {li}: {rep} paths test a feature twice (breaks the gradient)"
        ref = route_numpy(h, lay.feat.cpu().numpy(),
                          lay.leafbit().cpu().numpy().astype(np.uint8))
        bad = int((t.astype(np.uint8) != ref).sum())
        assert bad == 0, f"layer {li}: {bad} bits differ between multilinear and router"
        print(f"[ok] layer {li}             {t.shape[1]:,} bits match, all exactly 0/1")
        h = ref

    with torch.no_grad():
        cls_t = net(pix).argmax(1).cpu().numpy()
    cls_n = predict_numpy(net, data.val_x[:512])
    assert (cls_t == cls_n).all(), f"class differs on {int((cls_t != cls_n).sum())}/512"
    print("[ok] class              512/512 match")
    print("\nmultilinear forward == tree routing, bit for bit.")


# ==========================================================================================
# Check 2: the gradient w.r.t. an input bit is the exact finite difference, and is supported
# only on the path actually taken. This is the load-bearing claim of the whole design.
# ==========================================================================================
def gradcheck(a: argparse.Namespace, device: str) -> None:
    torch.manual_seed(a.seed)
    from tao import TreeLayer

    g = torch.Generator().manual_seed(a.seed)
    F, M, B, D = 64, 32, 16, a.depth
    lay = TreeLayer(F, M, D, g).to(device)
    with torch.no_grad():
        lay.leaf.normal_()

    x = (torch.rand(B, F, device=device) > 0.5).float().requires_grad_(True)
    out = lay(x)
    assert torch.isin(out.detach(), torch.tensor([0.0, 1.0], device=device)).all()

    # sparsity: one node at a time, so the count is per (sample, node)
    worst = 0
    for m in range(M):
        if x.grad is not None:
            x.grad = None
        out = lay(x)
        out[:, m].sum().backward()
        nz = (x.grad.abs() > 0).sum(1)                     # (B,) nonzeros per sample
        worst = max(worst, int(nz.max()))
    assert worst <= D, f"gradient touches {worst} inputs, but a depth-{D} path has {D}"
    print(f"[ok] sparsity           <= {worst} of {F} inputs get gradient (depth {D})")

    # exactness: grad == (output with the bit flipped) - (output now)
    x.grad = None
    out = lay(x)
    out.sum().backward()
    grad = x.grad.clone()
    base = out.detach()
    checked = 0
    for b, f in zip(*[t.tolist() for t in (grad.abs() > 0).nonzero(as_tuple=True)]):
        xf = x.detach().clone()
        xf[b, f] = 1.0 - xf[b, f]
        delta = (lay(xf)[b] - base[b]).sum()
        # the derivative of the multilinear extension at a binary point is out(1) - out(0), so
        # FLIPPING the bit moves the output by (1 - 2*x_f) times it. No truncation error: this
        # is an identity, not an approximation.
        want = (1.0 - 2.0 * x.detach()[b, f]) * grad[b, f]
        assert torch.isclose(delta, want, atol=1e-5), \
            f"b={b} f={f} x={x.detach()[b, f]:.0f}: expected {want:.4f}, flipping gives {delta:.4f}"
        checked += 1
    print(f"[ok] exactness          {checked} on-path partials equal their finite difference")
    print("\nthe gradient w.r.t. an input bit is exact, and lives only on the path taken.")


# ==========================================================================================
# Check 4: is the alternation earning its complexity?  Check 5: is depth doing anything?
# ==========================================================================================
def _train(a: argparse.Namespace, device: str, data, *, widths: str | None = None,
           tag: str = "", **kw) -> tuple[float, dict]:
    b = argparse.Namespace(**vars(a))
    if widths:
        b.widths = widths
    print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}", flush=True)
    net = make(b, device)
    opts = dict(signal=a.signal, do_grad=a.do_grad, do_refit=True, revert=not a.no_revert,
                refit_steps=a.refit_steps, topk=a.topk)
    opts.update(kw)
    acc = fit(net, data, device=device, seed=a.seed, epochs=a.epochs, batch=a.batch, lr=a.lr,
              patience=a.patience, refit_every=a.refit_every, refit_rows=a.refit_rows,
              refit_frac=a.refit_frac, mtry=a.mtry, chunk=a.chunk, bag=a.bag,
              log_every=a.log_every, **opts)
    return acc, report(net, acc, tag)


def signal_ablate(a: argparse.Namespace, device: str) -> None:
    """Does a node need a real gradient, or is an exact discrete message from the trees above
    enough? The last two arms touch no derivative anywhere: messages down, trees rebuilt, and
    nothing else."""
    data = load()
    runs = [
        ("grad signal + leaf gradient", dict(signal="grad", do_grad=True)),
        ("flip signal + leaf gradient", dict(signal="flip", do_grad=True)),
        ("flip signal only -- NO derivative anywhere", dict(signal="flip", do_grad=False)),
        ("flip signal only, no loss-gated revert (fully local)",
         dict(signal="flip", do_grad=False, revert=False)),
    ]
    out = [(tag, *_train(a, device, data, tag=tag, **kw)) for tag, kw in runs]
    print(f"\n{'=' * 78}\nsignal ablation: is a discrete message enough?\n{'=' * 78}")
    for tag, acc, e in out:
        print(f"  {acc:6.2f}%  ~{e['ge_est']:>8,} GE   {tag}")


def ablate(a: argparse.Namespace, device: str) -> None:
    data = load()
    runs = [
        ("full: init + refit + leaf gradient", dict(do_grad=True, do_refit=True)),
        ("leaf gradient only (structure frozen after init)", dict(do_grad=True, do_refit=False)),
        ("refit only (no gradient on the leaves)", dict(do_grad=False, do_refit=True)),
    ]
    out = [(tag, *_train(a, device, data, tag=tag, **kw)) for tag, kw in runs]
    print(f"\n{'=' * 78}\nablation: does the alternation earn its complexity?\n{'=' * 78}")
    for tag, acc, e in out:
        print(f"  {acc:6.2f}%  ~{e['ge_est']:>8,} GE   {tag}")
    if out[0][1] <= max(o[1] for o in out[1:]):
        print("\n  the full loop does NOT beat both halves -- the alternation is not paying for\n"
              "  itself, and the design needs rethinking before any Verilog is written.")


def depth_ablate(a: argparse.Namespace, device: str) -> None:
    """One layer at matched node count vs the deep stack. If they tie, nothing is composing and
    this is just a worse forest."""
    data = load()
    widths = [int(w) for w in a.widths.split(",")]
    flat = sum(widths)
    flat -= flat % N_CLASSES
    out = [
        ("deep: " + a.widths, *_train(a, device, data, tag="deep: " + a.widths)),
        (f"flat: {flat}", *_train(a, device, data, widths=str(flat), tag=f"flat: {flat}")),
    ]
    print(f"\n{'=' * 78}\ndepth ablation: is anything composing?\n{'=' * 78}")
    for tag, acc, e in out:
        print(f"  {acc:6.2f}%  ~{e['ge_est']:>8,} GE   {tag}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bits", type=int, default=3, help="thermometer bits per pixel")
    p.add_argument("--widths", default="1024,512,320", help="nodes per layer; last %% 10 == 0")
    p.add_argument("--depth", type=int, default=3, help="tree depth per node")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--refit-every", type=int, default=2, help="epochs between structure refits")
    p.add_argument("--refit-rows", type=int, default=2048, help="rows the refit is fit on")
    p.add_argument("--refit-frac", type=float, default=0.1,
                   help="fraction of nodes rebuilt per refit round (the trust region)")
    p.add_argument("--mtry", type=int, default=1024, help="candidate features per node-chunk, 0=all")
    p.add_argument("--chunk", type=int, default=256, help="nodes per refit GEMM")
    p.add_argument("--bag", type=float, default=0.7, help="per-node weight bagging rate, 0=off")
    p.add_argument("--signal", default="flip", choices=("flip", "grad"),
                   help="where a node's target comes from: exact discrete counterfactual votes "
                        "from the trees above (flip), or a backward pass (grad)")
    p.add_argument("--do-grad", action="store_true",
                   help="also run Adam on the leaf latents (off = no derivative anywhere)")
    p.add_argument("--no-revert", action="store_true",
                   help="skip the loss-gated revert, making the update purely local")
    p.add_argument("--topk", type=int, default=0,
                   help="how many of a node's 2^depth-1 decisions may be rewired per step "
                        "(0 = all). The step size on structure -- small k keeps each move quiet")
    p.add_argument("--refit-steps", type=int, default=1,
                   help="refit rounds per epoch, each on a fresh batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--selfcheck", action="store_true", help="multilinear forward == tree routing")
    p.add_argument("--gradcheck", action="store_true", help="grad == finite difference, on-path")
    p.add_argument("--ablate", action="store_true", help="full vs gradient-only vs refit-only")
    p.add_argument("--signal-ablate", action="store_true",
                   help="gradient signal vs discrete flip votes vs fully-local flip votes")
    p.add_argument("--depth-ablate", action="store_true", help="deep stack vs one flat layer")
    a = p.parse_args()

    dev = a.device
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[dev ] {dev}", flush=True)

    if a.selfcheck:
        return selfcheck(a, dev)
    if a.gradcheck:
        return gradcheck(a, dev)
    if a.ablate:
        return ablate(a, dev)
    if a.signal_ablate:
        return signal_ablate(a, dev)
    if a.depth_ablate:
        return depth_ablate(a, dev)

    data = load()
    net = make(a, dev)
    acc = fit(net, data, device=dev, seed=a.seed, epochs=a.epochs, batch=a.batch, lr=a.lr,
              patience=a.patience, refit_every=a.refit_every, refit_rows=a.refit_rows,
              refit_frac=a.refit_frac, mtry=a.mtry, chunk=a.chunk, bag=a.bag,
              signal=a.signal, do_grad=a.do_grad, revert=not a.no_revert,
              refit_steps=a.refit_steps, topk=a.topk, log_every=a.log_every)
    report(net, acc)

    # the harness will demand predict() == the circuit; prove the numpy path agrees now
    cls_t = net(torch.from_numpy(np.ascontiguousarray(data.val_x[:512])).to(dev)).argmax(1)
    cls_n = predict_numpy(net, data.val_x[:512])
    bad = int((cls_t.cpu().numpy() != cls_n).sum())
    print(f"  torch vs numpy path: {512 - bad}/512 agree" + ("" if not bad else "  <-- BUG"))


if __name__ == "__main__":
    main()

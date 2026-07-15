"""CLI.

    python -m mnistbench run records/sbuehrer/backprop [--point s] [--device cuda]
    python -m mnistbench pareto
    python -m mnistbench.selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import bench, data, pareto


def main() -> None:
    p = argparse.ArgumentParser(prog="mnistbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="train, synthesize and measure a record's points")
    r.add_argument("record", type=Path, help="records/<user>/<method>")
    r.add_argument("--point", action="append", help="only this point (repeatable)")
    r.add_argument("--device", default="cpu", help="passed straight to the submission")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--force", action="store_true", help="re-measure points already in results.json")

    s = sub.add_parser("rescore", help="re-measure stored .sv artifacts without retraining")
    s.add_argument("record", type=Path)

    g = sub.add_parser("merge", help="rebuild results.json from per-point files (after a "
                                     "parallel run, where each point ran as its own job)")
    g.add_argument("record", type=Path)

    sub.add_parser("pareto", help="rebuild the Pareto plot and the leaderboard table")

    args = p.parse_args()
    if args.cmd == "pareto":
        return pareto.main()
    if args.cmd == "merge":
        return bench.merge_record(args.record)

    d = data.load()
    print(f"MNIST: train {d.train_x.shape}, val {d.val_x.shape}, test {d.test_x.shape}",
          flush=True)
    if args.cmd == "rescore":
        return bench.rescore_record(args.record, d)
    bench.run_record(args.record, d, device=args.device, seed=args.seed, only=args.point,
                     force=args.force)


if __name__ == "__main__":
    main()

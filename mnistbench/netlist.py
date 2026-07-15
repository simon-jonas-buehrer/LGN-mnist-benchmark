"""Read the synthesized NAND netlist back in and run it on the test set. Pure numpy.

This is the y-axis: a submission reports a circuit, not an accuracy, and the accuracy is measured
on that circuit. A straight-through estimator that disagrees with its hard forward pass, a soft
surrogate used at eval, or a float leaking from the head does not survive here, because none of
it exists in the netlist.

The netlist is yosys JSON with only $_NAND_ and $_NOT_ cells (NOT(a) = NAND(a,a), so one kernel
covers both). Gates are grouped into levels by longest path, and 64 images are packed into each
uint64 word, so `~(a & b)` evaluates a whole level on 64 images in one instruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import N_CLASSES, N_PIXELS, PIXEL_BITS

CONST0, CONST1 = 0, 1  # signal ids for the constant nets yosys may leave behind
WORD = 64


@dataclass
class NandNet:
    """A levelled NAND-only circuit. Signals: [const0, const1, inputs..., gates...]."""

    n_in: int
    n_sig: int
    src_a: list[np.ndarray]  # per level, source signal ids
    src_b: list[np.ndarray]
    offs: list[int]          # level l writes signals [offs[l], offs[l+1])
    out_sig: list[int]       # signal id of each output bit, LSB first

    @property
    def n_gates(self) -> int:
        return self.offs[-1] - self.offs[0]

    @property
    def depth(self) -> int:
        return len(self.src_a)


def from_json(nl: dict, top: str = "top") -> NandNet:
    mod = nl["modules"][top]

    in_bits, out_bits = [], []
    for port in mod["ports"].values():
        (in_bits if port["direction"] == "input" else out_bits).extend(port["bits"])
    if len(in_bits) != N_PIXELS * PIXEL_BITS:
        raise RuntimeError(f"top has {len(in_bits)} input bits, expected {N_PIXELS * PIXEL_BITS}")

    sig: dict = {"0": CONST0, "1": CONST1, "x": CONST0, "z": CONST0}
    for i, b in enumerate(in_bits):
        sig[b] = 2 + i
    n_in = len(in_bits)

    cells = []  # (y, a, b); an inverter is NAND(a, a)
    for name, c in mod["cells"].items():
        conn = c["connections"]
        if c["type"] == "$_NAND_":
            cells.append((conn["Y"][0], conn["A"][0], conn["B"][0]))
        elif c["type"] == "$_NOT_":
            cells.append((conn["Y"][0], conn["A"][0], conn["A"][0]))
        else:
            raise RuntimeError(f"cell {name} is {c['type']}; the netlist must be NAND-only")

    driver = {y: i for i, (y, _, _) in enumerate(cells)}
    if len(driver) != len(cells):
        raise RuntimeError("a net is driven by two gates -- the simulation would be ambiguous")

    # longest-path level of every gate (iterative, so a deep adder tree cannot blow the stack)
    level, state = [0] * len(cells), [0] * len(cells)
    for root in range(len(cells)):
        if state[root]:
            continue
        stack = [(root, False)]
        while stack:
            g, done = stack.pop()
            if done:
                lv = 0
                for net in cells[g][1:]:
                    src = driver.get(net)
                    if src is not None:
                        lv = max(lv, level[src] + 1)
                level[g], state[g] = lv, 2
            elif state[g] == 0:
                state[g] = 1
                stack.append((g, True))
                for net in cells[g][1:]:
                    src = driver.get(net)
                    if src is not None and state[src] == 0:
                        stack.append((src, False))
            elif state[g] == 1:
                raise RuntimeError("combinational loop in the netlist")

    depth = max(level) + 1 if cells else 0
    buckets: list[list[int]] = [[] for _ in range(depth)]
    for g, lv in enumerate(level):
        buckets[lv].append(g)

    # ids level by level, so every source has a strictly smaller id than its gate
    offs, nxt = [2 + n_in], 2 + n_in
    for lv in range(depth):
        for g in buckets[lv]:
            sig[cells[g][0]] = nxt
            nxt += 1
        offs.append(nxt)

    def sid(net):
        if net not in sig:
            raise RuntimeError(f"net {net!r} is read but never driven")
        return sig[net]

    src_a = [np.array([sid(cells[g][1]) for g in b], dtype=np.int64) for b in buckets]
    src_b = [np.array([sid(cells[g][2]) for g in b], dtype=np.int64) for b in buckets]
    return NandNet(n_in, nxt, src_a, src_b, offs, [sid(b) for b in out_bits])


def run(net: NandNet, x_bits: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Evaluate the circuit. x_bits is (N, n_in) of 0/1; returns (N,) predicted classes."""
    preds = []
    for i in range(0, len(x_bits), chunk):
        xb = x_bits[i : i + chunk]
        n = len(xb)
        pad = (-n) % WORD  # pack() needs whole words; the padding images are simply ignored
        if pad:
            xb = np.concatenate([xb, np.zeros((pad, xb.shape[1]), np.uint8)])

        # (n_in, n_words) uint64: one bit per image, 64 images per word
        packed = np.packbits(np.ascontiguousarray(xb.T), axis=1, bitorder="little").view(np.uint64)

        acts = np.zeros((net.n_sig, packed.shape[1]), dtype=np.uint64)
        acts[CONST1] = np.uint64(0xFFFFFFFFFFFFFFFF)
        acts[2 : 2 + net.n_in] = packed
        for lv, (a, b) in enumerate(zip(net.src_a, net.src_b)):
            acts[net.offs[lv] : net.offs[lv + 1]] = ~(acts[a] & acts[b])

        # unpack the 4 output bits back to one class index per image
        out = np.unpackbits(acts[net.out_sig].view(np.uint8), axis=1, bitorder="little")
        cls = (out.astype(np.int64) << np.arange(len(net.out_sig), dtype=np.int64)[:, None]).sum(0)
        preds.append(cls[:n])

    pred = np.concatenate(preds)
    if (pred >= N_CLASSES).any():
        raise RuntimeError("circuit produced a class index >= 10 -- its argmax head is broken")
    return pred

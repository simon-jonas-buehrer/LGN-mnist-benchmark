"""Verilog emitters: turn a trained logic network into the benchmark's top module.

Every submission is judged as a circuit, and the circuit has a fixed interface:

    module top (input [6271:0] pix, output [3:0] cls);

`pix[8*p +: 8]` is the uint8 value of pixel p (row-major, 0..783), and `cls` is the predicted
digit, 0..9. EVERYTHING between those two ports is yours and is counted: the binarizer, the
learned logic, the readout head, the argmax. There is no free preprocessing and no free
softmax, which is the whole point -- it is the only way two different architectures can be
compared on one axis.

The pieces below are the reusable ones. Most logic networks are "a fan-in-2 gate net with a
popcount head", so `emit_lutnet()` covers them end to end:

    thermometer thresholds -> encoder bits -> L layers of 2-input LUTs -> group popcount -> argmax

A 2-input LUT *is* the general case of a 2-input gate: `tt` is a 4-bit truth table, bit (2a+b)
holding f(a, b). NAND is tt=7, XOR is tt=6, AND is tt=8. So a NAND-only network (the genetic
record) and a learned-truth-table network (the backprop record) go through the same emitter
and are synthesized by exactly the same flow -- neither gets an advantage from how its Verilog
happens to be written.

If your model is not a fan-in-2 net, write your own Verilog and reuse `emit_popcount_argmax()`
for the head. The only hard requirement is the module signature above.
"""

from __future__ import annotations

from typing import Sequence

from .data import N_CLASSES, N_PIXELS, PIXEL_BITS

# The 16 boolean functions of two inputs, as Verilog expressions.
# Truth table `tt` is a 4-bit int: bit (2*a + b) is f(a, b). So bit0=f(0,0) ... bit3=f(1,1).
_LUT2 = {
    0b0000: "1'b0",
    0b0001: "(~{a} & ~{b})",  # NOR
    0b0010: "(~{a} & {b})",
    0b0011: "~{a}",
    0b0100: "({a} & ~{b})",
    0b0101: "~{b}",
    0b0110: "({a} ^ {b})",  # XOR
    0b0111: "~({a} & {b})",  # NAND
    0b1000: "({a} & {b})",  # AND
    0b1001: "~({a} ^ {b})",  # XNOR
    0b1010: "{b}",
    0b1011: "(~{a} | {b})",
    0b1100: "{a}",
    0b1101: "({a} | ~{b})",
    0b1110: "({a} | {b})",  # OR
    0b1111: "1'b1",
}


def lut2_expr(tt: int, a: str, b: str) -> str:
    """Verilog for the 2-input boolean function `tt` applied to signals a, b."""
    return _LUT2[int(tt) & 0xF].format(a=a, b=b)


def even_thresholds(bits: int) -> list[int]:
    """`bits` evenly spaced thermometer thresholds on a uint8 pixel.

    They land on 2^k-1 boundaries, which is worth knowing: `pix > 127` is bit 7 of the byte, a
    WIRE, and costs zero gates; `pix > 63` costs two. Picking thresholds that are cheap in
    silicon is exactly the pressure this benchmark is meant to create.
    """
    return [round(256 * (j + 1) / (bits + 1)) - 1 for j in range(bits)]


def emit_thermometer(thresholds: Sequence[int], sig: str = "s") -> tuple[str, int]:
    """Encoder: each pixel becomes len(thresholds) bits, `pix[p] > t`.

    Returns (verilog body, n_bits). Bit layout is pixel-major: encoder bit `p*k + j` is
    `pix[p] > thresholds[j]`, so it lands in sig[p*k + j]. These occupy the first n_bits
    signals; the first LUT layer indexes into them.

    A comparison against a constant is a handful of gates, but it is *not* free, and it is in
    your gate count -- a 7-bit thermometer costs real area. That is the intended trade-off.
    """
    k = len(thresholds)
    lines = [f"  // thermometer encoder: {N_PIXELS} pixels x {k} thresholds = {N_PIXELS * k} bits"]
    for p in range(N_PIXELS):
        for j, t in enumerate(thresholds):
            t = int(t)
            if not 0 <= t <= 254:
                raise ValueError(f"threshold {t} is degenerate (a uint8 pixel is never > 255)")
            lines.append(f"  assign {sig}[{p * k + j}] = pix[{p * PIXEL_BITS} +: {PIXEL_BITS}] > 8'd{t};")
    return "\n".join(lines), N_PIXELS * k


def emit_popcount_argmax(bit_names: Sequence[str], n_classes: int = N_CLASSES) -> str:
    """Readout head: split bits into n_classes contiguous groups, popcount, argmax.

    Group c owns bits [c*g, (c+1)*g). Ties go to the lowest class index, which is also what
    torch.argmax does, so a python model and its circuit agree bit for bit.
    """
    n = len(bit_names)
    if n % n_classes != 0:
        raise ValueError(f"{n} readout bits not divisible by {n_classes} classes")
    g = n // n_classes
    w = max(1, int(g).bit_length())  # counts run 0..g

    lines = [f"  // readout: {n_classes} groups x {g} bits -> popcount -> argmax (ties: lowest class)"]
    lines.append(f"  logic [{w - 1}:0] cnt [0:{n_classes - 1}];")
    for c in range(n_classes):
        terms = " + ".join(bit_names[c * g : (c + 1) * g])
        lines.append(f"  assign cnt[{c}] = {terms};")

    lines.append(f"  logic [{w - 1}:0] best;")
    lines.append("  always_comb begin")
    lines.append("    best = cnt[0];")
    lines.append("    cls  = 4'd0;")
    for c in range(1, n_classes):
        lines.append(f"    if (cnt[{c}] > best) begin best = cnt[{c}]; cls = 4'd{c}; end")
    lines.append("  end")
    return "\n".join(lines)


def emit_lutnet(
    thresholds: Sequence[int],
    layers: Sequence[tuple[Sequence[int], Sequence[int], Sequence[int]]],
    *,
    n_classes: int = N_CLASSES,
    top: str = "top",
) -> str:
    """The whole circuit for a fan-in-2 logic net.

    layers is a list of (idx_a, idx_b, tt), one entry per layer:
      idx_a[i], idx_b[i]  signal ids the i-th gate of this layer reads
      tt[i]               its 4-bit truth table (see lut2_expr)

    Signal ids are global and assigned in order: the encoder owns 0..n_in-1, then layer 0's
    gates, then layer 1's, and so on. A gate may read ANY earlier signal -- that keeps the
    graph acyclic by construction, and it is checked here rather than discovered as a
    combinational loop deep inside yosys.

    The last layer is the readout; its width must be divisible by n_classes.
    """
    enc, n_in = emit_thermometer(thresholds)

    body = [enc]
    off = n_in
    offs = [n_in]
    for li, (idx_a, idx_b, tt) in enumerate(layers):
        w = len(idx_a)
        if not (len(idx_b) == len(tt) == w):
            raise ValueError(f"layer {li}: idx_a/idx_b/tt length mismatch")
        body.append(f"  // layer {li}: {w} gates, sources < {off}")
        for i in range(w):
            a, b = int(idx_a[i]), int(idx_b[i])
            if not (0 <= a < off and 0 <= b < off):
                raise ValueError(
                    f"layer {li} gate {i} reads signal {a}/{b}, but only 0..{off - 1} exist yet "
                    "(a gate may only read strictly earlier signals)"
                )
            body.append(f"  assign s[{off + i}] = {lut2_expr(tt[i], f's[{a}]', f's[{b}]')};")
        off += w
        offs.append(off)

    last_w = offs[-1] - offs[-2]
    head = emit_popcount_argmax([f"s[{i}]" for i in range(offs[-2], offs[-1])], n_classes)

    n_gates = off - n_in
    return f"""// generated by mnistbench.hw -- {len(layers)} layers, {n_gates} fan-in-2 gates,
// {len(thresholds)} thermometer bits/pixel, readout width {last_w}
module {top} (input [{N_PIXELS * PIXEL_BITS - 1}:0] pix, output logic [3:0] cls);
  wire [{off - 1}:0] s;

{chr(10).join(body)}

{head}
endmodule
"""

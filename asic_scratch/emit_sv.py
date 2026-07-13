"""Emit one conv kernel as combinational SystemVerilog with the weights baked in.

The unit of hardware here is the KERNEL, not the whole feature map. A conv layer applies the
same kernel at every spatial position, so the distinct logic of a layer is one kernel instance
per output channel -- spatially unrolling would just replicate it H*W times.

A kernel is an int8 dot product over C_in*K*K taps:

    acc = sum_i w_i * x_i + bias        (w_i known at synthesis time)
    y   = relu(requantize(acc))         (requant = multiply by M, shift right)

Because every w_i is a constant, no multiplier is instantiated. `w * x` becomes a shift-and-add
of the set bits of w, and the synthesizer folds the whole thing into one adder tree. That is the
"bake the weights into the gates" step: the weights stop being data and become topology.
"""

from __future__ import annotations

from dataclasses import dataclass


def slit(value: int, bits: int) -> str:
    """A signed Verilog literal. The sign must sit OUTSIDE the base: `-22'sd5`, never `22'sd-5`."""
    return f"{bits}'sd{value}" if value >= 0 else f"(-{bits}'sd{-value})"


@dataclass
class Kernel:
    """One output channel of one conv layer, fully specified as integers."""

    weights: list[int]  # int8, length C_in*K*K
    bias: int  # int32, folded from conv bias + BN
    mult: int  # requant multiplier (int32)
    shift: int  # requant right-shift
    relu: bool = True

    @property
    def n_taps(self) -> int:
        return len(self.weights)


def emit_kernel(k: Kernel, name: str = "kernel", in_bits: int = 8, out_bits: int = 8) -> str:
    """Return SystemVerilog for a single kernel, weights folded in as constants."""
    n = k.n_taps
    # accumulator must not overflow: int8 * int8 = 16 bits, plus log2(n) for the sum, plus bias
    acc_bits = 16 + max(1, (n - 1).bit_length()) + 1

    ports = ", ".join(f"input logic signed [{in_bits-1}:0] x{i}" for i in range(n))

    # weights become constants -> each product is a shift-add, not a multiplier
    terms = [f"{slit(w, acc_bits)}*x{i}" for i, w in enumerate(k.weights) if w != 0]
    # a zero weight costs nothing: the tap simply disappears from the circuit
    sum_expr = " + ".join(terms) if terms else f"{acc_bits}'sd0"

    # requantize: (acc * mult) >>> shift, then saturate to out_bits (and relu)
    wide = acc_bits + 32
    lo, hi = (0, (1 << out_bits) - 1) if k.relu else (-(1 << (out_bits - 1)), (1 << (out_bits - 1)) - 1)

    return f"""// {name}: {n} taps, {sum(1 for w in k.weights if w != 0)} nonzero, weights folded in
module {name}({ports}, output logic signed [{out_bits-1}:0] y);
  logic signed [{acc_bits-1}:0] acc;
  logic signed [{wide-1}:0]     scaled;

  assign acc    = {sum_expr} + {slit(k.bias, acc_bits)};
  assign scaled = (acc * {slit(k.mult, wide)}) >>> {k.shift};

  always_comb begin
    if (scaled < {slit(lo, wide)})      y = {slit(lo, out_bits)};
    else if (scaled > {slit(hi, wide)}) y = {slit(hi, out_bits)};
    else                                y = scaled[{out_bits-1}:0];
  end
endmodule
"""


def emit_generic_kernel(n: int, name: str = "kernel_generic", in_bits: int = 8) -> str:
    """Baseline for comparison: the SAME dot product but with weights as runtime INPUTS.

    This is what the circuit costs if you *don't* fold the weights in -- you must instantiate
    n real int8*int8 multipliers. Synthesizing this next to emit_kernel() is what quantifies
    the weight-folding win.
    """
    xs = ", ".join(f"input logic signed [{in_bits-1}:0] x{i}" for i in range(n))
    ws = ", ".join(f"input logic signed [{in_bits-1}:0] w{i}" for i in range(n))
    acc_bits = 16 + max(1, (n - 1).bit_length())
    terms = " + ".join(f"x{i}*w{i}" for i in range(n))
    return f"""module {name}({xs}, {ws}, output logic signed [{acc_bits-1}:0] y);
  assign y = {terms};
endmodule
"""

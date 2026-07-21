"""The circuit: thermometer bits in, 2-input LUT layers, popcount readout.

Every gate is a 2-input lookup table with its own 4-bit truth table, and each of its two ports
names a source. With `codebook=K` a port stores one of K seeded candidates (log2(K) bits) instead
of a full index; that is the wiring scheme both records search over.

The forward pass is bit-packed: 8 samples ride in one uint8 lane, so a whole population is scored
with bitwise ops. `_forward` is the function the emitted Verilog reproduces gate for gate.

Carved verbatim out of `ga_bits_wiring_mnist.py` in the record this came from -- the training
scripts, the CIFAR loader and the standalone CLI are not part of the measured path and were left
behind.
"""

from __future__ import annotations

import math
from itertools import accumulate

import jax
import jax.numpy as jnp

CODEBOOK_SEED = 12345  # structural: the K candidate wirings are regenerated from this at load


def pack(X: jnp.ndarray) -> jnp.ndarray:
    """(N, features) 0/1 -> (features, N/8): samples packed 8-per-byte along the bit axis."""
    assert X.shape[0] % 8 == 0, "packed forward needs N divisible by 8"
    return jnp.packbits(X, axis=0).T


class Net:
    """Fixed layer *shape*; per-gate truth tables AND their two input wires are the genome.

    Two wiring modes:
      - `codebook == 0` (free): each gate's wire is any source index in its layer -> must be stored
        at ceil(log2(prev_width)) bits each (~13), the expensive but maximally free scheme.
      - `codebook == K` (K > 0): K candidate wirings are generated from a FIXED structural seed (so
        they are regenerated at load and cost 0 bytes); each gate stores only a small *choice* index
        into them, at ceil(log2(K)) bits per wire. Per-gate optimisation survives, at a fraction of
        the storage — the middle ground between fixed random wiring and fully-free wiring.
    """

    def __init__(
        self, n_in: int, widths: list[int], classes: int, codebook: int = 0, dag: bool = False
    ):
        self.n_in, self.widths, self.classes = n_in, widths, classes
        self.n_gates = sum(widths)
        self.n_bits = self.n_gates * 4  # truth-table bits only (wiring stored as int indices)
        self.offs = [0, *accumulate(widths)]
        self.codebook, self.dag = codebook, dag
        assert widths[-1] % classes == 0, "final width must divide #classes"
        # How far back may a gate reach?
        #   layered (default): only the previous layer -- the shape is a funnel and stays one.
        #   dag: ANY strictly earlier signal (the inputs, plus every gate in every earlier layer),
        #        which is the search space a Cartesian-GP / graph encoding gives you. Acyclic by
        #        construction, since a gate can only ever read something already computed.
        self.src_limits = (
            [n_in + self.offs[k] for k in range(len(widths))] if dag else [n_in, *widths[:-1]]
        )
        self.srcmax = jnp.array(
            [lim for lim, w in zip(self.src_limits, widths) for _ in range(w)], jnp.int32
        )  # (n_gates,)
        self.gate_ar = jnp.arange(self.n_gates)
        if codebook:
            # K structural candidate wirings (regenerated at load from CODEBOOK_SEED -> 0 bytes)
            k = jax.random.PRNGKey(CODEBOOK_SEED)
            ka, kb = jax.random.split(k)
            shape = (codebook, self.n_gates)
            self.cand_a = (jax.random.uniform(ka, shape) * self.srcmax).astype(jnp.int32)
            self.cand_b = (jax.random.uniform(kb, shape) * self.srcmax).astype(jnp.int32)
        # the genome's wire values live in [0, wire_max): a source index (free) or a choice (codebook)
        self.wire_max = codebook if codebook else self.srcmax
        self.eval_pop = jax.jit(
            self._eval_pop
        )  # (codes, wa, wb, Xp, y) -> (margin, acc), vmapped over pop

    def init_pop(self, key, pop: int):
        """Random genome: tables ~ Bernoulli(0.5) bits; each wire ~ Uniform over its allowed range."""
        kt, ka, kb = jax.random.split(key, 3)
        tables = jax.random.bernoulli(kt, 0.5, (pop, self.n_bits)).astype(jnp.uint8)
        wa = (jax.random.uniform(ka, (pop, self.n_gates)) * self.wire_max).astype(jnp.int32)
        wb = (jax.random.uniform(kb, (pop, self.n_gates)) * self.wire_max).astype(jnp.int32)
        return tables, wa, wb

    def sources(self, wa: jnp.ndarray, wb: jnp.ndarray):
        """Resolve one genome's wire values to actual source indices (identity unless codebook)."""
        if not self.codebook:
            return wa, wb
        return self.cand_a[wa, self.gate_ar], self.cand_b[wb, self.gate_ar]

    def codes(self, tables: jnp.ndarray) -> jnp.ndarray:
        g = tables.reshape(tables.shape[0], self.n_gates, 4).astype(jnp.uint8)
        return (g[..., 0] << 3) | (g[..., 1] << 2) | (g[..., 2] << 1) | g[..., 3]  # (P, n_gates)

    def model_memory_bytes(self) -> int:
        """Bytes of the deployable circuit: 4-bit truth tables + bit-packed wiring, 1 bit/element.

        Free wiring is learned and not regenerable, so each of a gate's two source indices costs
        ceil(log2(prev_layer_width)) bits. Codebook wiring stores only a choice among K structural
        candidates -> ceil(log2(K)) bits per wire, with the candidates themselves costing nothing.
        """
        table_bits = self.n_gates * 4
        if self.codebook:
            wire_bits = self.n_gates * 2 * max(1, math.ceil(math.log2(self.codebook)))
        else:
            # a free wire costs log2(how many sources it could have named) -- which is wider under
            # dag, so reaching further back is not free unless the codebook pays for it
            wire_bits = sum(
                w * 2 * max(1, math.ceil(math.log2(lim)))
                for lim, w in zip(self.src_limits, self.widths)
            )
        return math.ceil((table_bits + wire_bits) / 8)

    def _forward(
        self, codes: jnp.ndarray, wa: jnp.ndarray, wb: jnp.ndarray, Xp: jnp.ndarray
    ) -> jnp.ndarray:
        """One genome. codes/wa/wb (n_gates,), Xp (features, n_words) -> class logits (classes, N).

        Layered mode keeps only the previous layer alive, which is all a funnel can read. Dag mode
        keeps every signal computed so far, because any of them may be named as a source -- so the
        buffer grows to (n_in + n_gates, n_words) instead of one layer's worth.
        """
        sa, sb = self.sources(wa, wb)  # genome wire values -> actual source indices
        sig = prev = Xp
        for k in range(len(self.widths)):
            lo, hi = self.offs[k], self.offs[k + 1]
            c = codes[lo:hi, None]  # (w, 1)
            src = sig if self.dag else prev
            a, b = src[sa[lo:hi]], src[sb[lo:hi]]
            na, nb = ~a, ~b
            prev = (
                ((na & nb) * (c & 1))
                | ((na & b) * ((c >> 1) & 1))
                | ((a & nb) * ((c >> 2) & 1))
                | ((a & b) * ((c >> 3) & 1))
            )
            if self.dag:
                sig = jnp.concatenate([sig, prev], 0)
        bits = jnp.unpackbits(prev, axis=1)  # (w_out, N)
        return bits.reshape(self.classes, -1, bits.shape[1]).sum(1)  # (classes, N)

    def _eval_pop(self, codes, wa, wb, Xp, y):
        fwd = jax.vmap(self._forward, in_axes=(0, 0, 0, None))
        logits = fwd(codes, wa, wb, Xp).astype(jnp.int32)  # (P,C,N)
        ar = jnp.arange(y.shape[0])
        correct = logits[:, y, ar]  # (P, N) true-class popcount
        masked = logits.at[:, y, ar].set(-1).max(1)  # best distractor per sample
        margin = (correct - masked).mean(1)  # (P,) smooth selection signal
        acc = (logits.argmax(1) == y).mean(1)  # (P,)
        return margin, acc

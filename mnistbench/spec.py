"""The submission contract. This is the whole API.

A record is a directory `records/<you>/<method>/` with a `submission.py` that exposes:

    POINTS: list[dict]                 # one dict per point on YOUR curve (e.g. model sizes)
    def build(**point) -> Submission    # turn one of those dicts into a model

Everything crossing this boundary is numpy, so the model inside can be torch, JAX, TensorFlow
or a pile of bit-twiddling -- the harness never imports your framework, and you never import
its.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .data import Mnist


class Submission(ABC):
    """One point on one curve: a model, plus the optimizer that trained it."""

    @abstractmethod
    def train(self, data: Mnist, *, device: str = "cpu", seed: int = 0) -> None:
        """Fit the model. Use data.train_* and data.val_*; data.test_* is off limits."""

    @abstractmethod
    def emit_verilog(self) -> str:
        """The trained model as one Verilog source defining:

            module top (input [6271:0] pix, output [3:0] cls);

        pix[8*p +: 8] is pixel p (uint8, row-major); cls is the predicted digit. Everything in
        between is counted. mnistbench.hw emits the usual pieces; see records/sbuehrer/*.
        """

    @abstractmethod
    def predict(self, pix: np.ndarray) -> np.ndarray:
        """(N, 784) uint8 -> (N,) predicted classes.

        Must be the EXACT function emit_verilog() describes. The harness checks this on 512
        test images and rejects the point if they differ on even one of them -- not to police
        you, but because the check catches the bug that otherwise silently costs you accuracy.
        """

    def scores(self, pix: np.ndarray) -> np.ndarray | None:
        """(N, 784) uint8 -> (N, 10) per-class scores in [0, 1], or None if you don't provide them.

        A gate circuit outputs a hard class, so there is no probability to take a cross-entropy of
        -- except that the readout, just before the argmax, holds one integer per class: how many
        of that class's gates fired. Divided by the group size that is a firing FRACTION in [0, 1],
        and its argmax is exactly `predict()` (the groups are equal size), so it is the circuit's
        own signal, not an invented one. The harness softmaxes these into the cross-entropy on the
        y-axis. Return None to sit out that axis (you still get accuracy).
        """
        return None

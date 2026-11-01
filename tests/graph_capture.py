# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""CUDA-graph capture for the tests that record a park's ticks in a graph and replay them against eager ticks.

- `capture(fn, pool, warm=None)`: `fn()` recorded in a CUDA graph on a side stream (torch.cuda.graph's recipe
  without its release of the cached memory). `warm()`, if given, runs first on that stream outside the capture. A
  capture that fails is ended, the allocator's recording into the pool ended with it (else every later capture into
  that pool fails), and its error raised.
- `Pins(objs)`: the tensors a tick may rebind (the objects' tensor attributes), each held at one fixed tensor. After
  a tick, `pin()` copies a rebound tensor back into its fixed tensor (its values unchanged) and binds the fixed one
  again, so every replay reads and writes the same memory.
"""

from __future__ import annotations

import threading

import torch

CAPTURE_MODE = "global"                 # torch's capture_error_mode
_SIDE: dict = {}
CAPTURE_LOCK = threading.RLock()        # one capture at a time in the process


class Unsupported(RuntimeError):
    """A tensor attribute rebound to another shape, type or device, or a new one: the recorded tick no longer fits."""


def capture(fn, pool, warm=None) -> torch.cuda.CUDAGraph:
    """fn() recorded in a CUDA graph into `pool` on a side stream; the graph."""
    dev = torch.cuda.current_device()
    key = (dev, threading.get_ident())
    side = _SIDE.get(key)
    if side is None:
        side = _SIDE[key] = torch.cuda.Stream()
    with CAPTURE_LOCK:
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            if warm is not None:
                warm()
                side.synchronize()
            g.capture_begin(pool=pool, capture_error_mode=CAPTURE_MODE)
            try:
                fn()
            except BaseException:
                try:
                    g.capture_end()
                except Exception:
                    pass
                try:
                    torch._C._cuda_endAllocateToPool(dev, pool)
                except Exception:
                    pass
                raise
            g.capture_end()
        torch.cuda.current_stream().wait_stream(side)
    return g


class Pins:
    """The objects' tensor attributes, each held at one fixed tensor."""

    def __init__(self, objs: list):
        self.objs = objs
        self.fixed = [(o, k, v) for o in objs for k, v in vars(o).items() if torch.is_tensor(v)]

    def pin(self, strict: bool = True) -> None:
        """Every rebound tensor copied back into its fixed tensor and bound to it again; a tensor attribute rebound to
        another shape, type or device, or (strict) a new one, is refused (Unsupported)."""
        seen = set()
        for o, k, old in self.fixed:
            new = getattr(o, k)
            seen.add((id(o), k))
            if new is old:
                continue
            if not torch.is_tensor(new) or new.shape != old.shape or new.dtype != old.dtype or new.device != old.device:
                raise Unsupported(f"{type(o).__name__}.{k} rebound to another shape or type")
            old.copy_(new)
            setattr(o, k, old)
        if strict:
            for o in self.objs:
                for k, v in vars(o).items():
                    if torch.is_tensor(v) and (id(o), k) not in seen:
                        raise Unsupported(f"{type(o).__name__}.{k}: a new tensor attribute")

    def unbind(self) -> None:
        """Every tensor bound back to its fixed tensor without a copy (after a failed capture, which ran nothing)."""
        for o, k, old in self.fixed:
            setattr(o, k, old)

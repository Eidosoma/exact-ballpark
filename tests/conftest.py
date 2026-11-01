# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The test suite's configuration.

**torch before numpy.** A process that imports numpy before torch can bind other native maths routines (the BLAS),
so torch is imported here first, for every run: the whole suite or a single file.

**The subsets** (`pytest -m cpu`, `-m oracle`, `-m gpu`; README.md). Each test is given exactly one, by these rules
in this order:
- `gpu`: the test is skipped where no CUDA GPU is visible (a `skipif` whose reason names CUDA or a GPU);
- `oracle`: the test runs destiny's own code, compiled by the conformance kit (a `skipif` whose reason names a
  harness or g++, a module named `*_oracle`, or a test whose name says it runs the compiled code);
- `cpu`: every other test.
A test of one subset may still skip for another reason (no `nvcc`, another C library); the skip says why.
"""

from __future__ import annotations

import re

import torch  # noqa: F401  (first, before any test module imports numpy)

GPU = re.compile(r"CUDA|GPU")
ORACLE_REASON = re.compile(r"harness|g\+\+")
ORACLE_NAME = re.compile(r"destiny|compiled|harness|oracle", re.I)
SUBSETS = ("cpu", "oracle", "gpu")


def pytest_configure(config):
    config.addinivalue_line("markers", "cpu: the reference and the batched parks on the CPU")
    config.addinivalue_line("markers", "oracle: destiny compiled from source against the reference (the conformance kit)")
    config.addinivalue_line("markers", "gpu: the CUDA kernels against the batched parks (a CUDA GPU)")


def subset(item) -> str:
    """The item's subset by the rules above."""
    reasons = [str(m.kwargs.get("reason", "")) for m in item.iter_markers(name="skipif")]
    if any(GPU.search(r) for r in reasons):
        return "gpu"
    module = item.module.__name__ if getattr(item, "module", None) is not None else ""
    name = getattr(item, "originalname", None) or item.name
    if (any(ORACLE_REASON.search(r) for r in reasons) or module.endswith("_oracle") or ORACLE_NAME.search(name)):
        return "oracle"
    return "cpu"


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        if not any(item.get_closest_marker(s) for s in SUBSETS):
            item.add_marker(getattr(pytest.mark, subset(item)))

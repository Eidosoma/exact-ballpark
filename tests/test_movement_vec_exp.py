# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The C library's exp on tensors (exact_ballpark.vec_exp), which the batched collision methods need for Destiny's
integration over part of a tick: bit for bit against math.exp (glibc 2.34 here, the function compiled Destiny calls),
on the CPU and (when one is visible) the GPU, over every path of the algorithm (the main path, |x| below 2^-54, the
special case from 512 with subnormal and near-overflow results, 1024 and beyond, infinities, NaN, every bit pattern);
its emulated fused multiply-add bit for bit against math.fma (exact ties, cancellations, a product's exact error,
random bit patterns, sums within a hair of a tie); the table generated from its definition against the words of the loaded library; and planted
faults: exp evaluated without its fusions, and the emulation without its rounding to odd or its product's error term,
each of which these arguments catch.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import math
import random
import struct
from pathlib import Path

import pytest
import torch

from exact_ballpark import vec_exp

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                                reason="no visible GPU"))]
LIBM_SHA256 = "b4fa1652bf3a7267d0fa4256fade78f0df2954b7b39ee842ec410e0e7b0f6c86"   # /lib64/libm.so.6, glibc 2.34
EXP_OFFSET, EXP_DATA_OFFSET = 0x30E70, 0x74298                                      # __exp and __exp_data (objdump)


def u64(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def f64(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def libm_exp(x: float) -> float:
    """math.exp, with the infinity C returns where Python raises OverflowError."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def same(a: float, b: float) -> bool:
    return u64(a) == u64(b) or (a != a and b != b)


def exp_args(seed: int, n: int) -> list:
    r = random.Random(seed)
    xs = [r.uniform(-20.0, 0.0) for _ in range(n // 4)]                  # Integrate's -k/m t over part of a tick
    xs += [-10 ** r.uniform(-20, 2) for _ in range(n // 8)]
    xs += [r.uniform(-745.2, 709.8) for _ in range(n // 8)]
    xs += [r.uniform(-746.0, -700.0) for _ in range(n // 8)]             # the special case: subnormal results
    xs += [r.uniform(700.0, 709.8) for _ in range(n // 16)]              # and near the overflow
    xs += [10 ** r.uniform(-30, 2) * r.choice([-1, 1]) for _ in range(n // 8)]
    xs += [f64(r.getrandbits(64)) for _ in range(n // 8)]                # every bit pattern: NaN, inf, subnormals
    return xs + [0.0, -0.0, math.inf, -math.inf, math.nan, 5e-324, -5e-324, 709.782712893384, 709.7827128933841,
                 -745.1332191019411, -745.1332191019412, -708.3964185322641, 512.0, -512.0, 1024.0, -1024.0,
                 2.0 ** -54, -2.0 ** -54, 2.0 ** -55, -1e-28, -1e-300]


def fma_triples(seed: int, n: int) -> list:
    r = random.Random(seed)
    out = []
    for _ in range(n):
        kind = r.randrange(7)
        a = r.choice([-1, 1]) * r.uniform(1, 2) * 2.0 ** r.randint(-200, 200)
        b = r.choice([-1, 1]) * r.uniform(1, 2) * 2.0 ** r.randint(-200, 200)
        if kind == 0:
            c = r.choice([-1, 1]) * r.uniform(1, 2) * 2.0 ** r.randint(-400, 400)
        elif kind == 1:                                                  # cancellation
            c = -(a * b) * (1 + r.choice([0.0, 2.0 ** -52, -2.0 ** -53, r.uniform(-1e-10, 1e-10)]))
        elif kind == 2:                                                  # exact ties: ab = 2^e, c on the grid above
            e = r.randint(-60, 60)
            a, b = 2.0 ** (e - 30), 2.0 ** 30
            c = r.choice([-1, 1]) * (1 + r.randrange(1 << 52) * 2.0 ** -52) * 2.0 ** (e + 53)
        elif kind == 3:                                                  # the product's exact error
            c = -(a * b)
        elif kind == 4:                                                  # exp's own operand scales
            a = r.uniform(-0.003, 0.003)
            b = r.choice([vec_exp.C3, vec_exp.C5, r.uniform(-1e-5, 1e-5)])
            c = r.choice([vec_exp.C2, vec_exp.C4, r.uniform(-0.01, 0.01)])
        elif kind == 5:
            bits = lambda lo, hi: r.getrandbits(52) | (r.randint(lo, hi) << 52) | (r.getrandbits(1) << 63)
            a, b, c = f64(bits(900, 1150)), f64(bits(900, 1150)), f64(bits(800, 1250))
        else:                                                            # near ties: ab about half an ulp of c,
            c = r.choice([-1, 1]) * r.uniform(1, 2) * 2.0 ** r.randint(-50, 50)    # its low bits deciding
            a = r.uniform(1, 2)
            b = r.choice([-1, 1]) * math.ulp(c) / 2 / a * (1 + r.choice([-1, 1]) * r.uniform(0, 2.0 ** -50))
        out.append((a, b, c))
    return out


def t(xs, device):
    return torch.tensor(xs, dtype=torch.float64, device=device)


def exp_differences(device: str, xs: list) -> list:
    got = vec_exp.exp(t(xs, device)).cpu().tolist()
    return [(x, g, libm_exp(x)) for x, g in zip(xs, got) if not same(g, libm_exp(x))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [1, 2])
def test_exp_matches_the_c_library_bit_for_bit(device, seed):
    bad = exp_differences(device, exp_args(seed, 200_000))
    assert not bad, f"{len(bad)} arguments differ, first {bad[:3]}"


@pytest.mark.parametrize("device", DEVICES)
def test_fma_matches_math_fma_bit_for_bit(device):
    tr = fma_triples(3, 200_000)
    got = vec_exp.fma(*(t([x[i] for x in tr], device) for i in range(3))).cpu().tolist()
    bad = [(x, g) for x, g in zip(tr, got) if u64(g) != u64(math.fma(*x))]
    assert not bad, f"{len(bad)} triples differ, first {bad[:3]}"


def test_the_table_is_glibcs():
    """The 256 generated words hash as recorded; where this machine's libm is the recorded one, they equal the words
    of its loaded __exp_data, read at the offset its __exp loads them from."""
    assert hashlib.sha256(struct.pack("<256Q", *vec_exp.TABLE)).hexdigest() == vec_exp.TABLE_SHA256
    path = Path("/lib64/libm.so.6")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != LIBM_SHA256:
        pytest.skip("not the recorded libm")
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    base = ctypes.cast(libm.exp, ctypes.c_void_p).value - EXP_OFFSET
    data = ctypes.string_at(base + EXP_DATA_OFFSET, 112 + 256 * 8)
    head = struct.unpack("<8d", data[:64])
    assert head == (vec_exp.INVLN2N, float.fromhex("0x1.8p52"), vec_exp.NEGLN2HIN, vec_exp.NEGLN2LON, vec_exp.C2,
                    vec_exp.C3, vec_exp.C4, vec_exp.C5)
    assert struct.unpack("<256Q", data[112:]) == vec_exp.TABLE


def test_planted_fault_unfused_exp_is_caught(monkeypatch):
    """glibc's exp evaluated without its fusions differs from it: the fused multiply-add is needed."""
    monkeypatch.setattr(vec_exp, "fma", lambda a, b, c, b_split=None: a * b + c)
    bad = exp_differences("cpu", exp_args(1, 200_000))
    assert len(bad) > 20


@pytest.mark.parametrize("fault", ["no_round_to_odd", "no_product_error"])
def test_planted_faults_in_the_emulation_are_caught(monkeypatch, fault):
    if fault == "no_round_to_odd":
        monkeypatch.setattr(vec_exp, "add_odd", lambda a, b: a + b)
    else:
        real = vec_exp.two_product
        monkeypatch.setattr(vec_exp, "two_product", lambda a, b, b_split=None: (real(a, b, b_split)[0],
                                                                                torch.zeros_like(a)))
    tr = fma_triples(4, 100_000)
    got = vec_exp.fma(*(t([x[i] for x in tr], "cpu") for i in range(3))).tolist()
    assert sum(u64(g) != u64(math.fma(*x)) for x, g in zip(tr, got)) > 20

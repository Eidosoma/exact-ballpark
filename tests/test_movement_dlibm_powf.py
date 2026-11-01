# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The C library's powf transcribed (exact_ballpark.dlibm_powf) against the C library itself (glibc 2.34
aarch64's exported powf through ctypes: what Destiny's compiled mushroom growth calls, Ballpark.cpp:569, and what the
reference exact_ballpark.full calls), bit for bit (a NaN equal to a NaN), on arguments over every branch: y zero, infinite
or NaN with every kind of x; x zero, infinite or NaN with y's sign and parity; a negative finite x with y an odd or even
integer or not one; subnormal x; |y log2 x| past 126 to overflow, to underflow and to the subnormal edge with either
sign; random patterns; and the mushroom's own domain (x in [0, 1) at y = 0.25). The tables and constants are held to the
library's bytes when the library is the one they were read from, and the exp2 data to vec_libm's own reading of the
same table. Planted faults (table entries and a coefficient off in a high bit, frinta truncating, the overflow bound
halved, the subnormal's scaling dropped, the odd integer's sign lost) must each be caught. A fault in the double's last
bits (an unfused step, a table entry's last bit) is rarely seen: powf computes in double and rounds once to float, so
only a result within such an error of a float's rounding boundary can show it; the fused steps are the disassembly's
(dlibm_powf's generator checks each of the nine fmadd), and the device's unfused copy is measured against all 2^32 x at
y = 0.25 by a separate check. On the GPU the combat kernel's g_powf
(exact_ballpark.vec_combat_cuda.powf) is held to the C library on the same arguments and on signaling NaNs, which a
Python float cannot hold."""

from __future__ import annotations

import ctypes
import hashlib
import math
import random
import struct
from collections import Counter
from pathlib import Path

import pytest
import torch

from exact_ballpark import dlibm_powf as D
from exact_ballpark import vec_libm as VL

CUDA = torch.cuda.is_available()
LIBM = Path("/usr/lib64/libm.so.6")
_C = ctypes.CDLL("libm.so.6")
_C.powf.restype = ctypes.c_float
_C.powf.argtypes = [ctypes.c_float, ctypes.c_float]


def same(a: float, b: float) -> bool:
    return (a != a and b != b) or struct.pack("<f", a) == struct.pack("<f", b)


SPECIAL = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5, 0.25, -0.25, 1.5, -1.5, 127.0, -127.0, 128.0, 150.0,
           -150.0, 16777216.0, 16777218.0, -16777217.0 + 1.0, 8388609.0, -8388609.0, float("inf"), float("-inf"),
           float("nan"), D.ffrom(0x00000001), D.ffrom(0x807FFFFF), D.ffrom(0x00400000), D.ffrom(0x7F7FFFFF),
           D.ffrom(0xFF7FFFFF), D.ffrom(0x00800000), D.ffrom(0x33000000), D.ffrom(0x3F7FFFFF), D.ffrom(0x3F800001),
           1e-30, -1e-30, 1e30, 0.1, 10.0, -10.0]


def kind(x: float, y: float) -> str:
    """Which branch of powf the arguments take (the disassembly's order)."""
    ix, iy = D.fbits(x), D.fbits(y)
    zinf_y = ((((iy << 1) & D.M32) - 1) & D.M32) >= D.POWF_ZINF
    if ((ix - D.POWF_X_MIN) & D.M32) > D.POWF_X_SPAN:
        if zinf_y:
            return "y_special"
        if ((((ix << 1) & D.M32) - 1) & D.M32) >= D.POWF_ZINF:
            return "x_zero_inf_nan"
        if ix >> 31:
            return "x_negative"
        return "x_subnormal"
    if zinf_y:
        return "y_special_normal_x"
    ylogx = D.ffrom(iy) * D._log2_inline(ix)
    if ((D.dbits(ylogx) >> 47) & 0xFFFF) > D.POWF_BIG:
        if ylogx > D.POWF_OFLOW:
            return "overflow"
        if ylogx <= D.dfrom(D.POWF_UFLOW):
            return "underflow"
        if ylogx < D.dfrom(D.POWF_MAY_UFLOW):
            return "may_underflow"
        return "big_but_exp2"
    return "main"


def arguments(seed: int, n: int) -> list:
    """Every pairing of the special values, and n random pairs over every branch."""
    r = random.Random(seed)
    out = [(x, y) for x in SPECIAL for y in SPECIAL]
    for _ in range(n):
        c = r.randrange(8)
        x = D.ffrom(r.getrandbits(32))
        y = D.ffrom(r.getrandbits(32))
        if c == 1:
            y = r.choice(SPECIAL)
        elif c == 2:
            x = r.choice(SPECIAL)
        elif c == 3:
            x = D.ffrom(r.getrandbits(23) | (r.getrandbits(1) << 31))                  # subnormal or zero
        elif c == 4:
            x = -D.ffrom(r.getrandbits(31))                                               # negative x ...
            y = float(r.randint(-300, 300)) * r.choice([1.0, 1.0, 0.5])                   # ... integer y, or a half
        elif c == 5:                                                                      # near over- and underflow
            x = r.choice([D.ffrom(r.randrange(0x3F000000, 0x3FFFFFFF)), D.ffrom(r.randrange(0x00000001, 0x00800000))])
            y = D.f32(r.choice([1.0, -1.0]) * r.uniform(100.0, 3000.0))
        elif c == 6:
            x = D.f32(r.uniform(0.0, 1.0))
            y = 0.25
        elif c == 7:                                                                      # |y log2 x| in (149, 150)
            x = r.choice([0.5, 2.0, 0.25, 4.0, 0.125, 8.0])
            y = D.f32(r.uniform(149.0, 150.0) / -math.log2(x))
        out.append((x, y))
    return out


def test_the_tables_are_the_librarys_bytes():
    """Every table and constant read at the addresses the disassembly loads, when the library is the one read."""
    data = LIBM.read_bytes() if LIBM.exists() else b""
    if hashlib.sha256(data).hexdigest() != D.LIBM_SHA256:
        pytest.skip("this machine's libm.so.6 is not the library the constants were read from")
    rd = lambda a: struct.unpack_from("<d", data, a)[0]
    log2, exp2 = D.ADDRESSES["__powf_log2_data"], D.ADDRESSES["__exp2f_data"]
    n = len(D.POWF_LOG2_INVC)
    assert n == 1 << D.POWF_I_BITS
    assert D.POWF_LOG2_INVC == tuple(rd(log2 + 16 * i) for i in range(n))
    assert D.POWF_LOG2_LOGC == tuple(rd(log2 + 16 * i + 8) for i in range(n))
    assert D.POWF_A == tuple(rd(log2 + 256 + 8 * j) for j in range(5))
    assert D.POWF_EXP2_TAB == tuple(struct.unpack_from("<Q", data, exp2 + 8 * k)[0] for k in range(32))
    assert D.POWF_EXP2_C == tuple(rd(exp2 + 304 + 8 * j) for j in range(3))
    assert D.POWF_OFLOW == rd(0x70A80) and D.POWF_RND == rd(0x70A88)


def test_the_exp2_data_is_vec_libms():
    """powf's exp2 part takes expf's table and cubic (__exp2f_data.tab and .poly_scaled), as vec_libm read them."""
    assert D.POWF_EXP2_TAB == tuple(VL.EXP2F_TAB)
    assert D.POWF_EXP2_C == tuple(VL.EXPF_C)


def test_powf_is_the_c_librarys_on_every_branch():
    args = arguments(1, 150000)
    seen, bad = Counter(), []
    for x, y in args:
        seen[kind(x, y)] += 1
        if not same(_C.powf(x, y), D.powf(x, y)):
            bad.append((x.hex(), y.hex()))
    assert not bad, bad[:5]
    for k in ("y_special", "y_special_normal_x", "x_zero_inf_nan", "x_negative", "x_subnormal", "overflow",
              "underflow", "may_underflow", "big_but_exp2", "main"):
        assert seen[k] >= 20, (k, seen)


def test_powf_on_the_mushrooms_domain():
    """x in [0, 1) at y = 0.25 (Ballpark.cpp:569's one call): random patterns, and every 2^12-th pattern of the range."""
    r = random.Random(2)
    xs = [D.ffrom(r.randrange(0, 0x3F800000)) for _ in range(150000)] + [D.ffrom(u) for u in range(0, 0x3F800000, 4096)]
    bad = [x.hex() for x in xs if not same(_C.powf(x, 0.25), D.powf(x, 0.25))]
    assert not bad, bad[:5]


def _truncating(x):
    return float(math.trunc(x))


def _bit30(v: float) -> float:
    return D.dfrom(D.dbits(v) ^ (1 << 22))                  # the 30th bit of the significand


def _bit12(v: float) -> float:
    return D.dfrom(D.dbits(v) ^ (1 << 40))                  # the 12th bit of the significand


FAULTS = {
    "frinta truncating": ("_frinta", _truncating),
    "the subnormal's scaling dropped": ("POWF_TWO23_F", 0x3F800000),
    "the odd integer's sign lost": ("POWF_SIGN_BIAS", 0),
    "the overflow bound halved": ("POWF_OFLOW", D.POWF_OFLOW / 2),
    "the exp2 cubic's middle coefficient off in its 12th bit": ("POWF_EXP2_C", None),
    "an entry of the log2 table off in its 30th bit": ("POWF_LOG2_INVC", None),
    "an entry of the exp2 table off in its 12th bit": ("POWF_EXP2_TAB", None),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    name, value = FAULTS[fault]
    if name == "POWF_LOG2_INVC":
        t = list(D.POWF_LOG2_INVC)
        t[5] = _bit30(t[5])
        value = tuple(t)
    elif name == "POWF_EXP2_TAB":
        t = list(D.POWF_EXP2_TAB)
        t[7] ^= 1 << 40
        value = tuple(t)
    elif name == "POWF_EXP2_C":
        value = (D.POWF_EXP2_C[0], _bit12(D.POWF_EXP2_C[1]), D.POWF_EXP2_C[2])
    monkeypatch.setattr(D, name, value)
    args = arguments(3, 40000)
    assert any(not same(_C.powf(x, y), D.powf(x, y)) for x, y in args), f"the planted fault {fault} went unnoticed"


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
def test_the_device_powf_is_the_c_librarys():
    """g_powf on the GPU (vec_combat_cuda.powf) against the C library on the same arguments, signaling NaNs among them
    (passed and compared as bit patterns), 2^20 random pattern pairs and the mushroom's domain."""
    from exact_ballpark import vec_combat_cuda as KC

    def cbits(a: int, b: int) -> int:
        """The C library's powf of two bit patterns: c_float instances made over the bits pass unconverted (a
        signaling NaN stays one); the result, a float held in a Python float, back to its bits (a NaN stays a NaN)."""
        x = ctypes.c_float.from_buffer(ctypes.c_uint32(a))
        y = ctypes.c_float.from_buffer(ctypes.c_uint32(b))
        return D.fbits(_C.powf(x, y))
    args = [(D.fbits(x), D.fbits(y)) for x, y in arguments(4, 60000)]
    r = random.Random(5)
    args += [(r.getrandbits(32), r.getrandbits(32)) for _ in range(1 << 20)]
    args += [(D.fbits(D.ffrom(r.randrange(0, 0x3F800000))), D.fbits(0.25)) for _ in range(1 << 18)]
    snan = [0x7F800001, 0x7FA00000, 0xFFBFFFFF]
    args += [(s, 0) for s in snan] + [(0x3F800000, s) for s in snan] + [(s, 0x3F800000) for s in snan]
    xb = torch.tensor([a for a, _ in args], dtype=torch.int64)
    yb = torch.tensor([b for _, b in args], dtype=torch.int64)
    to32 = lambda v: torch.where(v >= 1 << 31, v - (1 << 32), v).to(torch.int32)
    got = KC.powf(to32(xb).cuda().view(torch.float32), to32(yb).cuda().view(torch.float32)).view(torch.int32).cpu()
    got = (got.to(torch.int64) & 0xFFFFFFFF).tolist()
    is_nan = lambda u: (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF) != 0
    bad = []
    for (a, b), g in zip(args, got):
        w = cbits(a, b)
        if w != g and not (is_nan(w) and is_nan(g)):
            bad.append((hex(a), hex(b), hex(w), hex(g)))
    assert not bad, bad[:5]

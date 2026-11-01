# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The C library's double asin and atan2 transcribed (exact_ballpark.dlibm_inv) against the C library itself (glibc 2.34
aarch64's exported asin and atan2, called through ctypes: what Destiny's compiled code and Python's math call), bit for
bit, on arguments over every branch: asin's tiny, Taylor and six table ranges (each bound from both sides), the Newton
step near 1, +-1, beyond 1, the infinities and NaN; atan2's special values (every pairing of +-0, +-inf, NaN and finite
values), the exponents' distance at its two bounds, the rescaling below 2^-500 and above 2^500, and its four quadrant
cases each with u below 1/16 and from the table, both orders of the two-sum. The tables and constants are held to the
library's bytes when the library is the one they were read from. Planted faults (every fused step unfused, an entry of
`asncs` and of `cij` wrong in its last bit, the low halves of pi/2 dropped, the sign of the result lost) must each be
caught. The device copies are held to these in tests/test_movement_vec_ypr_cuda.py."""

from __future__ import annotations

import ctypes
import hashlib
import math
import random
import struct
from pathlib import Path

import pytest

from exact_ballpark import dlibm_inv as I

LIBM = Path("/usr/lib64/libm.so.6")
_C = ctypes.CDLL("libm.so.6")
_C.asin.restype = _C.atan2.restype = ctypes.c_double
_C.asin.argtypes = [ctypes.c_double]
_C.atan2.argtypes = [ctypes.c_double, ctypes.c_double]


def d64(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & ((1 << 64) - 1)))[0]


def same(a: float, b: float) -> bool:
    return struct.pack("<d", a) == struct.pack("<d", b) or (a != a and b != b)


def asin_arguments(seed: int, n: int) -> list:
    """Arguments over every branch of asin, with every bound from both sides and the special values."""
    r = random.Random(seed)
    out = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.0 + 2 ** -52, -1.0 - 2 ** -52, 1 - 2 ** -53, -1 + 2 ** -53, 2.0, -3.5,
           5e-324, -5e-324, 2.2250738585072014e-308, 2 ** -1030, 1e300, math.inf, -math.inf, math.nan, -math.nan]
    for k in (I.ASIN_K_TINY, I.ASIN_K_TAYLOR, I.ASIN_K_QUARTER, I.ASIN_K_HALF, I.ASIN_K_075, I.ASIN_K_0921875,
              I.ASIN_K_0953125, I.ASIN_K_096875, I.ASIN_K_NEAR_ONE, I.ASIN_K_INF):
        for low in (0, 1, 0xFFFFFFFF):
            for hi in (k, k - 1):
                out += [d64((hi << 32) | low), -d64((hi << 32) | low)]
    for j in range(n):
        kind = j % 8
        if kind == 0:
            x = d64(r.getrandbits(64))
        elif kind == 1:
            x = r.uniform(-1.0, 1.0)
        elif kind == 2:
            x = r.choice([-1, 1]) * (1.0 - 10 ** r.uniform(-17, -1.5))
        elif kind == 3:
            x = r.choice([-1, 1]) * 10 ** r.uniform(-12, 0)
        elif kind == 4:
            x = d64(r.randrange(0x3FC0000000000000, 0x3FF0000000000001)) * r.choice([-1, 1])
        elif kind == 5:
            x = d64(r.randrange(0, 0x3FF0000000000001)) * r.choice([-1, 1])
        elif kind == 6:
            x = r.uniform(0.9, 1.0) * r.choice([-1, 1])
        else:
            x = math.sin(r.uniform(-4.0, 4.0))
        out.append(x)
    return out


def atan2_arguments(seed: int, n: int) -> list:
    """(y, x) pairs over every branch of atan2: the special values paired with each other and with finite values, the
    exponents' distance at its bounds, the rescaling thresholds, each quadrant case below 1/16 and from the table."""
    r = random.Random(seed)
    special = [0.0, -0.0, math.inf, -math.inf, math.nan, 1.0, -1.0, 5e-324, -5e-324, 1.5e-310, 1e300, -1e300, 0.3, -7.0]
    out = [(a, b) for a in special for b in special]
    for i in range(0, len(I.ATAN2_CIJ), 7):                      # u at a table point: the two-sum's other order
        c = I.ATAN2_CIJ[i]
        out += [(c, 1.0), (-c, 1.0), (1.0, -1.0 / c), (math.nextafter(c, 1.0), 1.0), (math.nextafter(c, 0.0), -1.0)]
    for e in (56, 57, 58, 500, 501, 520, 600, 1000):              # the exponents' distance and the rescaling
        for s in (1.0, -1.0):
            for t in (1.0, -1.0):
                out += [(s * 2.0 ** e * 1.25, t * 0.75), (s * 0.75, t * 2.0 ** e * 1.25), (s * 2.0 ** -e, t * 1.5),
                        (s * 1.5, t * 2.0 ** -e), (s * 2.0 ** (e - 1050), t * 2.0 ** -1060)]
    for j in range(n):
        kind = j % 9
        if kind == 0:
            y, x = d64(r.getrandbits(64)), d64(r.getrandbits(64))
        elif kind == 1:
            y, x = r.uniform(-1e3, 1e3), r.uniform(-1e3, 1e3)
        elif kind == 2:
            y, x = r.gauss(0, 1) * 10 ** r.uniform(-300, 300), r.gauss(0, 1) * 10 ** r.uniform(-300, 300)
        elif kind == 3:                                           # |y| near |x|: u near 1, both orders
            x = r.uniform(-1, 1)
            y = x * r.uniform(-1.2, 1.2)
        elif kind == 4:                                           # u below 1/16
            x = r.gauss(0, 1)
            y = x * r.uniform(-0.07, 0.07)
            if r.random() < 0.5:
                x, y = y, x
        elif kind == 5:                                           # u at the table's cells
            x = r.choice([-1, 1]) * r.uniform(0.5, 2.0)
            y = x * (r.randint(16, 256) / 256 + r.uniform(-2 ** -10, 2 ** -10)) * r.choice([-1, 1])
            if r.random() < 0.5:
                x, y = y, x
        elif kind == 6:                                           # subnormal and tiny
            y, x = d64(r.randrange(1, 1 << 60)) * r.choice([-1, 1]), d64(r.randrange(1, 1 << 60)) * r.choice([-1, 1])
        elif kind == 7:                                           # the yaw's arguments: a heading's x and z
            y, x = r.gauss(0, 1) * 10 ** r.uniform(-3, 5), r.gauss(0, 1) * 10 ** r.uniform(-3, 5)
        else:
            y, x = r.gauss(0, 1), r.gauss(0, 1) * 10 ** r.uniform(-20, 20)
        out.append((y, x))
    return out


def asin_branch(x: float) -> str:
    u = I.bits(x)
    k, lo = (u >> 32) & 0x7FFFFFFF, u & 0xFFFFFFFF
    for name, bound in (("tiny", I.ASIN_K_TINY), ("taylor", I.ASIN_K_TAYLOR), ("q1", I.ASIN_K_QUARTER),
                        ("q2", I.ASIN_K_HALF), ("h", I.ASIN_K_075), ("a", I.ASIN_K_0921875), ("b", I.ASIN_K_0953125),
                        ("c", I.ASIN_K_096875), ("near_one", I.ASIN_K_NEAR_ONE)):
        if k < bound:
            return name
    if k == I.ASIN_K_ONE and lo == 0:
        return "one"
    if k > I.ASIN_K_INF or (k == I.ASIN_K_INF and lo != 0):
        return "nan"
    return "beyond"


def atan2_branch(y: float, x: float) -> str:
    """Which path of __ieee754_atan2 a pair takes (special values by kind; then the case and its polynomial)."""
    if x != x or y != y:
        return "nan"
    if y == 0.0 or x == 0.0 or math.isinf(x) or math.isinf(y):
        return "special"
    ax, ay = abs(x), abs(y)
    de = I._s32(((I.bits(y) >> 32) & 0x7FF00000) - ((I.bits(x) >> 32) & 0x7FF00000))
    if de >= I.ATAN2_EP:
        return "de_ep"
    if de <= I.ATAN2_EM:
        return "de_em_pos" if x > 0 else "de_em_neg"
    tag = ""
    if ax < I.ATAN2_TWOM500 or ay < I.ATAN2_TWOM500:
        ax, ay, tag = ax * I.ATAN2_TWO500, ay * I.ATAN2_TWO500, "+up"
    if ax > I.ATAN2_TWO500 or ay > I.ATAN2_TWO500:
        ax, ay, tag = ax * I.ATAN2_TWOM500, ay * I.ATAN2_TWOM500, tag + "+down"
    u = ay / ax if ay < ax else ax / ay
    case = ("i" if ay < ax else "ii") if x > 0 else ("iii" if ax < ay else "iv")
    return case + ("_small" if u < I.ATAN2_INV16 else "_table") + tag


def asin_differences(xs: list) -> list:
    return [x for x in xs if not same(I.asin(x), _C.asin(x))]


def atan2_differences(pairs: list) -> list:
    return [(y, x) for y, x in pairs if not same(I.atan2(y, x), _C.atan2(y, x))]


def test_asin_is_the_c_librarys():
    bad = asin_differences(asin_arguments(1, 60000))
    assert not bad, (len(bad), [x.hex() for x in bad[:5]])


def test_atan2_is_the_c_librarys():
    bad = atan2_differences(atan2_arguments(2, 60000))
    assert not bad, (len(bad), [(y.hex(), x.hex()) for y, x in bad[:5]])


def test_the_samples_reach_every_branch():
    """Every branch of both functions is taken by the samples (the rescaled arguments in both directions too); the
    floors are the counts a branch must reach."""
    seen_a = {}
    for x in asin_arguments(3, 20000):
        b = asin_branch(x)
        seen_a[b] = seen_a.get(b, 0) + 1
    assert set(seen_a) == {"tiny", "taylor", "q1", "q2", "h", "a", "b", "c", "near_one", "one", "nan", "beyond"}
    assert min(seen_a.values()) >= 4, seen_a
    seen_t = {}
    for y, x in atan2_arguments(4, 20000):
        b = atan2_branch(y, x)
        seen_t[b] = seen_t.get(b, 0) + 1
    base = {b.split("+")[0] for b in seen_t}
    assert base >= {"nan", "special", "de_ep", "de_em_pos", "de_em_neg", "i_small", "i_table", "ii_small", "ii_table",
                    "iii_small", "iii_table", "iv_small", "iv_table"}, sorted(seen_t)
    assert any("+up" in b for b in seen_t) and any("+down" in b for b in seen_t), sorted(seen_t)
    assert min(seen_t[b] for b in seen_t if "+" not in b) >= 20, seen_t


def test_the_two_sum_takes_both_orders():
    """Case (i)'s EADD(t3, du) takes both of its formulas (|t3| > |du| and not) on the samples."""
    orders = set()
    for y, x in atan2_arguments(5, 20000):
        if atan2_branch(y, x) != "i_table":
            continue
        ax, ay = abs(x), abs(y)
        u = ay / ax
        v = ax * u
        du = ((ay - v) - math.fma(ax, u, -v)) / ax
        t3 = u - I._cij_row(u)[0]
        orders.add(abs(t3) > abs(du))
    assert orders == {True, False}


@pytest.mark.skipif(not LIBM.exists() or hashlib.sha256(LIBM.read_bytes()).hexdigest() != I.LIBM_SHA256,
                    reason="not the library the tables were read from")
def test_the_tables_and_constants_are_the_librarys_bytes():
    b = LIBM.read_bytes()
    at = lambda addr, n=1: list(struct.unpack_from(f"<{n}d", b, addr))
    a = I.ADDRESSES
    assert at(a["ASIN_INROOT"], 128) == list(I.ASIN_INROOT)
    assert at(a["ASIN_POWTWO"], len(I.ASIN_POWTWO)) == list(I.ASIN_POWTWO)
    assert at(a["ASIN_ASNCS"], len(I.ASIN_ASNCS)) == list(I.ASIN_ASNCS)
    assert at(a["ATAN2_CIJ"], len(I.ATAN2_CIJ)) == list(I.ATAN2_CIJ) and len(I.ATAN2_CIJ) == 7 * 241
    for name, addr in a.items():
        if name not in ("ASIN_INROOT", "ASIN_POWTWO", "ASIN_ASNCS", "ATAN2_CIJ"):
            assert at(addr) == [getattr(I, name)], name


@pytest.mark.parametrize("fault", ["unfused", "asncs_entry", "cij_entry", "asin_hp1", "atan2_hpi1", "atan2_opi1",
                                   "sign"])
def test_planted_faults_are_caught(fault, monkeypatch):
    if fault == "unfused":                                # every fused step a multiplication and an addition
        monkeypatch.setattr(I, "fma", lambda a, b, c: a * b + c)
    elif fault == "asncs_entry":                          # the 0.5-0.75 range's cell 20: its constant term's last bit
        tab = list(I.ASIN_ASNCS)
        n = I.ASIN_N_H_OFF + 3 * 4 * 20
        tab[n + 9] = d64(I.bits(tab[n + 9]) ^ 1)
        monkeypatch.setattr(I, "ASIN_ASNCS", tuple(tab))
    elif fault == "cij_entry":                            # row 100 of cij: atan at its point, last bit
        tab = list(I.ATAN2_CIJ)
        tab[7 * 100 + 1] = d64(I.bits(tab[7 * 100 + 1]) ^ 1)
        monkeypatch.setattr(I, "ATAN2_CIJ", tuple(tab))
    elif fault == "asin_hp1":                             # near 1: pi/2's low half dropped
        monkeypatch.setattr(I, "ASIN_HP1", 0.0)
    elif fault == "atan2_hpi1":                           # cases (ii) and (iii): pi/2's low half dropped
        monkeypatch.setattr(I, "ATAN2_HPI1", 0.0)
    elif fault == "atan2_opi1":                           # case (iv): pi's low half dropped
        monkeypatch.setattr(I, "ATAN2_OPI1", 0.0)
    else:                                                 # the sign of y not given to the result
        monkeypatch.setattr(I, "_signed", lambda z, y: z)
    xs = asin_arguments(6, 20000) + [d64(I.bits(0.5) + (j << 40) + (80 << 43)) for j in range(-2000, 2000)]
    pairs = atan2_arguments(7, 20000) + [(1.0, 256 / (116 + k / 997)) for k in range(1000)]
    caught = len(asin_differences(xs)) + len(atan2_differences(pairs))
    assert caught, f"the planted fault {fault} went unnoticed"

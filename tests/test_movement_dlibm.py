# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The C library's double sin, cos and log transcribed (exact_ballpark.dlibm) against the C library itself (glibc 2.34
aarch64: Python's math.sin, math.cos and math.log call it), bit for bit, on arguments over every branch: the tiny,
Taylor, table, around-pi/2, reduced and huge (branred) arguments of sin and cos, both halves of their quadrants, log
near 1, normal, subnormal and special. The tables and constants are held to the library's bytes when the library is
the one they were read from. Planted faults (an unfused multiply-add, the Taylor polynomial's bound moved, branred's other
pi/2 piece, a table entry of each table) must each be caught. (A coefficient's last bit is not planted: the Taylor
polynomial's high coefficients are too small for their last bit to reach a result.) The kernel's device copies are held to these in tests/test_movement_vec_modes.py."""

from __future__ import annotations

import hashlib
import math
import random
import struct
from pathlib import Path

import pytest

from exact_ballpark import dlibm as D

LIBM = Path("/usr/lib64/libm.so.6")


def d64(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def same(a: float, b: float) -> bool:
    return struct.pack("<d", a) == struct.pack("<d", b) or (a != a and b != b)


def arguments(seed: int, n: int) -> list:
    """Arguments over every branch of sin, cos and log, with their boundaries and special values."""
    r = random.Random(seed)
    pio2 = math.pi / 2
    out = [0.0, -0.0, 1.0, -1.0, 0.5, D.TAYLOR_LIMIT, -D.TAYLOR_LIMIT, math.pi, -math.pi, pio2, 2 * math.pi, 1e8,
           1.0541435e8, 1e300, 1.7976931348623157e308, 5e-324, 2.2250738585072014e-308, 1 - 2 ** -53, 1 + 2 ** -52,
           0.001, 0.9375, 1.0647, math.inf, -math.inf, math.nan]
    for k in (D.K_TINY_SIN, D.K_TINY_COS, D.K_DO_SIN, D.K_MID, D.K_REDUCE, D.K_HUGE):   # each branch's bound, both sides
        for low in (0, 1, 0xFFFFFFFF):
            for hi in (k, k - 1):
                out += [d64((hi << 32) | low), -d64((hi << 32) | low)]
    for j in range(n):
        kind = j % 12
        if kind == 0:
            x = d64(r.getrandbits(64))
        elif kind == 1:
            x = r.uniform(-2.5, 2.5)
        elif kind == 2:
            x = r.uniform(-1e9, 1e9)
        elif kind == 3:
            x = r.choice([-1, 1]) * 10 ** r.uniform(-30, 308)
        elif kind == 4:
            x = pio2 * r.randint(-10 ** 6, 10 ** 6) + r.uniform(-1e-6, 1e-6)
        elif kind == 5:
            x = d64(r.randrange(0x3FE0000000000000, 0x4010000000000000))
        elif kind == 6:
            x = 1.0 + r.uniform(-0.07, 0.07)
        elif kind == 7:
            x = d64(r.randrange(1, 1 << 52))
        elif kind == 8:
            x = r.uniform(0, 2e5)
        elif kind == 9:
            x = 10 ** r.uniform(-3, 3)
        elif kind == 10:
            x = d64(r.randrange(0x3E30000000000000, 0x3E60000000000000)) * r.choice([-1, 1])
        else:
            x = r.uniform(1e8, 1e16) * r.choice([-1, 1])
        out.append(x)
    return out


def c_value(name: str, x: float) -> float:
    """The C library's value: Python raises where C returns NaN or an infinity."""
    if name == "log":
        if x == 0.0:
            return -math.inf
        if x < 0.0 or x != x:
            return math.nan
        return math.log(x)
    return getattr(math, name)(x) if math.isfinite(x) else math.nan


def differences(f, name: str, xs: list) -> list:
    return [x for x in xs if not same(f(x), c_value(name, x))]


@pytest.mark.parametrize("name", ["sin", "cos", "log"])
def test_each_function_is_the_c_librarys(name):
    xs = arguments({"sin": 1, "cos": 2, "log": 3}[name], 60000)
    bad = differences(getattr(D, name), name, xs)
    assert not bad, (len(bad), [x.hex() for x in bad[:5]])


def test_branred_and_the_reduction_reach_every_quadrant():
    """Every quadrant of both reductions is taken by the samples (each with a do_sin and a do_cos)."""
    seen = set()
    for x in arguments(4, 6000):
        k = (D.bits(x) >> 32) & 0x7FFFFFFF
        if D.K_MID <= k < D.K_REDUCE:
            seen.add(("reduce", D.reduce_sincos(x)[0]))
        elif D.K_REDUCE <= k < D.K_HUGE:
            seen.add(("branred", D.branred(x)[0]))
    assert seen == {(w, n) for w in ("reduce", "branred") for n in range(4)}


@pytest.mark.skipif(not LIBM.exists() or hashlib.sha256(LIBM.read_bytes()).hexdigest() != D.LIBM_SHA256,
                    reason="not the library the tables were read from")
def test_the_tables_and_constants_are_the_librarys_bytes():
    b = LIBM.read_bytes()
    at = lambda addr, n=1: list(struct.unpack_from(f"<{n}d", b, addr))
    a = D.ADDRESSES
    assert at(a["SINCOSTAB"], 440) == list(D.SINCOSTAB)
    assert at(a["BR_TOVERP"], 75) == list(D.BR_TOVERP)
    log = at(a["LOG_DATA"], 2 + 5 + 11 + 256)
    assert (log[0], log[1], log[2:7], log[7:18]) == (D.LOG_LN2HI, D.LOG_LN2LO, list(D.LOG_A), list(D.LOG_B))
    assert [tuple(log[18 + 2 * i:20 + 2 * i]) for i in range(128)] == list(D.LOG_TAB)
    for name, addr in a.items():
        if name not in ("SINCOSTAB", "BR_TOVERP", "LOG_DATA"):
            assert at(addr) == [getattr(D, name)], name


@pytest.mark.parametrize("fault", ["unfused", "taylor_limit", "branred_mp2", "table_row", "log_table"])
def test_planted_faults_are_caught(fault, monkeypatch):
    if fault == "unfused":
        monkeypatch.setattr(D, "fma", lambda a, b, c: a * b + c)
    elif fault == "taylor_limit":                         # the Taylor polynomial's bound moved from 0.126 to 0.125
        monkeypatch.setattr(D, "TAYLOR_LIMIT", 0.125)
    elif fault == "branred_mp2":                          # branred's own pi/2 piece swapped for reduce_sincos's
        monkeypatch.setattr(D, "BR_MP2", D.MP2)
    elif fault == "table_row":                            # a table row's cos wrong in its last bit (|x| near 50/128)
        tab = list(D.SINCOSTAB)
        tab[4 * 50 + 2] = d64(D.bits(tab[4 * 50 + 2]) ^ 1)
        monkeypatch.setattr(D, "SINCOSTAB", tuple(tab))
    else:                                                 # a log table cell's log c wrong in its last bit
        tab = list(D.LOG_TAB)
        tab[37] = (tab[37][0], d64(D.bits(tab[37][1]) ^ 1))
        monkeypatch.setattr(D, "LOG_TAB", tuple(tab))
    xs = arguments(5, 30000) + [50 / 128 + i * 1e-6 for i in range(-1000, 1000)] + [0.125 + i * 1e-6 for i in range(1000)]
    off = (1 << 64) - D.LOG_NEG_OFF                      # log's table cell 37, at a few exponents
    xs += [d64(off + (37 << 45) + (k << 52) + i * (1 << 33)) for k in (-3, 0, 2, 7) for i in range(500)]
    caught = sum(len(differences(getattr(D, n), n, xs)) for n in ("sin", "cos", "log"))
    assert caught, f"the planted fault {fault} went unnoticed"

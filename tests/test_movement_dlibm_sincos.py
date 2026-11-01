# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""glibc's sincos transcribed (exact_ballpark.dlibm.sincos) against the C library's own sincos (glibc 2.34 aarch64,
through ctypes: what Destiny's compiled code calls for a sine and a cosine of one argument), bit for bit, on the dlibm
tests' arguments over every branch and on every argument within 2,048 patterns of each point where its sine can part
from sin's (|x| = hp0 -+ an odd multiple of 1/256, and |x| = hp0; both signs); and the one place in those windows where
glibc's sin and sincos differ (x = +-0x1.ad1fb54442d18p+0, the sine's last bit): there dlibm.sin is sin's value and
dlibm.sincos sincos's. A planted fault (sincos taken as sin and cos) must be caught. The device copy is held to it in
tests/test_movement_vec_ypr_cuda.py; the work tool sincos_windows.c checked 2^20 patterns a side."""

from __future__ import annotations

import ctypes
import math
import struct

import test_movement_dlibm as TD
from exact_ballpark import dlibm as D

_C = ctypes.CDLL("libm.so.6")
_C.sin.restype = _C.cos.restype = ctypes.c_double
_C.sin.argtypes = _C.cos.argtypes = [ctypes.c_double]
_C.sincos.restype = None
_C.sincos.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
X0 = float.fromhex("0x1.ad1fb54442d18p+0")          # hp0 + 27/256: sin and sincos differ here


def same(a: float, b: float) -> bool:
    return struct.pack("<d", a) == struct.pack("<d", b) or (a != a and b != b)


def c_sincos(x: float) -> tuple:
    s, c = ctypes.c_double(), ctypes.c_double()
    _C.sincos(x, ctypes.byref(s), ctypes.byref(c))
    return s.value, c.value


def windows(w: int) -> list:
    """Every argument within w patterns of |x| = hp0 -+ (2j + 1)/256 (in range) and of |x| = hp0, both signs."""
    out = []
    for j in range(-1, 128):
        b = 0.0 if j < 0 else (2 * j + 1) / 256
        for side in ((-1,) if j < 0 else (-1, 1)):
            c = D.HP0 + side * b
            if 0.855469 <= c < 2.426265:
                u = D.bits(c)
                for d in range(-w, w + 1):
                    out += [D.dbl(u + d), -D.dbl(u + d)]
    return out


def test_sincos_is_the_c_librarys():
    xs = TD.arguments(41, 60000) + windows(2048) + [X0, -X0]
    bad = [x for x in xs if not all(same(a, b) for a, b in zip(D.sincos(x), c_sincos(x)))]
    assert not bad, (len(bad), [x.hex() for x in bad[:5]])


def test_where_glibcs_sin_and_sincos_differ():
    """In the windows glibc's sin and sincos differ only at +-X0, in the sine's last bit; dlibm has both values."""
    differ = [x for x in windows(2048) if not same(_C.sin(x), c_sincos(x)[0])]
    assert sorted(differ) == [-X0, X0]
    assert all(same(_C.cos(x), c_sincos(x)[1]) for x in windows(256))
    for x in (X0, -X0):
        s, c = D.sincos(x)
        assert same(s, c_sincos(x)[0]) and same(D.sin(x), _C.sin(x)) and not same(s, D.sin(x))
        assert abs(D.bits(s) - D.bits(D.sin(x))) == 1 and same(c, D.cos(x))
    assert math.isnan(D.sincos(math.inf)[0]) and math.isnan(D.sincos(math.nan)[1])


def test_a_planted_fault_is_caught():
    """sincos taken as sin and cos (what the reference calls today) differs from the C library's sincos at X0."""
    xs = windows(64) + [X0, -X0]
    caught = [x for x in xs if not all(same(a, b) for a, b in zip((D.sin(x), D.cos(x)), c_sincos(x)))]
    assert caught == [x for x in xs if abs(x) == X0] and caught

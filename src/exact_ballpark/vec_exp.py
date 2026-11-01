# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (c) 2026 Eidosoma
# Transcribed from the GNU C Library 2.34; see THIRD_PARTY_NOTICES.md.
"""The C library's exp on float64 tensors, bit for bit on the CPU and the GPU: what Destiny's integrator calls over a
part of a tick (`Ballpark::Integrate`, Ballpark.cpp:751: exp(-k/m t) when t is not the tick), which both batched
collision methods need (exact_ballpark.vec_collide).

The scalar reference (ref.py, modes.py, space.py) calls Python's `math.exp`, which is the C library's, and so does
Destiny's compiled code: glibc 2.34 on this machine (aarch64, /lib64/libm.so.6). Its exp is ARM's optimized-routines
algorithm (glibc's sysdeps/ieee754/dbl-64/e_exp.c): k = round(x 128/ln 2), r = x - k ln2/128 in two parts, a degree-5
polynomial in r around a table of 2^(i/128), with special cases for |x| below 2^-54, at or above 512 (results near the
overflow or in the subnormal range) and at or above 1024. On aarch64 GCC fused seven of its multiply-adds into fmadd
(objdump of `__exp`, 0x30e70 to 0x30eec, and one in its special case at 0x30fc4), so an unfused evaluation differs from
it: in 134 of 200,000 arguments drawn from [-20, 0]. Its special case for
subnormal results computes scale x tmp once and does not fuse it (0x30f34).

torch has no fused multiply-add with a defined rounding, so `fma` emulates one exactly from correctly rounded
operations only (+, -, *, comparisons and integer steps on the bits): Dekker's product with Veltkamp's split, Knuth's
TwoSum, and an addition rounded to odd, combined as Boldo and Melquiond's emulated FMA ("Emulation of FMA and
correctly rounded sums: proved algorithms using rounding to odd", IEEE Trans. Computers 57(4), 2008, Algorithm 5):
uh + ul = a b exactly, th + tl = c + uh exactly, v = tl + ul rounded to odd, fma = th + v rounded to nearest. It is
exact while no operand or partial product overflows or underflows; every fma inside exp stays far from both (|r| <
0.003 or exactly 0, the table's scales are normal; tests/test_movement_vec_exp.py checks it against math.fma, and exp
against math.exp, on the CPU and the GPU).

The constants are glibc's `__exp_data` (EXP_TABLE_BITS 7, EXP_POLY_ORDER 5): 1/ln2 x 128, ln2/128 in a 37-bit head and
a tail, the four polynomial coefficients; the table is generated here from its definition, scale_i = 2^(i/128) rounded
to nearest and tail_i = 2^(i/128)/scale_i - 1 rounded to nearest, stored as (bits of tail_i, bits of scale_i - i 2^45),
which reproduces the 256 words read from the loaded library (SHA-256 `TABLE_SHA256`). Nothing here is compiled or
fused (no torch.compile): a fused multiply-add anywhere else would change the last bits.
"""

from __future__ import annotations

import hashlib
import struct
from decimal import Context
from fractions import Fraction

import torch

F64, I64 = torch.float64, torch.int64

INVLN2N = float.fromhex("0x1.71547652b82fep+7")       # 128 / ln 2
NEGLN2HIN = float.fromhex("-0x1.62e42fefa0000p-8")    # -ln 2 / 128, its first 37 bits
NEGLN2LON = float.fromhex("-0x1.cf79abc9e3b3ap-47")   # the rest
C2 = float.fromhex("0x1.ffffffffffdbdp-2")            # the polynomial's coefficients for r^2 .. r^5
C3 = float.fromhex("0x1.555555555543cp-3")
C4 = float.fromhex("0x1.55555cf172b91p-5")
C5 = float.fromhex("0x1.1111167a4d017p-7")
TWO_1009 = 2.0 ** 1009                                 # the special case's rescalings (0x7f00000000000000, 2^-1022)
TWO_M1022 = 2.0 ** -1022
TABLE_SHA256 = "a4014caf214c2bab90ea0fdf3f57941e0235ae04dc2282a6c2f72e7a480c87e4"   # the 256 words, little-endian
SPLIT = 134217729.0                                    # 2^27 + 1: Veltkamp's splitter for 53-bit significands
_M64 = (1 << 64) - 1


def _u64(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def exp_table() -> tuple:
    """The 256 words of `__exp_data.tab` from their definition (80 decimal digits of 2^(i/128), each rounded once)."""
    ctx = Context(prec=80)
    ln2 = ctx.ln(2)
    out = []
    for i in range(128):
        v = Fraction(ctx.exp(ctx.divide(ctx.multiply(ln2, i), 128)))
        scale = float(v)                                # float(Fraction) rounds to nearest, ties to even
        tail = float(v / Fraction(scale) - 1)
        out += [_u64(tail), (_u64(scale) - (i << 45)) & _M64]
    return tuple(out)


TABLE = exp_table()
if hashlib.sha256(struct.pack("<256Q", *TABLE)).hexdigest() != TABLE_SHA256:
    raise RuntimeError("the generated exp table differs from glibc's")
_TABLES: dict = {}


def _table(device) -> torch.Tensor:
    """The table as int64 words (two's complement of the unsigned words) on a device, made once."""
    key = str(device)
    if key not in _TABLES:
        _TABLES[key] = torch.tensor([t - (1 << 64) if t >> 63 else t for t in TABLE], dtype=I64, device=device)
    return _TABLES[key]


# ---- the exact fused multiply-add ----

def split(a):
    """Veltkamp's split: a = hi + lo exactly, each with at most 26 significant bits (|a| below 2^996)."""
    c = a * SPLIT
    hi = c - (c - a)
    return hi, a - hi


def two_product(a: torch.Tensor, b, b_split=None) -> tuple:
    """Dekker's product: (p, e) with p = a b rounded and p + e = a b exactly (no overflow or underflow). `b` may be a
    Python float, with its split given."""
    p = a * b
    ah, al = split(a)
    bh, bl = split(b) if b_split is None else b_split
    return p, al * bl - (((p - ah * bh) - al * bh) - ah * bl)


def two_sum(a, b) -> tuple:
    """Knuth's TwoSum: (s, e) with s = a + b rounded and s + e = a + b exactly."""
    s = a + b
    z = s - a
    return s, (a - (s - z)) + (b - z)


def add_odd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a + b rounded to odd: the sum when it is a double, else whichever of its two neighbouring doubles has an odd last
    bit (the rounded sum, or one step from it toward the exact sum)."""
    s, e = two_sum(a, b)
    bits = s.view(I64)
    step = torch.where((e > 0.0) == (s > 0.0), 1, -1)                  # away from zero when e has s's sign
    odd = (bits + step).view(F64)
    return torch.where((e != 0.0) & ((bits & 1) == 0), odd, s)


def fma(a: torch.Tensor, b, c, b_split=None) -> torch.Tensor:
    """a b + c rounded once (Boldo and Melquiond's emulated FMA); exact while nothing overflows or underflows."""
    uh, ul = two_product(a, b, b_split)
    th, tl = two_sum(c, uh)
    return th + add_odd(tl, ul)


_NEGLN2HIN_SPLIT, _NEGLN2LON_SPLIT = split(NEGLN2HIN), split(NEGLN2LON)
_C3_SPLIT, _C5_SPLIT = split(C3), split(C5)


# ---- exp ----

def exp(x: torch.Tensor) -> torch.Tensor:
    """glibc 2.34 aarch64's exp of a float64 tensor, elementwise, bit for bit (math.exp's value; +inf where math.exp
    raises OverflowError)."""
    bits = x.view(I64)
    abstop = (bits >> 52) & 0x7FF
    tiny = abstop < 0x3C9                               # |x| < 2^-54: 1 + x
    large = abstop == 0x408                             # 512 <= |x| < 1024: the special case
    huge = abstop > 0x408                               # |x| >= 1024, infinities and NaN
    xs = torch.where(tiny | huge, torch.ones_like(x), x)
    z = xs * INVLN2N
    kd = torch.trunc(z + torch.where(z < 0.0, -0.5, 0.5))              # frinta: half away from zero (|z| < 2^18)
    ki = kd.to(I64)
    r = fma(kd, NEGLN2HIN, xs, _NEGLN2HIN_SPLIT)
    r = fma(kd, NEGLN2LON, r, _NEGLN2LON_SPLIT)
    tab = _table(x.device)
    idx = (ki & 0x7F) * 2
    tail = tab[idx].view(F64)
    sb = tab[idx + 1]
    top = ki * (1 << 45)                                # ki << 45, exact for |ki| < 2^18
    p23 = fma(r, C3, C2, _C3_SPLIT)
    p45 = fma(r, C5, C4, _C5_SPLIT)
    r2 = r * r
    tmp = r + tail
    r4 = r2 * r2
    tmp = fma(p23, r2, tmp)
    tmp = fma(r4, p45, tmp)
    scale = (sb + top).view(F64)
    main = fma(tmp, scale, scale)
    # the special case (abstop forced to 0 for 512 <= |x| < 1024); the additions are ordered to stay within int64
    pos = (ki & 0x80000000) == 0
    sc_hi = (sb + (top - (1009 << 52))).view(F64)
    y_hi = fma(tmp, sc_hi, sc_hi) * TWO_1009
    sc_lo = (sb + (top + (1022 << 52))).view(F64)
    st = tmp * sc_lo
    y = sc_lo + st
    hi = y + 1.0
    lo = (sc_lo - y) + st
    lo = ((1.0 - hi) + y) + lo
    y2 = (lo + hi) - 1.0
    y2 = torch.where(y2 == 0.0, torch.zeros_like(y2), y2)
    y_lo = torch.where(y < 1.0, y2, y) * TWO_M1022
    special = torch.where(pos, y_hi, y_lo)
    over = torch.where(bits < 0, torch.zeros_like(x), torch.full_like(x, float("inf")))
    over = torch.where((abstop == 0x7FF) & (x != float("-inf")), x + 1.0, over)
    out = torch.where(large, special, main)
    out = torch.where(huge, over, out)
    return torch.where(tiny, x + 1.0, out)

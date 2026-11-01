# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (c) 2026 Eidosoma
# Transcribed from the GNU C Library 2.34; see THIRD_PARTY_NOTICES.md.
"""The C library's powf as scalar Python, operation for operation: what Destiny's mushroom growth calls in float
(Ballpark::Evolve, Ballpark.cpp:569: `mRadius = range*pow(timeFraction,0.25f)`, the float overload, so glibc's powf;
MOV-734; the compiled harness calls powf@plt there and nowhere else), transcribed here so that the combat kernel
(exact_ballpark.vec_combat_cuda's `g_powf`) can follow it line for line on the GPU. The reference exact_ballpark.full calls
the C library's own powf through ctypes.

The C library is glibc 2.34 on this machine (aarch64, /usr/lib64/libm.so.6, SHA-256 `LIBM_SHA256`), and powf is its
compiled code read from the disassembly (__powf_finite at 0x411e0, exported as powf), every fused multiply-add GCC made
fused here too (math.fma, rounded once) and only those: nine, six in log2_inline and three in exp2_inline.

- powf is ARM's optimized-routines (glibc's sysdeps/ieee754/flt-32/e_powf.c, MIT-licensed upstream, as vec_libm's expf
  and logf are): x = 2^k z with z near one of 16 points c (the table __powf_log2_data: 1/c and log2 c, scaled by 32),
  log2 x by a quintic in r = z/c - 1, times y in double; then 2^(y log2 x) as 2^(k/32) from the 32-entry table
  __exp2f_data and a cubic in the rest (aarch64's frinta and fcvtas: TOINT_INTRINSICS), rounded to float once. First the
  special cases: y zero, infinite or NaN; x zero, infinite or NaN (a negative one with y's parity); a negative finite x
  (NaN unless y is an integer, the result's sign from its parity); a subnormal x normalised; and, where
  |y log2 x| >= 126, glibc's __math_oflowf, __math_uflowf and __math_may_uflowf ((sign ? -t : t) times t in float, t =
  0x1p97f, 0x1p-95f, 0x1.4p-75f), __math_divzerof and __math_invalidf.
- Constants and tables were read out of the library at the addresses the disassembly references, or assembled from its
  mov/movk/movi immediates (the generated block below, which checks each loading instruction's text first); a test holds them to the library's bytes when it is that
  library, and the exp2 table and cubic to vec_libm's (expf takes the same `__exp2f_data.poly_scaled`).

A float is held in a Python float (as exact_ballpark.full holds it): `powf(x, y)` takes two floats so held and returns its
float so held. A NaN result is NaN, not necessarily with the C library's payload; errno and the floating-point flags are
not modelled. The rounding mode is to nearest, as Python's (the library's check of 1 + 0x1p-25f is then never taken).
Checked by tests/test_movement_dlibm_powf.py against the C library itself, and the device's transcription against both.
"""

from __future__ import annotations

import math
import struct

_H = float.fromhex
LIBM_SHA256 = "b4fa1652bf3a7267d0fa4256fade78f0df2954b7b39ee842ec410e0e7b0f6c86"   # /usr/lib64/libm.so.6
M32, M64 = 0xFFFFFFFF, (1 << 64) - 1
fma = math.fma
_F32, _U32, _F64, _U64 = struct.Struct("<f"), struct.Struct("<I"), struct.Struct("<d"), struct.Struct("<Q")
INF, NAN = float("inf"), float("nan")

# ---- generated from this machine's libm.so.6 (SHA-256
# b4fa1652bf3a7267...): read at the addresses the disassembly of __powf_finite (0x411e0) and its error helpers
# references, or assembled from their mov/movk/movi immediates; not typed ----

POWF_X_MIN = 0x800000
POWF_X_SPAN = 0x7effffff
POWF_ZINF = 0xfeffffff
POWF_NEG_OFF = 0xc0cd0000
POWF_TOP_MASK = 0xff800000
POWF_I_SHIFT = 19
POWF_I_BITS = 4
POWF_K_SHIFT = 18
POWF_LOG2_INVC = (_H("0x1.661ec79f8f3bep+0"), _H("0x1.571ed4aaf883dp+0"), _H("0x1.49539f0f010b0p+0"),
                  _H("0x1.3c995b0b80385p+0"), _H("0x1.30d190c8864a5p+0"), _H("0x1.25e227b0b8ea0p+0"),
                  _H("0x1.1bb4a4a1a343fp+0"), _H("0x1.12358f08ae5bap+0"), _H("0x1.0953f419900a7p+0"),
                  _H("0x1.0000000000000p+0"), _H("0x1.e608cfd9a47acp-1"), _H("0x1.ca4b31f026aa0p-1"),
                  _H("0x1.b2036576afce6p-1"), _H("0x1.9c2d163a1aa2dp-1"), _H("0x1.886e6037841edp-1"),
                  _H("0x1.767dcf5534862p-1"))
POWF_LOG2_LOGC = (_H("-0x1.efec65b963019p+3"), _H("-0x1.b0b6832d4fca4p+3"), _H("-0x1.7418b0a1fb77bp+3"),
                  _H("-0x1.39de91a6dcf7bp+3"), _H("-0x1.01d9bf3f2b631p+3"), _H("-0x1.97c1d1b3b7af0p+2"),
                  _H("-0x1.2f9e393af3c9fp+2"), _H("-0x1.960cbbf788d5cp+1"), _H("-0x1.a6f9db6475fcep+0"), _H("0x0.0p+0"),
                  _H("0x1.338ca9f24f53dp+1"), _H("0x1.476a9543891bap+2"), _H("0x1.e840b4ac4e4d2p+2"),
                  _H("0x1.40645f0c6651cp+3"), _H("0x1.88e9c2c1b9ff8p+3"), _H("0x1.ce0a44eb17bccp+3"))
POWF_A = (_H("0x1.27616c9496e0bp+3"), _H("-0x1.71969a075c67ap+3"), _H("0x1.ec70a6ca7baddp+3"),
          _H("-0x1.7154748bef6c8p+4"), _H("0x1.71547652ab82bp+5"))
POWF_BIG = 0x815e
POWF_OFLOW = _H("0x1.fffffffd1d571p+11")
POWF_RND = _H("0x1.fffffffa3aae2p+11")
POWF_UFLOW = 0xc0b2c00000000000
POWF_MAY_UFLOW = 0xc0b2a00000000000
POWF_TINY_F = 0x33000000
POWF_EXP2_C = (_H("0x1.c6af84b912394p-20"), _H("0x1.ebfce50fac4f3p-13"), _H("0x1.62e42ff0c52d6p-6"))
POWF_EXP2_TAB = (0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51, 0x3fef72b83c7d517b,
                 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1, 0x3fef06fe0a31b715, 0x3feef1a7373aa9cb,
                 0x3feedea64c123422, 0x3feece086061892d, 0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429,
                 0x3feea47eb03a5585, 0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
                 0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d, 0x3feee89f995ad3ad,
                 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069, 0x3fef5818dcfba487, 0x3fef7c97337b9b5f,
                 0x3fefa4afa2a490da, 0x3fefd0765b6e4540)
POWF_EXP2_SHIFT = 47
POWF_INT_E = 0x96
POWF_SIGN_BIAS = 0x10000
POWF_SUBNORMAL_MAX = 0x7fffff
POWF_TWO23_F = 0x4b000000
POWF_NORM_SUB = 0xb800000
POWF_ONE_BITS = 0x3f800000
POWF_INF2 = 0xff000000
POWF_ONE2 = 0x7f000000
POWF_ONE2_M1 = 0x7effffff
POWF_E_HALF = 0x7e
POWF_E_ONE = 0x7f
POWF_E_SPAN = 23
POWF_QNAN_BIT = 0x400000
POWF_QNAN = 0x7fc00000
POWF_UFLOWF_X = 0x10000000
POWF_MAY_UFLOWF_X = 0x1a200000
POWF_OFLOWF_X = 0x70000000
ADDRESSES = {'__powf_log2_data': 0x76bf8, '__exp2f_data': 0x76870, 'powf': 0x411e0}

# ---- end of the generated block ----


def fbits(x: float) -> int:
    """The bit pattern of the float nearest x (x itself when it holds a float's value; an overflow is an infinity)."""
    return _U32.unpack(_F32.pack(f32(x)))[0]


def ffrom(u: int) -> float:
    """The float with bit pattern u, as a Python float."""
    return _F32.unpack(_U32.pack(u & M32))[0]


def dbits(x: float) -> int:
    return _U64.unpack(_F64.pack(x))[0]


def dfrom(u: int) -> float:
    return _F64.unpack(_U64.pack(u & M64))[0]


def f32(x: float) -> float:
    """A double rounded to the nearest float (an overflow to an infinity of its sign)."""
    try:
        return _F32.unpack(_F32.pack(x))[0]
    except OverflowError:
        return math.copysign(INF, x)


def _s32(u: int) -> int:
    """A 32-bit pattern as a signed int32."""
    return u - (1 << 32) if u & 0x80000000 else u


def _fmul(a: float, b: float) -> float:
    return f32(a * b)                       # float operands: the double product is exact, rounded once


def _fadd(a: float, b: float) -> float:
    return f32(a + b)                       # float operands: the double sum rounded to float is the float sum


def _fdiv(a: float, b: float) -> float:
    """a / b in float (IEEE: a zero divisor gives an infinity of the sign's product, or NaN for 0/0 and NaN/0)."""
    if b == 0.0:
        if a != a or a == 0.0:
            return NAN
        return math.copysign(INF, a) * math.copysign(1.0, b)
    return f32(a / b)


def _frinta(x: float) -> float:
    """aarch64's frinta: to the nearest integer, ties away from zero (x - trunc(x) is exact)."""
    if not math.isfinite(x) or abs(x) >= 4503599627370496.0:          # 2^52: already an integer
        return x
    t = math.copysign(float(math.trunc(x)), x)
    d = x - t
    if d >= 0.5:
        return t + 1.0
    if d <= -0.5:
        return t - 1.0
    return t


# ---- glibc's error helpers (0x4ab60-0x4abf8) ----

def _xflowf(sign: int, tbits: int) -> float:
    """xflowf (math_errf.c): (sign ? -t : t) * t in float (0x4ab74-0x4ab84)."""
    t = ffrom(tbits)
    return _fmul(-t if sign else t, t)


def _divzerof(sign: int) -> float:
    """__math_divzerof (0x4abc0): (sign ? -1.0f : 1.0f) / 0.0f."""
    return _fdiv(-1.0 if sign else 1.0, 0.0)


def _invalidf(x: float) -> float:
    """__math_invalidf (0x4abe0): (x - x) / (x - x)."""
    d = f32(x - x)
    return _fdiv(d, d)


def _signaling(u: int) -> bool:
    """issignalingf_inline: ((u ^ 0x00400000) & 0x7fffffff) > 0x7fc00000 (0x414c4-0x414d8)."""
    return ((u ^ POWF_QNAN_BIT) & 0x7FFFFFFF) > POWF_QNAN


# ---- the two inline parts (e_powf.c) ----

def _log2_inline(ix: int) -> float:
    """log2(x) x 32 for the bits ix of a positive normal (or normalised) x (0x41214-0x41288)."""
    tmp = (ix + POWF_NEG_OFF) & M32                         # 0x41218: ix - OFF
    i = (tmp >> POWF_I_SHIFT) & ((1 << POWF_I_BITS) - 1)    # 0x41224 ubfx
    top = tmp & POWF_TOP_MASK                               # 0x4121c
    iz = (ix - top) & M32                                   # 0x4122c
    k = _s32(top) >> POWF_K_SHIFT                           # 0x41248 asr (arithmetic)
    invc, logc = POWF_LOG2_INVC[i], POWF_LOG2_LOGC[i]       # 0x4124c, 0x41258
    z = ffrom(iz)                                           # 0x41230, 0x4123c: the float, exactly in double
    r = fma(invc, z, -1.0)                                  # 0x41260 fmadd d2, d5, d2, d3: invc z - 1
    y0 = float(k) + logc                                    # 0x41254 scvtf (exact), 0x4126c fadd: k + logc
    p = fma(r, POWF_A[2], POWF_A[3])                        # 0x41270 fmadd d4, d2, d7, d4: A2 r + A3
    q = fma(r, POWF_A[4], y0)                               # 0x41274 fmadd d0, d2, d5, d0: A4 r + y0
    r2 = r * r                                              # 0x41278
    y = fma(r, POWF_A[0], POWF_A[1])                        # 0x4127c fmadd d2, d2, d6, d3: A0 r + A1
    r4 = r2 * r2                                            # 0x41280
    q = fma(r2, p, q)                                       # 0x41284 fmadd d0, d5, d4, d0: p r2 + q
    return fma(y, r4, q)                                    # 0x41288 fmadd d0, d2, d3, d0: y r4 + q


def _exp2_inline(xd: float, sign_bias: int) -> float:
    """2^(xd / 32) rounded to float, its sign from sign_bias (0x412a0-0x412e8)."""
    kd = _frinta(xd)                                        # 0x412a0 frinta
    ki = int(kd)                                            # 0x412ac fcvtas: the same integer (|xd| < 2^63 here)
    r = xd - kd                                             # 0x412b4
    y = fma(r, POWF_EXP2_C[2], 1.0)                         # 0x412c8 fmadd d0, d1, d4, d0: C2 r + 1
    z = fma(r, POWF_EXP2_C[0], POWF_EXP2_C[1])              # 0x412cc fmadd d2, d1, d3, d2: C0 r + C1
    r2 = r * r                                              # 0x412d0
    t = POWF_EXP2_TAB[ki & (len(POWF_EXP2_TAB) - 1)]        # 0x412c0 and, 0x412d4 ldr
    t = (t + (((ki + sign_bias) & M64) << POWF_EXP2_SHIFT)) & M64   # 0x412c4, 0x412d8
    y = fma(z, r2, y)                                       # 0x412e0 fmadd d0, d2, d3, d0: z r2 + y
    return f32(y * dfrom(t))                                # 0x412e4 fmul, 0x412e8 fcvt


def _y_special(x: float, y: float, ix: int, iy: int, normal_x: bool) -> float:
    """y zero, infinite or NaN (0x41404 for a normal x, 0x414c0 otherwise)."""
    two_iy = (iy << 1) & M32
    if two_iy == 0:                                         # 0x41404 cbz / 0x414c0 cbnz
        return _fadd(x, y) if _signaling(ix) else 1.0       # 0x414c4-0x414e0
    if normal_x and ix == POWF_ONE_BITS:                    # 0x41408-0x41410 (never a special x)
        return _fadd(x, y) if _signaling(iy) else 1.0       # 0x414f4-0x4150c
    two_ix = (ix << 1) & M32                                # 0x41418
    if two_iy != POWF_INF2 or two_ix > POWF_INF2:           # 0x4141c-0x41424: y NaN, or x NaN
        return _fadd(x, y)                                  # 0x41428
    if two_ix == POWF_ONE2:                                 # 0x41438: |x| == 1
        return 1.0
    if (two_ix > POWF_ONE2_M1) != ((~iy & M32) >> 31):      # 0x41440-0x41458: |x| < 1 with y = +inf, or > 1 with -inf
        return 0.0
    return _fmul(y, y)                                      # 0x4145c


def powf(x: float, y: float) -> float:
    """glibc 2.34 aarch64's powf(x, y) (0x411e0) of two floats held in Python floats (a double passed is rounded to
    float first, as C's call converts it)."""
    ix, iy = fbits(x), fbits(y)
    x, y = ffrom(ix), ffrom(iy)
    two_iy = (iy << 1) & M32                                # 0x411f4
    zinf_y = ((two_iy - 1) & M32) >= POWF_ZINF              # 0x411fc, 0x41208: zeroinfnan(iy)
    sign_bias = 0
    if ((ix - POWF_X_MIN) & M32) > POWF_X_SPAN:             # 0x411f0-0x41204: x < 0x1p-126, negative, inf or NaN
        if zinf_y:                                          # 0x412f0-0x412f4
            return _y_special(x, y, ix, iy, False)
        two_ix = (ix << 1) & M32                            # 0x412f8
        if ((two_ix - 1) & M32) >= POWF_ZINF:               # 0x412fc-0x41304: x zero, infinite or NaN
            x2 = _fmul(x, x)                                # 0x41464
            sb = 0
            if ix >> 31:                                    # 0x4146c: checkint(iy) == 1 (0x4147c-0x414b8)
                e = (iy >> 23) & 0xFF
                if ((e - POWF_E_ONE) & M32) <= POWF_E_SPAN:
                    bit = 1 << (POWF_INT_E - e)
                    if not iy & (bit - 1) and iy & bit:
                        x2, sb = -x2, 1
            if two_ix == 0 and iy >> 31:                    # 0x41470-0x41478
                return _divzerof(sb)
            return _fdiv(1.0, x2) if iy >> 31 else x2       # 0x414e4-0x414ec, 0x412ec
        if ix >> 31:                                        # 0x4130c: a finite x < 0 (checkint, 0x41384-0x413bc)
            e = (iy >> 23) & 0xFF
            if e <= POWF_E_HALF:                            # 0x41388: |y| < 1
                return _invalidf(x)
            if e <= POWF_INT_E:                             # 0x41390 (above it: an even integer)
                bit = 1 << (POWF_INT_E - e)
                if iy & (bit - 1):                          # 0x413ac: not an integer
                    return _invalidf(x)
                if iy & bit:                                # 0x413b4-0x413bc: odd
                    sign_bias = POWF_SIGN_BIAS
            ix &= 0x7FFFFFFF                                # 0x413c0
        if ix <= POWF_SUBNORMAL_MAX:                        # 0x41310-0x41318: subnormal: x 2^23, then 23 << 23 off
            ix = ((fbits(_fmul(x, ffrom(POWF_TWO23_F))) & 0x7FFFFFFF) - POWF_NORM_SUB) & M32   # 0x4131c-0x41330
    elif zinf_y:                                            # 0x41208-0x4120c
        return _y_special(x, y, ix, iy, True)
    ylogx = y * _log2_inline(ix)                            # 0x41250 fcvt (exact), 0x4128c fmul
    if ((dbits(ylogx) >> 47) & 0xFFFF) > POWF_BIG:          # 0x41290-0x4129c: |y log2 x| >= 126 (scaled)
        if ylogx > POWF_OFLOW:                              # 0x41338-0x41344
            return _xflowf(sign_bias, POWF_OFLOWF_X)
        if ylogx > POWF_RND:                                # 0x41348-0x41354, 0x413c8-0x413fc: directed rounding
            tiny = ffrom(POWF_TINY_F)
            if (_fadd(1.0, tiny) != 1.0) if not sign_bias else (f32(-1.0 - tiny) != -1.0):
                return _xflowf(sign_bias, POWF_OFLOWF_X)
        if ylogx <= dfrom(POWF_UFLOW):                      # 0x41358-0x41368
            return _xflowf(sign_bias, POWF_UFLOWF_X)
        if ylogx < dfrom(POWF_MAY_UFLOW):                   # 0x4136c-0x4137c
            return _xflowf(sign_bias, POWF_MAY_UFLOWF_X)
    return _exp2_inline(ylogx, sign_bias)                   # 0x412a0

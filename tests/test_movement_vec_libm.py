# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""exact_ballpark.vec_libm: the C library's float functions and pow on tensors against the C library itself (glibc 2.34
aarch64, called through ctypes, as the reference calls it), bit for bit, on the CPU and (when one is visible) the GPU.

Once, outside the suite, every
float bit pattern went through the six one-argument functions on the CPU: 4,294,967,296 each, none differs. The suite
holds them to the library on samples that reach every branch (random bit patterns: subnormals, infinities, NaNs and
huge values; the unit interval; angles; the thresholds themselves and their neighbours), atan2f on pairs over every
quadrant and special value, pow on Destiny's kappa = pow(float in (0, 1], integer) and over its every special case,
and the GPU against the CPU. The tables are checked against the loaded library's bytes when it is the one they were
read from. The float FMA is held to the library's fmaf at constructed near ties, where rounding to double first goes
wrong, and frinta to ties away from zero.

What fusion does not change: the float functions' own fused multiply-adds never change their float results on the
patterns walked (expf and logf with every FMA unfused, atanf with fmaf rounded twice: no input in [0x30000000,
0xc3000000) differs of port_TURN1), so an unfused variant is not a catchable fault
there; the fusions are transcribed anyway, as compiled. Planted faults that do change values (a lost carry in sinf's
large-argument reduction, torch's own sine, exchanged coefficients or table entries, pow's exact-error steps
unfused) must each be caught.
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

import exact_ballpark.vec_libm as V

LIBM = Path("/usr/lib64/libm.so.6")
_M = ctypes.CDLL(ctypes.util.find_library("m"))
CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _cf(name: str, nargs: int, double: bool = False):
    fn = getattr(_M, name)
    t = ctypes.c_double if double else ctypes.c_float
    fn.restype, fn.argtypes = t, [t] * nargs
    return fn


C_F1 = {name: _cf(name, 1) for name in ("expf", "logf", "sinf", "cosf", "acosf", "atanf")}
C_ATAN2F, C_POW = _cf("atan2f", 2), _cf("pow", 2, double=True)


def u32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f_of(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def same(a: float, b: float) -> bool:
    return struct.pack("<d", a) == struct.pack("<d", b) or (a != a and b != b)


def floats(seed: int, n: int) -> list:
    """Float arguments over every branch: random bit patterns, [-1, 1], angles, around 88, -103 and 120, tiny."""
    r = random.Random(seed)
    out = [f_of(r.getrandbits(32)) for _ in range(n)]
    out += [f_of(u32(r.uniform(-1.0, 1.0))) for _ in range(n // 2)]
    out += [f_of(u32(r.uniform(-12.0, 12.0))) for _ in range(n // 2)]
    for centre in (88.0, -88.0, 88.72, -103.28, -103.97, 120.0, -120.0, 0.4375, 0.6875, 1.1875, 2.4375, 0.5, 2.0 ** 25,
                   2.0 ** -12, 2.0 ** -26, 2.0 ** -29, 2.0 ** -126, 1.0, 0.785398):
        u = u32(centre)
        out += [f_of(u + d) for d in range(-40, 41)]
    out += [0.0, -0.0, math.inf, -math.inf, math.nan, f_of(1), f_of(0x807FFFFF), 3.4028234663852886e38]
    return out


def port(fn, xs: list, device: str) -> list:
    return fn(torch.tensor(xs, dtype=torch.float64, device=device)).cpu().tolist()


@pytest.mark.parametrize("name", sorted(C_F1))
def test_one_argument_float_functions_are_the_c_librarys(name):
    xs = floats(11, 60000)
    want = [C_F1[name](x) for x in xs]
    got = port(getattr(V, name), xs, "cpu")
    bad = [(x, w, g) for x, w, g in zip(xs, want, got) if not same(w, g)]
    assert not bad, bad[:5]


def pairs(seed: int, n: int) -> tuple:
    r = random.Random(seed)
    special = [0.0, -0.0, 1.0, -1.0, math.inf, -math.inf, math.nan, f_of(1), 3.4e38, 0.5, -2.0]
    ys = [f_of(r.getrandbits(32)) for _ in range(n)] + [r.uniform(-1, 1) * 2.0 ** r.randint(-70, 70) for _ in range(n)]
    xs = [f_of(r.getrandbits(32)) for _ in range(n)] + [r.uniform(-1, 1) for _ in range(n)]
    ys += [a for a in special for _ in special] + [r.uniform(-5, 5) for _ in range(200)]
    xs += [b for _ in special for b in special] + [1.0] * 200
    return [f_of(u32(y)) for y in ys], [f_of(u32(x)) for x in xs]


def test_atan2f_is_the_c_librarys():
    ys, xs = pairs(12, 40000)
    want = [C_ATAN2F(y, x) for y, x in zip(ys, xs)]
    got = V.atan2f(torch.tensor(ys, dtype=torch.float64), torch.tensor(xs, dtype=torch.float64)).tolist()
    bad = [(y, x, w, g) for y, x, w, g in zip(ys, xs, want, got) if not same(w, g)]
    assert not bad, bad[:5]


def pow_args(seed: int, n: int) -> tuple:
    """Destiny's kappa (a float in (0, 1] to an integer power, up to 2^31 - 1), random bit patterns, results near
    overflow and in the subnormal range, negative bases with integer powers, subnormal bases, special values."""
    r = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        xs.append(f_of(r.randrange(0x3F000000, 0x3F800001)) if r.random() < 0.5 else f_of(u32(r.random())))
        ys.append(float(r.choice([r.randrange(0, 200), r.randrange(0, 5000), r.randrange(0, 2 ** 31)])))
    for _ in range(n // 2):
        xs.append(struct.unpack("<d", struct.pack("<Q", r.getrandbits(64)))[0])
        ys.append(struct.unpack("<d", struct.pack("<Q", r.getrandbits(64)))[0])
    for _ in range(n // 2):
        x = r.uniform(0.5, 2.0)
        xs.append(x)
        ys.append(r.choice([-1, 1]) * r.uniform(700.0, 1100.0) / math.log(x) if abs(math.log(x)) > 1e-9 else 3.0)
        xs.append(-r.uniform(0.0, 10.0))
        ys.append(float(r.randrange(-60, 60)))
        xs.append(r.random() * 2.0 ** -1060)
        ys.append(r.uniform(-1.2, 1.2))
    special = [0.0, -0.0, 1.0, -1.0, math.inf, -math.inf, math.nan, 5e-324, -5e-324, 1.7e308, 2.0, 0.5, -2.0, -0.5,
               3.0, -3.0, 2.0 ** -1022, 1 - 2 ** -53, 1 + 2 ** -52, 2.0 ** 63, 2.0 ** -66, -(2.0 ** 53), 1023.5]
    xs += [a for a in special for _ in special]
    ys += [b for _ in special for b in special]
    return xs, ys


def test_pow_is_the_c_librarys():
    xs, ys = pow_args(13, 30000)
    want = [C_POW(x, y) for x, y in zip(xs, ys)]
    got = V.pow(torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)).tolist()
    bad = [(x, y, w, g) for x, y, w, g in zip(xs, ys, want, got) if not same(w, g)]
    assert not bad, bad[:5]


def test_the_samples_reach_every_branch():
    xs = floats(11, 60000)
    top = [(u32(x) >> 20) & 0x7FF for x in xs]
    assert sum(t > 0x42A and -103.97 < x < -103.28 for t, x in zip(top, xs)) >= 5          # expf's may_uflowf
    assert sum(0x42F <= t < 0x7F8 for t in top) > 1000 and sum(t < 0x398 for t in top) > 1000   # sinf large, tiny
    assert sum(0x3F4 <= t < 0x42F for t in top) > 1000                                    # reduce_fast
    assert sum(0 < u32(x) < 0x800000 for x in xs) > 10                                    # logf's subnormals
    for lo, hi in ((0.0, 0.4375), (0.4375, 0.6875), (0.6875, 1.1875), (1.1875, 2.4375), (2.4375, 2.0 ** 25)):
        assert sum(lo <= abs(x) < hi for x in xs) > 100                                   # atanf's five ranges
    for lo, hi in ((0.0, 0.5), (0.5, 1.0)):
        assert sum(lo <= x < hi for x in xs) > 100 and sum(-hi < x <= -lo for x in xs) > 100   # acosf's three
    xp, yp = pow_args(13, 30000)
    ehi = []
    for x, y in zip(xp, yp):
        if x > 0 and math.isfinite(x) and math.isfinite(y) and x != 1.0:
            ehi.append(abs(y * math.log(x)))
    assert sum(512 <= e < 1024 for e in ehi) > 100                                        # pow's specialcase
    assert sum(e < 2.0 ** -54 for e in ehi) > 5 and sum(e >= 1024 for e in ehi) > 100     # tiny, huge


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
def test_the_gpu_gives_the_cpus_bits():
    xs = floats(14, 40000)
    for name in sorted(C_F1):
        assert all(same(a, b) for a, b in zip(port(getattr(V, name), xs, "cpu"), port(getattr(V, name), xs, "cuda")))
    ys, xa = pairs(15, 20000)
    cpu = V.atan2f(torch.tensor(ys, dtype=torch.float64), torch.tensor(xa, dtype=torch.float64)).tolist()
    gpu = V.atan2f(torch.tensor(ys, dtype=torch.float64, device="cuda"),
                   torch.tensor(xa, dtype=torch.float64, device="cuda")).cpu().tolist()
    assert all(same(a, b) for a, b in zip(cpu, gpu))
    xp, yp = pow_args(16, 20000)
    cpu = V.pow(torch.tensor(xp, dtype=torch.float64), torch.tensor(yp, dtype=torch.float64)).tolist()
    gpu = V.pow(torch.tensor(xp, dtype=torch.float64, device="cuda"),
                torch.tensor(yp, dtype=torch.float64, device="cuda")).cpu().tolist()
    assert all(same(a, b) for a, b in zip(cpu, gpu))


def test_the_tables_are_the_loaded_librarys():
    """The constants were read out of this machine's libm.so.6: check them
    against its bytes at the referenced addresses, when it is that library."""
    if not LIBM.is_file() or hashlib.sha256(LIBM.read_bytes()).hexdigest() != V.LIBM_SHA256:
        pytest.skip("not the libm.so.6 the tables were read from")
    b = LIBM.read_bytes()
    d = lambda off, n: [struct.unpack_from("<d", b, off + 8 * i)[0] for i in range(n)]
    q = lambda off, n: [struct.unpack_from("<Q", b, off + 8 * i)[0] for i in range(n)]
    assert list(V.EXP2F_TAB) == q(0x76870, 32) and d(0x76870 + 296, 1) == [V.EXPF_INVLN2]
    assert list(V.EXPF_C) == d(0x76870 + 304, 3)
    assert [c for pair in V.LOGF_TAB for c in pair] == d(0x769B8, 32) and d(0x769B8 + 256, 4) == [V.LOGF_LN2,
                                                                                                    *V.LOGF_A]
    t = d(0x76D80, 14)
    assert list(V.SINCOSF_SIGN) == t[:4] and [V.HPI_INV, V.HPI] == t[4:6] and list(V.COSF_C) == t[6:11]
    assert list(V.SINF_S) == t[11:14] and list(V.COSF_C_TABLE1) == d(0x76DF0 + 48, 5)
    assert list(V.INV_PIO4) == [struct.unpack_from("<I", b, 0x76D20 + 4 * i)[0] for i in range(24)]
    assert d(0x6EE50, 1) == [V.PI63]
    assert list(V.ATANF_HI) + list(V.ATANF_LO) == [struct.unpack_from("<f", b, 0x70A90 + 4 * i)[0] for i in range(8)]
    assert d(0x75828, 9) == [V.POW_LN2HI, V.POW_LN2LO, *V.POW_A]
    assert [c for e in V.POW_TAB for c in e] == [x for i in range(128) for j, x in enumerate(d(0x75828 + 72 + 32 * i, 4))
                                                 if j != 1]


# ---- the float FMA and frinta on their own; planted faults ----

C_FMAF = _cf("fmaf", 3)


def near_ties(seed: int, n: int) -> list:
    """(a, b, c) floats whose exact a b + c lies just below a float midpoint (by half an ulp times 2^-k, 30 <= k <= 46):
    c = f, a float with an odd significand, and a b = (2^(k/2) - 1)(2^(k/2) + 1) scaled to half an ulp of f times
    (1 - 2^-k). Rounded to double first, the sum lands on the midpoint and ties to even, the wrong neighbour."""
    r = random.Random(seed)
    out = []
    while len(out) < n:
        f = f_of(r.randrange(0x3F800001, 0x46FFFFFF, 2) | 1)            # odd significand, in [1, 32768)
        k = r.randrange(30, 47, 2)
        h = (math.frexp(f)[1] - 1) - 24                                  # half an ulp of f is 2^h
        e1 = r.randrange(-20, 10)
        a = math.ldexp(2 ** (k // 2) - 1, e1)
        b = math.ldexp(2 ** (k // 2) + 1, h - k - e1)
        if f_of(u32(a)) == a and f_of(u32(b)) == b:
            out.append((a, b, f) if r.random() < 0.5 else (-a, b, -f))
    return out


def test_fmaf_is_the_c_librarys_also_at_near_ties():
    r = random.Random(17)
    triples = near_ties(18, 2000) + [(f_of(r.getrandbits(32) & 0x7EFFFFFF), f_of(r.getrandbits(32) & 0x7EFFFFFF),
                                      f_of(r.getrandbits(32) & 0x7EFFFFFF)) for _ in range(5000)]
    triples = [t for t in triples if all(math.isfinite(v) for v in t) and math.isfinite(C_FMAF(*t))]
    a, b, c = (torch.tensor([t[i] for t in triples], dtype=torch.float64) for i in range(3))
    got = V.fmaf(a, b, c).tolist()
    assert all(same(C_FMAF(*t), g) for t, g in zip(triples, got))
    plain = V.f32(a * b + c).tolist()                                    # double rounding: wrong at the near ties
    assert sum(not same(C_FMAF(*t), g) for t, g in zip(triples, plain)) >= 1000


def test_frinta_rounds_ties_away_from_zero():
    z = torch.tensor([0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999999999999994, 4096.5, -0.3, 1e300], dtype=torch.float64)
    assert V.frinta(z).tolist() == [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.0, 4097.0, -0.0, 1e300]
    assert torch.round(z).tolist() != V.frinta(z).tolist()              # ties to even would differ


def _no_carry(ix):
    idx = (ix >> 26) & 15
    m = ((ix & 0x7FFFFF) | 0x800000) << ((ix >> 23) & 7)
    tab = V._tab("inv_pio4", ix.device)
    hi = (m * tab[idx]) & 0xFFFFFFFF
    p1, p2 = m * tab[idx + 4], m * tab[idx + 8]
    lo = (p2 >> 32) + (p1 & 0xFFFFFFFF)
    hi = (hi + (p1 >> 32)) & 0xFFFFFFFF                    # the carry from the low half dropped
    lo = lo & 0xFFFFFFFF
    n = ((hi + 0x20000000) & 0xFFFFFFFF) >> 30
    hi = (hi - (n << 30)) & 0xFFFFFFFF
    v = ((hi ^ 0x80000000) - 0x80000000) * 4294967296 + lo
    return n, v.to(torch.float64) * V.PI63


def _swap(name: str, i: int, j: int):
    def patch(mp):
        t = list(getattr(V, name))
        t[i], t[j] = t[j], t[i]
        mp.setattr(V, name, tuple(t))
        V._TABS.clear()
    return patch


FAULTS = {
    # sinf/cosf: reduce_large without the carry between its 32-bit halves
    "cosf_reduce_large_no_carry": ("cosf", lambda mp: mp.setattr(V, "_reduce_large", _no_carry)),
    # sinf: torch's own sine (SLEEF on the CPU, CUDA's on the GPU) in place of the transcription
    "sinf_torch": ("sinf", lambda mp: mp.setattr(V, "sinf", lambda x: torch.sin(x.to(torch.float32))
                                                 .to(torch.float64))),
    # acosf: two polynomial coefficients exchanged
    "acosf_coefficients": ("acosf", lambda mp: (mp.setattr(V, "ACOSF_PS4", V.ACOSF_PS5),
                                                mp.setattr(V, "ACOSF_PS5", V.ACOSF_PS4))),
    # atanf: the table's low parts in the high parts' order exchanged
    "atanf_table": ("atanf", _swap("ATANF_LO", 1, 2)),
    # logf: two table rows exchanged
    "logf_table": ("logf", _swap("LOGF_TAB", 3, 4)),
    # expf: two table words exchanged
    "expf_table": ("expf", _swap("EXP2F_TAB", 5, 6)),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    name, patch = FAULTS[fault]
    xs = floats(11, 60000)
    want = [C_F1[name](x) for x in xs]
    patch(monkeypatch)
    try:
        got = port(getattr(V, name), xs, "cpu")
    finally:
        V._TABS.clear()
    assert any(not same(w, g) for w, g in zip(want, got)), f"the planted fault {fault} went unnoticed"


def test_pow_without_its_fused_exact_errors_is_caught(monkeypatch):
    """pow with its fnmsub steps (a b - (a b rounded): log's lo3, y log x's low part) left unfused."""
    xs, ys = pow_args(13, 30000)
    want = [C_POW(x, y) for x, y in zip(xs, ys)]
    real = V.fma

    def unfused_exact_errors(a, b, c, b_split=None):
        if torch.is_tensor(b) and torch.is_tensor(c) and torch.equal(c, -(a * b)):
            return a * b + c
        return real(a, b, c, b_split)
    monkeypatch.setattr(V, "fma", unfused_exact_errors)
    got = V.pow(torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)).tolist()
    assert any(not same(w, g) for w, g in zip(want, got))

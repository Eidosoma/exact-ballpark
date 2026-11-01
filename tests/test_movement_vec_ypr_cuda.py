# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The C library's double sin, cos, asin and atan2 on a CUDA device (exact_ballpark.vec_ypr_cuda) and the yaw, pitch and
roll without dynamical orientation computed there (VecOrientPark(..., ypr="cuda")), bit for bit: doubles by their bits,
two NaNs equal, integers exactly. No tolerance anywhere.

- The device functions against their scalar transcriptions (exact_ballpark.dlibm's sin, cos and sincos,
  exact_ballpark.dlibm_inv's asin and atan2) and against the C library itself (glibc 2.34 aarch64 through ctypes) on the
  CPU tests' arguments over every branch (tests/test_movement_dlibm.py, test_movement_dlibm_inv.py,
  test_movement_dlibm_sincos.py, sincos's windows among them); the device sin and cos also against
  exact_ballpark.vec_modes_cuda's, whose source they copy (the copy held to it line for line).
- The park against the scalar reference (exact_ballpark.full): tests/turn_cases.py's "vec" scenarios (one to six free
  ships without dynamical orientation under GOTO, GotoDirection and STOP, every setter in flight), every ball-state
  field every step, development seed 1 and held-out seed 2; the reference rows are the ones
  tests/test_movement_turn_oracle.py holds to Destiny's own code compiled verbatim, so this is exact against Destiny.
- The park against the host path (ypr="host", Python's math on the host): random ships at widths 1 to 65,536 with
  commands and setters in flight, every field every tick, on the GPU and against the CPU.
- A tick replayed from a CUDA graph (YprTickGraph) is the eager tick: no host round trip is left in a tick.
- Planted faults must each be caught: an unfused step of asin, atan2's error term of the quotient unfused, an entry of
  each table wrong in its last bit (asncs, cij, sin's table), pi/2's low half dropped near 1, CUDA's own atan2 (refused
  by the source scan too).

The CPU part (the generated header, the source scan, the copy's identity, the build, the argument checks) runs
everywhere; the rest skips where no GPU is visible.
"""

from __future__ import annotations

import ctypes
import math
import random
import re
import struct

import pytest
import torch

import orient_batch as ob
import test_movement_dlibm as TD
import test_movement_dlibm_inv as TI
import test_movement_dlibm_sincos as TS
import turn_cases as tc
from exact_ballpark import dlibm as D
from exact_ballpark import dlibm_inv as I
from exact_ballpark import vec_modes_cuda as M
from exact_ballpark import vec_orient as V
from exact_ballpark import vec_orient_cuda as K
from exact_ballpark import vec_ypr_cuda as Y

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
try:
    K._nvcc()
    NVCC = True
except RuntimeError:
    NVCC = False
needs_nvcc = pytest.mark.skipif(not NVCC, reason="nvcc not found")
F64 = torch.float64
_REF: dict = {}
_C = ctypes.CDLL("libm.so.6")
for _name in ("sin", "cos", "asin"):
    getattr(_C, _name).restype = ctypes.c_double
    getattr(_C, _name).argtypes = [ctypes.c_double]
_C.atan2.restype = ctypes.c_double
_C.atan2.argtypes = [ctypes.c_double, ctypes.c_double]


def d64(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def same_bits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise: equal bit for bit, or both NaN."""
    return (a.view(torch.int64) == b.view(torch.int64)) | (torch.isnan(a) & torch.isnan(b))


# ============================================================================== the CPU part: header, scan, build

def parse_header(h: str) -> tuple:
    defs = {k: int(v, 16) for k, v in re.findall(r"#define (\w+) 0x([0-9a-f]+)(?:ULL|u)\n", h)}
    arrays = {k: [int(w, 16) for w in re.findall(r"0x([0-9a-f]+)", body)]
              for k, body in re.findall(r"(\w+)\[\d+\] = \{(.*?)\};", h)}
    ints = {k: int(v) for k, v in re.findall(r"#define (\w+) \((-?\d+)\)\n", h)}
    return defs, arrays, ints


def test_the_generated_header_is_dlibms_and_dlibm_invs_tables():
    """Every constant and table of the device code, read back from the header, is the Python modules' own value."""
    defs, arrays, ints = parse_header(Y.tables_header())
    for name in Y.SIN_DOUBLES:
        assert d64(defs[f"YL_{name}"]) == getattr(D, name), name
    for name in Y.SIN_WORDS:
        assert defs[f"YL_{name}"] == getattr(D, name), name
    for name in Y.SIN_INTS:
        assert ints[f"YL_{name}"] == getattr(D, name), name
    assert [d64(w) for w in arrays["YL_SINCOSTAB"]] == list(D.SINCOSTAB)
    assert [d64(w) for w in arrays["YL_BR_TOVERP"]] == list(D.BR_TOVERP)
    names = [n for n in dir(I) if n.startswith(("ASIN_", "ATAN2_")) and n != "ASIN_DBL_MIN" and n != "ATAN2_DBL_MIN"]
    for name in names:
        key = f"YI_{name}"
        v = getattr(I, name)
        if isinstance(v, tuple):
            assert [d64(w) for w in arrays[key]] == list(v), name
        elif isinstance(v, float):
            assert d64(defs[key]) == v, name
        elif key in ints:
            assert ints[key] == v, name
        else:
            assert defs[key] == v, name
    assert len(arrays) == 6 and len(defs) + len(ints) + len(arrays) == len(Y.table_values())


def test_a_perturbed_header_differs_in_one_bit_of_one_entry():
    a, b = Y.tables_header(), Y.tables_header({"YI_ATAN2_CIJ": 701})
    _, ta, _ = parse_header(a)
    _, tb, _ = parse_header(b)
    diff = [(k, i) for k in ta for i, (x, y) in enumerate(zip(ta[k], tb[k])) if x != y]
    assert diff == [("YI_ATAN2_CIJ", 701)] and ta["YI_ATAN2_CIJ"][701] ^ tb["YI_ATAN2_CIJ"][701] == 1
    with pytest.raises(KeyError):
        Y.tables_header({"YI_NONE": 0})


def test_the_source_calls_no_function_of_cudas_math_library():
    assert K.scan_source(Y.SOURCE.read_text()) == []
    planted = "out[i] = atan2(a[i], b[i]);\nconst double s = ::asin(x) + g_asin(y);\n"
    assert K.scan_source(planted) == [("atan2", 1), ("asin", 2)]


def _sin_cos_block(text: str, first: str, last: str) -> list:
    """The lines from `first` to the end of the function starting at `last`, whitespace collapsed and wrapped lines
    joined (so a line broken in two reads as one)."""
    start = text.index(first)
    end = text.index("\n}\n", text.index(last)) + 3
    body = re.sub(r"\s+", " ", text[start:end])
    return body.replace("( ", "(").replace(" )", ")")


def test_the_sin_and_cos_are_vec_modes_cudas_code():
    """The device sin and cos are vec_modes_cuda's own source (the transcription of dlibm), unchanged but for their
    constants' names (DL_ to YL_, KD to YKD) and two wrapped lines."""
    first, last = "static __device__ __forceinline__ double taylor_sin(", "static __device__ __noinline__ double g_cos("
    theirs = _sin_cos_block(M.SOURCE.read_text(), first, last)
    ours = _sin_cos_block(Y.SOURCE.read_text(), first, last)
    assert re.sub(r"\bDL_", "YL_", theirs).replace("KD(", "YKD(") == ours


@needs_nvcc
def test_the_build_pins_every_rounding():
    m = Y.build()
    assert m["ptx_check"]["ok"] and not m["ptx_check"]["refused"]
    for flag in ("-fmad=false", "-prec-div=true", "-prec-sqrt=true", "-ftz=false"):
        assert flag in m["flags"]
    assert not any("fast" in f and "no-fast" not in f for f in m["flags"])
    assert m["header_sha256"] == Y._sha(Y.tables_header().encode())
    assert m["source_sha256"] == Y._sha(Y.SOURCE.read_bytes())
    ops = m["ptx_check"]["fp_ops"]
    assert ops.get("fma.rn.f64", 0) and not any(k.endswith(".f32") for k in ops)    # only double, fused explicitly
    assert not any(k.startswith(("mad", "ex2", "lg2", "sin", "cos", "rsqrt", "sqrt", "rcp")) or ".ftz" in k
                   for k in ops)


def test_the_arguments_are_checked():
    with pytest.raises(ValueError):
        V.VecOrientPark(2, ypr="numba")
    with pytest.raises(ValueError):
        V.VecOrientPark(2, ypr="cuda")                                          # the CPU
    if CUDA:
        with pytest.raises(ValueError):
            V.VecOrientPark(2, device="cuda", dynamical_orientation=True, ypr="cuda")
        a = torch.zeros(3, dtype=F64, device="cuda")
        with pytest.raises(ValueError):
            Y.dlibm("atan2", a)
        with pytest.raises(ValueError):
            Y.dlibm("asin", a, a)
        with pytest.raises(ValueError):
            Y.dlibm("tan", a)
        with pytest.raises(ValueError):
            Y.asin(a.float())
    with pytest.raises(ValueError):
        Y.asin(torch.zeros(3, dtype=F64))                                       # not on a CUDA device
    assert V.VecOrientPark(2).ypr == "host"


# =================================================================================== the device functions, samples

def samples(name: str) -> tuple:
    """(a, b or None) of the CPU tests' arguments over every branch."""
    if name in ("sin", "cos", "asin_sin"):
        return TD.arguments({"sin": 31, "cos": 32, "asin_sin": 33}[name], 60000), None
    if name in ("sincos_sin", "sincos_cos"):
        return TD.arguments({"sincos_sin": 37, "sincos_cos": 38}[name], 30000) + TS.windows(256), None
    if name == "asin":
        return TI.asin_arguments(34, 60000), None
    pairs = TI.atan2_arguments(35, 60000)
    return [y for y, _ in pairs], [x for _, x in pairs]


def scalar(name: str, a: float, b: float | None = None) -> tuple:
    """(the scalar transcription's value, the C library's value)."""
    if name == "sin":
        return D.sin(a), _C.sin(a)
    if name == "cos":
        return D.cos(a), _C.cos(a)
    if name == "asin":
        return I.asin(a), _C.asin(a)
    if name == "atan2":
        return I.atan2(a, b), _C.atan2(a, b)
    if name in ("sincos_sin", "sincos_cos"):
        j = 0 if name == "sincos_sin" else 1
        return D.sincos(a)[j], TS.c_sincos(a)[j]
    return I.asin(D.sin(a)), _C.asin(_C.sin(a))


def device(name: str, a: list, b: list | None, lib: dict | None = None) -> torch.Tensor:
    ta = torch.tensor(a, dtype=F64, device="cuda")
    tb = None if b is None else torch.tensor(b, dtype=F64, device="cuda")
    return Y.dlibm(name, ta, tb, lib=lib).cpu()


@needs_cuda
@pytest.mark.parametrize("name", ["sin", "cos", "asin", "atan2", "asin_sin", "sincos_sin", "sincos_cos"])
def test_the_device_functions_are_the_transcriptions_and_the_c_librarys(name):
    a, b = samples(name)
    got = device(name, a, b)
    want = [scalar(name, x, None if b is None else b[i]) for i, x in enumerate(a)]
    twin = torch.tensor([w[0] for w in want], dtype=F64)
    clib = torch.tensor([w[1] for w in want], dtype=F64)
    assert bool(same_bits(twin, clib).all())
    bad = (~same_bits(got, clib)).nonzero().flatten().tolist()
    assert not bad, (len(bad), [(a[i], None if b is None else b[i]) for i in bad[:5]])


@needs_cuda
def test_the_device_sin_and_cos_are_vec_modes_cudas():
    """Random bit patterns (every exponent) and arguments up to 1e9: the two kernels' sin and cos agree bit for bit."""
    g = torch.Generator().manual_seed(36)
    u = torch.randint(-(1 << 63), (1 << 63) - 1, (1 << 20,), generator=g, dtype=torch.int64)
    x = torch.cat([u.view(F64), (torch.rand(1 << 20, generator=g, dtype=F64) - 0.5) * 2e9]).to("cuda")
    for name in ("sin", "cos"):
        assert bool(same_bits(Y.dlibm(name, x), M.dlibm(name, x)).all()), name


# =========================================================================================== the park, scenarios

def cases(seed: int) -> list:
    return [c for c in tc.cases(seed) if c["kind"] == "vec"]


def reference(seed: int) -> dict:
    if seed not in _REF:
        _REF[seed] = ob.reference_rows(cases(seed))
    return _REF[seed]


def use_ypr(park) -> None:
    if not park.dynamical_orientation:
        park.ypr = "cuda"


def park_differences(seed: int, cs=None) -> list:
    """The park (ypr="cuda") against the reference rows of cs (all of the seed's "vec" scenarios by default)."""
    want = reference(seed) if cs is None else ob.reference_rows(cs)
    return ob.differences(want, ob.batched(cases(seed) if cs is None else cs, "cuda", park_hook=use_ypr))


@needs_cuda
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_park_matches_the_reference(seed):
    bad = park_differences(seed)
    assert not bad, (len(bad), bad[:8])


# ================================================================================ the host path, at every width

def ships(n: int, seed: int) -> list:
    """Hulls over Destiny's ranges, some at rest (the goto direction alone then sets the heading)."""
    r = random.Random(seed)
    return [{"mass": 10 ** r.uniform(5.5, 9.5), "radius": r.uniform(20.0, 300.0), "max_vel": r.uniform(20.0, 900.0),
             "pos": [r.uniform(-1e4, 1e4) for _ in range(3)],
             "vel": r.choice([[0.0, 0.0, 0.0], [r.uniform(-300.0, 300.0) for _ in range(3)]]),
             "agility": r.uniform(0.05, 3.0), "speed_fraction": r.choice([1.0, r.uniform(0.05, 1.0)])}
            for _ in range(n)]


FIELDS = ("pos", "vel", "old_pos", "old_vel", "goto", "mode", "last_g", "mass", "agility", "speed_fraction", "max_vel",
          "time_factor", "yaw", "old_yaw", "pitch", "old_pitch", "roll", "old_roll", "roll_speed", "old_roll_speed",
          "yaw_delta", "radius", "refiled")


def state_differences(a, b) -> list:
    bad = []
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f).to(getattr(a, f).device)
        n = int((~same_bits(x, y)).sum()) if x.dtype == F64 else int((x != y).sum())
        if n:
            bad.append((f, n))
    return bad


def commands(park, gen: torch.Generator, tick: int) -> None:
    """Re-aims, stops and the setters in flight (refused and non-finite values among them, a goto onto the ship's own
    position, a velocity set to zero); one draw for every park, every value made on the CPU, then moved."""
    n, dev = park.n, park.dev
    u = lambda *s: torch.rand(*s, generator=gen, dtype=F64)
    pick = lambda p: (torch.rand(n, generator=gen) < p).to(dev)
    if tick % 5 == 0:
        d = torch.randn(n, 3, generator=gen, dtype=F64)
        park.goto_point(pick(0.8), park.pos + (d * (u(n, 1) * 4e4 + 50.0)).to(dev))
        park.goto_point(pick(0.03), park.pos.clone())
    if tick % 5 == 2:
        park.goto_direction(pick(0.3), torch.randn(n, 3, generator=gen, dtype=F64).to(dev))
        park.stop(pick(0.15))
    if tick % 5 == 3:
        v = (u(n, 3) - 0.5) * 600.0
        v = torch.where((u(n) < 0.1)[:, None], 0.0, v)
        park.set_velocity(pick(0.1), v.to(dev))
        m = torch.where(u(n) < 0.05, math.nan, 10 ** (u(n) * 4.0 + 5.5))
        park.set_mass(pick(0.2), m.to(dev))
        park.set_agility(pick(0.2), torch.where(u(n) < 0.05, 0.0, u(n) * 3.0).to(dev))
        park.set_speed_fraction(pick(0.2), u(n).to(dev))
        park.set_max_velocity(pick(0.1), (u(n) * 900.0).to(dev))


def run_pair(a, b, ticks: int, seed: int) -> list:
    ga, gb = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
    for tick in range(ticks):
        commands(a, ga, tick)
        commands(b, gb, tick)
        a.evolve()
        b.evolve()
        bad = state_differences(a, b)
        if bad:
            return [(tick, bad)]
    return []


def park_pair(n: int, seed: int, host_device: str = "cuda") -> tuple:
    rows = ships(n, seed)
    k = V.VecOrientPark(n, device="cuda", ypr="cuda")
    h = V.VecOrientPark(n, device=host_device)
    k.load(rows)
    h.load(rows)
    return k, h


@needs_cuda
@pytest.mark.parametrize("width", [1, 64, 1000, 65536])
def test_the_park_matches_the_host_path_on_the_gpu(width):
    k, h = park_pair(width, 200 + width)
    assert not state_differences(k, h)                                  # the snap of AddBall
    assert not run_pair(k, h, 12 if width > 10000 else 30, width)


@needs_cuda
def test_the_park_matches_the_host_path_on_the_cpu():
    """The chain to the reference: the device's yaw, pitch and roll against the host path on the CPU."""
    k, h = park_pair(500, 9, host_device="cpu")
    assert not run_pair(k, h, 25, 9)


# ================================================================================================ CUDA graphs

@needs_cuda
def test_a_tick_replayed_from_a_cuda_graph_is_the_eager_tick():
    """YprTickGraph's replays against eager ticks of the same park, commands and setters in flight, every field every
    tick; and both against the host path."""
    rows = ships(777, 11)
    e = V.VecOrientPark(777, device="cuda", ypr="cuda")
    g = V.VecOrientPark(777, device="cuda", ypr="cuda")
    h = V.VecOrientPark(777, device="cuda")
    for p in (e, g, h):
        p.load(rows)
    graph = Y.YprTickGraph(g)
    gens = [torch.Generator().manual_seed(12) for _ in range(3)]
    for tick in range(30):
        for p, gen in zip((e, g, h), gens):
            commands(p, gen, tick)
        e.evolve()
        graph.tick()
        h.evolve()
        assert not state_differences(e, g), tick
        assert not state_differences(e, h), tick
    assert e.current_time == g.current_time == 30
    with pytest.raises(ValueError):
        Y.YprTickGraph(h)


# ============================================================================================== planted faults

FAULTS = {          # name: (source edits (old, new, count), table perturbation, the checks that must catch it)
    # asin's table ranges: t = a1 xx + p unfused
    "asin_unfused": ([("const double t = dfma(YASNCS(n + 1), xx, p);",
                       "const double t = dadd(dmul(YASNCS(n + 1), xx), p);", 1)], None, ("asin",)),
    # atan2's EMULV unfused: the quotient's error du lost (an unfused last step of a polynomial is not planted: the
    # term it rounds is too small to reach a result)
    "atan2_unfused": ([("const double vv = dfnmsub(ax, u, v);", "const double vv = dsub(dmul(ax, u), v);", 1),
                       ("const double vv = dfnmsub(ay, u, v);", "const double vv = dsub(dmul(ay, u), v);", 1)], None,
                      ("atan2", "scenarios")),
    # an entry of asncs, of cij and of sin's table wrong in its last bit
    "asncs_entry": ([], {"YI_ASIN_ASNCS": I.ASIN_N_H_OFF + 3 * 4 * 20 + 9}, ("asin",)),
    "cij_entry": ([], {"YI_ATAN2_CIJ": 7 * 100 + 1}, ("atan2",)),
    "sin_table": ([], {"YL_SINCOSTAB": 4 * 50 + 2}, ("sin",)),
    # near 1: pi/2's low half dropped from the correction
    "asin_hp1": ([("double cor = dfmsub(cc, YKD(YI_ASIN_TWO), YKD(YI_ASIN_HP1));",
                   "double cor = dfmsub(cc, YKD(YI_ASIN_TWO), 0.0);", 1)], None, ("asin",)),
    # CUDA's own atan2 for the yaw (the source scan refuses it)
    "cuda_atan2": ([("case 3: out[i] = g_atan2(a[i], b[i]); break;", "case 3: out[i] = atan2(a[i], b[i]); break;", 1)],
                   None, ("atan2",)),
}


def faulted_library(name: str) -> dict:
    """The library built from the source with one planted fault (cached under the build directory)."""
    edits, perturb, _ = FAULTS[name]
    text = Y.SOURCE.read_text()
    for old, new, count in edits:
        assert text.count(old) == count, (name, old)
        text = text.replace(old, new)
    d = Y.build_dir() / "faults"
    d.mkdir(parents=True, exist_ok=True)
    src = d / f"{name}.cu"
    if not src.exists() or src.read_text() != text:
        src.write_text(text)
    return Y.load_library(source=src, perturb=perturb, checks=False)


def fault_samples(name: str) -> tuple:
    """The planted-fault test's arguments of tests/test_movement_dlibm_inv.py and tests/test_movement_dlibm.py, which
    reach the faulted cells."""
    if name == "asin":
        xs = TI.asin_arguments(6, 20000) + [d64(I.bits(0.5) + (j << 40) + (80 << 43)) for j in range(-2000, 2000)]
        return xs, None
    if name == "atan2":
        pairs = TI.atan2_arguments(7, 20000) + [(1.0, 256 / (116 + k / 997)) for k in range(1000)]
        return [y for y, _ in pairs], [x for _, x in pairs]
    return TD.arguments(5, 30000) + [50 / 128 + i * 1e-6 for i in range(-1000, 1000)], None


@needs_cuda
@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    lib = faulted_library(fault)
    if fault.startswith("cuda_"):
        assert lib["manifest"]["source_scan"], "the source scan missed it"
    for check in FAULTS[fault][2]:
        if check == "scenarios":
            with monkeypatch.context() as mp:
                mp.setattr(Y, "load_library", lambda *a, **k: lib)
                cs = [c for c in cases(1) if c["friction"] == 1.0e6][:60]
                assert park_differences(1, cs), f"the planted fault {fault} went unnoticed by the scenarios"
        else:
            a, b = fault_samples(check)
            got = device(check, a, b, lib=lib)
            want = device(check, a, b)
            assert not bool(same_bits(got, want).all()), f"the planted fault {fault} went unnoticed by {check}"

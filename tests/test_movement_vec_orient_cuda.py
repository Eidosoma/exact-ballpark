# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Dynamical orientation as one CUDA kernel (exact_ballpark.vec_orient_cuda; VecOrientPark(..., kernel="cuda")) against
the torch module (exact_ballpark.vec_orient, kernel="torch") and the scalar reference (exact_ballpark.full), bit for bit:
doubles by their bits, two NaNs equal, integers exactly. No tolerance anywhere.

- The reference: tests/turn_cases.py's "vec_do" scenarios (one to six free ships turning hard under GOTO,
  GotoDirection and STOP, every setter in flight, the refused values among them), every ball-state field every step;
  development seed 1 and held-out seed 2. The reference rows are the ones tests/test_movement_turn_oracle.py holds to
  Destiny's own code compiled with DirectXMath, so this is exact against Destiny.
- The torch module: random ships turning hard at widths 1 to 262,144 with commands and the turning setters in flight
  (NaN, zero and refused values among them), every field every tick.
- A ship's result is the same at every width and block size, and a tick replayed from a CUDA graph is the eager tick.
- The kernel's own float functions equal vec_libm's and the C library's (glibc 2.34 aarch64) on samples over every
  branch, and its native fused multiply-adds equal vec_exp's and vec_libm's exact emulations.
- Planted faults must each be caught: a contracted multiply-add, CUDA's own sinf and expf (also refused by the
  source scan), a wrong table entry, a round of the torque search skipped, kappa1 by pow.

The CPU part (the generated header against vec_libm's and vec_exp's tables, the source scan, the PTX check, the build,
the argument checks) runs everywhere; the rest skips where no GPU is visible.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import random
import re
import struct

import pytest
import torch

import orient_batch as ob
import test_movement_vec_exp as TE
import test_movement_vec_libm as TL
import turn_cases as tc
from exact_ballpark import vec_exp as E
from exact_ballpark import vec_libm as L
from exact_ballpark import vec_orient as V
from exact_ballpark import vec_orient_cuda as K

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


def d64(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def f32b(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u))[0]


def same_bits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise: equal bit for bit, or both NaN."""
    return (a.view(torch.int64) == b.view(torch.int64)) | (torch.isnan(a) & torch.isnan(b))


# ============================================================================== the CPU part: header, scan, build

def parse_header(h: str) -> tuple:
    defs = {k: int(v, 16) for k, v in re.findall(r"#define (\w+) 0x([0-9a-f]+)(?:ULL|u)\n", h)}
    arrays = {k: [int(w, 16) for w in re.findall(r"0x([0-9a-f]+)", body)]
              for k, body in re.findall(r"(\w+)\[\d+\] = \{(.*?)\};", h)}
    ints = {k: int(v) for k, v in re.findall(r"#define (\w+) (\d+)\n", h)}
    return defs, arrays, ints


def test_the_generated_header_is_vec_libms_and_vec_exps_tables():
    """Every constant and table of the kernel, read back from the header, is the Python module's own value."""
    defs, arrays, ints = parse_header(K.tables_header())
    dd = lambda k: d64(defs[k])
    ff = lambda k: f32b(defs[k])
    assert [dd(f"K_EXPF_C{i}") for i in range(3)] == list(L.EXPF_C) and dd("K_EXPF_INVLN2") == L.EXPF_INVLN2
    assert (ff("K_EXPF_OFLOW"), ff("K_EXPF_UFLOW"), ff("K_EXPF_MAY_UFLOW"), ff("K_MAY_UFLOWF")) == (
        L.EXPF_OFLOW, L.EXPF_UFLOW, L.EXPF_MAY_UFLOW, L.MAY_UFLOWF)
    assert arrays["T_EXP2F"] == list(L.EXP2F_TAB)
    assert [d64(w) for w in arrays["T_LOGF_INVC"]] == [c for c, _ in L.LOGF_TAB]
    assert [d64(w) for w in arrays["T_LOGF_LOGC"]] == [c for _, c in L.LOGF_TAB]
    assert (dd("K_LOGF_LN2"), [dd(f"K_LOGF_A{i}") for i in range(3)]) == (L.LOGF_LN2, list(L.LOGF_A))
    assert [dd(f"K_SINF_S{i}") for i in range(3)] == list(L.SINF_S)
    assert [dd(f"K_COSF_C{i}") for i in range(5)] == list(L.COSF_C)
    assert [d64(w) for w in arrays["T_SINCOSF_SIGN"]] == list(L.SINCOSF_SIGN)
    assert (dd("K_HPI_INV"), dd("K_HPI"), dd("K_PI63")) == (L.HPI_INV, L.HPI, L.PI63)
    assert arrays["T_INV_PIO4"] == list(L.INV_PIO4)
    assert [f32b(w) for w in arrays["T_ATANF_HI"]] == list(L.ATANF_HI)
    assert [f32b(w) for w in arrays["T_ATANF_LO"]] == list(L.ATANF_LO)
    for name in ("PS0", "PS1", "PS2", "PS3", "PS4", "PS5", "QS1", "QS2", "QS3", "QS4", "PI", "PIO2_HI", "PIO2_LO",
                 "NEG_PIO2_LO", "TWO_PIO2_LO"):
        assert ff(f"K_ACOSF_{name}") == getattr(L, f"ACOSF_{name}"), name
    for i in range(11):
        assert ff(f"K_ATANF_T{i}") == getattr(L, f"ATANF_T{i}")
    assert (ff("K_ATANF_HI3"), ff("K_ATANF_LO3")) == (L.ATANF_HI3, L.ATANF_LO3)
    for name in ("PI", "NEG_PI", "TINY", "NEG_TINY", "PIO2", "NEG_PIO2", "PIO4", "NEG_PIO4", "NEG_PI_LO",
                 "NEG_HALF_PI_LO"):
        assert ff(f"K_ATAN2F_{name}") == getattr(L, f"ATAN2F_{name}"), name
    assert defs["K_POW_OFF"] == L.POW_OFF and (dd("K_POW_LN2HI"), dd("K_POW_LN2LO")) == (L.POW_LN2HI, L.POW_LN2LO)
    assert [dd(f"K_POW_A{i}") for i in range(7)] == list(L.POW_A)
    for j, name in enumerate(("T_POW_INVC", "T_POW_LOGC", "T_POW_LOGCTAIL")):
        assert [d64(w) for w in arrays[name]] == [t[j] for t in L.POW_TAB], name
    assert arrays["T_EXP"] == list(E.TABLE)
    assert (dd("K_EXP_INVLN2N"), dd("K_EXP_NEGLN2HIN"), dd("K_EXP_NEGLN2LON")) == (E.INVLN2N, E.NEGLN2HIN, E.NEGLN2LON)
    assert [dd(f"K_EXP_C{i}") for i in range(2, 6)] == [E.C2, E.C3, E.C4, E.C5]
    assert (dd("K_TWO_1009"), dd("K_TWO_M1022"), dd("K_TWO_52")) == (2.0 ** 1009, 2.0 ** -1022, 2.0 ** 52)
    assert [dd(k) for k in ("K_D_1EM2", "K_D_1EM3", "K_D_1EM4", "K_D_1EM10", "K_D_1EM12", "K_D_0_05")] == [
        1e-2, 1e-3, 1e-4, 1e-10, 1e-12, 0.05]
    assert [ff(k) for k in ("K_F_0_01", "K_F_0_95", "K_F_0_05")] == [L.f32(torch.tensor(v, dtype=F64)).item()
                                                                    for v in (0.01, 0.95, 0.05)]
    assert ints == {"XBPO_TORQUE_STEPS": V.TORQUE_STEPS, "XBPO_TORQUE_ROUNDS": V.TORQUE_ROUNDS}


def test_a_perturbed_header_differs_in_one_bit_of_one_entry():
    a, b = K.tables_header(), K.tables_header({"T_POW_LOGC": 37})
    _, ta, _ = parse_header(a)
    _, tb, _ = parse_header(b)
    diff = [(k, i) for k in ta for i, (x, y) in enumerate(zip(ta[k], tb[k])) if x != y]
    assert diff == [("T_POW_LOGC", 37)] and ta["T_POW_LOGC"][37] ^ tb["T_POW_LOGC"][37] == 1
    with pytest.raises(KeyError):
        K.tables_header({"T_NONE": 0})


def test_the_source_calls_no_function_of_cudas_math_library():
    assert K.scan_source(K.SOURCE.read_text()) == []
    planted = "float a = sinf(h);\nx = std::pow(a, b) + __expf(c);\n// cosf(x) in a comment\ny = g_sinf(h) + fsqrt(z);\n"
    assert K.scan_source(planted) == [("sinf", 1), ("pow", 2), ("__expf", 2)]


def test_the_ptx_check_refuses_unpinned_instructions():
    ok = "add.rn.f32 %f1, %f2, %f3;\nmul.rn.f64 %fd1, %fd2, %fd3;\nfma.rn.f64 %fd4, %fd1, %fd2, %fd3;\n" \
         "div.rn.f32 %f4, %f1, %f2;\nsqrt.rn.f64 %fd5, %fd4;\nrcp.rn.f64 %fd6, %fd5;\nneg.f32 %f5, %f4;\n" \
         "min.f32 %f6, %f5, 0f3F800000;\ncvt.rn.f32.f64 %f7, %fd6;\nadd.s64 %rd1, %rd2, %rd3;\n"
    assert K.ptx_check(ok)["ok"]
    for bad in ("add.f32 %f1, %f2, %f3;", "mul.f64 %fd1, %fd2, %fd3;", "add.rn.ftz.f32 %f1, %f2, %f3;",
                "div.full.f32 %f1, %f2, %f3;", "div.approx.f32 %f1, %f2, %f3;", "ex2.approx.ftz.f32 %f1, %f2;",
                "sin.approx.f32 %f1, %f2;", "lg2.approx.f32 %f1, %f2;", "rcp.approx.ftz.f64 %fd1, %fd2;",
                "rsqrt.approx.f64 %fd1, %fd2;", "fma.rz.f64 %fd1, %fd2, %fd3, %fd4;", "mad.f32 %f1, %f2, %f3, %f4;",
                "sqrt.approx.f32 %f1, %f2;", "neg.ftz.f32 %f1, %f2;", "max.ftz.f32 %f1, %f2, %f3;"):
        assert not K.ptx_check(ok + bad)["ok"], bad


@needs_nvcc
def test_the_build_pins_every_rounding():
    m = K.build()
    assert m["ptx_check"]["ok"] and not m["ptx_check"]["refused"]
    for flag in ("-fmad=false", "-prec-div=true", "-prec-sqrt=true", "-ftz=false"):
        assert flag in m["flags"]
    assert not any("fast" in f and "no-fast" not in f for f in m["flags"])
    assert m["header_sha256"] == K._sha(K.tables_header().encode())
    assert m["source_sha256"] == K._sha(K.SOURCE.read_bytes())
    ops = m["ptx_check"]["fp_ops"]
    assert ops.get("fma.rn.f32", 0) and ops.get("fma.rn.f64", 0)       # the C library's fused steps, explicit
    assert not any(k.startswith(("mad", "ex2", "lg2", "sin", "cos", "rsqrt")) or ".ftz" in k for k in ops)


def test_the_kernel_argument_is_checked():
    with pytest.raises(ValueError):
        V.VecOrientPark(2, dynamical_orientation=True, kernel="numba")
    with pytest.raises(ValueError):
        V.VecOrientPark(2, dynamical_orientation=True, kernel="cuda")              # the CPU
    if CUDA:
        with pytest.raises(ValueError):
            V.VecOrientPark(2, device="cuda", dynamical_orientation=False, kernel="cuda")
    assert V.VecOrientPark(2).kernel == "torch"


# ======================================================================================= the reference, on the GPU

def cases(seed: int) -> list:
    return [c for c in tc.cases(seed) if c["kind"] == "vec_do"]


def reference(seed: int) -> dict:
    if seed not in _REF:
        _REF[seed] = ob.reference_rows(cases(seed))
    return _REF[seed]


def use_kernel(park) -> None:
    if park.dynamical_orientation:
        park.kernel = "cuda"


def kernel_differences(seed: int, cs=None) -> list:
    cs = cases(seed) if cs is None else cs
    ids = {c["id"] for c in cs}
    want = {k: v for k, v in reference(seed).items() if k[0] in ids}
    return ob.differences(want, ob.batched(cs, "cuda", park_hook=use_kernel))


@needs_cuda
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_kernel_matches_the_reference(seed):
    bad = kernel_differences(seed)
    assert not bad, (len(bad), bad[:8])


# ========================================================================== the torch module, at every width

def ships(n: int, seed: int) -> list:
    """Hulls over Destiny's ranges (benchmarks/orient_speed.py's, wider), some at rest, some with the defaults."""
    r = random.Random(seed)
    return [{"mass": 10 ** r.uniform(5.5, 9.5), "radius": r.uniform(20.0, 300.0), "max_vel": r.uniform(20.0, 900.0),
             "pos": [r.uniform(-1e4, 1e4) for _ in range(3)],
             "vel": r.choice([[0.0, 0.0, 0.0], [r.uniform(-300.0, 300.0) for _ in range(3)]]),
             "agility": r.uniform(0.05, 3.0), "speed_fraction": r.choice([1.0, r.uniform(0.05, 1.0)]),
             "max_ang_vel": r.choice([1.0, r.uniform(0.02, 4.0)]), "rot_agility": r.choice([0.9, r.uniform(0.02, 3.0)])}
            for _ in range(n)]


def park_pair(n: int, seed: int, device: str = "cuda") -> tuple:
    rows = ships(n, seed)
    k = V.VecOrientPark(n, device=device, dynamical_orientation=True, kernel="cuda")
    t = V.VecOrientPark(n, device=device, dynamical_orientation=True)
    k.load(rows)
    t.load(rows)
    return k, t


FIELDS = ("pos", "vel", "old_pos", "old_vel", "goto", "mode", "last_g", "mass", "agility", "speed_fraction", "max_vel",
          "time_factor", "max_ang_vel", "rot_agility", "rot", "old_rot", "ang_vel", "old_ang_vel", "torque", "rot_roll",
          "radius", "refiled")


def state_differences(a, b) -> list:
    bad = []
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f).to(getattr(a, f).device)
        if x.dtype == F64:
            n = int((~same_bits(x, y)).sum())
        else:
            n = int((x != y).sum())
        if n:
            bad.append((f, n))
    return bad


def commands(park, gen: torch.Generator, tick: int) -> None:
    """Re-aims and the turning setters in flight, refused and non-finite values among them; one draw for every park,
    every value computed on the CPU (torch's pow differs between devices) and then moved to the park's device."""
    n, dev = park.n, park.dev
    u = lambda *s: torch.rand(*s, generator=gen, dtype=F64)
    pick = lambda p: (torch.rand(n, generator=gen) < p).to(dev)
    if tick % 6 == 0:
        d = torch.randn(n, 3, generator=gen, dtype=F64)
        park.goto_point(pick(0.8), park.pos + (d * (u(n, 1) * 4e4 + 50.0)).to(dev))
    if tick % 6 == 3:
        park.goto_direction(pick(0.3), torch.randn(n, 3, generator=gen, dtype=F64).to(dev))
        park.stop(pick(0.1))
    if tick % 4 == 1:
        park.set_max_angular_speed(pick(0.3), torch.where(u(n) < 0.1, -1.0, u(n) * 4.0).to(dev))
        park.set_rot_agility(pick(0.3), torch.where(u(n) < 0.1, 0.0, u(n) * 3.0).to(dev))
        park.set_angular_velocity(pick(0.2), ((u(n, 3) - 0.5) * 4.0).to(dev))
    if tick % 4 == 2:
        park.set_rotation(pick(0.15), torch.randn(n, 4, generator=gen, dtype=F64).to(dev))
        park.set_velocity(pick(0.1), ((u(n, 3) - 0.5) * 600.0).to(dev))
        m = torch.where(u(n) < 0.05, math.nan, 10 ** (u(n) * 4.0 + 5.5))
        park.set_mass(pick(0.2), m.to(dev))
        park.set_agility(pick(0.2), torch.where(u(n) < 0.05, 0.0, u(n) * 3.0).to(dev))
        park.set_speed_fraction(pick(0.2), u(n).to(dev))


def run_pair(k, t, ticks: int, seed: int) -> list:
    gk, gt = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
    for tick in range(ticks):
        commands(k, gk, tick)
        commands(t, gt, tick)
        k.evolve()
        t.evolve()
        bad = state_differences(k, t)
        if bad:
            return [(tick, bad)]
    return []


@needs_cuda
@pytest.mark.parametrize("width", [1, 64, 1000, 4096])
def test_the_kernel_matches_the_torch_module_on_the_gpu(width):
    k, t = park_pair(width, 100 + width)
    assert not run_pair(k, t, 25, width)


@needs_cuda
def test_the_kernel_matches_the_torch_module_on_the_cpu():
    """The chain to the reference: the kernel on the GPU against the torch path on the CPU (itself the reference)."""
    rows = ships(300, 7)
    k = V.VecOrientPark(300, device="cuda", dynamical_orientation=True, kernel="cuda")
    t = V.VecOrientPark(300, device="cpu", dynamical_orientation=True)
    k.load(rows)
    t.load(rows)
    assert not run_pair(k, t, 15, 7)


@needs_cuda
@pytest.mark.parametrize("width", [65536, 262144])
def test_the_kernel_matches_the_torch_module_wide(width):
    k, t = park_pair(width, width)
    assert not run_pair(k, t, 3, width)


def random_state(n: int, seed: int, device="cuda") -> dict:
    """One tick's inputs over every branch: thrusts of every size (zero, tiny, huge, NaN), rotations and angular
    velocities at random (unnormalised, zero, NaN), hull values with zeros, NaNs and infinities."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=F64)
    u = lambda *s: torch.rand(*s, generator=g, dtype=F64)
    f = lambda x: L.f32(x)
    a = r(n, 3) * 10 ** (u(n, 1) * 8 - 4)
    a[u(n) < 0.05] = 0.0
    a[u(n) < 0.01] = math.nan
    mass = 10 ** (u(n) * 5 + 5)
    mass[u(n) < 0.01] = math.nan
    mass[u(n) < 0.01] = 0.0
    rot = f(r(n, 4))
    rot[u(n) < 0.02] = 0.0
    w = f(r(n, 3) * 10 ** (u(n, 1) * 4 - 3))
    w[u(n) < 0.05] = 0.0
    w[u(n) < 0.005] = math.inf
    mav = f(u(n) * 3)
    mav[u(n) < 0.05] = 0.0
    st = {"a": a, "mass": mass, "agility": f(u(n) * 3 + 0.01), "speed_fraction": f(u(n)),
          "max_vel": f(u(n) * 900), "max_ang_vel": mav, "rot_agility": f(u(n) * 3 + 0.01), "rot": rot, "ang_vel": w}
    return {k: v.to(device).contiguous() for k, v in st.items()}


def kernel_tick(st: dict, block: int = 128, lib=None) -> dict:
    a = st["a"]
    out = {"a": torch.empty_like(a), "rot": torch.empty_like(st["rot"]), "old_rot": torch.empty_like(st["rot"]),
           "ang_vel": torch.empty_like(a), "old_ang_vel": torch.empty_like(a), "torque": torch.empty_like(a),
           "rot_roll": torch.empty_like(st["mass"])}
    K.orient(a, out["a"], st["mass"], st["agility"], st["speed_fraction"], st["max_vel"], st["max_ang_vel"],
             st["rot_agility"], st["rot"], st["ang_vel"], out["rot"], out["old_rot"], out["ang_vel"],
             out["old_ang_vel"], out["torque"], out["rot_roll"], 1.0, 1.0, 1.0e6, block, lib)
    return out


def torch_tick(st: dict) -> dict:
    """The torch module's _orient and its evolve's commit on the same inputs."""
    n = st["a"].shape[0]
    p = V.VecOrientPark(n, device=st["a"].device, dynamical_orientation=True)
    for k in ("mass", "agility", "speed_fraction", "max_vel", "max_ang_vel", "rot_agility", "rot", "ang_vel"):
        setattr(p, k, st[k].clone())
    a = p._orient(st["a"].clone())
    rot, w = (V._stack(x) for x in p._next_rotation)
    f = V.f32                                                       # evolve's commit, as written there
    rot_ok = torch.isfinite(f(f(f(f(rot[:, 0] * rot[:, 0]) + f(rot[:, 1] * rot[:, 1])) + f(rot[:, 2] * rot[:, 2]))
                              + f(rot[:, 3] * rot[:, 3])))
    w_ok = torch.isfinite(V.v3_dot(V._cols(w), V._cols(w)))
    return {"a": a, "rot": torch.where(rot_ok[:, None], rot, p.rot), "old_rot": torch.where(rot_ok[:, None], p.rot, rot),
            "ang_vel": torch.where(w_ok[:, None], w, p.ang_vel), "old_ang_vel": torch.where(w_ok[:, None], p.ang_vel, w),
            "torque": p.torque, "rot_roll": p.rot_roll}


def tick_differences(x: dict, y: dict) -> list:
    return [(k, int((~same_bits(x[k], y[k].to(x[k].device))).sum())) for k in x
            if not bool(same_bits(x[k], y[k].to(x[k].device)).all())]


@needs_cuda
def test_one_tick_over_every_branch_matches_the_torch_module():
    """Inputs no park reaches by flying (unnormalised and zero rotations, NaN and infinite values), one tick."""
    st = random_state(20000, 5)
    assert not tick_differences(kernel_tick(st), torch_tick(st))


@needs_cuda
def test_a_ships_result_does_not_depend_on_the_width_or_the_block():
    st = random_state(37, 6)
    one = kernel_tick(st)
    for reps, block in ((1, 32), (29, 128), (1772, 256), (7085, 640)):
        tiled = {k: v.repeat(reps, *([1] * (v.dim() - 1))).contiguous() for k, v in st.items()}
        out = kernel_tick(tiled, block)
        for k, v in out.items():
            assert bool(same_bits(v.view(reps, 37, -1), one[k].view(1, 37, -1).expand(reps, -1, -1)).all()), \
                (reps, block, k)


# ======================================================================= the float functions and the native FMA

C_FMA = ctypes.CDLL(ctypes.util.find_library("m")).fma
C_FMA.restype, C_FMA.argtypes = ctypes.c_double, [ctypes.c_double] * 3


@needs_cuda
@pytest.mark.parametrize("name", ["expf", "logf", "sinf", "cosf", "acosf", "atanf"])
def test_the_kernels_float_functions_are_vec_libms_and_the_c_librarys(name):
    xs = TL.floats(21, 40000)
    x = torch.tensor(xs, dtype=F64, device="cuda")
    got = K.libm(name, x)
    assert bool(same_bits(got, getattr(L, name)(x)).all())
    want = torch.tensor([TL.C_F1[name](v) for v in xs], dtype=F64, device="cuda")
    assert bool(same_bits(got, want).all())


@needs_cuda
def test_the_kernels_atan2f_and_pow_are_vec_libms_and_the_c_librarys():
    ys, xs = TL.pairs(22, 30000)
    y, x = (torch.tensor(v, dtype=F64, device="cuda") for v in (ys, xs))
    got = K.libm("atan2f", x, y)
    assert bool(same_bits(got, L.atan2f(y, x)).all())
    assert bool(same_bits(got, torch.tensor([TL.C_ATAN2F(a, b) for a, b in zip(ys, xs)], dtype=F64,
                                            device="cuda")).all())
    px, py = TL.pow_args(23, 30000)
    x, y = (torch.tensor(v, dtype=F64, device="cuda") for v in (px, py))
    got = K.libm("pow", x, y)
    assert bool(same_bits(got, L.pow(x, y)).all())
    assert bool(same_bits(got, torch.tensor([TL.C_POW(a, b) for a, b in zip(px, py)], dtype=F64, device="cuda")).all())


@needs_cuda
def test_the_native_fused_multiply_adds_are_the_emulated_ones():
    """__fma_rn against vec_exp.fma (Boldo and Melquiond's emulation) and the C library's fma on 200,000 triples (exact
    ties, cancellations, the product's exact error, near ties, random bit patterns); __fmaf_rn against vec_libm.fmaf and
    fmaf at constructed near ties, where rounding to double first goes wrong."""
    tr = TE.fma_triples(31, 200_000)
    a, b, c = (torch.tensor([t[i] for t in tr], dtype=F64, device="cuda") for i in range(3))
    got = K.fma(a, b, c)
    assert bool(same_bits(got, E.fma(a, b, c)).all())
    assert bool(same_bits(got, torch.tensor([C_FMA(*t) for t in tr], dtype=F64, device="cuda")).all())
    r = random.Random(32)
    tri = TL.near_ties(33, 3000) + [tuple(TL.f_of(r.getrandbits(32) & 0x7EFFFFFF) for _ in range(3))
                                    for _ in range(20000)]
    tri = [t for t in tri if all(math.isfinite(v) for v in t) and math.isfinite(TL.C_FMAF(*t))]
    a, b, c = (torch.tensor([t[i] for t in tri], dtype=F64, device="cuda") for i in range(3))
    got = K.fma(a, b, c, single=True)
    assert bool(same_bits(got, L.fmaf(a, b, c)).all())
    assert bool(same_bits(got, torch.tensor([TL.C_FMAF(*t) for t in tri], dtype=F64, device="cuda")).all())


# ================================================================================================ CUDA graphs

@needs_cuda
def test_a_tick_replayed_from_a_cuda_graph_is_the_eager_tick():
    """TickGraph's replays against eager ticks of the same park, commands and setters in flight, every field every
    tick; and both against the torch path."""
    rows = ships(777, 11)
    e = V.VecOrientPark(777, device="cuda", dynamical_orientation=True, kernel="cuda")
    g = V.VecOrientPark(777, device="cuda", dynamical_orientation=True, kernel="cuda")
    t = V.VecOrientPark(777, device="cuda", dynamical_orientation=True)
    for p in (e, g, t):
        p.load(rows)
    graph = K.TickGraph(g)
    gens = [torch.Generator().manual_seed(12) for _ in range(3)]
    for tick in range(30):
        for p, gen in zip((e, g, t), gens):
            commands(p, gen, tick)
        e.evolve()
        graph.tick()
        t.evolve()
        assert not state_differences(e, g), tick
        assert not state_differences(e, t), tick
    assert e.current_time == g.current_time == 30


@needs_cuda
def test_the_kernel_alone_is_captured_and_replayed():
    """One launch on fixed buffers captured in a graph, replayed on new inputs: the eager launch's values."""
    st = random_state(5000, 13)
    out = kernel_tick(st)                                           # warm-up, and the buffers
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        K.orient(st["a"], out["a"], st["mass"], st["agility"], st["speed_fraction"], st["max_vel"],
                 st["max_ang_vel"], st["rot_agility"], st["rot"], st["ang_vel"], out["rot"], out["old_rot"],
                 out["ang_vel"], out["old_ang_vel"], out["torque"], out["rot_roll"], 1.0, 1.0, 1.0e6)
    for seed in (14, 15):
        new = random_state(5000, seed)
        for k in st:
            st[k].copy_(new[k])
        graph.replay()
        torch.cuda.synchronize()
        assert not tick_differences(out, kernel_tick(new))


# ============================================================================================== planted faults

FAULTS = {          # name: (source edits (old, new, count), table perturbation, the checks that must catch it)
    # a multiply-add contracted (v3_dot's first product fused into its sum)
    "contracted_fma": ([("return fadd(fadd(fmul(a.x, b.x), fmul(a.y, b.y)), fmul(a.z, b.z));",
                         "return fadd(ffma(a.x, b.x, fmul(a.y, b.y)), fmul(a.z, b.z));", 1)], None, ("scenarios",)),
    # CUDA's own sinf in RotationQuaternion (the source scan refuses it too)
    "cuda_sinf": ([("const float s = g_sinf(h);", "const float s = sinf(h);", 1)], None, ("scenarios",)),
    # CUDA's own expf for tf = expf(-dt t_scale) (the source scan and the PTX check refuse it too)
    "cuda_expf": ([("sh.tf = g_expf(fmul(-fdt, sh.t_scale));", "sh.tf = expf(fmul(-fdt, sh.t_scale));", 1)], None,
                  ("scenarios",)),
    # one entry of a float table wrong in its last bit (atanf's atan(1) high part: the roll torque's atan2f)
    "table_entry": ([], {"T_ATANF_HI": 1}, ("scenarios", "atanf")),
    # one entry of a double table wrong in its last bit (pow's log c of cell 37: seen in pow's double results; a float
    # result rounds such an error away, as it does a last bit of expf's 2^(i/32))
    "table_entry_pow": ([], {"T_POW_LOGC": 37}, ("pow",)),
    # GotoTorqueBreaks' outer correction with one round less
    "round_skipped": ([("    float kappa1 = 1.0f;\n    for (int round = 0; round < XBPO_TORQUE_ROUNDS; ++round) {",
                        "    float kappa1 = 1.0f;\n    for (int round = 0; round < XBPO_TORQUE_ROUNDS - 1; ++round) {",
                        1)], None, ("scenarios",)),
    # kappa1 = tf^n1 by pow, not Destiny's repeated float products
    "kappa_by_pow": ([("k1 = fmul(k1, sh.tf);", "k1 = tof(g_pow(tod(sh.tf), (double)n1));", 2)], None, ("scenarios",)),
}


def faulted_library(name: str) -> dict:
    """The library built from the source with one planted fault (cached under the build directory)."""
    edits, perturb, _ = FAULTS[name]
    text = K.SOURCE.read_text()
    for old, new, count in edits:
        assert text.count(old) == count, (name, old)
        text = text.replace(old, new)
    d = K.build_dir() / "faults"
    d.mkdir(parents=True, exist_ok=True)
    src = d / f"{name}.cu"
    if not src.exists() or src.read_text() != text:
        src.write_text(text)
    return K.load_library(source=src, perturb=perturb, checks=False)


@needs_cuda
@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    lib = faulted_library(fault)
    if fault.startswith("cuda_"):                                   # refused by the build's checks, too
        assert lib["manifest"]["source_scan"], "the source scan missed it"
    for check in FAULTS[fault][2]:
        if check == "scenarios":
            with monkeypatch.context() as mp:
                mp.setattr(K, "load_library", lambda *a, **k: lib)
                cs = [c for c in cases(1) if c["friction"] == 1.0e6][:80]
                assert kernel_differences(1, cs), f"the planted fault {fault} went unnoticed by the scenarios"
        elif check == "pow":
            px, py = TL.pow_args(25, 30000)
            x, y = (torch.tensor(v, dtype=F64, device="cuda") for v in (px, py))
            assert not bool(same_bits(K.libm("pow", x, y, lib=lib), L.pow(x, y)).all()), f"{fault} unnoticed by pow"
        else:
            x = torch.tensor(TL.floats(24, 40000), dtype=F64, device="cuda")
            assert not bool(same_bits(K.libm(check, x, lib=lib), getattr(L, check)(x)).all()), \
                f"{fault} unnoticed by {check}"

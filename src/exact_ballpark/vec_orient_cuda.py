# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's dynamical orientation (MOV-728) as one CUDA kernel, one thread a ship: the step that
`exact_ballpark.vec_orient.VecOrientPark` computes in torch operations (`_orient` and the rotation's commit at the end of
`evolve`), run as the scalar algorithm of `exact_ballpark.full.FullBallpark` (GotoTorqueBreaks and GotoRollBreaks with
their rounds, each ship leaving them when its own rounds end; ApplyTorque; the thrust laid along the heading), so a
tick is one launch instead of some 23,000 small operations. `VecOrientPark(..., kernel="cuda")` selects it; the torch
path (`kernel="torch"`, the default) stays the exactness reference, unchanged. Checked bit for bit against the torch
module and the scalar reference by tests/test_movement_vec_orient_cuda.py (Destiny's own code compiled verbatim is the
reference's reference: tests/test_movement_turn_oracle.py).

The source is vec_orient_cuda.cu beside this module; the library is built on first use (or when the source, the
generated header, the flags or nvcc change) into a build directory outside any source tree ($EXACT_BALLPARK_ORIENT_CUDA_DIR; by default $EXACT_BALLPARK_CACHE/orient_cuda), with a manifest; it is loaded through ctypes and launched
on torch's current stream, with no allocation and no host synchronisation, so a launch can be captured in a CUDA graph.

**Exactness by construction.**
- Every float operation is a correctly rounded binary32 operation (__fadd_rn, __fsub_rn, __fmul_rn, __fdiv_rn,
  __fsqrt_rn) exactly where the torch module rounds a double operation of two floats to float (equal for +, -, *, /,
  sqrt: 53 >= 2 x 24 + 2); every double operation (the thrust's sums of squares, comparisons against double literals,
  n x dt, the C library's double internals) is __dadd_rn ... __dsqrt_rn in double.
- A fused multiply-add only where the C library's compiled code fuses one (exact_ballpark.vec_libm's transcription of
  glibc 2.34 aarch64): __fma_rn and __fmaf_rn are the IEEE fused operations, rounded once, which vec_exp's and
  vec_libm's emulations compute exactly wherever no partial product overflows or underflows; the tests hold the
  native operations to the emulations (`fma`), and the device functions to vec_libm and the C library (`libm`).
- nvcc with -fmad=false, -prec-div=true, -prec-sqrt=true, -ftz=false and no fast math; the build reads the PTX back
  and refuses one with an unrounded, flushed, approximate or contracted floating-point instruction (`ptx_check`), and
  refuses a source that calls any function of CUDA's math library (`scan_source`): the C library's expf, logf, sinf,
  cosf, acosf, atanf, atan2f and pow are transcribed in the source, their constants generated from vec_libm's and
  vec_exp's own tables into a header (`tables_header`), never typed.

The kernel reads a ship's float fields (agility, speed fraction, maximum speed, maximum angular speed, rotational
agility, rotation, angular velocity) as floats: the park stores them as floats in float64 tensors (its loading and its
setters round them), as Destiny stores them.
"""

from __future__ import annotations

import ctypes
import datetime
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import torch

from . import vec_exp as E
from . import vec_libm as L

F64 = torch.float64
from ._cuda_build import SourcePath, cache_root, default_nvcc, read_source  # noqa: E402,F401
SOURCE = SourcePath(Path(__file__).with_name("vec_orient_cuda.cu"))
HEADER = "vec_orient_cuda_tables.h"
NVCC = default_nvcc()
FLAGS = ["-O3", "-std=c++17", "-fmad=false", "-prec-div=true", "-prec-sqrt=true", "-ftz=false",
         "-Xcompiler", "-fPIC,-ffp-contract=off,-fno-fast-math", "-shared", "-Xptxas", "-v"]
TARGETS = {
    "native": [("compute_90", "sm_90"), ("compute_100", "sm_100"), ("compute_90", "compute_90")],
    "ptx": [("compute_90", "compute_90")],
}
ABI = 1
LIBM_FUNCS = {"expf": 0, "logf": 1, "sinf": 2, "cosf": 3, "acosf": 4, "atanf": 5, "atan2f": 6, "pow": 7}
_LIBS: dict = {}

# CUDA's math library (and its intrinsics) by name: none may be called in the source (`scan_source`)
MATH_LIBRARY = (
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "ceil", "copysign", "cos", "cosh", "cospi",
    "erf", "erfc", "erfinv", "exp", "exp10", "exp2", "expm1", "fabs", "fdim", "floor", "fma", "fmax", "fmin", "fmod",
    "frexp", "hypot", "ilogb", "ldexp", "lgamma", "llrint", "llround", "log", "log10", "log1p", "log2", "logb",
    "lrint", "lround", "modf", "nearbyint", "nextafter", "norm3d", "pow", "rcbrt", "remainder", "remquo", "rhypot",
    "rint", "round", "rsqrt", "scalbn", "sin", "sincos", "sincospi", "sinh", "sinpi", "sqrt", "tan", "tanh", "tgamma",
    "trunc", "j0", "j1", "y0", "y1")
MATH_INTRINSICS = ("__expf", "__exp10f", "__logf", "__log2f", "__log10f", "__powf", "__sinf", "__cosf", "__tanf",
                   "__sincosf", "__fdividef", "__saturatef", "__frsqrt_rn", "__frcp_rn", "__drcp_rn", "__fsqrt_rd",
                   "__fsqrt_ru", "__fsqrt_rz", "__fdiv_rd", "__fdiv_ru", "__fdiv_rz", "__fmaf_rz", "__fmaf_ru",
                   "__fmaf_rd", "__fma_rz", "__fma_ru", "__fma_rd")


# ============================================================================================ the generated header

def _dbits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _fbits(x: float) -> int:
    """A float's bit pattern; refuses a double that is not a float."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    if struct.unpack("<f", struct.pack("<I", u))[0] != x:
        raise ValueError(f"{x!r} is not a float")
    return u


def _f32(x: float) -> float:
    """A double rounded to the nearest float (the torch module's f32 of a literal)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def table_values() -> dict:
    """Every constant and table the kernel uses, by its name in the header: (kind, value or values), read from
    exact_ballpark.vec_libm and exact_ballpark.vec_exp (and vec_orient's own literals and loop bounds), never typed.
    Kinds: d (a double), f (a float), u64 and u32 (integers); tables are tuples."""
    from . import vec_orient as V
    d, f = "d", "f"
    out = {
        "K_EXPF_INVLN2": (d, L.EXPF_INVLN2), "K_EXPF_C0": (d, L.EXPF_C[0]), "K_EXPF_C1": (d, L.EXPF_C[1]),
        "K_EXPF_C2": (d, L.EXPF_C[2]), "K_EXPF_OFLOW": (f, L.EXPF_OFLOW), "K_EXPF_UFLOW": (f, L.EXPF_UFLOW),
        "K_EXPF_MAY_UFLOW": (f, L.EXPF_MAY_UFLOW), "K_MAY_UFLOWF": (f, L.MAY_UFLOWF),
        "T_EXP2F": ("u64", tuple(L.EXP2F_TAB)),
        "T_LOGF_INVC": (d, tuple(c for c, _ in L.LOGF_TAB)), "T_LOGF_LOGC": (d, tuple(c for _, c in L.LOGF_TAB)),
        "K_LOGF_LN2": (d, L.LOGF_LN2), "K_LOGF_A0": (d, L.LOGF_A[0]), "K_LOGF_A1": (d, L.LOGF_A[1]),
        "K_LOGF_A2": (d, L.LOGF_A[2]),
        "T_SINCOSF_SIGN": (d, tuple(L.SINCOSF_SIGN)), "K_HPI_INV": (d, L.HPI_INV), "K_HPI": (d, L.HPI),
        "K_PI63": (d, L.PI63), "T_INV_PIO4": ("u32", tuple(L.INV_PIO4)),
        "T_ATANF_HI": (f, tuple(L.ATANF_HI)), "T_ATANF_LO": (f, tuple(L.ATANF_LO)),
        "K_POW_OFF": ("u64", L.POW_OFF), "K_POW_LN2HI": (d, L.POW_LN2HI), "K_POW_LN2LO": (d, L.POW_LN2LO),
        "T_POW_INVC": (d, tuple(t[0] for t in L.POW_TAB)), "T_POW_LOGC": (d, tuple(t[1] for t in L.POW_TAB)),
        "T_POW_LOGCTAIL": (d, tuple(t[2] for t in L.POW_TAB)),
        "K_EXP_INVLN2N": (d, E.INVLN2N), "K_EXP_NEGLN2HIN": (d, E.NEGLN2HIN), "K_EXP_NEGLN2LON": (d, E.NEGLN2LON),
        "K_EXP_C2": (d, E.C2), "K_EXP_C3": (d, E.C3), "K_EXP_C4": (d, E.C4), "K_EXP_C5": (d, E.C5),
        "T_EXP": ("u64", tuple(E.TABLE)), "K_TWO_1009": (d, E.TWO_1009), "K_TWO_M1022": (d, E.TWO_M1022),
        "K_TWO_52": (d, float(2 ** 52)),
        # vec_orient's literals (Ballpark.cpp's): double comparisons, and the floats f32(0.01), f32(0.95), f32(0.05)
        "K_D_1EM2": (d, 1e-2), "K_D_1EM3": (d, 1e-3), "K_D_1EM4": (d, 1e-4), "K_D_1EM10": (d, 1e-10),
        "K_D_1EM12": (d, 1e-12), "K_D_0_05": (d, 0.05), "K_F_0_01": (f, _f32(0.01)), "K_F_0_95": (f, _f32(0.95)),
        "K_F_0_05": (f, _f32(0.05)),
    }
    for i, c in enumerate(L.SINF_S):
        out[f"K_SINF_S{i}"] = (d, c)
    for i, c in enumerate(L.COSF_C):
        out[f"K_COSF_C{i}"] = (d, c)
    for i, c in enumerate(L.POW_A):
        out[f"K_POW_A{i}"] = (d, c)
    for name in ("PS0", "PS1", "PS2", "PS3", "PS4", "PS5", "QS1", "QS2", "QS3", "QS4", "PI", "PIO2_HI", "PIO2_LO",
                 "NEG_PIO2_LO", "TWO_PIO2_LO"):
        out[f"K_ACOSF_{name}"] = (f, getattr(L, f"ACOSF_{name}"))
    for i in range(11):
        out[f"K_ATANF_T{i}"] = (f, getattr(L, f"ATANF_T{i}"))
    out["K_ATANF_HI3"], out["K_ATANF_LO3"] = (f, L.ATANF_HI3), (f, L.ATANF_LO3)
    for name in ("PI", "NEG_PI", "TINY", "NEG_TINY", "PIO2", "NEG_PIO2", "PIO4", "NEG_PIO4", "NEG_PI_LO",
                 "NEG_HALF_PI_LO"):
        out[f"K_ATAN2F_{name}"] = (f, getattr(L, f"ATAN2F_{name}"))
    out["XBPO_TORQUE_STEPS"] = ("int", V.TORQUE_STEPS)
    out["XBPO_TORQUE_ROUNDS"] = ("int", V.TORQUE_ROUNDS)
    return out


def _word(kind: str, v) -> int:
    return _dbits(v) if kind == "d" else _fbits(v) if kind == "f" else int(v)


def tables_header(perturb: dict | None = None) -> str:
    """The C header of `table_values()`: scalars as #defines of their bit patterns, tables as device arrays of bit
    patterns. `perturb` ({table name: index}) flips the lowest bit of those entries: the planted fault of a wrong
    table entry, for the tests only."""
    perturb = dict(perturb or {})
    lines = ["// Generated by exact_ballpark.vec_orient_cuda.tables_header() from exact_ballpark.vec_libm and",
             "// exact_ballpark.vec_exp (glibc 2.34 aarch64's tables and constants, read from the library); do not edit.",
             "#pragma once", "#include <cstdint>", ""]
    for name, (kind, v) in table_values().items():
        if kind == "int":
            lines.append(f"#define {name} {int(v)}")
        elif isinstance(v, tuple):
            words = [_word(kind, x) for x in v]
            if name in perturb:
                words[perturb.pop(name)] ^= 1
            ctype, width = ("uint64_t", 16) if kind in ("d", "u64") else ("uint32_t", 8)
            suffix = "ULL" if width == 16 else "u"
            body = ", ".join(f"0x{w:0{width}x}{suffix}" for w in words)
            lines.append(f"static __device__ const {ctype} {name}[{len(words)}] = {{{body}}};")
        else:
            w = _word(kind, v)
            lines.append(f"#define {name} 0x{w:016x}ULL" if kind in ("d", "u64") else f"#define {name} 0x{w:08x}u")
    if perturb:
        raise KeyError(f"no such table: {sorted(perturb)}")
    return "\n".join(lines) + "\n"


# ================================================================================================ the static checks

def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def scan_source(text: str) -> list:
    """Every call of a function of CUDA's math library, or of an approximate or directed-rounding intrinsic, in a
    CUDA source (comments left out): (name, line). The kernel's own functions have other names (g_expf, ...)."""
    names = set(MATH_LIBRARY) | {n + "f" for n in MATH_LIBRARY} | set(MATH_INTRINSICS)
    found = []
    for k, line in enumerate(_strip_comments(text).splitlines(), 1):
        for m in re.finditer(r"(?<![\w.:>])(?:std::|::)?([A-Za-z_]\w*)\s*\(", line):
            if m.group(1) in names:
                found.append((m.group(1), k))
    return found


_FP = re.compile(r"\b(add|sub|mul|div|sqrt|rcp|rsqrt|fma|mad|neg|abs|min|max|ex2|lg2|sin|cos|tanh)"
                 r"((?:\.[a-z0-9]+)*)\.(f32|f64)\b")


def ptx_check(ptx: str) -> dict:
    """Every floating-point arithmetic instruction of the PTX, by form. The build fails unless each add, sub, mul,
    div, sqrt and rcp carries .rn (which no later stage contracts or approximates; rcp.rn is the IEEE reciprocal,
    nvcc's form of 1.0 / x, equal to div.rn of 1.0), each fma is fma.rn (only the source's __fma_rn and __fmaf_rn:
    nvcc's -fmad=false contracts nothing), and nothing is flushed (.ftz), approximate (.approx, .full), a mad, or a
    transcendental instruction (ex2, lg2, sin, cos, tanh, rsqrt)."""
    forms = {}
    for m in _FP.finditer(ptx):
        key = f"{m.group(1)}{m.group(2)}.{m.group(3)}"
        forms[key] = forms.get(key, 0) + 1
    bad = {}
    for k, v in forms.items():
        op = k.split(".")[0]
        mods = k.split(".")[1:-1]
        if op in ("neg", "abs", "min", "max"):
            ok = "ftz" not in mods
        elif op in ("add", "sub", "mul", "div", "sqrt", "rcp", "fma"):
            ok = mods == ["rn"]
        else:
            ok = False
        if not ok:
            bad[k] = v
    return {"fp_ops": dict(sorted(forms.items())), "refused": bad, "ok": not bad}


# ============================================================================================ build and load

def build_dir() -> Path:
    d = os.environ.get("EXACT_BALLPARK_ORIENT_CUDA_DIR")
    if d:
        return Path(d)
    return cache_root() / "orient_cuda"


def _nvcc() -> Path:
    if NVCC.exists():
        return NVCC
    found = shutil.which("nvcc")
    if not found:
        raise RuntimeError(f"nvcc not found at {NVCC} nor on PATH (set EXACT_BALLPARK_NVCC)")
    return Path(found)


_NVCC_VERSIONS: dict = {}


def nvcc_version(nvcc: Path) -> str:
    """`nvcc --version`'s whole output for the compiler at this path, run once a process: every kernel
    library's build key holds it beside the compiler's path (vec_orient_cuda, vec_collide_cuda, vec_modes_cuda,
    vec_combat_cuda, vec_composed_cuda, vec_ypr_cuda, vec_server_cuda, vec_server_combat_cuda, vec_server_space_cuda,
    vec_space), so a new nvcc installed at the same path builds anew rather than reusing a library an older one
    built."""
    k = str(nvcc)
    if k not in _NVCC_VERSIONS:
        _NVCC_VERSIONS[k] = subprocess.run([k, "--version"], capture_output=True, text=True, check=True).stdout
    return _NVCC_VERSIONS[k]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(out_dir: Path | None = None, variant: str = "native", source: Path = SOURCE, perturb: dict | None = None,
          checks: bool = True) -> dict:
    """Compile `source` with its generated header into out_dir (by default `build_dir()`); returns the manifest written
    beside the library: the source's and the header's SHA-256, nvcc's version, flags and targets, the kernels'
    registers and spills, the PTX check, the library's SHA-256 and the UTC time. A library already built from the same
    inputs is reused. `perturb` and `checks=False` (the source scan and the PTX check recorded, not enforced) are for
    the tests' planted faults only."""
    if variant not in TARGETS:
        raise ValueError(f"unknown variant {variant!r}")
    out_dir = Path(out_dir) if out_dir is not None else build_dir()
    text = read_source(source)
    calls = scan_source(text)
    if checks and calls:
        raise RuntimeError(f"the source calls CUDA's math library: {calls}")
    header = tables_header(perturb)
    nvcc = _nvcc()
    gencode = [f"-gencode=arch={a},code={c}" for a, c in TARGETS[variant]]
    key = _sha(json.dumps([_sha(text.encode()), _sha(header.encode()), FLAGS, gencode, str(nvcc), nvcc_version(nvcc),
                           checks]).encode())[:16]
    stem = f"vec_orient_cuda_{variant}_{key}"
    lib, man = out_dir / f"{stem}.so", out_dir / f"{stem}.json"
    if lib.exists() and man.exists():
        m = json.loads(man.read_text())
        if m.get("library_sha256") == _sha(lib.read_bytes()) and (m.get("ptx_check", {}).get("ok") or not checks):
            return m
    work = out_dir / f".{stem}.{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        (work / HEADER).write_text(header)
        src = work / Path(source).name
        src.write_text(text)
        tmp = work / "tmp"
        tmp.mkdir(exist_ok=True)
        env = dict(os.environ, TMPDIR=os.environ.get("EXACT_BALLPARK_BUILD_TMP", str(tmp)))
        cmd = [str(nvcc), *FLAGS, *gencode, "-I", str(work), "-o", str(work / "lib.so"), str(src)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
        cuobjdump = nvcc.parent / "cuobjdump"
        listing = subprocess.run([str(cuobjdump), "--list-elf", "--list-ptx", str(work / "lib.so")],
                                 capture_output=True, text=True, check=True).stdout.splitlines()
        ptx = subprocess.run([str(cuobjdump), "-ptx", str(work / "lib.so")], capture_output=True, text=True,
                             check=True).stdout
        check = ptx_check(ptx)
        if checks and not check["ok"]:
            raise RuntimeError(f"the PTX has unpinned floating-point instructions: {check['refused']}")
        resources = [ln.strip() for ln in (r.stdout + r.stderr).splitlines()
                     if "registers" in ln or "spill" in ln or "Compiling entry" in ln or "stack frame" in ln]
        version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True).stdout
        manifest = {
            "variant": variant, "source": str(source), "source_sha256": _sha(text.encode()),
            "header": HEADER, "header_sha256": _sha(header.encode()), "perturb": perturb or {}, "checks": checks,
            "source_scan": calls,
            "nvcc": version.strip().splitlines()[-1], "nvcc_path": str(nvcc), "flags": FLAGS,
            "targets": [f"{a}->{c}" for a, c in TARGETS[variant]], "embedded": listing, "ptxas": resources,
            "ptx_check": check, "library": lib.name, "library_sha256": _sha((work / "lib.so").read_bytes()),
            "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (work / HEADER).replace(out_dir / f"{stem}.h")
        (work / "lib.so").replace(lib)
        mtmp = work / "manifest.json"
        mtmp.write_text(json.dumps(manifest, indent=1) + "\n")
        mtmp.replace(man)
        return manifest
    finally:
        shutil.rmtree(work, ignore_errors=True)


def load_library(variant: str = "native", source: Path = SOURCE, perturb: dict | None = None,
                 checks: bool = True) -> dict:
    """The compiled library (built if missing or stale), its functions typed, and its manifest; once a process."""
    k = (variant, str(source), json.dumps(perturb or {}, sort_keys=True), checks)
    if k in _LIBS:
        return _LIBS[k]
    manifest = build(None, variant, source, perturb, checks)
    lib = ctypes.CDLL(str(build_dir() / manifest["library"]))
    if lib.xbpo_abi() != ABI:
        raise RuntimeError(f"the library's ABI {lib.xbpo_abi()} is not {ABI}")
    p, i64, dbl, flt, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double, ctypes.c_float, ctypes.c_int
    lib.xbpo_error_string.restype = ctypes.c_char_p
    lib.xbpo_error_string.argtypes = [i32]
    lib.xbpo_orient.argtypes = [i64] + [p] * 16 + [dbl, flt, flt, p, i32]
    lib.xbpo_orient.restype = i32
    lib.xbpo_libm.argtypes = [i32, i64, p, p, p, p, i32]
    lib.xbpo_libm.restype = i32
    lib.xbpo_fma.argtypes = [i32, i64, p, p, p, p, p, i32]
    lib.xbpo_fma.restype = i32
    lib.xbpo_info.argtypes = [ctypes.POINTER(i32)] * 5
    lib.xbpo_info.restype = i32
    entry = {"lib": lib, "manifest": manifest}
    _LIBS[k] = entry
    return entry


def _check(rc: int, lib, fn: str) -> None:
    if rc:
        msg = lib.xbpo_error_string(rc).decode() if rc > 0 else "refused (block size or arguments)"
        raise RuntimeError(f"{fn} failed ({rc}: {msg})")


def _dev(t: torch.Tensor, shape: tuple, name: str, device) -> None:
    if t.dtype != F64 or t.device != device or tuple(t.shape) != shape or not t.is_contiguous():
        raise ValueError(f"{name} must be a contiguous float64 tensor of shape {shape} on {device}, not "
                         f"{t.dtype} {tuple(t.shape)} on {t.device}")


# ======================================================================================================= launches

def orient(a: torch.Tensor, a_out: torch.Tensor, mass, agility, speed_fraction, max_vel, max_ang_vel, rot_agility,
           rot: torch.Tensor, ang_vel: torch.Tensor, rot_out: torch.Tensor, old_rot_out: torch.Tensor,
           ang_vel_out: torch.Tensor, old_ang_vel_out: torch.Tensor, torque_out: torch.Tensor,
           rot_roll_out: torch.Tensor, dt: float, fdt: float, ffriction: float, block: int = 128,
           lib: dict | None = None) -> None:
    """One tick's dynamical orientation of B ships, one launch on torch's current stream of their device: from the
    thrust `a` (B, 3) and each ship's values, the thrust laid along the predicted heading into `a_out`, the heading
    torque and the roll torque, and ApplyTorque's rotation and angular velocity committed where finite (the new values
    into rot_out and ang_vel_out, the ones they replace into old_rot_out and old_ang_vel_out). All tensors are contiguous
    float64 on one CUDA device; an output may be its input (a_out = a, rot_out = rot, ang_vel_out = ang_vel). dt is the
    tick in seconds, fdt and ffriction the floats of dt and of the friction. No host synchronisation."""
    e = lib or load_library()
    device = a.device
    if device.type != "cuda":
        raise ValueError("the orientation kernel needs CUDA tensors")
    B = a.shape[0]
    for t, shape, name in ((a, (B, 3), "a"), (a_out, (B, 3), "a_out"), (mass, (B,), "mass"),
                           (agility, (B,), "agility"), (speed_fraction, (B,), "speed_fraction"),
                           (max_vel, (B,), "max_vel"), (max_ang_vel, (B,), "max_ang_vel"),
                           (rot_agility, (B,), "rot_agility"), (rot, (B, 4), "rot"), (ang_vel, (B, 3), "ang_vel"),
                           (rot_out, (B, 4), "rot_out"), (old_rot_out, (B, 4), "old_rot_out"),
                           (ang_vel_out, (B, 3), "ang_vel_out"), (old_ang_vel_out, (B, 3), "old_ang_vel_out"),
                           (torque_out, (B, 3), "torque_out"), (rot_roll_out, (B,), "rot_roll_out")):
        _dev(t, shape, name, device)
    outs = [a_out, rot_out, old_rot_out, ang_vel_out, old_ang_vel_out, torque_out, rot_roll_out]
    if len({t.data_ptr() for t in outs}) != len(outs):
        raise ValueError("the outputs must not share memory")
    if not B:
        return
    with torch.cuda.device(device):
        rc = e["lib"].xbpo_orient(B, a.data_ptr(), a_out.data_ptr(), mass.data_ptr(), agility.data_ptr(),
                                  speed_fraction.data_ptr(), max_vel.data_ptr(), max_ang_vel.data_ptr(),
                                  rot_agility.data_ptr(), rot.data_ptr(), ang_vel.data_ptr(), rot_out.data_ptr(),
                                  old_rot_out.data_ptr(), ang_vel_out.data_ptr(), old_ang_vel_out.data_ptr(),
                                  torque_out.data_ptr(), rot_roll_out.data_ptr(), float(dt), float(fdt),
                                  float(ffriction), torch.cuda.current_stream(device).cuda_stream, int(block))
    _check(rc, e["lib"], "xbpo_orient")


def orient_park(park, a: torch.Tensor, block: int = 128, lib: dict | None = None) -> torch.Tensor:
    """VecOrientPark's dynamical orientation and rotation commit for one tick, by the kernel: the park's torque,
    rot_roll, rot, old_rot, ang_vel and old_ang_vel rebound to new tensors (as the torch path rebinds them); returns the
    thrust along the heading (the torch path's _orient's value)."""
    c = lambda t: t if t.is_contiguous() else t.contiguous()
    a = c(a.to(F64))
    rot, w = c(park.rot), c(park.ang_vel)
    a_out, torque, rot_roll = torch.empty_like(a), torch.empty_like(a), torch.empty_like(park.mass)
    rot_new, old_rot, w_new, old_w = torch.empty_like(rot), torch.empty_like(rot), torch.empty_like(w), torch.empty_like(w)
    orient(a, a_out, c(park.mass), c(park.agility), c(park.speed_fraction), c(park.max_vel), c(park.max_ang_vel),
           c(park.rot_agility), rot, w, rot_new, old_rot, w_new, old_w, torque, rot_roll, park.dt, park.fdt,
           park.ffriction, block, lib)
    park.torque, park.rot_roll = torque, rot_roll
    park.rot, park.old_rot, park.ang_vel, park.old_ang_vel = rot_new, old_rot, w_new, old_w
    return a_out


def libm(name: str, x: torch.Tensor, y: torch.Tensor | None = None, block: int = 128,
         lib: dict | None = None) -> torch.Tensor:
    """The kernel's own device function `name` (expf, logf, sinf, cosf, acosf, atanf: of floats held in float64;
    atan2f(y, x); pow(x, y) of doubles) on a CUDA tensor, for the tests that hold it to vec_libm and the C library."""
    e = lib or load_library()
    x = x.to(F64).contiguous()
    y = (x if y is None else y.to(F64).contiguous())
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        rc = e["lib"].xbpo_libm(LIBM_FUNCS[name], x.numel(), x.data_ptr(), y.data_ptr(), out.data_ptr(),
                                torch.cuda.current_stream(x.device).cuda_stream, int(block))
    _check(rc, e["lib"], "xbpo_libm")
    return out


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, single: bool = False, block: int = 128,
        lib: dict | None = None) -> torch.Tensor:
    """The device's native fused multiply-add, a b + c rounded once: __fma_rn of doubles, or __fmaf_rn of floats held
    in float64 (single=True), for the tests that hold it to vec_exp.fma and vec_libm.fmaf."""
    e = lib or load_library()
    a, b, c = (t.to(F64).contiguous() for t in (a, b, c))
    out = torch.empty_like(a)
    with torch.cuda.device(a.device):
        rc = e["lib"].xbpo_fma(int(single), a.numel(), a.data_ptr(), b.data_ptr(), c.data_ptr(), out.data_ptr(),
                               torch.cuda.current_stream(a.device).cuda_stream, int(block))
    _check(rc, e["lib"], "xbpo_fma")
    return out


def kernel_info(device=None, lib: dict | None = None) -> dict:
    """The orientation kernel as loaded on a device: registers a thread, local memory, binary and PTX versions; and the
    library's manifest (targets, ptxas, the PTX check)."""
    e = lib or load_library()
    vals = [ctypes.c_int() for _ in range(5)]
    with torch.cuda.device(torch.device(device or "cuda")):
        rc = e["lib"].xbpo_info(*[ctypes.byref(v) for v in vals])
    _check(rc, e["lib"], "xbpo_info")
    regs, local, binary, ptx, threads = (v.value for v in vals)
    return {"registers": regs, "local_bytes": local, "binary_version": binary, "ptx_version": ptx,
            "max_threads_per_block": threads, "library": e["manifest"]["library"],
            "library_sha256": e["manifest"]["library_sha256"], "targets": e["manifest"]["targets"],
            "ptxas": e["manifest"]["ptxas"]}


# ================================================================================== a tick captured as a CUDA graph

class TickGraph:
    """A VecOrientPark's tick (`evolve` with kernel="cuda") captured once as a CUDA graph, its new state copied back
    into its inputs inside the graph, so each replay is one tick in one launch: the same kernels on the same buffers
    (exact by construction; held to eager ticks bit for bit by the tests), without the host's cost of launching them
    one by one. The park's state tensors become the graph's fixed buffers; commands and setters rebind them, and
    `tick()` first copies any rebound one into its buffer (`sync()`). Engineering: a caller's own graphs would
    capture the kernel the same way. With `masked=True` the capture is `evolve(active=mask)`, OnTick's
    per-world mask a device buffer written by `tick(active)` (every row when active is None); a masked graph also
    captures a park without dynamical orientation and with ypr="cuda" (vec_ypr_cuda.YprTickGraph's buffers and
    warm-up)."""

    READ = ("pos", "vel", "goto", "mode", "mass", "agility", "speed_fraction", "max_vel", "time_factor", "max_ang_vel",
            "rot_agility", "rot", "ang_vel")
    WRITE = ("pos", "vel", "old_pos", "old_vel", "last_g", "rot", "old_rot", "ang_vel", "old_ang_vel", "torque",
             "rot_roll")

    def __init__(self, park, block: int = 128, masked: bool = False):
        ypr = bool(masked) and not park.dynamical_orientation and getattr(park, "ypr", "host") == "cuda"
        if not (park.dynamical_orientation and park.kernel == "cuda") and not ypr:
            raise ValueError('TickGraph captures a park with dynamical orientation and kernel="cuda" (masked: or one '
                             'without it and with ypr="cuda")')
        self.park, self.block = park, block
        if ypr:
            from . import vec_ypr_cuda as Y
            self.READ, self.WRITE = Y.YprTickGraph.READ, Y.YprTickGraph.WRITE
        self.buf = {f: getattr(park, f).clone() for f in dict.fromkeys(self.READ + self.WRITE)}
        for f, t in self.buf.items():
            setattr(park, f, t)
        if ypr:                                                         # YprTickGraph's warm-up
            z = torch.zeros(4, dtype=F64, device=park.dev)
            for name in ("asin", "asin_sin"):
                Y.dlibm(name, z, block=block)
            Y.dlibm("atan2", z, z, block=block)
        else:
            z = torch.zeros(1, 4, dtype=F64, device=park.dev)           # warm-up: the module loaded before capture
            orient(z[:, :3], z[:, :3].clone(), *(z[:, 0].clone() for _ in range(6)), z, z[:, :3].contiguous(),
                   z.clone(), z.clone(), z[:, :3].clone(), z[:, :3].clone(), z[:, :3].clone(), z[:, 0].clone(),
                   park.dt, park.fdt, park.ffriction, block)
        torch.cuda.synchronize(park.dev)
        self.mask = torch.ones(park.n, dtype=torch.bool, device=park.dev) if masked else None
        self.graph = torch.cuda.CUDAGraph()
        t0 = park.current_time
        with torch.cuda.graph(self.graph):
            if self.mask is None:
                park.evolve()
            else:
                park.evolve(active=self.mask)
            outs = {f: getattr(park, f) for f in self.WRITE}
            # evolve's old_pos and old_vel are its input pos and vel themselves: copied before those are overwritten
            outs = {f: t.clone() if any(t is b for b in self.buf.values()) else t for f, t in outs.items()}
            for f, t in outs.items():
                self.buf[f].copy_(t)
        park.current_time = t0
        for f, t in self.buf.items():
            setattr(park, f, t)

    def sync(self) -> None:
        """Copy every state tensor a command or setter rebound into the graph's buffer, and rebind it to the buffer."""
        for f, b in self.buf.items():
            t = getattr(self.park, f)
            if t is not b:
                b.copy_(t)
                setattr(self.park, f, b)

    def tick(self, active: torch.Tensor | None = None) -> None:
        if active is not None and self.mask is None:
            raise ValueError("a graph made without OnTick's mask (masked=True) ticks every row")
        self.sync()
        if self.mask is not None:
            if active is None:
                self.mask.fill_(True)
            else:
                self.mask.copy_(active.to(device=self.mask.device, dtype=torch.bool).reshape(self.park.n))
        self.graph.replay()
        self.park.current_time += 1

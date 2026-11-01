# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's gradient collision step as one CUDA kernel:
one thread a responder (a world's free ball in one of the park's rows) running the algorithm of
`exact_ballpark.vec_collide.VecCollidePark._gradient` (Destiny's Gradient and Potential, Ballpark.cpp:2745 and 2789: the
window's candidates as the park holds them, the time of impact, the response and its recursion, the contact order of
the partition's walk, the impact record and the collision list), so a tick's collision step is one launch instead
of some thousands of small operations. The park's other parts of the tick are one launch each, by the same rules:
`behaviour` (GOTO's thrust, STOP's damping, the lists cleared), `commit` (the end of the tick, Integrate with mLastG +
mLastC, the commit and the re-filing) and `filing` (the partition's filing), so
Destiny's tick is four launches. `VecCollidePark(..., kernel="cuda")` selects them; the torch path (`kernel="torch"`,
the default) stays the exactness reference, unchanged. Checked bit for bit against the torch module (on the card and
on the CPU) and the scalar reference by tests/test_movement_vec_collide_cuda.py; the
iterative method stays on the torch path. A park with static collidable slots (capsules and
oriented boxes) runs the statics build (XBPC_STATICS=1, `statics`), its own library: each responder walks its world's
static collidables after its balls, in the partition walk's order, through CollideWithBall and ReactToCollision, as
vec_collide._gradient_statics (tests/test_movement_vec_static_collidables.py); every addition is inside
`#if XBPC_STATICS`, so the default build compiles to the machine code it compiled to before. The statics build also
carries the minis' order and the force fields (tests/test_movement_vec_minis_and_fields.py): a park
made with `minis` or `fields` runs it, with or without static slots; its candidates in BallSortComparer's order (the
park's `sort_id`, a ball's rank) and less the force fields the responder passes (the park's mode, harmonic,
corporation and alliance), as vec_collide's `_sorted` and `_field_passed`.

The source is vec_collide_cuda.cu beside this module. The library is built on first use (or when the source, the
generated header, the flags, the planted-fault switches or nvcc change) into a build directory outside any source tree
($EXACT_BALLPARK_COLLIDE_CUDA_DIR; by default $EXACT_BALLPARK_CACHE/collide_cuda), with a manifest; it is loaded
through ctypes, its kernels' module loaded at once (`xbpc_info`), and launched on torch's current stream with no
allocation and no host synchronisation, so a launch can be captured in a CUDA graph (tests/graph_capture.py).

**Exactness by construction.**
- Every float64 operation is a correctly rounded IEEE operation in the torch module's order (+, -, *, /, __dsqrt_rn):
  nvcc with -fmad=false (nothing contracted), -prec-div=true, -prec-sqrt=true, -ftz=false and no fast math; the build
  reads the PTX back and refuses an add, sub, mul, div, sqrt or rcp without .rn, any mad, flush, approximation,
  transcendental instruction or float32 arithmetic, and a count of fma.rn other than the source's own (`ptx_check`).
- A fused multiply-add only where exact_ballpark.vec_exp emulates one (glibc 2.34 aarch64's exp fuses its multiply-adds):
  __fma_rn is the IEEE fused operation, rounded once, which vec_exp's emulation (Boldo and Melquiond's, from correctly
  rounded operations) computes exactly while no partial product overflows or underflows; the tests hold the native
  operation to the emulation and to math.fma (`fma`), and the kernel's exp to vec_exp and to the C library (`exp`).
- No function of CUDA's math library (`scan_source`): the C library's exp is transcribed in the source, its constants
  and its table generated from vec_exp into a header (`tables_header`), with vec_collide's own constants and literals,
  never typed.
- The contact order is vec_collide._order_keys's key (integers), then the other ball's id, then its slot: the two
  stable sorts' order.

**Differences in form from the torch path** (the values are the same): the kernels update the park's tensors in place
(the collision step last_c, last_collision, hits, hits_n, imp_slot, imp_s, imp_closing, imp_n, overflow OR-ed in; the
behaviours last_g, vel, last_c and the lists; the commit pos, vel, old_pos, old_vel, level, key, mod), where the torch
path rebinds them; and the collision kernel computes the impact branch for every pair that reaches it, so
VecCollidePark's impact_slots (the torch path's cap on that branch, an overflow event past it) does not apply.
contact_slots does: with a number, a ball's first contact_slots contacts in order are applied and more is the same
overflow event (check()).
"""

from __future__ import annotations

import ctypes
import datetime
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import torch

from . import vec_exp as E

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
from ._cuda_build import SourcePath, cache_root, default_nvcc, read_source  # noqa: E402,F401
SOURCE = SourcePath(Path(__file__).with_name("vec_collide_cuda.cu"))
HEADER = "vec_collide_cuda_tables.h"
NVCC = default_nvcc()
FLAGS = ["-O3", "-std=c++17", "-fmad=false", "-prec-div=true", "-prec-sqrt=true", "-ftz=false",
         "-Xcompiler", "-fPIC,-ffp-contract=off,-fno-fast-math", "-shared", "-Xptxas", "-v"]
TARGETS = {
    "native": [("compute_90", "sm_90"), ("compute_100", "sm_100"), ("compute_90", "compute_90")],
    "ptx": [("compute_90", "compute_90")],
    "sm90": [("compute_90", "sm_90"), ("compute_90", "compute_90")],    # the tests' planted-fault builds
}
ABI = 1
BLOCK = 128
FAULTS = {"contracted_fma": 1, "libm_exp": 2, "reversed_order": 3}   # planted faults (XBPC_FAULT), tests only
COMPOSED_MODES = ("MISSILE", "MINIBALL", "MUSHROOM")       # the composed build's modes (exact_ballpark.modes)
# the statics build's constants, read from exact_ballpark.vec_collide by these names: the record's
# layout (static_record), the kind of a capsule, ReactToCollision's margin, BoxShape's thresholds and DBL_MAX; and its
# two tables, BoxShape's connected sides and side mapping
STATIC_INTS = ("G_A", "G_B", "G_AB", "G_RAD", "G_CENTER", "G_LO", "G_HI", "G_AXES", "G_UNIT", "G_HALF", "G_LENSQ",
               "G_CORNERS", "G_RECT", "G_RNORM", "G_TNORM", "G_SIZE", "I_KIND", "I_ID", "I_PARENT", "I_LEVEL", "I_NBOX",
               "I_SIZE", "STATIC_CAPSULE",
               "FIELD_MODE", "HARMONIC_PASS", "NO_GROUP")      # the force fields' filter
STATIC_DOUBLES = {"K_EJECT_MARGIN": "EJECT_MARGIN", "K_STATIC_TINY": "STATIC_TINY", "K_STATIC_FLOOR": "STATIC_FLOOR",
                  "K_DBL_MAX": "DBL_MAX"}
EXP_FMAS = 8                                                 # vec_exp.exp's fma calls (one xbpc_exp in the PTX)
_LIBS: dict = {}

# vec_collide's literals in its gradient expressions, by name in the header, each with the text that holds it there
# (`literals_in_source` checks that vec_collide still writes them so)
LITERALS = {"K_TAYLOR": (1e-10, "(k < 1e-10 * m * t)"), "K_FIXED_MASS": (1.0e34, "torch.full_like(self.mass, 1.0e34)"),
            "K_EARLY": (0.0001, "_dot(dv, dv) > 0.0001"), "K_STOP_FRACTION": (0.07, "(vy - 0.07 * vy) * STOP_DAMP"),
            "K_MOVED": (0.1, "~(_dot(d, d) < 0.1)")}

# CUDA's math library and its approximate or directed intrinsics, by name: none may be called (`scan_source`); the
# kernel's own functions have other names (xbpc_exp, trunc_rz)
MATH_LIBRARY = (
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "ceil", "copysign", "cos", "cosh", "cospi",
    "erf", "erfc", "erfinv", "exp", "exp10", "exp2", "expm1", "fabs", "fdim", "floor", "fma", "fmax", "fmin", "fmod",
    "frexp", "hypot", "ilogb", "ldexp", "lgamma", "llrint", "llround", "log", "log10", "log1p", "log2", "logb",
    "lrint", "lround", "modf", "nearbyint", "nextafter", "norm3d", "pow", "rcbrt", "remainder", "remquo", "rhypot",
    "rint", "round", "rsqrt", "scalbn", "sin", "sincos", "sincospi", "sinh", "sinpi", "sqrt", "tan", "tanh", "tgamma",
    "trunc", "j0", "j1", "y0", "y1")
MATH_INTRINSICS = ("__expf", "__exp10f", "__logf", "__log2f", "__log10f", "__powf", "__sinf", "__cosf", "__tanf",
                   "__sincosf", "__fdividef", "__saturatef", "__frsqrt_rn", "__frcp_rn", "__drcp_rn", "__dsqrt_rd",
                   "__dsqrt_ru", "__dsqrt_rz", "__ddiv_rd", "__ddiv_ru", "__ddiv_rz", "__dadd_rd", "__dadd_ru",
                   "__dadd_rz", "__dmul_rd", "__dmul_ru", "__dmul_rz", "__fma_rz", "__fma_ru", "__fma_rd")


# ============================================================================================ the generated header

def literals_in_source() -> dict:
    """Each of LITERALS: whether vec_collide's source still holds it in the expression named (the header's value is
    the torch module's literal)."""
    from . import vec_collide as V
    text = inspect.getsource(V)
    return {name: phrase in text for name, (_, phrase) in LITERALS.items()}


def table_values(composed: bool = False, statics: bool = False) -> dict:
    """Every constant and table the kernel uses, by its name in the header: (kind, value or values), read from
    exact_ballpark.vec_exp (glibc's exp: its constants and the 256 table words) and exact_ballpark.vec_collide and space
    (the response's constants, the partition's sort-key constants, the literals of LITERALS), never typed. Kinds: d (a
    double), u64 (words), i64 (integers)."""
    from . import ref
    from . import space as S
    from . import vec_collide as V
    out = {"K_EXP_INVLN2N": ("d", E.INVLN2N), "K_EXP_NEGLN2HIN": ("d", E.NEGLN2HIN),
           "K_EXP_NEGLN2LON": ("d", E.NEGLN2LON), "K_EXP_C2": ("d", E.C2), "K_EXP_C3": ("d", E.C3),
           "K_EXP_C4": ("d", E.C4), "K_EXP_C5": ("d", E.C5), "K_TWO_1009": ("d", E.TWO_1009),
           "K_TWO_M1022": ("d", E.TWO_M1022), "T_EXP": ("u64", tuple(E.TABLE)),
           "K_LIGHT_BALL_RATIO": ("d", S.LIGHT_BALL_RATIO), "K_RESPONSE_SCALE": ("d", S.RESPONSE_SCALE),
           "XBPC_NUM_LEVELS": ("i64", S.NUM_LEVELS), "XBPC_CODE_BASE": ("i64", V.CODE_BASE),
           "XBPC_CODE_SPAN": ("i64", V.CODE_SPAN), "XBPC_NOT_A_CANDIDATE": ("i64", V.NOT_A_CANDIDATE),
           "XBPC_GOTO": ("i64", ref.GOTO), "XBPC_STOP": ("i64", ref.STOP), "K_STOP_DAMP": ("d", ref.STOP_DAMP),
           "K_LEVEL_WIDTH": ("d", tuple(S.LEVEL_WIDTH)), "K_LEVEL_HALF": ("d", tuple(V.HALF)),
           "K_LEVEL_FAC": ("d", tuple(V.FAC)), "K_LEVEL_CENTER": ("d", tuple(V.CENTER)),
           "K_LEVEL_GRID": ("i64", tuple(S.LEVEL_GRID))}
    for name, (value, _) in LITERALS.items():
        out[name] = ("d", value)
    if V.CODE_SPAN != V.CODE_BASE ** (S.NUM_LEVELS - 1):
        raise RuntimeError("vec_collide's CODE_SPAN is not CODE_BASE ** (NUM_LEVELS - 1), the kernel's powers")
    if composed:                                   # the composed park's modes (the switch)
        from . import modes as MO
        for name in COMPOSED_MODES:
            out[f"XBPC_{name}"] = ("i64", getattr(MO, name))
    if statics:                                    # the static collidables' constants (the switch)
        for name in STATIC_INTS:
            out[f"XBPC_{name}"] = ("i64", getattr(V, name))
        for name, attr in STATIC_DOUBLES.items():
            out[name] = ("d", getattr(V, attr))
        out["XBPC_CONNECTED_SIDES"] = ("i64", tuple(x for row in V.CONNECTED_SIDES for x in row))
        out["XBPC_SIDE_MAPPING"] = ("i64", tuple(x for row in V.SIDE_MAPPING for x in row))
    return out


def tables_header(perturb: dict | None = None, composed: bool = False, statics: bool = False) -> str:
    """The C header of `table_values()`: doubles as exact hexadecimal literals, integers as 64-bit constants, the
    table as a device array of words. `perturb` ({table name: index}) flips the lowest bit of those entries: the
    planted fault of a wrong table entry, for the tests only. `composed` adds the modes the composed park's
    candidates read (the switch); `statics` the static collidables' constants and tables (the switch); without them the header is the one it was."""
    perturb = dict(perturb or {})
    lines = ["// Generated by exact_ballpark.vec_collide_cuda.tables_header() from exact_ballpark.vec_exp (glibc 2.34",
             "// aarch64's exp, its constants and table as read from the library) and exact_ballpark.vec_collide; do not",
             "// edit.", "#pragma once", "#include <cstdint>", ""]
    for name, (kind, v) in table_values(composed, statics).items():
        if kind == "u64":
            words = list(v)
            if name in perturb:
                words[perturb.pop(name)] ^= 1
            body = ", ".join(f"0x{w:016x}ULL" for w in words)
            lines.append(f"static __device__ const uint64_t {name}[{len(words)}] = {{{body}}};")
        elif isinstance(v, tuple):                  # the partition's per-level constants
            ctype, body = (("long long", ", ".join(f"{int(x)}LL" for x in v)) if kind == "i64" else
                           ("double", ", ".join(float(x).hex() for x in v)))
            lines.append(f"static __device__ const {ctype} {name}[{len(v)}] = {{{body}}};")
        elif kind == "i64":
            lines.append(f"constexpr long long {name} = {int(v)}LL;")
        else:
            lines.append(f"constexpr double {name} = {float(v).hex()};   // {float(v)!r}")
    if perturb:
        raise KeyError(f"no such table: {sorted(perturb)}")
    return "\n".join(lines) + "\n"


# ================================================================================================ the static checks

def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def active_text(text: str, fault: int = 0) -> str:
    """The source as compiled with XBPC_FAULT = fault: the lines of each `#if XBPC_FAULT == k` block (to its #else or
    #endif) kept only for k = fault, its #else part only for k != fault (the planted faults' blocks, unnested)."""
    out, state = [], None                    # None outside a fault block; else (k, in_else)
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"#if\s+XBPC_FAULT\s*==\s*(\d+)", s)
        if m and state is None:
            state = (int(m.group(1)), False)
            continue
        if state is not None and s.startswith("#else"):
            state = (state[0], True)
            continue
        if state is not None and s.startswith("#endif"):
            state = None
            continue
        if state is None or (state[0] == fault) != state[1]:
            out.append(line)
    return "\n".join(out) + "\n"


def scan_source(text: str) -> list:
    """Every call of a function of CUDA's math library, or of an approximate or directed-rounding intrinsic, in a CUDA
    source (comments left out): (name, line)."""
    names = set(MATH_LIBRARY) | {n + "f" for n in MATH_LIBRARY} | set(MATH_INTRINSICS)
    found = []
    for k, line in enumerate(_strip_comments(text).splitlines(), 1):
        for m in re.finditer(r"(?<![\w.:>])(?:std::|::)?([A-Za-z_]\w*)\s*\(", line):
            if m.group(1) in names:
                found.append((m.group(1), k))
    return found


_FP = re.compile(r"\b(add|sub|mul|div|sqrt|rcp|rsqrt|fma|mad|neg|abs|min|max|ex2|lg2|sin|cos|tanh)"
                 r"((?:\.[a-z0-9]+)*)\.(f16|f32|f64)\b")
_CVT = re.compile(r"\bcvt((?:\.[a-z0-9]+)+)")
# the floating-point instructions a pinned build may hold: float64 arithmetic rounded to nearest (rcp.rn is the IEEE
# reciprocal, nvcc's form of 1.0 / x, equal to div.rn of 1.0), exact sign operations, the two conversions toward zero
# (vec_exp's frinta to an integral double, an integral double to int64; vec_collide's _to_int64), and int64 to double
# rounded to nearest (vec_collide's key.to(float64), exact below 2^53)
PTX_ALLOWED = ("add.rn.f64", "sub.rn.f64", "mul.rn.f64", "div.rn.f64", "sqrt.rn.f64", "rcp.rn.f64", "fma.rn.f64",
               "neg.f64", "abs.f64", "cvt.rzi.f64.f64", "cvt.rzi.s64.f64", "cvt.rn.f64.s64")


def ptx_check(ptx: str, fmas: int | None = None) -> dict:
    """Every floating-point instruction of the PTX, by form. The build fails on any form outside PTX_ALLOWED (an
    add, sub, mul, div or sqrt without .rn, which a later stage may contract; a mad; anything flushed, approximate,
    transcendental, float32 or float16) and on a count of fma.rn.f64 other than `fmas` (the source's own __fma_rn:
    nvcc's -fmad=false contracts nothing)."""
    forms: dict = {}
    for m in _FP.finditer(ptx):
        key = f"{m.group(1)}{m.group(2)}.{m.group(3)}"
        forms[key] = forms.get(key, 0) + 1
    for m in _CVT.finditer(ptx):                      # conversions that touch a floating-point type
        if re.search(r"\.f(16|32|64)\b", m.group(1)):
            key = "cvt" + m.group(1)
            forms[key] = forms.get(key, 0) + 1
    bad = {k: v for k, v in forms.items() if k not in PTX_ALLOWED}
    n_fma = forms.get("fma.rn.f64", 0)
    if fmas is not None and n_fma != fmas:
        bad["fma.rn.f64 (count)"] = n_fma
    return {"fp_ops": dict(sorted(forms.items())), "fmas_expected": fmas, "refused": bad, "ok": not bad}


# ============================================================================================ build and load

def build_dir() -> Path:
    d = os.environ.get("EXACT_BALLPARK_COLLIDE_CUDA_DIR")
    if d:
        return Path(d)
    return cache_root() / "collide_cuda"


def _nvcc() -> Path:
    if NVCC.exists():
        return NVCC
    found = shutil.which("nvcc")
    if not found:
        raise RuntimeError(f"nvcc not found at {NVCC} nor on PATH (set EXACT_BALLPARK_NVCC)")
    return Path(found)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(out_dir: Path | None = None, variant: str = "native", source: Path = SOURCE, perturb: dict | None = None,
          fault: str | None = None, checks: bool = True, composed: bool = False, statics: bool = False) -> dict:
    """Compile `source` with its generated header into out_dir (by default `build_dir()`); returns the manifest written
    beside the library: the source's and the header's SHA-256, nvcc's version, flags and targets, the kernels'
    registers and spills, the source scan and the PTX check, the library's SHA-256 and the UTC time. A library already
    built from the same inputs is reused. `perturb`, `fault` (a planted fault of FAULTS) and `checks=False` (the scan
    and the PTX check recorded, not enforced) are for the tests' planted faults only. `composed` builds the kernel for
    exact_ballpark.vec_composed's view (XBPC_COMPOSED=1): its own library, the default one untouched.
    `statics` builds the default kernel with the static collidables (XBPC_STATICS=1): its own library,
    the default one untouched; not with `composed`."""
    if variant not in TARGETS:
        raise ValueError(f"unknown variant {variant!r}")
    if composed and statics:
        raise ValueError("the static collidables are the default build's: not with composed")
    level = 0 if fault is None else FAULTS[fault]
    out_dir = Path(out_dir) if out_dir is not None else build_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    text = read_source(source)
    calls = scan_source(active_text(text, level))
    if checks and calls:
        raise RuntimeError(f"the source calls CUDA's math library: {calls}")
    missing = [k for k, ok in literals_in_source().items() if not ok]
    if missing:
        raise RuntimeError(f"vec_collide no longer writes the literals {missing} as the kernel's header holds them")
    header = tables_header(perturb, composed, statics)
    nvcc = _nvcc()
    gencode = [f"-gencode=arch={a},code={c}" for a, c in TARGETS[variant]]
    defines = [f"-DXBPC_FAULT={level}", f"-DXBPC_BLOCK={BLOCK}"] + (["-DXBPC_COMPOSED=1"] if composed else []) + (
        ["-DXBPC_STATICS=1"] if statics else [])
    from .vec_orient_cuda import nvcc_version         # (the compiler's version, once a process)
    key = _sha(json.dumps([_sha(text.encode()), _sha(header.encode()), FLAGS, gencode, defines, str(nvcc),
                           nvcc_version(nvcc), checks]).encode())[:16]
    stem = f"vec_collide_cuda_{variant}{'_composed' if composed else ''}{'_statics' if statics else ''}_{key}"
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
        cmd = [str(nvcc), *FLAGS, *defines, *gencode, "-I", str(work), "-o", str(work / "lib.so"), str(src)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
        cuobjdump = nvcc.parent / "cuobjdump"
        listing = subprocess.run([str(cuobjdump), "--list-elf", "--list-ptx", str(work / "lib.so")],
                                 capture_output=True, text=True, check=True).stdout.splitlines()
        ptx = subprocess.run([str(cuobjdump), "-ptx", str(work / "lib.so")], capture_output=True, text=True,
                             check=True).stdout
        check = ptx_check(ptx, EXP_FMAS + 1)
        if checks and not check["ok"]:
            raise RuntimeError(f"the PTX has unpinned floating-point instructions: {check['refused']}")
        resources = [ln.strip() for ln in (r.stdout + r.stderr).splitlines()
                     if "registers" in ln or "spill" in ln or "Compiling entry" in ln or "stack frame" in ln]
        version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True).stdout
        manifest = {
            "variant": variant, "source": source.name if Path(source) == SOURCE else str(source),
            "source_sha256": _sha(text.encode()), "header": HEADER, "header_sha256": _sha(header.encode()),
            "perturb": perturb or {}, "fault": fault, "checks": checks, "source_scan": calls,
            **({"composed": True} if composed else {}), **({"statics": True} if statics else {}),
            "nvcc": version.strip().splitlines()[-1], "nvcc_path": str(nvcc), "flags": FLAGS, "defines": defines,
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


def load_library(variant: str = "native", perturb: dict | None = None, fault: str | None = None,
                 checks: bool = True, composed: bool = False, source: Path = SOURCE, statics: bool = False) -> dict:
    """The compiled library (built if missing or stale), its functions typed, its kernels' module loaded on the current
    CUDA device (`xbpc_info`: no later launch loads it, as a captured one must not), and its manifest; once a
    process. `composed` and `source` (a planted fault's copy of the source) are the composed park's;
    `statics` the static collidables' build."""
    k = (variant, json.dumps(perturb or {}, sort_keys=True), fault, checks) + (
        (composed, str(source), _sha(Path(source).read_bytes())) if composed or source != SOURCE else ()) + (
        ("statics",) if statics else ())
    if k in _LIBS:
        return _LIBS[k]
    manifest = build(None, variant, source, perturb, fault, checks, composed, statics)
    lib = ctypes.CDLL(str(build_dir() / manifest["library"]))
    if lib.xbpc_abi() != ABI:
        raise RuntimeError(f"the library's ABI {lib.xbpc_abi()} is not {ABI}")
    p, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.xbpc_error_string.restype = ctypes.c_char_p
    lib.xbpc_error_string.argtypes = [i32]
    lib.xbpc_gradient.argtypes = [p, p, p, p, i32]
    lib.xbpc_gradient.restype = i32
    lib.xbpc_exp.argtypes = [p, p, i64, p, i32]
    lib.xbpc_exp.restype = i32
    lib.xbpc_fma.argtypes = [p, p, p, p, i64, p, i32]
    lib.xbpc_fma.restype = i32
    lib.xbpc_tick.argtypes = [i32, p, p, p, p, i32]
    lib.xbpc_tick.restype = i32
    lib.xbpc_info.argtypes = [ctypes.POINTER(i32)]
    lib.xbpc_info.restype = i32
    if lib.xbpc_fault() != (0 if fault is None else FAULTS[fault]):
        raise RuntimeError("the library's planted-fault switch is not the one asked for")
    if composed and lib.xbpc_composed() != 1:
        raise RuntimeError("the library's composed switch is not on")
    if statics and lib.xbpc_statics() != 1:
        raise RuntimeError("the library's statics switch is not on")
    entry = {"lib": lib, "manifest": manifest}
    entry["info"] = _info(entry)
    _LIBS[k] = entry
    return entry


def _info(e: dict) -> dict:
    out = (ctypes.c_int * 6)()
    _check(e["lib"].xbpc_info(out), e["lib"], "xbpc_info")
    keys = ("registers", "local_bytes", "max_threads_per_block", "binary_version", "ptx_version", "device_cc")
    return dict(zip(keys, list(out)))


def info(lib: dict | None = None) -> dict:
    """What this process runs: the gradient kernel as loaded on the current device (registers, local memory, versions,
    the device's compute capability) and the library's manifest."""
    e = lib or load_library()
    return {**e["info"], "library": e["manifest"]["library"], "library_sha256": e["manifest"]["library_sha256"],
            "targets": e["manifest"]["targets"], "ptxas": e["manifest"]["ptxas"]}


def _check(rc: int, lib, fn: str) -> None:
    if rc:
        msg = lib.xbpc_error_string(rc).decode() if rc > 0 else "refused (block size or arguments)"
        raise RuntimeError(f"{fn} failed ({rc}: {msg})")


# ======================================================================================================= launches

# the park's tensors the kernel reads and writes, in xbpc_gradient's order: (attribute, dtype, shape by (W, N, R, C))
_IN = (("pos", F64, "WN3"), ("vel", F64, "WN3"), ("last_g", F64, "WN3"), ("mass", F64, "WN"),
       ("agility", F64, "WN"), ("time_factor", F64, "WN"), ("radius", F64, "WN"), ("exists", BOOL, "WN"),
       ("free", BOOL, "WN"), ("massive", BOOL, "WN"), ("junk", BOOL, "WN"), ("cloaked", BOOL, "WN"))
_IDS = (("id", I64, "WN"), ("level", I64, "WN"), ("key", I64, "WN3"), ("mod", I64, "WN3"))
_OUT = (("last_c", F64, "WN3"), ("last_collision", F64, "WN"), ("imp_s", F64, "WRI"), ("imp_closing", F64, "WRI"),
        ("hits", I64, "WRH"), ("hits_n", I64, "WR"), ("imp_slot", I64, "WRI"), ("imp_n", I64, "WR"),
        ("overflow", BOOL, "W"))


def _shape(code: str, W: int, N: int, R: int, H: int, C: int) -> tuple:
    return tuple({"W": W, "N": N, "R": R, "H": H, "I": C, "3": 3}[ch] for ch in code)


def _on(t: torch.Tensor, dev: torch.device) -> bool:
    """Whether a tensor is on the park's device (a device without an index: the current one's)."""
    return t.device.type == dev.type and (dev.index is None or t.device.index == dev.index)


def _held(park, name: str, dtype, shape: tuple) -> torch.Tensor:
    """The park's tensor `name`, contiguous (rebound to a contiguous copy if it is not: the kernel writes in place),
    checked for its type, shape and device."""
    t = getattr(park, name)
    if t.dtype != dtype or tuple(t.shape) != shape or not _on(t, park.dev):
        raise ValueError(f"park.{name}: {t.dtype} {tuple(t.shape)} on {t.device}, the kernel takes {dtype} {shape} on "
                         f"{park.dev}")
    if not t.is_contiguous():
        t = t.contiguous()
        setattr(park, name, t)
    return t


def gradient(park, responder: torch.Tensor, block: int = BLOCK, lib: dict | None = None) -> None:
    """One tick's gradient collision step of a VecCollidePark (its `_gradient`'s work, the module's docstring) for the
    balls in `responder` [W, N] (live, massive, free), by one launch on torch's current stream of the park's device:
    last_c, last_collision, the collision lists and the impact record updated in place, overflow OR-ed in; with the
    park's `bubble` set (the master's bubbles) a candidate in another bubble is skipped. With the
    park's static slots (`statics` > 0), the statics build, each responder's static collidables after
    its balls; with `minis` or `fields` the statics build as well, its candidates in
    BallSortComparer's order and less the force fields passed; with `global_balls` the statics build
    as well, a candidate in another bubble kept when it is global. No host synchronisation and no allocation (a launch
    can be captured in a CUDA graph)."""
    S = getattr(park, "S", 0)
    minis, fields = bool(getattr(park, "minis", False)), bool(getattr(park, "fields", False))
    glob = bool(getattr(park, "global_balls", False))
    e = lib or (load_library(statics=True) if S or minis or fields or glob else load_library())
    W, N, R = park.W, park.N, park.R
    H, C = park.hits.shape[-1], park.imp_slot.shape[-1]
    if park.dev.type != "cuda":
        raise ValueError("the collision kernel needs the park on a CUDA device")
    if responder.dtype != BOOL or tuple(responder.shape) != (W, N) or not _on(responder, park.dev):
        raise ValueError(f"responder must be a bool tensor of shape {(W, N)} on {park.dev}")
    responder = responder.contiguous()
    ins = [_held(park, n, dt, _shape(s, W, N, R, H, C)) for n, dt, s in _IN]
    ids = [_held(park, n, dt, _shape(s, W, N, R, H, C)) for n, dt, s in _IDS]
    outs = [_held(park, n, dt, _shape(s, W, N, R, H, C)) for n, dt, s in _OUT]
    if len({t.data_ptr() for t in outs}) != len(outs):
        raise ValueError("the park's collision tensors must not share memory")
    limit = park.contact_slots is not None
    slots = min(int(park.contact_slots), N - 1) if limit else N - 1
    tensors = ins + [responder] + ids + outs
    # the default build's last pointer: the master's bubble per slot, or null for a client's view
    bubble = None if getattr(park, "bubble", None) is None else _held(park, "bubble", I64, (W, N))
    if not (S or minis or fields or glob):
        dims = (ctypes.c_int64 * 7)(W, N, R, slots, int(limit), H, C)
        ptrs = (ctypes.c_void_p * (len(tensors) + 1))(*[t.data_ptr() for t in tensors],
                                                      None if bubble is None else bubble.data_ptr())
    else:                                           # the statics build: two more dims, four pointers
        from .vec_collide import G_SIZE, I_SIZE
        if bubble is not None and park.has_statics:
            raise ValueError("static collidables on the master's view (bubble set) are not in this module")
        if bubble is not None and (minis or fields):
            raise ValueError("minis and force fields on the master's view (bubble set) are not in this module")
        if S:
            B = park.B
            st = [_held(park, n, dt, shape) for n, dt, shape in (("s_exists", BOOL, (W, S)),
                                                                  ("s_geo", F64, (W, S, G_SIZE)),
                                                                  ("s_int", I64, (W, S, I_SIZE)),
                                                                  ("s_keys", I64, (W, S, B, 3)))]
            st_ptrs = [t.data_ptr() for t in st]
        else:                                       # minis or fields without static slots
            B, st_ptrs = 1, [None] * 4
        # five more pointers, BallSortComparer's rank per slot (or null: the id) and the force
        # fields' mode, harmonic, corporation and alliance (or null: no fields)
        sid = [_held(park, "sort_id", I64, (W, N)).data_ptr() if minis else None]
        fl = ([_held(park, n, I64, (W, N)).data_ptr() for n in ("mode", "harmonic", "corporation", "alliance")]
              if fields else [None] * 4)
        # one more pointer, the global flag per slot (or null: no global balls)
        gl = [_held(park, "glob", BOOL, (W, N)).data_ptr() if glob else None]
        dims = (ctypes.c_int64 * 9)(W, N, R, slots, int(limit), H, C, S, B)
        ptrs = (ctypes.c_void_p * (len(tensors) + 11))(*[t.data_ptr() for t in tensors],
                                                       None if bubble is None else bubble.data_ptr(), *st_ptrs, *sid,
                                                       *fl, *gl)
    if W == 0:
        return
    with torch.cuda.device(park.dev):
        rc = e["lib"].xbpc_gradient(dims, ptrs, _consts(park), torch.cuda.current_stream(park.dev).cuda_stream,
                                    int(block))
    _check(rc, e["lib"], "xbpc_gradient")


def _consts(park):
    """The host's doubles the kernels take, computed in Python as vec_collide computes them: k, dt, k2, 1 / k2, 1 / k,
    -k, -k dt (the Taylor branch's), 1 / dt, dt k."""
    k, dt = park.friction, park.dt
    k2 = k * k
    return (ctypes.c_double * 9)(k, dt, k2, 1.0 / k2, 1.0 / k, -k, -k * dt, 1.0 / dt, dt * k)


# xbpc_tick's pointers, in order
_TICK = ("live", "free", "which", "mode", "mass", "agility", "max_vel", "speed_fraction", "time_factor", "radius",
         "goto", "pos", "vel", "old_pos", "old_vel", "last_g", "last_c", "level_in", "key_in", "mod_in", "level", "key",
         "mod", "hits", "hits_n", "imp_slot", "imp_n")
_BALL = {"free": (BOOL, "WN"), "mode": (I64, "WN"), "mass": (F64, "WN"), "agility": (F64, "WN"),
         "max_vel": (F64, "WN"), "speed_fraction": (F64, "WN"), "time_factor": (F64, "WN"), "radius": (F64, "WN"),
         "goto": (F64, "WN3"), "pos": (F64, "WN3"), "vel": (F64, "WN3"), "old_pos": (F64, "WN3"),
         "old_vel": (F64, "WN3"), "last_g": (F64, "WN3"), "last_c": (F64, "WN3"), "level": (I64, "WN"),
         "key": (I64, "WN3"), "mod": (I64, "WN3"), "hits": (I64, "WRH"), "hits_n": (I64, "WR"),
         "imp_slot": (I64, "WRI"), "imp_n": (I64, "WR")}


def _tick(park, part: int, mask_name: str, mask: torch.Tensor, names: tuple, writes: tuple, extra: dict | None = None,
          block: int = BLOCK, lib: dict | None = None) -> None:
    e = lib or load_library()
    W, N, R = park.W, park.N, park.R
    H, C = park.hits.shape[-1], park.imp_slot.shape[-1]
    if park.dev.type != "cuda":
        raise ValueError("the collision kernel needs the park on a CUDA device")
    if mask.dtype != BOOL or tuple(mask.shape) != (W, N) or not _on(mask, park.dev):
        raise ValueError(f"{mask_name} must be a bool tensor of shape {(W, N)} on {park.dev}")
    t = {mask_name: mask.contiguous()}
    for n in names:
        dt, code = _BALL[n]
        t[n] = _held(park, n, dt, _shape(code, W, N, R, H, C))
    t.update(extra or {})
    if len({t[n].data_ptr() for n in writes}) != len(writes):
        raise ValueError("the park's tensors a launch writes must not share memory")
    dims = (ctypes.c_int64 * 6)(W, N, R, H, C, int(park.collisions))
    ptrs = (ctypes.c_void_p * len(_TICK))(*[t[n].data_ptr() if n in t else None for n in _TICK])
    if W == 0:
        return
    with torch.cuda.device(park.dev):
        rc = e["lib"].xbpc_tick(part, dims, ptrs, _consts(park), torch.cuda.current_stream(park.dev).cuda_stream,
                                int(block))
    _check(rc, e["lib"], "xbpc_tick")


def behaviour(park, live: torch.Tensor, block: int = BLOCK, lib: dict | None = None) -> None:
    """VecCollidePark._behaviour (phase 1 for the free balls in `live` [W, N]: mLastG from GOTO's thrust, STOP's
    vertical damping, mLastC and the rows' collision lists cleared) by one launch, in place."""
    writes = ("vel", "last_g", "last_c", "hits", "hits_n", "imp_slot", "imp_n")
    _tick(park, 0, "live", live, ("mode", "mass", "agility", "max_vel", "speed_fraction", "goto", "pos") + writes,
          writes, block=block, lib=lib)


def commit(park, live: torch.Tensor, block: int = BLOCK, lib: dict | None = None) -> None:
    """VecCollidePark.evolve's end of the tick for the free balls in `live` on the gradient path (Integrate over the
    tick with mLastG + mLastC, or mLastG alone with collisions off; the commit; the re-filing of the balls that moved
    at least sqrt(0.1) m) by one launch, in place."""
    writes = ("pos", "vel", "old_pos", "old_vel", "level", "key", "mod")
    _tick(park, 1, "live", live, ("free", "mass", "agility", "max_vel", "speed_fraction", "time_factor", "radius",
                                  "last_g", "last_c") + writes, writes, block=block, lib=lib)


def filing(park, which: torch.Tensor, block: int = BLOCK, lib: dict | None = None) -> tuple:
    """VecCollidePark._filing (Ball::InsertInBoxes's outcome for the balls in `which`, the others keeping theirs) by one
    launch: new (level, key, mod) tensors."""
    W, N = park.W, park.N
    level = torch.empty((W, N), dtype=I64, device=park.dev)
    key, mod = torch.empty((W, N, 3), dtype=I64, device=park.dev), torch.empty((W, N, 3), dtype=I64, device=park.dev)
    old = {"level_in": _held(park, "level", I64, (W, N)), "key_in": _held(park, "key", I64, (W, N, 3)),
           "mod_in": _held(park, "mod", I64, (W, N, 3)), "level": level, "key": key, "mod": mod}
    _tick(park, 2, "which", which, ("free", "mass", "agility", "max_vel", "speed_fraction", "time_factor", "radius",
                                    "pos", "vel"), ("level", "key", "mod"), extra=old, block=block, lib=lib)
    return level, key, mod


def exp(x: torch.Tensor, block: int = BLOCK, lib: dict | None = None) -> torch.Tensor:
    """The kernel's own exp (xbpc_exp, the transcription of vec_exp.exp) on a float64 CUDA tensor, for the tests that
    hold it to vec_exp and the C library."""
    e = lib or load_library()
    x = x.to(F64).contiguous()
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        rc = e["lib"].xbpc_exp(x.data_ptr(), out.data_ptr(), x.numel(), torch.cuda.current_stream(x.device).cuda_stream,
                               int(block))
    _check(rc, e["lib"], "xbpc_exp")
    return out


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, block: int = BLOCK, lib: dict | None = None) -> torch.Tensor:
    """The native __fma_rn on float64 CUDA tensors (the operation the kernel's exp uses), for the tests that hold it to
    vec_exp.fma's emulation and math.fma."""
    e = lib or load_library()
    a, b, c = (t.to(F64).contiguous() for t in (a, b, c))
    if not (a.shape == b.shape == c.shape):
        raise ValueError("fma takes three tensors of one shape")
    out = torch.empty_like(a)
    with torch.cuda.device(a.device):
        rc = e["lib"].xbpc_fma(a.data_ptr(), b.data_ptr(), c.data_ptr(), out.data_ptr(), a.numel(),
                               torch.cuda.current_stream(a.device).cuda_stream, int(block))
    _check(rc, e["lib"], "xbpc_fma")
    return out

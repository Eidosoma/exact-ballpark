# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's flight modes (GOTO, STOP, FOLLOW, ORBIT under both laws, WARP) with their transitions and dynamical
orientation as one CUDA kernel a tick, one thread a ball: the step that `exact_ballpark.vec_modes.VecModesPark.evolve`
launches, run as the scalar algorithm of `exact_ballpark.modes.ModeBallpark` and `exact_ballpark.full.FullBallpark`. A tick
is two launches: a prepass that records what each ball shows the others during phase 1 (its start-of-tick position and
velocity, its vertical velocity after its own turn, whether it starts warp proper this tick), then the main kernel,
which runs every ball's phases 1, 3 and 4 in place and advances its world's clock. Checked bit for bit against the
scalar reference by tests/test_movement_vec_modes.py (Destiny's own code compiled verbatim is the reference's
reference: tests/test_movement_modes_oracle.py, tests/test_movement_full_oracle.py, tests/test_movement_turn_oracle.py).
(31 October 2026; tests/test_movement_vec_modes_leftovers.py): the follower sets as StopAllFollowers acts
on them (a member that SetBoid took out of a follow mode included), the troll's petrification at the end of the tick it
is due, and the sine and cosine of one argument from glibc's sincos (g_sincos), as compiled Destiny takes them.

The source is vec_modes_cuda.cu beside this module. It includes vec_orient_cuda.cu (the kernel: the C
library's float functions and pow, carbonengine's vectors and quaternions, the torque searches, ApplyTorque), so the
dynamical orientation here is that code, not a copy. The library is built on first use (or when either source, either
generated header, the flags or nvcc change) into a build directory outside any source tree ($EXACT_BALLPARK_MODES_CUDA_DIR; by default $EXACT_BALLPARK_CACHE/modes_cuda), with a manifest; it is loaded through ctypes and launched on
torch's current stream, with no allocation and no host synchronisation, so a tick can be captured in a CUDA graph.

**Exactness by construction** (as vec_orient_cuda, whose source scan and PTX check this build applies to both sources):
every double operation is __dadd_rn ... __dsqrt_rn in the reference's order, every float operation __fadd_rn ...
__fsqrt_rn where Destiny computes in float; a fused multiply-add only where the C library's compiled code fuses one
(the double sin, cos, sincos and log are exact_ballpark.dlibm's transcription of glibc 2.34 aarch64, exp is
exact_ballpark.vec_exp's, the float functions and pow vec_libm's through vec_orient_cuda); nvcc with -fmad=false,
-prec-div=true,
-prec-sqrt=true, -ftz=false and no fast math; no function of CUDA's math library; every constant generated into a
header from the Python modules' own values (`tables_header`), never typed: dlibm's tables and constants (read out of
libm.so.6 by script), and the reference's literals, taken from exact_ballpark.modes and exact_ballpark.ref (the named ones)
or written here once with the reference's own spelling (`MODE_LITERALS`; a test finds each in the reference's source).
"""

from __future__ import annotations

import ctypes
import datetime
import hashlib
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import torch

from . import dlibm as D
from . import modes as MO
from . import ref as R
from . import vec_orient_cuda as K
from .vec_orient import GraphMask

F64 = torch.float64
from ._cuda_build import SourcePath, cache_root, default_nvcc, read_source  # noqa: E402,F401
SOURCE = SourcePath(Path(__file__).with_name("vec_modes_cuda.cu"))
HEADER = "vec_modes_cuda_tables.h"
FLAGS = K.FLAGS
TARGETS = K.TARGETS
ABI = 1
DLIBM_FUNCS = {"sin": 0, "cos": 1, "log": 2, "exp": 3, "sincos_sin": 4, "sincos_cos": 5}
_LIBS: dict = {}

# The reference's literals that have no name there, with the reference's own spelling (modes.py, full.py); each is
# found in the source by a test. Python's float literal is the C++ double literal Destiny writes.
MODE_LITERALS = {
    "M_STOP_DECAY": 0.07,              # EvolveStop: v.y - 0.07 v.y (MOV-710)
    "M_DIR_DIST": 1.0e17,              # GotoDirection's point 1e17 m away (MOV-709)
    "M_ALIGN_COS": 0.01,               # IsAlignedForWarp: |1 - cos| < 0.01 (TRV-706)
    "M_ALIGN_SPEED2": 0.5625,          # ... and |v|^2 > 0.5625 maxVel^2 (75 %)
    "M_RF_SCALE": 40000.0,             # EvolveOldStyleOrbit: exp(-(r - dist)^2 / 40000)
    "M_TRUNC7": 10000000.0,            # (int64_t)(x 10000000) / 10000000
    "M_GOTO_FAR": 10.0,                # the old orbit's goto point 10 AU along the thrust
    "M_NORMAL_MIN": 1e-12,             # GetOrbitalNormal: |to x v|^2 > 1e-12
    "M_ANGLE_SCALE": 0.1,              # the new orbit's angle 0.1 (r - dist) / speed, and its distance floor 0.1
    "M_ANGLE_ZERO": 1e-10,             # |angle| < 1e-10: a circle; |factor| > 1e-10: the logarithm
    "M_FACTOR_FLOOR": -0.999,          # factor < -0.999: the steepest spiral
    "M_LOG_FLOOR": 0.001,              # log(0.001)
    "M_SLOW_TARGET": 5e-2,             # GotoThrustFollow: a target slower than 5 % of maxVel x speedFraction
    "M_THRUST_OFF": 0.1,               # OrbitThrust: zero below a tenth of the maximum acceleration
    "M_DROP_SPEED": 100.0,             # the warp drops out below min(maxVel / 2, 100) m/s
    "M_HALF": 2.0,                     # ... maxVel / 2.0
    "M_TAYLOR_LIMIT": 1e-10,           # Integrate: k < 1e-10 m t takes the Taylor branch
    "M_DO_THRUST": 1e-2,               # dynamical orientation steers above 1 % of the maximum acceleration
}


# ============================================================================================ the generated header

def _dbits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def table_values() -> dict:
    """Every constant and table this file's code uses, by its name in the header: (kind, value or values), read from
    exact_ballpark.dlibm (glibc's tables and constants), exact_ballpark.modes and exact_ballpark.ref (Destiny's named
    constants) and MODE_LITERALS; never typed. Kinds: d (a double), u64 and u32 (unsigned integers), int (a decimal
    #define); tables are tuples. vec_orient_cuda's own header supplies exp's table and constants."""
    d = "d"
    out = {}
    for name in ("TAYLOR_LIMIT", "BIG", "TOINT", "S1", "S2", "S3", "S4", "S5", "CS2", "CS4", "CS6", "SN3", "SN5", "HPINV",
                 "MP1", "MP2", "PP3", "PP4", "HP0", "HP1", "BR_TM600", "BR_SPLIT", "BR_TM24", "BR_BIG", "BR_BIG1",
                 "BR_TOVERP0", "BR_HP0", "BR_HP1", "BR_MP1", "BR_MP2", "LOG_TWO27", "LOG_TWO52", "LOG_LN2HI",
                 "LOG_LN2LO"):
        out[f"DL_{name}"] = (d, getattr(D, name))
    for name in ("K_TINY_SIN", "K_TINY_COS", "K_DO_SIN", "K_MID", "K_REDUCE", "K_HUGE", "BR_T576_HI"):
        out[f"DL_{name}"] = ("u32", getattr(D, name))
    for name in ("BR_EXP_LIMIT", "BR_EXP_BIAS"):
        out[f"DL_{name}"] = ("int", getattr(D, name))
    for name in ("LOG_NEG_LO", "LOG_RANGE", "LOG_NEG_OFF", "LOG_NEG_52"):
        out[f"DL_{name}"] = ("u64", getattr(D, name))
    out["DL_SINCOSTAB"] = (d, tuple(D.SINCOSTAB))
    out["DL_BR_TOVERP"] = (d, tuple(D.BR_TOVERP))
    out["DL_LOG_A"] = (d, tuple(D.LOG_A))
    out["DL_LOG_B"] = (d, tuple(D.LOG_B))
    out["DL_LOG_INVC"] = (d, tuple(c for c, _ in D.LOG_TAB))
    out["DL_LOG_LOGC"] = (d, tuple(c for _, c in D.LOG_TAB))
    # Destiny's named constants (modes.py, ref.py) and the mode numbers
    out["M_AU"] = (d, MO.AU)
    out["M_ORBITAL_PRECESSION"] = (d, MO.ORBITAL_PRECESSION)
    out["M_WARP_AU_PER_SECOND"] = (d, MO.WARP_FACTOR_TO_AU_PER_SECOND)
    out["M_WARP_ACCELERATION"] = (d, MO.WARP_FACTOR_TO_ACCELERATION)
    out["M_WARP_DECELERATION"] = (d, MO.WARP_FACTOR_TO_DECELERATION)
    out["M_MIN_WARP_DISTANCE_SQ"] = (d, MO.MIN_WARP_DISTANCE_SQ)
    out["M_STOP_DAMP"] = (d, R.STOP_DAMP)
    out["M_MAX_ALIGN_TICKS"] = ("int", MO.MAX_ALIGN_TICKS)
    for name, v in MODE_LITERALS.items():
        out[name] = (d, v)
    for name in ("GOTO", "STOP"):
        out[f"MODE_{name}"] = ("int", getattr(R, name))
    for name in ("FOLLOW", "WARP", "ORBIT", "TROLL", "RIGID"):
        out[f"MODE_{name}"] = ("int", getattr(MO, name))
    return out


def _word(kind: str, v) -> int:
    return _dbits(v) if kind == "d" else int(v)


def tables_header(perturb: dict | None = None) -> str:
    """The C header of `table_values()`: scalars as #defines of their bit patterns, tables as device arrays of bit
    patterns. `perturb` ({table name: index}) flips the lowest bit of those entries: the planted fault of a wrong
    table entry, for the tests only."""
    perturb = dict(perturb or {})
    lines = ["// Generated by exact_ballpark.vec_modes_cuda.tables_header() from exact_ballpark.dlibm (glibc 2.34 aarch64's",
             "// tables and constants, read from the library), exact_ballpark.modes and exact_ballpark.ref; do not edit.",
             "#pragma once", "#include <cstdint>", ""]
    for name, (kind, v) in table_values().items():
        if kind == "int":
            lines.append(f"#define {name} {int(v)}")
        elif isinstance(v, tuple):
            words = [_word(kind, x) for x in v]
            if name in perturb:
                words[perturb.pop(name)] ^= 1
            body = ", ".join(f"0x{w:016x}ULL" for w in words)
            lines.append(f"static __device__ const uint64_t {name}[{len(words)}] = {{{body}}};")
        else:
            w = _word(kind, v)
            lines.append(f"#define {name} 0x{w:016x}ULL" if kind in ("d", "u64") else f"#define {name} 0x{w:08x}u")
    if perturb:
        raise KeyError(f"no such table: {sorted(perturb)}")
    return "\n".join(lines) + "\n"


# ============================================================================================ build and load

def build_dir() -> Path:
    d = os.environ.get("EXACT_BALLPARK_MODES_CUDA_DIR")
    if d:
        return Path(d)
    return cache_root() / "modes_cuda"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(out_dir: Path | None = None, variant: str = "native", source: Path = SOURCE,
          orient_source: Path = K.SOURCE, perturb: dict | None = None, checks: bool = True) -> dict:
    """Compile `source` (which includes `orient_source`) with both generated headers into out_dir (by default
    `build_dir()`); returns the manifest written beside the library: both sources' and both headers' SHA-256, nvcc's
    version, flags and targets, the kernels' registers and spills, the PTX check, the library's SHA-256 and the UTC
    time. A library already built from the same inputs is reused. `orient_source`, `perturb` and `checks=False` (the
    source scan and the PTX check recorded, not enforced) are for the tests' planted faults only."""
    if variant not in TARGETS:
        raise ValueError(f"unknown variant {variant!r}")
    out_dir = Path(out_dir) if out_dir is not None else build_dir()
    text, otext = read_source(source), read_source(orient_source)
    calls = K.scan_source(text) + [(n, f"{K.SOURCE.name}:{k}") for n, k in K.scan_source(otext)]
    if checks and calls:
        raise RuntimeError(f"the sources call CUDA's math library: {calls}")
    header, oheader = tables_header(perturb), K.tables_header()
    nvcc = K._nvcc()
    gencode = [f"-gencode=arch={a},code={c}" for a, c in TARGETS[variant]]
    key = _sha(json.dumps([_sha(text.encode()), _sha(otext.encode()), _sha(header.encode()), _sha(oheader.encode()),
                           FLAGS, gencode, str(nvcc), K.nvcc_version(nvcc), checks]).encode())[:16]
    stem = f"vec_modes_cuda_{variant}_{key}"
    lib, man = out_dir / f"{stem}.so", out_dir / f"{stem}.json"
    if lib.exists() and man.exists():
        m = json.loads(man.read_text())
        if m.get("library_sha256") == _sha(lib.read_bytes()) and (m.get("ptx_check", {}).get("ok") or not checks):
            return m
    work = out_dir / f".{stem}.{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        (work / HEADER).write_text(header)
        (work / K.HEADER).write_text(oheader)
        (work / K.SOURCE.name).write_text(otext)
        src = work / SOURCE.name
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
        check = K.ptx_check(ptx)
        if checks and not check["ok"]:
            raise RuntimeError(f"the PTX has unpinned floating-point instructions: {check['refused']}")
        resources = [ln.strip() for ln in (r.stdout + r.stderr).splitlines()
                     if "registers" in ln or "spill" in ln or "Compiling entry" in ln or "stack frame" in ln]
        version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True).stdout
        manifest = {
            "variant": variant, "source": str(source), "source_sha256": _sha(text.encode()),
            "orient_source": str(orient_source), "orient_source_sha256": _sha(otext.encode()),
            "header": HEADER, "header_sha256": _sha(header.encode()), "orient_header_sha256": _sha(oheader.encode()),
            "perturb": perturb or {}, "checks": checks, "source_scan": calls,
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


def load_library(variant: str = "native", source: Path = SOURCE, orient_source: Path = K.SOURCE,
                 perturb: dict | None = None, checks: bool = True) -> dict:
    """The compiled library (built if missing or stale), its functions typed, and its manifest; once a process."""
    k = (variant, str(source), str(orient_source), json.dumps(perturb or {}, sort_keys=True), checks)
    if k in _LIBS:
        return _LIBS[k]
    manifest = build(None, variant, source, orient_source, perturb, checks)
    lib = ctypes.CDLL(str(build_dir() / manifest["library"]))
    if lib.xbpm_abi() != ABI:
        raise RuntimeError(f"the library's ABI {lib.xbpm_abi()} is not {ABI}")
    p, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.xbpo_error_string.restype = ctypes.c_char_p
    lib.xbpo_error_string.argtypes = [i32]
    lib.xbpm_dlibm.argtypes = [i32, i64, p, p, p, i32]
    lib.xbpm_dlibm.restype = i32
    lib.xbpm_tick.argtypes = [ctypes.POINTER(State), p, i32]
    lib.xbpm_tick.restype = i32
    lib.xbpm_info.argtypes = [i32] + [ctypes.POINTER(i32)] * 5
    lib.xbpm_info.restype = i32
    entry = {"lib": lib, "manifest": manifest}
    _LIBS[k] = entry
    return entry


def _check(rc: int, lib, fn: str) -> None:
    if rc:
        msg = lib.xbpo_error_string(rc).decode() if rc > 0 else "refused (block size or arguments)"
        raise RuntimeError(f"{fn} failed ({rc}: {msg})")


# ======================================================================================================= launches

# The kernel's tensors, in vec_modes_cuda.cu's `struct Modes` order: (park attribute, dtype, columns; 0 for one)
STATE_TENSORS = (
    ("pos", F64, 3), ("vel", F64, 3), ("old_pos", F64, 3), ("old_vel", F64, 3), ("goto", F64, 3), ("last_g", F64, 3),
    ("mass", F64, 0), ("time_factor", F64, 0), ("last_collision", F64, 0), ("radius", F64, 0), ("max_vel", F64, 0),
    ("agility", F64, 0), ("speed_fraction", F64, 0), ("follow_range", F64, 0), ("max_ang_vel", F64, 0),
    ("rot_agility", F64, 0), ("rot_roll", F64, 0), ("rot", F64, 4), ("old_rot", F64, 4), ("ang_vel", F64, 3),
    ("old_ang_vel", F64, 3), ("torque", F64, 3), ("ids", torch.int64, 0), ("mode", torch.int64, 0),
    ("effect_stamp", torch.int64, 0), ("follow_id", torch.int64, 0), ("owner_id", torch.int64, 0),
    ("follow_ptr", torch.int64, 0), ("clock", torch.int64, 0), ("free", torch.bool, 0), ("moribund", torch.bool, 0),
    ("massive", torch.bool, 0), ("interactive", torch.bool, 0), ("sensor", torch.bool, 0), ("act", torch.uint8, 0),
    ("pos0", F64, 3), ("vel0", F64, 3), ("vy_after", F64, 0))


class State(ctypes.Structure):
    """vec_modes_cuda.cu's `struct Modes`, field for field: the tensors' addresses, the count and the park's scalars."""
    _fields_ = ([(name, ctypes.c_void_p) for name, _, _ in STATE_TENSORS]
                + [("n", ctypes.c_int64), ("friction", ctypes.c_double), ("dt", ctypes.c_double),
                   ("ffriction", ctypes.c_float), ("fdt", ctypes.c_float), ("new_orbit", ctypes.c_int),
                   ("dynamical", ctypes.c_int)])


def make_state(park) -> State:
    """The kernel's arguments for a VecModesPark: every tensor checked (dtype, shape, device), made contiguous and
    rebound to the park if it was not (a command may leave a strided one)."""
    if park.dev.type != "cuda":
        raise ValueError("the modes kernel needs a park on a CUDA device")
    s = State()
    for name, dtype, cols in STATE_TENSORS:
        t = getattr(park, name)
        shape = (park.n, cols) if cols else (park.n,)
        if t.dtype != dtype or tuple(t.shape) != shape or t.device != park.dev:
            raise ValueError(f"{name} must be a {dtype} tensor of shape {shape} on {park.dev}, not {t.dtype} "
                             f"{tuple(t.shape)} on {t.device}")
        if not t.is_contiguous():
            t = t.contiguous()
            setattr(park, name, t)
        setattr(s, name, t.data_ptr())
    s.n, s.friction, s.dt, s.ffriction, s.fdt = park.n, park.friction, park.dt, park.ffriction, park.fdt
    s.new_orbit, s.dynamical = int(park.use_new_orbit), int(park.dynamical_orientation)
    return s


def tick(park, block: int = 128, lib: dict | None = None) -> None:
    """One tick of a VecModesPark in place: the prepass and the main kernel, on torch's current stream of the park's
    device; no allocation, no host synchronisation."""
    e = lib or load_library()
    s = make_state(park)
    if not park.n:
        return
    with torch.cuda.device(park.dev):
        rc = e["lib"].xbpm_tick(ctypes.byref(s), torch.cuda.current_stream(park.dev).cuda_stream, int(block))
    _check(rc, e["lib"], "xbpm_tick")


def kernel_info(device=None, lib: dict | None = None) -> dict:
    """The main kernel and the prepass as loaded on a device: registers a thread, local memory, binary and PTX
    versions; and the library's manifest (targets, ptxas, the PTX check)."""
    e = lib or load_library()
    out = {}
    for which, name in ((0, "main"), (1, "prepass")):
        vals = [ctypes.c_int() for _ in range(5)]
        with torch.cuda.device(torch.device(device or "cuda")):
            rc = e["lib"].xbpm_info(which, *[ctypes.byref(v) for v in vals])
        _check(rc, e["lib"], "xbpm_info")
        regs, local, binary, ptx, threads = (v.value for v in vals)
        out[name] = {"registers": regs, "local_bytes": local, "binary_version": binary, "ptx_version": ptx,
                     "max_threads_per_block": threads}
    out.update(library=e["manifest"]["library"], library_sha256=e["manifest"]["library_sha256"],
               targets=e["manifest"]["targets"], ptxas=e["manifest"]["ptxas"])
    return out


def dlibm(name: str, x: torch.Tensor, block: int = 128, lib: dict | None = None) -> torch.Tensor:
    """The kernel's own double function `name` (sin, cos, log, exp, and sincos_sin and sincos_cos: sincos's sine and
    cosine) on a CUDA tensor, for the tests that hold it to exact_ballpark.dlibm, exact_ballpark.vec_exp
    and the C library."""
    e = lib or load_library()
    x = x.to(F64).contiguous()
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        rc = e["lib"].xbpm_dlibm(DLIBM_FUNCS[name], x.numel(), x.data_ptr(), out.data_ptr(),
                                 torch.cuda.current_stream(x.device).cuda_stream, int(block))
    _check(rc, e["lib"], "xbpm_dlibm")
    return out


# ================================================================================== a tick captured as a CUDA graph

class ModesGraph:
    """A VecModesPark's tick (the prepass and the main kernel) captured once as a CUDA graph on the park's own tensors,
    so each replay is one tick in one launch: the same kernels on the same buffers (exact by construction; held to
    eager ticks bit for bit by the tests). The kernels update the state in place, so the park's state tensors are the
    graph's fixed buffers; commands and setters rebind them, and `tick()` first copies any rebound one into its buffer
    (`sync()`). With `masked=True` the capture holds OnTick's per-world mask as a device buffer
    (vec_orient.GraphMask): the state saved, the tick, the rows outside the mask put back; `tick(active)` writes the
    mask and replays (every row when active is None)."""

    def __init__(self, park, block: int | None = None, lib: dict | None = None, masked: bool = False):
        self.park, self.block = park, park.block if block is None else int(block)
        self.lib = lib or load_library()
        self.buf = {}
        for name, _, _ in STATE_TENSORS:
            t = getattr(park, name).contiguous().clone()
            self.buf[name] = t
            setattr(park, name, t)
        self._warm_up()
        self.gmask = GraphMask(self.buf, park.n, park.dev) if masked else None
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            if self.gmask is not None:
                self.gmask.save()
            tick(park, self.block, self.lib)
            if self.gmask is not None:
                self.gmask.put_back()

    def _warm_up(self) -> None:
        """Both kernels launched once on a one-ball state of fresh zeros (the module loaded before the capture)."""
        class One:
            pass
        one = One()
        one.dev, one.n = self.park.dev, 1
        for name, dtype, cols in STATE_TENSORS:
            setattr(one, name, torch.zeros((1, cols) if cols else (1,), dtype=dtype, device=self.park.dev))
        one.follow_ptr -= 1
        for k in ("friction", "dt", "ffriction", "fdt", "use_new_orbit", "dynamical_orientation"):
            setattr(one, k, getattr(self.park, k))
        tick(one, self.block, self.lib)
        torch.cuda.synchronize(self.park.dev)

    def sync(self) -> None:
        """Copy every state tensor a command or setter rebound into the graph's buffer, and rebind it to the buffer."""
        for name, b in self.buf.items():
            t = getattr(self.park, name)
            if t is not b:
                b.copy_(t)
                setattr(self.park, name, b)

    def tick(self, active: torch.Tensor | None = None) -> None:
        if active is not None and self.gmask is None:
            raise ValueError("a graph made without OnTick's mask (masked=True) ticks every row")
        self.sync()
        if self.gmask is not None:
            self.gmask.set(active)
        self.graph.replay()
        self.park.current_time += 1
        if self.gmask is not None:
            self.gmask.mark(self.park)

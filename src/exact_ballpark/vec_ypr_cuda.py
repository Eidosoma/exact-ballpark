# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The C library's double sin, cos, asin and atan2 on a CUDA device, bit for bit glibc 2.34 aarch64's: the three
calls of Destiny's yaw, pitch and roll without dynamical orientation (Ball::CalculateYawPitchRoll, Ball.cpp:335;
MOV-729: atan2 of the heading, asin of its sine, asin of the sine of the yaw's change), so that
`exact_ballpark.vec_orient.VecOrientPark(..., ypr="cuda")` keeps them on the device instead of evaluating them on the host
with Python's math (two host round trips a tick), and a tick can be captured as a CUDA graph (`YprTickGraph`).

The device functions follow the scalar transcriptions line for line: sin and cos are exact_ballpark.dlibm's (the
device code copied from exact_ballpark.vec_modes_cuda), sincos is dlibm.sincos (glibc's sincos, which is
not always sin and cos), asin and atan2 are exact_ballpark.dlibm_inv's (IBM's Accurate Mathematical Library, read from
libm.so.6's disassembly; LGPL 2.1 inside glibc). Held bit for bit to those modules and
to the C library itself by tests/test_movement_vec_ypr_cuda.py.

The source is vec_ypr_cuda.cu beside this module; the library is built on first use (or when the source, the generated
header, the flags or nvcc change) into a build directory outside any source tree ($EXACT_BALLPARK_YPR_CUDA_DIR; by default $EXACT_BALLPARK_CACHE/ypr_cuda), with a manifest; it is loaded through ctypes and launched on torch's
current stream, with no host synchronisation, so a launch can be captured in a CUDA graph.

**Exactness by construction.** As exact_ballpark.vec_orient_cuda's kernel, whose build rules this module applies: nvcc
with -fmad=false, -prec-div=true, -prec-sqrt=true, -ftz=false and no fast math; the PTX read back and refused if it has
an unrounded, flushed, approximate or contracted floating-point instruction (`vec_orient_cuda.ptx_check`), and the
source refused if it calls any function of CUDA's math library (`vec_orient_cuda.scan_source`); every constant and
table generated into a header from dlibm's and dlibm_inv's values (`tables_header`), never typed.
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
from . import dlibm_inv as I
from . import vec_orient_cuda as K

F64 = torch.float64
from ._cuda_build import SourcePath, cache_root, default_nvcc, read_source  # noqa: E402,F401
SOURCE = SourcePath(Path(__file__).with_name("vec_ypr_cuda.cu"))
HEADER = "vec_ypr_cuda_tables.h"
FLAGS = K.FLAGS
TARGETS = K.TARGETS
ABI = 1
FUNCS = {"sin": 0, "cos": 1, "asin": 2, "atan2": 3, "asin_sin": 4, "sincos_sin": 5, "sincos_cos": 6}
_LIBS: dict = {}

# dlibm's constants the device sin and cos use (the names vec_modes_cuda's header gives them, prefixed YL_ here)
SIN_DOUBLES = ("TAYLOR_LIMIT", "BIG", "TOINT", "S1", "S2", "S3", "S4", "S5", "CS2", "CS4", "CS6", "SN3", "SN5", "HPINV",
               "MP1", "MP2", "PP3", "PP4", "HP0", "HP1", "BR_TM600", "BR_SPLIT", "BR_TM24", "BR_BIG", "BR_BIG1",
               "BR_TOVERP0", "BR_HP0", "BR_HP1", "BR_MP1", "BR_MP2")
SIN_WORDS = ("K_TINY_SIN", "K_TINY_COS", "K_DO_SIN", "K_MID", "K_REDUCE", "K_HUGE", "BR_T576_HI")
SIN_INTS = ("BR_EXP_LIMIT", "BR_EXP_BIAS")


# ============================================================================================ the generated header

def _dbits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def table_values() -> dict:
    """Every constant and table the device code uses, by its name in the header: (kind, value or values), read from
    exact_ballpark.dlibm and exact_ballpark.dlibm_inv (glibc's tables and constants, read out of the library), never
    typed. Kinds: d (a double), u32 (an unsigned word), int (a decimal #define); tables are tuples."""
    out = {}
    for name in SIN_DOUBLES:
        out[f"YL_{name}"] = ("d", getattr(D, name))
    for name in SIN_WORDS:
        out[f"YL_{name}"] = ("u32", getattr(D, name))
    for name in SIN_INTS:
        out[f"YL_{name}"] = ("int", getattr(D, name))
    out["YL_SINCOSTAB"] = ("d", tuple(D.SINCOSTAB))
    out["YL_BR_TOVERP"] = ("d", tuple(D.BR_TOVERP))
    for name in ("K_TINY", "K_TAYLOR", "K_QUARTER", "K_HALF", "K_075", "K_0921875", "K_0953125", "K_096875",
                 "K_NEAR_ONE", "K_ONE", "K_INF"):
        out[f"YI_ASIN_{name}"] = ("u32", getattr(I, f"ASIN_{name}"))
    for name in ("F1", "F2", "F3", "F4", "F5", "F6", "RT0", "RT1", "RT2", "RT3", "HP0", "HP1", "NEG_HP0", "ONE", "HALF",
                 "T24", "THREE_HALVES", "TWO"):
        out[f"YI_ASIN_{name}"] = ("d", getattr(I, f"ASIN_{name}"))
    for name in ("POWTWO_TOP", "N_Q1_MUL", "N_Q2_OFF", "N_H_OFF", "N_A_MUL", "N_A_OFF", "N_B_OFF", "N_C_OFF"):
        out[f"YI_ASIN_{name}"] = ("int", getattr(I, f"ASIN_{name}"))
    for name in ("INROOT", "POWTWO", "ASNCS"):
        out[f"YI_ASIN_{name}"] = ("d", tuple(getattr(I, f"ASIN_{name}")))
    for name in ("HPI", "MHPI", "OPI", "MOPI", "QPI", "MQPI", "TQPI", "MTQPI", "OPI1", "HPI1", "D3", "D5", "D7", "D9",
                 "D11", "D13", "TWOM500", "TWO500", "INV16", "TWO8", "TWO52"):
        out[f"YI_ATAN2_{name}"] = ("d", getattr(I, f"ATAN2_{name}"))
    for name in ("EP", "EM", "I_OFF"):
        out[f"YI_ATAN2_{name}"] = ("int", getattr(I, f"ATAN2_{name}"))
    out["YI_ATAN2_CIJ"] = ("d", tuple(I.ATAN2_CIJ))
    return out


def _word(kind: str, v) -> int:
    return _dbits(v) if kind == "d" else int(v)


def tables_header(perturb: dict | None = None) -> str:
    """The C header of `table_values()`: scalars as #defines of their bit patterns (integers in parentheses), tables as
    device arrays of bit patterns. `perturb` ({table name: index}) flips the lowest bit of those entries: the planted
    fault of a wrong table entry, for the tests only."""
    perturb = dict(perturb or {})
    lines = ["// Generated by exact_ballpark.vec_ypr_cuda.tables_header() from exact_ballpark.dlibm and",
             "// exact_ballpark.dlibm_inv (glibc 2.34 aarch64's tables and constants, read from the library);",
             "// do not edit.",
             "#pragma once", "#include <cstdint>", ""]
    for name, (kind, v) in table_values().items():
        if kind == "int":
            lines.append(f"#define {name} ({int(v)})")
        elif isinstance(v, tuple):
            words = [_word(kind, x) for x in v]
            if name in perturb:
                words[perturb.pop(name)] ^= 1
            body = ", ".join(f"0x{w:016x}ULL" for w in words)
            lines.append(f"static __device__ const uint64_t {name}[{len(words)}] = {{{body}}};")
        else:
            w = _word(kind, v)
            lines.append(f"#define {name} 0x{w:016x}ULL" if kind == "d" else f"#define {name} 0x{w:08x}u")
    if perturb:
        raise KeyError(f"no such table: {sorted(perturb)}")
    return "\n".join(lines) + "\n"


# ============================================================================================ build and load

def build_dir() -> Path:
    d = os.environ.get("EXACT_BALLPARK_YPR_CUDA_DIR")
    if d:
        return Path(d)
    return cache_root() / "ypr_cuda"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(out_dir: Path | None = None, variant: str = "native", source: Path = SOURCE, perturb: dict | None = None,
          checks: bool = True) -> dict:
    """Compile `source` with its generated header into out_dir (by default `build_dir()`); returns the manifest written
    beside the library (the source's and the header's SHA-256, nvcc's version, flags and targets, ptxas's registers and
    spills, the PTX check, the library's SHA-256, the UTC time). A library already built from the same inputs is
    reused. `perturb` and `checks=False` (the source scan and the PTX check recorded, not enforced) are for the tests'
    planted faults only."""
    if variant not in TARGETS:
        raise ValueError(f"unknown variant {variant!r}")
    out_dir = Path(out_dir) if out_dir is not None else build_dir()
    text = read_source(source)
    calls = K.scan_source(text)
    if checks and calls:
        raise RuntimeError(f"the source calls CUDA's math library: {calls}")
    header = tables_header(perturb)
    nvcc = K._nvcc()
    gencode = [f"-gencode=arch={a},code={c}" for a, c in TARGETS[variant]]
    key = _sha(json.dumps([_sha(text.encode()), _sha(header.encode()), FLAGS, gencode, str(nvcc),
                           K.nvcc_version(nvcc), checks]).encode())[:16]
    stem = f"vec_ypr_cuda_{variant}_{key}"
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
        check = K.ptx_check(ptx)
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
    if lib.xbpy_abi() != ABI:
        raise RuntimeError(f"the library's ABI {lib.xbpy_abi()} is not {ABI}")
    p, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.xbpy_error_string.restype = ctypes.c_char_p
    lib.xbpy_error_string.argtypes = [i32]
    lib.xbpy_dlibm.argtypes = [i32, i64, p, p, p, p, i32]
    lib.xbpy_dlibm.restype = i32
    lib.xbpy_info.argtypes = [ctypes.POINTER(i32)] * 5
    lib.xbpy_info.restype = i32
    entry = {"lib": lib, "manifest": manifest}
    _LIBS[k] = entry
    return entry


def _check(rc: int, lib, fn: str) -> None:
    if rc:
        msg = lib.xbpy_error_string(rc).decode() if rc > 0 else "refused (block size or arguments)"
        raise RuntimeError(f"{fn} failed ({rc}: {msg})")


# ======================================================================================================= launches

def dlibm(name: str, a: torch.Tensor, b: torch.Tensor | None = None, block: int = 128,
          lib: dict | None = None) -> torch.Tensor:
    """The device function `name` (sin, cos, asin, asin_sin = asin(sin(a)), sincos_sin and sincos_cos = sincos's two
    values, or atan2(a, b) with a = y and b = x) on a CUDA float64 tensor, elementwise, on torch's current stream: a new
    tensor of a's shape; no host synchronisation."""
    if name not in FUNCS:
        raise ValueError(f"unknown function {name!r}")
    if a.dtype != F64 or a.device.type != "cuda":
        raise ValueError(f"a must be a float64 CUDA tensor, not {a.dtype} on {a.device}")
    if (b is None) != (name != "atan2"):
        raise ValueError("atan2 takes a and b; the others take a alone")
    e = lib or load_library()
    a = a.contiguous()
    if b is not None:
        if b.dtype != F64 or b.device != a.device or b.shape != a.shape:
            raise ValueError("b must be a float64 tensor of a's shape on a's device")
        b = b.contiguous()
    out = torch.empty_like(a)
    with torch.cuda.device(a.device):
        rc = e["lib"].xbpy_dlibm(FUNCS[name], a.numel(), a.data_ptr(), (a if b is None else b).data_ptr(),
                                 out.data_ptr(), torch.cuda.current_stream(a.device).cuda_stream, int(block))
    _check(rc, e["lib"], "xbpy_dlibm")
    return out


def sin(x: torch.Tensor, **kw) -> torch.Tensor:
    """glibc's sin on the device (exact_ballpark.dlibm.sin's bits)."""
    return dlibm("sin", x, **kw)


def cos(x: torch.Tensor, **kw) -> torch.Tensor:
    """glibc's cos on the device (exact_ballpark.dlibm.cos's bits)."""
    return dlibm("cos", x, **kw)


def asin(x: torch.Tensor, **kw) -> torch.Tensor:
    """glibc's asin on the device (exact_ballpark.dlibm_inv.asin's bits)."""
    return dlibm("asin", x, **kw)


def atan2(y: torch.Tensor, x: torch.Tensor, **kw) -> torch.Tensor:
    """glibc's atan2(y, x) on the device (exact_ballpark.dlibm_inv.atan2's bits)."""
    return dlibm("atan2", y, x, **kw)


def sincos(x: torch.Tensor, **kw) -> tuple:
    """glibc's sincos on the device (exact_ballpark.dlibm.sincos's bits): the sine and the cosine, two launches."""
    return dlibm("sincos_sin", x, **kw), dlibm("sincos_cos", x, **kw)


def asin_sin(x: torch.Tensor, **kw) -> torch.Tensor:
    """asin(sin(x)) in one launch, both glibc's (the roll's yaw delta: Ball.cpp's asin(sin(yaw - oldYaw)))."""
    return dlibm("asin_sin", x, **kw)


def kernel_info(device=None, lib: dict | None = None) -> dict:
    """The kernel as loaded on a device: registers a thread, local memory, binary and PTX versions; and the library's
    manifest (targets, ptxas, the PTX check)."""
    e = lib or load_library()
    vals = [ctypes.c_int() for _ in range(5)]
    with torch.cuda.device(torch.device(device or "cuda")):
        rc = e["lib"].xbpy_info(*[ctypes.byref(v) for v in vals])
    _check(rc, e["lib"], "xbpy_info")
    regs, local, binary, ptx, threads = (v.value for v in vals)
    return {"registers": regs, "local_bytes": local, "binary_version": binary, "ptx_version": ptx,
            "max_threads_per_block": threads, "library": e["manifest"]["library"],
            "library_sha256": e["manifest"]["library_sha256"], "targets": e["manifest"]["targets"],
            "ptxas": e["manifest"]["ptxas"]}


# ================================================================================== a tick captured as a CUDA graph

class YprTickGraph:
    """A VecOrientPark's tick without dynamical orientation (`evolve` with ypr="cuda") captured once as a CUDA graph,
    its new state copied back into its inputs inside the graph, so each replay is one tick: the same kernels on the same
    buffers (exact by construction; held to eager ticks bit for bit by the tests), with no host round trip. The park's
    state tensors become the graph's fixed buffers; commands and setters rebind them, and `tick()` first copies any
    rebound one into its buffer (`sync()`). As vec_orient_cuda.TickGraph does for dynamical orientation."""

    READ = ("pos", "vel", "goto", "mode", "mass", "agility", "speed_fraction", "max_vel", "time_factor", "yaw", "pitch",
            "roll", "old_yaw", "old_pitch", "old_roll", "roll_speed", "old_roll_speed", "yaw_delta")
    WRITE = ("pos", "vel", "old_pos", "old_vel", "last_g", "yaw", "pitch", "old_yaw", "old_pitch", "roll", "old_roll",
             "roll_speed", "old_roll_speed", "yaw_delta")

    def __init__(self, park, block: int = 128):
        if park.dynamical_orientation or getattr(park, "ypr", "host") != "cuda":
            raise ValueError('YprTickGraph captures a park without dynamical orientation and with ypr="cuda"')
        self.park, self.block = park, block
        self.buf = {f: getattr(park, f).clone() for f in dict.fromkeys(self.READ + self.WRITE)}
        for f, t in self.buf.items():
            setattr(park, f, t)
        z = torch.zeros(4, dtype=F64, device=park.dev)               # warm-up: the library loaded before capture
        for name in ("asin", "asin_sin"):
            dlibm(name, z, block=block)
        dlibm("atan2", z, z, block=block)
        torch.cuda.synchronize(park.dev)
        self.graph = torch.cuda.CUDAGraph()
        t0 = park.current_time
        with torch.cuda.graph(self.graph):
            park.evolve()
            outs = {f: getattr(park, f) for f in self.WRITE}
            # evolve's old_* values are its input tensors themselves: copied before those are overwritten
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

    def tick(self) -> None:
        self.sync()
        self.graph.replay()
        self.park.current_time += 1

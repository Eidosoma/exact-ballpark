# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's combat pieces on the GPU: a VecCombatPark's tick as four CUDA launches
(the prepass, phase 1, the missiles' contacts with their targets, phases 3 and 4 with the dispatch's counts) and
GetAccuracy for B pairs, one thread a ball or a pair: the kernels of vec_combat_cuda.cu beside this module, which
includes vec_modes_cuda.cu (the flight, and through it vec_orient_cuda.cu, the turning and the
C library's float functions and pow) unedited. Checked bit for bit against the scalar reference by
tests/test_movement_vec_combat.py and tests/test_movement_vec_accuracy.py.

(1 November 2026): a fifth launch after the host's yaw, pitch and roll (`after`): the release of a
removed mushroom's members whose turns came first, and PetrifyTroll; with it the commands (SetBallFree,
SetBallRigid, SetBoid, SetBallTroll) and mushrooms (AddMushroom's ball grows by glibc's powf, `g_powf`, transcribed from
exact_ballpark.dlibm_powf, and is removed at its turn when grown) in VecCombatPark's tick; `powf` runs the device's powf
elementwise (its tests). Checked by tests/test_movement_vec_combat_leftovers.py and tests/test_movement_dlibm_powf.py.

The library is built on first use (or when a source, a generated header, the flags or nvcc change) into a build
directory outside any source tree ($EXACT_BALLPARK_COMBAT_CUDA_DIR; by default $EXACT_BALLPARK_CACHE/combat_cuda),
with a manifest; it is loaded through ctypes and launched on torch's current stream, with no allocation and no host
synchronisation, so a tick can be captured in a CUDA graph (`CombatGraph`).

**Exactness by construction**, as vec_modes_cuda: the three sources are scanned for CUDA math-library calls and the
PTX is checked (vec_orient_cuda.scan_source, ptx_check); nvcc with -fmad=false, -prec-div=true, -prec-sqrt=true,
-ftz=false and no fast math; every constant generated into a header from the reference's modules (`tables_header`):
Destiny's named constants from exact_ballpark.full and exact_ballpark.modes, the literals the reference writes without
a name, here once with the reference's own spelling (`COMBAT_LITERALS`; a test finds each in the reference's source),
and powf's tables and constants from exact_ballpark.dlibm_powf (read from the C library by its generator).
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

from . import dlibm_powf as DP
from . import full as FU
from . import modes as MO
from . import vec_modes_cuda as MK
from . import vec_orient_cuda as K
from .vec_orient import GraphMask

F64, I64 = torch.float64, torch.int64
from ._cuda_build import SourcePath, cache_root, default_nvcc, read_source  # noqa: E402,F401
SOURCE = SourcePath(Path(__file__).with_name("vec_combat_cuda.cu"))
HEADER = "vec_combat_cuda_tables.h"
FLAGS = K.FLAGS
TARGETS = K.TARGETS
ABI = 2                                            # 2: the two records in `struct Combat`, the fifth launch
_LIBS: dict = {}
# The batched park's domain flags, a bit each (exact_ballpark.vec_combat; the kernels set FLAG_UNDEFINED)
FLAG_DEFENDER, FLAG_UNDEFINED, FLAG_BUBBLES, FLAG_DEAD, FLAG_POOL, FLAG_MASSIVE = 1, 2, 4, 8, 16, 32

# The reference's literals that have no name there, with the reference's own spelling (space.py's Potential, full.py's
# SnapOrientation and RotationQuaternion(yaw, pitch, roll)); each is found in the source by a test. "f" is a float.
COMBAT_LITERALS = {
    "C_FIXED_MASS": ("d", 1.0e34, "space.py", "1.0e34"),           # Potential: a fixed ball's mass
    "C_SEPARATING_SPEED2": ("d", 0.0001, "space.py", "0.0001"),    # Potential: |v1 - v2|^2 > 0.0001 returns early
    "C_SNAP_MIN_SPEED2": ("d", 1e-12, "full.py", "1e-12"),         # SnapOrientation: |v|^2 > 1e-12
    "C_SNAP_MIN_LENGTH": ("d", 1e-12, "full.py", "len_dir < 1e-12"),   # ... and a float length below 1e-12
    "F_UP_LIMIT": ("f", 0.99, "full.py", "_f(0.99)"),              # ... the roll only while up.z < 0.99f
    "F_HALF": ("f", 0.5, "full.py", "pitch * 0.5"),                # RotationQuaternion(yaw, pitch, roll)'s halves
    "F_MILLI": ("f", 0.001, "full.py", "_f(0.001)"),               # a mushroom's time fraction: ms to s (0.001f)
    "F_QUARTER": ("f", 0.25, "full.py", "_powf(fraction, 0.25)"),  # its growth: range x powf(fraction, 0.25f)
}

# glibc's powf (exact_ballpark.dlibm_powf, generated from the C library): its scalars and tables, by their names there
POWF_SCALARS = ("POWF_X_MIN", "POWF_X_SPAN", "POWF_ZINF", "POWF_NEG_OFF", "POWF_TOP_MASK", "POWF_I_SHIFT", "POWF_I_BITS",
                "POWF_K_SHIFT", "POWF_BIG", "POWF_UFLOW", "POWF_MAY_UFLOW", "POWF_TINY_F", "POWF_EXP2_SHIFT",
                "POWF_INT_E", "POWF_SIGN_BIAS", "POWF_SUBNORMAL_MAX", "POWF_TWO23_F", "POWF_NORM_SUB", "POWF_ONE_BITS",
                "POWF_INF2", "POWF_ONE2", "POWF_ONE2_M1", "POWF_E_HALF", "POWF_E_ONE", "POWF_E_SPAN", "POWF_QNAN_BIT",
                "POWF_QNAN", "POWF_UFLOWF_X", "POWF_MAY_UFLOWF_X", "POWF_OFLOWF_X")
POWF_DOUBLES = ("POWF_OFLOW", "POWF_RND")
POWF_TABLES = ("POWF_LOG2_INVC", "POWF_LOG2_LOGC", "POWF_A", "POWF_EXP2_C")


def _dbits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _fbits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def table_values() -> dict:
    """Every constant vec_combat_cuda.cu uses beyond vec_modes_cuda's, by its name in the header: (kind, value), read
    from exact_ballpark.full and exact_ballpark.modes, or COMBAT_LITERALS; never typed."""
    out = {"M_MISSILE_STRAIGHT_MS": ("int", FU.MISSILE_STRAIGHT_MS),
           "C_DISPATCH_MIN_MOVE_SQ": ("d", FU.DISPATCH_MIN_MOVE_SQ),
           "FORMATION_SLOT_BITS": ("int", FU.FORMATION_MAX_SLOTS), "FLAG_UNDEFINED": ("int", FLAG_UNDEFINED)}
    for name in ("MISSILE", "FORMATION", "MUSHROOM", "MINIBALL"):
        out[f"MODE_{name}"] = ("int", getattr(MO, name))
    for name, (kind, v, _, _) in COMBAT_LITERALS.items():
        out[name] = (kind, v)
    for name in POWF_SCALARS:
        out[name] = ("int", getattr(DP, name))
    for name in POWF_DOUBLES:
        out[name] = ("d", getattr(DP, name))
    for name in POWF_TABLES:
        out["T_" + name] = ("d", tuple(getattr(DP, name)))
    out["T_POWF_EXP2_TAB"] = ("u64", tuple(DP.POWF_EXP2_TAB))
    out["POWF_EXP2_N"] = ("int", len(DP.POWF_EXP2_TAB))
    return out


def tables_header() -> str:
    lines = ["// Generated by exact_ballpark.vec_combat_cuda.tables_header() from exact_ballpark.full,",
             "// exact_ballpark.modes, the reference's own literals (COMBAT_LITERALS) and exact_ballpark.dlibm_powf; do not",
             "// edit.", "#pragma once", "#include <cstdint>", ""]
    for name, (kind, v) in table_values().items():
        if isinstance(v, tuple):                    # a device table of bit patterns (doubles or 64-bit words)
            body = ", ".join(f"0x{(_dbits(x) if kind == 'd' else int(x)):016x}ULL" for x in v)
            lines.append(f"static __device__ const uint64_t {name}[{len(v)}] = {{{body}}};")
        elif kind == "int" and int(v) >= 2 ** 32:      # a 64-bit pattern
            lines.append(f"#define {name} 0x{int(v):016x}ULL")
        elif kind == "int" and int(v) >= 2 ** 31:      # a 32-bit mask
            lines.append(f"#define {name} 0x{int(v):08x}u")
        elif kind == "int":
            lines.append(f"#define {name} {int(v)}")
        elif kind == "f":
            lines.append(f"#define {name} 0x{_fbits(v):08x}u")
        else:
            lines.append(f"#define {name} 0x{_dbits(v):016x}ULL")
    return "\n".join(lines) + "\n"


# ============================================================================================ build and load

def build_dir() -> Path:
    d = os.environ.get("EXACT_BALLPARK_COMBAT_CUDA_DIR")
    if d:
        return Path(d)
    return cache_root() / "combat_cuda"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(out_dir: Path | None = None, variant: str = "native", source: Path = SOURCE, checks: bool = True) -> dict:
    """Compile `source` (which includes vec_modes_cuda.cu and, through it, vec_orient_cuda.cu) with the three generated
    headers into out_dir (by default `build_dir()`); returns the manifest written beside the library: the sources' and
    headers' SHA-256, nvcc's version, flags and targets, the kernels' registers and spills, the PTX check, the library's
    SHA-256 and the UTC time. A library already built from the same inputs is reused. `checks=False` (the source scan
    and the PTX check recorded, not enforced) is for the tests' planted faults only."""
    if variant not in TARGETS:
        raise ValueError(f"unknown variant {variant!r}")
    out_dir = Path(out_dir) if out_dir is not None else build_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = {Path(source).name: read_source(source), MK.SOURCE.name: MK.SOURCE.read_text(),
             K.SOURCE.name: K.SOURCE.read_text()}
    calls = [(n, f"{f}:{k}") for f, t in texts.items() for n, k in K.scan_source(t)]
    if checks and calls:
        raise RuntimeError(f"the sources call CUDA's math library: {calls}")
    headers = {HEADER: tables_header(), MK.HEADER: MK.tables_header(), K.HEADER: K.tables_header()}
    nvcc = K._nvcc()
    gencode = [f"-gencode=arch={a},code={c}" for a, c in TARGETS[variant]]
    key = _sha(json.dumps([[_sha(t.encode()) for t in texts.values()], [_sha(h.encode()) for h in headers.values()],
                           FLAGS, gencode, str(nvcc), K.nvcc_version(nvcc), checks]).encode())[:16]
    stem = f"vec_combat_cuda_{variant}_{key}"
    lib, man = out_dir / f"{stem}.so", out_dir / f"{stem}.json"
    if lib.exists() and man.exists():
        m = json.loads(man.read_text())
        if m.get("library_sha256") == _sha(lib.read_bytes()) and (m.get("ptx_check", {}).get("ok") or not checks):
            return m
    work = out_dir / f".{stem}.{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        for name, text in {**texts, **headers}.items():
            (work / name).write_text(text)
        src = work / Path(source).name
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
            "variant": variant, "source": str(source),
            "sources_sha256": {n: _sha(t.encode()) for n, t in texts.items()},
            "headers_sha256": {n: _sha(h.encode()) for n, h in headers.items()},
            "checks": checks, "source_scan": calls, "nvcc": version.strip().splitlines()[-1], "nvcc_path": str(nvcc),
            "flags": FLAGS, "targets": [f"{a}->{c}" for a, c in TARGETS[variant]], "embedded": listing,
            "ptxas": resources, "ptx_check": check, "library": lib.name,
            "library_sha256": _sha((work / "lib.so").read_bytes()),
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


def load_library(variant: str = "native", source: Path = SOURCE, checks: bool = True) -> dict:
    """The compiled library (built if missing or stale), its functions typed, and its manifest; once a process."""
    k = (variant, str(source), checks)
    if k in _LIBS:
        return _LIBS[k]
    manifest = build(None, variant, source, checks)
    lib = ctypes.CDLL(str(build_dir() / manifest["library"]))
    if lib.xbpc_abi() != ABI:
        raise RuntimeError(f"the library's ABI {lib.xbpc_abi()} is not {ABI}")
    p, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.xbpo_error_string.restype = ctypes.c_char_p
    lib.xbpo_error_string.argtypes = [i32]
    lib.xbpc_tick.argtypes = [ctypes.POINTER(State), p, i32]
    lib.xbpc_tick.restype = i32
    lib.xbpc_after.argtypes = [ctypes.POINTER(State), p, i32]
    lib.xbpc_after.restype = i32
    lib.xbpc_powf.argtypes = [i64, p, p, p, p, i32]
    lib.xbpc_powf.restype = i32
    lib.xbpc_accuracy.argtypes = [i64] + [p] * 12 + [p, i32]
    lib.xbpc_accuracy.restype = i32
    lib.xbpc_info.argtypes = [i32] + [ctypes.POINTER(i32)] * 5
    lib.xbpc_info.restype = i32
    entry = {"lib": lib, "manifest": manifest}
    _LIBS[k] = entry
    return entry


def _check(rc: int, lib, fn: str) -> None:
    if rc:
        msg = lib.xbpo_error_string(rc).decode() if rc > 0 else "refused (block size or arguments)"
        raise RuntimeError(f"{fn} failed ({rc}: {msg})")


# ======================================================================================================= launches

# vec_combat_cuda.cu's `struct Combat` after its Modes: (park attribute, dtype, columns; 0 for one)
COMBAT_TENSORS = (
    ("exists", torch.bool, 0), ("formation_id", I64, 0), ("owner_ptr", I64, 0), ("hits_n", I64, 0), ("ev_n", I64, 0),
    ("ev_self", torch.bool, 0), ("bubble", I64, 0), ("cloaked", I64, 0), ("yaw", F64, 0), ("pitch", F64, 0),
    ("roll", F64, 0), ("formation_slots", I64, 0), ("flags", I64, 0),
    ("dying", torch.uint8, 0), ("petrified", torch.uint8, 0))   # the records of a tick


class State(ctypes.Structure):
    """vec_combat_cuda.cu's `struct Combat`, field for field."""
    _fields_ = ([("m", MK.State)] + [(name, ctypes.c_void_p) for name, _, _ in COMBAT_TENSORS]
                + [("formations", ctypes.c_void_p), ("formation_len", ctypes.c_void_p),
                   ("n_formations", ctypes.c_int64), ("max_slots", ctypes.c_int64), ("tick_ms", ctypes.c_int64),
                   ("do_missiles_off", ctypes.c_int), ("master", ctypes.c_int)])


def make_state(park) -> State:
    """The kernels' arguments for a VecCombatPark: every tensor checked (dtype, shape, device), made contiguous and
    rebound to the park if it was not."""
    s = State()
    s.m = MK.make_state(park)
    for name, dtype, cols in COMBAT_TENSORS:
        t = getattr(park, name)
        shape = (park.n, cols) if cols else (park.n,)
        if t.dtype != dtype or tuple(t.shape) != shape or t.device != park.dev:
            raise ValueError(f"{name} must be a {dtype} tensor of shape {shape} on {park.dev}, not {t.dtype} "
                             f"{tuple(t.shape)} on {t.device}")
        if not t.is_contiguous():
            t = t.contiguous()
            setattr(park, name, t)
        setattr(s, name, t.data_ptr())
    s.formations, s.formation_len = park.formation_table.data_ptr(), park.formation_lengths.data_ptr()
    s.n_formations, s.max_slots = park.n_formations, park.max_slots
    s.tick_ms, s.do_missiles_off, s.master = park.tick_ms, int(park.disable_do_for_missiles), int(park.master)
    return s


def tick(park, block: int = 128, lib: dict | None = None) -> None:
    """One tick of a VecCombatPark in place: the four launches, on torch's current stream of the park's device; no
    allocation, no host synchronisation. `after` follows (after the host's yaw, pitch and roll without dynamical
    orientation: VecCombatPark.evolve)."""
    e = lib or load_library()
    s = make_state(park)
    if not park.n:
        return
    with torch.cuda.device(park.dev):
        rc = e["lib"].xbpc_tick(ctypes.byref(s), torch.cuda.current_stream(park.dev).cuda_stream, int(block))
    _check(rc, e["lib"], "xbpc_tick")


def after(park, block: int = 128, lib: dict | None = None) -> None:
    """The fifth launch, after phase 4 and the host's yaw, pitch and roll: the members of a mushroom
    removed this tick whose turns came first are released, then each troll due is petrified (its row marked in
    `petrified`, whose yaw, pitch and roll the host then snaps without dynamical orientation); no host sync."""
    e = lib or load_library()
    s = make_state(park)
    if not park.n:
        return
    with torch.cuda.device(park.dev):
        rc = e["lib"].xbpc_after(ctypes.byref(s), torch.cuda.current_stream(park.dev).cuda_stream, int(block))
    _check(rc, e["lib"], "xbpc_after")


def powf(x: torch.Tensor, y: torch.Tensor, block: int = 128, lib: dict | None = None) -> torch.Tensor:
    """glibc's powf on the device, elementwise (`g_powf`, the combat kernel's transcription of exact_ballpark.dlibm_powf):
    x and y float32 tensors of one shape on a CUDA device; returns float32. For the tests."""
    e = lib or load_library()
    x, y = x.to(torch.float32).contiguous(), y.to(torch.float32).contiguous()
    if x.shape != y.shape or x.device.type != "cuda":
        raise ValueError("x and y: float32 tensors of one shape on a CUDA device")
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        with torch.cuda.device(x.device):
            rc = e["lib"].xbpc_powf(n, x.data_ptr(), y.data_ptr(), out.data_ptr(),
                                    torch.cuda.current_stream(x.device).cuda_stream, int(block))
        _check(rc, e["lib"], "xbpc_powf")
    return out


def accuracy(pos_s, vel_s, rad_s, pos_d, vel_d, rad_d, args, same, present, block: int = 128,
             lib: dict | None = None) -> tuple:
    """GetAccuracy for B pairs on the device (one thread a pair): args [B, 5] (optimal, falloff, tracking, signature,
    optimal signature), same and present [B] masks; returns (2^E, the surface distance, the angular speed) as
    exact_ballpark.vec_accuracy.get_accuracy does."""
    e = lib or load_library()
    dev = pos_s.device
    c = lambda t, dt=F64: t.to(dtype=dt).contiguous()
    ins = [c(pos_s), c(vel_s), c(rad_s), c(pos_d), c(vel_d), c(rad_d), c(args), c(same, torch.uint8),
           c(present, torch.uint8)]
    n = ins[0].shape[0]
    out = [torch.empty(n, dtype=F64, device=dev) for _ in range(3)]
    with torch.cuda.device(dev):
        rc = e["lib"].xbpc_accuracy(n, *[t.data_ptr() for t in ins], *[t.data_ptr() for t in out],
                                    torch.cuda.current_stream(dev).cuda_stream, int(block))
    _check(rc, e["lib"], "xbpc_accuracy")
    return tuple(out)


def kernel_info(device=None, lib: dict | None = None) -> dict:
    """The seven kernels as loaded on a device: registers a thread, local memory, binary and PTX versions; and the
    library's manifest (targets, ptxas, the PTX check)."""
    e = lib or load_library()
    out = {}
    for which, name in enumerate(("prepass", "phase1", "contact", "commit", "accuracy", "after", "powf")):
        vals = [ctypes.c_int() for _ in range(5)]
        with torch.cuda.device(torch.device(device or "cuda")):
            rc = e["lib"].xbpc_info(which, *[ctypes.byref(v) for v in vals])
        _check(rc, e["lib"], "xbpc_info")
        regs, local, binary, ptx, threads = (v.value for v in vals)
        out[name] = {"registers": regs, "local_bytes": local, "binary_version": binary, "ptx_version": ptx,
                     "max_threads_per_block": threads}
    out.update(library=e["manifest"]["library"], library_sha256=e["manifest"]["library_sha256"],
               targets=e["manifest"]["targets"], ptxas=e["manifest"]["ptxas"], ptx_check=e["manifest"]["ptx_check"])
    return out


# ================================================================================== a tick captured as a CUDA graph

class CombatGraph:
    """A VecCombatPark's tick (its four launches and, the fifth: `after`) captured once as a CUDA graph
    on the park's own tensors, so each replay is one tick in one launch: the same kernels on the same buffers (held to
    eager ticks bit for bit by the tests). As vec_modes_cuda.ModesGraph: commands and setters rebind tensors, and
    `tick()` first copies any rebound one into its buffer. Only with dynamical orientation (without it the yaw, pitch
    and roll are computed on the host after each tick, before the fifth launch: `VecCombatPark.after_tick`). With
    `masked=True` the capture holds OnTick's per-world mask as a device buffer (vec_orient.GraphMask,
    as vec_modes_cuda.ModesGraph's): `tick(active)`."""

    def __init__(self, park, block: int | None = None, lib: dict | None = None, masked: bool = False):
        if not park.dynamical_orientation:
            raise ValueError("without dynamical orientation a tick needs the host (the yaw, pitch and roll)")
        self.park, self.block = park, park.block if block is None else int(block)
        self.lib = lib or load_library()
        self.names = [n for n, _, _ in MK.STATE_TENSORS] + [n for n, _, _ in COMBAT_TENSORS]
        self.buf = {}
        for name in self.names:
            t = getattr(park, name).contiguous().clone()
            self.buf[name] = t
            setattr(park, name, t)
        saved = {n: b.clone() for n, b in self.buf.items()}
        tick(park, self.block, self.lib)                # the warm-up runs one real tick on the buffers ...
        after(park, self.block, self.lib)
        torch.cuda.synchronize(park.dev)
        for n, b in self.buf.items():                   # ... and the state is put back as it was
            b.copy_(saved[n])
        torch.cuda.synchronize(park.dev)
        self.gmask = GraphMask(self.buf, park.n, park.dev) if masked else None
        self._capture()

    def _capture(self) -> None:
        """The tick captured on the buffers, with the formation table's addresses as they are now."""
        park = self.park
        self._tables = (park.formation_table, park.formation_lengths)   # addresses captured in the graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):              # captured, not run
            if self.gmask is not None:
                self.gmask.save()
            tick(park, self.block, self.lib)
            after(park, self.block, self.lib)
            if self.gmask is not None:
                self.gmask.put_back()

    def sync(self) -> None:
        for name, b in self.buf.items():
            t = getattr(self.park, name)
            if t is not b:
                b.copy_(t)
                setattr(self.park, name, b)

    def tick(self, active: torch.Tensor | None = None) -> None:
        """One replay; after a LoadFormations (new table tensors) the tick is captured anew first (PyLoadFormations mid-run, Thunkers.cpp:1921, replaces the park's table). `active` [n]: OnTick's mask (a graph
        made with masked=True)."""
        if active is not None and self.gmask is None:
            raise ValueError("a graph made without OnTick's mask (masked=True) ticks every row")
        self.sync()
        if self.park.formation_table is not self._tables[0] or self.park.formation_lengths is not self._tables[1]:
            self._capture()
        if self.gmask is not None:
            self.gmask.set(active)
        self.graph.replay()
        self.park.current_time += 1
        if self.gmask is not None:
            self.gmask.mark(self.park)

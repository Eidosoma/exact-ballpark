# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's flight modes, missiles and ship-to-ship collisions in one park on the GPU (the GPU parity report's G1 and G2, the kernel path): `exact_ballpark.vec_composed.VecComposedPark`'s tick as thirteen
launches on torch's current stream, with no allocation and no host synchronisation, so a tick can be captured in a
CUDA graph (`ComposedGraph`). Two libraries:

- this module's, built from vec_composed_cuda.cu, which includes vec_combat_cuda.cu unedited (the
  prepass and phase-1 kernels, and through it the flight and the turning) and adds the
  composed park's own kernels (phase 1's clearing and the warp paths, the partition check, the view's gather and
  scatter, the commit);
- the vec_collide_cuda.cu built with its composed switch (XBPC_COMPOSED=1: the composed candidates and
  COL-711's exception; `vec_collide_cuda.load_library(composed=True)`): its gradient kernel on the park's [world, slot]
  view, and its filing kernel on the rows (the drop-out's re-filing at rest, the re-filing of a ball that moved).

The tick, in the order of `VecComposedPark.evolve_torch` (the torch path, the exactness reference): this
library's part 0 (the prepass, the release mark, the phase 1; a released missile's owner row cleared;
mLastC and the lists cleared; mLastCollision; the warp paths), the
filings at rest (the view's into scratch, the rows' in place), part 1 (the check; the gather), the gradient, part 2
(the scatter; the commit), the filing of the
balls that moved, part 3 (the check). Held to the torch path bit for bit, every tensor every tick, by
tests/test_movement_vec_composed_cuda.py.

(1 November 2026; ABI 2): the commands and the mushrooms in the composed tick.
The release mark takes StopAllFollowers' members; the commit grows a free mushroom at its turn by glibc's powf without the
swap and marks it for the re-filing, or removes it, releasing its members with higher ids before their own commit;
after the host's yaw, pitch and roll, `after` launches the fifth launch (the members with lower ids; PetrifyTroll),
marks the petrified trolls and re-files them with the filing kernel, then checks those filings; ComposedGraph captures
it with the tick. Held to the torch path and to FullBallpark by tests/test_movement_vec_composed_leftovers.py.

**Exactness by construction**, as vec_combat_cuda: the sources scanned for CUDA math-library calls and the PTX checked
(vec_orient_cuda.scan_source, ptx_check); nvcc with -fmad=false, -prec-div=true, -prec-sqrt=true, -ftz=false and no
fast math; every constant generated into a header from the reference's modules (`tables_header`): the composed park's
flags and its at-rest bound (exact_ballpark.vec_composed), the partition's grid and top width (exact_ballpark.space).

The library is built on first use (or when a source, a header, the flags or nvcc change) into a build directory
outside any source tree ($EXACT_BALLPARK_COMPOSED_CUDA_DIR; by default $EXACT_BALLPARK_CACHE/composed_cuda), with
a manifest; it is loaded through ctypes and launched on torch's current stream.
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

from . import space as SP
from . import vec_collide_cuda as KD
from . import vec_combat_cuda as KC
from . import vec_modes_cuda as MK
from . import vec_orient_cuda as K
from .vec_orient import GraphMask

F64, I64, U8 = torch.float64, torch.int64, torch.uint8
from ._cuda_build import SourcePath, cache_root, default_nvcc, read_source  # noqa: E402,F401
SOURCE = SourcePath(Path(__file__).with_name("vec_composed_cuda.cu"))
HEADER = "vec_composed_cuda_tables.h"
FLAGS = K.FLAGS
TARGETS = K.TARGETS
ABI = 2                         # 2: (xbpp_after: the fifth launch, the troll's re-filing)
BLOCK = 128                     # the kernels take at most 128 threads a block (vec_collide_cuda.BLOCK)
FAULTS = {"not_filed_at_rest": 1, "no_last_c": 2, "no_self_collision": 3, "drop_at_warp_position": 4}
# kept apart from FAULTS (whose existing tests run worlds without mushrooms or trolls, where these
# cannot show); a build looks a name up in ALL_FAULTS
G11_FAULTS = {"growth_not_refiled": 5, "later_members_after_commit": 6, "mushroom_swapped": 7,
              "petrified_not_refiled": 8}
ALL_FAULTS = {**FAULTS, **G11_FAULTS}
_LIBS: dict = {}


def _dbits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def table_values() -> dict:
    """Every constant vec_composed_cuda.cu uses beyond vec_combat_cuda's, by its name in the header: (kind, value), read
    from exact_ballpark.vec_composed and exact_ballpark.space; never typed."""
    from . import vec_composed as VP
    return {"P_FLAG_WARP_MASSIVE": ("int", VP.FLAG_WARP_MASSIVE), "P_FLAG_PARTITION": ("int", VP.FLAG_PARTITION),
            "P_REST_SQ": ("d", VP.REST_SQ), "P_LEVEL_WIDTH0": ("d", SP.LEVEL_WIDTH[0]),
            "P_NUM_LEVELS": ("int", SP.NUM_LEVELS), "P_LEVEL_GRID": ("i64", tuple(SP.LEVEL_GRID))}


def tables_header() -> str:
    lines = ["// Generated by exact_ballpark.vec_composed_cuda.tables_header() from exact_ballpark.vec_composed and",
             "// exact_ballpark.space; do not edit.", "#pragma once", "#include <cstdint>", ""]
    for name, (kind, v) in table_values().items():
        if kind == "int":
            lines.append(f"#define {name} {int(v)}")
        elif kind == "i64":
            lines.append(f"static __device__ const long long {name}[{len(v)}] = {{{', '.join(f'{int(x)}LL' for x in v)}}};")
        else:
            lines.append(f"#define {name} 0x{_dbits(v):016x}ULL")
    return "\n".join(lines) + "\n"


# ============================================================================================ build and load

def build_dir() -> Path:
    d = os.environ.get("EXACT_BALLPARK_COMPOSED_CUDA_DIR")
    if d:
        return Path(d)
    return cache_root() / "composed_cuda"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def active_text(text: str, fault: int = 0) -> str:
    """The source as compiled with XBPP_FAULT = fault (the planted faults' blocks, unnested: `#if XBPP_FAULT == k` or
    `#if XBPP_FAULT != k`, to its #else or #endif)."""
    out, state = [], None                    # None outside a fault block; else (keep the first part, in_else)
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"#if\s+XBPP_FAULT\s*(==|!=)\s*(\d+)", s)
        if m and state is None:
            hit = int(m.group(2)) == fault
            state = (hit if m.group(1) == "==" else not hit, False)
            continue
        if state is not None and s.startswith("#else"):
            state = (state[0], True)
            continue
        if state is not None and s.startswith("#endif"):
            state = None
            continue
        if state is None or state[0] != state[1]:
            out.append(line)
    return "\n".join(out) + "\n"


def build(out_dir: Path | None = None, variant: str = "native", source: Path = SOURCE, fault: str | None = None,
          checks: bool = True) -> dict:
    """Compile `source` (which includes vec_combat_cuda.cu, vec_modes_cuda.cu and vec_orient_cuda.cu) with the four
    generated headers into out_dir (by default `build_dir()`); returns the manifest written beside the library: the
    sources' and headers' SHA-256, nvcc's version, flags, defines and targets, the kernels' registers and spills, the
    source scan, the PTX check, the library's SHA-256 and the UTC time. A library already built from the same inputs is
    reused. `fault` (a planted fault of FAULTS) and `checks=False` are for the tests only."""
    if variant not in TARGETS:
        raise ValueError(f"unknown variant {variant!r}")
    level = 0 if fault is None else ALL_FAULTS[fault]
    out_dir = Path(out_dir) if out_dir is not None else build_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = {Path(source).name: read_source(source), KC.SOURCE.name: KC.SOURCE.read_text(),
             MK.SOURCE.name: MK.SOURCE.read_text(), K.SOURCE.name: K.SOURCE.read_text()}
    scanned = {n: (active_text(t, level) if n == Path(source).name else t) for n, t in texts.items()}
    calls = [(n, f"{f}:{k}") for f, t in scanned.items() for n, k in K.scan_source(t)]
    if checks and calls:
        raise RuntimeError(f"the sources call CUDA's math library: {calls}")
    headers = {HEADER: tables_header(), KC.HEADER: KC.tables_header(), MK.HEADER: MK.tables_header(),
               K.HEADER: K.tables_header()}
    nvcc = K._nvcc()
    gencode = [f"-gencode=arch={a},code={c}" for a, c in TARGETS[variant]]
    defines = [f"-DXBPP_FAULT={level}"]
    key = _sha(json.dumps([[_sha(t.encode()) for t in texts.values()], [_sha(h.encode()) for h in headers.values()],
                           FLAGS, gencode, defines, str(nvcc), K.nvcc_version(nvcc), checks]).encode())[:16]
    stem = f"vec_composed_cuda_{variant}_{key}"
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
        cmd = [str(nvcc), *FLAGS, *defines, *gencode, "-I", str(work), "-o", str(work / "lib.so"), str(src)]
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
            "variant": variant, "source": str(source), "fault": fault,
            "sources_sha256": {n: _sha(t.encode()) for n, t in texts.items()},
            "headers_sha256": {n: _sha(h.encode()) for n, h in headers.items()},
            "checks": checks, "source_scan": calls, "nvcc": version.strip().splitlines()[-1], "nvcc_path": str(nvcc),
            "flags": FLAGS, "defines": defines, "targets": [f"{a}->{c}" for a, c in TARGETS[variant]],
            "embedded": listing, "ptxas": resources, "ptx_check": check, "library": lib.name,
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


def load_library(variant: str = "native", source: Path = SOURCE, fault: str | None = None,
                 checks: bool = True) -> dict:
    """This module's compiled library (built if missing or stale), its functions typed, and its manifest; once a
    process."""
    k = (variant, str(source), _sha(Path(source).read_bytes()), fault, checks)    # the text: a rewritten copy
    if k in _LIBS:
        return _LIBS[k]
    manifest = build(None, variant, source, fault, checks)
    lib = ctypes.CDLL(str(build_dir() / manifest["library"]))
    if lib.xbpp_abi() != ABI:
        raise RuntimeError(f"the library's ABI {lib.xbpp_abi()} is not {ABI}")
    if lib.xbpp_fault() != (0 if fault is None else ALL_FAULTS[fault]):
        raise RuntimeError("the library's planted-fault switch is not the one asked for")
    p, i32 = ctypes.c_void_p, ctypes.c_int
    lib.xbpo_error_string.restype = ctypes.c_char_p
    lib.xbpo_error_string.argtypes = [i32]
    lib.xbpp_tick.argtypes = [i32, ctypes.POINTER(State), p, i32]
    lib.xbpp_tick.restype = i32
    lib.xbpp_after.argtypes = [i32, ctypes.POINTER(State), p, i32]
    lib.xbpp_after.restype = i32
    lib.xbpp_info.argtypes = [i32] + [ctypes.POINTER(i32)] * 5
    lib.xbpp_info.restype = i32
    entry = {"lib": lib, "manifest": manifest}
    _LIBS[k] = entry
    return entry


def libraries(variant: str = "native", composed_lib: dict | None = None, collide_lib: dict | None = None) -> dict:
    """The two libraries a composed tick launches from: this module's and the composed build."""
    return {"composed": composed_lib or load_library(variant),
            "collide": collide_lib or KD.load_library(variant, composed=True)}


def _check(rc: int, lib, fn: str) -> None:
    if rc:
        msg = lib.xbpo_error_string(rc).decode() if rc > 0 else "refused (block size or arguments)"
        raise RuntimeError(f"{fn} failed ({rc}: {msg})")


# ======================================================================================================= the state

ROWS = (("last_c", F64, 3), ("hits", I64, -1), ("level", I64, 0), ("key", I64, 3), ("mod", I64, 3))
SCRATCH = (("wp", F64, 3), ("wv", F64, 3), ("dropped", U8, 0), ("rest", U8, 0), ("moved", U8, 0),
           ("released", U8, 0), ("fpos", F64, 3), ("fvel", F64, 3), ("flevel", I64, 0), ("fkey", I64, 3),
           ("fmod", I64, 3))
VIEW = (("v_pos", F64, 3), ("v_vel", F64, 3), ("v_last_g", F64, 3), ("v_mass", F64, 0), ("v_agility", F64, 0),
        ("v_tf", F64, 0), ("v_radius", F64, 0), ("v_last_c", F64, 3), ("v_last_col", F64, 0),
        ("v_exists", U8, 0), ("v_free", U8, 0), ("v_massive", U8, 0), ("v_junk", U8, 0), ("v_cloaked", U8, 0),
        ("v_responder", U8, 0), ("v_missile", U8, 0), ("v_dropped", U8, 0),
        ("v_id", I64, 0), ("v_level", I64, 0), ("v_key", I64, 3), ("v_mod", I64, 3), ("v_mode", I64, 0),
        ("v_owner", I64, 0), ("v_follow", I64, 0), ("v_bubble", I64, 0), ("v_tslot", I64, 0),
        ("v_hits", I64, -1), ("v_hits_n", I64, 0), ("v_imp_n", I64, 0), ("v_imp_slot", I64, 0))
EXTRA = (("v_imp_s", F64), ("v_imp_cl", F64), ("v_overflow", U8))     # the record and overflow (not read here)


class State(ctypes.Structure):
    """vec_composed_cuda.cu's `struct Composed`, field for field."""
    _fields_ = ([("c", KC.State), ("last_c", ctypes.c_void_p), ("wp", ctypes.c_void_p), ("wv", ctypes.c_void_p),
                 ("hits", ctypes.c_void_p), ("level", ctypes.c_void_p), ("key", ctypes.c_void_p),
                 ("mod", ctypes.c_void_p), ("dropped", ctypes.c_void_p), ("rest", ctypes.c_void_p),
                 ("moved", ctypes.c_void_p), ("released", ctypes.c_void_p), ("fpos", ctypes.c_void_p),
                 ("fvel", ctypes.c_void_p), ("flevel", ctypes.c_void_p), ("fkey", ctypes.c_void_p),
                 ("fmod", ctypes.c_void_p), ("slot_rows", ctypes.c_void_p), ("row_slot", ctypes.c_void_p)]
                + [(name, ctypes.c_void_p) for name, _, _ in VIEW]
                + [("H", ctypes.c_int64), ("W", ctypes.c_int64), ("S", ctypes.c_int64), ("Hv", ctypes.c_int64)])


def buffers(park) -> dict:
    """The kernel path's own tensors for a loaded park (its scratch rows and the view), allocated once a load."""
    W, S = (int(x) for x in park.slot_rows.shape)
    n, H = park.n, int(park.hits.shape[1])
    Hv = H - 1
    dev = park.dev
    out = {}
    for name, dt, cols in SCRATCH:
        out[name] = torch.zeros((n, cols) if cols else (n,), dtype=dt, device=dev)
    for name, dt, cols in VIEW:
        shape = (W * S, Hv) if cols == -1 else ((W * S, cols) if cols else (W * S,))
        out[name] = torch.zeros(shape, dtype=dt, device=dev)
    out["v_imp_s"] = torch.zeros(W * S, dtype=F64, device=dev)
    out["v_imp_cl"] = torch.zeros(W * S, dtype=F64, device=dev)
    out["v_overflow"] = torch.zeros(W, dtype=U8, device=dev)
    out["shape"] = (n, W, S, H)
    return out


def _held(park, name: str, dtype, shape: tuple) -> torch.Tensor:
    t = getattr(park, name)
    if t.dtype != dtype or tuple(t.shape) != shape or t.device != park.dev:
        raise ValueError(f"{name} must be a {dtype} tensor of shape {shape} on {park.dev}, not {t.dtype} "
                         f"{tuple(t.shape)} on {t.device}")
    if not t.is_contiguous():
        t = t.contiguous()
        setattr(park, name, t)
    return t


def make_state(park) -> tuple:
    """The launches' arguments for a VecComposedPark on a CUDA device: this library's struct (every row tensor checked
    and made contiguous, as vec_combat_cuda.make_state), and the dims, pointers and constants for the gradient on the
    view and for the two filings on the rows."""
    if park.dev.type != "cuda":
        raise ValueError("the composed kernels need a park on a CUDA device")
    b = park.kernel_buffers()
    n, W, S, H = b["shape"]
    s = State()
    s.c = KC.make_state(park)
    rows = {}
    for name, dt, cols in ROWS:
        shape = (n, H) if cols == -1 else ((n, cols) if cols else (n,))
        rows[name] = _held(park, name, dt, shape)
        setattr(s, name, rows[name].data_ptr())
    for name, _, _ in SCRATCH:
        setattr(s, name, b[name].data_ptr())
    slot_rows = _held(park, "slot_rows", I64, (W, S))
    row_slot = _held(park, "row_slot", I64, (n,))
    s.slot_rows, s.row_slot = slot_rows.data_ptr(), row_slot.data_ptr()
    for name, _, _ in VIEW:
        setattr(s, name, b[name].data_ptr())
    s.H, s.W, s.S, s.Hv = H, W, S, H - 1
    view = park.view
    consts = KD._consts(view)
    # the gradient on the view (xbpc_gradient's order; the composed fields after overflow)
    g_dims = (ctypes.c_int64 * 7)(W, S, S, S - 1, 0, H - 1, 1)
    g_names = ("v_pos", "v_vel", "v_last_g", "v_mass", "v_agility", "v_tf", "v_radius", "v_exists", "v_free",
               "v_massive", "v_junk", "v_cloaked", "v_responder", "v_id", "v_level", "v_key", "v_mod", "v_last_c",
               "v_last_col", "v_imp_s", "v_imp_cl", "v_hits", "v_hits_n", "v_imp_slot", "v_imp_n", "v_overflow",
               "v_missile", "v_dropped", "v_mode", "v_owner", "v_follow", "v_bubble", "v_tslot")
    g_ptrs = (ctypes.c_void_p * len(g_names))(*[b[k].data_ptr() for k in g_names])
    # the filing on the rows (xbpc_tick part 2: W = 1, N = n, R = 1), from the drop-out point or the committed state
    f_dims = (ctypes.c_int64 * 6)(1, n, 1, 1, 1, 0)
    ball = {k: _held(park, k, dt, (n,)) for k, dt in (("free", torch.bool), ("mass", F64), ("agility", F64),
                                                     ("max_vel", F64), ("speed_fraction", F64),
                                                     ("time_factor", F64), ("radius", F64))}
    pos, vel = _held(park, "pos", F64, (n, 3)), _held(park, "vel", F64, (n, 3))

    def filing_ptrs(which, p, v, out=None):
        t = {"free": ball["free"], "which": which, "mass": ball["mass"], "agility": ball["agility"],
             "max_vel": ball["max_vel"], "speed_fraction": ball["speed_fraction"],
             "time_factor": ball["time_factor"], "radius": ball["radius"], "pos": p, "vel": v,
             "level_in": rows["level"], "key_in": rows["key"], "mod_in": rows["mod"],
             "level": (out or rows)["level"], "key": (out or rows)["key"], "mod": (out or rows)["mod"]}
        return (ctypes.c_void_p * len(KD._TICK))(*[t[k].data_ptr() if k in t else None for k in KD._TICK])
    keep = (rows, ball, pos, vel, slot_rows, row_slot)          # the tensors the pointers name, kept alive
    return s, {"g_dims": g_dims, "g_ptrs": g_ptrs, "consts": consts, "f_dims": f_dims,
               "rest_view": filing_ptrs(b["rest"], b["fpos"], b["fvel"],
                                       {"level": b["flevel"], "key": b["fkey"], "mod": b["fmod"]}),
               "rest": filing_ptrs(b["rest"], b["wp"], b["wv"]), "moved": filing_ptrs(b["moved"], pos, vel),
               "keep": keep}


def tick(park, block: int = BLOCK, libs: dict | None = None) -> None:
    """One composed tick of a VecComposedPark in place (collisions on): thirteen launches on torch's current stream of the
    park's device, from this library and the composed build; no allocation, no host synchronisation."""
    if park.dev.type != "cuda":
        raise ValueError("the composed kernels need a park on a CUDA device (evolve_torch is the CPU path)")
    L = libs or libraries()
    lib, col = L["composed"]["lib"], L["collide"]["lib"]
    s, d = make_state(park)
    if not park.n:
        return
    with torch.cuda.device(park.dev):
        st = torch.cuda.current_stream(park.dev).cuda_stream
        _check(lib.xbpp_tick(0, ctypes.byref(s), st, int(block)), lib, "xbpp_tick(0)")
        KD._check(col.xbpc_tick(2, d["f_dims"], d["rest_view"], d["consts"], st, int(block)), col,
                  "xbpc_tick(filing)")
        KD._check(col.xbpc_tick(2, d["f_dims"], d["rest"], d["consts"], st, int(block)), col, "xbpc_tick(filing)")
        _check(lib.xbpp_tick(1, ctypes.byref(s), st, int(block)), lib, "xbpp_tick(1)")
        KD._check(col.xbpc_gradient(d["g_dims"], d["g_ptrs"], d["consts"], st, int(block)), col, "xbpc_gradient")
        _check(lib.xbpp_tick(2, ctypes.byref(s), st, int(block)), lib, "xbpp_tick(2)")
        KD._check(col.xbpc_tick(2, d["f_dims"], d["moved"], d["consts"], st, int(block)), col, "xbpc_tick(filing)")
        _check(lib.xbpp_tick(3, ctypes.byref(s), st, int(block)), lib, "xbpp_tick(3)")


def after(park, block: int = BLOCK, libs: dict | None = None) -> None:
    """The fifth launch of a composed tick, after the host's yaw, pitch and roll (VecComposedPark.evolve):
    the combat_after_kernel (a removed mushroom's members whose turns came before its own, released; PetrifyTroll,
    each troll petrified marked in `petrified`), the re-filing mark, the filing of the petrified trolls (SetBallFree
    (false)'s InsertBallInBoxes, from their state at rest and fixed) and the check of those filings: three launches of
    this library and one of the collision kernel's on torch's current stream; no allocation, no host synchronisation."""
    if park.dev.type != "cuda":
        raise ValueError("the composed kernels need a park on a CUDA device (evolve_torch is the CPU path)")
    L = libs or libraries()
    lib, col = L["composed"]["lib"], L["collide"]["lib"]
    s, d = make_state(park)
    if not park.n:
        return
    with torch.cuda.device(park.dev):
        st = torch.cuda.current_stream(park.dev).cuda_stream
        _check(lib.xbpp_after(0, ctypes.byref(s), st, int(block)), lib, "xbpp_after(0)")
        KD._check(col.xbpc_tick(2, d["f_dims"], d["moved"], d["consts"], st, int(block)), col, "xbpc_tick(filing)")
        _check(lib.xbpp_after(1, ctypes.byref(s), st, int(block)), lib, "xbpp_after(1)")


def kernel_info(device=None, libs: dict | None = None) -> dict:
    """This library's kernels as loaded on a device (registers, local memory, versions), its manifest's PTX check, and
    the composed gradient kernel's (vec_collide_cuda.info)."""
    L = libs or libraries()
    e = L["composed"]
    out = {}
    names = ("prepass", "phase1", "pre", "check", "gather", "scatter", "commit", "mark", "after", "refile_mark")
    for which, name in enumerate(names):
        vals = [ctypes.c_int() for _ in range(5)]
        with torch.cuda.device(torch.device(device or "cuda")):
            rc = e["lib"].xbpp_info(which, *[ctypes.byref(v) for v in vals])
        _check(rc, e["lib"], "xbpp_info")
        regs, local, binary, ptx, threads = (v.value for v in vals)
        out[name] = {"registers": regs, "local_bytes": local, "binary_version": binary, "ptx_version": ptx,
                     "max_threads_per_block": threads}
    out.update(library=e["manifest"]["library"], library_sha256=e["manifest"]["library_sha256"],
               targets=e["manifest"]["targets"], ptx_check=e["manifest"]["ptx_check"],
               gradient=KD.info(L["collide"]))
    return out


# ================================================================================== a tick captured as a CUDA graph

class ComposedGraph:
    """A VecComposedPark's composed tick (its thirteen launches) captured once as a CUDA graph on the park's own tensors,
    so each replay is one tick in one launch: the same kernels on the same buffers (held to eager ticks bit for bit by
    the tests). As vec_combat_cuda.CombatGraph: commands and setters rebind tensors, and `tick()` first copies any
    rebound one into its buffer. Only with dynamical orientation (without it the yaw, pitch and roll are computed on the
    host after each tick: `VecCombatPark.after_tick`). `after`'s four launches (three of this
    library's and one of the collision kernel's) are captured with the tick (with dynamical orientation nothing waits for the host
    between them). With `masked=True` the capture holds OnTick's per-world mask as a device buffer (vec_orient.GraphMask, as
    vec_modes_cuda.ModesGraph's): `tick(active)`."""

    def __init__(self, park, block: int = BLOCK, libs: dict | None = None, masked: bool = False):
        if not park.dynamical_orientation:
            raise ValueError("without dynamical orientation a tick needs the host (the yaw, pitch and roll)")
        if not park.collisions:
            raise ValueError("ComposedGraph captures the composed tick (collisions on); VecCombatPark's is CombatGraph")
        self.park, self.block = park, int(block)
        self.libs = libs or libraries()
        self.names = ([n for n, _, _ in MK.STATE_TENSORS] + [n for n, _, _ in KC.COMBAT_TENSORS]
                      + [n for n, _, _ in ROWS])
        self.buf = {}
        for name in self.names:
            t = getattr(park, name).contiguous().clone()
            self.buf[name] = t
            setattr(park, name, t)
        self._kb = park.kernel_buffers()
        saved = {n: b.clone() for n, b in self.buf.items()}
        tick(park, self.block, self.libs)               # the warm-up runs one real tick on the buffers ...
        after(park, self.block, self.libs)
        torch.cuda.synchronize(park.dev)
        for n, b in self.buf.items():                   # ... and the state is put back as it was
            b.copy_(saved[n])
        torch.cuda.synchronize(park.dev)
        self._tables = (park.formation_table, park.formation_lengths, park.slot_rows, park.row_slot)
        self.gmask = GraphMask(self.buf, park.n, park.dev) if masked else None
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):              # captured, not run
            if self.gmask is not None:
                self.gmask.save()
            tick(park, self.block, self.libs)
            after(park, self.block, self.libs)          # the fifth launch: no host step between
            if self.gmask is not None:
                self.gmask.put_back()

    def sync(self) -> None:
        for name, b in self.buf.items():
            t = getattr(self.park, name)
            if t is not b:
                b.copy_(t)
                setattr(self.park, name, b)

    def tick(self, active: torch.Tensor | None = None) -> None:
        p = self.park
        if active is not None and self.gmask is None:
            raise ValueError("a graph made without OnTick's mask (masked=True) ticks every row")
        if (p.formation_table is not self._tables[0] or p.formation_lengths is not self._tables[1]
                or p.slot_rows is not self._tables[2] or p.row_slot is not self._tables[3]
                or p.kernel_buffers() is not self._kb):
            raise RuntimeError("the park was loaded again or LoadFormations ran after the graph was made: a new graph")
        self.sync()
        if self.gmask is not None:
            self.gmask.set(active)
        self.graph.replay()
        p.current_time += 1
        if self.gmask is not None:
            self.gmask.mark(p)

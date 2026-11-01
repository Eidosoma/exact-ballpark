# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer for VecModesPark as one CUDA kernel a tick, one thread a world: the pre-tick
of `exact_ballpark.vec_server.VecServer` (the queue's duplicate test and lookups, every queued action applied in queue
order by the park's own rules, the bubble update, the history's routes, CloakBall applied after the bubble update, the
uncloaks' notices) and its send flags, which the torch path runs in some hundreds of small launches
(`VecServer(..., kernel="cuda")` selects it; `kernel="torch"`, the default, stays the exactness reference, unchanged).
Checked bit for bit against the torch path, and through it against exact_ballpark.server, by
tests/test_movement_vec_server_cuda.py.

The source is vec_server_cuda.cu beside this module. It includes vec_modes_cuda.cu (the kernel, which
includes vec_orient_cuda.cu), unedited, for their device functions: the double vectors, glibc's exp
(the decay factor), the C library's float functions, carbonengine's vectors, quaternions and matrices (dynamical
orientation's snap, the rotation setter). The library is built on first use (or when a source, a generated header,
the flags, the planted-fault switch or nvcc change) into a build directory outside any source tree
($EXACT_BALLPARK_SERVER_CUDA_DIR; by default $EXACT_BALLPARK_CACHE/server_cuda), with a manifest; it is loaded
through ctypes, its kernel's module loaded at once (`xbpv_info`), and launched on torch's current stream with no
allocation and no host synchronisation, so a launch can be captured in a CUDA graph (vec_server.ServerGraph).

**Exactness by construction** (as vec_modes_cuda, whose flags, source scan and PTX check this build applies to all three
sources): every double operation __dadd_rn ... __dsqrt_rn in the torch path's order, every float operation __fadd_rn
... __fsqrt_rn where the park computes in float; a fused multiply-add only inside the included C-library functions;
nvcc with -fmad=false, -prec-div=true, -prec-sqrt=true, -ftz=false and no fast math; no function of CUDA's math
library; every constant from the generated headers: vec_modes_cuda's and vec_orient_cuda's, and this module's
(`tables_header`): the kinds' codes and the routes from exact_ballpark.vec_server, and the three literals the commands use
that the modes' header lacks, each with the torch path's own spelling (`LITERALS`; a test finds each in its source).

**Differences in form from the torch path** (the values are the same): the kernel updates the park's tensors and the
server's state (stamp_for_system, settled, the bubble update's lists and time) in place where the torch path rebinds
them; it computes the send flags in the pre-tick (they do not depend on the park's tick), so the post-tick does
nothing; a tick's queue holds at most 256 pushes.
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
import struct
import subprocess
from pathlib import Path

import torch

from . import vec_modes_cuda as M
from . import vec_orient_cuda as K

F64, I64 = torch.float64, torch.int64
from ._cuda_build import SourcePath, cache_root, default_nvcc, read_source  # noqa: E402,F401
SOURCE = SourcePath(Path(__file__).with_name("vec_server_cuda.cu"))
HEADER = "vec_server_cuda_tables.h"
FLAGS = K.FLAGS
TARGETS = dict(K.TARGETS, sm90=[("compute_90", "sm_90"), ("compute_90", "compute_90")])
ABI = 1
BLOCK = 128
MAX_SLOTS = 256
FAULTS = {"unit_unnormalised": 1, "cloak_with_the_others": 2, "no_release": 3, "speed_fraction_unrounded": 4,
          "interactive_ignored": 5, "duplicate_without_local": 6, "cloaked_to_the_bubble": 7}   # tests only
# the pending queries' planted fault (kept apart from FAULTS, whose tests run the worlds without an
# undo): the kernel tests again the calls an undo already tested (their mask byte's bit 1 ignored)
PENDING_FAULTS = {"undo_retested": 8}
# the (apart from FAULTS, whose tests run worlds without these): StopAllFollowers' members taken by the
# follow modes (a member SetBoid took out of one kept), SetBallTroll's delay floor left out, SetBallFree(false) keeping
# mLastG
G11_FAULTS = {"release_by_follow_modes": 9, "troll_delay_unfloored": 10, "fixed_keeps_last_g": 11}
ALL_FAULTS = {**FAULTS, **PENDING_FAULTS, **G11_FAULTS}
_LIBS: dict = {}

# the commands' literals the modes' header lacks: (value, the torch path's spelling, the module that holds it)
LITERALS = {
    "SV_SNAP_MOVING": (1e-12, "v[:, 2] * v[:, 2]) > 1e-12)", "vec_orient"),   # SnapOrientation: |v|^2 > 1e-12
    "SV_SNAP_TINY": (1e-12, "_where(len_dir < 1e-12,", "vec_orient"),         # ... a direction under 1e-12: identity
    "SV_SNAP_UPRIGHT": (0.99, "L.f32(zero + 0.99)", "vec_orient"),            # ... rolled upright below 0.99
    "SV_ROTATION_MIN": (1e-20, "(length > 1e-20)", "vec_orient"),              # SetBallRotation: normalised above
}
INT_LITERALS = {
    "SV_WARPIN_BACK": (5, "torch.clamp(self.clock - 5, min=0)", "vec_modes"),  # EntityWarpIn: begun five ticks ago
}


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def literals_in_source() -> dict:
    """Each of LITERALS and INT_LITERALS: whether its module still writes it as the header holds it."""
    from . import vec_modes, vec_orient
    mods = {"vec_modes": inspect.getsource(vec_modes), "vec_orient": inspect.getsource(vec_orient)}
    return {name: phrase in mods[mod] for name, (_, phrase, mod) in {**LITERALS, **INT_LITERALS}.items()}


def _dbits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def tables_header(perturb: dict | None = None) -> str:
    """The kinds' codes (K_<KIND>), the routes, the normal cloak's mode and the literals, generated from
    exact_ballpark.vec_server and LITERALS; `perturb` ({name: value}) changes one, for the tests only."""
    from . import vec_server as VS
    lines = ["// generated by exact_ballpark.vec_server_cuda.tables_header: do not edit", "#pragma once"]
    ints = {f"K_{_snake(name)}": code for name, code in VS.CODE.items()}
    ints.update(SV_NONE=VS.NONE, SV_BUBBLE=VS.BUBBLE, SV_CHARACTER=VS.CHARACTER, SV_NORMAL_CLOAK=VS.DSTNORMALCLOAK)
    from . import vec_modes                             # SetBallTroll's floor, the park's constant
    ints.update(SV_TROLL_MIN_DELAY=vec_modes.TROLL_MIN_DELAY)
    ints.update({k: v for k, (v, _, _) in INT_LITERALS.items()})
    dbl = {k: v for k, (v, _, _) in LITERALS.items()}
    for k, v in (perturb or {}).items():
        if k in ints:
            ints[k] = v
        elif k in dbl:
            dbl[k] = v
        else:
            raise KeyError(k)
    for k, v in ints.items():
        lines.append(f"#define {k} {int(v)}")
    for k, v in dbl.items():
        lines.append(f"#define {k} 0x{_dbits(v):016x}ULL")
    return "\n".join(lines) + "\n"


# ============================================================================================ build and load

def build_dir() -> Path:
    d = os.environ.get("EXACT_BALLPARK_SERVER_CUDA_DIR")
    if d:
        return Path(d)
    return cache_root() / "server_cuda"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def active_text(text: str, fault: int = 0) -> str:
    """The source as compiled with XBPV_FAULT = fault: each `#if XBPV_FAULT == k` block's lines kept only for k = fault,
    its #else part only otherwise (the planted faults' blocks, unnested)."""
    out, state = [], None
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"#if\s+XBPV_FAULT\s*==\s*(\d+)", s)
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


def build(out_dir: Path | None = None, variant: str = "native", perturb: dict | None = None, fault: str | None = None,
          checks: bool = True) -> dict:
    """Compile vec_server_cuda.cu (with vec_modes_cuda.cu, vec_orient_cuda.cu and the three generated headers) into
    out_dir (by default `build_dir()`); returns the manifest written beside the library: the sources' and headers'
    SHA-256, nvcc's version, flags and targets, the kernel's registers and spills, the source scan and the PTX check,
    the library's SHA-256 and the UTC time. A library already built from the same inputs is reused. `perturb`, `fault`
    (a planted fault of FAULTS) and `checks=False` (the scan and the PTX check recorded, not enforced) are for the
    tests only."""
    if variant not in TARGETS:
        raise ValueError(f"unknown variant {variant!r}")
    level = 0 if fault is None else ALL_FAULTS[fault]
    out_dir = Path(out_dir) if out_dir is not None else build_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    text, mtext, otext = SOURCE.read_text(), M.SOURCE.read_text(), K.SOURCE.read_text()
    calls = ([(n, f"{SOURCE.name}:{k}") for n, k in K.scan_source(active_text(text, level))]
             + [(n, f"{M.SOURCE.name}:{k}") for n, k in K.scan_source(mtext)]
             + [(n, f"{K.SOURCE.name}:{k}") for n, k in K.scan_source(otext)])
    if checks and calls:
        raise RuntimeError(f"the sources call CUDA's math library: {calls}")
    missing = [k for k, ok in literals_in_source().items() if not ok]
    if missing:
        raise RuntimeError(f"the torch path no longer writes the literals {missing} as the kernel's header holds them")
    header, mheader, oheader = tables_header(perturb), M.tables_header(), K.tables_header()
    nvcc = K._nvcc()
    gencode = [f"-gencode=arch={a},code={c}" for a, c in TARGETS[variant]]
    defines = [f"-DXBPV_FAULT={level}"]
    key = _sha(json.dumps([_sha(t.encode()) for t in (text, mtext, otext, header, mheader, oheader)]
                          + [FLAGS, gencode, defines, str(nvcc), K.nvcc_version(nvcc), checks]).encode())[:16]
    stem = f"vec_server_cuda_{variant}_{key}"
    lib, man = out_dir / f"{stem}.so", out_dir / f"{stem}.json"
    if lib.exists() and man.exists():
        m = json.loads(man.read_text())
        if m.get("library_sha256") == _sha(lib.read_bytes()) and (m.get("ptx_check", {}).get("ok") or not checks):
            return m
    work = out_dir / f".{stem}.{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        (work / HEADER).write_text(header)
        (work / M.HEADER).write_text(mheader)
        (work / K.HEADER).write_text(oheader)
        (work / M.SOURCE.name).write_text(mtext)
        (work / K.SOURCE.name).write_text(otext)
        src = work / SOURCE.name
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
        check = K.ptx_check(ptx)
        if checks and not check["ok"]:
            raise RuntimeError(f"the PTX has unpinned floating-point instructions: {check['refused']}")
        resources = [ln.strip() for ln in (r.stdout + r.stderr).splitlines()
                     if "registers" in ln or "spill" in ln or "Compiling entry" in ln or "stack frame" in ln]
        version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True).stdout
        manifest = {
            "variant": variant, "source_sha256": _sha(text.encode()), "modes_source_sha256": _sha(mtext.encode()),
            "orient_source_sha256": _sha(otext.encode()), "header": HEADER, "header_sha256": _sha(header.encode()),
            "modes_header_sha256": _sha(mheader.encode()), "orient_header_sha256": _sha(oheader.encode()),
            "perturb": perturb or {}, "fault": fault, "checks": checks, "source_scan": calls,
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


class Park(ctypes.Structure):
    """vec_server_cuda.cu's `struct Park`."""
    _fields_ = [("cloaked", ctypes.c_void_p), ("bubble", ctypes.c_void_p), ("refiled", ctypes.c_void_p)]


class Queue(ctypes.Structure):
    """vec_server_cuda.cu's `struct Queue`."""
    _fields_ = ([(n, ctypes.c_void_p) for n in ("mask", "ball", "target", "f", "n", "loc_in", "ok", "row", "trow",
                                                 "loc", "route", "notice")]
                + [("J", ctypes.c_int64), ("kind", ctypes.c_int8 * MAX_SLOTS)])


class Srv(ctypes.Structure):
    """vec_server_cuda.cu's `struct Srv`."""
    _fields_ = ([(n, ctypes.c_void_p) for n in ("rows_of_world", "ids_of_world", "pilot", "stamp_for_system", "stamp",
                                                 "clock", "bubble_time", "settled", "fresh", "listed",
                                                 "has_character", "gets_bubble")]
                + [("W", ctypes.c_int64), ("N", ctypes.c_int64), ("R", ctypes.c_int64), ("neg_kdt", ctypes.c_double),
                   ("master", ctypes.c_int)])


def load_library(variant: str = "native", perturb: dict | None = None, fault: str | None = None,
                 checks: bool = True) -> dict:
    """The compiled library (built if missing or stale), its functions typed, its kernel's module loaded on the current
    CUDA device (`xbpv_info`: no later launch loads it, as a captured one must not), and its manifest; once a
    process."""
    k = (variant, json.dumps(perturb or {}, sort_keys=True), fault, checks)
    if k in _LIBS:
        return _LIBS[k]
    manifest = build(None, variant, perturb, fault, checks)
    lib = ctypes.CDLL(str(build_dir() / manifest["library"]))
    if lib.xbpv_abi() != ABI:
        raise RuntimeError(f"the library's ABI {lib.xbpv_abi()} is not {ABI}")
    if lib.xbpv_fault() != (0 if fault is None else ALL_FAULTS[fault]):
        raise RuntimeError("the library's planted-fault switch is not the one asked for")
    p, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.xbpv_error_string.restype = ctypes.c_char_p
    lib.xbpv_error_string.argtypes = [i32]
    lib.xbpv_pre_tick.argtypes = [ctypes.POINTER(M.State), ctypes.POINTER(Park), ctypes.POINTER(Queue),
                                  ctypes.POINTER(Srv), i32, i64, i64, p, i32]
    lib.xbpv_pre_tick.restype = i32
    lib.xbpv_info.argtypes = [ctypes.POINTER(i32)]
    lib.xbpv_info.restype = i32
    entry = {"lib": lib, "manifest": manifest}
    out = (ctypes.c_int * 5)()
    _check(lib.xbpv_info(out), lib, "xbpv_info")
    entry["info"] = dict(zip(("registers", "local_bytes", "max_threads_per_block", "binary_version", "ptx_version"),
                             list(out)))
    _LIBS[k] = entry
    return entry


def _check(rc: int, lib, fn: str) -> None:
    if rc != 0:
        msg = lib.xbpv_error_string(rc).decode() if rc > 0 else "refused (the slots, the block size or the phase)"
        raise RuntimeError(f"{fn} failed ({rc}: {msg})")


def _ptr(t: torch.Tensor, dtype, shape: tuple, what: str) -> int:
    if t.dtype != dtype or tuple(t.shape) != tuple(shape) or not t.is_contiguous() or t.device.type != "cuda":
        raise ValueError(f"{what}: a contiguous {dtype} tensor of shape {tuple(shape)} on a CUDA device, not {t.dtype} "
                         f"{tuple(t.shape)} on {t.device}")
    return t.data_ptr()


def launch(server, q: dict, phase: int, j0: int = 0, j1: int = 0, block: int | None = None,
           lib: dict | None = None) -> None:
    """One launch of the server kernel on torch's current stream: `q` holds the tick's stacked pushes and the outputs
    (mask, ball, target, f, n, loc_in, ok, row, trow, loc, route, notice, kinds, stamp, clock, has_character,
    gets_bubble); the park's and the server's state are read and written in place. No allocation, no host
    synchronisation."""
    e = lib or server._lib
    block = BLOCK if block is None else int(block)
    if not 1 <= block <= e["info"]["max_threads_per_block"]:
        raise ValueError(f"block {block}: the kernel takes 1 to {e['info']['max_threads_per_block']} threads a block "
                         f"(its registers)")
    a, park = server.a, server.a.park
    W, R, N = server.W, server.R, a.rows_of_world.shape[1]
    J = len(q["kinds"])
    if J > MAX_SLOTS:
        raise ValueError(f"{J} pushes in a tick: the kernel takes at most {MAX_SLOTS}")
    m = M.make_state(park)
    p = Park(*(_ptr(getattr(park, n), I64, (R,), f"park.{n}") for n in ("cloaked", "bubble", "refiled")))
    qs = Queue(_ptr(q["mask"], torch.uint8, (J, W), "mask"), _ptr(q["ball"], I64, (J, W), "ball"),
               _ptr(q["target"], I64, (J, W), "target"), _ptr(q["f"], F64, (J, W, 4), "f"),
               _ptr(q["n"], I64, (J, W), "n"), _ptr(q["loc_in"], I64, (J, W), "loc_in"),
               _ptr(q["ok"], torch.bool, (J, W), "ok"), _ptr(q["row"], I64, (J, W), "row"),
               _ptr(q["trow"], I64, (J, W), "trow"), _ptr(q["loc"], I64, (J, W), "loc"),
               _ptr(q["route"], torch.int8, (W, J), "route"), _ptr(q["notice"], torch.bool, (J, R), "notice"), J)
    for j, k in enumerate(q["kinds"]):
        qs.kind[j] = int(k)
    s = Srv(_ptr(a.rows_of_world, I64, (W, N), "rows_of_world"), _ptr(a.ids_of_world, I64, (W, N), "ids_of_world"),
            _ptr(server.pilot, I64, (R,), "pilot"), _ptr(server.stamp_for_system, I64, (W,), "stamp_for_system"),
            _ptr(q["stamp"], I64, (W,), "stamp"), _ptr(q["clock"], I64, (W,), "clock"),
            _ptr(server._bubble_time, I64, (W,), "bubble_time"), _ptr(server.settled, torch.bool, (R,), "settled"),
            _ptr(server._fresh, torch.bool, (R,), "fresh"), _ptr(server._listed, torch.bool, (R,), "listed"),
            _ptr(q["has_character"], torch.bool, (R,), "has_character"),
            _ptr(q["gets_bubble"], torch.bool, (R,), "gets_bubble"), W, N, R,
            -park.friction * park.dt, int(a.master))
    with torch.cuda.device(park.dev):
        rc = e["lib"].xbpv_pre_tick(ctypes.byref(m), ctypes.byref(p), ctypes.byref(qs), ctypes.byref(s), int(phase),
                                    int(j0), int(j1), torch.cuda.current_stream(park.dev).cuda_stream, int(block))
    _check(rc, e["lib"], "xbpv_pre_tick")

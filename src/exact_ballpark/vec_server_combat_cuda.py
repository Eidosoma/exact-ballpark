# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer over the combat park as one CUDA kernel a tick, one thread a world: the
pre-tick of `exact_ballpark.vec_server_combat.CombatServer` (VecServer over VecCombatPark on the master: the queue's
duplicate test and each ball's row among the balls in the park, every queued action applied in queue order by the
combat park's own rules, LaunchMissile among them, Stop with the master's formation-slot free and StopAllFollowers
with its missile branch; the bubble update over the balls in the park; the history's routes with CloakBall applied
there, releasing the cloaked ball's missiles and formation followers; the uncloaks' notices) and its send flags, which
the torch path runs in some hundreds of small launches (`CombatServer(..., kernel="cuda")` selects it; "torch", the
default, stays the exactness reference, unchanged). Checked bit for bit against the torch path, and through it against
exact_ballpark.server over FullBallpark, by tests/test_movement_vec_server_combat_cuda.py.

The source is vec_server_combat_cuda.cu beside this module. It includes vec_server_cuda.cu (the server
kernel, and through it vec_modes_cuda.cu and vec_orient_cuda.cu) and vec_ypr_cuda.cu (the device atan2
and asin), unedited, for their functions. The library is built on first use (or when a source, a generated header, the
flags, the planted-fault switch or nvcc change) into a build directory outside any source tree
($EXACT_BALLPARK_SERVER_COMBAT_CUDA_DIR; by default beside vec_server_cuda's, `server_combat_cuda`), with a manifest; it is loaded
through ctypes, its kernel's module loaded at once (`xbpk_info`), and launched on torch's current stream with no
allocation and no host synchronisation, so a tick can be captured in a CUDA graph (vec_server.ServerGraph; the combat
park's tick waits for no host only with dynamical orientation).

**Exactness by construction**, as the kernels it includes: their flags (nvcc -fmad=false, -prec-div=true,
-prec-sqrt=true, -ftz=false, no fast math), the source scan over all five sources (no function of CUDA's math
library), the PTX check (every add, sub, mul, div, sqrt and rcp `.rn`; no mad, flush or approximation); every constant
from the generated headers: the kinds and routes, the modes' and the orientation's, the tables, and this
module's (`tables_header`): the combat park's two modes, slots and flags, the launch speed and the missile's goto
distance, 2 pi, each read from the reference's modules, and the two literals the torch path writes without a name
(`LITERALS`, each found in its source by a test). The arithmetic is the torch path's, operation for operation; C's
fmod (the yaw and pitch moved within 2 pi of the old values at SetBallVelocity's snap) is computed exactly on the bits.

**Differences in form from the torch path** (the values are the same): the kernel updates the park's tensors and the
server's state in place where the torch path rebinds them; it computes the send flags at the pre-tick (the combat
park's tick changes no ball's interactive flag or presence), so the post-tick does nothing; a tick's queue holds at
most 256 pushes; LaunchMissile's aimed and massive are converted from their float slots by C's truncation (the torch
path's `.to(int64)`: the same for every value an int64 holds, and NaN to 0 on this platform's conversions).

Not here (as CombatServer): VecComposedPark (its collisions: refused), defender missiles (the park flags them),
AddBall and RemoveBall through the server layer, the partition (SpaceServer).
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
from . import vec_server_cuda as V
from . import vec_ypr_cuda as Y

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
from ._cuda_build import SourcePath, cache_root, default_nvcc, read_source  # noqa: E402,F401
SOURCE = SourcePath(Path(__file__).with_name("vec_server_combat_cuda.cu"))
HEADER = "vec_server_combat_cuda_tables.h"
FLAGS = K.FLAGS
TARGETS = V.TARGETS                                  # native: sm_90 and sm_100 cubins, compute_90 PTX; sm90; ptx
ABI = 1
BLOCK = 128
MAX_SLOTS = V.MAX_SLOTS
# planted faults (XBPK_FAULT), the tests only: each compiles one wrong rule into the library
FAULTS = {"lookup_ignores_exists": 1, "stop_without_slot_free": 2, "release_without_missiles": 3,
          "launch_at_rest_unturned": 4, "velocity_without_ypr_snap": 5, "modulo_without_fmod": 6,
          "missile_range_unrounded": 7, "listed_ignores_exists": 8}
# the pending queries' planted fault (kept apart from FAULTS, whose tests run the worlds without an
# undo): the kernel tests again the calls an undo already tested (their mask byte's bit 1 ignored)
PENDING_FAULTS = {"undo_retested": 9}
# the (apart from FAULTS, whose tests run worlds without these): StopAllFollowers' members taken by the
# follow modes, SetBallTroll's delay floor left out, SetBallFree(false) keeping mLastG
G11_FAULTS = {"release_by_follow_modes": 10, "troll_delay_unfloored": 11, "fixed_keeps_last_g": 12}
ALL_FAULTS = {**FAULTS, **PENDING_FAULTS, **G11_FAULTS}
# the torch path's literals without a name: (kind, value, the spelling in the module's source, the module)
LITERALS = {
    "SK_BACK": ("f3", (0.0, 0.0, -1.0), "self.rotated_vector(lsafe, (0.0, 0.0, -1.0))", "vec_combat"),  # at rest
}
_LIBS: dict = {}


def build_dir() -> Path:
    d = os.environ.get("EXACT_BALLPARK_SERVER_COMBAT_CUDA_DIR")
    if d:
        return Path(d)
    return V.build_dir().parent / "server_combat_cuda"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dbits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _fbits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def literals_in_source() -> dict:
    """Each of LITERALS: whether its module still writes it as the header holds it."""
    from . import vec_combat
    mods = {"vec_combat": inspect.getsource(vec_combat)}
    return {name: phrase in mods[mod] for name, (_, _, phrase, mod) in LITERALS.items()}


def table_values() -> dict:
    """Every constant vec_server_combat_cuda.cu uses beyond the included kernels' headers, by its name in the header:
    (kind, value), read from the reference's modules (exact_ballpark.modes, full, vec_combat, vec_orient,
    vec_combat_cuda's spelling-checked literal) or LITERALS; never typed here."""
    from . import full as FU
    from . import modes as MO
    from . import vec_combat as VC
    from . import vec_combat_cuda as KC
    from . import vec_orient as VO
    out = {"SK_MODE_MISSILE": ("int", MO.MISSILE), "SK_MODE_FORMATION": ("int", MO.FORMATION),
           "SK_FORMATION_SLOTS": ("int", FU.FORMATION_MAX_SLOTS), "SK_FLAG_UNDEFINED": ("int", VC.FLAG_UNDEFINED),
           "SK_FLAG_DEFENDER": ("int", VC.FLAG_DEFENDER), "SK_LAUNCH_SPEED": ("d", FU.MISSILE_LAUNCH_SPEED),
           "SK_MISSILE_GOTO": ("d", VC.MISSILE_GOTO), "SK_TWO_PI": ("d", VO.TWO_PI),
           "SK_F_HALF": ("f", KC.COMBAT_LITERALS["F_HALF"][1])}
    for name, (kind, v, _, _) in LITERALS.items():
        if kind == "f3":
            for axis, x in zip("XYZ", v):
                out[f"{name}_{axis}"] = ("f", x)
        else:
            out[name] = (kind, v)
    return out


def tables_header(perturb: dict | None = None) -> str:
    """The C header of `table_values()`; `perturb` ({name: value}) changes one, for the tests only."""
    vals = table_values()
    for k, v in (perturb or {}).items():
        if k not in vals:
            raise KeyError(k)
        vals[k] = (vals[k][0], v)
    lines = ["// generated by exact_ballpark.vec_server_combat_cuda.tables_header from exact_ballpark.modes, full,",
             "// vec_combat, vec_orient and the torch path's own literals: do not edit", "#pragma once", ""]
    for name, (kind, v) in vals.items():
        if kind == "int":
            lines.append(f"#define {name} {int(v)}")
        elif kind == "f":
            lines.append(f"#define {name} 0x{_fbits(v):08x}u")
        else:
            lines.append(f"#define {name} 0x{_dbits(v):016x}ULL")
    return "\n".join(lines) + "\n"


def active_text(text: str, fault: int = 0) -> str:
    """The source as compiled with XBPK_FAULT = fault: each `#if XBPK_FAULT == k` block's lines kept only for k = fault
    (its #else part only otherwise), each `#if XBPK_FAULT != k` block's only otherwise (the planted faults' blocks,
    unnested)."""
    out, state = [], None
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"#if\s+XBPK_FAULT\s*(==|!=)\s*(\d+)", s)
        if m and state is None:
            want = int(m.group(2)) == fault if m.group(1) == "==" else int(m.group(2)) != fault
            state = (want, False)
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


def _sources() -> list:
    """(path, text) of the five sources, this module's first."""
    return [(p, read_source(p)) for p in (SOURCE, V.SOURCE, M.SOURCE, K.SOURCE, Y.SOURCE)]


def _headers(perturb: dict | None = None) -> dict:
    """The generated headers the sources include, by file name."""
    return {HEADER: tables_header(perturb), V.HEADER: V.tables_header(), M.HEADER: M.tables_header(),
            K.HEADER: K.tables_header(), Y.HEADER: Y.tables_header()}


def source_scan(fault: int = 0) -> list:
    """The CUDA math-library calls in the five sources as compiled ([] when clean)."""
    calls = []
    for k, (path, text) in enumerate(_sources()):
        t = active_text(text, fault) if k == 0 else (V.active_text(text, 0) if path == V.SOURCE else text)
        calls += [(n, f"{Path(path).name}:{line}") for n, line in K.scan_source(t)]
    return calls


def build(out_dir: Path | None = None, variant: str = "native", fault: str | None = None, checks: bool = True,
          perturb: dict | None = None) -> dict:
    """Compile vec_server_combat_cuda.cu (with the four sources it includes and their generated headers) into out_dir
    (by default `build_dir()`); returns the manifest written beside the library: the sources' and headers' SHA-256,
    nvcc's version, flags and targets, the kernel's registers and spills, the source scan, the PTX check, the
    library's SHA-256 and the UTC time. A library already built from the same inputs is reused. `fault` (a planted
    fault of FAULTS), `perturb` and `checks=False` (the scan and the check recorded, not enforced) are for the tests."""
    if variant not in TARGETS:
        raise ValueError(f"unknown variant {variant!r}")
    level = 0 if fault is None else ALL_FAULTS[fault]
    out_dir = Path(out_dir) if out_dir is not None else build_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = _sources()
    calls = source_scan(level)
    if checks and calls:
        raise RuntimeError(f"the sources call CUDA's math library: {calls}")
    for found in (V.literals_in_source(), literals_in_source()):
        missing = [k for k, ok in found.items() if not ok]
        if missing:
            raise RuntimeError(f"the torch path no longer writes the literals {missing} as the kernels' headers "
                               f"hold them")
    headers = _headers(perturb)
    nvcc = K._nvcc()
    gencode = [f"-gencode=arch={a},code={c}" for a, c in TARGETS[variant]]
    defines = [f"-DXBPK_FAULT={level}"]
    key = _sha(json.dumps([_sha(t.encode()) for _, t in sources] + [_sha(h.encode()) for h in headers.values()]
                          + [FLAGS, gencode, defines, str(nvcc), K.nvcc_version(nvcc), checks]).encode())[:16]
    stem = f"vec_server_combat_cuda_{variant}_{key}"
    lib, man = out_dir / f"{stem}.so", out_dir / f"{stem}.json"
    if lib.exists() and man.exists():
        m = json.loads(man.read_text())
        if m.get("library_sha256") == _sha(lib.read_bytes()) and (m.get("ptx_check", {}).get("ok") or not checks):
            return m
    work = out_dir / f".{stem}.{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        for name, h in headers.items():
            (work / name).write_text(h)
        for path, text in sources:
            (work / Path(path).name).write_text(text)
        src = work / SOURCE.name
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
            "variant": variant, "sources_sha256": {Path(p).name: _sha(t.encode()) for p, t in sources},
            "headers_sha256": {n: _sha(h.encode()) for n, h in headers.items()}, "fault": fault, "checks": checks,
            "perturb": perturb or {}, "source_scan": calls, "nvcc": version.strip().splitlines()[-1],
            "nvcc_path": str(nvcc), "flags": FLAGS, "defines": defines,
            "targets": [f"{a}->{c}" for a, c in TARGETS[variant]], "embedded": listing, "ptxas": resources,
            "ptx_check": check, "library": lib.name, "library_sha256": _sha((work / "lib.so").read_bytes()),
            "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (work / "lib.so").replace(lib)
        mtmp = work / "manifest.json"
        mtmp.write_text(json.dumps(manifest, indent=1) + "\n")
        mtmp.replace(man)
        return manifest
    finally:
        shutil.rmtree(work, ignore_errors=True)


# VecCombatPark's tensors in vec_server_combat_cuda.cu's `struct Combat`: (park attribute, dtype)
COMBAT_TENSORS = (("exists", BOOL), ("formation_id", I64), ("formation_slots", I64), ("owner_ptr", I64),
                  ("flags", I64), ("yaw", F64), ("pitch", F64), ("roll", F64), ("old_yaw", F64), ("old_pitch", F64),
                  ("old_roll", F64), ("roll_speed", F64), ("old_roll_speed", F64), ("yaw_delta", F64))


class Combat(ctypes.Structure):
    """vec_server_combat_cuda.cu's `struct Combat`."""
    _fields_ = ([(n, ctypes.c_void_p) for n, _ in COMBAT_TENSORS]
                + [("formation_len", ctypes.c_void_p), ("n_formations", ctypes.c_int64), ("master", ctypes.c_int)])


def _open(manifest: dict, out_dir: Path | None = None):
    lib = ctypes.CDLL(str((Path(out_dir) if out_dir is not None else build_dir()) / manifest["library"]))
    lib.xbpk_layout.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    lib.xbpk_layout.restype = ctypes.c_int
    return lib


def layout_check(manifest: dict | None = None, out_dir: Path | None = None) -> dict:
    """The ctypes structures (Combat; the Queue and Srv; the modes' State) against the compiled ones, {name:
    (compiled, ctypes)}: needs no GPU (the library is opened, its module not loaded)."""
    manifest = manifest or build(out_dir)
    lib = _open(manifest, out_dir)
    out = (ctypes.c_int64 * 12)()
    lib.xbpk_layout(out)
    mine = [ctypes.sizeof(Combat), Combat.formation_id.offset, Combat.yaw.offset, Combat.yaw_delta.offset,
            Combat.formation_len.offset, Combat.n_formations.offset, Combat.master.offset, ctypes.sizeof(V.Queue),
            V.Queue.J.offset, ctypes.sizeof(V.Srv), V.Srv.neg_kdt.offset, ctypes.sizeof(M.State)]
    names = ["Combat", "Combat.formation_id", "Combat.yaw", "Combat.yaw_delta", "Combat.formation_len",
             "Combat.n_formations", "Combat.master", "Queue", "Queue.J", "Srv", "Srv.neg_kdt", "Modes"]
    return {n: (int(out[k]), int(mine[k])) for k, n in enumerate(names)}


def load_library(variant: str = "native", fault: str | None = None, checks: bool = True) -> dict:
    """The compiled library (built if missing or stale), its functions typed, its kernel's module loaded on the current
    CUDA device (`xbpk_info`: no later launch loads it, as a captured one must not), and its manifest; once a
    process."""
    k = (variant, fault, checks)
    if k in _LIBS:
        return _LIBS[k]
    manifest = build(None, variant, fault, checks)
    lib = _open(manifest)
    if lib.xbpk_abi() != ABI:
        raise RuntimeError(f"the library's ABI {lib.xbpk_abi()} is not {ABI}")
    if lib.xbpk_fault() != (0 if fault is None else ALL_FAULTS[fault]):
        raise RuntimeError("the library's planted-fault switch is not the one asked for")
    p, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.xbpk_error_string.restype = ctypes.c_char_p
    lib.xbpk_error_string.argtypes = [i32]
    lib.xbpk_pre_tick.argtypes = [ctypes.POINTER(M.State), ctypes.POINTER(V.Park), ctypes.POINTER(V.Queue),
                                  ctypes.POINTER(V.Srv), ctypes.POINTER(Combat), i32, i64, i64, p, i32]
    lib.xbpk_pre_tick.restype = i32
    lib.xbpk_info.argtypes = [ctypes.POINTER(i32)]
    lib.xbpk_info.restype = i32
    entry = {"lib": lib, "manifest": manifest}
    out = (ctypes.c_int * 5)()
    rc = lib.xbpk_info(out)
    if rc:
        raise RuntimeError(f"xbpk_info failed ({rc}: {lib.xbpk_error_string(rc).decode()})")
    entry["info"] = dict(zip(("registers", "local_bytes", "max_threads_per_block", "binary_version", "ptx_version"),
                             list(out)))
    _LIBS[k] = entry
    return entry


def _ptr(t: torch.Tensor, dtype, shape: tuple, what: str) -> int:
    return V._ptr(t, dtype, shape, what)


def combat_struct(park) -> Combat:
    """The kernel's view of VecCombatPark's own tensors (each checked: dtype, shape, a CUDA device, contiguous)."""
    R = park.n
    c = Combat(*(_ptr(getattr(park, n), dt, (R,), f"park.{n}") for n, dt in COMBAT_TENSORS))
    c.formation_len = _ptr(park.formation_lengths, I64, (max(1, park.n_formations),), "park.formation_lengths")
    c.n_formations, c.master = int(park.n_formations), int(park.master)
    return c


def launch(server, q: dict, phase: int, j0: int = 0, j1: int = 0, block: int | None = None,
           lib: dict | None = None) -> None:
    """One launch of the kernel on torch's current stream (vec_server_cuda.launch's arguments: `q` holds the tick's
    stacked pushes and the outputs); the park's and the server's state are read and written in place. No allocation,
    no host synchronisation."""
    e = lib or server._lib
    block = (server.block if getattr(server, "block", None) is not None else BLOCK) if block is None else int(block)
    if not 1 <= block <= e["info"]["max_threads_per_block"]:
        raise ValueError(f"block {block}: the kernel takes 1 to {e['info']['max_threads_per_block']} threads a block "
                         f"(its registers)")
    a, park = server.a, server.a.park
    W, R, N = server.W, server.R, a.rows_of_world.shape[1]
    J = len(q["kinds"])
    if J > MAX_SLOTS:
        raise ValueError(f"{J} pushes in a tick: the kernel takes at most {MAX_SLOTS}")
    m = M.make_state(park)
    p = V.Park(*(_ptr(getattr(park, n), I64, (R,), f"park.{n}") for n in ("cloaked", "bubble", "refiled")))
    qs = V.Queue(_ptr(q["mask"], torch.uint8, (J, W), "mask"), _ptr(q["ball"], I64, (J, W), "ball"),
                 _ptr(q["target"], I64, (J, W), "target"), _ptr(q["f"], F64, (J, W, 4), "f"),
                 _ptr(q["n"], I64, (J, W), "n"), _ptr(q["loc_in"], I64, (J, W), "loc_in"),
                 _ptr(q["ok"], BOOL, (J, W), "ok"), _ptr(q["row"], I64, (J, W), "row"),
                 _ptr(q["trow"], I64, (J, W), "trow"), _ptr(q["loc"], I64, (J, W), "loc"),
                 _ptr(q["route"], torch.int8, (W, J), "route"), _ptr(q["notice"], BOOL, (J, R), "notice"), J)
    for j, k in enumerate(q["kinds"]):
        qs.kind[j] = int(k)
    s = V.Srv(_ptr(a.rows_of_world, I64, (W, N), "rows_of_world"), None,     # ids_of_world: unread (the park's ids)
              _ptr(server.pilot, I64, (R,), "pilot"), _ptr(server.stamp_for_system, I64, (W,), "stamp_for_system"),
              _ptr(q["stamp"], I64, (W,), "stamp"), _ptr(q["clock"], I64, (W,), "clock"),
              _ptr(server._bubble_time, I64, (W,), "bubble_time"), _ptr(server.settled, BOOL, (R,), "settled"),
              _ptr(server._fresh, BOOL, (R,), "fresh"), _ptr(server._listed, BOOL, (R,), "listed"),
              _ptr(q["has_character"], BOOL, (R,), "has_character"),
              _ptr(q["gets_bubble"], BOOL, (R,), "gets_bubble"), W, N, R, -park.friction * park.dt, int(a.master))
    c = combat_struct(park)
    with torch.cuda.device(park.dev):
        rc = e["lib"].xbpk_pre_tick(ctypes.byref(m), ctypes.byref(p), ctypes.byref(qs), ctypes.byref(s),
                                    ctypes.byref(c), int(phase), int(j0), int(j1),
                                    torch.cuda.current_stream(park.dev).cuda_stream, int(block))
    if rc != 0:
        msg = e["lib"].xbpk_error_string(rc).decode() if rc > 0 else "refused (the slots, the block or the phase)"
        raise RuntimeError(f"xbpk_pre_tick failed ({rc}: {msg})")

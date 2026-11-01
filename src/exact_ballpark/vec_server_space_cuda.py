# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer over the bubble partition as one CUDA kernel a tick, one thread a world:
the pre-tick of `exact_ballpark.vec_server_space.SpaceServer` over VecModesPark (the follower clean-up's Stops, the
queue's duplicate test and lookups, CloakBall's captured bubble, every queued action applied in queue order by the
park's own rules with the partition's re-filings and sets after each, the bubble update once per park time with its
lists, the history's routes per bubble with CloakBall applied there, the uncloaks' notices) as one launch, and its
post-tick send flags as another, where the torch path runs some hundreds of small launches and asks the host for the
clean-up's count each tick. `CudaSpaceServer` is SpaceServer with the kernel (SpaceServer, the torch path, unchanged,
stays the exactness reference); `SpaceServerGraph` captures its tick as a CUDA graph. Checked bit for bit against the
torch path, and through it against exact_ballpark.server, by tests/test_movement_vec_server_space_cuda.py.

The source is vec_server_space_cuda.cu beside this module. It includes vec_server_cuda.cu (the server
kernel: VecModesPark's commands and setters, through vec_modes_cuda.cu and vec_orient_cuda.cu) and vec_space.cu
(the partition: its operations run as its tape runs them), unedited. The library is built on first use
(or when a source, a generated header, the flags, the planted-fault switch or nvcc change) into a build directory
outside any source tree ($EXACT_BALLPARK_SERVER_SPACE_CUDA_DIR; by default beside vec_server_cuda's, `server_space_cuda`), with a
manifest; it is loaded through ctypes, its kernel's module loaded at once (`xbpq_info`), and launched on torch's
current stream with no allocation and no host synchronisation, so a tick can be captured in a CUDA graph.

**Exactness by construction**, as the two kernels it includes: their flags (nvcc -fmad=false, -prec-div=true,
-prec-sqrt=true, -ftz=false, no fast math; the host compiler -ffp-contract=off), the source scan over all five
sources (no function of CUDA's math library), the PTX check (every add, sub, mul, div, sqrt and rcp `.rn`; no mad,
flush or approximation) and the partition's host check (no fused instruction in its host code); every constant from
the generated headers (vec_server_cuda's kinds and routes, vec_space's operations and the bubble update's kinds,
vec_modes_cuda's and vec_orient_cuda's). The kernel adds index work only: the setters' acceptance test is the torch
path's (a float argument rounded to float, compared with zero).

**Differences in form from the torch path** (the values are the same):
- the clean-up's Stops take `cleanup_slots` fixed slots behind the pushed calls (by default the world's N, the most a
  world can need), where the torch path pushes as many as the most a world has this tick (the host reads it); the
  slots past a world's count are not queued (ok False, routed nowhere) and `decode` skips them;
- the kernel updates the park's and the partition's tensors in place where the torch path rebinds some of them;
- a slot's row, target row and local argument are compared where it was queued, and a route's bubble where the route
  is BUBBLE (the torch path computes them for unqueued calls from another world's first ball, which means nothing);
- a tick's queue holds at most 256 slots, the clean-up's included.

Not here (as SpaceServer): VecOrientPark (its torch path stays), make_ball_global and make_ball_local (the kinds are
refused: this port left them to the torch path), removals, missiles, the game's SetState. global
balls at AddBall (the flags held by the partition layer's view, `space_for(adapter, global_balls)`) are read by the
partition's operations in the kernel (the Park structure's flag pointer, as on the torch path), and the snapshots for
`decode` are SpaceServer's (`_stream_fields`: DSTBALL_ISGLOBAL in the streams); the kernel's source is unchanged.
OnTick's per-world mask is refused here (`on_tick`): the kernel runs each world's follower clean-up and
bubble update in its pre-tick, so a world without a tick would be changed by it (a skip of the worlds outside the
mask belongs in the kernel, in its own build); SpaceServer, the torch path, runs it over VecModesPark.
"""

from __future__ import annotations

import ctypes
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import torch

from . import vec_modes_cuda as M
from . import vec_orient_cuda as K
from . import vec_server as VS
from . import vec_server_cuda as V
from . import vec_server_space as VSS
from . import vec_space as SP

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
from ._cuda_build import SourcePath, cache_root, default_nvcc, read_source  # noqa: E402,F401
SOURCE = SourcePath(Path(__file__).with_name("vec_server_space_cuda.cu"))
FLAGS = K.FLAGS
TARGETS = V.TARGETS                                  # native: sm_90 and sm_100 cubins, compute_90 PTX; sm90; ptx
ABI = 1
BLOCK = 128
MAX_SLOTS = V.MAX_SLOTS
# planted faults (XBPQ_FAULT), the tests only: each compiles one wrong rule into the library
FAULTS = {"no_cleanup": 1, "cloak_current_bubble": 2, "no_refile": 3, "dead_bubble_routed": 4,
          "cloaked_to_the_bubble": 5, "notices_world_wide": 6, "narrow_by_bubble": 7}
# the pending queries' planted fault (kept apart from FAULTS, whose tests run the worlds without an
# undo): the kernel tests again the calls an undo already tested (their mask byte's bit 1 ignored)
PENDING_FAULTS = {"undo_retested": 8}
# the (apart from FAULTS, whose tests run worlds without SetBallFree): SetBallFree(false)'s re-filing
G11_FAULTS = {"free_not_refiled": 9}
ALL_FAULTS = {**FAULTS, **PENDING_FAULTS, **G11_FAULTS}
_LIBS: dict = {}


def build_dir() -> Path:
    d = os.environ.get("EXACT_BALLPARK_SERVER_SPACE_CUDA_DIR")
    if d:
        return Path(d)
    return V.build_dir().parent / "server_space_cuda"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def active_text(text: str, fault: int = 0) -> str:
    """The source as compiled with XBPQ_FAULT = fault: each `#if XBPQ_FAULT == k` block's lines kept only for
    k = fault, its #else part only otherwise (the planted faults' blocks, unnested)."""
    out, state = [], None
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"#if\s+XBPQ_FAULT\s*==\s*(\d+)", s)
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


def _sources() -> list:
    """(path, text) of the five sources, this module's first."""
    return [(p, read_source(p)) for p in (SOURCE, V.SOURCE, SP.SOURCE, M.SOURCE, K.SOURCE)]


def _headers() -> dict:
    """The generated headers the sources include, by file name."""
    return {V.HEADER: V.tables_header(), SP.HEADER: SP.tables_header(), M.HEADER: M.tables_header(),
            K.HEADER: K.tables_header()}


def build(out_dir: Path | None = None, variant: str = "native", fault: str | None = None,
          checks: bool = True) -> dict:
    """Compile vec_server_space_cuda.cu (with the four sources it includes and their generated headers) into out_dir
    (by default `build_dir()`); returns the manifest written beside the library: the sources' and headers' SHA-256,
    nvcc's version, flags and targets, the kernel's registers and spills, the source scan, the PTX check, the host
    check, the library's SHA-256 and the UTC time. A library already built from the same inputs is reused. `fault` (a
    planted fault of FAULTS) and `checks=False` (the scan and the checks recorded, not enforced) are for the tests."""
    if variant not in TARGETS:
        raise ValueError(f"unknown variant {variant!r}")
    level = 0 if fault is None else ALL_FAULTS[fault]
    out_dir = Path(out_dir) if out_dir is not None else build_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = _sources()
    calls = []
    for k, (path, text) in enumerate(sources):
        t = active_text(text, level) if k == 0 else (V.active_text(text, 0) if path == V.SOURCE else text)
        calls += [(n, f"{Path(path).name}:{line}") for n, line in K.scan_source(t)]
    if checks and calls:
        raise RuntimeError(f"the sources call CUDA's math library: {calls}")
    missing = [k for k, ok in V.literals_in_source().items() if not ok]
    if missing:
        raise RuntimeError(f"the torch path no longer writes the literals {missing} as the server kernel's header has")
    headers = _headers()
    nvcc = K._nvcc()
    gencode = [f"-gencode=arch={a},code={c}" for a, c in TARGETS[variant]]
    defines = [f"-DXBPQ_FAULT={level}"]
    key = _sha(json.dumps([_sha(t.encode()) for _, t in sources] + [_sha(h.encode()) for h in headers.values()]
                          + [FLAGS, gencode, defines, str(nvcc), K.nvcc_version(nvcc), checks]).encode())[:16]
    stem = f"vec_server_space_cuda_{variant}_{key}"
    lib, man = out_dir / f"{stem}.so", out_dir / f"{stem}.json"
    if lib.exists() and man.exists():
        m = json.loads(man.read_text())
        if m.get("library_sha256") == _sha(lib.read_bytes()) and (
                (m.get("ptx_check", {}).get("ok") and m.get("host_check", {}).get("ok")) or not checks):
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
        hcheck = SP.host_check(work / "lib.so")
        if checks and not check["ok"]:
            raise RuntimeError(f"the PTX has unpinned floating-point instructions: {check['refused']}")
        if checks and not hcheck["ok"]:
            raise RuntimeError(f"the host code has fused floating-point instructions: {hcheck}")
        resources = [ln.strip() for ln in (r.stdout + r.stderr).splitlines()
                     if "registers" in ln or "spill" in ln or "Compiling entry" in ln or "stack frame" in ln]
        version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True).stdout
        manifest = {
            "variant": variant, "sources_sha256": {Path(p).name: _sha(t.encode()) for p, t in sources},
            "headers_sha256": {n: _sha(h.encode()) for n, h in headers.items()}, "fault": fault, "checks": checks,
            "source_scan": calls, "nvcc": version.strip().splitlines()[-1], "nvcc_path": str(nvcc), "flags": FLAGS,
            "defines": defines, "targets": [f"{a}->{c}" for a, c in TARGETS[variant]], "embedded": listing,
            "ptxas": resources, "ptx_check": check, "host_check": hcheck, "library": lib.name,
            "library_sha256": _sha((work / "lib.so").read_bytes()),
            "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (work / "lib.so").replace(lib)
        mtmp = work / "manifest.json"
        mtmp.write_text(json.dumps(manifest, indent=1) + "\n")
        mtmp.replace(man)
        return manifest
    finally:
        shutil.rmtree(work, ignore_errors=True)


class Extra(ctypes.Structure):
    """vec_server_space_cuda.cu's `struct Extra`."""
    _fields_ = ([(n, ctypes.c_void_p) for n in ("ad_time", "cleanup", "cleanup_total", "rbubble", "ad_kind", "ad_adds",
                                                 "ad_addn", "ad_dels", "ad_deln", "narrow", "slot_of_row")]
                + [("J_user", ctypes.c_int64)])


def _open(manifest: dict, out_dir: Path | None = None):
    lib = ctypes.CDLL(str((Path(out_dir) if out_dir is not None else build_dir()) / manifest["library"]))
    lib.xbpq_layout.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    lib.xbpq_layout.restype = ctypes.c_int
    return lib


def layout_check(manifest: dict | None = None, out_dir: Path | None = None) -> dict:
    """The ctypes structures (Extra; the Queue and Srv) against the compiled ones, {name: (compiled, ctypes)}:
    needs no GPU (the library is opened, its module not loaded)."""
    manifest = manifest or build(out_dir)
    lib = _open(manifest, out_dir)
    out = (ctypes.c_int64 * 17)()
    lib.xbpq_layout(out)
    mine = [ctypes.sizeof(Extra)] + [getattr(Extra, n).offset for n, _ in Extra._fields_] + [
        ctypes.sizeof(V.Queue), V.Queue.J.offset, ctypes.sizeof(V.Srv), V.Srv.neg_kdt.offset]
    names = ["Extra"] + [f"Extra.{n}" for n, _ in Extra._fields_] + ["Queue", "Queue.J", "Srv", "Srv.neg_kdt"]
    return {n: (int(out[k]), int(mine[k])) for k, n in enumerate(names)}


def load_library(variant: str = "native", fault: str | None = None, checks: bool = True) -> dict:
    """The compiled library (built if missing or stale), its functions typed, its kernel's module loaded on the current
    CUDA device (`xbpq_info`: no later launch loads it, as a captured one must not), and its manifest; once a
    process."""
    k = (variant, fault, checks)
    if k in _LIBS:
        return _LIBS[k]
    manifest = build(None, variant, fault, checks)
    lib = _open(manifest)
    if lib.xbpq_abi() != ABI:
        raise RuntimeError(f"the library's ABI {lib.xbpq_abi()} is not {ABI}")
    if lib.xbpq_fault() != (0 if fault is None else ALL_FAULTS[fault]):
        raise RuntimeError("the library's planted-fault switch is not the one asked for")
    p, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.xbpq_error_string.restype = ctypes.c_char_p
    lib.xbpq_error_string.argtypes = [i32]
    lib.xbpq_pre_tick.argtypes = [ctypes.POINTER(M.State), ctypes.POINTER(V.Park), ctypes.POINTER(V.Queue),
                                  ctypes.POINTER(V.Srv), ctypes.POINTER(SP._Park), ctypes.POINTER(SP._Space),
                                  ctypes.POINTER(Extra), i32, i64, i64, p, i32]
    lib.xbpq_pre_tick.restype = i32
    lib.xbpq_info.argtypes = [ctypes.POINTER(i32)]
    lib.xbpq_info.restype = i32
    entry = {"lib": lib, "manifest": manifest}
    out = (ctypes.c_int * 5)()
    rc = lib.xbpq_info(out)
    if rc:
        raise RuntimeError(f"xbpq_info failed ({rc}: {lib.xbpq_error_string(rc).decode()})")
    entry["info"] = dict(zip(("registers", "local_bytes", "max_threads_per_block", "binary_version", "ptx_version"),
                             list(out)))
    _LIBS[k] = entry
    return entry


def _ptr(t: torch.Tensor, dtype, shape: tuple, what: str) -> int:
    return V._ptr(t, dtype, shape, what)


def launch(server: "CudaSpaceServer", q: dict, phase: int, j0: int = 0, j1: int = 0, block: int | None = None,
           lib: dict | None = None) -> None:
    """One launch of the kernel on torch's current stream: `q` holds the tick's stacked slots and the outputs (the
    queue and flags, and this layer's: rbubble [W, J], the bubble update's lists per row, narrow [R]); the park's, the
    partition's and the server's state are read and written in place. No allocation, no host synchronisation."""
    e = lib or server._lib
    block = (server.block if server.block is not None else BLOCK) if block is None else int(block)
    if not 1 <= block <= e["info"]["max_threads_per_block"]:
        raise ValueError(f"block {block}: the kernel takes 1 to {e['info']['max_threads_per_block']} threads a block "
                         f"(its registers)")
    a, park, sp = server.a, server.a.park, server.space
    W, R, N = server.W, server.R, a.rows_of_world.shape[1]
    J = len(q["kinds"])
    if J > MAX_SLOTS:
        raise ValueError(f"{J} slots in a tick (the clean-up's {server.C} included): the kernel takes at most "
                         f"{MAX_SLOTS}")
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
    s = V.Srv(_ptr(a.rows_of_world, I64, (W, N), "rows_of_world"), _ptr(a.ids_of_world, I64, (W, N), "ids_of_world"),
              _ptr(server.pilot, I64, (R,), "pilot"), _ptr(server.stamp_for_system, I64, (W,), "stamp_for_system"),
              _ptr(q["stamp"], I64, (W,), "stamp"), _ptr(q["clock"], I64, (W,), "clock"),
              _ptr(server._bubble_time, I64, (W,), "bubble_time"), _ptr(server.settled, BOOL, (R,), "settled"),
              _ptr(server._fresh, BOOL, (R,), "fresh"), _ptr(server._listed, BOOL, (R,), "listed"),
              _ptr(q["has_character"], BOOL, (R,), "has_character"),
              _ptr(q["gets_bubble"], BOOL, (R,), "gets_bubble"), W, N, R, -park.friction * park.dt, int(a.master))
    x = Extra(_ptr(server._ad_time, I64, (W,), "ad_time"), _ptr(server.cleanup, I64, (W, 1 + N), "cleanup"),
              _ptr(server.cleanup_total, I64, (W,), "cleanup_total"), _ptr(q["rbubble"], I64, (W, J), "rbubble"),
              _ptr(q["ad_kind"], I64, (R,), "ad_kind"), _ptr(q["ad_adds"], I64, (R, N), "ad_adds"),
              _ptr(q["ad_addn"], I64, (R,), "ad_addn"), _ptr(q["ad_dels"], I64, (R, N), "ad_dels"),
              _ptr(q["ad_deln"], I64, (R,), "ad_deln"), _ptr(q["narrow"], I64, (R,), "narrow"),
              _ptr(server.slot_of_row, I64, (R,), "slot_of_row"), int(q["J_user"]))
    sp_park, keep = sp._park_struct()
    sp_space = sp._space_struct()
    with torch.cuda.device(park.dev):
        rc = e["lib"].xbpq_pre_tick(ctypes.byref(m), ctypes.byref(p), ctypes.byref(qs), ctypes.byref(s),
                                    ctypes.byref(sp_park), ctypes.byref(sp_space), ctypes.byref(x), int(phase),
                                    int(j0), int(j1), torch.cuda.current_stream(park.dev).cuda_stream, int(block))
    del keep
    if rc != 0:
        msg = e["lib"].xbpq_error_string(rc).decode() if rc > 0 else "refused (the slots, the block, the phase, sizes)"
        raise RuntimeError(f"xbpq_pre_tick failed ({rc}: {msg})")


# ===================================================================================== the server on the kernel

class CudaSpaceServer(VSS.SpaceServer):
    """SpaceServer over VecModesPark on a CUDA device with the kernel: `push` queues (the duplicate test, the lookups
    and CloakBall's bubble run in the kernel, at the pre-tick, from the partition as it stands before anything of the
    tick is applied, which is as it stood when the calls were pushed); `pre_tick` is one launch (with snapshots for
    `decode`, as phases); `evolve` is SpaceServer's (exact_ballpark.vec_space's tick); `post_tick` is one launch.
    `cleanup_slots`: the clean-up's fixed slots (by default the world's N). `batch` is the tick's SpaceBatch, its
    clean-up slots all present (those past a world's count not queued). `fault`: a planted fault's library (the tests
    only)."""

    def __init__(self, adapter, pilots=None, space: SP.VecSpace | None = None, cleanup_slots: int | None = None,
                 block: int | None = None, fault: str | None = None):
        if not (isinstance(adapter, VSS.SpaceModesAdapter) and adapter.dev.type == "cuda"):
            raise ValueError("CudaSpaceServer runs the server layer of a VecModesPark (SpaceModesAdapter) on a CUDA "
                             "device")
        super().__init__(adapter, pilots, space)
        N = int(self.a.rows_of_world.shape[1])
        self.C = N if cleanup_slots is None else int(cleanup_slots)
        if not 0 <= self.C <= MAX_SLOTS:
            raise ValueError(f"cleanup_slots {self.C}: 0 to {MAX_SLOTS}")
        if self.C < N:                                      # a world may need N Stops
            raise ValueError(f"cleanup_slots {self.C}: at least the world's {N} slots (each ball may follow away)")
        self.block = block
        d = self.dev
        self.cleanup = torch.zeros(self.W, 1 + N, dtype=I64, device=d)
        self.cleanup_total = torch.zeros(self.W, dtype=I64, device=d)
        with torch.cuda.device(d):
            self._lib = load_library(variant="native" if fault is None else "sm90", fault=fault,
                                     checks=fault is None)

    @property
    def cleanup_stops(self) -> int:
        """The clean-up's Stops so far (SpaceServer's count; a host read)."""
        return int(self.cleanup_total.sum()) if hasattr(self, "cleanup_total") else 0

    @cleanup_stops.setter
    def cleanup_stops(self, value: int) -> None:            # SpaceServer's constructor sets 0
        if int(value) != 0:
            raise ValueError("the clean-up's count is the kernel's")

    def push(self, kind, mask, ball, target=None, f=None, n=None, loc=None):
        """An Actions call for every world in mask [W] (as VecServer.push), queued for the kernel."""
        k = VS.CODE[kind] if isinstance(kind, str) else int(kind)
        if k not in self.a.kinds:
            raise ValueError(f"{VS.KINDS[k] or kind} is not applied by the park ({type(self.a).__name__})")
        W, d = self.W, self.dev
        z = lambda dt: torch.zeros(W, dtype=dt, device=d)                    # noqa: E731
        mask = mask.to(BOOL).to(d)
        ball = ball.to(I64).to(d)
        target = z(I64) if target is None else target.to(I64).to(d)
        ff = torch.zeros(W, 4, dtype=F64, device=d)
        if f is not None:
            f = f.to(F64).to(d)
            ff[:, :f.shape[1]] = f.reshape(W, -1)
        n = z(I64) if n is None else n.to(I64).to(d)
        loc = z(I64) if loc is None else loc.to(I64).to(d)
        slot = VS.Slot(k, None, ball, None, target, None, ff, n, loc)
        slot.mask = mask
        self.slots.append(slot)
        return slot

    def _queue(self, slots: list) -> dict:
        """The tick's slots stacked (the pushed calls, then the clean-up's Stops) with the kernel's outputs."""
        W, R, d = self.W, self.R, self.dev
        N = int(self.a.rows_of_world.shape[1])
        Ju, C = len(slots), self.C
        J = Ju + C
        if J > MAX_SLOTS:
            raise ValueError(f"{Ju} pushes and {C} clean-up slots in a tick: the kernel takes at most {MAX_SLOTS}")

        def stack(attr, dtype, tail=()):
            out = torch.zeros(J, W, *tail, dtype=dtype, device=d)
            if Ju:
                out[:Ju] = torch.stack([getattr(s, attr) for s in slots]).to(dtype)
            return out

        mask = torch.zeros(J, W, dtype=torch.uint8, device=d)     # the pushes' mask bytes (VS.mask_bytes)
        if Ju:
            mask[:Ju] = VS.mask_bytes(slots, W, d)
        return {"kinds": [s.kind for s in slots] + [VSS.STOP] * C, "J_user": Ju,
                "mask": mask, "ball": stack("ball", I64), "target": stack("target", I64),
                "f": stack("f", F64, (4,)), "n": stack("n", I64), "loc_in": stack("loc", I64),
                "ok": torch.zeros(J, W, dtype=BOOL, device=d), "row": torch.zeros(J, W, dtype=I64, device=d),
                "trow": torch.zeros(J, W, dtype=I64, device=d), "loc": torch.zeros(J, W, dtype=I64, device=d),
                "route": torch.zeros(W, J, dtype=torch.int8, device=d),
                "notice": torch.zeros(J, R, dtype=BOOL, device=d), "stamp": torch.zeros(W, dtype=I64, device=d),
                "clock": torch.zeros(W, dtype=I64, device=d), "has_character": torch.zeros(R, dtype=BOOL, device=d),
                "gets_bubble": torch.zeros(R, dtype=BOOL, device=d),
                "rbubble": torch.full((W, J), -1, dtype=I64, device=d),
                "ad_kind": torch.zeros(R, dtype=I64, device=d), "ad_adds": torch.full((R, N), -1, dtype=I64, device=d),
                "ad_addn": torch.zeros(R, dtype=I64, device=d), "ad_dels": torch.full((R, N), -1, dtype=I64, device=d),
                "ad_deln": torch.zeros(R, dtype=I64, device=d), "narrow": torch.full((R,), -1, dtype=I64, device=d)}

    def pre_tick(self, snapshot: bool | None = None) -> None:
        """DoPreTick's Ticker.tick with the partition, as the kernel: one launch, or, with snapshots (for `decode`; the
        default, as SpaceServer's), the commands and the bubble update, then the history split at each UncloakBall
        (the streams they send are written there)."""
        slots, self.slots = self.slots, []
        self._klast = None                                  # the next tick's queue starts empty (the undo's state)
        q = self._queue(slots)
        kinds = q["kinds"]
        if snapshot is None:
            snapshot = True
        self._seen_first = True
        J = len(kinds)
        notices, ad_fields = [], None
        if not snapshot:
            launch(self, q, 0)
            notices = [(j, q["notice"][j], None) for j, k in enumerate(kinds) if k == VS.UNCLOAK]
        else:
            launch(self, q, 1)
            ad_fields = self._stream_fields()               # (SpaceServer's, with the view's global flags)
            j0 = 0
            for j in [j for j, k in enumerate(kinds) if k == VS.UNCLOAK]:
                launch(self, q, 2, j0, j)
                notices.append((j, q["notice"][j], self._stream_fields()))
                j0 = j
            launch(self, q, 2, j0, J)
        W, d = self.W, self.dev
        z = torch.zeros(W, 4, dtype=F64, device=d)
        out = []
        for j, k in enumerate(kinds):
            if j < q["J_user"]:
                s = slots[j]
            else:                                           # a clean-up Stop: its ball written by the kernel
                s = VS.Slot(k, None, q["ball"][j], None, torch.zeros(W, dtype=I64, device=d), None, z,
                            torch.zeros(W, dtype=I64, device=d), torch.zeros(W, dtype=I64, device=d))
            s.ok, s.row, s.trow, s.loc = q["ok"][j], q["row"][j], q["trow"][j], q["loc"][j]
            out.append(s)
        ad = {"kind": q["ad_kind"], "adds": q["ad_adds"], "addn": q["ad_addn"], "dels": q["ad_dels"],
              "deln": q["ad_deln"]}
        self._q = q
        self.batch = VSS.SpaceBatch(stamp=q["stamp"], clock=q["clock"], slots=out, route=q["route"],
                                    notices=notices, additions=None, rbubble=q["rbubble"], ad=ad, ad_fields=ad_fields)

    def post_tick(self) -> None:
        """DoPostTick's send_batch after Evolve, as the kernel: per row (its pilot), the bubble whose narrowcast it gets
        and whether it has a character update."""
        q = self._q
        launch(self, q, 3)
        b = self.batch
        b.has_character, b.gets_bubble, b.narrow = q["has_character"], q["gets_bubble"], q["narrow"]

    def on_tick(self, sim_time) -> list:
        """Refused: OnTick's per-world mask is not on this kernel, whose pre-tick runs each world's
        follower clean-up and bubble update itself (a world without a tick would be changed by it); SpaceServer, the
        torch path, runs it over this adapter."""
        raise ValueError("OnTick's per-world mask is not on the space server's kernel (CudaSpaceServer): SpaceServer, "
                         "the torch path, runs on_tick over VecModesPark")


# ===================================================================================== a tick as a CUDA graph

class SpaceServerGraph:
    """A CudaSpaceServer's tick with a fixed schedule of pushes, captured once as a CUDA graph: before each replay the
    caller writes each push's arguments into `inputs[j]` (mask, ball, target, f, n, loc); `tick()` replays the pushes,
    the pre-tick's launch, the partition's tick with the park's kernel and the post-tick's launch, and returns the
    graph's SpaceBatch (its tensors refilled; no snapshots: `decode` needs an eager tick). Every launch updates the
    park's, the partition's and the server's tensors in place, so the state carries from replay to replay; an eager
    operation between replays must too: one that rebinds a tensor of the park (an eager setter of VecModesPark)
    invalidates the graph (`stale()`, and `tick()` refuses). Exact by construction (the same launches); held to eager
    kernel ticks bit for bit by the tests."""

    def __init__(self, server: CudaSpaceServer, schedule: list):
        if not isinstance(server, CudaSpaceServer):
            raise ValueError("SpaceServerGraph captures a CudaSpaceServer's tick")
        if server.slots:
            raise ValueError("pushes are queued: tick them eagerly first")
        self.server, self.schedule = server, [VS.CODE[k] if isinstance(k, str) else int(k) for k in schedule]
        W, d = server.W, server.dev
        self.inputs = [{"mask": torch.zeros(W, dtype=BOOL, device=d), "ball": torch.zeros(W, dtype=I64, device=d),
                        "target": torch.zeros(W, dtype=I64, device=d), "f": torch.zeros(W, 4, dtype=F64, device=d),
                        "n": torch.zeros(W, dtype=I64, device=d), "loc": torch.zeros(W, dtype=I64, device=d)}
                       for _ in self.schedule]
        park = server.a.park
        with torch.cuda.device(d):                          # every kernel's module loaded before the capture (a
            M.kernel_info(lib=M.load_library())             # launch inside it must not load one)
            SP.kernel_info(lib=server.space.lib())
        self._ptrs = self._pointers()
        torch.cuda.synchronize(d)
        self.graph = torch.cuda.CUDAGraph()
        t0 = park.current_time
        with torch.cuda.graph(self.graph):
            for k, inp in zip(self.schedule, self.inputs):
                server.push(k, inp["mask"], inp["ball"], inp["target"], inp["f"], inp["n"], inp["loc"])
            server.pre_tick(snapshot=False)
            server.evolve()
            server.post_tick()
        park.current_time = t0                              # the capture ran nothing: the state is unchanged
        server.slots = []
        self.out = server.batch
        if self._pointers() != self._ptrs:
            raise RuntimeError("a tensor of the park or the server was rebound during the capture")

    def _pointers(self) -> tuple:
        s = self.server
        objs = (s.a.park, s)
        return tuple((id(o), n, v.data_ptr()) for o in objs for n, v in sorted(vars(o).items())
                     if torch.is_tensor(v)) + tuple((n, v.data_ptr()) for n, v in sorted(s.space.st.items())
                                                    if torch.is_tensor(v))

    def stale(self) -> bool:
        return self._pointers() != self._ptrs

    def tick(self) -> VSS.SpaceBatch:
        if self.server.slots:
            raise RuntimeError("eager pushes before a graph tick: the graph's tick pushes its own schedule")
        if self.stale():
            raise RuntimeError("a tensor of the park or the server was rebound since the capture: capture again")
        self.graph.replay()
        self.server.a.park.current_time += 1
        return self.out

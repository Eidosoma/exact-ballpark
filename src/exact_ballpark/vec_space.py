# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's space on the master batched: the partition's filing and the bubble partition (bubbles
created, joined, left and removed as ships move, in Destiny's order, the re-initialisation quirk included), the
proximity sensors with target tracking and the notification range (their events in Destiny's order), and the
visibility and range queries, for W independent worlds of up to N balls each, around a batched park
(exact_ballpark.vec_modes.VecModesPark, exact_ballpark.vec_orient.VecOrientPark, or exact_ballpark.vec_collide.VecCollidePark
with its collisions kept within bubbles: `CollideView`), on the CPU or a CUDA device.

It must reproduce `exact_ballpark.full.FullBallpark` with `is_master=True` (its partition, bubbles and sensors are
space.py's, `SpaceBallpark`) bit for bit, which is held bit for bit to Destiny's own code compiled verbatim
(tests/test_movement_space_oracle.py, tests/test_movement_full_oracle.py), so it is exact against Destiny (Carbon's
`destiny` at 114e89fa, MIT, (c) CCP Games). Checked by
tests/test_movement_vec_space.py on generated worlds, on the CPU and the GPU. Rules, by rule id:

- SEN-705, SEN-706: each ball's filing (the level of its inflated radius, the key of the box holding its centre, the
  sides its inflated sphere reaches), re-filed where Destiny re-files it: AddBall, SetBallRadius, SetMaxSpeed,
  SetBallPosition (`refile`), a warp's drop-out (phase 3) and the commit of a ball that moved at least sqrt(0.1) m
  (phase 4, ascending id; `after_evolve`), and the sensor's two re-filings.
- SEN-707: on the master a ball entering a top box without a bubble takes the first neighbour's (26, i-j-k order)
  or a new id (`Ballpark::InsertInBoxes`, Ballpark.cpp:3115); a top box exists while a ball's filing reaches it, and
  a released top loses its bubble (so a lone ship takes a new id at each top face it crosses, DESTINY_FINDINGS 1.6);
  the bubble dictionaries (+1, 0, -1 in insertion order) and the interactive and keep-alive sets follow every change
  (`AddToBubble`); `InitializeBubbles` (the seed fill from each ball's top in the ball dictionary's order) drops every
  ball whose bubble does not change from its dictionary and sets, which then count as dead (DESTINY_FINDINGS 1.4);
  `UpdateBallBubble`; `CopyBubbles`.
- TIME-714: a ball in a dead bubble (no interactive and no keep-alive ball, not in WARP) is skipped by the whole tick
  (`dead_rows` before the park's tick; the park skips them: `evolve`).
- SEN-701 to SEN-704: HandleProximities at the end of the tick in the proximity map's order (libstdc++'s
  unordered_map order), the notification range (DoRange), target tracking (DoTargetTracking) and the sensor every
  period (DoProximity, entering and leaving, the vanished first in the sensor list's order): the candidates of
  GetNearbyBalls with the radius inflated by the range and the velocity zeroed, the ball re-filed twice (SEN-702's
  correction), the inside test summed in float; a ball in a dead bubble gets one forced check and its list emptied.
  A shuffled sensor (AddProximitySensor's shuffle) starts its period at floatrand(0, period), drawn
  from the world's own glibc rand() (`seed`: srand per world; TYPE_3's table, fptr and rptr as state, `rnd`), as the
  master's process-wide rand() draws it for its one park (space.GlibcRandom, held to compiled Destiny).
- SEN-708, SEN-709, SEN-710: the sensor's candidates share the ball's bubble (unless only interactives are sensed);
  CheckVisibility (-2 between bubbles; the first massive, uncloaked ball entered on the segment, in the ball
  dictionary's order); GetBallIdsInRange and GetBallIdsAndDistInRange from a ball or a point (the bubble's dictionary
  in its order, balls that left it included until CopyBubbles: DESTINY_FINDINGS 1.5); GetBubbleAtCoordinates.
- MOV-731's query: GetRemoteFollowers (free balls following a ball missing or in another bubble, ascending id).
- TIME-711's partition side: RemoveBall outside Evolve (the ball leaves its boxes and bubble at once).

**One algorithm, two devices.** The layer is one sequential algorithm a world, the transcription of space.py's code
in C++ (vec_space.cu beside this module), compiled once by nvcc into a library holding host functions (the CPU path:
a loop over the worlds) and CUDA kernels (one thread a world). The state is in torch tensors [W, N, ...] on the park's
device; an operation is a tape: per world K operations applied in order (`tape`), one launch for all worlds. Exact by
construction: correctly rounded IEEE double operations (__dadd_rn ... __dsqrt_rn on the device; the plain operators
on the host, compiled with -ffp-contract=off, no fast math), float sums as binary32 operations, IEEE division, no
transcendental, no float atomic; every constant generated from the reference's modules (`tables_header`); the build
reads the PTX back (vec_orient_cuda's check) and the host code's disassembly (`host_check`: no fused instruction).
Without nvcc (a CPU-only machine) `build` compiles the host functions alone with g++
(`build_host`): the same source and header, nvcc's host-compiler flags (`HOST_FLAGS`: -O3, -std=c++17, -fPIC,
-ffp-contract=off, -fno-fast-math), the kernels and their launchers left out by the source's own `__CUDACC__` guards,
and the three CUDA qualifiers outside them (`HD`'s __host__ and __device__, the device tables' __constant__) defined
empty on the command line (`HOST_DEFINES`), so the source, and with it every nvcc library's key, is unchanged. The
library holds the host functions only (`host_only` in its manifest); a layer on a CUDA device still needs nvcc.

**What is not stored.** Boxes below the top level: a box exists while a ball is filed in it or below it, so the walk
of GetNearbyBalls follows from the filings (the order key, vec_collide, here per pair). Top boxes are a
per-world table: those some filing reaches (a reference count), with their bubble id. BoxAllocator keeps up to 256
released boxes, and Destiny reads a released box's memory in one path (Ballpark.cpp:3141-3185): undefined only once
the box was deleted, which needs the high-water mark of live boxes to reach 256; a world whose bound (each ball's
boxes with their ancestors, summed, plus 8) reaches 256 is flagged (`FLAG_BOXES`) and refused by `check()`.

**Global balls** (SEN-708). Over VecCollidePark made with `global_balls` (CollideView) the
layer reads the park's global flag (`ParkView.global_flags`; set at the load and by SetBallGlobal, `set_global`): a
sensor that senses every ball (not only the interactive ones) takes a global ball of another bubble as a candidate,
as the master's GetNearbyBalls does (Partition.cpp:344: a ball in another bubble is skipped unless it is global), and
the bubble update leaves global balls out of its additions (PopulateAddDelForBubble and AdditionsAndDeletions,
Thunkers.cpp:2509-2895: a global ball is never added to a client's park by a bubble's lists; the server announces it
to every bubble instead). Over a view without the flag (the other views) no ball is global, as before. 
ModesView and OrientView made with `global_balls` hold the flag themselves (their parks have none: VecModesPark and
VecOrientPark have no collisions, so the layer's sensors and bubble lists are all that read it), set in place by
`ParkView.set_global`; `VecSpace.last_dead` keeps the rows the last tick left out of the park's tick (the client's
view reads it).

**The domain.** Free or fixed balls with positive ids, none global (but over CollideView with the park's global flag; over a view holding the flag), none huge (two radii above a top box), none at the
grid's edge (a top box and its 26 neighbours in the grid; Destiny clamps there), no space junk, harmonics or force
fields, no static collidables, no collision contact over VecModesPark or
VecOrientPark (they have none: two massive balls must not meet there). Each refusal is a flag (`FLAG_*`), raised by
`check()`.

**Collisions within bubbles**. Over VecCollidePark (`CollideView`) the master's collisions
are Destiny's: the park's gradient (or iterative method) skips a candidate in another bubble, as the master's
GetNearbyBalls skips it (SEN-708; the park's `bubble`, which the layer writes after every tape), the dead bubbles'
balls are left out of its tick (TIME-714; its `skip`), and its filing, which orders a ball's contacts, is the layer's
(the master's re-filings: the setters', the commits', a sensor's two), copied into the park before and after its
tick. Without the layer the park stays a client's view (every candidate, whatever its bubble). Not over it: RemoveBall
(the park keeps no moribund ball; refused by `check()`).

**The unordered maps**. The proximity balls and each sensor's list hold any number of balls
up to the world's slots, in libstdc++'s order: the bucket count and the rehash policy's next resize are state beside
each list, and _Prime_rehash_policy (__prime_list's entries the layer can reach, generated from the reference) grows
them as the compiled code does (space.StdHashMap). A layer built with a smaller capacity (`umap_cap`; 13, the first
bucket count, is the former fixed one) keeps that memory and flags a map past it (`FLAG_PROXIMITY`,
`FLAG_SENSOR_LIST`).

Not here: missiles, mushrooms, trolls, formations, the delayed removal (moribund until BringOutTheDeads), AddBall of
an existing ball, SetBallFree, SetBallGlobal.
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

from . import full as _full
from . import modes as _modes
from . import ref as _ref
from . import space as _space
from . import vec_orient_cuda as _K

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
from ._cuda_build import SourcePath, cache_root, default_nvcc, read_source  # noqa: E402,F401
SOURCE = SourcePath(Path(__file__).with_name("vec_space.cu"))
HEADER = "vec_space_tables.h"
ABI = 4                              # 3: the Space structure gained `rnd`; 4: Park `global_`
BLOCK = 128
MAX_SLOTS = 64                       # N, the slots a world, at most (the kernel's stack arrays)
UMAP_CAP = MAX_SLOTS                 # the most keys of a proximity map or sensor list (every slot of a world)
CODE_BASE = 4 ** 3 + 1               # a post-order digit: 64 children a box, then "past the ball's level"
CODE_SPAN = CODE_BASE ** (_space.NUM_LEVELS - 1)
NOT_A_CANDIDATE = 1 << 62
NO_TOP = -(1 << 63)                  # GetBubbleAtCoordinates' None
_LIBS: dict = {}
# the build without nvcc (`build_host`). nvcc hands its host compiler -O3 and -std=c++17 and the
# -Xcompiler list of vec_orient_cuda.FLAGS (`nvcc --dryrun`: gcc -std=c++17 ... -fPIC -ffp-contract=off -fno-fast-math
# -O3), so g++ gets the same; the CUDA qualifiers outside the source's __CUDACC__ guards are defined empty.
HOST_FLAGS = ["-O3", "-std=c++17", "-fPIC", "-ffp-contract=off", "-fno-fast-math", "-shared"]
HOST_DEFINES = ["-D__host__=", "-D__device__=", "-D__constant__="]
_CXX_VERSIONS: dict = {}
RAND_DEG = len(_space.GlibcRandom(1).s)          # glibc's TYPE_3 table (31 words), from the reference
RAND_WORDS = RAND_DEG + 2                         # a world's rand state: the table, then fptr and rptr (indices)


def glibc_srand(seeds) -> torch.Tensor:
    """srand(seed) for each world (glibc's srandom_r, TYPE_3): [W, RAND_WORDS] int64, the table's int32 values, then
    fptr and rptr, taken from the reference's own generator (exact_ballpark.space.GlibcRandom) for each distinct seed,
    so the state is the reference's by construction. The layer then draws rand() on the device
    (vec_space.cu's glibc_rand) for each shuffled sensor, as AddProximitySensor's floatrand does."""
    seeds = [int(x) for x in torch.as_tensor(seeds, dtype=I64).reshape(-1).tolist()]
    rows = {}
    for x in set(seeds):
        g = _space.GlibcRandom(x)
        rows[x] = list(g.s) + [g.f, g.r]
    return torch.tensor([rows[x] for x in seeds], dtype=I64).reshape(len(seeds), RAND_WORDS)

OPS = ("NONE", "ADD", "REFILE", "INIT_BUBBLES", "COPY_BUBBLES", "KEEP_ALIVE", "SET_INTERACTIVE", "ADD_SENSOR",
       "REMOVE_SENSOR", "NOTIFY_RANGE", "TRACK", "REMOVE_BALL", "DEAD", "AFTER_EVOLVE", "CLEAR_EVENTS",
       "CLOAK_SENSOR", "UNCLOAK", "ADDITIONS_AND_DELETIONS")
OP = {name: k for k, name in enumerate(OPS)}
QUERIES = ("NONE", "VISIBILITY", "RANGE_BALL", "RANGE_POINT", "BUBBLE_AT", "REMOTE_FOLLOWERS")
AD = {"NONE": 0, "SAME_BUBBLE": 1, "SAME_OWN": 2, "CHANGED": 3}     # AdditionsAndDeletions' per-user-ball kinds
Q = {name: k for k, name in enumerate(QUERIES)}
EVENTS = ("", "DoProximity", "DoRange", "DoTargetTracking", "DoFinalCleanup")
EV = {name: k for k, name in enumerate(EVENTS) if name}
EV_MACROS = {"DoProximity": "EV_PROXIMITY", "DoRange": "EV_RANGE", "DoTargetTracking": "EV_TRACKING",
             "DoFinalCleanup": "EV_FINAL_CLEANUP"}
FLAGS = ("EVENTS", "TOPS", "EDGE", "BOXES", "DICTS", "ENTRIES", "PROXIMITY", "SENSOR_LIST", "HUGE", "BAD_OP", "SETS")
FLAG = {name: 1 << k for k, name in enumerate(FLAGS)}
# the branches counted (the reference's _hit names, for the coverage checks)
HITS = ("bubble.no_bubble", "bubble.new", "bubble.neighbour", "bubble.changed", "insert.unchanged",
        "insert.adjacent", "init.traversed", "nearby.no_interactives", "nearby.other_bubble", "sensor.wait",
        "sensor.check", "sensor.vanished", "sensor.left", "sensor.entered", "sensor.dead_bubble_check", "range.event",
        "tracking.event", "tick.dead_bubble", "warp.drop_out_insert", "visibility.other_bubble",
        "visibility.occluded", "range.moribund", "sensor.shuffle")
# planted faults (the tests only): a header value of XBPS_FAULT compiles one wrong rule into the library
FAULTS = {"no_neighbour_bubble": 1, "top_keeps_bubble": 2, "reentry_appended": 3, "init_keeps_unchanged": 4,
          "dead_ignored": 5, "walk_by_id": 6, "list_appended": 7, "range_skips_left": 8, "copy_keeps_empty": 9,
          "commit_unsorted": 10, "no_drop_out_insert": 11, "rehash_skipped": 12, "rehash_keeps_order": 13,
          "clear_resets_buckets": 14, "growth_without_doubling": 15,
          "shuffle_ignored": 16, "rand_unshifted": 17, "rand_no_feedback": 18}      # 16-18
# the reference's literals the header holds: (value, the reference's own spelling, its module)
LITERALS = {
    "K_TRACKING_MIN": (0.00001, "b.tracking_range >= 0.00001", "space"),     # Ball.cpp:598
    "K_RAND_MAX": (2147483647.0, "self.rng.next() / 2147483647.0", "space"),   # floatrand's RAND_MAX (Ball.cpp:19)
}


def _hit_macro(name: str) -> str:
    """A branch's macro in the source: bubble.new -> H_NEW, insert.adjacent -> H_ADJACENT, sensor.check ->
    H_SENSOR_CHECK."""
    head, tail = name.split(".", 1)
    return "H_" + (tail if head in ("bubble", "insert") else f"{head}_{tail}").upper()


def _umap_buckets() -> int:
    """libstdc++'s bucket count after a first insertion, as space.StdHashMap computes it (13)."""
    m = _space.StdHashMap()
    m.insert(1, None)
    return m.bucket_count


def _umap_primes() -> tuple:
    """__prime_list's entries a map of at most UMAP_CAP keys can ask for: through the bucket count UMAP_CAP keys reach
    (space.StdHashMap's policy; the bucket count grows with the most keys a map ever held, never shrinks)."""
    m = _space.StdHashMap()
    for k in range(1, UMAP_CAP + 1):
        m.insert(k, None)
    return _space._PRIME_LIST[:_space._PRIME_LIST.index(m.bucket_count) + 1]


UMAP_FAST_CAP = _umap_buckets()      # the former fixed capacity: maps of at most the first bucket count (13)


def literals_in_source() -> dict:
    """Each of LITERALS: whether its module still writes it as the header holds it."""
    mods = {"space": inspect.getsource(_space)}
    return {name: phrase in mods[mod] for name, (_, phrase, mod) in LITERALS.items()}


def _dbits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def table_values() -> dict:
    """Every constant of the header, from the reference's modules: {name: (kind, value)}."""
    hm = _space.StdHashMap
    if hm._MAX_LOAD != 1.0:
        raise ValueError("the kernel's rehash policy is integer arithmetic: max_load_factor 1 only")
    primes = _umap_primes()
    v = {"K_NUM_LEVELS": ("i", _space.NUM_LEVELS), "K_BOX_CHUNK": ("i", _space.BOX_CHUNK_SIZE),
         "K_CODE_BASE": ("i", CODE_BASE), "K_CODE_SPAN": ("i", CODE_SPAN), "K_NOT_A_CANDIDATE": ("i", NOT_A_CANDIDATE),
         "K_NO_TOP": ("i", NO_TOP), "K_MAX_SLOTS": ("i", MAX_SLOTS), "K_UMAP_BUCKETS": ("i", _umap_buckets()),
         "K_MODE_GOTO": ("i", _ref.GOTO), "K_MODE_FOLLOW": ("i", _modes.FOLLOW), "K_MODE_WARP": ("i", _modes.WARP),
         "K_MODE_ORBIT": ("i", _modes.ORBIT), "K_MODE_MISSILE": ("i", _modes.MISSILE),
         "K_MODE_FORMATION": ("i", _modes.FORMATION), "K_HITS": ("i", len(HITS)),
         "K_MIN_MOVE_SQ": ("d", _full.DISPATCH_MIN_MOVE_SQ), "K_TWO63": ("d", float(2 ** 63)),
         "XBPS_FAULT": ("i", 0), "K_NORMAL_CLOAK": ("i", _full.DSTNORMALCLOAK),
         "K_CLOAK_SENSOR_PERIOD": ("d", _full.CLOAK_SENSOR_PERIOD), "K_UMAP_CAP": ("i", UMAP_CAP),
         "K_NFAST_BKT": ("i", len(hm._FAST_BKT)), "K_FAST_BKT": ("l", hm._FAST_BKT), "K_NPRIMES": ("i", len(primes)),
         "K_PRIME_LIST": ("l", primes), "K_PRIME_FROM": ("i", hm._SEARCH_FROM), "K_UMAP_FIRST_MIN": ("i", hm._FIRST_MIN),
         "K_UMAP_GROWTH": ("i", hm._GROWTH), "K_RAND_DEG": ("i", RAND_DEG), "K_RAND_WORDS": ("i", RAND_WORDS)}
    for i in range(_space.NUM_LEVELS):
        v[f"K_LW{i}"] = ("d", _space.LEVEL_WIDTH[i])
        v[f"K_LG{i}"] = ("i", _space.LEVEL_GRID[i])
    v.update({k: ("d", val) for k, (val, _, _) in LITERALS.items()})
    v.update({f"OP_{name}": ("i", k) for name, k in OP.items()})
    v.update({f"Q_{name}": ("i", k) for name, k in Q.items()})
    v.update({f"AD_{name}": ("i", k) for name, k in AD.items()})
    v.update({EV_MACROS[name]: ("i", k) for name, k in EV.items()})
    v.update({f"FLAG_{name}": ("i", bit) for name, bit in FLAG.items()})
    v.update({_hit_macro(name): ("i", k) for k, name in enumerate(HITS)})
    return v


def tables_header(perturb: dict | None = None) -> str:
    """The header the source includes, generated from table_values(); `perturb` ({name: value}) changes one, for the
    tests' planted faults only."""
    vals = table_values()
    for k, val in (perturb or {}).items():
        if k not in vals:
            raise KeyError(k)
        vals[k] = (vals[k][0], val)
    lines = ["// generated by exact_ballpark.vec_space.tables_header: do not edit", "#pragma once"]
    for k, (kind, val) in vals.items():
        if kind == "d":
            lines.append(f"#define {k} 0x{_dbits(float(val)):016x}ULL")
        elif kind == "l":
            lines.append(f"#define {k} {{" + ", ".join(f"{int(x)}LL" for x in val) + "}")
        else:
            lines.append(f"#define {k} ({int(val)}LL)" if int(val) >= 0 else f"#define {k} ({int(val) + 1}LL - 1)")
    return "\n".join(lines) + "\n"


# ============================================================================================ build and load

def build_dir() -> Path:
    d = os.environ.get("EXACT_BALLPARK_SPACE_CUDA_DIR")
    if d:
        return Path(d)
    return cache_root() / "space_cuda"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_FUSED = re.compile(r"\b(fmadd|fmsub|fnmadd|fnmsub|fmla|fmls)\b")


def _host_fused(text: str) -> dict:
    """The fused floating-point instructions in a disassembly (objdump -d -C), by function, in this module's own code
    (namespace xbps and the xbps_ entry points): {"functions": count, "fused": {name: count}}."""
    fused, current, functions = {}, None, 0
    for line in text.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m:
            name = m.group(1)
            current = name if ("xbps::" in name or name.startswith("xbps_")) else None
            functions += current is not None
            continue
        if current is not None and _FUSED.search(line):
            fused[current] = fused.get(current, 0) + 1
    return {"functions": functions, "fused": fused}


def host_check(lib: Path) -> dict:
    """The host functions' fused floating-point instructions (aarch64: fmadd, fmsub, fnmadd, fnmsub, fmla, fmls) in
    the library's disassembly of this module's own code: none may appear (the host compiler's -ffp-contract=off)."""
    objdump = shutil.which("objdump")
    if objdump is None:
        return {"ok": False, "reason": "objdump not found", "functions": 0, "fused": {}}
    text = subprocess.run([objdump, "-d", "-C", "--no-show-raw-insn", str(lib)], capture_output=True, text=True,
                          check=True).stdout
    out = _host_fused(text)
    return {"ok": not out["fused"] and out["functions"] > 0, **out}


def build(out_dir: Path | None = None, variant: str = "native", source: Path = SOURCE, perturb: dict | None = None,
          checks: bool = True, host: bool = False) -> dict:
    """Compile `source` with its generated header (nvcc with exact_ballpark.vec_orient_cuda's flags and targets) into
    out_dir (by default `build_dir()`); returns the manifest written beside the library (the inputs' SHA-256, nvcc,
    the flags, the kernels' resources, the PTX check, the host check, the library's SHA-256, the UTC time). A library
    built from the same inputs is reused. `perturb` and `checks=False` are for the tests' planted faults only. Without
    nvcc (none at vec_orient_cuda's path nor on PATH), or with `host` (the tests' comparison of the two compilers), the
    host functions alone, by g++ (`build_host`): whatever the variant, which names device targets
    only."""
    if variant not in _K.TARGETS:
        raise ValueError(f"unknown variant {variant!r}")
    out_dir = Path(out_dir) if out_dir is not None else build_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    text = read_source(source)
    calls = _K.scan_source(text)
    if checks and calls:
        raise RuntimeError(f"the source calls CUDA's math library: {calls}")
    header = tables_header(perturb)
    if host:                                        # g++'s build, even where nvcc is found
        return build_host(out_dir, source, text, header, perturb, checks, calls)
    try:
        nvcc = _K._nvcc()
    except RuntimeError:                            # no nvcc, the host functions by g++
        return build_host(out_dir, source, text, header, perturb, checks, calls)
    gencode = [f"-gencode=arch={a},code={c}" for a, c in _K.TARGETS[variant]]
    key = _sha(json.dumps([_sha(text.encode()), _sha(header.encode()), _K.FLAGS, gencode, str(nvcc),
                           _K.nvcc_version(nvcc), checks]).encode())[:16]
    stem = f"vec_space_{variant}_{key}"
    lib, man = out_dir / f"{stem}.so", out_dir / f"{stem}.json"
    if lib.exists() and man.exists():
        m = json.loads(man.read_text())
        if m.get("library_sha256") == _sha(lib.read_bytes()) and (
                (m.get("ptx_check", {}).get("ok") and m.get("host_check", {}).get("ok")) or not checks):
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
        cmd = [str(nvcc), *_K.FLAGS, *gencode, "-I", str(work), "-o", str(work / "lib.so"), str(src)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
        cuobjdump = nvcc.parent / "cuobjdump"
        listing = subprocess.run([str(cuobjdump), "--list-elf", "--list-ptx", str(work / "lib.so")],
                                 capture_output=True, text=True, check=True).stdout.splitlines()
        ptx = subprocess.run([str(cuobjdump), "-ptx", str(work / "lib.so")], capture_output=True, text=True,
                             check=True).stdout
        check = _K.ptx_check(ptx)
        hcheck = host_check(work / "lib.so")
        if checks and not check["ok"]:
            raise RuntimeError(f"the PTX has unpinned floating-point instructions: {check['refused']}")
        if checks and not hcheck["ok"]:
            raise RuntimeError(f"the host code has fused floating-point instructions: {hcheck}")
        resources = [ln.strip() for ln in (r.stdout + r.stderr).splitlines()
                     if "registers" in ln or "spill" in ln or "Compiling entry" in ln or "stack frame" in ln]
        version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True).stdout
        manifest = {
            "variant": variant, "source": str(source), "source_sha256": _sha(text.encode()),
            "header": HEADER, "header_sha256": _sha(header.encode()), "perturb": perturb or {}, "checks": checks,
            "source_scan": calls, "nvcc": version.strip().splitlines()[-1], "nvcc_path": str(nvcc),
            "flags": _K.FLAGS, "targets": [f"{a}->{c}" for a, c in _K.TARGETS[variant]], "embedded": listing,
            "ptxas": resources, "ptx_check": check, "host_check": hcheck, "library": lib.name,
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


def _cxx() -> Path:
    """The host compiler of the build without nvcc: EXACT_BALLPARK_CXX, else g++ on PATH."""
    c = os.environ.get("EXACT_BALLPARK_CXX")
    if c:
        if not Path(c).exists():
            raise RuntimeError(f"EXACT_BALLPARK_CXX names no file: {c}")
        return Path(c)
    found = shutil.which("g++")
    if not found:
        raise RuntimeError("neither nvcc nor g++ found: the space layer's host functions need one (set EXACT_BALLPARK_NVCC or "
                           "EXACT_BALLPARK_CXX)")
    return Path(found)


def cxx_version(cxx: Path) -> str:
    """`<cxx> --version`'s whole output, asked once a process through this module's `subprocess` (a fleet job's guard,
    exact_ballpark.no_compile, refuses it, as it refuses every call but the pinned nvcc's version): the host build's key
    holds it."""
    k = str(cxx)
    if k not in _CXX_VERSIONS:
        _CXX_VERSIONS[k] = subprocess.run([k, "--version"], capture_output=True, text=True, check=True).stdout
    return _CXX_VERSIONS[k]


def build_host(out_dir: Path, source: Path, text: str, header: str, perturb: dict | None, checks: bool,
               calls: list) -> dict:
    """The host functions alone (`xbps_tape_host`, `xbps_query_host`, `xbps_layout`, `xbps_abi`), compiled by g++ from
    the source's text and its generated header (`build` reads, scans and makes them, then calls this)
    into out_dir, as nvcc's host compiler compiles them: `HOST_FLAGS` (nvcc's -O3 and -std=c++17 and
    vec_orient_cuda.FLAGS' -Xcompiler list), `HOST_DEFINES` (the CUDA qualifiers outside the source's __CUDACC__
    guards, empty), the source as C++. The host's check as `build`'s: `host_check` (no fused instruction). Returns the
    manifest written beside the library (`host_only`: no kernel; the inputs' SHA-256, the compiler and its version, the
    flags, the host check, the library's SHA-256, the UTC time); a library built from the same inputs is reused. Its
    key holds the compiler's path and `--version` text, never nvcc's, so it never takes an nvcc library's name."""
    cxx = _cxx()
    version = cxx_version(cxx)
    key = _sha(json.dumps([_sha(text.encode()), _sha(header.encode()), HOST_FLAGS, HOST_DEFINES, str(cxx), version,
                           checks]).encode())[:16]
    stem = f"vec_space_host_{key}"
    lib, man = out_dir / f"{stem}.so", out_dir / f"{stem}.json"
    if lib.exists() and man.exists():
        m = json.loads(man.read_text())
        if m.get("library_sha256") == _sha(lib.read_bytes()) and (m.get("host_check", {}).get("ok") or not checks):
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
        cmd = [str(cxx), *HOST_FLAGS, *HOST_DEFINES, "-I", str(work), "-o", str(work / "lib.so"), "-x", "c++",
               str(src)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"{cxx.name} failed:\n{r.stdout}\n{r.stderr}")
        hcheck = host_check(work / "lib.so")
        if checks and not hcheck["ok"]:
            raise RuntimeError(f"the host code has fused floating-point instructions: {hcheck}")
        manifest = {
            "variant": "host", "host_only": True, "source": str(source), "source_sha256": _sha(text.encode()),
            "header": HEADER, "header_sha256": _sha(header.encode()), "perturb": perturb or {}, "checks": checks,
            "source_scan": calls, "compiler": (version.strip().splitlines() or [""])[0], "compiler_path": str(cxx),
            "flags": HOST_FLAGS, "defines": HOST_DEFINES, "host_check": hcheck, "library": lib.name,
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


class _Park(ctypes.Structure):
    _fields_ = [("R", ctypes.c_int64)] + [(n, ctypes.c_void_p) for n in (
        "ids", "pos", "vel", "old_pos", "old_vel", "radius", "max_vel", "agility", "sf", "mass", "tf", "free_",
        "massive", "interactive", "cloaked", "moribund", "mode", "follow_id", "follow_range", "goto_", "sensor",
        "effect_stamp", "bubble_out", "global_")] + [("friction", ctypes.c_double), ("dt", ctypes.c_double)]


_SPACE_SIZES = ("W", "N", "T", "E", "D", "M", "S", "I")
_SPACE_PTRS = ("rows", "by_id", "lvl", "key", "mod", "nbox", "bubble", "old_bubble", "in_park",
               "ka_ball", "bseq", "tkey", "tbub", "tref", "tn", "ebub", "eslot", "eact", "eseq", "en", "eseq_next",
               "dbub", "dn", "iab", "ias", "ian", "kab", "kas", "kan", "bubble_id", "box_sum", "bseq_next", "srange",
               "speriod", "selapsed", "sonly",
               "has_prox", "slist", "slen", "trk", "trk_range", "trk_in", "ntf_range", "ntf_in", "pmap", "pn", "ev",
               "evn", "flags", "dead", "dropped", "hits", "ad_kind", "ad_add", "ad_addn", "ad_del", "ad_deln",
               "pbkt", "pnr", "sbkt", "snr", "rnd")


class _Space(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int64) for n in _SPACE_SIZES] + [(n, ctypes.c_void_p) for n in _SPACE_PTRS]


class _Tape(ctypes.Structure):
    _fields_ = [("K", ctypes.c_int64), ("code", ctypes.c_void_p), ("a", ctypes.c_void_p), ("b", ctypes.c_void_p),
                ("f", ctypes.c_void_p)]


class _Query(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int64), ("a", ctypes.c_void_p), ("b", ctypes.c_void_p), ("f", ctypes.c_void_p),
                ("out", ctypes.c_void_p), ("outf", ctypes.c_void_p)]


def load_library(variant: str = "native", source: Path = SOURCE, perturb: dict | None = None,
                 checks: bool = True, host: bool = False) -> dict:
    """The compiled library (built if missing or stale), its functions typed, and its manifest; once a process.
    a library built without nvcc (`build_host`, which `build` takes when nvcc is not found) holds the
    host functions only (`host_only` in the entry: no kernel to type); `host` builds that one even where nvcc is found
    (the tests' comparison of the two compilers)."""
    k = (variant, str(source), json.dumps(perturb or {}, sort_keys=True), checks) + (("host",) if host else ())
    if k in _LIBS:
        return _LIBS[k]
    manifest = build(None, variant, source, perturb, checks, host=host)
    lib = ctypes.CDLL(str(build_dir() / manifest["library"]))
    if lib.xbps_abi() != ABI:
        raise RuntimeError(f"the library's ABI {lib.xbps_abi()} is not {ABI}")
    P, S, T, QQ = (ctypes.POINTER(x) for x in (_Park, _Space, _Tape, _Query))
    lib.xbps_tape_host.argtypes = [P, S, T]
    lib.xbps_tape_host.restype = ctypes.c_int
    lib.xbps_query_host.argtypes = [P, S, QQ]
    lib.xbps_query_host.restype = ctypes.c_int
    host_only = bool(manifest.get("host_only"))
    if not host_only:
        lib.xbps_tape_cuda.argtypes = [P, S, T, ctypes.c_void_p, ctypes.c_int]
        lib.xbps_tape_cuda.restype = ctypes.c_int
        lib.xbps_query_cuda.argtypes = [P, S, QQ, ctypes.c_void_p, ctypes.c_int]
        lib.xbps_query_cuda.restype = ctypes.c_int
    lib.xbps_layout.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    lib.xbps_layout.restype = ctypes.c_int
    if not host_only:
        lib.xbps_info.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 5
        lib.xbps_info.restype = ctypes.c_int
        lib.xbps_error_string.restype = ctypes.c_char_p
        lib.xbps_error_string.argtypes = [ctypes.c_int]
    entry = {"lib": lib, "manifest": manifest, "host_only": host_only}
    _LIBS[k] = entry
    return entry


def layout_check(lib: dict | None = None) -> dict:
    """The ctypes structures against the compiled ones: sizes and field offsets, {name: (compiled, ctypes)}."""
    lib = lib or load_library()
    out = (ctypes.c_int64 * 16)()
    lib["lib"].xbps_layout(out)
    mine = (ctypes.sizeof(_Park), ctypes.sizeof(_Space), ctypes.sizeof(_Tape), ctypes.sizeof(_Query),
            _Park.effect_stamp.offset, _Park.bubble_out.offset, _Park.dt.offset, _Space.iab.offset,
            _Space.hits.offset, _Space.ad_deln.offset, _Space.snr.offset, _Space.rnd.offset, _Park.global_.offset)
    names = ("Park", "Space", "Tape", "Query", "Park.effect_stamp", "Park.bubble_out", "Park.dt", "Space.iab",
             "Space.hits", "Space.ad_deln", "Space.snr", "Space.rnd", "Park.global_")
    return {n: (int(out[k]), int(mine[k])) for k, n in enumerate(names)}


def kernel_info(device=None, lib: dict | None = None) -> dict:
    """The tape kernel's and the query kernel's resources on the current device (registers, local memory, versions,
    maxThreadsPerBlock)."""
    lib = lib or load_library()
    if lib.get("host_only"):                        # built by g++, without nvcc
        raise RuntimeError("the space library was built without nvcc (host functions only): it holds no kernel")
    out = {}
    for which, name in ((0, "tape"), (1, "query")):
        vals = [ctypes.c_int() for _ in range(5)]
        rc = lib["lib"].xbps_info(which, *(ctypes.byref(v) for v in vals))
        if rc:
            raise RuntimeError(f"xbps_info failed: {lib['lib'].xbps_error_string(rc).decode()}")
        out[name] = dict(zip(("registers", "local_bytes", "binary_version", "ptx_version", "max_threads"),
                             (v.value for v in vals)))
    return out


# ================================================================================================= the park's view

class ParkView:
    """The park's tensors the layer reads (and in part writes: radius and velocity during a sensor's check, the
    interactive, moribund and sensor flags, the park's own bubble field), by Destiny's names; re-read at each launch
    (a park rebinds its tensors at its tick). Subclasses map a park. `global_balls`: each row's global
    flag at the load (AddBall's isGlobal), held here for a park that holds none (VecModesPark, VecOrientPark): the
    layer reads it, `set_global` sets it."""

    def __init__(self, park, global_balls=None):
        self.park = park
        dev = torch.device(park.dev)
        if dev.type == "cuda" and dev.index is None:        # "cuda" as tensors report it: cuda:N
            dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        self.R = park.n
        self.friction, self.dt = float(park.friction), float(park.dt)
        self.glob = None if global_balls is None else \
            torch.as_tensor(global_balls, dtype=BOOL).to(dev).reshape(self.R).contiguous().clone()

    def tensors(self) -> dict:
        raise NotImplementedError

    def bubble_out(self):
        return None

    def global_flags(self):
        """[R] bool: each row's global flag, or None: no ball is global (the flags the
        view holds when made with `global_balls`)."""
        return getattr(self, "glob", None)

    def set_global(self, rows: torch.Tensor, flag: bool) -> None:
        """SetBallGlobal: each row in rows [R] (bool) made global (flag true) or local, in place (the
        layer keeps the tensor's address); refused without the flags."""
        if self.global_flags() is None:
            raise ValueError("SetBallGlobal needs a view made with global_balls (or a VecCollidePark made with them)")
        self.glob.masked_fill_(rows.to(BOOL).to(self.dev).reshape(self.R), bool(flag))

    def check(self) -> None:
        """Raise where the park leaves the view's domain (VecSpace.check calls it)."""


class ModesView(ParkView):
    """exact_ballpark.vec_modes.VecModesPark: every field is the park's own."""

    def tensors(self) -> dict:
        p = self.park
        return {"ids": p.ids, "pos": p.pos, "vel": p.vel, "old_pos": p.old_pos, "old_vel": p.old_vel,
                "radius": p.radius, "max_vel": p.max_vel, "agility": p.agility, "sf": p.speed_fraction,
                "mass": p.mass, "tf": p.time_factor, "free_": p.free, "massive": p.massive,
                "interactive": p.interactive, "cloaked": p.cloaked, "moribund": p.moribund, "mode": p.mode,
                "follow_id": p.follow_id, "follow_range": p.follow_range, "goto_": p.goto, "sensor": p.sensor,
                "effect_stamp": p.effect_stamp}

    def bubble_out(self):
        return self.park.bubble


class OrientView(ParkView):
    """exact_ballpark.vec_orient.VecOrientPark (GOTO and STOP, free balls): the flags it lacks are held here (`flags`:
    an object with interactive, massive, cloaked and sensor tensors, such as vec_server's OrientAdapter), the ids
    given; every ball free; no follow fields."""

    def __init__(self, park, ids, flags=None, global_balls=None):
        super().__init__(park, global_balls)            # (the global flags held here)
        d = self.dev
        self.ids = torch.as_tensor(ids, dtype=I64).to(d)
        self.flags = flags
        n = self.R
        self.own = {"interactive": torch.zeros(n, dtype=BOOL, device=d),
                    "massive": torch.zeros(n, dtype=BOOL, device=d),
                    "cloaked": torch.zeros(n, dtype=I64, device=d), "sensor": torch.zeros(n, dtype=BOOL, device=d)}
        self.free = torch.ones(n, dtype=BOOL, device=d)
        self.moribund = torch.zeros(n, dtype=BOOL, device=d)
        self.follow_id = torch.zeros(n, dtype=I64, device=d)
        self.effect_stamp = torch.zeros(n, dtype=I64, device=d)
        self.follow_range = torch.zeros(n, dtype=F64, device=d)

    def flag(self, name: str) -> torch.Tensor:
        return getattr(self.flags, name) if self.flags is not None else self.own[name]

    def set_flag(self, name: str, value: torch.Tensor) -> None:
        if self.flags is not None:
            setattr(self.flags, name, value)
        else:
            self.own[name] = value

    def tensors(self) -> dict:
        p = self.park
        return {"ids": self.ids, "pos": p.pos, "vel": p.vel, "old_pos": p.old_pos, "old_vel": p.old_vel,
                "radius": p.radius, "max_vel": p.max_vel, "agility": p.agility, "sf": p.speed_fraction,
                "mass": p.mass, "tf": p.time_factor, "free_": self.free, "massive": self.flag("massive"),
                "interactive": self.flag("interactive"), "cloaked": self.flag("cloaked"), "moribund": self.moribund,
                "mode": p.mode, "follow_id": self.follow_id, "follow_range": self.follow_range, "goto_": p.goto,
                "sensor": self.flag("sensor"), "effect_stamp": self.effect_stamp}


class CollideView(ParkView):
    """exact_ballpark.vec_collide.VecCollidePark (W worlds of N slots, GOTO and STOP with Destiny's collisions): row w x N + slot, the park's own fields as flat views [W x N, ...] (re-read at each launch: the park
    rebinds its tensors at its tick; a view, never a copy, so the layer's writes during a sensor's check reach the
    park); the flags it lacks held here or by `flags` (an object with interactive, cloaked (the cloak's mode, int64)
    and sensor tensors [R], such as exact_ballpark.vec_server_space's SpaceCollideAdapter): interactive (by default
    `interactive`, else none), the cloak's mode (by default the park's flag as 1 or 0), the sensor flag. Every ball is
    free or fixed as the park has it, none moribund, none following.

    The master's collisions: the park's own `bubble` [W, N] is created here (unless it has one) and written by the
    layer after every tape (`bubble_out`), so the park's gradient and iterative method skip a candidate in another
    bubble, as the master's GetNearbyBalls does (SEN-708); the park's filing (level, key, mod), which orders its
    contacts, is the layer's (the master's re-filings: the sensor's two, the setters', the commits'), copied into the
    park before each of its ticks and after the layer's (`sync_filing`); the dead bubbles' balls are left out of the
    park's tick (`VecCollidePark.evolve`'s skip). Not modelled: RemoveBall (the park keeps no moribund ball: a ball the
    layer removed would stay a collision candidate; `check()` refuses it)."""

    def __init__(self, park, flags=None, interactive=None):
        self.park = park
        dev = torch.device(park.dev)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        self.R = park.W * park.N
        self.friction, self.dt = float(park.friction), float(park.dt)
        R, d = self.R, dev
        self.flags = flags
        inter = (torch.zeros(R, dtype=BOOL, device=d) if interactive is None else
                 torch.as_tensor(interactive, dtype=BOOL).to(d).reshape(R).clone())
        self.own = {"interactive": inter, "cloaked": park.cloaked.reshape(R).to(I64).contiguous(),
                    "sensor": torch.zeros(R, dtype=BOOL, device=d)}
        self.moribund = torch.zeros(R, dtype=BOOL, device=d)
        self.follow_id = torch.zeros(R, dtype=I64, device=d)
        self.effect_stamp = torch.zeros(R, dtype=I64, device=d)
        self.follow_range = torch.zeros(R, dtype=F64, device=d)
        if getattr(park, "bubble", None) is None:
            park.bubble = torch.full((park.W, park.N), -1, dtype=I64, device=d)

    def flag(self, name: str) -> torch.Tensor:
        if self.flags is not None and hasattr(self.flags, name):
            return getattr(self.flags, name)
        return self.own[name]

    def set_flag(self, name: str, value: torch.Tensor) -> None:
        """A flag by name (the view's own written in place, so that a captured graph keeps reading it); the cloak's
        mode also sets the park's own cloaked flag (its collisions' candidates)."""
        if self.flags is not None and hasattr(self.flags, name):
            setattr(self.flags, name, value)
            return
        self.own[name].copy_(value)
        if name == "cloaked":
            self.park.cloaked = (value != 0).reshape(self.park.W, self.park.N)

    def tensors(self) -> dict:
        p, R = self.park, self.R
        return {"ids": p.id.view(R), "pos": p.pos.view(R, 3), "vel": p.vel.view(R, 3), "old_pos": p.old_pos.view(R, 3),
                "old_vel": p.old_vel.view(R, 3), "radius": p.radius.view(R), "max_vel": p.max_vel.view(R),
                "agility": p.agility.view(R), "sf": p.speed_fraction.view(R), "mass": p.mass.view(R),
                "tf": p.time_factor.view(R), "free_": p.free.view(R), "massive": p.massive.view(R),
                "interactive": self.flag("interactive"), "cloaked": self.flag("cloaked"), "moribund": self.moribund,
                "mode": p.mode.view(R), "follow_id": self.follow_id, "follow_range": self.follow_range,
                "goto_": p.goto.view(R, 3), "sensor": self.flag("sensor"), "effect_stamp": self.effect_stamp}

    def bubble_out(self):
        return self.park.bubble.view(self.R)

    def global_flags(self):
        """The park's own global flag as a flat view [R] (a park made with global_balls; set at its
        load and by its set_global, in place), or None."""
        return self.park.glob.view(self.R) if getattr(self.park, "global_balls", False) else None

    def set_global(self, rows: torch.Tensor, flag: bool) -> None:
        """SetBallGlobal through the park's own flag (VecCollidePark.set_global refuses a park made
        without global balls)."""
        self.park.set_global(rows.to(BOOL).to(self.dev).reshape(self.park.W, self.park.N), bool(flag))

    def sync_filing(self, space: "VecSpace") -> None:
        """The park's filing (level, key, mod) made the layer's, slot by slot, in place (index work; after the first
        call, which finds the slots' rows on the host once, no host synchronisation: capturable)."""
        p, R = self.park, self.R
        idx = getattr(space, "_filing_rows", None)
        if idx is None:                                     # the slots in the park and their rows (fixed: `rows`)
            keep = space._flat < R
            idx = space._filing_rows = (space._flat[keep].contiguous(), torch.nonzero(keep).flatten())
        at, src = idx
        p.level.view(R).index_copy_(0, at, space.st["lvl"].flatten().index_select(0, src))
        p.key.view(R, 3).index_copy_(0, at, space.st["key"].reshape(-1, 3).index_select(0, src))
        p.mod.view(R, 3).index_copy_(0, at, space.st["mod"].reshape(-1, 3).index_select(0, src))

    def check(self) -> None:
        if bool(self.moribund.any()):
            raise RuntimeError("a ball removed by the layer (RemoveBall): VecCollidePark keeps no moribund ball, so it "
                               "would stay a collision candidate (not modelled)")


_PARK_TYPES = {"ids": I64, "pos": F64, "vel": F64, "old_pos": F64, "old_vel": F64, "radius": F64, "max_vel": F64,
               "agility": F64, "sf": F64, "mass": F64, "tf": F64, "free_": BOOL, "massive": BOOL,
               "interactive": BOOL, "cloaked": I64, "moribund": BOOL, "mode": I64, "follow_id": I64,
               "follow_range": F64, "goto_": F64, "sensor": BOOL, "effect_stamp": I64}


# ======================================================================================================= the layer

class VecSpace:
    """The master's partition, bubbles and sensors for W worlds of N slots over a park's rows. `rows` [W, N]: the park's
    row of each slot (-1 for none), each world's slots in the order its balls are added (the ball dictionary's order).
    `kernel`: "host" (the CPU path) or "cuda" (one thread a world), by default the park's device's. `umap_cap`: the
    most keys of the proximity map and of a sensor's list, by default N (every slot: nothing to flag); a smaller cap
    (13, UMAP_FAST_CAP, the former fixed one) keeps less memory and flags a world whose map passes it. `seed`: each
    world's srand seed (an int for every world, or [W]; FullBallpark's `seed`, 1 by default), the source of its
    shuffled sensors' first period (an ADD_SENSOR whose f[2] is nonzero)."""

    def __init__(self, view: ParkView, rows, kernel: str | None = None, events: int = 256, hits: bool = False,
                 block: int = BLOCK, lib: dict | None = None, server: bool = False, umap_cap: int | None = None,
                 seed=1):
        self.view = view
        self.dev = view.dev
        rows = torch.as_tensor(rows, dtype=I64).to(self.dev)
        if rows.dim() != 2:
            raise ValueError("rows: [W, N]")
        self.W, self.N = int(rows.shape[0]), int(rows.shape[1])
        if not 1 <= self.N <= MAX_SLOTS:
            raise ValueError(f"{self.N} slots a world: from 1 to {MAX_SLOTS}")
        self.kernel = kernel or ("cuda" if self.dev.type == "cuda" else "host")
        if self.kernel not in ("host", "cuda") or (self.kernel == "cuda") != (self.dev.type == "cuda"):
            raise ValueError(f"kernel {self.kernel!r} on {self.dev}: 'host' on the CPU, 'cuda' on a CUDA device")
        self.block = int(block)
        self.rows = rows.contiguous()
        W, N, d = self.W, self.N, self.dev
        S = N if umap_cap is None else int(umap_cap)
        if not 1 <= S <= UMAP_CAP:
            raise ValueError(f"umap_cap {S}: from 1 to {UMAP_CAP}")
        self.T, self.E, self.D, self.M, self.S = 8 * N + 8, 4 * N + 8, 8 * N + 16, int(events), S
        self.I = 2 * N + 8
        ids = view.tensors()["ids"]
        safe = self.rows.clamp(min=0)
        idw = torch.where(self.rows >= 0, ids[safe], torch.full_like(self.rows, (1 << 62)))
        order = torch.sort(idw, dim=1, stable=True).indices
        self.by_id = torch.where(torch.gather(self.rows, 1, order) >= 0, order, -1).contiguous()
        R = view.R
        self._flat = torch.where(self.rows >= 0, self.rows, R).flatten()      # slot -> row, R for an empty slot
        zi = lambda *s: torch.zeros(*s, dtype=I64, device=d)          # noqa: E731
        zb = lambda *s: torch.zeros(*s, dtype=torch.uint8, device=d)  # noqa: E731
        zf = lambda *s: torch.zeros(*s, dtype=F64, device=d)          # noqa: E731
        self.st = {
            "lvl": zi(W, N) - 1, "key": zi(W, N, 3), "mod": zi(W, N, 3), "nbox": zi(W, N),
            "bubble": zi(W, N) - 1, "old_bubble": zi(W, N) - 1, "in_park": zb(W, N),
            "ka_ball": zb(W, N), "iab": zi(W, self.I), "ias": zi(W, self.I), "ian": zi(W), "kab": zi(W, self.I),
            "kas": zi(W, self.I), "kan": zi(W), "bseq": zi(W, N), "tkey": zi(W, self.T, 3), "tbub": zi(W, self.T),
            "tref": zi(W, self.T), "tn": zi(W), "ebub": zi(W, self.E), "eslot": zi(W, self.E), "eact": zi(W, self.E),
            "eseq": zi(W, self.E), "en": zi(W), "eseq_next": zi(W), "dbub": zi(W, self.D), "dn": zi(W),
            "bubble_id": zi(W), "box_sum": zi(W), "bseq_next": zi(W), "srange": zf(W, N),
            "speriod": zf(W, N) + 2.0, "selapsed": zf(W, N), "sonly": zb(W, N), "has_prox": zb(W, N),
            "slist": zi(W, N, self.S), "slen": zi(W, N), "trk": zi(W, N) - 1, "trk_range": zf(W, N),
            "trk_in": zb(W, N), "ntf_range": zf(W, N), "ntf_in": zb(W, N), "pmap": zi(W, self.S), "pn": zi(W),
            "ev": zi(W, self.M, 4), "evn": zi(W), "flags": zi(W), "dead": zb(W, N), "dropped": zb(W, N),
            "hits": zi(W, len(HITS)) if hits else None,
            "ad_kind": zi(W, N) if server else None, "ad_add": zi(W, N, N) if server else None,
            "ad_addn": zi(W, N) if server else None, "ad_del": zi(W, N, N) if server else None,
            "ad_deln": zi(W, N) if server else None,
            "pbkt": zi(W) + 1, "pnr": zi(W), "sbkt": zi(W, N) + 1, "snr": zi(W, N)}
        if isinstance(seed, int):                           # one generator's state for every world
            rnd = glibc_srand([seed]).expand(W, RAND_WORDS)
        else:
            rnd = glibc_srand(seed)
            if rnd.shape[0] != W:
                raise ValueError(f"seed: an int or one a world ({W}), not {rnd.shape[0]}")
        self.st["rnd"] = rnd.to(d).contiguous()
        self._lib = lib

    # ---- plumbing ----

    def lib(self) -> dict:
        if self._lib is None:
            self._lib = load_library()
        if self.kernel == "cuda" and self._lib.get("host_only"):     # a g++ build has no kernel
            raise RuntimeError("the space layer on a CUDA device needs nvcc (EXACT_BALLPARK_NVCC): its library was built by g++ "
                               "without it (host functions only)")
        return self._lib

    def _park_struct(self) -> tuple:
        t = dict(self.view.tensors())
        keep = []
        for name, dt in _PARK_TYPES.items():
            x = t[name]
            if x.dtype != dt or x.device != self.dev:
                raise ValueError(f"the park's {name}: {x.dtype} on {x.device}, not {dt} on {self.dev}")
            if not x.is_contiguous():
                raise ValueError(f"the park's {name} is not contiguous")
            keep.append(x)
        out = self.view.bubble_out()
        if out is not None and (out.dtype != I64 or not out.is_contiguous()):
            raise ValueError("the park's bubble field: a contiguous int64 tensor")
        gl = self.view.global_flags()                       # the global flags, or None
        if gl is not None and (gl.dtype != BOOL or not gl.is_contiguous() or gl.device != self.dev
                               or tuple(gl.shape) != (self.view.R,)):
            raise ValueError("the park's global flags: a contiguous bool tensor [R] on the layer's device")
        s = _Park(self.view.R, *(t[n].data_ptr() for n in _PARK_TYPES), out.data_ptr() if out is not None else None,
                  gl.data_ptr() if gl is not None else None, self.view.friction, self.view.dt)
        return s, keep + ([out] if out is not None else []) + ([gl] if gl is not None else [])

    def _space_struct(self) -> _Space:
        st = self.st
        ptr = lambda name: (st[name].data_ptr() if st.get(name) is not None else None)   # noqa: E731
        vals = [self.W, self.N, self.T, self.E, self.D, self.M, self.S, self.I, self.rows.data_ptr(),
                self.by_id.data_ptr()]
        vals += [ptr(n) for n in _SPACE_PTRS[2:]]
        return _Space(*vals)

    def _stream(self):
        return ctypes.c_void_p(torch.cuda.current_stream(self.dev).cuda_stream) if self.kernel == "cuda" else None

    def tape(self, code, a=None, b=None, f=None) -> None:
        """Per world, K operations in order: code [W, K] (OP), a [W, K] (a slot), b [W, K] (an int), f [W, K, 4]."""
        code = torch.as_tensor(code, dtype=I64).to(self.dev).reshape(self.W, -1).contiguous()
        K = code.shape[1]
        a = (torch.zeros_like(code) - 1 if a is None else torch.as_tensor(a, dtype=I64).to(self.dev)
             .reshape(self.W, K).contiguous())
        b = torch.zeros_like(code) if b is None else torch.as_tensor(b, dtype=I64).to(self.dev).reshape(self.W, K) \
            .contiguous()
        f = (torch.zeros(self.W, K, 4, dtype=F64, device=self.dev) if f is None else
             torch.as_tensor(f, dtype=F64).to(self.dev).reshape(self.W, K, 4).contiguous())
        park, keep = self._park_struct()
        space = self._space_struct()
        t = _Tape(K, code.data_ptr(), a.data_ptr(), b.data_ptr(), f.data_ptr())
        lib = self.lib()["lib"]
        if self.kernel == "cuda":
            rc = lib.xbps_tape_cuda(ctypes.byref(park), ctypes.byref(space), ctypes.byref(t), self._stream(),
                                    self.block)
        else:
            rc = lib.xbps_tape_host(ctypes.byref(park), ctypes.byref(space), ctypes.byref(t))
        if rc:
            raise RuntimeError(f"the space tape failed ({rc})")
        del keep

    def ops(self, per_world: list) -> None:
        """A tape from per-world lists of (op name, slot, int, floats)."""
        K = max((len(x) for x in per_world), default=0)
        if K == 0:
            return
        code = torch.zeros(self.W, K, dtype=I64)
        a = torch.full((self.W, K), -1, dtype=I64)
        b = torch.zeros(self.W, K, dtype=I64)
        f = torch.zeros(self.W, K, 4, dtype=F64)
        for w, lst in enumerate(per_world):
            for k, op in enumerate(lst):
                name = op[0]
                slot = op[1] if len(op) > 1 else -1
                n = op[2] if len(op) > 2 else 0
                fl = op[3] if len(op) > 3 else ()
                code[w, k] = OP[name]
                a[w, k] = int(slot)
                b[w, k] = int(n)
                for i, x in enumerate(fl):
                    f[w, k, i] = float(x)
        self.tape(code, a, b, f)

    def broadcast(self, name: str, mask: torch.Tensor | None = None) -> None:
        """One operation without arguments for every world (or the worlds in mask [W]), on the device."""
        code = torch.full((self.W, 1), OP[name], dtype=I64, device=self.dev)
        if mask is not None:
            code = torch.where(mask.to(self.dev).reshape(self.W, 1), code, torch.zeros_like(code))
        self.tape(code)

    def query(self, kind: str, a=None, b=None, f=None) -> tuple:
        """A query for every world: (out [W, 1 + N], outf [W, N])."""
        W, N, d = self.W, self.N, self.dev
        a = torch.full((W,), -1, dtype=I64, device=d) if a is None else torch.as_tensor(a, dtype=I64).to(d) \
            .reshape(W).contiguous()
        b = torch.zeros(W, dtype=I64, device=d) if b is None else torch.as_tensor(b, dtype=I64).to(d).reshape(W) \
            .contiguous()
        ff = torch.zeros(W, 5, dtype=F64, device=d)
        if f is not None:
            f = torch.as_tensor(f, dtype=F64).to(d).reshape(W, -1)
            ff[:, :f.shape[1]] = f
        out = torch.zeros(W, 1 + N, dtype=I64, device=d)
        outf = torch.zeros(W, N, dtype=F64, device=d)
        park, keep = self._park_struct()
        space = self._space_struct()
        q = _Query(Q[kind], a.data_ptr(), b.data_ptr(), ff.data_ptr(), out.data_ptr(), outf.data_ptr())
        lib = self.lib()["lib"]
        if self.kernel == "cuda":
            rc = lib.xbps_query_cuda(ctypes.byref(park), ctypes.byref(space), ctypes.byref(q), self._stream(),
                                     self.block)
        else:
            rc = lib.xbps_query_host(ctypes.byref(park), ctypes.byref(space), ctypes.byref(q))
        if rc:
            raise RuntimeError(f"the space query failed ({rc})")
        del keep
        return out, outf

    # ---- the load and the tick ----

    def load(self) -> None:
        """AddBall of every ball, each world's slots in order (the master files each, InsertBallInBoxes): one tape,
        built on the device."""
        code = torch.where(self.rows >= 0, OP["ADD"], OP["NONE"]).to(I64)
        slots = torch.arange(self.N, dtype=I64, device=self.dev).expand(self.W, self.N).contiguous()
        self.tape(code, slots)

    def dead_rows(self, worlds: torch.Tensor | None = None) -> torch.Tensor:
        """Phase 1's dead-bubble flags [R] (TIME-714), for the park's tick to skip (no host synchronisation); `worlds`
        [W] (OnTick's worlds with a tick) limits the operation to them."""
        self.broadcast("DEAD", worlds)
        out = torch.zeros(self.view.R + 1, dtype=BOOL, device=self.dev)
        out.scatter_(0, self._flat, self.st["dead"].flatten().to(BOOL))
        return out[:self.view.R]

    def row_mask(self, worlds: torch.Tensor) -> torch.Tensor:
        """[R] the park's rows of the worlds in `worlds` [W] (OnTick's mask as the rows a park's tick
        takes; every row is a slot of its world, the padding scattered past R and dropped; no host synchronisation)."""
        out = torch.zeros(self.view.R + 1, dtype=BOOL, device=self.dev)
        out.scatter_(0, self._flat, worlds.to(BOOL).to(self.dev).reshape(self.W, 1).expand(self.W, self.N).flatten())
        return out[:self.view.R]

    def after_evolve(self, dropped_rows: torch.Tensor | None = None, worlds: torch.Tensor | None = None) -> None:
        """After the park's tick: the warp drop-outs' inserts, the commits' re-filings, HandleProximities (no host
        synchronisation); `worlds` [W] limits it to them."""
        dr = self.st["dropped"]
        if dropped_rows is None:
            dr.zero_()
        else:
            pad = torch.cat([dropped_rows.to(self.dev).to(torch.uint8), torch.zeros(1, dtype=torch.uint8,
                                                                                  device=self.dev)])
            dr.copy_(pad[self._flat].reshape(self.W, self.N))
        self.broadcast("AFTER_EVOLVE", worlds)

    def tick(self, park, worlds: torch.Tensor | None = None) -> None:
        """One tick of the park with this layer, as Destiny's Evolve runs it on the master: the dead bubbles' balls
        left out of the park's tick (TIME-714), then the warp drop-outs' and the commits' re-filings and
        HandleProximities; no host synchronisation (VecModesPark, and VecCollidePark on its kernel or with
        contact_slots: capturable as a CUDA graph, `SpaceGraph`). Over VecCollidePark (CollideView)
        the park's filing is the layer's before and after its tick, and its collisions skip other bubbles' balls
        through the park's bubble field, which the layer keeps. `worlds` [W] (OnTick's per-world mask)
        ticks those worlds alone, the others left as they are, the layer's operations and the park's tick alike
        (VecCollidePark's `active`; VecModesPark's and VecOrientPark's `evolve(active)` on the worlds'
        rows, `row_mask`, inside the dead bubbles' skip, so a dead bubble's rows of a world with a tick advance their
        clock and a world without a tick keeps every row, its clock included)."""
        from .vec_collide import VecCollidePark
        from .vec_modes import VecModesPark
        dead = self.dead_rows() if worlds is None else self.dead_rows(worlds)     # (the call as before, unmasked)
        self.last_dead = dead                   # the rows this tick left out (the client's view reads it)
        if isinstance(park, VecModesPark):
            from .modes import WARP as _WARP
            warp_proper = (park.mode == _WARP) & (park.effect_stamp >= 0) & park.free & ~park.moribund & ~dead
            if worlds is None:                              # the calls as before
                evolve_skipping(park, dead)
                self.after_evolve(warp_proper & (park.mode != _WARP))
            else:                                           # OnTick's worlds alone
                evolve_skipping(park, dead, rows=self.row_mask(worlds))
                self.after_evolve(warp_proper & (park.mode != _WARP), worlds)
        elif isinstance(park, VecCollidePark):
            if not isinstance(self.view, CollideView) or self.view.park is not park:
                raise ValueError("VecCollidePark's tick with the layer needs the layer on its CollideView")
            self.view.sync_filing(self)
            if worlds is None:                              # the calls as before (a test may replace these functions)
                evolve_skipping(park, dead)
                self.after_evolve(None)
            else:
                evolve_skipping(park, dead, worlds)
                self.after_evolve(None, worlds)
            self.view.sync_filing(self)
        elif worlds is None:
            evolve_skipping(park, dead | self.view.tensors()["moribund"])
            self.after_evolve(None)
        else:                                               # VecOrientPark, OnTick's worlds alone
            evolve_skipping(park, dead | self.view.tensors()["moribund"], rows=self.row_mask(worlds))
            self.after_evolve(None, worlds)

    def clear_events(self) -> None:
        self.st["evn"].zero_()

    def check(self) -> None:
        """Raise on a world flagged outside this layer's domain (FLAG_*), or on the park's view's own refusal."""
        self.view.check()
        bad = torch.nonzero(self.st["flags"]).flatten().tolist()
        if bad:
            fl = int(self.st["flags"][bad[0]])
            names = [n for n, bit in FLAG.items() if fl & bit]
            raise RuntimeError(f"worlds {bad[:8]} left this layer's domain (world {bad[0]}: {names})")

    # ---- readers (host) ----

    def slot_ids(self, w: int) -> dict:
        ids = self.view.tensors()["ids"].cpu()
        return {j: int(ids[int(r)]) for j, r in enumerate(self.rows[w].tolist()) if r >= 0}

    def dicts(self, w: int) -> dict:
        """The bubble dictionaries of world w: {bubble: [(ball id, action)] in insertion order}."""
        st = {k: (v[w].cpu() if v is not None else None) for k, v in self.st.items()}
        ids = self.slot_ids(w)
        out = {int(b): [] for b in st["dbub"][:int(st["dn"])].tolist()}
        n = int(st["en"])
        ent = sorted(zip(st["eseq"][:n].tolist(), st["ebub"][:n].tolist(), st["eslot"][:n].tolist(),
                         st["eact"][:n].tolist()))
        for _, bub, slot, act in ent:
            out.setdefault(int(bub), []).append((ids[int(slot)], int(act)))
        return out

    def tops(self, w: int) -> dict:
        """The top boxes that exist in world w: {key: bubble}."""
        n = int(self.st["tn"][w])
        keys = self.st["tkey"][w, :n].cpu().tolist()
        bub = self.st["tbub"][w, :n].cpu().tolist()
        return {tuple(k): int(b) for k, b in zip(keys, bub)}

    def _pairs(self, w: int, b: str, s: str, n: str) -> dict:
        ids = self.slot_ids(w)
        k = int(self.st[n][w])
        out = {}
        for bub, slot in zip(self.st[b][w, :k].tolist(), self.st[s][w, :k].tolist()):
            out.setdefault(int(bub), set()).add(ids[int(slot)])
        return out

    def interactives(self, w: int) -> dict:
        """bubbleInteractives of world w: {bubble: set of ball ids}."""
        return self._pairs(w, "iab", "ias", "ian")

    def keep_alives(self, w: int) -> dict:
        """bubbleKeepAlives of world w: {bubble: set of ball ids}."""
        return self._pairs(w, "kab", "kas", "kan")

    def proximity_order(self, w: int) -> list:
        ids = self.slot_ids(w)
        return [ids[int(j)] for j in self.st["pmap"][w, :int(self.st["pn"][w])].tolist()]

    def sensor_list(self, w: int, slot: int) -> list:
        ids = self.slot_ids(w)
        return [ids[int(j)] for j in self.st["slist"][w, slot, :int(self.st["slen"][w, slot])].tolist()]

    def proximity_buckets(self, w: int) -> tuple:
        """The proximity map's (bucket count, next resize): space.StdHashMap's bucket_count and next_resize."""
        return int(self.st["pbkt"][w]), int(self.st["pnr"][w])

    def sensor_buckets(self, w: int, slot: int) -> tuple:
        return int(self.st["sbkt"][w, slot]), int(self.st["snr"][w, slot])

    def events(self, w: int) -> list:
        """The world's events since the last clear: (name, ball id, a[, b])."""
        n = int(self.st["evn"][w])
        out = []
        for kind, ball, a, b in self.st["ev"][w, :n].cpu().tolist():
            name = EVENTS[kind]
            out.append((name, ball, a, b) if name == "DoProximity" else (name, ball, a) if name != "DoFinalCleanup"
                       else (name, ball))
        return out

    def hit_counts(self) -> dict:
        h = self.st["hits"]
        if h is None:
            return {}
        tot = h.sum(dim=0).cpu().tolist()
        return {name: int(v) for name, v in zip(HITS, tot) if v}


# ============================================================================================ the park's tick

def evolve_skipping(park, skip_rows: torch.Tensor, worlds: torch.Tensor | None = None,
                    rows: torch.Tensor | None = None) -> None:
    """The park's tick with the rows in skip_rows (dead bubbles, TIME-714) left out of it, as Destiny's tick leaves
    them: VecModesPark's kernel skips a moribund row in every phase (as Destiny's Evolve skips a dead or moribund
    ball), so those rows are flagged moribund for the call; VecCollidePark's tick takes them as its skip (rows w x N +
    slot; its balls are not independent, so restoring rows would not do); any other park's rows are
    independent (VecOrientPark), so their state is restored after its tick. `worlds` [W]: only those
    worlds tick (VecCollidePark's `active`; refused for the other parks). `rows` [n] (OnTick's mask as
    rows, a world's rows together): only those rows tick, VecModesPark's and VecOrientPark's `evolve(active)` (every other row keeps every per-row tensor, its clock included), inside the moribund flags and before the
    restore of the skipped rows."""
    from .vec_collide import VecCollidePark
    from .vec_modes import VecModesPark
    if isinstance(park, VecCollidePark):
        if rows is not None:
            raise ValueError("VecCollidePark's tick takes OnTick's mask by its worlds (`worlds`), not rows")
        if worlds is None:
            park.evolve(skip=skip_rows.reshape(park.W, park.N))
        else:
            park.evolve(active=worlds, skip=skip_rows.reshape(park.W, park.N))
        return
    if worlds is not None:
        raise ValueError("a tick of some worlds alone (OnTick's mask) by worlds runs over VecCollidePark only (the "
                         "other parks take `rows`)")
    evolve = park.evolve if rows is None else (lambda: park.evolve(active=rows))
    if isinstance(park, VecModesPark):
        saved = park.moribund
        park.moribund = (saved | skip_rows).contiguous()
        try:
            evolve()
        finally:
            park.moribund = saved
        return
    if skip_rows.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("only VecModesPark's tick is captured with the layer (VecOrientPark restores rows on the "
                           "host's decision)")
    n = park.n
    if not bool(skip_rows.any()):
        evolve()
        return
    before = {k: v.clone() for k, v in vars(park).items()
              if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n}
    evolve()
    if True:
        for k, old in before.items():
            cur = getattr(park, k)
            if torch.is_tensor(cur) and cur.shape == old.shape:
                cur = cur.clone()
                cur[skip_rows] = old[skip_rows]
                setattr(park, k, cur)


class SpaceGraph:
    """A tick of VecModesPark with its VecSpace layer (`VecSpace.tick`) captured once as a CUDA graph and replayed:
    the park's kernel and the layer's state are updated in place, so the state carries from replay to replay; the
    park's clock moves on the host as its own evolve does. Exact by construction (the same launches); held to eager
    ticks bit for bit by tests/test_movement_vec_space.py. An eager operation between replays (a tape, a setter)
    must update the tensors in place: a setter that rebinds one of the park's tensors invalidates the graph
    (`stale()`).

    Over VecCollidePark (the gradient method on its kernel, kernel="cuda", or on its torch path with
    contact_slots: otherwise its tick waits for the host), whose tick rebinds its tensors: the park's state lives in
    fixed buffers, the capture ends by copying each rebound tensor back into its buffer, and `tick()` first copies a
    tensor rebound since (an eager setter between replays) into its buffer, as the ServerGraph does; held to eager
    ticks bit for bit by tests/test_movement_vec_space_collide.py."""

    def __init__(self, space: VecSpace, park, warm: int = 0):
        from .vec_collide import VecCollidePark
        from .vec_modes import VecModesPark
        collide = isinstance(park, VecCollidePark)
        if not (isinstance(park, VecModesPark) or collide) or space.dev.type != "cuda":
            raise ValueError("SpaceGraph: a VecModesPark or a VecCollidePark on a CUDA device")
        if collide and (park.iterative or (park.kernel != "cuda" and park.contact_slots is None)):
            raise ValueError("SpaceGraph over VecCollidePark: the gradient method on its kernel or with contact_slots "
                             "(its torch path otherwise waits for the host)")
        self.space, self.park, self.collide = space, park, collide
        for _ in range(warm):
            space.tick(park)
        self.state: list = []                               # VecCollidePark: (attribute, buffer)
        if collide:
            if park.kernel == "cuda":                       # every kernel's module loaded before the capture
                from . import vec_collide_cuda
                with torch.cuda.device(space.dev):
                    vec_collide_cuda.info()
            space.view.sync_filing(space)                   # its slots' rows found once on the host, before it
            bufs: dict = {}
            for name, v in list(vars(park).items()):
                if not torch.is_tensor(v) or v.device.type != "cuda":
                    continue
                if id(v) not in bufs:
                    bufs[id(v)] = v.contiguous().clone()
                setattr(park, name, bufs[id(v)])
                self.state.append((name, bufs[id(v)]))
            self._buf_ids = {id(b) for _, b in self.state}
        self._ptrs = self._pointers()
        torch.cuda.synchronize(space.dev)
        self.graph = torch.cuda.CUDAGraph()
        t0 = park.current_time
        with torch.cuda.graph(self.graph):
            space.tick(park)
            if collide:
                self._copy_back()
        park.current_time = t0                              # the capture ran no tick

    def _pointers(self) -> tuple:
        return tuple(v.data_ptr() for v in vars(self.park).values() if torch.is_tensor(v))

    def _copy_back(self) -> None:
        """VecCollidePark: each state attribute's current tensor into its buffer (a current tensor that is another
        attribute's buffer is cloned first: the commit's swap), then rebound to it (ServerGraph._copy_back's way)."""
        pairs = []
        for name, b in self.state:
            cur = getattr(self.park, name)
            if cur is b:
                continue
            pairs.append((name, b, cur.clone() if id(cur) in self._buf_ids else cur))
        for name, b, cur in pairs:
            b.copy_(cur)
        for name, b in self.state:
            setattr(self.park, name, b)

    def stale(self) -> bool:
        return self._pointers() != self._ptrs

    def tick(self) -> None:
        if self.collide:
            self._copy_back()                               # an eager setter's tensor into its buffer
        elif self.stale():
            raise RuntimeError("a tensor of the park was rebound since the capture: capture again")
        self.graph.replay()
        self.park.current_time += 1

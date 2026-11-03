# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's space on the master batched (exact_ballpark.vec_space): the partition's
filing, the bubble partition, the proximity sensors and the queries for many worlds, against the scalar reference
(exact_ballpark.full.FullBallpark with is_master=True, which tests/test_movement_space_oracle.py and
tests/test_movement_full_oracle.py hold to Destiny's code compiled verbatim), bit for bit, after every step: each
ball's bubble, old bubble, filing, set memberships, position and velocity, sensor, tracking and notification state and
sensor list in order; each world's bubble dictionaries in insertion order, interactive and keep-alive sets, bubble id
counter, top boxes with their bubbles, proximity balls in order, events in order and query answers
(tests/space_batch.py).

On the CPU: the generated header, the source scan, the build's PTX and host checks, the layer's host path over
VecOrientPark on generated worlds (development seed 1 and held-out seed 2), the branches reached (each count equal to
the reference's), a world alone and reordered, Destiny's findings pinned (each on both sides), the refusals, and ten
planted faults (each built into its own library, each caught). On the GPU (skipped without one): the kernel over
VecModesPark (warps, follows and orbits too) and over VecOrientPark, the kernel against the host path, worlds tiled,
block sizes, and the GPU-only planted fault.
"""

from __future__ import annotations

import ctypes
import math
import os
import re
import shutil

import pytest
import torch

import space_batch as SB
from exact_ballpark import vec_space as V
from exact_ballpark.space import LEVEL_GRID, LEVEL_WIDTH, NUM_LEVELS

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
NVCC = pytest.mark.skipif(not (V._K.NVCC.exists() or shutil.which("nvcc")), reason="nvcc not found")
TOP = LEVEL_WIDTH[0]
_RUNS: dict = {}


def run(seed: int, n: int = 300, park: str = "orient", steps: int = 20, device: str = "cpu") -> dict:
    k = (seed, n, park, steps, device)
    if k not in _RUNS:
        _RUNS[k] = SB.compare(SB.cases(seed, n, park, steps), device, park)
    return _RUNS[k]


def assert_exact(res: dict, minimum: dict) -> None:
    assert not res["flagged"], f"worlds left the domain: {res['flagged'][:10]}"
    assert not res["diffs"], [str(d)[:400] for d in res["diffs"][:6]]
    for key, low in minimum.items():
        assert res["compared"].get(key, 0) >= low, (key, res["compared"])


# ==================================================================================================== on the CPU

def test_the_header_holds_the_references_constants():
    """The partition's level widths and grids, the box allocator's 256, the modes' codes, the commit's sqrt(0.1)
    threshold and the tracking minimum come from the reference's modules; the tracking literal is found in space.py's
    source with its spelling; the walk's code base and span equal exact_ballpark.vec_collide's; a perturbed value changes
    one line."""
    from exact_ballpark import full, modes, ref, space, vec_collide
    h = V.tables_header()
    for i in range(NUM_LEVELS):
        assert f"#define K_LW{i} 0x{V._dbits(LEVEL_WIDTH[i]):016x}ULL\n" in h
        assert f"#define K_LG{i} ({LEVEL_GRID[i]}LL)\n" in h
    for name, val in (("K_BOX_CHUNK", space.BOX_CHUNK_SIZE), ("K_MODE_WARP", modes.WARP), ("K_MODE_GOTO", ref.GOTO),
                      ("K_MODE_FOLLOW", modes.FOLLOW), ("K_MODE_ORBIT", modes.ORBIT), ("K_UMAP_BUCKETS", 13)):
        assert f"#define {name} ({val}LL)\n" in h
    assert f"#define K_MIN_MOVE_SQ 0x{V._dbits(full.DISPATCH_MIN_MOVE_SQ):016x}ULL\n" in h
    assert all(V.literals_in_source().values()), V.literals_in_source()
    assert (V.CODE_BASE, V.CODE_SPAN, V.NOT_A_CANDIDATE) == (vec_collide.CODE_BASE, vec_collide.CODE_SPAN,
                                                             vec_collide.NOT_A_CANDIDATE)
    p = V.tables_header({"K_MIN_MOVE_SQ": 0.2})
    assert sum(a != b for a, b in zip(h.splitlines(), p.splitlines())) == 1


def test_the_source_calls_no_cuda_math():
    """The source scan (exact_ballpark.vec_orient_cuda's) finds nothing, and flags a planted call."""
    from exact_ballpark import vec_orient_cuda as K
    text = V.SOURCE.read_text()
    assert K.scan_source(text) == []
    planted = text.replace("return __builtin_sqrt(a);", "return sqrt(a);")
    assert any(name == "sqrt" for name, _ in K.scan_source(planted))


@NVCC
def test_the_build_pins_every_rounding():
    """The library builds with the pinned flags: its PTX passes the check (every add, sub, mul, div, sqrt and rcp
    rounded to nearest; no mad, flush or approximation) and its host code has no fused instruction; the kernels'
    resources recorded. (Without nvcc the build is g++'s host functions: the host-build tests below.)"""
    m = V.build()
    assert m["ptx_check"]["ok"] and not m["source_scan"], m["ptx_check"]
    assert m["host_check"]["ok"] and m["host_check"]["functions"] > 0, m["host_check"]
    assert any("tape_kernel" in ln for ln in m["ptxas"])
    assert not any(k.startswith(("fma", "mad")) for k in m["ptx_check"]["fp_ops"])


def test_the_host_check_flags_a_fused_instruction():
    """host_check reads a library's disassembly; a fused instruction in one of this module's functions is refused."""
    text = ("0000000000001000 <xbps::insert_in_boxes(xbps::Ctx const&, long)>:\n    1000:\tfmul\td0, d1, d2\n"
            "0000000000002000 <xbps_tape_host>:\n    2000:\tfmadd\td0, d1, d2, d3\n"
            "0000000000003000 <other>:\n    3000:\tfmadd\td0, d1, d2, d3\n")
    out = V._host_fused(text)
    assert out == {"functions": 2, "fused": {"xbps_tape_host": 1}}


# ---- the host functions built by g++ where nvcc is absent ----

CUDA_WORDS = ("__global__", "__device__", "__host__", "__constant__", "__shared__", "__managed__", "__launch_bounds__",
              "__forceinline__", "__noinline__", "__grid_constant__")


def _as_without_cuda(text: str) -> str:
    """The source as a compiler without CUDA reads it, comments left out: each `#ifdef __CUDACC__` or
    `#ifdef __CUDA_ARCH__` block dropped up to its #else (whose part is kept) or its #endif, nested #if's counted."""
    out, state, depth = [], None, 0
    for line in text.splitlines():
        s = line.strip()
        if state is None and s in ("#ifdef __CUDACC__", "#ifdef __CUDA_ARCH__"):
            state, depth = "cuda", 0
            continue
        if state is not None and s.startswith("#if"):
            depth += 1
        elif state is not None and s.startswith("#endif"):
            if depth == 0:
                state = None
                continue
            depth -= 1
        elif state == "cuda" and depth == 0 and s.startswith("#else"):
            state = "host"
            continue
        if state != "cuda":
            out.append(re.sub(r"//.*", "", line))
    return "\n".join(out)


def test_the_cuda_words_g_plus_plus_meets_are_the_host_defines():
    """The source's kernels and launchers are behind its own __CUDACC__ guards and its device arithmetic behind
    __CUDA_ARCH__; outside them the only CUDA qualifiers are HD's __host__ and __device__ and the device tables'
    __constant__, the three `HOST_DEFINES` define empty for g++ (so the source itself is unchanged). Every CUDA word
    g++ meets is one of them; a copy of the source with a clean-room region may meet fewer of them."""
    text = _as_without_cuda(V.SOURCE.read_text())
    words = set(re.findall(r"__[a-z_]+__", text)) & set(CUDA_WORDS)
    defined = {d[2:].split("=")[0] for d in V.HOST_DEFINES}
    assert defined == {"__host__", "__device__", "__constant__"}, defined
    assert words <= defined and {"__host__", "__device__"} <= words, words
    assert all(d.endswith("=") for d in V.HOST_DEFINES)
    assert "<<<" not in text and "cudaStream_t" not in text and "blockIdx" not in text


def test_the_host_flags_are_those_nvcc_gives_its_host_compiler():
    """nvcc hands its host compiler -O3, -std=c++17 and the -Xcompiler list of vec_orient_cuda.FLAGS (`nvcc --dryrun`:
    gcc -std=c++17 ... -fPIC -ffp-contract=off -fno-fast-math -O3); HOST_FLAGS are those and -shared, nothing else, so
    the floating point is pinned alike: no contraction into a fused multiply-add, no fast math."""
    F = V._K.FLAGS
    xc = F[F.index("-Xcompiler") + 1].split(",")
    assert "-O3" in F and "-std=c++17" in F and "-shared" in F
    assert V.HOST_FLAGS == ["-O3", "-std=c++17", *xc, "-shared"]
    assert "-ffp-contract=off" in xc and "-fno-fast-math" in xc


def _hide_nvcc(monkeypatch, tmp_path) -> None:
    """No nvcc at vec_orient_cuda's path nor on PATH (as on a machine without the CUDA toolkit)."""
    monkeypatch.setattr(V._K, "NVCC", tmp_path / "no_nvcc")
    keep = [p for p in os.environ.get("PATH", "").split(os.pathsep)
            if p and not os.path.exists(os.path.join(p, "nvcc"))]
    monkeypatch.setenv("PATH", os.pathsep.join(keep))
    with pytest.raises(RuntimeError, match="nvcc not found"):
        V._K._nvcc()


def test_without_nvcc_the_host_functions_are_built_by_g_plus_plus(tmp_path, monkeypatch):
    """With nvcc hidden, `build` compiles the host functions alone with g++ (`build_host`): the source as it is,
    nvcc's host flags, the host check passed; built once (a second build reuses it). The library holds the four host
    entry points and no kernel's; `load_library` types those only, and the kernels' paths refuse it with a reason."""
    _hide_nvcc(monkeypatch, tmp_path)
    out = tmp_path / "lib"
    m = V.build(out)
    assert m["host_only"] and m["variant"] == "host" and m["library"].startswith("vec_space_host_"), m
    assert m["host_check"]["ok"] and m["host_check"]["functions"] > 0 and not m["source_scan"], m["host_check"]
    assert m["flags"] == V.HOST_FLAGS and m["defines"] == V.HOST_DEFINES
    assert m["source_sha256"] == V._sha(V.SOURCE.read_text().encode()) and m["compiler"].startswith("g++")
    assert V.build(out)["built_utc"] == m["built_utc"]
    so = ctypes.CDLL(str(out / m["library"]))
    assert all(hasattr(so, f) for f in ("xbps_abi", "xbps_layout", "xbps_tape_host", "xbps_query_host"))
    assert not any(hasattr(so, f) for f in ("xbps_tape_cuda", "xbps_query_cuda", "xbps_info", "xbps_error_string"))
    monkeypatch.setenv("EXACT_BALLPARK_SPACE_CUDA_DIR", str(out))
    monkeypatch.setattr(V, "_LIBS", {})
    e = V.load_library()
    assert e["host_only"] and e["manifest"]["library"] == m["library"]
    assert all(a == b for a, b in V.layout_check(e).values())
    with pytest.raises(RuntimeError, match="without nvcc"):
        V.kernel_info(lib=e)
    bw = SB.BatchedWorlds(SB.cases(1, 4, "orient", 2), "cpu", "orient", lib=e)
    bw.space.kernel = "cuda"                        # as a layer on a card would be (nothing runs)
    with pytest.raises(RuntimeError, match="needs nvcc"):
        bw.space.lib()


def _same_bits(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.is_floating_point():
        it = torch.int64 if x.element_size() == 8 else torch.int32
        return torch.equal(x.contiguous().view(it), y.contiguous().view(it))
    return torch.equal(x, y)


@NVCC
def test_g_plus_plus_and_nvccs_host_compiler_give_the_same_bits():
    """The host functions of the two builds, nvcc's library (its host code compiled by GCC under nvcc) and g++'s
    (`load_library(host=True)`), side by side on the same worlds: every tensor of the layer's state and of the park
    after every step, bit for bit, and the events, the answers and the branch counts; VecOrientPark's worlds (seeds 1
    and 2) and VecModesPark's (seeds 1 and 7)."""
    a, b = V.load_library(), V.load_library(host=True)
    assert not a["host_only"] and b["host_only"]
    seen = {"tensors": 0, "events": 0, "answers": 0, "branches": 0}
    for park, seed, n, T in (("orient", 1, 300, 20), ("orient", 2, 300, 20), ("modes", 1, 200, 24),
                             ("modes", HELD_OUT, 200, 24)):
        cs = SB.cases(seed, n, park, T)
        A = SB.BatchedWorlds(cs, "cpu", park, hits=True, lib=a)
        B = SB.BatchedWorlds(cs, "cpu", park, hits=True, lib=b)
        for t in range(T):
            for X in (A, B):
                X.space.clear_events()
                X.step(t)
            ta = {**{("st", k): v for k, v in A.space.st.items() if isinstance(v, torch.Tensor)},
                  **{("park", k): v for k, v in A.view.tensors().items()}}
            tb = {**{("st", k): v for k, v in B.space.st.items() if isinstance(v, torch.Tensor)},
                  **{("park", k): v for k, v in B.view.tensors().items()}}
            assert ta.keys() == tb.keys()
            for k in ta:
                assert _same_bits(ta[k], tb[k]), (park, seed, t, k)
            seen["tensors"] += len(ta)
            for wi in range(len(cs)):
                ea = A.space.events(wi)
                assert ea == B.space.events(wi), (park, seed, t, wi)
                seen["events"] += len(ea)
            assert A.answers == B.answers, (park, seed, t)
            seen["answers"] += sum(len(x) for x in A.answers)
            A.answers, B.answers = [[] for _ in cs], [[] for _ in cs]
        assert A.space.hit_counts() == B.space.hit_counts(), (park, seed)
        seen["branches"] += sum(A.space.hit_counts().values())
    assert (seen["tensors"] >= 6776 and seen["events"] >= 1693 and seen["answers"] >= 7272
            and seen["branches"] >= 121384), seen                 # the run: these, exactly


def test_the_layer_matches_the_reference_development_worlds():
    """Seed 1: 300 worlds of 2 to 6 balls, 20 steps each, every record kind; the host path over VecOrientPark."""
    assert_exact(run(1), {"ball_fields": 200000, "world_fields": 30000, "events": 300, "answers": 1500})


HELD_OUT = 7                                                # first run after the generator was frozen


def test_the_layer_matches_the_reference_second_development_worlds():
    """Seed 2, run while the layer was written (a second development seed)."""
    assert_exact(run(2), {"ball_fields": 200000, "world_fields": 30000, "events": 300, "answers": 1500})


def test_the_layer_matches_the_reference_held_out_worlds():
    """The held-out seed, first run after the generator was frozen."""
    assert_exact(run(HELD_OUT), {"ball_fields": 200000, "world_fields": 30000, "events": 300, "answers": 1500})


SHARED = ("bubble.no_bubble", "bubble.new", "bubble.neighbour", "bubble.changed", "insert.unchanged",
          "insert.adjacent", "init.traversed", "nearby.no_interactives", "sensor.wait", "sensor.check",
          "sensor.vanished", "sensor.left", "sensor.entered", "sensor.dead_bubble_check", "range.event",
          "tracking.event", "tick.dead_bubble", "visibility.other_bubble", "visibility.occluded")


def test_the_worlds_reach_every_branch_with_the_references_counts():
    """Every branch the layer counts is reached on the development and held-out worlds, each as often as the
    reference reaches it (nearby.other_bubble counts the reference's collision searches too, so only reached here;
    range.moribund needs a delayed removal, outside the domain)."""
    for seed in (1, 2, HELD_OUT):
        res = run(seed)
        for name in SHARED:
            assert res["hits"].get(name, 0) > 0, (seed, name)
            assert res["hits"].get(name, 0) == res["ref_hits"].get(name, 0), (seed, name, res["hits"].get(name),
                                                                               res["ref_hits"].get(name))
        assert res["hits"].get("nearby.other_bubble", 0) > 0


def test_a_world_alone_and_reordered():
    """30 worlds in one batch, the same reversed, and each of five alone: every world's state the same each time."""
    cs = SB.cases(5, 30, "orient", 12)
    full_run = SB.BatchedWorlds(cs, "cpu")
    rev = SB.BatchedWorlds(list(reversed(cs)), "cpu")
    for t in range(12):
        full_run.space.clear_events()
        rev.space.clear_events()
        full_run.step(t)
        rev.step(t)
        for wi in range(30):
            assert SB.batched_state(full_run, wi) == SB.batched_state(rev, 29 - wi), (t, wi)
            assert full_run.space.events(wi) == rev.space.events(29 - wi)
    for wi in (0, 7, 13, 21, 29):
        alone = SB.BatchedWorlds([cs[wi]], "cpu")
        for t in range(12):
            alone.step(t)
        assert SB.batched_state(alone, 0) == SB.batched_state(full_run, wi)


# ---- Destiny's code as written: each finding on both sides ----

def _ball(bid, pos, interactive=True, vel=(0.0, 0.0, 0.0), radius=50.0, max_vel=200.0):
    return {"id": bid, "mass": 1e7, "radius": radius, "max_vel": max_vel, "agility": 0.5, "sf": 1.0, "pos": list(pos),
            "vel": list(vel), "interactive": interactive, "massive": False, "free": True, "occluder": False}


def _both(balls: list, steps: list, device: str = "cpu", park: str = "orient") -> tuple:
    """(the scalar world, the batched worlds, the scalar states after each step, the answers of each step) for one
    constructed world, both sides asserted equal after every step."""
    w = {"id": 0, "park": park, "balls": balls, "anchors": [], "steps": steps, "pair": None}
    sw = SB.ScalarWorld(w)
    bw = SB.BatchedWorlds([w], device, park)
    states, answers = [], []
    for t in range(len(steps)):
        for rec in steps[t]:
            sw.apply(rec)
        sw.evolve()
        bw.step(t)
        a, b = SB.scalar_state(sw), SB.batched_state(bw, 0)
        assert {k: v for k, v in a["world"].items() if k != "live_boxes"} == \
            {k: v for k, v in b["world"].items() if k != "box_bound"}, t
        assert a["balls"] == b["balls"], t
        assert sw.answers == bw.answers[0], t
        states.append(a)
        answers.append(sw.answers)
        sw.answers, bw.answers[0] = [], []
    return sw, bw, states, answers


def test_finding_initialize_bubbles_freezes_a_lone_moving_ball():
    """DESTINY_FINDINGS 1.4: InitializeBubbles adds and removes a ball whose bubble does not change in one AddToBubble,
    so it ends in no dictionary and no interactive set; its bubble counts as dead and the moving interactive ball stops
    being evolved (it freezes) from that tick on."""
    a = 11
    sw, bw, st, ans = _both([_ball(a, (1000.0, 2000.0, 3000.0), vel=(150.0, 0.0, 0.0))],
                       [[("goto", a, 1e7, 0.0, 0.0)], [], [("init", 0)], [], []])
    assert st[1]["balls"][a]["pos"] != st[0]["balls"][a]["pos"]           # flying
    assert st[2]["world"]["dicts"] == {0: []} and st[2]["world"]["interactives"] == {}    # an empty dictionary
    assert st[2]["balls"][a]["bubble"] == 0 and st[2]["balls"][a]["old_bubble"] == 0
    assert st[2]["balls"][a]["pos"] == st[3]["balls"][a]["pos"] == st[4]["balls"][a]["pos"]   # frozen
    assert sw.park.trace["tick.dead_bubble"] >= 2


def test_finding_a_lone_ship_takes_a_new_bubble_at_each_top_face():
    """DESTINY_FINDINGS 1.6: a ship alone in its top box releases it on leaving, so the next top finds no neighbour
    with a bubble and takes a new id: bubbles 0, 1, 2, 3 for three moves into the next top. With a second ship
    staying in the first top, the move into the adjacent top joins its bubble instead."""
    a, b = 21, 22
    y = 0.3 * TOP
    steps = [[("pos", a, 1.5 * TOP, y, y)], [("pos", a, 2.5 * TOP, y, y)], [("pos", a, 3.5 * TOP, y, y)]]
    sw, bw, st, ans = _both([_ball(a, (0.5 * TOP, y, y))], steps)
    assert [s["balls"][a]["bubble"] for s in st] == [1, 2, 3] and st[-1]["world"]["bubble_id"] == 4
    sw, bw, st, ans = _both([_ball(a, (0.5 * TOP, y, y)), _ball(b, (0.5 * TOP + 5e3, y, y))], steps[:1])
    assert st[0]["balls"][a]["bubble"] == 0 == st[0]["balls"][b]["bubble"]


def test_finding_range_queries_return_balls_that_left_until_copy_bubbles():
    """DESTINY_FINDINGS 1.5: GetBallIdsInRange walks the bubble's dictionary, whose removals (-1) stay until
    CopyBubbles: a ball that has moved into another bubble is still returned for its old one, then no longer. (c is
    filed first, bubble 0; a two tops away, bubble 1; b between them takes its first neighbour's, a's.)"""
    c, a, b = 31, 32, 33
    y = 0.3 * TOP
    balls = [_ball(c, (1.5 * TOP, y, y)), _ball(a, (-0.5 * TOP, y, y)), _ball(b, (0.5 * TOP - 1e3, y, y))]
    steps = [[("copy",)], [("pos", b, TOP + 1e3, y, y), ("range_ball", a, 3 * TOP, 0), ("copy",),
                           ("range_ball", a, 3 * TOP, 0)]]
    sw, bw, st, ans = _both(balls, steps)
    assert (st[0]["balls"][c]["bubble"], st[0]["balls"][a]["bubble"], st[0]["balls"][b]["bubble"]) == (0, 1, 1)
    assert st[1]["balls"][b]["bubble"] == 0
    assert ans[1] == [("range", [b]), ("range", [])]


def test_finding_update_ball_bubble_leaves_the_sets_behind():
    """New: PyInitializeBubbles(src) (UpdateBallBubble, Ballpark.cpp:5959) sets the ball's bubble
    to a nearby ball's without AddToBubble: the ball stays in its old bubble's dictionary and interactive set while its
    newBubbleId (which the server layer routes its events by, _baseticker.py:99-106) names another; after its next
    bubble change it is in two interactive sets."""
    a, b, c = 41, 42, 43
    y = 0.3 * TOP
    balls = [_ball(a, (-0.5 * TOP, y, y)), _ball(b, (1.5 * TOP, y, y)), _ball(c, (TOP - 3e3, y, y))]
    steps = [[("init", b)], [("pos", b, 5.5 * TOP, y, y)]]
    sw, bw, st, ans = _both(balls, steps)
    assert st[0]["balls"][b]["bubble"] == 0
    assert st[0]["world"]["interactives"] == {0: {a, c}, 1: {b}}
    assert [x[0] for x in st[0]["world"]["dicts"][1]] == [b]
    assert st[1]["balls"][b]["in_ia"] == [1, st[1]["balls"][b]["bubble"]]       # in two interactive sets


def test_finding_an_only_interactives_sensor_sees_other_bubbles():
    """SEN-702's notes: with onlyInteractives the candidate filter skips the same-bubble test (Partition.cpp:427-455),
    so a sensor sees an interactive ball across a top face in another bubble, which a plain sensor never lists."""
    c, d, x, e = 51, 52, 53, 54
    y = 0.3 * TOP
    balls = [_ball(c, (-0.5 * TOP, y, y)), _ball(d, (1.5 * TOP, y, y)), _ball(x, (TOP - 1e3, y, y)),
             _ball(e, (TOP + 1e3, y, y))]
    sw, bw, st, ans = _both(balls, [[("sensor", x, 5e3, 1.0, 1)], [], [("sensor", x, 5e3, 1.0, 0)], [], []])
    assert st[1]["balls"][x]["bubble"] != st[1]["balls"][e]["bubble"]
    assert e in st[1]["balls"][x]["sensor"][-1]
    assert e not in st[3]["balls"][x]["sensor"][-1]


# ---- the refusals ----

def test_the_layer_flags_what_it_leaves_out():
    """A ball at the grid's edge (Destiny clamps there) and a ball wider than a top box are flagged, and check()
    raises. A fourteenth proximity ball is flagged only on a layer built with the former fixed capacity (umap_cap 13,
    UMAP_FAST_CAP); by default a world holds as many as its slots, in libstdc++'s order,
    and that world equals the reference."""
    edge = (LEVEL_GRID[0] // 2 - 1) * TOP + 0.5 * TOP
    w1 = LEVEL_WIDTH[1]
    crowd = [_ball(k, (1.5 * w1 + k * 3e4, 1.5 * w1, 1.5 * w1), radius=6e4) for k in range(1, 15)]
    for balls, flag, cap in (([_ball(1, (edge, 0.0, 0.0))], "EDGE", None),
                             ([_ball(1, (0.0, 0.0, 0.0), radius=4e6)], "HUGE", None),
                             (crowd, "PROXIMITY", V.UMAP_FAST_CAP)):
        w = {"id": 0, "park": "orient", "balls": balls, "anchors": [], "pair": None,
             "steps": [[("notify", k["id"], 100.0) for k in balls]]}
        bw = SB.BatchedWorlds([w], "cpu", umap_cap=cap)
        bw.step(0)
        assert int(bw.space.st["flags"][0]) & V.FLAG[flag], flag
        with pytest.raises(RuntimeError):
            bw.space.check()
    w = {"id": 0, "park": "orient", "balls": crowd, "anchors": [], "pair": None,
         "steps": [[("notify", k["id"], 100.0) for k in crowd], [("sensor", 3, 5e5, 1.0, 0)], [], []]}
    res = SB.compare([w], "cpu")
    assert not res["flagged"] and not res["diffs"], (res["flagged"], res["diffs"][:2])
    bw = SB.BatchedWorlds([w], "cpu")
    for t in range(4):
        bw.step(t)
    assert len(bw.space.proximity_order(0)) == 14 and bw.space.proximity_buckets(0)[0] > V.UMAP_FAST_CAP
    good = SB.BatchedWorlds([{"id": 0, "park": "orient", "balls": [_ball(1, (0.0, 0.0, 0.0))], "anchors": [],
                              "pair": None, "steps": [[]]}], "cpu")
    good.step(0)
    good.space.check()


# ---- planted faults ----

FAULT_RULES = {"no_neighbour_bubble": "SEN-707 (NearbyBubbles)", "top_keeps_bubble": "SEN-707 (a released top)",
               "reentry_appended": "SEN-707 (AddToBubble's -1 to 0 in place)",
               "init_keeps_unchanged": "SEN-707 (InitializeBubbles' quirk)", "dead_ignored": "TIME-714",
               "walk_by_id": "SEN-702 (GetNearbyBalls' order)", "list_appended": "SEN-701 (unordered_map order)",
               "range_skips_left": "SEN-710", "copy_keeps_empty": "SEN-707 (CopyBubbles)",
               "commit_unsorted": "TIME-707 (ascending id)"}


@pytest.mark.parametrize("fault", sorted(FAULT_RULES))
def test_a_planted_fault_is_caught(fault):
    """Each fault built into its own library (the PTX variant: the host path is the same) differs from the reference
    on the development worlds."""
    lib = V.load_library("ptx", perturb={"XBPS_FAULT": V.FAULTS[fault]})
    res = SB.compare(SB.cases(1, 200, "orient", 16), "cpu", "orient", lib=lib)
    assert res["diffs"] or res["flagged"], fault
    assert not SB.compare(SB.cases(1, 40, "orient", 8), "cpu", "orient")["diffs"]     # the true library: none


# ==================================================================================================== on the GPU

@CUDA
def test_the_kernel_matches_the_reference_modes_worlds():
    """VecModesPark's worlds (every record kind and follows, orbits and warps between far top boxes, a ship in warp
    proper taking a new bubble each tick), development seed 1 and held-out seed 2, the kernel path."""
    for seed in (1, HELD_OUT):
        res = run(seed, 200, "modes", 24, "cuda")
        assert_exact(res, {"ball_fields": 100000, "world_fields": 15000, "events": 100, "answers": 800})
        for name in SHARED + ("warp.drop_out_insert",):
            assert res["hits"].get(name, 0) == res["ref_hits"].get(name, 0), (seed, name)
        assert res["hits"].get("warp.drop_out_insert", 0) > 0


def test_the_host_path_matches_the_reference_modes_worlds():
    """VecModesPark's worlds on the CPU (its tick on tensors, the layer's host path)."""
    for seed in (1, HELD_OUT):
        res = run(seed, 200, "modes", 24, "cpu")
        assert_exact(res, {"ball_fields": 100000, "world_fields": 15000, "events": 100, "answers": 800})
        for name in SHARED + ("warp.drop_out_insert",):
            assert res["hits"].get(name, 0) == res["ref_hits"].get(name, 0), (seed, name)
        assert res["hits"].get("warp.drop_out_insert", 0) > 0


@CUDA
def test_the_kernel_matches_the_reference_orient_worlds():
    """VecOrientPark on the GPU with the kernel path: the development and held-out worlds."""
    for seed in (1, HELD_OUT):
        assert_exact(run(seed, 300, "orient", 20, "cuda"), {"ball_fields": 200000, "world_fields": 30000})


@CUDA
def test_the_kernel_equals_the_host_path():
    """The same worlds on the CPU (host path) and the GPU (kernel): every state tensor of the layer and the park equal
    after every step, the events too."""
    cs = SB.cases(3, 120, "orient", 16)
    h, g = SB.BatchedWorlds(cs, "cpu"), SB.BatchedWorlds(cs, "cuda")
    for t in range(16):
        h.space.clear_events()
        g.space.clear_events()
        h.step(t)
        g.step(t)
        for k, v in h.space.st.items():
            if v is None:
                continue
            assert torch.equal(v, g.space.st[k].cpu()), (t, k)
        for k, v in h.view.tensors().items():
            assert torch.equal(v.view(torch.int64) if v.dtype == torch.float64 else v,
                               (g.view.tensors()[k].cpu().view(torch.int64) if v.dtype == torch.float64
                                else g.view.tensors()[k].cpu())), (t, k)


@CUDA
def test_the_kernel_on_tiled_worlds_and_block_sizes():
    """60 worlds tiled to 1,200 in one park: every copy's state equals its original's (and the originals the
    reference); blocks of 32 and 256 give the same state."""
    cs = SB.cases(4, 60, "modes", 12)
    res = SB.compare(cs, "cuda", "modes")
    assert_exact(res, {"ball_fields": 10000})
    tiled = SB.BatchedWorlds(cs * 20, "cuda", "modes")
    for t in range(12):
        tiled.step(t)
    first = [SB.batched_state(tiled, wi) for wi in range(60)]
    for k in range(1, 20):
        for wi in range(60):
            assert SB.batched_state(tiled, k * 60 + wi) == first[wi], (k, wi)
    a = SB.BatchedWorlds(cs, "cuda", "modes")
    b = SB.BatchedWorlds(cs, "cuda", "modes")
    a.space.block, b.space.block = 32, 256
    for t in range(12):
        a.step(t)
        b.step(t)
    for k, v in a.space.st.items():
        if v is not None:
            assert torch.equal(v, b.space.st[k]), k


@CUDA
def test_a_cuda_graph_of_the_tick_equals_eager_ticks():
    """VecModesPark worlds (several bubbles, flights across top faces, sensors, warps under way) set up eagerly, then
    12 ticks as replays of one captured graph (VecSpace.tick: the dead bubbles' skip, the park's kernel, the
    re-filings and HandleProximities) against 12 eager ticks: every tensor of the park and the layer equal after
    each tick, events too."""
    cs = SB.cases(6, 120, "modes", 4)
    a, b = SB.BatchedWorlds(cs, "cuda", "modes"), SB.BatchedWorlds(cs, "cuda", "modes")
    for t in range(4):
        a.step(t)
        b.step(t)
    g = V.SpaceGraph(b.space, b.park)
    for t in range(12):
        a.space.clear_events()
        b.space.clear_events()
        a.space.tick(a.park)
        g.tick()
        assert a.park.current_time == b.park.current_time
        for k, v in a.space.st.items():
            if v is not None:
                assert torch.equal(v, b.space.st[k]), (t, k)
        for k, v in a.view.tensors().items():
            w = b.view.tensors()[k]
            assert torch.equal(v.view(torch.int64), w.view(torch.int64)) if v.dtype == torch.float64 else \
                torch.equal(v, w), (t, k)
    assert int(a.space.st["evn"].sum()) > 0 or int(a.space.st["hits"].sum()) > 0


@CUDA
def test_the_kernel_resources_and_the_gpu_planted_fault():
    """The kernels load (their registers and stack recorded), and the drop-out's re-filing planted out (the GPU-only
    fault) is caught. In the generated worlds a warp's drop-out re-files its ship into the box its last commit chose
    (the ship crawls in its last ticks), so there only the branch count sees the fault. A slow ship whose exit point
    lies 5 cm below a box face, which it overshoots by about 11 cm on its last warp tick, is moved into the next box
    by the drop-out's re-filing alone: with the fault it stays filed in the wrong box."""
    info = V.kernel_info()
    assert info["tape"]["registers"] > 0 and info["tape"]["max_threads"] >= V.BLOCK
    lib = V.load_library("ptx", perturb={"XBPS_FAULT": V.FAULTS["no_drop_out_insert"]})
    res = SB.compare(SB.cases(1, 200, "modes", 24), "cuda", "modes", lib=lib)
    assert res["hits"].get("warp.drop_out_insert", 0) != res["ref_hits"].get("warp.drop_out_insert", 0)
    w7 = LEVEL_WIDTH[NUM_LEVELS - 1]
    face = math.floor(1e9 / w7) * w7                         # a face of the finest boxes near 1e9 m
    a = 7
    ship = _ball(a, (1000.0, 2000.0, 3000.0), max_vel=0.5)
    steps = [[("vel", a, 0.45, 0.0, 0.0), ("warp", a, face - 0.05, 2000.0, 3000.0, 0.0, 3000)]] + [[]] * 31
    w = {"id": 0, "park": "modes", "balls": [ship], "anchors": [], "steps": steps, "pair": None}
    good = SB.compare([w], "cuda", "modes")
    assert not good["diffs"]
    assert good["hits"].get("warp.drop_out_insert") == good["ref_hits"].get("warp.drop_out_insert") == 1
    bad = SB.compare([w], "cuda", "modes", lib=lib)
    assert any(d[3] == "filing" for d in bad["diffs"] if len(d) > 3), bad["diffs"][:3]

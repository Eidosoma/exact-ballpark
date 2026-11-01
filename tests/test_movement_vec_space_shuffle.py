# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Shuffled proximity sensors on the master batched: exact_ballpark.vec_space's per-world glibc rand() state and
AddProximitySensor's shuffle (a shuffled sensor's first period starts at floatrand(0, period)), against the scalar
reference (exact_ballpark.full.FullBallpark with is_master=True, one park a world; its generator,
exact_ballpark.space.GlibcRandom, and its shuffled sensors are held to Destiny's code compiled verbatim by
tests/test_movement_space_oracle.py), bit for bit after every step, each world's generator state included
(tests/space_shuffle_cases.py, over the frozen worlds).

On the CPU: the header's constants from the reference; srand from the reference's own generator for each world; the
host path over VecOrientPark on development seeds 101, 102 and 107 (107 was to be held out; it ran before the freeze was
recorded, so it counts as development) and held-out seed 113 (first run after the generator's freeze); the shuffle's
branch counted as the reference counts it; worlds without a shuffle draw nothing; three planted faults caught. On the
GPU (skipped without one): the kernel over VecModesPark (follows, orbits and warps too) on seed 101 and held-out seed
113, and over VecOrientPark on seed 102; a CUDA graph of the tick, shuffled sensors added eagerly between its replays,
equal to eager ticks every tensor; the planted faults on the kernel. Against Destiny's compiled code directly (the conformance kit's space harness, skipped where it is not built): the worlds as tests/space_cases.py's scenarios, each world's srand seed
its scenario's; the reference against the harness on every row kind, and the host path against the harness's ball rows
(added after the generators' freeze, on development seed 101 only).
"""

from __future__ import annotations

import random

import pytest
import torch

import space_batch as SB
import space_shuffle_cases as SC
from exact_ballpark import vec_space as V
from exact_ballpark.space import GlibcRandom

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
DEV = (101, 102, 107)
HELD_OUT = 113
FAULTS = ("shuffle_ignored", "rand_unshifted", "rand_no_feedback")
_RUNS: dict = {}


def run(seed: int, n: int, park: str, steps: int, device: str, lib: dict | None = None) -> dict:
    k = (seed, n, park, steps, device, id(lib))
    if k not in _RUNS:
        _RUNS[k] = SC.compare(SC.cases(seed, n, park, steps), device, park, lib=lib)
    return _RUNS[k]


def assert_exact(res: dict, minimum: dict) -> None:
    assert not res["flagged"], f"worlds left the domain: {res['flagged'][:10]}"
    assert not res["diffs"], [str(d)[:400] for d in res["diffs"][:6]]
    for key, low in minimum.items():
        assert res["compared"].get(key, 0) >= low, (key, res["compared"])
    assert res["hits"].get("sensor.shuffle", 0) == res["ref_hits"].get("sensor.shuffle", 0) > 0


# ==================================================================================================== on the CPU

def test_the_header_holds_the_generators_constants():
    """The table's degree and the state's width come from the reference's generator; RAND_MAX is the literal
    space.py divides by (its spelling found there); the shuffle's branch is counted."""
    h = V.tables_header()
    assert V.RAND_DEG == len(GlibcRandom(1).s) == 31 and V.RAND_WORDS == V.RAND_DEG + 2
    assert f"#define K_RAND_DEG ({V.RAND_DEG}LL)\n" in h and f"#define K_RAND_WORDS ({V.RAND_WORDS}LL)\n" in h
    assert f"#define K_RAND_MAX 0x{V._dbits(2147483647.0):016x}ULL\n" in h
    assert V.literals_in_source()["K_RAND_MAX"]
    assert "sensor.shuffle" in V.HITS and "#define H_SENSOR_SHUFFLE" in h


def test_srand_is_the_references_generator_for_each_world():
    """glibc_srand's rows are GlibcRandom's state (the table, fptr, rptr) for edge seeds (0, which glibc takes as 1;
    2**31 - 1; 2**31; 2**32 - 1; -1; past 32 bits) and 200 random ones; VecSpace(seed=...) holds them, an int for
    every world or one a world; a seed list of the wrong length is refused."""
    r = random.Random("srand-check")
    seeds = list(SC.SEED_EDGES) + [r.randrange(-2 ** 40, 2 ** 40) for _ in range(200)]
    got = V.glibc_srand(seeds).tolist()
    for s, row in zip(seeds, got):
        g = GlibcRandom(s)
        assert row == list(g.s) + [g.f, g.r], s
    assert V.glibc_srand([0]).tolist() == V.glibc_srand([1]).tolist()
    w = SB.cases(1, 3, "orient", 1)
    bw = SB.BatchedWorlds(w, "cpu")
    sp = V.VecSpace(bw.view, bw.space.rows.clone(), seed=[5, 6, 7])
    assert torch.equal(sp.st["rnd"], V.glibc_srand([5, 6, 7]))
    sp1 = V.VecSpace(bw.view, bw.space.rows.clone())
    assert torch.equal(sp1.st["rnd"], V.glibc_srand([1, 1, 1]))
    with pytest.raises(ValueError):
        V.VecSpace(bw.view, bw.space.rows.clone(), seed=[5, 6])


def test_the_layout_holds_the_generators_state():
    lay = V.layout_check()
    assert all(a == b for a, b in lay.values()), lay
    assert "Space.rnd" in lay


@pytest.mark.parametrize("seed", DEV)
def test_the_host_path_matches_the_reference(seed):
    """VecOrientPark's worlds with shuffled sensors, the host path, every step: the comparisons, each sensor's
    elapsed time by its bits, and each world's generator state."""
    res = run(seed, 120, "orient", 16, "cpu")
    assert_exact(res, {"ball_fields": 40000, "rand_states": 1900, "shuffled_records": 900, "events": 100})


def test_the_host_path_matches_the_reference_held_out():
    res = run(HELD_OUT, 120, "orient", 16, "cpu")
    assert_exact(res, {"ball_fields": 40000, "rand_states": 1900, "shuffled_records": 900, "events": 100})


def test_worlds_without_a_shuffle_draw_nothing():
    """the own worlds (no shuffle): every world's generator is where srand left it, and the shuffle's branch is
    never reached."""
    cs = SB.cases(1, 40, "orient", 8)
    bw = SB.BatchedWorlds(cs, "cpu", hits=True)
    for t in range(8):
        bw.step(t)
    assert torch.equal(bw.space.st["rnd"], V.glibc_srand([1] * len(cs)))
    assert bw.space.hit_counts().get("sensor.shuffle", 0) == 0


@pytest.mark.parametrize("fault", FAULTS)
def test_a_planted_fault_is_caught(fault):
    """Each fault built into its own library (the PTX variant; the host path is the same code): the shuffle ignored;
    rand() not shifted; the table never fed back. Each differs from the reference on development seed 101."""
    lib = V.load_library("ptx", perturb={"XBPS_FAULT": V.FAULTS[fault]})
    res = SC.compare(SC.cases(101, 40, "orient", 10), "cpu", "orient", lib=lib)
    assert res["diffs"] or res["flagged"], fault


# ======================================================================= against Destiny's compiled code (CPU)

HARNESS_FIELDS = ("bubble", "old_bubble", "box_level", "box_ix", "box_iy", "box_iz", "mod_x", "mod_y", "mod_z", "px",
                  "py", "pz", "vx", "vy", "vz", "radius", "mode", "sensor", "has_proximity", "only_interactives",
                  "sensor_range", "sensor_period", "sensor_elapsed", "sensor_balls", "notification_range",
                  "within_range", "tracking_range", "within_tracking", "tracking_ptr", "interactive", "cloaked")


def _scenarios(cs: list) -> list:
    """The worlds as tests/space_cases.py's scenarios (the mapping, tests/space_collide_cases.py's _as_record; an
    immediate removal; a shuffled sensor with its shuffle), each world's srand seed (its low 32 bits, srand's unsigned
    int) the scenario's: AddBall at step 0, each step t's records at step t + 1 before its tick."""
    import space_cases as sc
    import space_collide_cases as C

    def rec_of(step, rec):
        if rec[0] == "sensor" and len(rec) > 5:
            return sc._k(step, sc.SENSOR_OP, rec[1], b=int(rec[5]), w=int(rec[4]), x=rec[3], rr=rec[2])
        if rec[0] == "remove":
            return sc._k(step, sc.REMOVE_OP, rec[1], w=0)
        return C._as_record(sc, step, rec)

    out = []
    for i, w in enumerate(cs):
        recs = [("B", 0, {"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
                          "free": b["free"], "global": False, "massive": b["massive"], "interactive": b["interactive"],
                          "junk": False, "pos": list(b["pos"]), "vel": list(b["vel"]), "agility": b["agility"],
                          "sf": b["sf"]}) for b in w["balls"]]
        for t, step in enumerate(w["steps"]):
            recs += [k for k in (rec_of(t + 1, r) for r in step) if k is not None]
        out.append({"id": i + 1, "kind": "shuffle", "friction": 1.0e6, "tick": 1000, "steps": len(w["steps"]),
                    "new_orbit": False, "master": True, "seed": w["seed"] & 0xFFFFFFFF, "start": 0, "records": recs})
    return out


def harness_compare(cs: list, lib: dict | None = None) -> dict:
    """The reference against Destiny's compiled code on every row kind, and the host path's ball rows against the
    harness's: {"reference": differences, "diffs", "fields", "rows", "elapsed_nonzero"}."""
    import space_cases as sc
    import test_movement_space_oracle as SO
    scen = _scenarios(cs)
    want = sc.oracle(scen)
    got, undefined = sc.run_reference(scen)
    ref = sum(len(v) for v in SO.differences(want, got, undefined).values())
    idx = {f: i for i, f in enumerate(sc.S_FIELDS)}
    rows = {(r[0], r[1], r[2]): r for r in want["S"]}
    bw = SC.ShuffleBatchedWorlds(cs, "cpu", "orient", lib=lib)
    diffs, nf, nr, nz = [], 0, 0, 0
    for t in range(len(cs[0]["steps"])):
        bw.space.clear_events()
        bw.step(t)
        st = {k: v for k, v in bw.space.st.items() if v is not None}
        tv = bw.view.tensors()
        flags = st["flags"].tolist()
        for (wi, bid), (r, j) in bw.index.items():
            if flags[wi] or not st["in_park"][wi, j]:
                continue
            wr = rows[(wi + 1, t + 1, bid)]
            ids = bw.space.slot_ids(wi)
            trk = int(st["trk"][wi, j])
            mine = {"bubble": int(st["bubble"][wi, j]), "old_bubble": int(st["old_bubble"][wi, j]),
                    "box_level": int(st["lvl"][wi, j]), "radius": float(tv["radius"][r]), "mode": int(tv["mode"][r]),
                    "sensor": int(tv["sensor"][r]), "has_proximity": int(st["has_prox"][wi, j]),
                    "only_interactives": int(st["sonly"][wi, j]), "sensor_range": float(st["srange"][wi, j]),
                    "sensor_period": float(st["speriod"][wi, j]), "sensor_elapsed": float(st["selapsed"][wi, j]),
                    "sensor_balls": tuple(bw.space.sensor_list(wi, j)),
                    "notification_range": float(st["ntf_range"][wi, j]), "within_range": int(st["ntf_in"][wi, j]),
                    "tracking_range": float(st["trk_range"][wi, j]), "within_tracking": int(st["trk_in"][wi, j]),
                    "tracking_ptr": -1 if trk < 0 else ids[trk], "interactive": int(tv["interactive"][r]),
                    "cloaked": int(tv["cloaked"][r])}
            for i, ax in enumerate("xyz"):
                mine["box_i" + ax], mine["mod_" + ax] = int(st["key"][wi, j, i]), int(st["mod"][wi, j, i])
                mine["p" + ax], mine["v" + ax] = float(tv["pos"][r, i]), float(tv["vel"][r, i])
            nr += 1
            nz += mine["sensor_elapsed"] != 0.0
            for f in HARNESS_FIELDS:
                nf += 1
                if not sc.same(wr[idx[f]], mine[f]):
                    diffs.append((t, wi, bid, f, wr[idx[f]], mine[f]))
    return {"reference": ref, "diffs": diffs, "fields": nf, "rows": nr, "elapsed_nonzero": nz}


def _harness_built() -> bool:
    import space_cases as sc
    return sc.HARNESS.exists()


@pytest.mark.skipif(not _harness_built(), reason="the space harness is not built here")
def test_the_host_path_matches_destinys_compiled_code():
    """Development seed 101 (60 worlds x 12 steps): the reference against the harness on every row kind, and the host
    path's ball rows (filing, bubble, position and velocity bits, every sensor field with the elapsed time's bits and
    the list in order, notification and tracking) against the harness's, 0 differences; the table never fed back
    (planted) differs there."""
    cs = SC.cases(101, 60, "orient", 12)
    res = harness_compare(cs)
    assert res["reference"] == 0 and not res["diffs"], (res["reference"], res["diffs"][:4])
    assert res["fields"] > 50000 and res["elapsed_nonzero"] > 400, res["fields"]
    bad = harness_compare(SC.cases(101, 30, "orient", 8),
                          lib=V.load_library("ptx", perturb={"XBPS_FAULT": V.FAULTS["rand_no_feedback"]}))
    assert bad["diffs"]


# ==================================================================================================== on the GPU

@CUDA
def test_the_kernel_matches_the_reference_modes_worlds():
    """VecModesPark's worlds (follows, orbits and warps besides), the kernel, development seed 101 and held-out 107."""
    for seed in (101, HELD_OUT):
        res = run(seed, 120, "modes", 16, "cuda")
        assert_exact(res, {"ball_fields": 40000, "rand_states": 1900, "shuffled_records": 900, "events": 100})


@CUDA
def test_the_kernel_matches_the_reference_orient_worlds():
    res = run(102, 120, "orient", 16, "cuda")
    assert_exact(res, {"ball_fields": 40000, "rand_states": 1900, "shuffled_records": 900, "events": 100})


@CUDA
def test_a_graph_of_ticks_with_draws_between_equals_eager_ticks():
    """Two layers over their own VecModesPark: one ticks eagerly, the other replays one CUDA graph of the tick
    (vec_space.SpaceGraph); between ticks both get the same shuffled sensors (added, re-added, removed: the draws on
    the device, outside the graph). Every tensor of the layers and the parks equal after every tick."""
    cs = SC.cases(103, 64, "modes", 1)
    for w in cs:
        w["steps"] = [[]]
    a = SC.ShuffleBatchedWorlds(cs, "cuda", "modes")
    b = SC.ShuffleBatchedWorlds(cs, "cuda", "modes")
    g = V.SpaceGraph(b.space, b.park)
    r = random.Random("shuffle-graph")
    draws = 0
    for t in range(12):
        ops = [[] for _ in cs]
        for wi, w in enumerate(cs):
            for _ in range(r.choice((0, 1, 2))):
                bid = r.choice(w["balls"])["id"]
                if r.random() < 0.2:
                    ops[wi].append(("REMOVE_SENSOR", a.slot(wi, bid)))
                else:
                    ops[wi].append(("ADD_SENSOR", a.slot(wi, bid), r.choice((0, 1)),
                                    (r.uniform(500.0, 20000.0), r.choice(SC.PERIODS), r.choice((1.0, 2.0, -3.0)))))
                    draws += 1
        if any(ops):
            a.space.ops(ops)
            b.space.ops(ops)
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
    assert draws > 200 and not torch.equal(a.space.st["rnd"], V.glibc_srand([w["seed"] for w in cs]).cuda())
    assert int(a.space.st["evn"].sum()) > 0


@CUDA
@pytest.mark.parametrize("fault", FAULTS)
def test_a_planted_fault_is_caught_on_the_kernel(fault):
    lib = V.load_library("ptx", perturb={"XBPS_FAULT": V.FAULTS[fault]})
    res = SC.compare(SC.cases(101, 40, "modes", 10), "cuda", "modes", lib=lib)
    assert res["diffs"] or res["flagged"], fault

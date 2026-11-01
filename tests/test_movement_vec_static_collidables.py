# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's static collidables in the batched collision park: capsules and oriented boxes (COL-712) in the gradient method of
exact_ballpark.vec_collide.VecCollidePark, on its torch path and on the kernel (the statics build, XBPC_STATICS=1),
held bit for bit to the scalar reference (exact_ballpark.full.FullBallpark, a client's view), which is held to Destiny's
own code compiled verbatim (full_cases.HARNESS); the batched park is also held to the compiled code's ball rows
directly.

Worlds: tests/static_cases.py (ships flown through, along, beside, out of and inside capsules and oriented boxes, at
their faces, edges and corners; fields of static collidables and fixed spheres; a ship touching two or three static
collidables at once; boxes wider than a top box; static collidables on partition box faces). Development seeds 1 and
7; the held-out seed 11 was first run after the generator and this file were frozen (the note's section 4).

Compared after every step: every ball's state (positions, velocities and their old values, the goto point, mLastG,
mLastC, mLastCollision, the mode, the speed fraction, the filing and the boxes, the collision list in order), and each
static collidable's filing (its boxes) at the load. On the card: the kernel and the torch path in lockstep (every park
tensor after every tick), CUDA graphs against eager ticks, and planted faults (each must be seen).
"""

from __future__ import annotations

import pytest
import torch

import collide_cases as cc
import full_cases as fc
import static_cases as sc
import exact_ballpark.full as full
import exact_ballpark.vec_collide as V
from exact_ballpark import vec_collide_cuda as K
from exact_ballpark.vec_collide import VecCollidePark

DEV = (1, 7)
HELD_OUT = 11
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device (GPU tests)")
HARNESS = pytest.mark.skipif(not fc.HARNESS.is_file(),
                             reason="Destiny full harness not built (conformance/build_full.py)")
FIELDS = ("pos", "vel", "old_pos", "old_vel", "goto", "last_g", "last_c", "last_collision", "mode", "speed_fraction",
          "level", "key", "mod", "hits", "hits_n", "imp_slot", "imp_s", "imp_closing", "imp_n", "overflow")
_REF: dict = {}
_CPU: dict = {}


def reference(seed: int) -> tuple:
    """(scenarios, the reference's ball rows, its static filings), cached; no scenario is undefined here."""
    if seed not in _REF:
        cs = sc.cases(seed, 60)
        want, statics, _, undefined = sc.reference(cs)
        assert not undefined, undefined
        _REF[seed] = (cs, want, statics)
    return _REF[seed]


def torch_cpu(seed: int) -> dict:
    """The torch path on the CPU, cached: the card's tests compare with it."""
    if seed not in _CPU:
        _CPU[seed] = sc.batched(reference(seed)[0], "cpu")
    return _CPU[seed]


def static_ids(cs: list) -> set:
    return {(c["id"], s["id"]) for c in cs for s in sc.statics_of(c)}


def reach(cs: list, rows: dict) -> tuple:
    """Ball-steps with a static collidable on the collision list, and with two or more."""
    ids = static_ids(cs)
    n = [sum(1 for h in r["hits"] if (cid, h) in ids) for (cid, _, _), r in rows.items()]
    return sum(1 for x in n if x), sum(1 for x in n if x > 1)


def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu()
    return t.view(torch.int64) if t.dtype == torch.float64 else t


# ---- the torch path against the reference (CPU) ----

@pytest.mark.parametrize("seed", DEV)
def test_the_torch_path_matches_the_reference(seed):
    """The torch path on the CPU against FullBallpark, every ball every step, and every static collidable's filing;
    the worlds reach static collidables often, and two or more in one tick."""
    cs, want, statics = reference(seed)
    parks: list = []
    got = sc.batched(cs, "cpu", parks=parks)
    assert sc.differences(want, got) == []
    assert sc.park_statics(parks) == statics
    one, two = reach(cs, want)
    assert one >= 100 and two >= 10, (one, two)


def test_without_host_syncs_and_with_rows_is_the_same():
    """contact_slots (no host synchronisation: every static slot walked) and rows='free' (the free balls' rows):
    the same rows as the reference."""
    cs, want, _ = reference(DEV[0])
    assert sc.differences(want, sc.batched(cs, "cpu", contact_slots=5)) == []
    assert sc.differences(want, sc.batched(cs, "cpu", rows="free")) == []


@HARNESS
@pytest.mark.parametrize("seed", DEV)
def test_the_reference_and_the_park_match_compiled_destiny(seed):
    """Destiny's own code compiled verbatim on the same scenarios: the reference equals it on every row kind (balls,
    static collidables, the partition and its boxes, events), and the batched park's torch path equals its ball rows."""
    cs, _, _ = reference(seed)
    rows, undefined = fc.run_reference(cs)
    ora = fc.oracle(cs)
    assert fc.differences(rows, ora, undefined) == {}
    assert len(ora["T"]) > 0 and len(ora["S"]) > 1000
    assert sc.differences(sc.rows_of(ora["S"]), torch_cpu(seed)) == []


def test_the_static_records_are_the_references():
    """static_record's doubles are what AddCapsule and AddOrientedBox make in the reference (FullBallpark): the
    capsule's ends and radius, the box's corners, centre, AABB, sides' corners and normals (and the triangle (a, c, d)'s),
    its axes, their normalised forms, half lengths and squared lengths; the filings at its level."""
    cs, _, statics = reference(DEV[0])
    seen = 0
    for c in cs:
        park = full.FullBallpark(friction=c["friction"], tick_interval_ms=c["tick"], is_master=False)
        for s in sc.statics_of(c):
            geo, ints, keys = V.static_record(s)
            if s["kind"] == "capsule":
                cap = park.add_capsule(s["id"], s["parent"], *s["a"], *s["b"], s["radius"])
                assert geo[V.G_A:V.G_A + 3] == cap.a and geo[V.G_B:V.G_B + 3] == cap.b
                assert geo[V.G_AB:V.G_AB + 3] == cap.ab and geo[V.G_RAD] == cap.radius
            else:
                ob = park.add_oriented_box(s["id"], s["parent"], *s["corner"], *s["lx"], *s["ly"], *s["lz"])
                sh = ob.shape
                assert [geo[V.G_CORNERS + 3 * i:V.G_CORNERS + 3 * i + 3] for i in range(8)] == list(sh.corners)
                assert geo[V.G_CENTER:V.G_CENTER + 3] == sh.center()
                assert geo[V.G_LO:V.G_LO + 3] == ob.aabb.low and geo[V.G_HI:V.G_HI + 3] == ob.aabb.high
                for i, side in enumerate(sh.sides):
                    assert geo[V.G_RECT + 12 * i:V.G_RECT + 12 * i + 12] == [*side.a, *side.b, *side.c, *side.d]
                    assert geo[V.G_RNORM + 3 * i:V.G_RNORM + 3 * i + 3] == side.normal
                    assert geo[V.G_TNORM + 3 * i:V.G_TNORM + 3 * i + 3] == full.Triangle(side.a, side.c,
                                                                                        side.d).normal()
                for i, axis in enumerate(sh.local):
                    assert geo[V.G_AXES + 3 * i:V.G_AXES + 3 * i + 3] == axis
                    assert geo[V.G_UNIT + 3 * i:V.G_UNIT + 3 * i + 3] == full._normalized(axis)
                    assert geo[V.G_HALF + i] == full._length(axis) / 2.0 and geo[V.G_LENSQ + i] == full._lensq(axis)
            assert tuple((ints[V.I_LEVEL], *k) for k in keys) == statics[(c["id"], s["id"])]
            seen += 1
    assert seen > 50
    assert V.CONNECTED_SIDES == full.CONNECTED_SIDES and V.SIDE_MAPPING == full.SIDE_MAPPING


def test_static_slots_without_static_collidables_change_nothing():
    """A park with static slots and none loaded, or loaded empty, gives the worlds the rows a park without slots
    gives them (its collision lists one entry a slot longer, unused)."""
    cs = cc.cases(1, n=60)
    want = cc.batched(cs, "cpu")
    assert cc.differences(want, cc.batched(cs, "cpu", statics=3)) == []


# ---- refusals ----

def test_what_is_not_in_this_module_is_refused():
    """The iterative method with static collidables; the master's bubbles with them; duplicate ids in a world; more
    than the park's slots; a park without slots given some; an unknown kind; a box at the partition's edge; a capsule
    whose walk through the boxes never ends (Destiny's result undefined); the statics build with the composed one."""
    ship = {"id": 1, "mass": 1e7, "radius": 10.0, "max_vel": 100.0, "agility": 1.0, "speed_fraction": 1.0,
            "pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "free": True, "massive": True}
    cap = {"kind": "capsule", "id": 5, "parent": 0, "a": [50.0, 0.0, 0.0], "b": [90.0, 0.0, 0.0], "radius": 5.0}
    with pytest.raises(ValueError, match="iterative"):
        VecCollidePark(1, 1, iterative=True, statics=1).load([[ship]], [[cap]])
    p = VecCollidePark(1, 1, statics=1)
    p.load([[ship]], [[cap]])
    p.bubble = torch.zeros(1, 1, dtype=torch.int64)
    with pytest.raises(ValueError, match="bubble"):
        p.evolve()
    with pytest.raises(ValueError, match="distinct"):
        VecCollidePark(1, 1, statics=2).load([[ship]], [[cap, dict(cap)]])
    with pytest.raises(ValueError, match="slots"):
        VecCollidePark(1, 1, statics=1).load([[ship]], [[cap, dict(cap, id=6)]])
    with pytest.raises(ValueError, match="slots"):
        VecCollidePark(1, 1).load([[ship]], [[cap]])
    with pytest.raises(ValueError, match="kind"):
        VecCollidePark(1, 1, statics=1).load([[ship]], [[dict(cap, kind="cone")]])
    edge = V.LEVEL_WIDTH[0] * V.LEVEL_GRID[0] * 0.5 - 10.0
    with pytest.raises(ValueError, match="edge"):
        VecCollidePark(1, 1, statics=1).load([[ship]], [[dict(cap, a=[edge, 0.0, 0.0], b=[edge, 0.0, 0.0])]])
    with pytest.raises(ValueError, match="never ends"):
        VecCollidePark(1, 1, statics=1).load([[ship]], [[dict(cap, a=[0.0, 0.0, 0.0], b=[1.0e9, 0.0, 0.0],
                                                             radius=1.0)]])
    with pytest.raises(ValueError, match="composed"):
        K.build(variant="sm90", composed=True, statics=True)


def test_the_statics_header_and_the_default_header():
    """The statics header carries vec_collide's static constants and BoxShape's tables, exact; the default and
    composed headers carry none of them (they are the headers they were)."""
    vals = K.table_values(statics=True)
    for name in K.STATIC_INTS:
        assert vals[f"XBPC_{name}"] == ("i64", getattr(V, name))
    for name, attr in K.STATIC_DOUBLES.items():
        assert vals[name] == ("d", getattr(V, attr))
        assert float(getattr(V, attr)).hex() in K.tables_header(statics=True)
    assert vals["XBPC_CONNECTED_SIDES"][1] == tuple(x for row in full.CONNECTED_SIDES for x in row)
    assert vals["XBPC_SIDE_MAPPING"][1] == tuple(x for row in full.SIDE_MAPPING for x in row)
    for h in (K.tables_header(), K.tables_header(composed=True)):
        assert "XBPC_G_" not in h and "K_EJECT_MARGIN" not in h and "XBPC_CONNECTED_SIDES" not in h
    assert K.tables_header() == K.tables_header(statics=False)


# ---- planted faults on the torch path (CPU) ----

def _reversed_order(self, active, order):
    """The walk's order reversed: each responder's active static collidables last to first."""
    perm = _ORIG_SORTED(self, active, order)
    cnt = active.sum(-1, keepdim=True)
    idx = torch.arange(perm.shape[-1], device=perm.device).expand_as(perm)
    return perm.gather(-1, torch.where(idx < cnt, cnt - 1 - idx, idx))


def _box_order_reversed(self, active, order):
    """The boxes' positions in the walk reversed (the parent and id order within a box kept)."""
    return _ORIG_SORTED(self, active, torch.where(active, V.NOT_A_CANDIDATE - 1 - order, order))


def _equal_replaces(self, mask, toi, lc, sid):
    self.last_c = torch.where(mask[..., None], lc, self.last_c)
    self.last_collision = torch.where(mask, toi, self.last_collision)
    self.hits = self._append(self.hits, self.hits_n, mask, sid)
    self.hits_n = self.hits_n + mask.to(torch.int64)


_ORIG_SORTED = VecCollidePark._static_sorted
_ORIG_CAPSULE = VecCollidePark._capsule_contact


def _no_end(self, *a):
    toi, nrm, parallel = _ORIG_CAPSULE(self, *a)
    return toi, nrm, torch.zeros_like(parallel)


TORCH_FAULTS = {
    "order_reversed": ("_static_sorted", _reversed_order),
    "box_order_reversed": ("_static_sorted", _box_order_reversed),
    "equal_time_replaces": ("_static_update", _equal_replaces),
    "capsule_reaction_from_the_start": ("_capsule_contact", _no_end),
    "triangle_closest_point_skipped": ("_triangle_closest", lambda self, A, B, C, P: P),
}


@pytest.mark.parametrize("fault", sorted(TORCH_FAULTS) + ["no_eject_margin"])
def test_planted_torch_faults_are_caught(monkeypatch, fault):
    """Each fault planted in the torch path changes what the comparison with the reference sees (development seed 1):
    the static collidables walked in reverse order (or only their boxes' positions in the walk reversed), a response at
    an equal time replacing the held one, a capsule's
    reaction from the tick's start where the end velocity is parallel to its axis, the triangle's closest point
    skipped, the ejection's metre dropped."""
    cs, want, _ = reference(DEV[0])
    if fault == "no_eject_margin":
        monkeypatch.setattr(V, "EJECT_MARGIN", 0.0)
    else:
        name, fn = TORCH_FAULTS[fault]
        monkeypatch.setattr(VecCollidePark, name, fn)
    assert len(sc.differences(want, sc.batched(cs, "cpu"))) > 0


# ---- on the card ----

@CUDA
@pytest.mark.parametrize("seed", DEV)
def test_the_kernel_matches_the_torch_path_and_the_reference(seed):
    """The statics kernel (pruned, contact_slots 5, rows='free') and the torch path on the card: the torch path's CPU
    rows and the reference's, bit for bit."""
    cs, want, _ = reference(seed)
    cpu = torch_cpu(seed)
    for kw in ({"kernel": "cuda"}, {"kernel": "cuda", "contact_slots": 5}, {"kernel": "cuda", "rows": "free"},
               {"kernel": "torch"}):
        got = sc.batched(cs, "cuda", **kw)
        assert sc.differences(cpu, got) == [], kw
        assert sc.differences(want, got) == [], kw


@CUDA
def test_the_kernel_and_the_torch_path_in_lockstep():
    """Every park tensor after every tick, the kernel against the torch path on the card (development seed 1)."""
    cs, _, _ = reference(DEV[0])
    a, b = [], []
    sc.batched(cs, "cuda", kernel="torch", after_tick=lambda p, s: a.append({f: bits(getattr(p, f)) for f in FIELDS}))
    sc.batched(cs, "cuda", kernel="cuda", after_tick=lambda p, s: b.append({f: bits(getattr(p, f)) for f in FIELDS}))
    assert len(a) == len(b) >= 16
    assert [(k, f) for k, (x, y) in enumerate(zip(a, b)) for f in FIELDS if not torch.equal(x[f], y[f])] == []


@CUDA
@pytest.mark.parametrize("kernel", ["cuda", "torch"])
def test_a_cuda_graph_with_static_collidables_equals_eager(kernel):
    """A park's ticks with static collidables recorded in one CUDA graph (graph_capture's capture and pins), on the
    kernel and on the torch path (contact_slots: no host synchronisation), replayed 12 times against eager ticks:
    every tensor after every tick, bit for bit."""
    from graph_capture import Pins, capture
    cs = [c for c in reference(DEV[0])[0] if c["friction"] == 1.0e6 and c["tick"] == 1000]
    n = max(sum(1 for rec in c["records"] if rec[0] == "B") for c in cs)
    s_n = max(len(sc.statics_of(c)) for c in cs)
    parks = []
    for _ in range(2):
        p = VecCollidePark(len(cs), n, device="cuda", kernel=kernel, contact_slots=5, statics=s_n)
        p.load([list({rec[2]["id"]: sc._ball_dict(rec[2]) for rec in c["records"] if rec[0] == "B"}.values())
                for c in cs], [sc.statics_of(c) for c in cs])
        slot = [{int(i): j for j, i in enumerate(ids) if i} for ids in p.id.cpu().tolist()]
        sc._commands(p, slot, [(w, rec) for w, c in enumerate(cs) for rec in c["records"]
                               if rec[0] == "K" and rec[1] == 1])
        parks.append(p)
    eager, graphed = parks
    pins = Pins([graphed])

    def body():
        graphed.evolve()
        pins.pin(True)
    eager.evolve()
    body()
    torch.cuda.synchronize()
    g = capture(body, torch.cuda.graph_pool_handle())
    entries = 0
    for k in range(12):
        eager.evolve()
        g.replay()
        torch.cuda.synchronize()
        assert [f for f in FIELDS if not torch.equal(bits(getattr(eager, f)), bits(getattr(graphed, f)))] == [], k
        entries += int(eager.hits_n.sum())
    assert entries > 0


KERNEL_FAULTS = {
    "static_order_reversed": (
        "    return k1 < k2 || (k1 == k2 && (p1 < p2 || (p1 == p2 && (d1 < d2 || (d1 == d2 && s1 < s2)))));",
        "    return k1 > k2 || (k1 == k2 && (p1 > p2 || (p1 == p2 && (d1 > d2 || (d1 == d2 && s1 > s2)))));"),
    "no_outward_test": (
        "        if (cap && nc > 0.0) return false;                              // a capsule: mLastG points out of it\n",
        ""),
    "equal_time_replaces": (
        "    if (toi == st.last_col) {\n        if (dot(lc, lc) > dot(st.last_c, st.last_c)) st.last_c = lc;",
        "    if (false) {\n        if (dot(lc, lc) > dot(st.last_c, st.last_c)) st.last_c = lc;"),
    "triangle_flipped": ("        const bool hit = dot(vel, m) <= 0.0", "        const bool hit = dot(vel, m) > 0.0"),
    "parallel_from_start": ("        end = true;\n", "        end = false;\n"),
    "box_inside_skipped": ("    if (inside) {\n        V3 q = gv(g, XBPC_G_CENTER);",
                           "    if (false) {\n        V3 q = gv(g, XBPC_G_CENTER);"),
}


@CUDA
@pytest.mark.parametrize("fault", sorted(KERNEL_FAULTS))
def test_planted_kernel_faults_are_caught(monkeypatch, tmp_path, fault):
    """Each fault, built into a copy of the statics kernel's source (one change), changes what the comparison with
    the torch path sees (development seed 1): the static collidables walked in reverse order, the capsule's outward
    test dropped, a response at an equal time replacing the held one, the other triangle of a side, a parallel
    capsule's reaction from the tick's start, the box's inside test skipped."""
    old, new = KERNEL_FAULTS[fault]
    src = K.SOURCE.read_text()
    assert src.count(old) == 1
    copy = tmp_path / f"{fault}.cu"
    copy.write_text(src.replace(old, new))
    lib = K.load_library("sm90", statics=True, source=copy)
    monkeypatch.setattr(K, "load_library", lambda *a, **k: lib)
    cs, _, _ = reference(DEV[0])
    assert len(sc.differences(torch_cpu(DEV[0]), sc.batched(cs, "cuda", kernel="cuda"))) > 0


# ---- the held-out seed (first run after the freeze) ----

def test_held_out_the_torch_path_matches_the_reference():
    cs, want, statics = reference(HELD_OUT)
    parks: list = []
    got = sc.batched(cs, "cpu", parks=parks)
    assert sc.differences(want, got) == []
    assert sc.park_statics(parks) == statics
    assert sc.differences(want, sc.batched(cs, "cpu", contact_slots=5)) == []


@HARNESS
def test_held_out_the_reference_and_the_park_match_compiled_destiny():
    cs, _, _ = reference(HELD_OUT)
    rows, undefined = fc.run_reference(cs)
    ora = fc.oracle(cs)
    assert fc.differences(rows, ora, undefined) == {}
    assert sc.differences(sc.rows_of(ora["S"]), torch_cpu(HELD_OUT)) == []


@CUDA
def test_held_out_the_kernel_matches_the_torch_path_and_the_reference():
    cs, want, _ = reference(HELD_OUT)
    cpu = torch_cpu(HELD_OUT)
    for kw in ({"kernel": "cuda"}, {"kernel": "cuda", "contact_slots": 5}, {"kernel": "torch"}):
        got = sc.batched(cs, "cuda", **kw)
        assert sc.differences(cpu, got) == [], kw
        assert sc.differences(want, got) == [], kw

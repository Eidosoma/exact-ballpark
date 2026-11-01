# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's minis and force fields in the batched collision park (the collision park's leftovers: the minis, and MOV-737's force fields): miniballs (local ids in BallSortComparer's owner-first
order), minicapsules and miniboxes (COL-713, made by exact_ballpark.vec_collide.make_minis as a client's Destiny makes
them) and force fields (SetBallHarmonic, and GetNearbyBalls' filter in the collision searches) in the gradient method of
VecCollidePark, on its torch path and on the kernel (the statics build), held bit for bit to the scalar reference
(exact_ballpark.full.FullBallpark, a client's view), which is held to Destiny's own code compiled verbatim
(full_cases.HARNESS); the batched park is also held to the compiled code's ball rows directly.

Worlds: tests/minis_fields_cases.py (fixed parents with miniballs, minicapsules and miniboxes, packed and interleaved;
force fields fixed and drifting, turned on, off and changed during a run, with ships matching them or not). Development
seeds 1 and 7; the held-out seed 11 was first run after the generator and this file were frozen (the note).

Compared after every step: every ball's state (collide_cases' rows: positions, velocities and their old values, the
goto point, mLastG, mLastC, mLastCollision, the mode, the speed fraction, the filing and the boxes, the collision list
in order) with its mass, radius, maximum speed, agility, time factor, owner, harmonic, corporation and alliance; each
static collidable's filing at the load. On the card: the kernel and the torch path in lockstep (every park tensor after
every tick), CUDA graphs against eager ticks, and planted faults (each must be seen).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace

import pytest
import torch

import collide_cases as cc
import full_cases as fc
import minis_fields_cases as mf
import static_cases as stc
import exact_ballpark.full as full
import exact_ballpark.vec_collide as V
from exact_ballpark import modes
from exact_ballpark import vec_collide_cuda as K
from exact_ballpark.vec_collide import VecCollidePark

DEV = (1, 7)
HELD_OUT = 11
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device (GPU tests)")
HARNESS = pytest.mark.skipif(not fc.HARNESS.is_file(),
                             reason="Destiny full harness not built (conformance/build_full.py)")
FIELDS = ("pos", "vel", "old_pos", "old_vel", "goto", "last_g", "last_c", "last_collision", "mode", "speed_fraction",
          "level", "key", "mod", "hits", "hits_n", "imp_slot", "imp_s", "imp_closing", "imp_n", "overflow")
NEW = ("owner", "sort_id", "harmonic", "corporation", "alliance")
_REF: dict = {}
_CPU: dict = {}


def reference(seed: int) -> tuple:
    """(scenarios, the reference's ball rows, its static filings), cached; no scenario is undefined here."""
    if seed not in _REF:
        cs = mf.cases(seed)
        want, statics, _, undefined = mf.reference(cs)
        assert not undefined, undefined
        _REF[seed] = (cs, want, statics)
    return _REF[seed]


def torch_cpu(seed: int) -> dict:
    """The torch path on the CPU, cached: the card's tests compare with it."""
    if seed not in _CPU:
        _CPU[seed] = mf.batched(reference(seed)[0], "cpu")
    return _CPU[seed]


def filings(parks: list) -> dict:
    return stc.park_statics([(g, p) for g, p in parks if p.S])


def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu()
    return t.view(torch.int64) if t.dtype == torch.float64 else t


# ---- the torch path against the reference (CPU) ----

@pytest.mark.parametrize("seed", DEV)
def test_the_torch_path_matches_the_reference(seed):
    """The torch path on the CPU against FullBallpark, every ball every step with the extended rows, and every static
    collidable's filing; the worlds reach miniballs of two owners in the owner-first order, minicapsules and miniboxes,
    and force fields on the collision lists."""
    cs, want, statics = reference(seed)
    parks: list = []
    got = mf.batched(cs, "cpu", parks=parks)
    assert mf.differences(want, got) == []
    assert filings(parks) == statics
    reach = mf.reach(want)
    assert reach["owner_order_seen"] >= 20 and reach["mini_static"] >= 20 and reach["field_hit"] >= 20, reach
    kinds = Counter((p.minis, p.fields, p.S > 0) for _, p in parks)
    assert {(False, True, False), (True, False, False), (True, False, True)} <= set(kinds), kinds


def test_every_option_without_host_syncs_and_with_rows_is_the_same():
    """Every park made with minis and fields; contact_slots (no host synchronisation); rows='free': the reference's
    rows."""
    cs, want, _ = reference(DEV[0])
    for kw in ({"flags": "both"}, {"contact_slots": 10}, {"rows": "free"}):
        assert mf.differences(want, mf.batched(cs, "cpu", **kw)) == [], kw


def test_the_worlds_reach_every_clause_of_the_field_filter():
    """The reference's branch counts on the development seeds: each clause of GetNearbyBalls' force-field filter (the
    pass, the harmonic, the corporation, the alliance) skips candidates, and FIELD balls tick as an inert mode."""
    trace: Counter = Counter()
    for seed in DEV:
        fc.run_reference(reference(seed)[0], trace)
    for name in ("nearby.field_master_pass", "nearby.field_harmonic", "nearby.field_corporation",
                 "nearby.field_alliance"):
        assert trace[name] >= 20, (name, trace[name])
    assert trace["mode.inert"] > 0


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
    assert mf.differences(mf.rows_of(ora["S"]), torch_cpu(seed)) == []


# ---- make_minis, the ranks, set_harmonic ----

def _reference_at_step_0(c: dict):
    park = full.FullBallpark(friction=c["friction"], tick_interval_ms=c["tick"], is_master=False)
    out: dict = defaultdict(list)
    for rec in c["records"]:
        if rec[1] == 0:
            fc._apply(park, rec, c["id"], 0, out)
    return park


def test_make_minis_is_destinys():
    """make_minis against FullBallpark's own minis (AddMiniBall, AddMiniCapsule and AddMiniBox through the thunkers on
    a client's view) in every scenario of the development seeds: each miniball's id, owner, flags, mode, mass, radius,
    position, velocity, maximum speed, agility, speed fraction and time factor (as the load makes it); each
    minicapsule's and minibox's id, parent and shape; every local object, and the local counter after the calls."""
    seen: Counter = Counter()
    for seed in DEV:
        for c in reference(seed)[0]:
            ref = _reference_at_step_0(c)
            balls = list({rec[2]["id"]: stc._ball_dict(rec[2]) for rec in c["records"] if rec[0] == "B"}.values())
            made_b, made_s, local = V.make_minis(balls, mf.minis_of(c))
            assert local == ref.local_hi_cnt
            assert {i for i in ref.balls if i < 0} == {b["id"] for b in made_b}
            assert ({i for i in ref.capsules if i < 0} | {i for i in ref.oriented_boxes if i < 0}
                    == {s["id"] for s in made_s})
            park = VecCollidePark(1, max(1, len(balls) + len(made_b)), friction=c["friction"],
                                  tick_interval_ms=c["tick"], minis=True, statics=max(1, len(made_s)))
            park.load([balls + made_b])
            slot = {int(i): j for j, i in enumerate(park.id[0].tolist()) if i}
            for b in made_b:
                rb, j = ref.balls[b["id"]], slot[b["id"]]
                assert (rb.owner_id, rb.mode, rb.is_free, rb.is_massive, rb.is_interactive) == (
                    b["owner"], modes.MINIBALL, False, True, False)
                assert int(park.owner[0, j]) == rb.owner_id and int(park.mode[0, j]) == rb.mode
                for name, value in (("mass", rb.mass), ("radius", rb.radius), ("max_vel", rb.max_vel),
                                    ("agility", rb.agility), ("speed_fraction", rb.speed_fraction),
                                    ("time_factor", rb.time_factor)):
                    assert fc.same(float(getattr(park, name)[0, j]), float(value)), name
                for name, value in (("pos", rb.pos), ("vel", rb.vel), ("old_pos", rb.old_pos),
                                    ("old_vel", rb.old_vel)):
                    assert fc.same(tuple(getattr(park, name)[0, j].tolist()), tuple(float(x) for x in value)), name
                seen["ball"] += 1
            for s in made_s:
                if s["kind"] == "capsule":
                    rc = ref.capsules[s["id"]]
                    assert rc.parent_id == s["parent"] and fc.same(tuple(rc.a), tuple(s["a"]))
                    assert fc.same(tuple(rc.b), tuple(s["b"])) and fc.same(rc.radius, s["radius"])
                else:
                    rb = ref.oriented_boxes[s["id"]]
                    ob = full.OrientedBox()
                    ob.initialize(s["id"], s["parent"], s["corner"], s["lx"], s["ly"], s["lz"])
                    assert rb.parent_id == s["parent"]
                    assert fc.same(tuple(map(tuple, rb.shape.corners)), tuple(map(tuple, ob.shape.corners)))
                seen[s["kind"]] += 1
    assert min(seen["ball"], seen["capsule"], seen["box"]) >= 20, seen


def test_make_minis_edge_cases_are_destinys():
    """A call for a missing parent, a free parent's minis (kept, unmade), minicapsules whose float radius is not
    positive (the thunker's ValueError: nothing, no id), a counter carried from earlier calls: as FullBallpark."""
    parent = {"id": 7, "mass": 2e9, "radius": 50.0, "max_vel": 0.0, "pos": [10.0, -20.0, 30.5], "vel": [0.0] * 3,
              "free": False, "massive": True}
    loose = dict(parent, id=8, free=True, agility=1.0)
    calls = [{"kind": "ball", "parent": 9, "pos": [1.0, 2.0, 3.0], "radius": 5.0},
             {"kind": "ball", "parent": 8, "pos": [1.0, 2.0, 3.0], "radius": 5.0},
             {"kind": "capsule", "parent": 7, "a": [0.0, 0.0, 0.0], "b": [40.0, 0.0, 0.0], "radius": 0.0},
             {"kind": "capsule", "parent": 7, "a": [0.0, 0.0, 0.0], "b": [40.0, 0.0, 0.0], "radius": 1e-46},
             {"kind": "capsule", "parent": 7, "a": [0.0, 0.0, 0.0], "b": [40.0, 0.0, 0.0], "radius": -2.0},
             {"kind": "ball", "parent": 7, "pos": [0.1, 0.2, 0.3], "radius": -4.0},
             {"kind": "capsule", "parent": 7, "a": [0.3, 0.0, 0.0], "b": [40.0, 0.0, 0.0], "radius": 3.3},
             {"kind": "box", "parent": 7, "corner": [5.0, 5.0, 5.0], "lx": [10.0, 0.0, 0.0], "ly": [0.0, 7.0, 0.0],
              "lz": [0.0, 0.0, 3.0]},
             {"kind": "box", "parent": 8, "corner": [5.0, 5.0, 5.0], "lx": [10.0, 0.0, 0.0], "ly": [0.0, 7.0, 0.0],
              "lz": [0.0, 0.0, 3.0]}]
    ref = full.FullBallpark(is_master=False)
    for b in (parent, loose):
        ref.add_ball(b["id"], b["mass"], b["radius"], b["max_vel"], b["free"], False, b["massive"], False, False,
                     *b["pos"], *b["vel"], b.get("agility", 0.0), 0.0)
    ref.add_capsule(3, 0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)              # explicit ids take no local id
    ref.add_mini_ball(7, 0.0, 0.0, 0.0, 1.0)                             # an earlier call: the counter moves
    start = ref.local_hi_cnt
    for m in calls:
        try:
            if m["kind"] == "ball":
                ref.add_mini_ball(m["parent"], *m["pos"], m["radius"])
            elif m["kind"] == "capsule":
                ref.add_mini_capsule(m["parent"], *m["a"], *m["b"], m["radius"])
            else:
                ref.add_mini_box(m["parent"], *m["corner"], *m["lx"], *m["ly"], *m["lz"])
        except ValueError:
            pass
    made_b, made_s, local = V.make_minis([parent, loose], calls, start)
    assert local == ref.local_hi_cnt == start - 3
    assert [b["id"] for b in made_b] == [i for i in sorted(ref.balls, reverse=True) if start > i]
    assert made_b[0]["radius"] == ref.balls[made_b[0]["id"]].radius == 0.0
    assert [s["id"] for s in made_s] == [start - 2, start - 3]
    assert ref.capsules[start - 2].radius == made_s[0]["radius"] and ref.oriented_boxes[start - 3].parent_id == 7
    assert len(ref.balls[8].mini_balls) == 1 and len(ref.balls[8].mini_boxes) == 1     # kept by the free parent


def test_the_load_ranks_the_balls_in_ballsortcomparers_order():
    """sort_id is each ball's rank by the reference's own BallSortComparer (full._ball_key) in its world; a park made
    without minis or fields holds none of their tensors."""
    cs, _, _ = reference(DEV[0])
    parks: list = []
    mf.batched(cs, "cpu", parks=parks)
    checked = 0
    for g, park in parks:
        if not park.minis:
            continue
        for w in range(park.W):
            n = int(park.exists[w].sum())
            balls = [SimpleNamespace(id=int(park.id[w, j]), owner_id=int(park.owner[w, j]), slot=j) for j in range(n)]
            ranked = sorted(balls, key=full._ball_key)
            assert [int(park.sort_id[w, b.slot]) for b in ranked] == list(range(n))
            checked += n
    assert checked > 200
    plain = VecCollidePark(2, 3)
    assert not any(hasattr(plain, name) for name in NEW)
    assert VecCollidePark(2, 3, minis=True).owner.shape == (2, 3)
    assert VecCollidePark(2, 3, fields=True).harmonic.tolist() == [[V.NO_GROUP] * 3] * 2


def test_set_harmonic_is_space_s():
    """set_harmonic as SpaceBallpark's (and FullBallpark's) set_harmonic on a ball in GOTO, a fixed one and a FIELD one:
    the values set, a field stopped and set to FIELD, a cleared field stopped, another ball's mode kept; a missing slot
    untouched."""
    ship = {"id": 1, "mass": 1e7, "radius": 10.0, "max_vel": 100.0, "agility": 1.0, "speed_fraction": 1.0,
            "pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "free": True, "massive": True}
    worlds = [[ship, dict(ship, id=2, free=False, pos=[500.0, 0.0, 0.0])]]
    park = VecCollidePark(1, 3, fields=True)
    park.load(worlds)
    ref = full.FullBallpark(is_master=False)
    for b in worlds[0]:
        ref.add_ball(b["id"], b["mass"], b["radius"], b["max_vel"], b["free"], False, True, False, False, *b["pos"],
                     *b["vel"], b["agility"], b["speed_fraction"])
    park.goto_point(torch.tensor([[True, False, False]]), torch.tensor([[[9.0, 9.0, 9.0]] * 3], dtype=torch.float64))
    ref.goto_point(1, 9.0, 9.0, 9.0)
    t = lambda *x: torch.tensor([list(x)], dtype=torch.int64)                   # noqa: E731
    steps = [((True, True, True), (5, 6, 7), (8, 9, 10), (11, 12, 13), (False, True, True)),
             ((True, False, True), (-2, 0, 0), (-1, 0, 0), (3, 0, 0), (False, False, False)),
             ((False, True, False), (0, 4, 0), (0, -1, 0), (0, -1, 0), (False, False, False)),
             ((True, False, False), (1, 0, 0), (2, 0, 0), (3, 0, 0), (True, False, False))]
    for mask, h, c, a, field in steps:
        park.set_harmonic(torch.tensor([list(mask)]), t(*h), t(*c), t(*a), torch.tensor([list(field)]))
        for j, bid in enumerate((1, 2)):
            if mask[j]:
                ref.set_harmonic(bid, h[j], c[j], a[j], field[j])
        for j, bid in enumerate((1, 2)):
            rb = ref.balls[bid]
            assert (int(park.mode[0, j]), int(park.harmonic[0, j]), int(park.corporation[0, j]),
                    int(park.alliance[0, j])) == (rb.mode, rb.harmonic, rb.corporation_id, rb.alliance_id)
        assert int(park.harmonic[0, 2]) == V.NO_GROUP and int(park.mode[0, 2]) == modes.STOP


def test_parks_with_the_options_and_nothing_to_use_them_change_nothing():
    """A park made with minis and fields given the worlds (positive ids, no harmonics): the rows a park without them
    gives."""
    cs = cc.cases(1, n=60)
    want = cc.batched(cs, "cpu")
    assert cc.differences(want, cc.batched(cs, "cpu", minis=True, fields=True)) == []


# ---- refusals ----

def test_what_is_not_in_this_module_is_refused():
    """The iterative method with minis or fields; the master's bubbles with them (at the tick); local ids, owners and
    miniballs in a park without minis; ids 0 and -1 in one with them; a free miniball; SetBallHarmonic without fields;
    an unknown mini."""
    ship = {"id": 1, "mass": 1e7, "radius": 10.0, "max_vel": 100.0, "agility": 1.0, "speed_fraction": 1.0,
            "pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "free": True, "massive": True}
    mini = {"id": V.LOCAL_BALLS - 1, "owner": 5, "miniball": True, "mass": 1e9, "radius": 3.0, "max_vel": 0.0,
            "agility": 0.0, "speed_fraction": 0.0, "pos": [100.0, 0.0, 0.0], "vel": [0.0] * 3, "free": False,
            "massive": True}
    for kw in ({"minis": True}, {"fields": True}):
        with pytest.raises(ValueError, match="iterative"):
            VecCollidePark(1, 2, iterative=True, **kw)
        p = VecCollidePark(1, 2, **kw)
        p.load([[ship]])
        p.bubble = torch.zeros(1, 2, dtype=torch.int64)
        with pytest.raises(ValueError, match="bubble"):
            p.evolve()
    with pytest.raises(ValueError, match="positive"):
        VecCollidePark(1, 2).load([[ship, dict(ship, id=-5)]])
    with pytest.raises(ValueError, match="minis"):
        VecCollidePark(1, 2).load([[ship, mini]])
    for bad in (0, -1):
        with pytest.raises(ValueError, match="neither"):
            VecCollidePark(1, 2, minis=True).load([[ship, dict(ship, id=bad)]])
    with pytest.raises(ValueError, match="fixed"):
        VecCollidePark(1, 2, minis=True).load([[ship, dict(mini, free=True)]])
    p = VecCollidePark(1, 2)
    p.load([[ship]])
    z = torch.zeros(1, 2, dtype=torch.int64)
    with pytest.raises(ValueError, match="fields"):
        p.set_harmonic(torch.ones(1, 2, dtype=torch.bool), z, z, z, torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="kind"):
        V.make_minis([ship], [{"kind": "cone", "parent": 1}])


def test_the_statics_header_carries_the_field_constants():
    """The statics header carries the force fields' constants, read from vec_collide (FIELD from exact_ballpark.modes,
    the pass and the unset value as the reference writes them); the default and composed headers do not."""
    vals = K.table_values(statics=True)
    for name in ("FIELD_MODE", "HARMONIC_PASS", "NO_GROUP"):
        assert name in K.STATIC_INTS and vals[f"XBPC_{name}"] == ("i64", getattr(V, name))
    assert (V.FIELD_MODE, V.MINIBALL_MODE) == (modes.FIELD, modes.MINIBALL)
    import inspect
    text = inspect.getsource(full.FullBallpark._nearby)
    assert f"b.harmonic == {V.HARMONIC_PASS}" in text and f"other.harmonic != {V.NO_GROUP}" in text
    for h in (K.tables_header(), K.tables_header(composed=True)):
        assert "XBPC_FIELD_MODE" not in h and "XBPC_HARMONIC_PASS" not in h and "XBPC_NO_GROUP" not in h


# ---- planted faults on the torch path (CPU) ----

def _passed_without(clause: str):
    def passed(self):
        R = self.R
        field = (self.mode == V.FIELD_MODE)[:, None, :]

        def same(x, guard=True):
            eq = x[:, None, :] == x[:, :R, None]
            return eq & (x[:, None, :] != V.NO_GROUP) if guard else eq
        parts = {"pass": self.harmonic[:, :R, None] == V.HARMONIC_PASS, "harmonic": same(self.harmonic),
                 "corporation": same(self.corporation, clause != "guard"), "alliance": same(self.alliance)}
        out = torch.zeros_like(parts["pass"] & field)
        for name, part in parts.items():
            if name != clause:
                out = out | part
        return field & out
    return passed


def _made(transform):
    real = V.make_minis

    def made(balls, minis, local=V.LOCAL_BALLS):
        return transform(real, balls, minis, local)
    return made


def _shifted(real, balls, minis, local):
    b, s, n = real(balls, minis, local - 1)
    return b, s, n


def _free_parents_made(real, balls, minis, local):
    return real([dict(b, free=False) for b in balls], minis, local)


def _refused_capsules_made(real, balls, minis, local):
    return real(balls, [dict(m, radius=5.0) if m["kind"] == "capsule" and float(m["radius"]) <= 0.0 else m
                        for m in minis], local)


def _mode_lost(real, balls, minis, local):
    b, s, n = real(balls, minis, local)
    return [{k: v for k, v in x.items() if k != "miniball"} for x in b], s, n


TORCH_FAULTS = {
    "field_pass_dropped": ("_field_passed", _passed_without("pass")),
    "field_harmonic_dropped": ("_field_passed", _passed_without("harmonic")),
    "field_corporation_dropped": ("_field_passed", _passed_without("corporation")),
    "field_alliance_dropped": ("_field_passed", _passed_without("alliance")),
    "no_group_guard_dropped": ("_field_passed", _passed_without("guard")),
}
MINIS_FAULTS = {"ids_shifted": _shifted, "free_parents_made": _free_parents_made,
                "refused_capsules_made": _refused_capsules_made, "miniball_mode_lost": _mode_lost}


@pytest.mark.parametrize("fault", sorted(TORCH_FAULTS) + sorted(MINIS_FAULTS) + ["id_order"])
def test_planted_torch_faults_are_caught(monkeypatch, fault):
    """Each fault planted in the torch path or in make_minis changes what the comparison with the reference sees
    (development seed 1): a clause of the force-field filter dropped (the pass, the harmonic, the corporation, the
    alliance) or its unset guard dropped; the local ids from one further on; a free parent's minis made; refused
    minicapsules made; a miniball loaded as AddBall's ball; the candidates in id order (the owner-first order lost)."""
    cs, want, _ = reference(DEV[0])
    kw = {}
    if fault in TORCH_FAULTS:
        name, fn = TORCH_FAULTS[fault]
        monkeypatch.setattr(VecCollidePark, name, fn)
    elif fault in MINIS_FAULTS:
        monkeypatch.setattr(V, "make_minis", _made(MINIS_FAULTS[fault]))
    else:
        kw["after_load"] = lambda p: setattr(p, "sort_id", p.id.clone()) if p.minis else None
    got = mf.batched(cs, "cpu", **kw)
    assert len(mf.differences(want, got)) > 0


# ---- on the card ----

@CUDA
@pytest.mark.parametrize("seed", DEV)
def test_the_kernel_matches_the_torch_path_and_the_reference(seed):
    """The statics kernel (pruned, contact_slots 10, rows='free', every park with minis and fields) and the torch path
    on the card: the torch path's CPU rows and the reference's, bit for bit."""
    cs, want, _ = reference(seed)
    cpu = torch_cpu(seed)
    for kw in ({"kernel": "cuda"}, {"kernel": "cuda", "contact_slots": 10}, {"kernel": "cuda", "rows": "free"},
               {"kernel": "cuda", "flags": "both"}, {"kernel": "torch"}):
        got = mf.batched(cs, "cuda", **kw)
        assert mf.differences(cpu, got) == [], kw
        assert mf.differences(want, got) == [], kw


@CUDA
def test_the_kernel_and_the_torch_path_in_lockstep():
    """Every park tensor after every tick (the new ones among them), the kernel against the torch path on the card
    (development seed 1)."""
    cs, _, _ = reference(DEV[0])
    a, b = [], []

    def hook(store):
        return lambda p, s: store.append({f: bits(getattr(p, f)) for f in FIELDS + NEW if hasattr(p, f)})
    mf.batched(cs, "cuda", kernel="torch", after_tick=hook(a))
    mf.batched(cs, "cuda", kernel="cuda", after_tick=hook(b))
    assert len(a) == len(b) >= 100
    assert [(k, f) for k, (x, y) in enumerate(zip(a, b)) for f in x if not torch.equal(x[f], y[f])] == []


@CUDA
@pytest.mark.parametrize("kernel", ["cuda", "torch"])
def test_a_cuda_graph_with_minis_and_fields_equals_eager(kernel):
    """A park's ticks with minis, fields and static collidables recorded in one CUDA graph (graph_capture's capture and
    pins), on the kernel and on the torch path (contact_slots: no host synchronisation), replayed 12 times against eager
    ticks: every tensor after every tick, bit for bit."""
    from graph_capture import Pins, capture
    cs = [c for c in reference(DEV[0])[0] if c["friction"] == 1.0e6 and c["tick"] == 1000]
    built = [mf.world_of(c) for c in cs]
    n, s_n = max(len(b[0]) for b in built), max(len(b[1]) for b in built)
    parks = []
    for _ in range(2):
        p = VecCollidePark(len(cs), n, device="cuda", kernel=kernel, contact_slots=10, statics=s_n, minis=True,
                           fields=True)
        p.load([b[0] for b in built], [b[1] for b in built])
        slot = [{int(i): j for j, i in enumerate(ids) if i} for ids in p.id.cpu().tolist()]
        for st in (0, 1):
            mf._commands(p, slot, [(w, rec) for w, c in enumerate(cs) for rec in c["records"]
                                   if rec[0] == "K" and rec[1] == st])
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
        assert [f for f in FIELDS + NEW if not torch.equal(bits(getattr(eager, f)), bits(getattr(graphed, f)))] == [], k
        entries += int(eager.hits_n.sum())
    assert entries > 0


KERNEL_FAULTS = {
    "sort_id_ignored": ("    return P.sort_id != nullptr ? P.sort_id[j] : id;",
                        "    return P.sort_id != nullptr ? id : id;"),
    "field_pass_dropped": ("    if (h == XBPC_HARMONIC_PASS) return true;\n", ""),
    "field_harmonic_dropped": ("    if (P.harmonic[j] != XBPC_NO_GROUP && P.harmonic[j] == h) return true;\n", ""),
    "field_corporation_dropped": (
        "    if (P.corporation[j] != XBPC_NO_GROUP && P.corporation[j] == P.corporation[j0]) return true;\n", ""),
    "field_alliance_dropped": ("    return P.alliance[j] != XBPC_NO_GROUP && P.alliance[j] == P.alliance[j0];",
                               "    return false;"),
    "no_group_guard_dropped": ("P.corporation[j] != XBPC_NO_GROUP && P.corporation[j] == P.corporation[j0]",
                               "P.corporation[j] == P.corporation[j0]"),
    "second_loop_unfiltered": ("            if (field_passed(P, j0, j)) continue;"
                               "                       // a force field passed\n", ""),
}


@CUDA
@pytest.mark.parametrize("fault", sorted(KERNEL_FAULTS))
def test_planted_kernel_faults_are_caught(monkeypatch, tmp_path, fault):
    """Each fault, built into a copy of the statics kernel's source (one change), changes what the comparison with the
    torch path sees (development seed 1): the rank ignored (the id order), a clause of the force-field filter dropped,
    the unset guard dropped, the filter missing from the rescan for the next contact."""
    old, new = KERNEL_FAULTS[fault]
    src = K.SOURCE.read_text()
    assert src.count(old) == 1
    copy = tmp_path / f"{fault}.cu"
    copy.write_text(src.replace(old, new))
    lib = K.load_library("sm90", statics=True, source=copy)
    monkeypatch.setattr(K, "load_library", lambda *a, **k: lib)
    cs, _, _ = reference(DEV[0])
    assert len(mf.differences(torch_cpu(DEV[0]), mf.batched(cs, "cuda", kernel="cuda"))) > 0


# ---- the held-out seed (first run after the freeze) ----

def test_held_out_the_torch_path_matches_the_reference():
    cs, want, statics = reference(HELD_OUT)
    parks: list = []
    got = mf.batched(cs, "cpu", parks=parks)
    assert mf.differences(want, got) == []
    assert filings(parks) == statics
    assert mf.differences(want, mf.batched(cs, "cpu", contact_slots=10)) == []
    assert mf.differences(want, mf.batched(cs, "cpu", flags="both")) == []


@HARNESS
def test_held_out_the_reference_and_the_park_match_compiled_destiny():
    cs, _, _ = reference(HELD_OUT)
    rows, undefined = fc.run_reference(cs)
    ora = fc.oracle(cs)
    assert fc.differences(rows, ora, undefined) == {}
    assert mf.differences(mf.rows_of(ora["S"]), torch_cpu(HELD_OUT)) == []


@CUDA
def test_held_out_the_kernel_matches_the_torch_path_and_the_reference():
    cs, want, _ = reference(HELD_OUT)
    cpu = torch_cpu(HELD_OUT)
    for kw in ({"kernel": "cuda"}, {"kernel": "cuda", "contact_slots": 10}, {"kernel": "cuda", "flags": "both"},
               {"kernel": "torch"}):
        got = mf.batched(cs, "cuda", **kw)
        assert mf.differences(cpu, got) == [], kw
        assert mf.differences(want, got) == [], kw

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance against compiled code for the space layer: the partition, bubbles,
proximity sensors, visibility, range queries, collisions (gradient and iterative) and ball removal of the reference
(exact_ballpark.space) against Destiny's own code compiled verbatim (conformance/build_space.py), around
slice 1's tick.

Generated scenarios (tests/space_cases.py): balls on and near box faces at every level and every radius scale, moved,
resized, fixed and freed; the grid's clamped edges; bubbles over neighbouring and distant top boxes, InitializeBubbles,
CopyBubbles, interactive and keep-alive balls; overlapping, coincident, approaching and crossing collisions with fixed
walls, light balls, space junk, cloaked, non-massive, global balls and force fields; sensors with periods, shuffles and
only-interactive filters, notification ranges, target tracking, balls leaving the candidates and dying bubbles;
occluders on and off the line of sight; balls exactly at a sensor's or a range query's threshold; followers of a ball
that changes bubble with the pre-tick clean-up; balls removed at once or left moribund and brought out before a later
tick; balls added again under their id; long warps through many top boxes; and slice 1's modes mixed in, at ticks of
0.25 to 2 s, several frictions and both the server's and a client's view. After every step every ball's state, boxes,
collision list, followers and sensor list, the partition's checksum and box counts, the proximity balls in Destiny's
order, the bubble dictionaries and every query's answer are compared for equality (doubles by their bits), every box at
the end of each scenario, and the events in Destiny's order (the final clean-ups of one BringOutDeadBalls call as a set:
Destiny walks an unordered_set of pointers there). Development cases (seed 1) were used while writing the reference;
held-out cases (seed 2) are a separate test, first run after the generator was frozen (generator frozen 16 October 2026,
16:25:45 UTC; first held-out run 16:25:46 UTC, which failed in one scenario of 1,835 on the order of two final clean-ups
of one BringOutDeadBalls call, every state equal; the comparison then took that order as a set).

The iterative collision method (COL-714 to COL-718) has its own scenarios (space_cases.iterative_cases: overlapping,
coincident, approaching, chained, dense, fast and limited impacts, fixed balls, Destiny's test ships, and mixtures with
warps, removal, sensors and dead bubbles), compared in the same way: development seed 1, held-out seed 2 (generator
frozen 16 October 2026, 16:41:12 UTC; first held-out run 16:41:13 UTC, passed).

Destiny reads and writes a released partition box in one path (`Ballpark::InsertInBoxes`, Ballpark.cpp:3141-3185, and
`Ball::InsertInBoxes`, Ball.cpp:2365): defined while BoxAllocator keeps the box's memory (fewer than 256 kept),
undefined once it has deleted it. The reference refuses those scenarios (DestinyUndefined); they are counted and left
out of the comparison, and a constructed one shows the undefined read is real on this machine. Skipped when the harness
has not been built here.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

import space_cases as sc
import exact_ballpark.space as space
from exact_ballpark.space import DestinyUndefined, SpaceBallpark

pytestmark = pytest.mark.skipif(not sc.HARNESS.is_file(),
                                reason="Destiny space harness not built (conformance/build_space.py)")

KINDS_ROWS = ("S", "P", "U", "I", "V", "W", "D", "Q")
_RUNS: dict = {}


def run(seed: int) -> tuple:
    """(scenarios, Destiny's rows, the reference's rows, the undefined scenarios), cached per seed."""
    if seed not in _RUNS:
        cs = sc.cases(seed)
        want = sc.oracle(cs)
        got, undefined = sc.run_reference(cs)
        _RUNS[seed] = (cs, want, got, undefined)
    return _RUNS[seed]


def differences(want: dict, got: dict, undefined) -> dict:
    """Per row kind, the differing rows as (case, step, field or kind); events per case, in order."""
    want = sc.without(want, undefined)
    out = {}
    for kind in KINDS_ROWS:
        w, g = want.get(kind, []), got.get(kind, [])
        if len(w) != len(g):
            out[kind] = [("row count", len(w), len(g))]
            continue
        if kind == "S":
            bad = [(a[0], a[1], a[2], sc.S_FIELDS[i]) for a, b in zip(w, g) for i in range(len(sc.S_FIELDS))
                   if not sc.same(a[i], b[i])]
        else:
            bad = [(a[0], a[1], kind) for a, b in zip(w, g) if not sc.same(a, b)]
        if bad:
            out[kind] = bad
    we, ge = {}, {}
    for e in canonical_events(want.get("E", [])):
        we.setdefault(e[0], []).append(e[1:])
    for e in canonical_events(got.get("E", [])):
        ge.setdefault(e[0], []).append(e[1:])
    bad_events = sorted(c for c in set(we) | set(ge) if we.get(c) != ge.get(c))
    if bad_events:
        out["E"] = bad_events
    return out


def canonical_events(events: list) -> list:
    """The events in order, except that the final clean-ups of one BringOutDeadBalls call (consecutive, same tick) are
    sorted by ball: Destiny walks an unordered_set of Ball pointers there (Ballpark.cpp:5716), whose order is set by
    heap addresses, which no reference can know. Every other event keeps Destiny's order."""
    out, i = [], 0
    while i < len(events):
        j = i
        while (j < len(events) and events[j][2] == "DoFinalCleanup" and events[j][0] == events[i][0]
               and events[j][1] == events[i][1]):
            j += 1
        if j > i + 1:
            out.extend(sorted(events[i:j], key=lambda e: e[3]))
            i = j
        else:
            out.append(events[i])
            i += 1
    return out


def check(seed: int) -> None:
    cs, want, got, undefined = run(seed)
    diffs = differences(want, got, undefined)
    assert not diffs, {k: (len(v), v[:5]) for k, v in diffs.items()}
    assert len(undefined) <= len(cs) // 200, f"{len(undefined)} scenarios undefined: {sorted(undefined)}"
    assert len(got["S"]) > 100000 and len(got["E"]) > 3000


def test_reference_matches_compiled_destiny_development_cases():
    check(1)


def test_reference_matches_compiled_destiny_held_out_cases():
    check(2)


BRANCHES = [
    "box.create", "box.release", "insert.adjacent", "insert.corner", "insert.huge", "insert.unchanged",       # SEN-706
    "insert.stale_box",
    "bubble.no_bubble", "bubble.new", "bubble.neighbour", "bubble.changed", "init.traversed",               # SEN-707
    "nearby.other_bubble", "nearby.global_other_bubble", "nearby.no_interactives", "visibility.other_bubble",  # SEN-708
    "tick.dead_bubble", "sensor.dead_bubble_check",                                                        # TIME-714
    "sensor.check", "sensor.wait", "sensor.shuffle", "sensor.entered", "sensor.left", "sensor.vanished",    # SEN-701/2
    "tracking.event", "range.event",                                                                       # SEN-703/4
    "visibility.occluded",                                                                                 # SEN-709
    "potential.free", "potential.fixed", "potential.coincident", "potential.recurse", "potential.light",   # COL-705-711
    "potential.early_return", "potential.equal_time", "potential.earlier_kept", "gradient.skip_junk",
    "nearby.field_master_pass", "nearby.field_harmonic", "nearby.field_corporation", "nearby.field_alliance",  # COL-703
    "add.existing",                                                                                        # AddBall
    "remove.now", "remove.delayed", "remove.buffer", "nearby.moribund", "range.moribund",                  # TIME-713
    "warp.drop_out", "warp.drop_out_insert", "release.stop", "release.goto_direction",                     # slice 1
]


@pytest.mark.parametrize("seed", [1, 2])
def test_every_branch_of_the_slice_is_reached(seed):
    cs, want, _, undefined = run(seed)
    per_case = Counter()
    for c in cs:
        t = Counter()
        try:
            sc.reference(c, t)
        except DestinyUndefined:
            pass
        per_case.update(t.keys())
    thin = {b: per_case[b] for b in BRANCHES if per_case[b] < 5}
    assert not thin, f"branches reached by fewer than 5 cases: {thin}"
    events = Counter((e[2], e[4][-1] if e[2] in ("DoProximity", "DoRange", "DoTargetTracking") else None)
                     for e in want["E"])
    for name in ("DoProximity", "DoRange", "DoTargetTracking"):
        assert events[(name, 0)] > 0 and events[(name, 1)] > 0, (name, events)
    answers = Counter(q[2] for q in want["Q"] if q[6] not in ((), (0,), (-2,), (-999,)))
    assert all(answers[op] >= 5 for op in (sc.VIS_Q, sc.RANGE_BALL_Q, sc.RANGE_POINT_Q, sc.RANGE_DIST_Q,
                                           sc.BUBBLE_AT_Q, sc.CLEAN_FOLLOWERS_OP)), answers
    kinds = Counter(c["kind"] for c in cs)
    assert min(kinds.values()) >= 15 and sum(kinds.values()) >= 1600, kinds
    masters = Counter(c["master"] for c in cs)
    assert masters[True] > 1000 and masters[False] > 200, masters


def test_the_harness_runs_destinys_own_code():
    m = json.loads(sc.MANIFEST.read_text())
    assert m["destiny_commit"] == "114e89fa584f0e1a81eb0a679aabf41af8870072"
    assert m["command"][:5] == ["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math"]
    for name in ("Partition::Partition", "Partition::GetBox(double)", "Partition::GetNearbyBalls",
                 "Partition::ExplodeChildren", "Box::Box", "Box::~Box", "Box::Traverse", "Box::NearbyBubbles",
                 "BoxAllocator::ReleaseBox", "Ball::InsertInBoxes", "Ball::GetInflatedRadius", "Ball::DeleteFromBoxes",
                 "Ballpark::InsertInBoxes", "Ballpark::AddBall", "Ballpark::AddToBubble", "Ballpark::InitializeBubbles",
                 "Ballpark::InDeadBubble", "Ballpark::Gradient", "Ballpark::Potential", "CollideTwoSpheres",
                 "Quadratic", "Ballpark::HandleProximities", "Ball::CheckForProximity_Sensor",
                 "Ball::CheckForProximity_NotificationRange", "Ball::CheckForProximity_TargetTracking",
                 "Ballpark::CheckVisibility", "Ballpark::GetBallIdsInRange", "Ballpark::PyGetRemoteFollowers",
                 "Ballpark::Evolve", "Ballpark::EvolveBehaviorForBall", "Ballpark::StopAllFollowers"):
        assert len(m["extracted_sha256"][name]) == 64, name
    # the slice-1 functions are the same extractions as build.py's and build_modes.py's
    assert m["extracted_sha256"]["Ballpark::Integrate"].startswith("b03b4d51cb37")
    assert m["extracted_sha256"]["Ballpark::GotoThrust"].startswith("cdf7cfbb2b28")
    assert all("#pragma once" in w for w in m["compiler_warnings"])
    # COL-701: the gradient method is the default (Settings.h:9-10), the switch compiled at that default
    assert m["extracted_lines"]["g_useIterativeCollisionDefault"].strip() == \
        "constexpr bool g_useIterativeCollisionDefault = false;"
    assert m["extracted_lines"]["g_collisionMaxIterationsDefault"].strip() == \
        "constexpr size_t g_collisionMaxIterationsDefault = 20;"
    assert "g_useIterativeCollision = g_useIterativeCollisionDefault" in (sc.HARNESS.with_name("harness.cpp")
                                                                          .read_text())


# ---- the iterative collision method (COL-714 to COL-718), on its own scenarios ----

_IRUNS: dict = {}


def run_iterative(seed: int) -> tuple:
    if seed not in _IRUNS:
        cs = sc.iterative_cases(seed)
        want = sc.oracle(cs)
        got, undefined = sc.run_reference(cs)
        _IRUNS[seed] = (cs, want, got, undefined)
    return _IRUNS[seed]


def check_iterative(seed: int) -> None:
    cs, want, got, undefined = run_iterative(seed)
    diffs = differences(want, got, undefined)
    assert not diffs, {k: (len(v), v[:5]) for k, v in diffs.items()}
    assert len(undefined) <= len(cs) // 100, f"{len(undefined)} scenarios undefined: {sorted(undefined)}"
    assert len(got["S"]) > 25000


def test_iterative_method_matches_compiled_destiny_development_cases():
    check_iterative(1)


def test_iterative_method_matches_compiled_destiny_held_out_cases():
    check_iterative(2)


ITERATIVE_BRANCHES = ["iterative.found", "iterative.free", "iterative.fixed", "iterative.free_contact",
                      "iterative.static_contact", "iterative.no_impulse", "iterative.ignored", "iterative.substeps",
                      "iterative.invalidated", "iterative.limited", "iterative.coincident", "iterative.cap"]


@pytest.mark.parametrize("seed", [1, 2])
def test_every_branch_of_the_iterative_method_is_reached(seed):
    cs, _, _, _ = run_iterative(seed)
    per_case = Counter()
    for c in cs:
        t = Counter()
        try:
            sc.reference(c, t)
        except DestinyUndefined:
            pass
        per_case.update(t.keys())
    thin = {b: per_case[b] for b in ITERATIVE_BRANCHES if per_case[b] < 5}
    assert not thin, f"branches reached by fewer than 5 cases: {thin}"


def _caught_iterative(patch) -> bool:
    cs, want, _, _ = run_iterative(1)
    patch()
    got, undefined = sc.run_reference(cs)
    return bool(differences(want, got, undefined)) or bool(undefined)


def _impact_collider_minus_one(orig_init):
    def f(self, time):
        orig_init(self, time)
        self.collider = -1
    return f


ITERATIVE_FAULTS = {
    "max_iterations": lambda mp: mp.setattr(space, "COLLISION_MAX_ITERATIONS", 19),                  # COL-714
    "fuzzy_delta": lambda mp: mp.setattr(space, "FUZZY_DELTA", 1e-3),                                 # COL-715
    "velocity_limit": lambda mp: mp.setattr(space, "VELOCITY_LIMIT", 24.0),                           # COL-718
    "no_invalidation": lambda mp: mp.setattr(SpaceBallpark, "_remove_impact_from_free",               # COL-718
                                             lambda self, time, collider, b: None),
    # Destiny's value-initialised collider (0) on an impact without impulse, mended to -1 (COL-716, COL-717)
    "impact_collider": lambda mp: mp.setattr(space.CollisionImpact, "__init__",
                                             _impact_collider_minus_one(space.CollisionImpact.__init__)),
    # the gradient method in place of the iterative one (COL-701)
    "gradient_instead": lambda mp: mp.setattr(SpaceBallpark, "_collision_responses", lambda self: None),
}


@pytest.mark.parametrize("fault", sorted(ITERATIVE_FAULTS))
def test_planted_iterative_faults_are_caught(fault, monkeypatch):
    assert _caught_iterative(lambda: ITERATIVE_FAULTS[fault](monkeypatch)), f"the planted fault {fault} went unnoticed"


# ---- Destiny's undefined read of a deleted box, shown on a constructed scenario ----

def _deleted_box_case(prepared: int) -> dict:
    """`prepared` balls on a corner of 8 top boxes (8 chains of 8 boxes each) are moved to plain places (64 boxes
    released, 8 created, each); then ball 7 moves into its own neighbour box, in a top box without a bubble, so
    InsertInBoxes releases the box it is about to read."""
    top = sc.TOP
    recs = []
    for k in range(prepared):
        recs.append(("B", 0, {"id": 101 + k, "mass": 1e7, "radius": 1.0, "max_vel": 10.0, "free": True,
                              "global": False, "massive": False, "interactive": True, "junk": False,
                              "pos": [10.0 * (k + 1) * top, 10.0 * top, 0.0], "vel": [0.0, 0.0, 0.0],
                              "agility": 1.0, "sf": 1.0}))
    recs.append(("B", 0, {"id": 7, "mass": 1e7, "radius": 1.0, "max_vel": 100.0, "free": True, "global": False,
                          "massive": False, "interactive": True, "junk": False,
                          "pos": [top - 10.0, 0.5 * top, 0.5 * top], "vel": [0.0, 0.0, 0.0], "agility": 1.0,
                          "sf": 1.0}))
    for k in range(prepared):
        recs.append(sc._k(1, sc.POS_OP, 101 + k, x=1e9 + k * 1e7 + 123.4, y=2e9 + 77.7, z=3e9 + 55.5))
    recs.append(sc._k(1, sc.POS_OP, 7, x=top + 10.0, y=0.5 * top, z=0.5 * top))
    return {"id": 1, "kind": "deleted_box", "friction": 1e6, "tick": 1000, "steps": 2, "new_orbit": False,
            "master": True, "seed": 1, "start": 0, "records": recs}


@pytest.mark.parametrize("prepared", [0, 3])
def test_the_released_box_read_is_defined_while_its_memory_is_kept(prepared):
    c = _deleted_box_case(prepared)
    want, got = sc.oracle([c]), sc.reference(c)
    assert not differences(want, got, {})


def test_the_released_box_read_is_undefined_once_its_memory_is_deleted(monkeypatch):
    c = _deleted_box_case(4)
    with pytest.raises(DestinyUndefined):
        sc.reference(c)
    monkeypatch.setattr(space, "BOX_CHUNK_SIZE", 10 ** 9)      # pretend the memory were kept: Destiny differs
    diffs = differences(sc.oracle([c]), sc.reference(c), {})
    assert {d[3] for d in diffs["S"] if d[2] == 7} >= {"mod_x", "boxes"}


# ---- planted faults: each must be caught by the development comparison, on the scenarios it concerns ----

def _caught(kinds, monkeypatch, patch) -> bool:
    cs, want, _, _ = run(1)
    chosen = [c for c in cs if c["kind"] in kinds]
    ids = {c["id"] for c in chosen}
    want = {k: [row for row in v if row[0] in ids] for k, v in want.items()}
    patch()
    got, undefined = sc.run_reference(chosen)
    return bool(differences(want, got, undefined)) or bool(undefined)


FAULTS = {
    # SEN-705: the finest box 1 m wider
    "grid_unit": (("partition", "bubbles"), lambda mp: mp.setattr(
        space, "LEVEL_WIDTH", tuple(w * (481.0 / 480.0) for w in space.LEVEL_WIDTH))),
    # SEN-706: a ball filed at the finest level whose box is at least 2.9x (not 3x) its inflated radius
    "level_rule": (("partition",), lambda mp: mp.setattr(SpaceBallpark, "level_for", staticmethod(_level_29))),
    # SEN-706: the inflated radius without the tick of full thrust
    "inflation": (("partition", "collision"), lambda mp: mp.setattr(
        SpaceBallpark, "inflated_radius", lambda self, b: b.radius)),
    # SEN-707: NearbyBubbles scanning from +1 instead of -1
    "nearby_order": (("bubble_edges", "bubbles"), lambda mp: mp.setattr(
        SpaceBallpark, "_nearby_bubbles", _reversed_nearby(SpaceBallpark._nearby_bubbles))),
    # SEN-707: a removed ball deleted from its old bubble's dictionary instead of marked -1
    "bubble_removal": (("bubbles", "keepalive"), lambda mp: mp.setattr(
        SpaceBallpark, "_add_to_bubble", _no_minus(SpaceBallpark._add_to_bubble))),
    # SEN-708: the bubble filter of the candidates ignoring global balls' exemption
    "global_exemption": (("bubble_edges",), lambda mp: mp.setattr(
        SpaceBallpark, "_nearby_balls", _no_globals(SpaceBallpark._nearby_balls))),
    # TIME-714: keep-alive balls not keeping a bubble alive
    "keep_alive": (("keepalive",), lambda mp: mp.setattr(
        SpaceBallpark, "in_dead_bubble", _ignore_keep_alives(SpaceBallpark.in_dead_bubble))),
    # SEN-702: the sensor's reach summed in double instead of float
    "sensor_float": (("thresholds",), lambda mp: mp.setattr(space, "f32", lambda x: x)),
    # SEN-710: the range test strict
    "range_strict": (("thresholds",), lambda mp: mp.setattr(
        SpaceBallpark, "ball_ids_in_range", _strict_range(SpaceBallpark.ball_ids_in_range))),
    # SEN-701: the period test inclusive
    "sensor_period": (("sensor", "sensor_events"), lambda mp: mp.setattr(
        SpaceBallpark, "_check_sensor", _inclusive_period(SpaceBallpark._check_sensor))),
    # SEN-701: libstdc++'s order replaced by insertion order
    "map_order": (("sensor", "sensor_events", "mixed"), lambda mp: mp.setattr(
        space.StdHashMap, "_insert_bucket_begin", lambda self, k: self.order.append(k))),
    # SEN-709: a cloaked ball occluding
    "cloaked_occluder": (("visibility",), lambda mp: mp.setattr(
        SpaceBallpark, "check_visibility", _cloaked_occludes(SpaceBallpark.check_visibility))),
    # SEN-710: the reference ball's own radius added
    "range_radius": (("bubbles", "bubble_edges"), lambda mp: mp.setattr(
        SpaceBallpark, "ball_ids_in_range", _wider_range(SpaceBallpark.ball_ids_in_range))),
    # COL-711: 0.86 of the computed acceleration
    "response_scale": (("collision",), lambda mp: mp.setattr(space, "RESPONSE_SCALE", 0.86)),
    # COL-710: the light-ball ratio 24
    "light_ratio": (("collision",), lambda mp: mp.setattr(space, "LIGHT_BALL_RATIO", 24.0)),
    # COL-709/710: no recursion from separated positions
    "no_recursion": (("collision",), lambda mp: mp.setattr(
        SpaceBallpark, "_potential", _depth_cap(SpaceBallpark._potential))),
    # COL-705: CollideTwoSpheres taking the later root
    "later_root": (("collision",), lambda mp: mp.setattr(
        SpaceBallpark, "collide_two_spheres", staticmethod(_later_root))),
}


def _level_29(radius):
    level = 0
    for i in range(8):
        if 2.9 * radius > space.LEVEL_WIDTH[i]:
            break
        level = i
    return level


def _reversed_nearby(orig):
    def f(self, box):
        for i in (1, 0, -1):
            for j in (1, 0, -1):
                for k in (1, 0, -1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    other = self._check_box(box.key[0] + i, box.key[1] + j, box.key[2] + k, box.level)
                    if other is not None and other.bubble != -1:
                        return other.bubble
        return -1
    return f


def _no_minus(orig):
    def f(self, b):
        old = self.bubbles.get(b.old_bubble)
        had = old is not None and old.get(b.id) == 0
        orig(self, b)
        if had and old.get(b.id) == -1:
            del old[b.id]
    return f


def _no_globals(orig):
    def f(self, b, *flags):
        saved = {i: o.is_global for i, o in self.balls.items()}
        for o in self.balls.values():
            o.is_global = False
        try:
            return orig(self, b, *flags)
        finally:
            for i, g in saved.items():
                self.balls[i].is_global = g
    return f


def _ignore_keep_alives(orig):
    def f(self, b):
        saved, self.bubble_keep_alives = self.bubble_keep_alives, {}
        try:
            return orig(self, b)
        finally:
            self.bubble_keep_alives = saved
    return f


def _inclusive_period(orig):
    def f(self, b):
        if b.sensor_elapsed + self.dt == b.sensor_period:   # elapsed <= period: one tick more before the check
            b.sensor_elapsed += self.dt
            return None
        return orig(self, b)
    return f


def _cloaked_occludes(orig):
    def f(self, a, b):
        saved = {i: o.is_cloaked for i, o in self.balls.items()}
        for o in self.balls.values():
            o.is_cloaked = 0
        try:
            return orig(self, a, b)
        finally:
            for i, c in saved.items():
                self.balls[i].is_cloaked = c
    return f


def _wider_range(orig):
    def f(self, *args, include_dist=False):
        if len(args) >= 2 and isinstance(args[0], int) and args[0] in self.balls:
            args = (args[0], args[1] + self.balls[args[0]].radius, *args[2:])
        return orig(self, *args, include_dist=include_dist)
    return f


def _strict_range(orig):
    def f(self, *args, include_dist=False):
        out = orig(self, *args, include_dist=include_dist)
        if not out:
            return out
        if isinstance(args[0], int):
            center, rng = self.balls[args[0]].pos, args[1]
        else:
            center, rng = list(args[:3]), args[3]
        def strictly(i):
            o = self.balls[i[1] if include_dist else i]
            d2 = sum((o.pos[j] - center[j]) ** 2 for j in range(3))
            return d2 < (rng + o.radius) * (rng + o.radius)
        return [i for i in out if strictly(i)]
    return f


def _depth_cap(orig):
    def f(self, me, other, depth):
        return orig(self, me, other, max(depth, 2))
    return f


_COLLIDE = SpaceBallpark.collide_two_spheres


def _later_root(p0, p1, q0, q1, coll_radius):
    s = _COLLIDE(p0, p1, q0, q1, coll_radius)
    if s <= 0.0:
        return s
    p0q0 = [p0[i] - q0[i] for i in range(3)]
    dpq = [(p1[i] - p0[i]) - (q1[i] - q0[i]) for i in range(3)]
    a = sum(x * x for x in dpq)
    b = 2.0 * sum(p0q0[i] * dpq[i] for i in range(3))
    c = sum(x * x for x in p0q0) - coll_radius * coll_radius
    det = (b * b - 4.0 * a * c) ** 0.5
    s1 = (-b + det) * 0.5 / a
    return s1 if 0.0 <= s1 <= 1.0 else s


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    kinds, patch = FAULTS[fault]
    assert _caught(kinds, monkeypatch, lambda: patch(monkeypatch)), f"the planted fault {fault} went unnoticed"

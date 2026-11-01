# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Conformance against compiled code for the full ballpark: missiles, static collidables
(capsules, oriented boxes, miniballs, minicapsules, miniboxes, in both collision methods), cloaking, orientation with
and without dynamical orientation, CapAcceleration, mushrooms, trolls, formations, the accuracy helpers and OnTick of
the reference (exact_ballpark.full) against Destiny's own code compiled verbatim (conformance/build_full.py),
around slices 1 and 2.

Generated scenarios (tests/full_cases.py, thirteen families): missiles launched aimed or not, massive or not, from
moving, stationary and defending owners, sent with MissileFollow (and refused), their targets warping, cloaking,
removed or moved to another bubble, their owners gone; capsules and oriented boxes made directly or as minis of fixed
parents, approached at their cylinders, caps, faces, edges and corners, along their axes, grazed, started inside
(slowly and fast), removed at once or later, removed twice, re-added; cloaks of every mode with the master's sensor,
followers released, uncloaking in and out of warp proper; orientation from velocity (warp mass, aligning, snapping),
dynamical orientation (torque, roll, turning past 90 degrees, the missile exclusion); mushrooms growing and done;
trolls petrified inside Evolve (their ghost miniballs); formations loaded, changed, left, with slots in and out of
range, reserved and freed; GetAccuracy and the relative velocities; OnTick on server and client with backlogs, early
calls and the first tick; AddBall's clamps and the setters' refusals; and mixtures of all of these with slices 1 and
2, at ticks of 0.25 to 2 s, several frictions and both views. After every step every ball's state (with orientation,
rotation, angular velocity, torque, roll, formation, time stamps and minis), every static collidable, the partition's
checksum (with the static collidables) and box counts, the proximity balls in Destiny's order, the park's clock, local
id counters and moribund sets, the bubble dictionaries and every query's answer are compared for equality (doubles by
their bits), every box at the end of each scenario, and the events in Destiny's order (the final clean-ups of one call
as a set: Destiny walks unordered_sets of pointers there). Development cases (seed 1) were used while writing the
reference; held-out cases (seed 2) are a separate test, first run after the generator was frozen (generator frozen
16 October 2026, 19:25:12 UTC, tests/full_cases.py sha256
e34e846dd5947b671c382af2263f5011c19025861007afd8e51419b0bd1c68f2; first held-out run 19:25:19 UTC, which passed: the
comparison and the branch coverage of seed 2).

Where Destiny's result is undefined the reference refuses (DestinyUndefined); those scenarios are counted and left out
of the comparison, and constructed ones show the undefined behaviour is real on this machine: the harness dies on the
null capsule a stale RemoveCapsule leaves (Ballpark.cpp:5599, 3667) and never finishes the walk of a capsule whose
diameter is below half a float ulp of its length (Capsule.cpp:499, 578).

DirectXMath (23 October 2026): carbonengine/math's RotationQuaternion(yaw, pitch, roll)
(Quaternion.cpp:60-82) calls XMQuaternionRotationRollPitchYaw. Until then neither side had it and both stopped a
scenario with an "X" row at the same step (93 development and 83 held-out scenarios). The harness now compiles
DirectXMath at f5d82574 (its scalar path, _XM_NO_INTRINSICS_) and the reference transcribes it, so no scenario stops:
the X rows must be absent. The held-out cases' first run with DirectXMath (23 October 2026, 12:31:24 UTC) passed, with
4,696 ball-state rows after the former stops. The transcription is also held to the compiled function alone
(dxm_probe) on random angles. Skipped when the harness has not been built here.
"""

from __future__ import annotations

import json
import math
import resource
import subprocess
from collections import Counter, defaultdict

import pytest

import full_cases as fc
import exact_ballpark.full as full
import exact_ballpark.space as space
from exact_ballpark.full import FullBallpark
from exact_ballpark.modes import MISSILE, STOP
from exact_ballpark.ref import f32
from exact_ballpark.space import DestinyUndefined

pytestmark = pytest.mark.skipif(not fc.HARNESS.is_file(),
                                reason="Destiny full harness not built (conformance/build_full.py)")

_RUNS: dict = {}


def run(seed: int) -> tuple:
    """(scenarios, Destiny's rows, the reference's rows, the undefined scenarios, branch hits per scenario), cached per
    seed. The reference runs first: a scenario it refuses is not given to the harness (Destiny may crash or hang)."""
    if seed not in _RUNS:
        cs = fc.cases(seed)
        rows, undefined, coverage = defaultdict(list), {}, Counter()
        for c in cs:
            t = Counter()
            try:
                out = fc.reference(c, t)
            except DestinyUndefined as exc:
                undefined[c["id"]] = str(exc)
            else:
                for k, v in out.items():
                    rows[k].extend(v)
            coverage.update(t.keys())
        want = fc.oracle([c for c in cs if c["id"] not in undefined])
        _RUNS[seed] = (cs, want, rows, undefined, coverage)
    return _RUNS[seed]


def check(seed: int) -> None:
    cs, want, got, undefined, _ = run(seed)
    diffs = fc.differences(want, got, undefined)
    assert not diffs, {k: (len(v), v[:5]) for k, v in diffs.items()}
    assert len(undefined) <= len(cs) // 200, f"{len(undefined)} scenarios undefined: {sorted(undefined.items())}"
    assert len(got["S"]) > 100000 and len(got["T"]) > 10000 and len(got["E"]) > 8000, {k: len(v) for k, v in
                                                                                        got.items()}
    assert not got["X"] and not want["X"], "no scenario stops for DirectXMath now"


def test_reference_matches_compiled_destiny_development_cases():
    check(1)


def test_reference_matches_compiled_destiny_held_out_cases():
    check(2)


BRANCHES = [
    "missile.straight", "missile.pursue", "missile.on_target", "missile.other_bubble", "missile.cloaked_target",  # 723/4
    "launch.aimed", "launch.stationary", "launch.straight",                                                    # MOV-725
    "missile.collision_event", "missile.self_collision",                                                       # COL-719
    "candidates.missile_target", "candidates.defender", "gradient.skip", "gradient.static_skip",              # COL-703/4
    "iterative.static_skip",
    "capsule.check_cap", "capsule.check_cylinder", "capsule.check_inside", "capsule.cylinder_contact",         # COL-712
    "capsule.inside_cylinder", "capsule.parallel", "capsule.sphere", "capsule.reflect", "capsule.eject",
    "capsule.outward", "capsule.no_volume", "capsule.unchanged", "capsule.existing",
    "obox.excluded", "obox.reflect", "obox.eject", "obox.huge", "obox.unchanged", "obox.existing",
    "static.equal_time", "nearby.static_other_bubble",                                                         # SEN-708
    "static.removed", "static.delayed", "static.buffer", "static.null_entry", "miniball.ghost",               # COL-713
    "iterative.static", "iterative.static_contact", "iterative.spin_limited", "iterative.limited",           # COL-715-8
    "iterative.invalidated", "ellipse.turned",
    "cloak.applied", "cloak.refused", "cloak.sensor", "uncloak.massive", "uncloak.sensor_erase",               # SEN-711
    "uncloak.warping",
    "cap_acceleration", "mode.inert",                                                                          # 727, 734
    "orient.thrust", "orient.behind", "orient.coast", "orient.missile_snap", "torque.full", "torque.brake_only",  # 728
    "torque.none", "torque.flip", "roll.turn", "roll.brake_only", "roll.none", "snap.at_rest",
    "orbit_new.heading_normal",
    "ypr.evolve", "ypr.snap", "ypr.warp_mass",                                                                 # MOV-729
    "formation.slot", "formation.zero_offset", "formation.no_slot", "formation.no_formation",                 # MOV-726
    "formation.shrink", "formation.leave", "formation.refused",
    "troll.petrified", "mushroom.grow", "mushroom.done",                                                       # MOV-736
    "accuracy.falloff",                                                                                        # CMB-702
    "tick.first", "tick.too_early", "tick.resync", "tick.catch_up", "tick.dead_bubble",                       # TIME-702-6
    "add.existing", "remove.now", "remove.delayed", "nearby.moribund",                                         # MOV-711
]


@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_every_branch_of_the_slice_is_reached(seed):
    cs, want, _, undefined, coverage = run(seed)
    thin = {b: coverage[b] for b in BRANCHES if coverage[b] < 5}
    assert not thin, f"branches reached by fewer than 5 scenarios: {thin}"
    events = Counter(e[2] for e in want["E"])
    assert events["DoCollision"] > 100 and events["DoFinalCleanup"] > 100, events
    assert events["DoPreTick"] > 100 and events["DoPostTick"] == events["DoPreTick"], events
    statics_out = {e[3] for e in want["E"] if e[2] == "DoFinalCleanup"} & ({5000 + j for j in range(10)}
                                                                            | {5100 + j for j in range(4)})
    assert statics_out, "no static collidable was cleaned up"
    answered = Counter(q[2] for q in want["Q"] if q[5] == 1)
    for op in (fc.ACCURACY_Q, fc.RELVEL_Q, fc.RESERVE_SLOT_Q, fc.SHIFTED_Q, fc.ROTATED_Q, fc.DIST_Q):
        assert answered[op] >= 5, (op, answered)
    kinds = Counter(c["kind"] for c in cs)
    assert min(kinds.values()) >= 70 and sum(kinds.values()) >= 1900 and len(kinds) == 13, kinds
    flags = Counter()
    for c in cs:
        flags.update(k for k in ("master", "iterative", "do", "do_missiles_off", "manual", "new_orbit") if c.get(k))
    assert flags["master"] > 1000 and len(cs) - flags["master"] > 300, flags
    assert flags["iterative"] > 300 and flags["do"] > 300 and flags["manual"] > 100, flags
    assert flags["do_missiles_off"] > 20 and flags["new_orbit"] > 30, flags
    assert len({c["tick"] for c in cs}) >= 4


def test_the_harness_runs_destinys_own_code():
    m = json.loads(fc.MANIFEST.read_text())
    assert m["destiny_commit"] == "114e89fa584f0e1a81eb0a679aabf41af8870072"
    assert m["math_commit"] == "8273322093eb1f80168ae60a35891f2d454369f9"
    # -fsigned-char: Destiny's platforms (x86, MSVC) have a signed char; aarch64's is unsigned (mFormationID -1)
    assert m["command"][:6] == ["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math", "-fsigned-char"]
    for name in ("Ballpark::EvolveMissile", "Ballpark::MissileFollow", "Ballpark::PyLaunchMissile",
                 "Ball::DispatchCollisions", "Partition::GetCollisionCandidates", "Ballpark::Gradient",
                 "Capsule::HandleCollisionNonIteratively", "Capsule::CheckCollision", "Capsule::ReactToCollision",
                 "Capsule::InsertInBoxes", "Capsule::CollideWithBall", "OrientedBox::CheckCollision",
                 "OrientedBox::ReactToCollision", "OrientedBox::InsertInBoxes", "OrientedBox::CollideWithBall",
                 "BoxShape::CollideWithSphere", "BoxShape::RayIntersect", "IntersectSphereTriangle",
                 "Ballpark::AddCapsule", "Ballpark::AddOrientedBox", "Ballpark::RemoveCapsule",
                 "Ballpark::RemoveOrientedBox", "Ballpark::BringOutDeadCapsules", "Ballpark::AddMiniball",
                 "Ball::AddActualMinicapsule", "Ball::AddActualMiniBox", "Ball::PyAddMiniCapsule",
                 "Ballpark::GetNextCollisionStatic", "Ballpark::CalculateCollisionImpactStatic",
                 "Ballpark::ApproximateEllipsePointOfImpact", "Ballpark::AddImpact", "Ballpark::CloakBall",
                 "Ballpark::UncloakBall", "Ballpark::CapAcceleration", "Ballpark::GotoTorqueBreaks",
                 "Ballpark::GotoRollBreaks", "Ballpark::ApplyTorque", "Ballpark::CalculateBallRotationVelocity",
                 "CollisionBallProperties::CalculateRotation", "Ball::CalculateYawPitchRoll", "Ball::SnapOrientation",
                 "Ball::GetRotatedVector", "Ballpark::EvolveFormation", "Ballpark::SetBallFormation",
                 "Ballpark::FormationFollow", "Ball::ReserveFormationSlot", "Ball::FreeFormationSlot",
                 "Ballpark::SetBallTroll", "Ballpark::PetrifyTroll", "Ballpark::AddMushroom",
                 "Ballpark::GetAccuracy", "Ball::CalculateRelativeVelocities", "Ballpark::OnTick",
                 "ClientBall::GetShiftedTime", "Ballpark::Evolve", "Ballpark::EvolveBehaviorForBall"):
        assert len(m["extracted_sha256"][name]) == 64, name
    # the slice-1 and slice-2 functions are the same extractions as build.py's and build_space.py's
    assert m["extracted_sha256"]["Ballpark::Integrate"].startswith("b03b4d51cb37")
    assert m["extracted_sha256"]["Ballpark::GotoThrust"].startswith("cdf7cfbb2b28")
    # carbonengine/math: the inline vector, quaternion and matrix functions verbatim; RotationQuaternion(yaw, pitch,
    # roll) verbatim too, through DirectXMath's scalar path at the pinned commit, its headers checked and read by path
    for name in ("TransformCoord", "RotationMatrix(Quaternion)", "RotationQuaternion(axis, angle)",
                 "Normalize(Quaternion)", "Quaternion::operator*(Quaternion)", "Cross(Vector3)",
                 "RotationQuaternion(yaw, pitch, roll)", "Quaternion::Quaternion(XMVECTOR)"):
        assert len(m["math_extracted_sha256"][name]) == 64, name
    assert m["math_rotation_ypr"]["name"] == "RotationQuaternion(yaw, pitch, roll)"
    assert "XMQuaternionRotationRollPitchYaw(pitch, yaw, roll)" in m["math_rotation_ypr"]["body"]
    assert m["math_kept_lines"] == {"struct Quaternion": ["\tQuaternion( const XMVECTOR& other );"]}
    assert m["directxmath_commit"] == "f5d82574b03417d905fdd2f85bfa9bf7a2f1deac"
    assert m["command"][6] == m["directxmath_define"] == "-D_XM_NO_INTRINSICS_"
    assert m["directxmath_files_read"] == m["probe_directxmath_files_read"] == sorted(m["directxmath_header_sha256"])
    snapshot = fc.DXM_MANIFEST.read_text().splitlines() if fc.DXM_MANIFEST.is_file() else []
    for f, h in m["directxmath_header_sha256"].items():          # the snapshot's own checksums, where present
        assert not snapshot or f"{h}  ./{f}" in snapshot, f
    assert all("#pragma once" in w for w in m["compiler_warnings"])
    assert m["extracted_lines"]["g_useDynamicalOrientationDefault"].strip() == \
        "constexpr bool g_useDynamicalOrientationDefault = false;"
    assert m["extracted_lines"]["g_disableDynamicalOrientationForMissilesDefault"].strip() == \
        "constexpr bool g_disableDynamicalOrientationForMissilesDefault = false;"


# ---- Destiny's undefined behaviour and its missing dependency, on constructed scenarios ----

def _one(records: list, steps: int = 3, **kw) -> dict:
    c = {"id": 1, "kind": "constructed", "friction": 1.0e6, "tick": 1000, "steps": steps, "new_orbit": False,
         "iterative": False, "do": False, "do_missiles_off": False, "manual": False, "master": True, "seed": 7,
         "start": 0, "records": records}
    c.update(kw)
    return c


def _no_core_dump() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))       # a crash here is expected: no core file in the folder


def _harness(c: dict, timeout: float):
    return subprocess.run([str(fc.HARNESS)], input=fc.serialize([c]), capture_output=True, text=True,
                          timeout=timeout, preexec_fn=_no_core_dump)


def test_a_capsule_added_on_a_stale_entry_is_undefined_in_destiny():
    """RemoveCapsule of an unknown id inserts a null entry (operator[], Ballpark.cpp:5599); AddCapsule of that id then
    dereferences it (3667): the reference refuses, and the compiled code dies."""
    c = _one([fc._k(0, fc.REMOVE_CAPSULE_OP, 4999), fc._k(0, fc.CAPSULE_OP, 4999, d=(0.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                                                                                      3.0))])
    with pytest.raises(DestinyUndefined):
        fc.reference(c)
    assert _harness(c, 60).returncode != 0


def test_a_capsule_walk_that_never_ends_is_undefined_in_destiny():
    """Capsule::InsertInBoxes steps 2r along the float length (Capsule.cpp:499, 578); at r = 1e-45 (a float
    denormal) the length never shrinks: the reference refuses, the compiled code does not return."""
    c = _one([fc._parent(9000), fc._k(0, fc.MINICAPSULE_OP, 9000, d=(-20.0, 0.0, 0.0, 20.0, 0.0, 0.0, 1e-45))])
    with pytest.raises(DestinyUndefined):
        fc.reference(c)
    with pytest.raises(subprocess.TimeoutExpired):
        _harness(c, 3)
    ok = _one([fc._parent(9000), fc._k(0, fc.MINICAPSULE_OP, 9000, d=(-20.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.05))])
    assert not fc.differences(fc.oracle([ok]), fc.reference(ok), {})


def test_a_turned_leader_and_a_turned_launcher_are_computed_by_both():
    """MOV-726 and MOV-725 through DirectXMath: a formation offset turned by a leader that never moved (yaw, pitch and
    roll +0: the identity), then by a turned leader, and a missile launched from a launcher that turned and stopped
    (its rotated (0, 0, -1)): both sides compute every step (until 23 October both stopped with an X row), equal."""
    forms = (1.0, 2.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0)
    r = fc.random.Random(1)
    still = [fc._k(0, fc.FORMATIONS_OP, d=forms), fc._ball(r, 1, pos=[0.0, 0.0, 0.0], free=False),
             fc._k(0, fc.SET_FORMATION_OP, 1, w=0), *fc._test_ship(2, pos=(500.0, 0.0, 0.0)),
             fc._k(1, fc.FORMATION_FOLLOW_OP, 2, 1, w=1)]
    c = _one(still, steps=5)
    want, got = fc.oracle([c]), fc.reference(c)
    assert not fc.differences(want, got, {}) and not got.get("X")
    assert any(row[2] == 2 and row[3] == full.FORMATION for row in got["S"])
    moving = [fc._k(0, fc.FORMATIONS_OP, d=forms), fc._ball(r, 1, pos=[0.0, 0.0, 0.0], vel=[30.0, 5.0, 0.0]),
              fc._k(0, fc.SET_FORMATION_OP, 1, w=0), *fc._test_ship(2, pos=(500.0, 0.0, 0.0)),
              fc._k(1, fc.FORMATION_FOLLOW_OP, 2, 1, w=1)]
    c = _one(moving, steps=5)
    want, got = fc.oracle([c]), fc.reference(c)
    assert not want.get("X") and not got.get("X") and not fc.differences(want, got, {})
    yaw = fc.S_FIELDS.index("yaw")
    assert any(row[2] == 1 and row[yaw] != 0.0 for row in got["S"])          # the leader turned
    launcher = [fc._ball(r, 10, pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], max_vel=120.0),
                fc._k(0, fc.GOTO_OP, 10, d=(5000.0, 2000.0, -3000.0)),
                fc._k(2, fc.VEL_OP, 10, d=(0.0, 0.0, 0.0)),     # at rest: snapped toward its goto point, turned
                fc._ball(r, 12, step=2, pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0]),
                fc._k(2, fc.LAUNCH_OP, 12, 0, 10, w=0)]         # no target: along the launcher's turned (0, 0, -1)
    c = _one(launcher, steps=4)
    want, got = fc.oracle([c]), fc.reference(c)
    assert not want.get("X") and not got.get("X") and not fc.differences(want, got, {})
    vx = fc.S_FIELDS.index("vx")
    missile = [row for row in got["S"] if row[1] == 2 and row[2] == 12]   # flying straight on (GotoDirection)
    assert missile and missile[0][vx] != 0.0                              # along the turned vector, not (0, 0, -1)


def test_rotation_quaternion_is_directxmaths():
    """The reference's RotationQuaternion(yaw, pitch, roll) against the compiled function alone (the build's dxm_probe:
    carbonengine/math's code and DirectXMath's scalar path, compiled as in the harness), bit for bit on 100,000 angle
    triples: half in a turn either way, the rest tiny, large or any bit pattern (subnormals, infinities, NaNs)."""
    r = fc.random.Random(23)

    def pat() -> int:
        k = r.random()
        if k < 0.5:
            return fc.struct.unpack("<I", fc.struct.pack("<f", r.uniform(-7.0, 7.0)))[0]
        if k < 0.7:
            return fc.struct.unpack("<I", fc.struct.pack("<f", r.uniform(-1e-3, 1e-3)))[0]
        if k < 0.8:
            return fc.struct.unpack("<I", fc.struct.pack("<f", r.uniform(-1e6, 1e6)))[0]
        return r.getrandbits(32)
    rows = [(pat(), pat(), pat()) for _ in range(100000)] + [(0, 0, 0), (0x80000000, 0, 0), (0x7f800000, 0, 0)]
    out = subprocess.run([str(fc.DXM_PROBE)], input="".join(f"{a:08x} {b:08x} {c:08x}\n" for a, b, c in rows),
                         capture_output=True, text=True, check=True).stdout.splitlines()
    as_f = lambda u: fc.struct.unpack("<f", fc.struct.pack("<I", u))[0]
    bad = []
    for (a, b, c), line in zip(rows, out):
        want = [as_f(int(x, 16)) for x in line.split()]
        got = full.q_rotation_ypr(as_f(a), as_f(b), as_f(c))
        if not all(fc.same(w, g) for w, g in zip(want, got)):
            bad.append((a, b, c))
    assert len(out) == len(rows) and not bad, bad[:5]
    assert full.q_rotation_ypr(0.0, 0.0, 0.0) == full.QUAT_IDENTITY


def test_followers_left_in_another_bubble_are_found_for_the_pre_tick():
    """MOV-731's compiled half: GetRemoteFollowers (Thunkers.cpp:1193-1236) names the free balls in a follow mode
    (followModes: FOLLOW, ORBIT, MISSILE, FORMATION) whose leader is in another bubble; the pre-tick stops them
    (Actions.clean_up_followers, tests/test_destiny_server_layer.py). A leader with a formation, a missile, an orbiter
    and a follower on it; the leader moved to another top box."""
    r = fc.random.Random(5)
    forms = (1.0, 3.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0)
    recs = [fc._k(0, fc.FORMATIONS_OP, d=forms), fc._ball(r, 10, pos=[0.0, 0.0, 0.0], free=False, massive=True),
            fc._k(0, fc.SET_FORMATION_OP, 10, w=0)]
    for bid in (11, 12, 13, 14):
        recs += fc._test_ship(bid, pos=(300.0 * (bid - 10), 50.0, 0.0))
    recs += [fc._k(1, fc.FORMATION_FOLLOW_OP, 11, 10, w=0), fc._k(1, fc.FOLLOW_OP, 12, 10, d=fc._xyz(rr=200.0)),
             fc._k(1, fc.ORBIT_OP, 13, 10, d=fc._xyz(rr=500.0)), fc._k(1, fc.MISSILE_FOLLOW_OP, 14, 10, 11),
             fc._k(1, fc.CLEAN_FOLLOWERS_OP), fc._k(2, fc.POS_OP, 10, d=(5 * fc.TOP, 0.0, 0.0)),
             fc._k(3, fc.CLEAN_FOLLOWERS_OP), fc._k(4, fc.CLEAN_FOLLOWERS_OP)]
    c = _one(recs, steps=4)
    want, got = fc.oracle([c]), fc.reference(c)
    assert not fc.differences(want, got, {})
    answers = [q[6] for q in want["Q"] if q[2] == fc.CLEAN_FOLLOWERS_OP]
    assert answers[0] == () and set(answers[1]) == {11, 12, 13, 14} and answers[2] == ()


# ---- planted faults: each must be caught by the development comparison, on the scenarios it concerns ----

_BY_CASE: dict = {}


def _want_by_case() -> dict:
    if not _BY_CASE:
        cs, want, _, undefined, _ = run(1)
        for kind, rows in want.items():
            for row in rows:
                _BY_CASE.setdefault(row[0], defaultdict(list))[kind].append(row)
    return _BY_CASE


def _caught(kinds: tuple) -> bool:
    """Runs the patched reference on the development scenarios of `kinds`, one at a time, until one differs from
    Destiny's rows (or is refused); False when none does."""
    cs, _, _, undefined, _ = run(1)
    by_case = _want_by_case()
    for c in cs:
        if c["kind"] not in kinds or c["id"] in undefined:
            continue
        try:
            got = fc.reference(c)
        except DestinyUndefined:
            return True
        if fc.differences(by_case.get(c["id"], {}), got, {}):
            return True
    return False


def _after(name: str, extra):
    """A FullBallpark method that runs as before, then `extra(self, result, *args)`."""
    orig = getattr(FullBallpark, name)

    def f(self, *args, **kw):
        result = orig(self, *args, **kw)
        extra(self, result, *args)
        return result
    return f


def _speed_fraction_095(self, result, src, dst, owner):
    m = self.balls.get(int(src))
    if m is not None and m.mode == MISSILE:
        m.speed_fraction = f32(0.95)


def _aim_at_surface(self, result, src, dst, owner):
    m = self.balls.get(int(src))
    if m is not None and m.mode == MISSILE:
        m.follow_range = 0.0


def _uncloak_non_massive(self, result, bid):
    b = self.balls.get(bid)
    if b is not None:
        b.is_massive = False


def _troll_interactive(self, result, b):
    if b.id in self.balls and b.mode == full.RIGID:
        b.is_interactive = True


def _remove_due_boxes(self, result):
    if self.in_evolve:
        return
    for ob in sorted(self.moribund_boxes, key=lambda o: o.id):
        if ob.effect_stamp - self.current_time < 0:
            self.remove_oriented_box(ob.id)
            self.moribund_boxes.discard(ob)


def _cloak_keeps_followers(orig):
    def f(self, bid, mode, range_=2000.0):
        self._stop_all_followers = lambda b: None
        try:
            return orig(self, bid, mode, range_)
        finally:
            del self._stop_all_followers
    return f


def _static_response_086(orig):
    def f(self, b, s_id, last_c, toi):
        return orig(self, b, s_id, [x * (0.86 / 0.85) for x in last_c], toi)
    return f


def _start_velocity(orig):
    def f(self, c, b, p0, p1, m1, vp1):
        return orig(self, c, b, p0, p1, m1, list(b.vel))
    return f


def _moribund_statics_excluded(orig):
    def f(self, b, *flags):
        balls, stat = orig(self, b, *flags)
        return balls, [s for s in stat if not s.is_moribund]
    return f


def _target_filtered(orig):
    def f(self, b):
        balls, stat = orig(self, b)
        if b.mode == MISSILE and b.owner_id >= 0:
            balls = [o for o in balls if o.is_massive and not o.is_cloaked]
        return balls, stat
    return f


_ORIG_TAU = FullBallpark._tau
_ORIG_NEXT_STATIC = FullBallpark._next_collision_static


def _statics_after_fixed(self, prop, item, balls, collidables=()):
    """GetNextCollisionStatic with the fixed balls searched before the static collidables."""
    space.SpaceBallpark._next_collision_static(self, prop, item, balls)
    _ORIG_NEXT_STATIC(self, prop, item, [], collidables)


def _elastic_095(orig):
    def f(self, prop, item):
        orig(self, prop, item)
        if item.impact is not None and item.impact.dv is not None:
            item.impact.dv = [x * 0.95 for x in item.impact.dv]
    return f


def _first_tick_events(orig):
    def f(self, sim_time, real_time=0):
        if self.first_time and int(sim_time) - self.time >= self.tick_interval * 10000:
            self._event("DoPreTick", -2, self.current_time)
        return orig(self, sim_time, real_time)
    return f


def _deads_only_when_ticking(orig):
    def f(self, sim_time, real_time=0):
        if int(sim_time) - self.time < self.tick_interval * 10000:
            return None
        return orig(self, sim_time, real_time)
    return f


def _damped_with_do(orig):
    def f(self, b):
        if self.dynamical_orientation and b.mode == STOP:
            b.vel[1] = (b.vel[1] - 0.07 * b.vel[1]) * full.STOP_DAMP
        return orig(self, b)
    return f


def _unturned_formation(orig):
    def f(self, b):
        self._rotated_vector = lambda ball, vec: vec
        try:
            return orig(self, b)
        finally:
            del self._rotated_vector
    return f


def _accuracy_wide_falloff(orig):
    def f(self, src, dst, optimal, falloff, tracking, signature, optimal_signature):
        return orig(self, src, dst, optimal, float(falloff) * math.sqrt(2.0), tracking, signature, optimal_signature)
    return f


def _relvel_without_rotation(orig):
    def f(self, src, dst, flags):
        saved = self.dynamical_orientation
        self.dynamical_orientation = False
        try:
            return orig(self, src, dst, flags)
        finally:
            self.dynamical_orientation = saved
    return f


def _capped(self, b, a):
    """An agility cap: the thrust shortened by 0.1 % whenever it is not along the velocity."""
    if any(a) and any(b.vel) and space._dot(a, b.vel) ** 2 < space._dot(a, a) * space._dot(b.vel, b.vel) * 0.999999:
        return [x * 0.999 for x in a]
    return a


MISSILES, STATICS = ("missile", "corner", "mixed"), ("static", "approach", "corner")

FAULTS = {
    # MOV-723: the straight phase a tick longer at the default tick
    "missile_straight": (MISSILES, lambda mp: mp.setattr(full, "MISSILE_STRAIGHT_MS", 1000)),
    # MOV-725: the launch speed 151 m/s
    "launch_speed": (MISSILES, lambda mp: mp.setattr(full, "MISSILE_LAUNCH_SPEED", 151.0)),
    # MOV-724: the missile's speed fraction 0.95
    "missile_speed_fraction": (MISSILES, lambda mp: mp.setattr(
        FullBallpark, "missile_follow", _after("missile_follow", _speed_fraction_095))),
    # MOV-724: aimed at the surface (follow range 0), not the centre
    "missile_aim": (MISSILES, lambda mp: mp.setattr(
        FullBallpark, "missile_follow", _after("missile_follow", _aim_at_surface))),
    # COL-719: no self-collision for an orphaned missile
    "no_self_collision": (MISSILES, lambda mp: mp.setattr(FullBallpark, "_orphaned", lambda self, b: False)),
    # COL-719: collisions dispatched below sqrt(0.1) m of movement too
    "dispatch_threshold": (MISSILES, lambda mp: mp.setattr(full, "DISPATCH_MIN_MOVE_SQ", 0.0)),
    # COL-703: a normal missile's target filtered like other candidates (Destiny adds it unfiltered)
    "target_filtered": (MISSILES, lambda mp: mp.setattr(
        FullBallpark, "_collision_candidates", _target_filtered(FullBallpark._collision_candidates))),
    # COL-712: the static response 0.86
    "static_response": (STATICS, lambda mp: mp.setattr(
        FullBallpark, "_static_response", _static_response_086(FullBallpark._static_response))),
    # COL-712: ejected by half a metre more than the penetration, not one
    "eject_margin": (STATICS, lambda mp: mp.setattr(full, "EJECT_MARGIN", 0.5)),
    # COL-712: the capsule's parameter along the start velocity (Destiny uses the end velocity, Capsule.cpp:214)
    "capsule_start_velocity": (STATICS, lambda mp: mp.setattr(
        FullBallpark, "_capsule_non_iterative", _start_velocity(FullBallpark._capsule_non_iterative))),
    # COL-712: moribund static collidables no longer colliding (Destiny keeps them, Partition.cpp:459-472)
    "moribund_statics": (STATICS, lambda mp: mp.setattr(
        FullBallpark, "_nearby", _moribund_statics_excluded(FullBallpark._nearby))),
    # COL-713: a minicapsule's bounding radius |AB| + r
    "capsule_bounding": (STATICS, lambda mp: mp.setattr(
        full.Capsule, "bounding_radius", lambda self: full._f(full._f(full._length(self.ab)) + self.radius))),
    # COL-713: oriented boxes brought out like capsules (Destiny never removes them, Ballpark.cpp:5699-5706)
    "boxes_brought_out": (STATICS, lambda mp: mp.setattr(
        FullBallpark, "bring_out_the_deads", _after("bring_out_the_deads", _remove_due_boxes))),
    # MOV-732: freeing a parent leaves its minis in place
    "minis_kept": (STATICS, lambda mp: mp.setattr(FullBallpark, "_remove_minis", lambda self, b: None)),
    # COL-715: the fixed balls searched before the static collidables
    "statics_after_fixed": (STATICS, lambda mp: _patch_static_order(mp)),
    # COL-716: the static impulse 0.95 of the elastic one
    "inelastic_statics": (STATICS, lambda mp: mp.setattr(
        FullBallpark, "_impact_static", _elastic_095(FullBallpark._impact_static))),
    # COL-718: the spin limit 3 maxAngVel^2
    "spin_limit": (STATICS + ("orient_do",), lambda mp: mp.setattr(full, "SPIN_LIMIT", 3)),
    # SEN-711: the cloak's sensor period 2.5 s
    "cloak_period": (("cloak", "missile", "corner"), lambda mp: mp.setattr(full, "CLOAK_SENSOR_PERIOD", 2.5)),
    # SEN-711: an uncloaked ball left non-massive
    "uncloak_massive": (("cloak", "missile", "corner"), lambda mp: mp.setattr(
        FullBallpark, "uncloak_ball", _after("uncloak_ball", _uncloak_non_massive))),
    # MOV-730: a cloak keeping its followers
    "cloak_followers": (("cloak", "missile", "corner"), lambda mp: mp.setattr(
        FullBallpark, "cloak_ball", _cloak_keeps_followers(FullBallpark.cloak_ball))),
    # MOV-727: an agility cap on the thrust
    "agility_cap": (("orient", "missile", "mixed"), lambda mp: mp.setattr(FullBallpark, "_cap_acceleration", _capped)),
    # MOV-728: tau with the rotational agility twice
    "do_tau": (("orient_do",), lambda mp: mp.setattr(
        FullBallpark, "_tau", lambda self, b: full._f(_ORIG_TAU(self, b) * b.rot_agility))),
    # MOV-728: STOP's vertical damping kept with dynamical orientation
    "do_stop_damping": (("orient_do", "missile", "mixed"), lambda mp: mp.setattr(
        FullBallpark, "_behaviour", _damped_with_do(FullBallpark._behaviour))),
    # MOV-729: the roll decay -0.45
    "roll_decay": (("orient", "missile", "mixed"), lambda mp: mp.setattr(full, "ROLL_DECAY", -0.45)),
    # MOV-726: the slot offset not turned by the leader's rotation
    "formation_unturned": (("formation",), lambda mp: mp.setattr(
        FullBallpark, "_evolve_formation", _unturned_formation(FullBallpark._evolve_formation))),
    # MOV-736: a troll's minimum delay 0
    "troll_delay": (("troll",), lambda mp: mp.setattr(full, "TROLL_MIN_DELAY", 0)),
    # MOV-736: a petrified troll left interactive
    "troll_interactive": (("troll",), lambda mp: mp.setattr(
        FullBallpark, "_petrify_troll", _after("_petrify_troll", _troll_interactive))),
    # CMB-702: the falloff term halved (falloff^2 doubled)
    "accuracy_falloff": (("accuracy",), lambda mp: mp.setattr(
        FullBallpark, "get_accuracy", _accuracy_wide_falloff(FullBallpark.get_accuracy))),
    # CMB-701: the ball's own rotation not subtracted with dynamical orientation
    "angular_without_rotation": (("accuracy",), lambda mp: mp.setattr(
        FullBallpark, "relative_velocities", _relvel_without_rotation(FullBallpark.relative_velocities))),
    # TIME-703: the server's backlog limit 6 ticks
    "catch_up_limit": (("ontick",), lambda mp: mp.setattr(full, "MAX_CATCH_UP_TICKS", 6)),
    # TIME-704: a pre-tick event on the first tick
    "first_tick_events": (("ontick",), lambda mp: mp.setattr(
        FullBallpark, "on_tick", _first_tick_events(FullBallpark.on_tick))),
    # TIME-705: moribund balls brought out only when a tick runs
    "deads_when_ticking": (("ontick",), lambda mp: mp.setattr(
        FullBallpark, "on_tick", _deads_only_when_ticking(FullBallpark.on_tick))),
    # DirectXMath: the yaw-pitch-roll rotation taken as the identity everywhere
    "directxmath_identity": (("orient", "formation", "missile"), lambda mp: mp.setattr(
        full, "q_rotation_ypr", lambda yaw, pitch, roll: full.QUAT_IDENTITY)),
    # DirectXMath: XMQuaternionRotationRollPitchYaw given (yaw, pitch, roll) in carbonengine's order, not (pitch, yaw,
    # roll), so the yaw turns about the x axis and the pitch about the y axis
    "directxmath_argument_order": (("orient", "formation", "missile"), lambda mp: mp.setattr(
        full, "q_rotation_ypr", lambda yaw, pitch, roll, _q=full.q_rotation_ypr: _q(pitch, yaw, roll))),
    # DirectXMath: the whole angles instead of the half angles (DirectXMathMisc.inl:512)
    "directxmath_whole_angles": (("orient", "formation", "missile"), lambda mp: mp.setattr(
        full, "q_rotation_ypr", lambda yaw, pitch, roll, _q=full.q_rotation_ypr: _q(
            full._f(yaw * 2.0), full._f(pitch * 2.0), full._f(roll * 2.0)))),
}

def _patch_static_order(mp) -> None:
    mp.setattr(FullBallpark, "_next_collision_static", _statics_after_fixed)


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    kinds, patch = FAULTS[fault]
    patch(monkeypatch)
    assert _caught(kinds), f"the planted fault {fault} went unnoticed"

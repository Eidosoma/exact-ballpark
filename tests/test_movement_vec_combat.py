# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The batched combat pieces (exact_ballpark.vec_combat.VecCombatPark: missiles, formations, cloaking at the park; its
tick as four CUDA launches in exact_ballpark.vec_combat_cuda, and on tensors in evolve_torch) against the scalar reference
(exact_ballpark.full.FullBallpark), bit for bit, every field the batch holds of every ball at every step, which balls
exist, the ReserveFormationSlot answers and the DoCollision events, on generated scenarios (tests/
destiny_combat_cases.py): on the CPU (the tensor tick, for the worlds it holds: no warp) and, when a CUDA device is
visible, on the GPU (the kernels for every world, a CUDA graph against eager ticks, the tensor tick against the
kernels). The reference is held to Destiny's compiled code by tests/test_movement_full_oracle.py, so this is exact
against Destiny. Development scenarios are seed 1, held-out seed 2 (first compared after the generator was frozen).
Worlds the batch refuses (its domain flags: here two massive balls outside MISSILE mode) are counted, never compared
as equal. Planted faults, on the tensor tick and in the kernel's source, must each be caught.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

import destiny_combat_cases as dc
import exact_ballpark.vec_combat as vc
import exact_ballpark.vec_combat_cuda as KC

CUDA = torch.cuda.is_available()
_REF: dict = {}


def references(seed: int, nb: int, nw: int) -> list:
    key = (seed, nb, nw)
    if key not in _REF:
        _REF[key] = [[dc.reference(cfg, w) for w in cfg["worlds"]] for cfg in dc.batches(seed, nb, nw)]
    return _REF[key]


def run(seed: int, nb: int, nw: int, path: str = "torch", device: str = "cpu", graph: bool = False,
        honour_flags: bool = True, batches: list | None = None) -> dict:
    """Every batch of a seed on the batched park: counts and the first differences."""
    refs = references(seed, nb, nw)
    out = Counter()
    examples = []
    for bi, cfg in enumerate(dc.batches(seed, nb, nw)):
        if batches is not None and bi not in batches:
            continue
        if graph and not cfg["do"]:
            continue
        keep = [wi for wi, w in enumerate(cfg["worlds"]) if path == "kernel" or dc.torch_domain(w)]
        if not keep:
            continue
        got, park = dc.batched({**cfg, "worlds": [cfg["worlds"][wi] for wi in keep]}, device, graph=graph, path=path)
        flags = park.world_flags()
        for k, wi in enumerate(keep):
            want = refs[bi][wi]
            if not dc.in_domain(want):
                out["outside_reference_domain"] += 1
                continue
            if honour_flags and flags[k]:
                out["refused"] += 1
                continue
            out["worlds"] += 1
            out["ball_steps"] += sum(len(v) for v in want["S"].values())
            out["events"] += len(want["E"])
            bad = dc.differences(want, got[k])
            if bad:
                out["differ"] += 1
                if len(examples) < 4:
                    examples.append((bi, wi, bad[:3]))
    out["examples"] = examples
    return out


def test_tensor_tick_matches_reference_development_cpu():
    r = run(1, 12, 12)
    assert r["worlds"] > 100 and r["events"] > 100, r
    assert r["differ"] == 0, r["examples"]


def test_tensor_tick_matches_reference_held_out_cpu():
    r = run(2, 12, 12)
    assert r["worlds"] > 100 and r["events"] > 100, r
    assert r["differ"] == 0, r["examples"]


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_kernels_match_reference_on_the_gpu(seed):
    r = run(seed, 12, 12, path="kernel", device="cuda")
    assert r["worlds"] > 100 and r["events"] > 100, r
    assert r["differ"] == 0, r["examples"]


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
def test_a_graph_matches_the_reference_on_the_gpu():
    r = run(1, 12, 12, path="kernel", device="cuda", graph=True)
    assert r["worlds"] > 30, r
    assert r["differ"] == 0, r["examples"]


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
def test_the_tensor_tick_on_the_gpu_matches_the_reference():
    r = run(2, 12, 12, path="torch", device="cuda")
    assert r["worlds"] > 100, r
    assert r["differ"] == 0, r["examples"]


def test_the_scenarios_reach_every_branch():
    """The reference's branch counters over the development scenarios: missiles straight and pursuing, on target,
    launched aimed, from rest and straight on, refused for other bubbles or cloaked targets; their contacts and
    self-collisions; releases by a warp, a cloak and a removal; formations in a slot and at a zero offset,
    leaving and shrinking (a slot past a smaller formation is stopped by SetBallFormation, so EvolveFormation's own
    no-slot branch needs a LoadFormations mid-run, not generated here); the yaw, pitch and roll's warp branch;
    dynamical orientation's disabled missiles snapped."""
    trace = Counter()
    for cfg in dc.batches(1, 12, 12):
        for w in cfg["worlds"]:
            dc.reference(cfg, w, trace)
    for name in ("missile.straight", "missile.pursue", "launch.aimed", "launch.stationary", "launch.straight",
                 "missile.cloaked_target", "missile.collision_event", "missile.self_collision",
                 "release.goto_direction", "release.stop", "formation.slot", "formation.zero_offset", "formation.leave",
                 "formation.shrink", "cloak.applied", "cloak.refused", "uncloak.massive", "orient.missile_snap",
                 "ypr.evolve", "ypr.snap", "remove.now"):
        assert trace[name] > 0, (name, sorted(trace.items())[:80])
    events = Counter()
    for refs in references(1, 12, 12):
        for want in refs:
            for t, m, c in want["E"]:
                events["self" if m == c else "target"] += 1
            assert want.get("Z", True)                       # each DoCollision's mLastC is +0
    assert events["self"] > 10 and events["target"] > 10, events


def test_the_literals_are_the_references():
    """Each literal the kernels take without a name in the reference is found in the reference's source as written."""
    from pathlib import Path
    root = Path(vc.__file__).parent
    for name, (_, _, source, text) in KC.COMBAT_LITERALS.items():
        assert text in (root / source).read_text(), (name, source, text)


def test_a_worlds_result_does_not_depend_on_its_batch():
    """Three worlds alone and the batch reversed give the rows of the batch as generated (the tensor tick)."""
    cfg = dc.batches(1, 2, 12)[0]
    ws = [w for w in cfg["worlds"] if dc.torch_domain(w)]
    together, _ = dc.batched({**cfg, "worlds": ws}, "cpu", path="torch")
    rev, _ = dc.batched({**cfg, "worlds": ws[::-1]}, "cpu", path="torch")
    for k in range(len(ws)):
        assert not dc.differences(together[k], rev[len(ws) - 1 - k])
    for k in range(3):
        alone, _ = dc.batched({**cfg, "worlds": [ws[k]]}, "cpu", path="torch")
        assert not dc.differences(together[k], alone[0])


def test_the_domain_is_refused_not_guessed():
    """Two massive balls outside MISSILE mode in a world are flagged (the park has no collision code for them), and
    check() raises; a defender missile is flagged too."""
    cfg = dc.batches(1, 1, 2)[0]
    rows, where = dc.rows_for(cfg)
    p = vc.VecCombatPark(len(rows), device="cpu")
    p.load(rows)
    p.massive = torch.where(p.exists, True, p.massive)
    p.watch()
    assert all(f & vc.FLAG_MASSIVE for f in p.world_flags())
    with pytest.raises(RuntimeError, match="two massive balls"):
        p.check()


# ---- planted faults on the tensor tick: each must change the development comparison ----

def _taper_always(orig):
    def f(self, target, taper):
        return orig(self, target, torch.ones_like(taper))
    return f


FAULTS = {
    "a missile straight on for one tick more": ("FU_MISSILE_STRAIGHT_MS", 1000),
    "the contact's early return dropped": ("SEPARATING_SPEED2", 1.0e300),
    "the self-collision blind to a cloaked owner": ("VecCombatPark._orphans", None),
    "no dispatch without a move of a kilometre": ("FU_DISPATCH_MIN_MOVE_SQ", 1.0e6),
    "the pursuit tapered": ("VecCombatPark._thrust_to", _taper_always),
    "the formation's offset not turned": ("m_rotation_q", lambda orig: (lambda q: orig((q[0] * 0.0, q[1] * 0.0,
                                                                                      q[2] * 0.0, q[3] * 0.0 + 1.0)))),
    "a released missile kept massive": ("VecCombatPark._release_followers", None),
    "Stop on the master keeps a formation follower's slot": ("VecCombatPark._free_slots",
                                                             lambda orig: (lambda self, rows: None)),
}


def _release_keeping_mass(self, rows):
    massive = self.massive.clone()
    _ORIG_RELEASE(self, rows)
    self.massive = massive


_ORIG_RELEASE = vc.VecCombatPark._release_followers
_ORIG_ORPHANS = vc.VecCombatPark._orphans


def _orphans_ignoring_cloak(self):
    cloaked = self.cloaked
    self.cloaked = torch.zeros_like(cloaked)
    try:
        return _ORIG_ORPHANS(self)
    finally:
        self.cloaked = cloaked


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    name, patch = FAULTS[fault]
    if name == "VecCombatPark._orphans":
        monkeypatch.setattr(vc.VecCombatPark, "_orphans", _orphans_ignoring_cloak)
    elif name == "VecCombatPark._release_followers":
        monkeypatch.setattr(vc.VecCombatPark, "_release_followers", _release_keeping_mass)
    elif name.startswith("VecCombatPark."):
        attr = name.split(".", 1)[1]
        monkeypatch.setattr(vc.VecCombatPark, attr, patch(getattr(vc.VecCombatPark, attr)))
    elif callable(patch):
        monkeypatch.setattr(vc, name, patch(getattr(vc, name)))
    else:
        monkeypatch.setattr(vc, name, patch)
    r = run(1, 12, 12, honour_flags=False)
    assert r["differ"] > 0, f"the planted fault {fault} went unnoticed"


# ---- planted faults in the kernel's source (a build of a changed copy): each must be caught on the GPU ----

KERNEL_FAULTS = {
    "the straight phase cut to the launch tick": ("<= M_MISSILE_STRAIGHT_MS", "<= 0"),
    "the contact's early return dropped": ("return inner;\n    return inner + 1;",
                                           "return inner + 1;\n    return inner + 1;"),
    "a released missile kept massive": ("if (missile && apply) b.massive = false;", "if (missile && apply) { }"),
    "a released formation follower's slot kept":
        ("atomicAnd((unsigned long long*)&c.formation_slots[t], ~(1ULL << slot));", "(void)slot;"),
}


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("fault", sorted(KERNEL_FAULTS))
def test_planted_kernel_faults_are_caught(fault, tmp_path, monkeypatch):
    old, new = KERNEL_FAULTS[fault]
    text = KC.SOURCE.read_text()
    assert text.count(old) == 1, (fault, text.count(old))
    src = tmp_path / KC.SOURCE.name
    src.write_text(text.replace(old, new))
    lib = KC.load_library(source=src)
    real = KC.tick
    monkeypatch.setattr(KC, "tick", lambda park, block=128, l=None: real(park, block, lib))
    r = run(1, 12, 12, path="kernel", device="cuda", honour_flags=False)
    if r["differ"] == 0:                                    # the in-tick release of formation followers: its own world
        cfg = _formation_warp_cfg(False)
        want = dc.reference(cfg, cfg["worlds"][0])
        got, _ = dc.batched(cfg, "cuda", path="kernel")
        r["differ"] += 1 if dc.differences(want, got[0]) else 0
    assert r["differ"] > 0, f"the planted kernel fault {fault} went unnoticed"


# ---- findings in Destiny's code (as the reference transcribes it, held to the compiled code), each pinned ----

def _one_world(cfg: dict, w: dict) -> dict:
    return {**cfg, "worlds": [w]}


def test_lane_finding_a_missile_can_post_two_collisions_with_its_target_in_one_tick():
    """Potential's overlap recursion (Ballpark.cpp:2789-2990) appends the target at each depth unless the two separate
    faster than 0.01 m/s, and a missile's response to its own target is never kept (2982): so a missile touching its
    target at a tick's start can post DoCollision with it twice in one tick (2 of the 3,556 missile-target ticks with
    an event in the generated worlds of seeds 1 and 2, both in seed 1). The batch posts the same."""
    cfg = dc.batches(1, 24, 24)[7]
    w = cfg["worlds"][21]
    want = dc.reference(cfg, w)
    assert want["E"].count((1002, 103, 3)) == 2, want["E"]
    got, _ = dc.batched(_one_world(cfg, w), "cpu", path="torch")
    assert not dc.differences(want, got[0])


def _two_ships_and_a_missile(do: bool = False):
    from exact_ballpark.full import FullBallpark
    p = FullBallpark(is_master=False, dynamical_orientation=do)
    p.add_ball(1, 1e7, 50.0, 200.0, True, False, True, True, False, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.5, 1.0)
    p.add_ball(2, 1e7, 50.0, 200.0, True, False, False, True, False, 5000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0)
    p.add_ball(101, 1e4, 3.0, 2500.0, True, False, False, False, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 1.0)
    rows = [{"id": 1, "mass": 1e7, "radius": 50.0, "max_vel": 200.0, "pos": [0.0, 0.0, 0.0], "vel": [10.0, 0.0, 0.0],
             "agility": 0.5, "speed_fraction": 1.0, "massive": True, "interactive": True},
            {"id": 2, "mass": 1e7, "radius": 50.0, "max_vel": 200.0, "pos": [5000.0, 0.0, 0.0],
             "vel": [0.0, 0.0, 0.0], "agility": 0.5, "speed_fraction": 1.0, "interactive": True},
            {"id": 101, "mass": 1e4, "radius": 3.0, "max_vel": 2500.0, "pos": [0.0, 0.0, 0.0],
             "vel": [0.0, 0.0, 0.0], "agility": 0.05, "speed_fraction": 1.0}]
    q = vc.VecCombatPark(3, device="cpu", dynamical_orientation=do)
    q.load(rows)
    return p, q


def _same_state(p, q, bid: int) -> None:
    b = p.balls[bid]
    r = int((q.ids == bid).nonzero()[0, 0])
    assert int(q.mode[r]) == b.mode and bool(q.massive[r]) == b.is_massive
    assert int(q.follow_id[r]) == b.follow_id and int(q.effect_stamp[r]) == b.effect_stamp
    assert q.vel[r].tolist() == list(b.vel) and q.goto[r].tolist() == list(b.goto)


def test_lane_finding_missile_follow_stops_the_missile_before_refusing_a_cloaked_target():
    """MissileFollow (Ballpark.cpp:3842-3855) stops the missile and only then refuses a target that is cloaked or in
    another bubble: a missile flying toward a point (GOTO) is left in STOP by a refused MissileFollow."""
    from exact_ballpark.ref import STOP
    p, q = _two_ships_and_a_missile()
    p.goto_direction(101, 1.0, 0.0, 0.0)
    q.goto_direction(q.ids == 101, torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float64))
    p.cloak_ball(2, 1)
    q.cloak(q.ids == 2, torch.full((3,), 1))
    p.missile_follow(101, 2, 1)
    q.missile_follow(q.ids == 101, torch.full((3,), 2), torch.full((3,), 1))
    assert p.balls[101].mode == STOP
    _same_state(p, q, 101)


def test_lane_finding_a_missile_launched_at_a_cloaked_target_coasts_massive_in_stop():
    """PyLaunchMissile (Thunkers.cpp:857-946) sets the missile's position, velocity and (if asked) massiveness, then
    returns for a cloaked target: the missile neither follows nor flies on, but coasts in its mode (STOP) as a massive
    ball, one that ships see and collide with, until the game removes it."""
    from exact_ballpark.ref import STOP
    p, q = _two_ships_and_a_missile()
    p.cloak_ball(2, 2)
    q.cloak(q.ids == 2, torch.full((3,), 2))
    p.launch_missile(101, 2, 1, 1, 1)
    q.launch_missile(q.ids == 101, torch.full((3,), 2), torch.full((3,), 1), torch.full((3,), 1),
                     torch.full((3,), 1))
    b = p.balls[101]
    assert b.mode == STOP and b.is_massive and b.vel[0] == 210.0, (b.mode, b.is_massive, b.vel)   # 10 + max(150, 200)
    _same_state(p, q, 101)


def _released_world() -> dict:
    """Ships 10 (the owner) and 11 (the target, flying along +x at its maximum speed by GotoDirection, so aligned for
    warp at once) and missile 3, whose id is below both: launched at 11 at step 1; 11 warps along +x at step 4 and
    starts warp proper at its own turn of that step's tick, after the missile's turn."""
    ship = lambda bid, pos, vel: {"id": bid, "mass": 2e7, "radius": 40.0, "max_vel": 300.0, "free": True,
                                  "massive": False, "interactive": True, "pos": pos, "vel": vel, "agility": 0.5,
                                  "speed_fraction": 1.0, "max_ang_vel": 0.8, "rot_agility": 0.6}
    missile = {"id": 3, "mass": 1e4, "radius": 3.0, "max_vel": 900.0, "free": True, "massive": False,
               "interactive": False, "pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "agility": 0.05,
               "speed_fraction": 1.0, "max_ang_vel": 2.0, "rot_agility": 0.5}
    return {"ships": [ship(10, [0.0, 0.0, 0.0], [0.0, 50.0, 0.0]), ship(11, [9000.0, 0.0, 0.0], [300.0, 0.0, 0.0])],
            "pool": [3], "start": 0,
            "records": [(0, "dir", 11, 1.0, 0.0, 0.0), (1, "add", missile), (1, "launch", 3, 11, 10, 0, 1),
                        (4, "warp", 11, 1e9, 0.0, 0.0, 0.0, 10)]}


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("do", [False, True], ids=["yaw_pitch_roll", "dynamical_disabled_for_missiles"])
def test_a_missile_released_after_its_own_turn(do):
    """A path the generated worlds do not reach (every missile id there is above every ship id): a missile whose
    target, with a higher id, starts warp proper at its turn is released after the missile's own turn (its MISSILE
    thrust kept for the tick, then GOTO and non-massive); with dynamical orientation disabled for missiles its phase 1
    skips the steering and its phase 3 applies the stored torque. The kernels against the reference, every step."""
    w = _released_world()
    cfg = {"do": do, "do_off": do, "master": False, "tick": 1000, "friction": 1.0e6, "new_orbit": False,
           "steps": 10, "worlds": [w]}
    trace = Counter()
    want = dc.reference(cfg, w, trace)
    assert dc.in_domain(want) and trace["release.goto_direction"] >= 1, (want["X"], want["B"], trace)
    modes = [want["S"][s][3]["mode"] for s in range(1, 11) if 3 in want["S"][s]]
    assert vc.MISSILE in modes and vc.GOTO in modes, modes
    got, _ = dc.batched(cfg, "cuda", path="kernel")
    assert not dc.differences(want, got[0])


def test_lane_finding_stop_frees_a_formation_followers_slot_on_the_master():
    """Destiny's Stop(Ball*) on the master frees a FORMATION follower's slot in its leader (Ballpark.cpp:4629-4633:
    FreeFormationSlot(mEffectStamp)); on a client it does not. The reference did not free it until this port found the difference by running these scenarios on the compiled code (full.py's FullBallpark._stop); the batch
    frees it too (VecCombatPark._free_slots, and in the kernels a follower released by its leader's warp)."""
    from exact_ballpark.full import FullBallpark
    for master in (True, False):
        p = FullBallpark(is_master=master)
        for bid, x in ((1, 0.0), (2, 600.0)):
            p.add_ball(bid, 1e7, 50.0, 200.0, True, False, False, True, False, x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0)
        p.load_formations(dc.FORMATIONS)
        p.set_ball_formation(1, 0)
        slot = p.reserve_formation_slot(1)
        p.formation_follow(2, 1, slot)
        assert p.balls[1].formation_slots == 1
        p.stop(2)
        assert p.balls[1].formation_slots == (0 if master else 1), (master, p.balls[1].formation_slots)
        rows = [{"id": 1, "mass": 1e7, "radius": 50.0, "max_vel": 200.0, "pos": [0.0, 0.0, 0.0],
                 "vel": [0.0, 0.0, 0.0], "agility": 0.5, "speed_fraction": 1.0, "interactive": True},
                {"id": 2, "mass": 1e7, "radius": 50.0, "max_vel": 200.0, "pos": [600.0, 0.0, 0.0],
                 "vel": [0.0, 0.0, 0.0], "agility": 0.5, "speed_fraction": 1.0, "interactive": True}]
        q = vc.VecCombatPark(2, device="cpu", master=master)
        q.load(rows)
        q.load_formations(dc.FORMATIONS)
        q.set_ball_formation(q.ids == 1, torch.zeros(2, dtype=torch.int64))
        s = q.reserve_formation_slot(0)
        q.formation_follow(q.ids == 2, torch.ones(2, dtype=torch.int64), torch.full((2,), s))
        q.stop(q.ids == 2)
        assert int(q.formation_slots[0]) == p.balls[1].formation_slots


def test_the_scenarios_are_destinys_own():
    """The development scenarios run directly on Destiny's own code compiled verbatim (the conformance kit's full harness) and on
    the reference, every row of every step equal (balls, boxes, bubbles, events, answers): the chain's first link on
    these very worlds."""
    import full_cases as fc
    if not fc.HARNESS.is_file():
        pytest.skip("Destiny full harness not built (conformance/build_full.py)")
    cs, cid = [], 0
    for cfg in dc.batches(1, 12, 12):
        for w in cfg["worlds"]:
            cid += 1
            cs.append(_harness_case(cfg, w, cid))
    rows, undefined = fc.run_reference(cs)
    want = fc.oracle([c for c in cs if c["id"] not in undefined])
    assert len(undefined) <= 2, undefined
    assert not fc.differences(want, rows, undefined)
    assert len(rows["S"]) > 10000 and len(rows["E"]) > 1000, {k: len(v) for k, v in rows.items()}


def _harness_case(cfg: dict, w: dict, cid: int) -> dict:
    """A world as the full harness takes it (full_cases' records): ReserveFormationSlot-then-FormationFollow as the
    query and a FormationFollow with the slot the reference answered; LaunchMissile's aimed and massive as its bits."""
    import full_cases as fc
    char = lambda n: ((int(n) + 128) & 0xFF) - 128
    answers = iter(dc.reference(cfg, w)["A"])

    def ball(b, step):
        return ("B", step, {"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
                            "free": b["free"], "global": False, "massive": b["massive"],
                            "interactive": b["interactive"], "junk": False, "pos": list(b["pos"]),
                            "vel": list(b["vel"]), "agility": b["agility"], "sf": b["speed_fraction"],
                            "max_ang_vel": b["max_ang_vel"], "rot_agility": b["rot_agility"]})
    recs = [ball(s, 0) for s in w["ships"]]
    flat = [float(len(dc.FORMATIONS))]
    for vectors in dc.FORMATIONS:
        flat.append(float(len(vectors)))
        for v in vectors:
            flat.extend(float(c) for c in v)
    k = fc._k
    recs.append(k(0, fc.FORMATIONS_OP, d=flat))
    simple = {"goto": fc.GOTO_OP, "dir": fc.DIR_OP, "vel": fc.VEL_OP}
    for rec in w["records"]:
        s, op, a = rec[0], rec[1], rec[2:]
        if op in simple:
            recs.append(k(s, simple[op], a[0], d=a[1:4]))
        elif op == "stop":
            recs.append(k(s, fc.STOP_OP, a[0]))
        elif op == "warp":
            recs.append(k(s, fc.WARP_OP, a[0], w=a[5], d=(*a[1:4], a[4])))
        elif op == "massive":
            recs.append(k(s, fc.MASSIVE_OP, a[0], w=a[1]))
        elif op == "add":
            recs.append(ball(a[0], s))
        elif op == "launch":
            recs.append(k(s, fc.LAUNCH_OP, a[0], a[1], a[2], w=(char(a[3]) == 1) | ((char(a[4]) != 0) << 1)))
        elif op == "mfollow":
            recs.append(k(s, fc.MISSILE_FOLLOW_OP, a[0], a[1], a[2]))
        elif op == "cloak":
            recs.append(k(s, fc.CLOAK_OP, a[0], w=a[1]))
        elif op == "uncloak":
            recs.append(k(s, fc.UNCLOAK_OP, a[0]))
        elif op == "remove":
            recs.append(k(s, fc.REMOVE_OP, a[0], w=0))
        elif op == "setform":
            recs.append(k(s, fc.SET_FORMATION_OP, a[0], w=a[1]))
        elif op == "formfollow":
            recs.append(k(s, fc.FORMATION_FOLLOW_OP, a[0], a[1], w=a[2]))
        elif op == "reserve_follow":
            slot = next(answers, -1)              # a world the reference refuses is not run on the harness
            recs.append(k(s, fc.RESERVE_SLOT_Q, a[1]))
            if slot >= 0:
                recs.append(k(s, fc.FORMATION_FOLLOW_OP, a[0], a[1], w=slot))
        elif op == "reserve":
            next(answers, -1)
            recs.append(k(s, fc.RESERVE_SLOT_Q, a[0]))
        elif op == "free":
            recs.append(k(s, fc.FREE_SLOT_OP, a[0], w=a[1]))
        else:
            raise ValueError(op)
    return {"id": cid, "kind": "combat1", "friction": cfg["friction"], "tick": cfg["tick"], "steps": cfg["steps"],
            "new_orbit": cfg["new_orbit"], "iterative": False, "do": cfg["do"], "do_missiles_off": cfg["do_off"],
            "manual": False, "master": cfg["master"], "seed": 1, "start": w["start"], "records": recs}


def _formation_warp_cfg(do: bool) -> dict:
    """A formation leader (2) flying along +x at its maximum speed, its followers 1 and 3 in slots 0 and 1, on
    the master; the leader warps along +x at step 3 (aligned at once) and its slots are asked for at step 5."""
    ship = lambda bid, pos, vel: {"id": bid, "mass": 2e7, "radius": 40.0, "max_vel": 300.0, "free": True,
                                  "massive": False, "interactive": True, "pos": pos, "vel": vel, "agility": 0.5,
                                  "speed_fraction": 1.0, "max_ang_vel": 0.8, "rot_agility": 0.6}
    w = {"ships": [ship(2, [9000.0, 0.0, 0.0], [300.0, 0.0, 0.0]), ship(1, [0.0, 600.0, 0.0], [0.0, 0.0, 0.0]),
                   ship(3, [0.0, -600.0, 0.0], [0.0, 0.0, 0.0])],
         "pool": [], "start": 0,
         "records": [(0, "dir", 2, 1.0, 0.0, 0.0), (0, "setform", 2, 0), (0, "reserve_follow", 1, 2),
                     (0, "reserve_follow", 3, 2), (3, "warp", 2, 1e9, 0.0, 0.0, 0.0, 10), (5, "reserve", 2)]}
    return {"do": do, "do_off": False, "master": True, "tick": 1000, "friction": 1.0e6, "new_orbit": False, "steps": 6,
            "worlds": [w]}


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("do", [False, True], ids=["yaw_pitch_roll", "dynamical"])
def test_a_formation_leaders_warp_frees_its_followers_slots_on_the_master(do):
    """Inside the tick: a formation leader starting warp proper on the master releases its FORMATION followers, and
    each Stop frees the follower's slot in the leader (the kernels' integer atomic AND), one follower before the
    leader's turn and one after it by id. The kernels against the reference, every step (the leader's bubble moves
    with its warp; nothing compared here reads bubbles)."""
    cfg = _formation_warp_cfg(do)
    w = cfg["worlds"][0]
    trace = Counter()
    want = dc.reference(cfg, w, trace)
    assert want["X"] is None and trace["release.stop"] >= 2, (want["X"], trace)
    assert want["A"] == [0, 1, 0], want["A"]                 # both slots taken, then both freed by the release
    got, _ = dc.batched(cfg, "cuda", path="kernel")
    assert not dc.differences(want, got[0])

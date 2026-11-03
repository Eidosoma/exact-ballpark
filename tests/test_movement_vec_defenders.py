# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Defender missiles (a missile whose owner id is negative, the GPU parity report's G9) in the composed
park (exact_ballpark.vec_composed.VecComposedPark with collisions: GetNearbyBalls with missiles kept, in the partition's
walk order; Gradient's skips on the signed owner id, so a defender meets its launcher; COL-711's exception for its own
target, every other response kept; its collision list dispatched), and the owner row of the kernel path
(a released missile's, cleared where the torch path's Stop clears it: vec_combat_cuda.cu's release_in_phase1).

On the CPU: the torch path against the scalar reference (FullBallpark, held to Destiny's compiled code by
tests/test_movement_full_oracle.py), every field of every ball at every step, mLastC, the collision lists, the
filings, the DoCollision events and the answers, on tests/defender_cases.py's development seed 21 and held-out seed 23
(first compared after the generator was frozen; seed 22, which the generator's docstring names, ran once before its
freeze record and is reported apart); the scenarios reach the defenders' branches; a sample of them run on
Destiny's own code (the conformance kit's full harness); planted faults, one rule each, are caught; with collisions off a defender
is refused (FLAG_DEFENDER), as VecCombatPark refuses it; this port's finding (a defender meets its launcher) pinned.

On the card: the kernel path against the torch path, every tensor every tick (both seeds), a CUDA graph, the kernel
path against FullBallpark, three planted faults in the composed candidates; the kernel path (collisions off)
against the torch path every tensor every tick (the collision lists included: the kernels count a missile's
contacts and the park makes the lists from the counts), its owner row included, and the old kernel
(the fix taken out) caught on the same worlds.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

import defender_cases as dfc
import destiny_combat_cases as dc
import park_cases as pc
import test_movement_vec_composed as tc
from exact_ballpark import vec_combat as vc
from exact_ballpark import vec_composed as vp
from exact_ballpark.modes import MISSILE

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device visible")
NB, NW = 6, 16                        # the first six batches of a seed in the CPU tests


def run(seed: int, device: str = "cpu", park_cls=None, batches=None, **kw) -> Counter:
    """The first NB batches of a defender seed on the composed park against the reference (the `run`)."""
    return tc.run(seed, device, park_cls, nb=NB, nw=NW, batches=batches, gen=dfc, **kw)


@pytest.mark.parametrize("seed", [21, 23], ids=["development", "held_out"])
def test_defenders_match_fullballpark_on_the_cpu(seed):
    """The torch path against the reference: no difference, no world refused (defenders are held with collisions)."""
    out = run(seed)
    tc._clean(out, 85)
    assert out["refused"] == 0 and out["events"] > 500, dict(out)


def test_the_defender_scenarios_reach_every_branch():
    """The development worlds put defenders into contact with their launcher, their target, other missiles and more
    than one ball a tick, dispatch their events (with the master's self-collision by |owner|), and keep a defender's
    response to every ball but its target (the reference's own counters and rows)."""
    trace, rows = Counter(), Counter()
    for cfg in dfc.batches(21, NB, NW):
        for w in cfg["worlds"]:
            t = Counter()
            want = dfc.reference(cfg, w, trace=t)
            if not dc.in_domain(want):
                continue
            trace.update(t)
            S = want["S"]
            for s in range(1, cfg["steps"] + 1):
                for bid, row in S[s].items():
                    if bid not in dfc.DEFENDERS or row["mode"] != MISSILE or row["owner_id"] >= 0:
                        continue
                    hits = row["collisions"]
                    rows["defender_steps"] += 1
                    if not hits:
                        continue
                    rows["contact"] += 1
                    rows["launcher"] += -row["owner_id"] in hits
                    rows["target"] += row["follow_id"] in hits
                    rows["missile"] += any(h in dfc.POOL or h == dfc.EXTRA_MISSILE or h in dfc.DEFENDERS for h in hits)
                    rows["two_or_more"] += len(set(hits)) > 1
                    rows["kept"] += any(c != 0.0 for c in row["last_c"])
            for (_, mid, col) in want["E"]:
                if mid in dfc.DEFENDERS:
                    rows["events"] += 1
                    rows["self_events"] += col == mid
    floors = dict(contact=300, launcher=30, target=200, missile=20, two_or_more=60, kept=100, events=400,
                  self_events=5)
    for k, v in floors.items():
        assert rows[k] >= v, (k, rows[k], dict(rows))
    for k in ("candidates.defender", "candidates.missile_target", "gradient.skip", "potential.coincident",
              "potential.recurse", "potential.earlier_kept", "missile.self_collision", "launch.aimed",
              "launch.straight"):
        assert trace[k] >= 3, (k, trace[k])


def test_the_defender_scenarios_are_destinys_own():
    """The first four development batches (64 worlds) run directly on Destiny's own code compiled verbatim (the conformance kit's full harness, the adapter) and on the reference: every row of every step equal (balls, boxes,
    bubbles, events, answers)."""
    import full_cases as fc
    if not fc.HARNESS.is_file():
        pytest.skip("Destiny full harness not built (conformance/build_full.py)")
    cs, cid = [], 0
    for cfg in dfc.batches(21, 4, NW):
        for w in cfg["worlds"]:
            cid += 1
            cs.append(tc._harness_case(cfg, w, cid))
    rows, undefined = fc.run_reference(cs)
    want = fc.oracle([c for c in cs if c["id"] not in undefined])
    assert not undefined, undefined
    assert not fc.differences(want, rows, undefined)
    assert len(rows["S"]) > 5000 and len(rows["E"]) > 1000, {k: len(v) for k, v in rows.items()}


# ---- this port's finding, pinned: Gradient's owner skip compares the signed id ----

def launcher_world(owner: int, master: bool = False) -> dict:
    """Missile 201 (massive, aimed) launched by ship 1 (massive, radius 100 m, at rest) at ship 2 (massive, 8 km away)
    with the owner `owner`: -1 a defender, 1 a normal missile."""
    common = {"free": True, "massive": True, "agility": 1.0, "speed_fraction": 1.0, "max_ang_vel": 1.0,
              "rot_agility": 0.9, "vel": [0.0, 0.0, 0.0], "max_vel": 150.0, "mass": 1e8}
    centre = [2.0e6, 2.0e6, 2.0e6] if master else [1000.0, -2000.0, 500.0]
    ships = [{**common, "id": 1, "radius": 100.0, "interactive": True, "pos": list(centre)},
             {**common, "id": 2, "radius": 60.0, "interactive": False,
              "pos": [centre[0] + 8000.0, centre[1], centre[2]]}]
    m = {"id": 201, "mass": 1e4, "radius": 3.0, "max_vel": 1500.0, "free": True, "massive": False,
         "interactive": False, "pos": [centre[0], centre[1] + 500.0, centre[2]], "vel": [0.0, 0.0, 0.0],
         "agility": 0.1, "speed_fraction": 1.0, "max_ang_vel": 1.0, "rot_agility": 0.5}
    w = {"ships": ships, "pool": [201], "records": [(1, "add", m), (1, "launch", 201, 2, owner, 1, 1)], "start": 50}
    return {"do": False, "do_off": False, "master": master, "tick": 1000, "friction": 1.0e6, "new_orbit": False,
            "steps": 4, "worlds": [w]}


@pytest.mark.parametrize("master", [False, True], ids=["client", "master"])
def test_lane_finding_a_defender_meets_its_own_launcher(master):
    """Destiny's Gradient skips a missile's candidate whose id equals the missile's mOwnerID (Ballpark.cpp:2745), the
    signed id, while LaunchMissile finds the launcher by |owner| and places the missile at its centre: a massive
    defender launched from a massive launcher meets it (DoCollision with the launcher's id), a normal missile does
    not. The reference shows it (held to the compiled code by test_the_defender_scenarios_are_destinys_own and
    tests/test_movement_full_oracle.py), and the composed park equals the reference on both worlds."""
    events = {}
    for owner in (-1, 1):
        cfg = launcher_world(owner, master)
        want = dfc.reference(cfg, cfg["worlds"][0])
        assert dc.in_domain(want), (want["X"], want["B"])
        events[owner] = want["E"]
        got, park = pc.batched(cfg, "cpu")
        assert park.world_flags() == [0]
        assert dc.differences(want, got[0]) == []
    assert any(col == 1 for (_, mid, col) in events[-1] if mid == 201), events[-1]
    assert not any(col == 1 for (_, mid, col) in events[1] if mid == 201), events[1]


def test_with_collisions_off_a_defender_is_refused():
    """With collisions off the park has no partition, so a defender's candidates are not computed: the defender is
    flagged (FLAG_DEFENDER) and check() raises, in VecComposedPark as in VecCombatPark; with collisions it is held."""
    cfg = launcher_world(-1)
    _, off = pc.batched(cfg, "cpu", collisions=False)
    _, combat = dc.batched(cfg, "cpu", path="torch")
    _, on = pc.batched(cfg, "cpu")
    for p in (off, combat):
        assert p.world_flags()[0] & vc.FLAG_DEFENDER
        with pytest.raises(RuntimeError, match="defender"):
            p.check()
    assert on.world_flags() == [0]
    on.check()


# ---- planted faults on the torch path: one rule each, every one must change the comparison ----

class _AsNormal(vp.VecComposedPark):
    """Defenders taken for normal missiles: their one candidate their target, first in order."""
    @staticmethod
    def _defenders(missile, owner):
        return torch.zeros_like(missile)


class _NoMissilesSeen(vp.VecComposedPark):
    """A defender's candidates collisionCase1's: missiles left out."""
    @staticmethod
    def _defender_candidates(nearby, cand):
        return cand


class _TargetFirst(vp.VecComposedPark):
    """A defender's target first in its order (key 0), as a normal missile's."""
    @staticmethod
    def _target_first(missile, defender):
        return missile


class _KeptView(vp._GradientView):
    def _exempt(self, other):
        R = self.R
        return self.missile[:, :R] & (self.owner_v[:, :R] >= 0) & (other == self.follow_v[:, :R])


class _KeepsTarget(vp.VecComposedPark):
    """COL-711's exception for normal missiles only: a defender keeps its response to its own target."""
    view_cls = _KeptView


class _UnsignedOwnerSkip(vp.VecComposedPark):
    """Gradient's owner skip on |owner|: a defender skips its launcher."""
    def _pair_mask(self, responder_v, dropped_v):
        pair = super()._pair_mask(responder_v, dropped_v)
        v = self.view
        own = v.owner_v
        launcher = v.id[:, None, :] == -own[:, :, None]
        return pair & ~(((own < 0) & v.missile)[:, :, None] & launcher)


class _EventsOfTarget(vp.VecComposedPark):
    """The events as VecCombatPark posts them: the target's id ev_n times."""
    def events(self):
        return vc.VecCombatPark.events(self)


FAULTS = {"as_normal_missiles": _AsNormal, "no_missiles_seen": _NoMissilesSeen, "target_first": _TargetFirst,
          "keeps_its_targets_response": _KeepsTarget, "unsigned_owner_skip": _UnsignedOwnerSkip,
          "events_of_the_target": _EventsOfTarget}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault):
    out = run(21, park_cls=FAULTS[fault], batches=range(2))
    assert out["differ"] > 0, (fault, dict(out))


# ---- on the card: the kernel path against the torch path, every tensor every tick ----

@CUDA
@pytest.mark.parametrize("seed", [21, 23], ids=["development", "held_out"])
def test_the_kernel_equals_the_torch_path_with_defenders(seed):
    """The first three batches of each seed (48 worlds; both orientations), every state tensor of the composed park
    after every tick (the lockstep): no bit differs, the events and the answers are the same."""
    import test_movement_vec_composed_cuda as tk
    out, ex = tk.run_batches(dfc.batches(seed, 3, NW))
    tk._clean(out, ex)
    assert out["worlds"] == 48 and out["contacts"] > 300 and out["events"] > 200, dict(out)


@CUDA
def test_a_cuda_graph_with_defenders():
    """With dynamical orientation (batches 1 and 3 of the development seed): the tick captured once as a CUDA graph,
    against the torch path and against eager kernel ticks, every tensor every tick."""
    import test_movement_vec_composed_cuda as tk
    cfgs = [c for c in dfc.batches(21, 4, NW) if c["do"]]
    assert len(cfgs) == 2
    out, ex = tk.run_batches(cfgs, graph=True)
    tk._clean(out, ex)
    assert out["ticks"] > 20 and out["events"] > 100, dict(out)
    out, ex = tk.run_batches(cfgs, graph=True, a_kernel=True)
    tk._clean(out, ex)


@CUDA
def test_the_kernel_matches_fullballpark_with_defenders():
    """The kernel path against the scalar reference (the comparison), two batches of each seed."""
    import test_movement_vec_composed_cuda as tk
    for seed in (21, 23):
        out = tc.run(seed, "cuda", tk._KernelPark, nb=2, nw=NW, gen=dfc)
        tc._clean(out, 28)


# the composed candidates (vec_collide_cuda.cu under XBPC_COMPOSED): (old text, new text), each found once
KERNEL_FAULTS = {
    "defenders_see_no_missiles": ("(me_defender || !P.missile[j])", "!P.missile[j]"),
    "defenders_as_normal_missiles": ("const bool me_defender = me_missile && P.owner[j0] < 0;",
                                     "const bool me_defender = false;"),
    "defenders_target_first": ("const bool key0 = me_missile && P.owner[j0] >= 0;", "const bool key0 = me_missile;"),
}


@CUDA
@pytest.mark.parametrize("fault", sorted(KERNEL_FAULTS))
def test_planted_kernel_faults_are_caught(fault, tmp_path):
    import test_movement_vec_composed_cuda as tk
    from exact_ballpark import vec_collide_cuda as KD
    from exact_ballpark import vec_composed_cuda as KP
    old, new = KERNEL_FAULTS[fault]
    text = KD.SOURCE.read_text()
    assert text.count(old) == 1, (fault, text.count(old))
    src = tmp_path / KD.SOURCE.name
    src.write_text(text.replace(old, new))
    libs = KP.libraries(collide_lib=KD.load_library("native", composed=True, source=src))
    out, ex = tk.run_batches(dfc.batches(21, 2, NW), libs=libs)
    assert out["differ"] + out["events_differ"] > 0, (fault, dict(out))


# ---- on the card: the kernel path (collisions off) holds the torch path's owner row ----

class _Off(vp.VecComposedPark):
    """VecComposedPark with collisions off: its kernel path is VecCombatPark's four launches (the kernels)."""
    def __init__(self, *a, **kw):
        super().__init__(*a, collisions=False, **kw)


def released_world(missile_id: int) -> dict:
    """The world (tests/test_movement_vec_combat.py's _released_world): ships 10 (the owner) and 11 (the
    target, aligned for warp at once), a missile launched at 11 at step 1; 11 warps at step 4 and starts warp proper at
    its turn, which comes after the missile's (id 3) or before it (id 30)."""
    import test_movement_vec_combat as tcb
    w = tcb._released_world()
    m = dict(next(r for r in w["records"] if r[1] == "add")[2], id=missile_id)
    recs = [(0, "dir", 11, 1.0, 0.0, 0.0), (1, "add", m), (1, "launch", missile_id, 11, 10, 0, 1),
            (4, "warp", 11, 1e9, 0.0, 0.0, 0.0, 10)]
    return {**w, "pool": [missile_id], "records": recs}


def owner_cfgs() -> list:
    out = []
    for mid in (3, 30):
        for do in (False, True):
            out.append({"do": do, "do_off": False, "master": False, "tick": 1000, "friction": 1.0e6,
                        "new_orbit": False, "steps": 8, "worlds": [released_world(mid)]})
    return out + dc.batches(1, 6, 24)[:2]


def _lockstep_off(cfgs: list, monkeypatch, lib=None) -> tuple:
    """the kernel path (collisions off) against the torch path, every state tensor every tick, the collision lists
    included: with collisions off the torch path keeps the lists (the target's id once a contact, then the master's
    self-collision appended, as FullBallpark's list); the kernels count the contacts (hits_n) and the
    self-collision (ev_self), and the park's `evolve` makes the lists from those counts after the kernels (earlier,
    `hits` was left out here and the relation torch hits_n ==
    kernel hits_n + kernel ev_self checked in place of `hits_n`). `monkeypatch` is no longer used."""
    import test_movement_vec_composed_cuda as tk
    out, ex = Counter(), []
    for cfg in cfgs:
        o, e, A, B = tk.lockstep(cfg, kernel_cls=_Off, torch_cls=_Off, libs=lib)
        out.update(o)
        ex.extend(e[:3])
    return out, ex


@CUDA
def test_the_combat_kernel_path_holds_the_torch_paths_owner_row(monkeypatch):
    """A missile released in phase 1 (its target starts warp proper; before and after its own turn, with and without
    dynamical orientation) and two batches of the worlds: every state tensor of the collisions-off park after
    every tick, the owner row included, equal between the kernels and the torch path."""
    import test_movement_vec_composed_cuda as tk
    out, ex = _lockstep_off(owner_cfgs(), monkeypatch)
    tk._clean(out, ex)
    assert out["worlds"] == 52 and out["events"] > 50 and out["contacts"] > 50, dict(out)


@CUDA
def test_the_old_combat_kernel_kept_a_stale_owner_row(tmp_path, monkeypatch):
    """The fix taken out of the kernel (a build of a changed copy): the owner row of a released missile differs
    from the torch path's, and nothing else does."""
    from exact_ballpark import vec_combat_cuda as KC
    old = "    if (b.follow_ptr < 0) c.owner_ptr[i] = -1;\n"
    text = KC.SOURCE.read_text()
    assert text.count(old) == 1
    src = tmp_path / KC.SOURCE.name
    src.write_text(text.replace(old, ""))
    out, ex = _lockstep_off(owner_cfgs(), monkeypatch, lib=KC.load_library(source=src))
    differ = {k for k in out if k.startswith("differ:")}
    assert differ == {"differ:owner_ptr"}, (dict(out), ex[:4])

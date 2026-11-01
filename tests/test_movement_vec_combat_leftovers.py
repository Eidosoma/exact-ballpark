# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The pieces of the combat park (exact_ballpark.vec_combat.VecCombatPark): the commands
(SetBallFree, SetBallRigid, the Python SetBoid, SetBallTroll, SetBallInteractive) in its tick, with StopAllFollowers'
members, and mushrooms (AddMushroom, the growth by glibc's powf, the removal in a tick with its members released before
or after their own turn, BringOutTheDeads), against the scalar reference (exact_ballpark.full.FullBallpark), bit for bit:
every field the batch holds of every ball at every step (its free, interactive and moribund flags among them), which
balls exist, the ReserveFormationSlot answers and the DoCollision events, on generated scenarios
(tests/combat_leftover_cases.py). On the CPU the tensor tick (the worlds it holds: no warp, follow or orbit); with a CUDA
device the five launches for every world, a CUDA graph, the tensor tick on the GPU, and the yaw, pitch and roll on the
device (the `ypr="cuda"`) on these scenarios and on the (tests/destiny_combat_cases.py). The same
scenarios run on Destiny's own code compiled verbatim (the conformance kit's full harness), every row equal to the reference's,
so this is exact against Destiny. Development scenarios are seed 1, held-out seeds 2 and 3 (first compared after the
generator was frozen). Worlds the batch refuses (its domain flags) are counted, never compared as equal. Planted faults,
on the tensor tick and in the kernel's source, must each be caught.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest
import torch

import combat_leftover_cases as cl
import destiny_combat_cases as dc
import exact_ballpark.vec_combat as vc
import exact_ballpark.vec_combat_cuda as KC
from exact_ballpark.modes import BOID, FOLLOW, FORMATION, MISSILE, MUSHROOM, ORBIT, RIGID, STOP, TROLL, WARP

CUDA = torch.cuda.is_available()
_REF: dict = {}


def batches(gen, seed: int, nb: int, nw: int, reload: bool = False) -> list:
    out = gen.batches(seed, nb, nw)
    return [cl.with_reloads(cfg) for cfg in out] if reload else out


def references(gen, seed: int, nb: int, nw: int, reload: bool = False) -> list:
    key = (gen.__name__, seed, nb, nw, reload)
    if key not in _REF:
        _REF[key] = [[cl.reference(cfg, w) for w in cfg["worlds"]] for cfg in batches(gen, seed, nb, nw, reload)]
    return _REF[key]


def run(seed: int, nb: int = 12, nw: int = 12, path: str = "torch", device: str = "cpu", graph: bool = False,
        honour_flags: bool = True, park_cls=None, gen=cl, only=None, reload: bool = False) -> Counter:
    """Every batch of a seed on the batched park: counts and the first differences (`only`: a predicate on cfg;
    `reload`: every world with cl.with_reloads' mid-run LoadFormations)."""
    refs = references(gen, seed, nb, nw, reload)
    out = Counter()
    examples = []
    for bi, cfg in enumerate(batches(gen, seed, nb, nw, reload)):
        if (graph and not cfg["do"]) or (only is not None and not only(cfg)):
            continue
        keep = [wi for wi, w in enumerate(cfg["worlds"]) if path == "kernel" or cl.torch_domain(w)]
        if not keep:
            continue
        got, park = cl.batched({**cfg, "worlds": [cfg["worlds"][wi] for wi in keep]}, device, graph=graph, path=path,
                               park_cls=park_cls)
        flags = park.world_flags()
        for k, wi in enumerate(keep):
            want = refs[bi][wi]
            if not cl.in_domain(want):
                out["outside_reference_domain"] += 1
                continue
            if honour_flags and flags[k]:
                out["refused"] += 1
                continue
            out["worlds"] += 1
            out["ball_steps"] += sum(len(v) for v in want["S"].values())
            out["events"] += len(want["E"])
            bad = cl.differences(want, got[k])
            if bad:
                out["differ"] += 1
                if len(examples) < 4:
                    examples.append((bi, wi, bad[:3]))
    out["examples"] = examples
    return out


def test_tensor_tick_matches_reference_development_cpu():
    r = run(1)
    assert r["worlds"] > 60 and r["ball_steps"] > 5000 and r["events"] > 80, r
    assert r["differ"] == 0, r["examples"]


# ---- what the scenarios reach (the reference's own counts) ----

_MEMBERS: list = []


class _Recording(cl.FullBallpark):
    """The reference, recording before each tick each mushroom's members by StopAllFollowers' own test (follow_ptr and
    follow id name it), and after the tick which mushrooms went."""

    def evolve(self, timestamp: int = 0) -> None:
        before = {b.id: [(f.id, f.mode) for f in self.balls.values()
                         if f.id != b.id and f.follow_ptr == b.id and f.follow_id == b.id]
                  for b in self.balls.values() if b.mode == MUSHROOM and not b.is_moribund}
        super().evolve(timestamp)
        _MEMBERS.extend((mid, m) for mid, m in before.items() if self.balls[mid].is_moribund)


def test_the_scenarios_reach_every_branch(monkeypatch):
    """The development scenarios reach what the batch adds: mushrooms grown and removed in a tick, with members whose
    turns come after the mushroom's (missiles, formation followers, followers, orbiters, members SetBoid took out of a
    follow mode, trolls among them) and before it; refused AddMushrooms; trolls petrified; SetBallFree both ways."""
    from collections import Counter as C
    monkeypatch.setattr(cl, "FullBallpark", _Recording)
    trace, seen = C(), C()
    for cfg in cl.batches(1, 12, 12):
        for w in cfg["worlds"]:
            _MEMBERS.clear()
            want = cl.reference(cfg, w, trace)
            for mid, members in _MEMBERS:
                seen["removed"] += 1
                for fid, mode in members:
                    seen[(mode, fid > mid)] += 1
            for rec in w["records"]:
                seen["record " + rec[1]] += 1
            S = want["S"]
            for s in range(1, cfg["steps"] + 1):
                for bid, row in S[s].items():
                    prev = S[s - 1].get(bid)
                    if prev and prev["mode"] == TROLL and row["mode"] == RIGID:
                        seen["petrified"] += 1
                    if prev and prev["free"] and not row["free"] and prev["mode"] not in (TROLL, MUSHROOM):
                        seen["fixed"] += 1
                    if prev and not prev["free"] and row["free"]:
                        seen["freed"] += 1
    assert trace["mushroom.grow"] >= 200 and trace["mushroom.done"] >= 30 and seen["removed"] >= 30, (trace, seen)
    for mode in (MISSILE, FORMATION, FOLLOW, ORBIT, BOID, TROLL):
        assert seen[(mode, True)] >= 2, (mode, seen)              # members whose turns came after the mushroom's
    assert sum(v for k, v in seen.items() if isinstance(k, tuple) and not k[1]) >= 6, seen   # ... and before it
    assert seen[(MISSILE, False)] >= 2, seen
    assert seen["petrified"] >= 20 and seen["fixed"] >= 20 and seen["freed"] >= 20, seen
    for op in ("setfree", "rigid", "boid", "troll", "inter", "mushroom", "deads", "follow", "orbit"):
        assert seen["record " + op] >= 5, (op, seen)


@pytest.mark.parametrize("reload", [False, True], ids=["as_generated", "loadformations_mid_run"])
def test_the_scenarios_are_destinys_own(reload):
    """The development scenarios on Destiny's own code compiled verbatim (the conformance kit's full harness) and on the
    reference, every row of every kind equal (balls, partition, bubbles, events, answers): the chain's first link on
    these very worlds; and the same with a LoadFormations mid-run in every world (cl.with_reloads)."""
    import full_cases as fc
    if not fc.HARNESS.is_file():
        pytest.skip("Destiny full harness not built (conformance/build_full.py)")
    cs, cid = [], 0
    for cfg in batches(cl, 1, 12, 12, reload):
        for w in cfg["worlds"]:
            cid += 1
            cs.append(cl.harness_case(cfg, w, cid, cl.reference(cfg, w)["A"]))
    rows, undefined = fc.run_reference(cs)
    want = fc.oracle([c for c in cs if c["id"] not in undefined])
    assert not undefined, undefined
    assert not fc.differences(want, rows, undefined)
    assert len(rows["S"]) > 10000 and len(rows["E"]) > 1000, {k: len(v) for k, v in rows.items()}


def test_a_mid_run_load_formations_cpu():
    """PyLoadFormations mid-run (Thunkers.cpp:1921: the table cleared and loaded again; the park's own table, every
    world at once): a smaller table, then the first again, on the tensor tick."""
    r = run(1, reload=True)
    assert r["worlds"] > 60, r
    assert r["differ"] == 0, r["examples"]


# ---- the commands where a tick lacks them, AddMushroom's ids and refusals ----

def test_a_park_with_another_tick_refuses_the_commands_and_mushrooms():
    """A subclass of VecCombatPark with a tick of its own refuses the commands and AddMushroom (VecCombatPark._g11);
    the composed park batches them (tests/test_movement_vec_composed_leftovers.py), which
    changed this test from the composed park's refusal."""
    class Own(vc.VecCombatPark):
        def evolve_torch(self) -> None:
            super().evolve_torch()

    p = Own(2, device="cpu")
    m = torch.ones(2, dtype=torch.bool)
    one = torch.ones(2, dtype=torch.int64)
    for call in (lambda: p.set_free(m, m), lambda: p.set_ball_rigid(m), lambda: p.set_boid(m, m),
                 lambda: p.set_ball_troll(m, one), lambda: p.add_mushroom(m, one, one.double(), one.double())):
        with pytest.raises(NotImplementedError):
            call()


def _one_world(master: bool, do: bool, records: list, steps: int = 4) -> tuple:
    w = {"ships": [dc._ship(random.Random(3), k, [0.0, 0.0, 0.0], interactive=True) for k in (1, 2)],
         "pool": list(cl.POOL), "records": records, "start": 7, "kind": "mush"}
    cfg = {"do": do, "do_off": False, "master": master, "tick": 1000, "friction": 1.0e6, "new_orbit": False,
           "steps": steps, "worlds": [w]}
    want = cl.reference(cfg, w)
    got, park = cl.batched(cfg, "cpu", path="torch")
    return want, got[0], park


@pytest.mark.parametrize("master,do,mid,mid2", [(False, False, 0, 0), (False, True, 0, 0), (True, False, -2, -3),
                                                (True, True, -3, -5)])
def test_a_mushrooms_id_and_its_refusals(master, do, mid, mid2):
    """AddMushroom's ball takes id 0 on a client and the master's next local id (two taken with dynamical
    orientation); a range or lifetime not above 0 or a missing owner adds nothing (no id taken); grown (here at the
    fourth tick) it is removed, and after BringOutTheDeads a second mushroom is a new ball. A second mushroom while
    the first is in the park is refused by the batch, not guessed: on a client AddBall's "not created" path
    (FLAG_POOL), on the master two massive balls (FLAG_MASSIVE)."""
    recs = [(0, "mushroom", 1, 0.0, 3.0), (0, "mushroom", 1, 100.0, -1.0), (0, "mushroom", 99, 100.0, 3.0),
            (0, "mushroom", 2, 120.0, 2.5), (1, "setfree", mid, 1)]
    want, got, park = _one_world(master, do, recs + [(5, "deads"), (5, "mushroom", 1, 50.0, 1.0)], steps=6)
    assert mid in got["S"][3] and got["S"][4][mid]["moribund"], sorted(got["S"][4])
    assert mid2 in got["S"][5] and got["S"][5][mid2]["mode"] == MUSHROOM and not got["S"][5][mid2]["moribund"]
    assert got["S"][5][mid2]["effect_stamp"] > got["S"][4][mid]["effect_stamp"]        # a new ball
    assert mid == mid2 or mid not in got["S"][5]
    assert not park.world_flags()[0] and not cl.differences(want, got)
    _, _, park = _one_world(master, do, recs + [(2, "mushroom", 1, 50.0, 1.0)])
    assert park.world_flags()[0] & (vc.FLAG_MASSIVE if master else vc.FLAG_POOL)


# ---- planted faults on the tensor tick (the CPU) ----

class _PetrifyWithoutSnap(vc.VecCombatPark):
    def _after_phase4(self, later):
        self._release_members(later)
        due = self.free & ~self.moribund & self.exists & (self.mode == TROLL) & ~(self.effect_stamp > self.clock - 1)
        self.petrified = due.to(torch.uint8)
        self.vel = torch.where(due[:, None], 0.0, self.vel)
        self.old_vel = torch.where(due[:, None], 0.0, self.old_vel)
        self.last_g = torch.where(due[:, None], 0.0, self.last_g)
        self.free = self.free & ~due
        self.interactive = self.interactive & ~due
        self.mode = torch.where(due, RIGID, self.mode)


class _MembersByFollowModes(vc.VecCombatPark):
    def _members(self, rows):
        return super()._members(rows) & self._following()


class _NoBringOut(vc.VecCombatPark):
    def bring_out_the_deads(self, mask=None):
        return None


TORCH_FAULTS = {"PetrifyTroll without SetBallVelocity's snap": _PetrifyWithoutSnap,
                "the follower sets by the follow modes (the modes' rule)": _MembersByFollowModes,
                "BringOutTheDeads leaving the removed mushroom": _NoBringOut,
                "the growth's exponent nudged": "powf"}


@pytest.mark.parametrize("fault", sorted(TORCH_FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    cls = TORCH_FAULTS[fault]
    if cls == "powf":
        real = vc._C_POWF
        monkeypatch.setattr(vc, "_C_POWF", lambda x, y: real(x, 0.2500001))
        cls = None
    r = run(1, park_cls=cls, honour_flags=False)
    assert r["differ"] > 0, f"the planted fault {fault} went unnoticed"


# ---- the GPU: the five launches, a graph, the tensor tick, the yaw, pitch and roll on the device ----

@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("seed", [1, 2, 3], ids=["development", "held_out", "held_out_3"])
def test_kernels_match_reference_on_the_gpu(seed):
    r = run(seed, path="kernel", device="cuda")
    assert r["worlds"] > 100 and r["ball_steps"] > 9000 and r["events"] > 150, r
    assert r["differ"] == 0, r["examples"]


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
def test_a_graph_matches_the_reference_on_the_gpu():
    r = run(1, path="kernel", device="cuda", graph=True)
    assert r["worlds"] > 40, r
    assert r["differ"] == 0, r["examples"]


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_a_mid_run_load_formations_on_the_gpu(graph):
    """The same mid-run LoadFormations on the five launches, eager and in a CUDA graph (captured anew when the table's
    tensors change)."""
    r = run(1, path="kernel", device="cuda", graph=graph, reload=True)
    assert r["worlds"] > (40 if graph else 100), r
    assert r["differ"] == 0, r["examples"]


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
def test_the_tensor_tick_on_the_gpu_matches_the_reference():
    r = run(2, path="torch", device="cuda")
    assert r["worlds"] > 50, r
    assert r["differ"] == 0, r["examples"]


def _ypr_cuda(*a, **k):
    return vc.VecCombatPark(*a, **k, ypr="cuda")


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("gen", [cl, dc], ids=["these_scenarios", "combat1_scenarios"])
def test_the_yaw_pitch_and_roll_on_the_device(gen):
    """the switch (VecCombatPark(..., ypr="cuda"): phase 4's CalculateYawPitchRoll and the snaps with glibc's
    atan2, asin and sin on the device, vec_ypr_cuda) without dynamical orientation, the five launches, against the
    reference."""
    r = run(1, path="kernel", device="cuda", park_cls=_ypr_cuda, gen=gen, only=lambda cfg: not cfg["do"])
    assert r["worlds"] > 40, r
    assert r["differ"] == 0, r["examples"]


KERNEL_FAULTS = {
    "a removed mushroom's later members released after their commit": [
        ("        if (released) release_in_phase1(c, b, i);                    // the mushroom's turn came first\n", ""),
        ("c.dying[b.follow_ptr] && s.ids[b.follow_ptr] > b.id) {", "c.dying[b.follow_ptr] && s.ids[b.follow_ptr] != b.id) {")],
    "a mushroom's swap not skipped": [("        if (b.mode == MODE_MUSHROOM) {\n            d3_store(s.old_pos, i, p);",
                                       "        if (false) {\n            d3_store(s.old_pos, i, p);")],
    "the dying record never set": [("&& !(tod(mushroom_fraction(c, i)) < 1.0)) ? 1 : 0;",
                                    "&& !(tod(mushroom_fraction(c, i)) < 1.0) && false) ? 1 : 0;")],
    "the members by the follow modes (the modes' rule)": [
        ("return c.exists[i] && member_of(c.m, t, follow_id);",
         "return c.exists[i] && follow_mode(c.m.mode[i]) && member_of(c.m, t, follow_id);")],
    "a troll petrified a tick early": [("b.mode == MODE_TROLL && !(b.effect_stamp > now)) {",
                                        "b.mode == MODE_TROLL && !(b.effect_stamp > now + 1)) {")],
    "powf's exp2 table index one off": [("T_POWF_EXP2_TAB[(uint64_t)ki & (uint64_t)(POWF_EXP2_N - 1)]",
                                         "T_POWF_EXP2_TAB[((uint64_t)ki + 1) & (uint64_t)(POWF_EXP2_N - 1)]")],
    "the removed mushroom kept massive": [("                b.massive = false;\n                b.effect_stamp = b.clock;",
                                           "                b.effect_stamp = b.clock;")],
}


def faulty_source(fault: str, tmp_path) -> Path:
    text = KC.SOURCE.read_text()
    for old, new in KERNEL_FAULTS[fault]:
        assert text.count(old) == 1, (fault, old, text.count(old))
        text = text.replace(old, new)
    src = tmp_path / KC.SOURCE.name
    src.write_text(text)
    return src


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("fault", sorted(KERNEL_FAULTS))
def test_planted_kernel_faults_are_caught(fault, tmp_path, monkeypatch):
    lib = KC.load_library(source=faulty_source(fault, tmp_path))
    real_tick, real_after = KC.tick, KC.after
    monkeypatch.setattr(KC, "tick", lambda park, block=128, l=None: real_tick(park, block, lib))
    monkeypatch.setattr(KC, "after", lambda park, block=128, l=None: real_after(park, block, lib))
    r = run(1, path="kernel", device="cuda", honour_flags=False)
    assert r["differ"] > 0, f"the planted kernel fault {fault} went unnoticed"


def test_the_kernel_faults_apply_to_the_source():
    """Each planted kernel fault's text is in the kernel's source exactly once (checked without a GPU)."""
    text = KC.SOURCE.read_text()
    for fault, pairs in KERNEL_FAULTS.items():
        for old, _ in pairs:
            assert text.count(old) == 1, (fault, old)

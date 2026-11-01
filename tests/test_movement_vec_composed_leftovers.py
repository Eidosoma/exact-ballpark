# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The pieces of the composed park (exact_ballpark.vec_composed.VecComposedPark): the commands
(SetBallFree with its re-filing, SetBallRigid, the Python SetBoid, SetBallTroll with PetrifyTroll's re-filing,
SetBallInteractive), StopAllFollowers' own members (a ball SetBoid took out of a follow mode, released by its target's
warp start), and the mushrooms (AddMushroom, the growth by glibc's powf re-filed every tick, the removal in a
tick with its members released before or after their own turn, BringOutTheDeads) among the composed park's collisions,
against the scalar reference (exact_ballpark.full.FullBallpark), bit for bit: every field the batch holds of every ball at
every step (the collision acceleration, the collision list, the partition's filing, the free, interactive and moribund
flags among them), which balls exist, the ReserveFormationSlot answers and the DoCollision events, on generated
scenarios (tests/composed_leftover_cases.py). On the CPU the torch path; with a CUDA device the kernel path (thirteen
launches, the host's yaw, pitch and roll, the fifth launch's three), held to the torch path every tensor every tick, a
CUDA graph of it, and the torch path on the GPU. The same scenarios run on Destiny's own code compiled verbatim (the conformance kit's full harness), every row equal to the reference's, so this is exact against Destiny. Development scenarios are
seed 1, held-out seeds 2 and 3 (first compared after the generator was frozen). Worlds the batch refuses (its domain
flags) are counted, never compared as equal. Planted faults, on the torch path and in the kernel's source, must each be
caught.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest
import torch

import combat_leftover_cases as cl
import composed_leftover_cases as cc
import exact_ballpark.vec_combat as vc
import exact_ballpark.vec_composed as vp
import exact_ballpark.vec_composed_cuda as KP
from exact_ballpark.modes import BOID, FOLLOW, FORMATION, MISSILE, MUSHROOM, ORBIT, STOP, TROLL

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device visible")
_REF: dict = {}


def references(seed: int, nb: int, nw: int) -> list:
    key = (seed, nb, nw)
    if key not in _REF:
        _REF[key] = [[cc.reference(cfg, w) for w in cfg["worlds"]] for cfg in cc.batches(seed, nb, nw)]
    return _REF[key]


def run(seed: int, nb: int = 12, nw: int = 12, device: str = "cpu", path: str = "torch", graph: bool = False,
        libs: dict | None = None, only=None) -> Counter:
    """Every batch of a seed on the composed park: counts and the first differences (`only`: a predicate on cfg)."""
    refs = references(seed, nb, nw)
    out, examples = Counter(), []
    for bi, cfg in enumerate(cc.batches(seed, nb, nw)):
        if (graph and not cfg["do"]) or (only is not None and not only(cfg)):
            continue
        got, park = cc.batched(cfg, device, path=path, graph=graph, libs=libs)
        flags = park.world_flags()
        for wi, want in enumerate(refs[bi]):
            if not cc.in_domain(want):
                out["outside_reference_domain"] += 1
                continue
            if flags[wi]:
                out["refused"] += 1
                continue
            out["worlds"] += 1
            out["ball_steps"] += sum(len(v) for v in want["S"].values())
            out["events"] += len(want["E"])
            out["collision_rows"] += sum(1 for v in want["S"].values() for row in v.values() if row["collisions"])
            bad = cc.differences(want, got[wi])
            if bad:
                out["differ"] += 1
                if len(examples) < 4:
                    examples.append((bi, wi, bad[:3]))
    out["examples"] = examples
    return out


# ---- the torch path (CPU) ----

def test_the_torch_path_matches_fullballpark_development_cpu():
    r = run(1)
    assert r["worlds"] > 100 and r["ball_steps"] > 15000 and r["collision_rows"] > 1500 and r["events"] > 300, r
    assert r["differ"] == 0, r["examples"]


@pytest.mark.parametrize("seed", [2, 3])
def test_the_torch_path_matches_fullballpark_held_out_cpu(seed):
    """Held-out seeds (first compared after the generator's freeze)."""
    r = run(seed)
    assert r["worlds"] > 100 and r["collision_rows"] > 1500, r
    assert r["differ"] == 0, r["examples"]


# ---- what the scenarios reach (the reference's own counts) ----

_MEMBERS: list = []
_RELEASED: list = []


class _Recording(cl.FullBallpark):
    """The reference, recording at each tick each mushroom's members by StopAllFollowers' own test (follow_ptr and
    follow id name it) when the mushroom goes, and the members a warp start releases out of the follow modes."""

    def evolve(self, timestamp: int = 0) -> None:
        before = {b.id: [(f.id, f.mode) for f in self.balls.values()
                         if f.id != b.id and f.follow_ptr == b.id and f.follow_id == b.id]
                  for b in self.balls.values() if b.mode == MUSHROOM and not b.is_moribund}
        stale = {f.id: f.follow_id for f in self.balls.values()
                 if f.mode not in (FOLLOW, ORBIT, MISSILE, FORMATION) and f.follow_ptr is not None
                 and f.follow_ptr in self.balls and f.follow_id == f.follow_ptr and not f.is_moribund}
        warping = {b.id for b in self.balls.values() if b.mode == 3 and b.effect_stamp < 0}
        super().evolve(timestamp)
        _MEMBERS.extend((mid, m) for mid, m in before.items() if self.balls[mid].is_moribund)
        for fid, tid in stale.items():
            t = self.balls.get(tid)
            if tid in warping and t is not None and t.effect_stamp >= 0 and fid in self.balls:
                _RELEASED.append(self.balls[fid].mode)


def test_the_scenarios_reach_every_branch(monkeypatch):
    """The development scenarios reach what the composed park adds: mushrooms grown (and re-filed) and removed in a
    tick, with members whose turns come after the mushroom's (missiles, formation followers, followers, orbiters,
    members SetBoid took out of a follow mode, trolls among them) and before it; trolls petrified (re-filed fixed);
    balls fixed by SetBallFree (re-filed); members SetBoid took out of a follow mode released by their target's warp
    start; collisions with a growing mushroom."""
    monkeypatch.setattr(cc, "FullBallpark", _Recording)
    trace, seen = Counter(), Counter()
    _RELEASED.clear()
    for cfg in cc.batches(1, 12, 12):
        for w in cfg["worlds"]:
            _MEMBERS.clear()
            want = cc.reference(cfg, w, trace)
            for mid, members in _MEMBERS:
                seen["removed"] += 1
                for fid, mode in members:
                    seen[(mode, "after" if fid > mid else "before")] += 1
            mids = {b for s in want["S"].values() for b, row in s.items() if row["mode"] == MUSHROOM}
            for s in want["S"].values():
                for bid, row in s.items():
                    if bid not in mids and any(c in mids for c in row["collisions"]):
                        seen["collided_with_a_mushroom"] += 1
            for rec in w["records"]:
                seen["record " + rec[1]] += 1
    seen["boid_released_by_warp"] = len(_RELEASED)
    assert trace["mushroom.grow"] > 200 and trace["mushroom.done"] > 20 and trace["troll.petrified"] > 10, trace
    assert seen["removed"] > 20 and seen["collided_with_a_mushroom"] > 20, seen
    for mode in (MISSILE, FOLLOW, ORBIT, FORMATION):
        assert seen[(mode, "after")] > 0, (mode, seen)
    assert sum(v for k, v in seen.items() if isinstance(k, tuple) and k[1] == "before") > 0, seen
    assert seen[(BOID, "after")] + seen[(TROLL, "after")] > 0, seen
    assert seen["boid_released_by_warp"] > 0, seen
    for op in ("setfree", "rigid", "boid", "troll", "inter", "mushroom", "deads", "warp", "warpin", "launch"):
        assert seen["record " + op] >= 5, (op, seen)


# ---- planted faults on the torch path (each must change the comparison) ----

_REAL = {"powf": vp._C_POWF, "file": vp.VecComposedPark._file, "turns": vp.VecComposedPark._release_turns}


def _f_powf(x, y):
    return _REAL["powf"](x, 0.2500001)                      # the growth's exponent nudged


def _f_file(self, which, pos=None, vel=None, sf=None):
    return _REAL["file"](self, which & (self.mode != MUSHROOM), pos, vel, sf)     # a mushroom row never re-filed


def _f_turns(self, hit, ts):
    return _REAL["turns"](self, hit & self._following(), ts)      # phase 1's members by the follow modes


TORCH_FAULTS = {"growth_exponent": (vp, "_C_POWF", _f_powf),
                "growth_not_refiled": (vp.VecComposedPark, "_file", _f_file),
                "fixed_not_refiled": (vp.VecComposedPark, "set_free", vc.VecCombatPark.set_free),
                "phase1_members_by_follow_modes": (vp.VecComposedPark, "_release_turns", _f_turns)}


@pytest.mark.parametrize("fault", sorted(TORCH_FAULTS))
def test_planted_torch_path_faults_are_caught(fault, monkeypatch):
    owner, name, fn = TORCH_FAULTS[fault]
    monkeypatch.setattr(owner, name, fn)
    r = run(1)
    assert r["differ"] > 0, (fault, dict(r))


# ---- the commands where a tick lacks them ----

def test_the_composed_park_accepts_the_commands_and_mushrooms():
    """VecComposedPark's own tick (its torch path and its kernels) holds the commands and AddMushroom; a
    subclass with yet another tick refuses them (VecModesPark._g11's rule)."""
    p = vp.VecComposedPark(3, device="cpu")
    p.load([{"id": k + 1, "world": 0, "mass": 1e6, "radius": 10.0, "max_vel": 100.0, "pos": [k * 100.0, 0.0, 0.0],
             "vel": [0.0, 0.0, 0.0], "agility": 1.0, "speed_fraction": 1.0} for k in range(2)]
           + [{"id": 900001, "world": 0, "exists": False, "mass": 1.0, "radius": 1.0, "max_vel": 1.0,
               "pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "agility": 1.0, "speed_fraction": 1.0}])
    one = torch.tensor([True, False, False])
    p.set_ball_rigid(one)
    p.set_boid(one, torch.ones(3, dtype=torch.int64))
    p.set_ball_troll(one, torch.full((3,), 3, dtype=torch.int64))
    p.set_free(one, torch.zeros(3, dtype=torch.int64))
    p.add_mushroom(torch.tensor([False, False, True]), torch.tensor([0, 0, 2]), torch.full((3,), 50.0),
                   torch.full((3,), 2.0))
    assert bool(p.exists[2]) and int(p.mode[2]) == MUSHROOM and int(p.mode[0]) == STOP and not bool(p.free[0])

    class Other(vp.VecComposedPark):
        def evolve_torch(self) -> None:
            super().evolve_torch()

    q = Other(1, device="cpu")
    with pytest.raises(NotImplementedError):
        q.set_ball_rigid(torch.tensor([True]))


def test_the_kernel_faults_are_in_the_source():
    """Each of the planted kernel faults changes the compiled text, and FAULTS (the earlier tests'
    list, run on worlds without mushrooms or trolls) does not hold them."""
    text = KP.SOURCE.read_text()
    base = KP.active_text(text, 0)
    for name, level in KP.G11_FAULTS.items():
        assert KP.active_text(text, level) != base, name
        assert name not in KP.FAULTS
    assert set(KP.ALL_FAULTS) == set(KP.FAULTS) | set(KP.G11_FAULTS)


# ---- the same worlds on Destiny's own code ----

def test_the_scenarios_are_destinys_own():
    """The development scenarios on Destiny's own code compiled verbatim (the conformance kit's full harness) and on the
    reference, every row of every kind equal (balls, partition, bubbles, events, answers): the chain's first link on
    these very worlds."""
    import full_cases as fc
    if not fc.HARNESS.is_file():
        pytest.skip("Destiny full harness not built (conformance/build_full.py)")
    cs, cid = [], 0
    for cfg in cc.batches(1, 12, 12):
        for w in cfg["worlds"]:
            cid += 1
            ref = cc.reference(cfg, w)
            if cc.in_domain(ref):                   # (a world the reference refuses is compared nowhere)
                cs.append(cc.harness_case(cfg, w, cid, ref["A"]))
    rows, undefined = fc.run_reference(cs)
    want = fc.oracle([c for c in cs if c["id"] not in undefined])
    assert not fc.differences(want, rows, undefined)
    assert len(rows["S"]) > 15000 and len(rows["E"]) > 300, {k: len(v) for k, v in rows.items()}


# ---- the kernel path (a CUDA device) ----

@CUDA
@pytest.mark.parametrize("seed", [1, 2, 3], ids=["development", "held_out_2", "held_out_3"])
def test_the_kernel_path_matches_fullballpark(seed):
    r = run(seed, device="cuda", path="kernel")
    assert r["worlds"] > 100 and r["collision_rows"] > 1500, r
    assert r["differ"] == 0, r["examples"]


@CUDA
def test_the_torch_path_on_the_gpu_matches_fullballpark():
    r = run(2, device="cuda", path="torch")
    assert r["worlds"] > 100, r
    assert r["differ"] == 0, r["examples"]


@CUDA
def test_a_cuda_graph_matches_fullballpark():
    """The composed tick and its fifth launch captured in one CUDA graph (dynamical orientation), replayed tick by
    tick with the commands between replays, against the reference."""
    r = run(1, device="cuda", graph=True)
    assert r["worlds"] > 40, r
    assert r["differ"] == 0, r["examples"]


def lockstep(cfg: dict, libs: dict | None = None, graph: bool = False) -> tuple:
    """The batch on two parks on the GPU, the torch path (A) and the kernel path (B, or a CUDA graph of it), the same
    commands on both, every state tensor compared after the load and after every step's commands and tick
    (test_movement_vec_composed_cuda.compare: bit for bit, two NaNs equal); the events compared."""
    import test_movement_vec_composed_cuda as TC
    rows = cl.rows_for(cfg)
    W = len(cfg["worlds"])

    def make():
        p = vp.VecComposedPark(len(rows), cfg["friction"], cfg["tick"], "cuda", cfg["do"], cfg["new_orbit"],
                               master=cfg["master"], disable_do_for_missiles=cfg["do_off"])
        p.load(rows)
        p.load_formations(cc.FORMATIONS)
        ids0, world0 = p.ids.tolist(), p.world.tolist()
        p.mush_rows = {wi: [r for r in range(p.n) if world0[r] == wi and ids0[r] in cc.MUSH_POOL] for wi in range(W)}
        return p
    A, B = make(), make()
    g = KP.ComposedGraph(B, libs=libs) if graph else None
    outs = {k: [{"S": {}, "E": [], "A": [], "B": [], "X": None} for _ in range(W)] for k in "AB"}
    by_step = [defaultdict(list) for _ in range(W)]
    for wi, w in enumerate(cfg["worlds"]):
        for rec in w["records"]:
            by_step[wi][rec[0]].append(rec)
    out, examples = Counter(), []
    TC.compare(A, B, out, examples, ("load",))
    for s in range(cfg["steps"] + 1):
        slots = max(len(by_step[wi].get(s, ())) for wi in range(W))
        for j in range(slots):
            recs = {wi: by_step[wi][s][j] for wi in range(W) if j < len(by_step[wi].get(s, ()))}
            cc.apply_batch(A, recs, outs["A"])
            cc.apply_batch(B, recs, outs["B"])
        if s > 0:
            A.evolve_torch()
            if g is not None:
                g.tick()
            else:
                B.evolve(libs)
            ea, eb = A.events(), B.events()
            out["events"] += len(ea)
            if ea != eb:
                out["events_differ"] += 1
                examples.append((s, "events", ea[:3], eb[:3]))
            out["ticks"] += 1
            out["moribund"] += int((A.moribund & A.exists).sum())
            out["petrified"] += int(A.petrified.sum())
        TC.compare(A, B, out, examples, (s,))
    if [o["A"] for o in outs["A"]] != [o["A"] for o in outs["B"]]:
        out["answers_differ"] += 1
    return out, examples


@CUDA
def test_the_kernel_equals_the_torch_path_every_tensor_every_tick():
    total = Counter()
    for cfg in cc.batches(1, 12, 12)[:6]:
        out, ex = lockstep(cfg)
        assert out["differ"] == 0 and not out["events_differ"] and not out["answers_differ"], ex[:6]
        total.update(out)
    assert total["ticks"] > 80 and total["moribund"] > 0 and total["petrified"] > 0, total


@CUDA
def test_a_cuda_graph_equals_the_torch_path_every_tensor_every_tick():
    total = Counter()
    for cfg in [c for c in cc.batches(1, 12, 12) if c["do"]][:3]:
        out, ex = lockstep(cfg, graph=True)
        assert out["differ"] == 0 and not out["events_differ"] and not out["answers_differ"], ex[:6]
        total.update(out)
    assert total["ticks"] > 30, total


@CUDA
@pytest.mark.parametrize("fault", sorted(KP.G11_FAULTS))
def test_planted_kernel_faults_are_caught(fault):
    """Each of the planted kernel faults (a growing mushroom not re-filed, a removed mushroom's later
    members released after their commit, a mushroom's swap not skipped, the petrified trolls not re-filed) changes the
    kernel path's comparison with the reference on the development seed."""
    libs = KP.libraries(composed_lib=KP.load_library(fault=fault))
    r = run(1, device="cuda", path="kernel", libs=libs)
    assert r["differ"] > 0, (fault, dict(r))

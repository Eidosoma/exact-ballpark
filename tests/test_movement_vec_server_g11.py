# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The server-layer pieces against the scalar server layer (exact_ballpark.server over FullBallpark) and the
kernels against their torch paths, bit for bit, on tests/server_g11_cases.py's scenarios:

- SetBallTroll and SetBallFree as queued Actions (two kinds appended to vec_server.KINDS): through VecServer over
  VecModesPark (the torch path and the kernel), CombatServer over VecCombatPark (the torch path and the
  kernel), and SetBallFree through SpaceServer over VecModesPark (the torch path and the kernel; SetBallFree(false)
  re-files its ball: the layer's REFILE); PetrifyTroll in the parks' ticks; the streams writing a troll's stamp and
  leaving out a rigid ball's mass block (a mushroom's fields too), as WriteBallToStream does;
- has_pending_set_free and has_pending_set_not_free, the answer and its type (Destiny's int 0 where nothing is pending);
- StopAllFollowers' members on every path (CloakBall's and EntityWarpIn's releases): a ball the Python SetBoid took out
  of a follow mode between server ticks is released by its target's CloakBall or EntityWarpIn in the kernel (and
  an earlier generator's) as in the torch path and the scalar layer;
- tests/server_cases.py's reading of the collision lists (the finding 1): a collide world whose ships
  meet, its collision lists compared as ids.

Development seed 1; held-out seed 2 (first run after the generator's freeze). Planted kernel faults must be caught.
"""

from __future__ import annotations

import logging
import struct

import pytest
import torch

import server_cases as S
import server_g11_cases as G
from exact_ballpark import server as srv
from exact_ballpark import vec_server as VS

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
logging.getLogger("exact_ballpark.server").setLevel(logging.CRITICAL)


def _clean(out, bad) -> None:
    assert bad == [], bad[:4]
    assert out["rows"] > 0 and out["updates"] > 0 and out["asked"] > 0 and out["pending"] > 0, out


# ---- the kinds ----

def test_the_kinds_are_appended_never_renumbered():
    assert VS.KINDS[:25] == ("", "GotoPoint", "GotoDirection", "Stop", "FollowBall", "Orbit", "WarpTo",
                             "EntityWarpIn", "SetSpeedFraction", "SetMaxSpeed", "SetBallMass", "SetBallAgility",
                             "SetBallRadius", "SetBallVelocity", "SetBallMassive", "SetBallInteractive",
                             "SetBallPosition", "CloakBall", "UncloakBall", "SetMaxAngularSpeed",
                             "SetBallAngularVelocity", "SetBallRotation", "SetBallAngularAgility",
                             "SetMaxAngularVelocity", "LaunchMissile")
    assert (VS.CODE["SetBallTroll"], VS.CODE["SetBallFree"]) == (25, 26) == (VS.TROLL, VS.FREE)
    assert VS.ACTION_KINDS["set_ball_troll"] == "SetBallTroll" and VS.ACTION_KINDS["set_ball_free"] == "SetBallFree"
    for name in ("set_ball_troll", "set_ball_free"):                   # Destiny's Actions has both
        assert hasattr(srv.Actions, name)
    from exact_ballpark import vec_server_space as VSS
    from exact_ballpark.vec_modes import VecModesPark
    p = VecModesPark(2, device="cpu")
    p.load([{"id": k + 1, "world": 0, "mass": 1e6, "radius": 10.0, "max_vel": 100.0, "pos": [0.0, 0.0, 0.0],
             "vel": [0.0, 0.0, 0.0], "agility": 1.0, "speed_fraction": 1.0} for k in range(2)])
    assert {VS.TROLL, VS.FREE} <= VS.ModesAdapter(p).kinds
    assert VS.FREE in VSS.SpaceModesAdapter(p).kinds and VS.TROLL not in VSS.SpaceModesAdapter(p).kinds


# ---- on the CPU: VecModesPark's pre-tick (its tick is a kernel), CombatServer's torch path ----

@pytest.mark.parametrize("do", [False, True])
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_modes_pre_tick_matches_the_scalar_layer_cpu(do, seed):
    out, bad = G.run_modes(G.modes_cases(seed, 40, do), "cpu", do=do, evolve=False)
    _clean(out, bad)
    assert out["trolls"] > 0 and out["fixed"] > 0 and out["boids"] > 0 and out["stale_members_kept"] > 0, out


@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_combat_torch_path_matches_the_scalar_layer_cpu(seed):
    import server_combat_cases as SC
    r = SC.run(G.combat_cases(seed, 30, device="cpu"), "cpu")
    assert r["rows"] == [] and r["sends"] == [], (r["rows"][:2], r["sends"][:1])
    assert r["compared"] > 0 and r["updates"] > 0, r
    assert r["reached"].get("SetBallTroll", 0) > 0 and r["reached"].get("SetBallFree", 0) > 0, r["reached"]


# ---- the streams (WriteBallToStream) of a troll, a rigid ball, a mushroom, a boid ----

def test_the_streams_write_trolls_rigid_balls_and_mushrooms_as_destiny_does():
    """A master's park on the scalar layer (DestinyPark over FullBallpark) and on VecCombatPark: SetBallTroll, a
    mushroom, SetBoid, then ticks until the troll is petrified (RIGID); after each step every ball's stream from
    vec_server.ball_stream equals the scalar park's WriteBallsToStream, byte for byte."""
    from exact_ballpark.vec_combat import VecCombatPark
    from exact_ballpark.vec_server_combat import CombatAdapter
    srv.disable_dynamical_orientation()
    ships = [(k, 1e7 * k, 20.0 * k, 100.0, [100.0 * k, 50.0, -20.0 * k], [5.0, 0.0, 1.0 * k]) for k in (1, 2, 3)]
    park = srv.DestinyPark(is_master=True)
    for bid, mass, radius, vmax, pos, vel in ships:
        park.AddBall(bid, mass, radius, vmax, 1, 0, 0, 1, 0, *pos, *vel, 1.0, 1.0)     # free, not massive
    p = VecCombatPark(4, device="cpu", master=True)
    p.load([{"id": bid, "world": 0, "mass": mass, "radius": radius, "max_vel": vmax, "pos": pos, "vel": vel,
             "agility": 1.0, "speed_fraction": 1.0, "massive": False, "interactive": True, "bubble": 0}
            for bid, mass, radius, vmax, pos, vel in ships]
           + [{"id": 900001, "world": 0, "exists": False, "mass": 1.0, "radius": 1.0, "max_vel": 1.0,
               "pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "agility": 1.0, "speed_fraction": 1.0, "bubble": 0}])
    a = CombatAdapter(p)
    row = lambda i: torch.tensor([r == i for r in range(4)])              # noqa: E731
    seen = set()
    for step in range(5):
        if step == 0:
            park.SetBallTroll(2, 2)
            p.set_ball_troll(row(1), torch.full((4,), 2, dtype=torch.int64))
            park.lane.add_mushroom(1, 40.0, 3.0)
            p.add_mushroom(row(3), torch.full((4,), 1, dtype=torch.int64), torch.full((4,), 40.0),
                           torch.full((4,), 3.0))
            park.SetBoid(3, 1)
            p.set_boid(row(2), torch.ones(4, dtype=torch.int64))
        else:
            park.Evolve()
            p.evolve_torch()
        fl = a.stream_fields()
        ids = p.ids.tolist()
        for r in range(4):
            if not bool(p.exists[r]):
                continue
            s = srv.MemStream()
            park.WriteBallsToStream([ids[r]], s)
            s.Seek(0)
            want = s.Read()
            got = VS.ball_stream(fl, [r], park.lane.current_time)
            assert got == want, (step, ids[r], int(p.mode[r]), want.hex(), got.hex())
            seen.add(int(p.mode[r]))
    assert {6, 7, 8, 11} <= seen, seen                                  # MUSHROOM, BOID, TROLL, RIGID


# ---- the collision lists of a collide world (the finding) ----

def test_a_collide_world_with_a_collision_is_compared():
    """the runner on a VecCollidePark world whose ships meet: every row equal to the scalar layer's, collision lists
    included, now read as the ids Potential appends; the earlier reading (the entries as slots) gives other lists on
    the same rows, so its comparison could not have passed on such a world."""
    w = G.collide_world()
    res = S.run([w], "collide", "cpu")
    assert res["rows"] == [] and res["sends"] == [] and res["compared"] > 0, (res["rows"][:2], res["sends"][:1])
    server, index = S.build([G.collide_world()], "collide", "cpu")
    sw = S.ScalarWorld(G.collide_world())
    cs = [G.collide_world()]
    met = old_differs = 0
    for t in range(len(cs[0]["ticks"])):
        sw.tick(cs[0]["ticks"][t])
        for kind, *data in S.pushes(cs, t, "cpu"):
            server.push(kind, *data)
        server.tick()
        p = server.a.park
        hits, hn, ids = p.hits.cpu(), p.hits_n.cpu(), server.a.ids.tolist()
        for bid, b in sw.park.lane.balls.items():
            s = index[(0, bid)]
            new = tuple(int(o) for o in hits[0, s, :int(hn[0, s])].tolist())
            assert new == tuple(b.collisions), (t, bid, new, b.collisions)
            if b.collisions:
                met += 1
                try:
                    old = tuple(ids[o] if o >= 0 else o for o in new)
                except IndexError:
                    old = None
                old_differs += int(old != tuple(b.collisions))
    assert met > 0 and old_differs == met, (met, old_differs)


# ---- on a CUDA device ----

@CUDA
@pytest.mark.parametrize("server_kernel", ["torch", "cuda"])
@pytest.mark.parametrize("do", [False, True])
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_modes_layer_matches_the_scalar_layer_on_the_gpu(server_kernel, do, seed):
    """VecServer over VecModesPark with its tick (PetrifyTroll the modes kernel's), on the torch path and on the
    kernel, against the scalar layer: rows, updates, the pending queries; ships set to BOID between ticks."""
    out, bad = G.run_modes(G.modes_cases(seed, 40, do), "cuda", do=do, evolve=True, server_kernel=server_kernel)
    _clean(out, bad)
    assert out["trolls"] > 0 and out["rigid"] > 0 and out["fixed"] > 0 and out["stale_members_kept"] > 0, out


def _modes_pair(cs: list, do: bool, fault: str | None = None) -> tuple:
    ts, index = S.build(cs, "modes", "cuda", do=do)
    ks, _ = S.build(cs, "modes", "cuda", do=do, server_kernel="cuda")
    if fault is not None:
        from exact_ballpark import vec_server_cuda as V
        ks._lib = V.load_library(variant="sm90", fault=fault, checks=False)
    return ts, ks, index


@CUDA
@pytest.mark.parametrize("do", [False, True])
def test_the_modes_kernel_equals_its_torch_path(do):
    """the kernel with the two kinds and StopAllFollowers' members (BOID members between ticks) against its torch
    path: every tensor of the parks and adapters, the server's state, the Batch, the decoded updates, every tick."""
    cs = G.modes_cases(1, 40, do)
    ts, ks, index = _modes_pair(cs, do)
    out, bad = G.lockstep(ts, ks, cs, index, S.pushes)
    assert bad == [], bad[:4]
    assert out["queued SetBallTroll"] > 0 and out["queued SetBallFree"] > 0 and out["rigid"] > 0, out


@CUDA
@pytest.mark.parametrize("fault", ["release_by_follow_modes", "troll_delay_unfloored", "fixed_keeps_last_g"])
def test_the_modes_kernels_planted_faults_are_caught(fault):
    cs = G.modes_cases(1, 40, False)
    ts, ks, index = _modes_pair(cs, False, fault)
    _, bad = G.lockstep(ts, ks, cs, index, S.pushes)
    assert bad, fault


@CUDA
@pytest.mark.parametrize("fault", [None, "release_by_follow_modes", "troll_delay_unfloored", "fixed_keeps_last_g"])
def test_the_combat_kernel_equals_its_torch_path(fault):
    """the kernel with the two kinds and StopAllFollowers' members (BOID members between ticks) against its torch
    path (VecCombatPark's kernels under both), every tensor every tick; each planted fault caught."""
    import server_combat_kernel_cases as KC
    cs = G.combat_cases(1, 30, kernel_worlds=True)
    ts, index = KC.build(cs, "cuda")
    ks, _ = KC.build(cs, "cuda", kernel="cuda", fault=fault)
    out, bad = G.lockstep(ts, ks, cs, index, KC.pushes)
    if fault is None:
        assert bad == [], bad[:4]
        assert out["queued SetBallTroll"] > 0 and out["queued SetBallFree"] > 0 and out["boids"] > 0, out
    else:
        assert bad, fault


@CUDA
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_combat_kernel_matches_the_scalar_layer(seed):
    import server_combat_kernel_cases as KC
    r = KC.run_scalar(G.combat_cases(seed, 30, device="cuda"), kernel="cuda")
    assert r["rows"] == [] and r["sends"] == [] and not r["flagged"], (r["rows"][:2], r["sends"][:1], r["flagged"])
    assert r["reached"].get("SetBallTroll", 0) > 0 and r["reached"].get("SetBallFree", 0) > 0, r["reached"]


@CUDA
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_space_layer_matches_the_scalar_layer_on_the_gpu(seed):
    """SpaceServer over VecModesPark with SetBallFree (its REFILE) against the scalar layer over several bubbles."""
    import server_space_cases as SS
    r = SS.run(G.space_cases(seed, 40), "modes", "cuda")
    assert r["rows"] == [] and r["sends"] == [], (r["rows"][:2], r["sends"][:1])


@CUDA
@pytest.mark.parametrize("fault", [None, "free_not_refiled"])
def test_the_space_kernel_equals_its_torch_path(fault):
    import test_movement_vec_server_space_cuda as SQ
    bad, vol = SQ.kernel_against_torch(G.space_cases(1, 40), fault=fault)
    if fault is None:
        assert bad == [], bad[:4]
    else:
        assert bad, fault


@CUDA
@pytest.mark.parametrize("do", [False, True])
def test_a_server_graph_with_the_two_kinds_equals_eager_kernel_ticks(do):
    """the ServerGraph on its kernel with a fixed schedule holding SetBallTroll and SetBallFree (PetrifyTroll inside
    the captured park's tick, the send flags' phase after it): the same rows, routes and flags as eager kernel ticks,
    every tick (the schedule's arguments drawn here, on seed 1's worlds: development, added after the freeze)."""
    import random
    from collections import Counter
    cs = G.modes_cases(1, 64, do, ticks=10, boids=False)
    sched = ["SetBallTroll", "SetBallFree", "GotoDirection", "SetBallFree", "SetBallInteractive"]
    eager, index = S.build(cs, "modes", "cuda", do=do, server_kernel="cuda")
    graphed, _ = S.build(cs, "modes", "cuda", do=do, server_kernel="cuda")
    g = VS.ServerGraph(graphed, sched)
    r = random.Random(f"compose3-graph-{int(do)}")
    W, seen = len(cs), Counter()
    for t in range(10):
        per = []
        for kind in sched:
            mask = torch.tensor([r.random() < 0.5 for _ in range(W)])
            pick = [[b["id"] for b in w["balls"] if b["id"] not in (w["special"], w["anchor"])] or [w["anchor"]]
                    for w in cs]
            ball = torch.tensor([r.choice(ids) for ids in pick], dtype=torch.int64)
            f = torch.zeros(W, 4, dtype=torch.float64)
            n = torch.zeros(W, dtype=torch.int64)
            if kind == "SetBallTroll":
                n = torch.tensor([r.choice([-1, 1, 2, 4]) for _ in range(W)], dtype=torch.int64)
            elif kind in ("SetBallFree", "SetBallInteractive"):
                n = torch.tensor([r.choice([0, 1]) for _ in range(W)], dtype=torch.int64)
            else:
                f[:, :3] = torch.randn(W, 3, generator=torch.Generator().manual_seed(t), dtype=torch.float64)
            per.append((kind, mask, ball, torch.zeros(W, dtype=torch.int64), f, n, torch.zeros(W, dtype=torch.int64)))
        for kind, *data in per:
            eager.push(kind, *(x.cuda() for x in data))
        eager.pre_tick(snapshot=False)
        eager.evolve()
        eager.post_tick()
        for slot_in, (kind, *data) in zip(g.inputs, per):
            for key, x in zip(("mask", "ball", "target", "f", "n", "loc"), data):
                slot_in[key].copy_(x.cuda())
        out = g.tick()
        torch.cuda.synchronize()
        a, b = S.batched_rows(eager, cs, index, "modes", do), S.batched_rows(graphed, cs, index, "modes", do)
        assert all(S.same_rows(a[k], b[k]) for k in a), t
        assert torch.equal(eager.batch.route, out.route), t
        assert torch.equal(eager.batch.has_character, out.has_character), t
        assert torch.equal(eager.batch.gets_bubble, out.gets_bubble), t
        p = eager.a.park
        seen["rigid"] += int((p.mode == 11).sum())
        seen["trolls"] += int((p.mode == 8).sum())
        seen["fixed"] += int((~p.free).sum())
    assert seen["rigid"] > 0 and seen["trolls"] > 0 and seen["fixed"] > 0, seen


def _bits(x: float) -> str:
    return struct.pack("<d", float(x)).hex()

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer over the combat park as one CUDA kernel (exact_ballpark.vec_server_combat_cuda: LaunchMissile, the missile and formation branches of Stop and StopAllFollowers, the lookups of the balls in the park,
SetBallVelocity's snaps) against the torch path (CombatServer, kernel="torch", itself held to the scalar server layer
by tests/test_movement_vec_server_combat.py), bit for bit, every tick: every tensor of the two parks and adapters, the
server's state, every field of the tick's Batch (the queued actions' rows, duplicates and CloakBall's bubbles, the
routes, the uncloaks' notices, the bubble update's lists, the stamps, the send flags) and the update batches decoded;
and directly against the scalar server layer (exact_ballpark.server over FullBallpark). On the CPU: the generated header,
the source scan with and without each planted fault, the build and its PTX check, the layout, the refusals. On the GPU
(skipped without one): tests/server_combat_kernel_cases.py's command streams (development seeds 1 and 2, held-out seed
7, with and without dynamical orientation), tests/server_combat_cases.py's worlds against the scalar layer (seed 1 and
held-out seed 9, with and without dynamical orientation), one launch, block sizes, tiled worlds, a CUDA graph against
eager kernel ticks, the yaw and pitch's reduction by fmod on crafted angles, the kernel's resources, and eight planted
faults (each built into its own library, each caught). The case generator tests/server_combat_kernel_cases.py was
frozen before the held-out seeds ran.
"""

from __future__ import annotations

import logging
import math
import re
import struct

import pytest
import torch

import server_combat_cases as SC
import server_combat_kernel_cases as KC
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_combat as VSC
from exact_ballpark import vec_server_combat_cuda as SK

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
logging.getLogger("exact_ballpark.server").setLevel(logging.CRITICAL)
HELD_OUT = 7                                                # the kernel's streams: first run after the freeze
HELD_OUT_SCALAR = 9                                         # server_combat_cases' worlds: first run on the kernel
I64, F64, BOOL = torch.int64, torch.float64, torch.bool


# ==================================================================================================== on the CPU

def test_the_header_holds_the_constants_and_the_literals():
    """Every constant from the reference's modules, each literal found in the torch path's source with its spelling; a
    perturbed value changes exactly one line."""
    from exact_ballpark import full as FU
    from exact_ballpark import modes as MO
    from exact_ballpark import vec_combat as VC
    from exact_ballpark import vec_orient as VO
    h = SK.tables_header()
    for name, val in (("SK_MODE_MISSILE", MO.MISSILE), ("SK_MODE_FORMATION", MO.FORMATION),
                      ("SK_FORMATION_SLOTS", FU.FORMATION_MAX_SLOTS), ("SK_FLAG_UNDEFINED", VC.FLAG_UNDEFINED),
                      ("SK_FLAG_DEFENDER", VC.FLAG_DEFENDER)):
        assert f"#define {name} {val}\n" in h
    for name, val in (("SK_LAUNCH_SPEED", FU.MISSILE_LAUNCH_SPEED), ("SK_MISSILE_GOTO", VC.MISSILE_GOTO),
                      ("SK_TWO_PI", VO.TWO_PI)):
        assert f"#define {name} 0x{SK._dbits(val):016x}ULL\n" in h
    for name, val in (("SK_F_HALF", 0.5), ("SK_BACK_X", 0.0), ("SK_BACK_Y", 0.0), ("SK_BACK_Z", -1.0)):
        assert f"#define {name} 0x{SK._fbits(val):08x}u\n" in h
    assert all(SK.literals_in_source().values()), SK.literals_in_source()
    p = SK.tables_header({"SK_TWO_PI": 6.28})
    assert sum(a != b for a, b in zip(h.splitlines(), p.splitlines())) == 1


def test_the_sources_call_no_cuda_math_and_each_fault_changes_its_rule():
    """The source scan finds nothing in the five sources as compiled, with and without each planted fault, and flags a
    planted call; each fault's text differs from the clean text."""
    assert SK.source_scan(0) == []
    clean = SK.active_text(SK.SOURCE.read_text(), 0)
    for name, level in SK.FAULTS.items():
        assert SK.source_scan(level) == [], name
        assert SK.active_text(SK.SOURCE.read_text(), level) != clean, name
    from exact_ballpark import vec_orient_cuda as K
    text = SK.SOURCE.read_text().replace("b = fmod_exact(b, tp);", "b = fmod(b, tp);")
    assert any(n == "fmod" for n, _ in K.scan_source(SK.active_text(text, 0)))


def test_the_build_pins_every_rounding_and_the_layout(tmp_path):
    """The library builds with the pinned flags; its PTX passes the check (every add, sub, mul, div, sqrt and rcp
    rounded to nearest; no mad, flush or approximation); the kernel's resources recorded, no spill; the ctypes
    structures equal the compiled ones."""
    m = SK.build(tmp_path, variant="sm90")
    assert m["ptx_check"]["ok"] and not m["source_scan"]
    assert "-fmad=false" in m["flags"] and "-prec-div=true" in m["flags"] and "-ftz=false" in m["flags"]
    assert any("server_combat_kernel" in ln for ln in m["ptxas"])
    spills = [int(x) for ln in m["ptxas"] for x in re.findall(r"(\d+) bytes spill", ln)]
    assert spills and not any(spills)
    lay = SK.layout_check(m, tmp_path)
    assert all(a == b for a, b in lay.values()), lay


def test_refusals():
    """The kernel runs a VecCombatPark's server layer on a CUDA device only; a kernel's name is checked."""
    cs = SC.cases(1, 2, "cpu", 1)
    server, _ = SC.build(cs, "cpu")
    with pytest.raises(ValueError, match="VecCombatPark .not a subclass. on a CUDA device"):
        VSC.CombatServer(server.a, server.pilot, kernel="cuda")
    with pytest.raises(ValueError, match="'torch' or 'cuda'"):
        VSC.CombatServer(server.a, server.pilot, kernel="numba")


# ==================================================================================================== on the GPU

MINIMUM = {"LaunchMissile": 150, "launches_on_a_ball": 120, "CloakBall": 80, "Stop": 80, "SetBallVelocity": 60,
           "in_MISSILE": 200, "in_FORMATION": 120, "missiles_released": 15, "slots_freed": 15}


def _assert_volumes(vol: dict, minimum: dict) -> None:
    got = dict(vol["queued"], **{k: v for k, v in vol.items() if not isinstance(v, dict)})
    for key, low in minimum.items():
        assert got.get(key, 0) >= low, (key, got)


@CUDA
@pytest.mark.parametrize("do", [False, True])
@pytest.mark.parametrize("seed", [1, 2, HELD_OUT])
def test_the_kernel_equals_the_torch_path(seed, do):
    """Every tick bit for bit: both parks' and adapters' tensors, the server's state, the Batch, the decoded updates
    (snapshots on): 60 worlds x 12 ticks."""
    bad, vol = KC.kernel_against_torch(KC.cases(seed, 60, do=do, ticks=12))
    assert bad == [], bad[:6]
    _assert_volumes(vol, MINIMUM)


@CUDA
def test_the_kernel_in_one_launch_equals_the_torch_path():
    """Without snapshots the pre-tick is one launch: the same states and Batch."""
    bad, vol = KC.kernel_against_torch(KC.cases(2, 60, do=False, ticks=12), snapshot=False)
    assert bad == [], bad[:6]
    _assert_volumes(vol, MINIMUM)


SCALAR_MINIMUM = {"LaunchMissile": 300, "CloakBall": 300, "Stop": 300, "AddBallsToPark": 200, "FollowBall": 50,
                  "Orbit": 50}


@CUDA
@pytest.mark.parametrize("do", [False, True])
@pytest.mark.parametrize("seed", [1, HELD_OUT_SCALAR])
def test_the_kernel_equals_the_scalar_server_layer(seed, do):
    """Directly against exact_ballpark.server over FullBallpark (no torch path between): every client's update list and
    every ball's state, tests/server_combat_cases.py's worlds (200 x 12), as generated (no dynamical orientation) and
    with dynamical orientation on both sides."""
    res = KC.run_scalar([dict(w, do=do) for w in SC.cases(seed, 200, "cuda", 12)])
    assert not res["flagged"], res["flagged"][:10]
    assert res["rows"] == [] and res["sends"] == [] and res["compared"] > 0, (res["rows"][:2], res["sends"][:1])
    for key, low in SCALAR_MINIMUM.items():
        assert res["reached"].get(key, 0) >= low, (key, res["reached"])


@CUDA
@pytest.mark.parametrize("block", [32, 96])
def test_block_sizes(block):
    bad, _ = KC.kernel_against_torch(KC.cases(3, 70, do=False, ticks=8), snapshot=False, block=block)
    assert bad == [], bad[:6]


@CUDA
def test_a_block_past_the_kernels_limit_is_refused():
    """A block larger than the kernel's registers allow (maxThreadsPerBlock) is refused before the launch."""
    server, _ = KC.build(KC.cases(1, 4, ticks=1), "cuda", kernel="cuda", block=1024)
    limit = server._lib["info"]["max_threads_per_block"]
    assert 128 <= limit < 1024
    with pytest.raises(ValueError, match="threads a block"):
        server.tick()


@CUDA
def test_tiled_worlds():
    """40 worlds tiled to 2,000 in one park: kernel and torch path equal every tick (no snapshots)."""
    cs = KC.cases(1, 40, do=False, ticks=6)
    tiled = [dict(cs[k % len(cs)], id=k) for k in range(2000)]
    bad, vol = KC.kernel_against_torch(tiled, snapshot=False)
    assert bad == [], bad[:6]
    assert vol["queued"].get("LaunchMissile", 0) >= 1000


@CUDA
def test_the_time1_kernel_and_composed_parks_are_refused():
    """VecServer's own kernel (VecModesPark's) refuses a combat adapter; the combat kernel refuses VecComposedPark."""
    cs = KC.cases(1, 3, ticks=1)
    server, _ = KC.build(cs, "cuda")
    with pytest.raises(ValueError, match="CombatServer"):
        VS.VecServer(server.a, server.pilot, kernel="cuda")
    from exact_ballpark.vec_composed import VecComposedPark
    p = server.a.park
    q = VecComposedPark.__new__(VecComposedPark)
    q.__dict__.update(p.__dict__)
    with pytest.raises(ValueError, match="not a subclass"):
        VSC.CombatServer(VSC.CombatAdapter(q), server.pilot, kernel="cuda")


def _graph_case(seed: int, worlds: int, ticks: int) -> tuple:
    """A fixed schedule of pushes every tick (each pilot's ship GotoDirection then SetSpeedFraction; LaunchMissile;
    a Stop of a formation follower; a cloak and an uncloak of the formation's leader; SetBallVelocity; an Orbit) with
    random arguments and masks, as per-tick input tensors, on dynamical orientation's worlds."""
    import random
    r = random.Random(f"graph-combat-{seed}")
    cs = KC.cases(seed, worlds, do=True, ticks=1, empty=False)
    sched = [("GotoDirection", 1), ("SetSpeedFraction", 1), ("LaunchMissile", "missile"), ("Stop", "follower"),
             ("CloakBall", "leader"), ("UncloakBall", "leader"), ("SetBallVelocity", "any"), ("Orbit", "any")]
    W = len(cs)
    inputs = []
    for t in range(ticks):
        per = []
        for kind, who in sched:
            mask = torch.tensor([r.random() < 0.6 for _ in range(W)])
            ball, target, n = (torch.zeros(W, dtype=I64) for _ in range(3))
            f = torch.zeros(W, 4, dtype=F64)
            for wi, w in enumerate(cs):
                ids = [b["id"] for b in w["balls"]]
                b = {1: 1, "missile": r.choice(w["missiles"]), "follower": r.choice(w["formation"]["followers"]),
                     "leader": w["formation"]["leader"], "any": r.choice(ids)}[who]
                ball[wi] = b
                if kind == "GotoDirection":
                    f[wi, :3] = torch.tensor([r.uniform(-1, 1) for _ in range(3)], dtype=F64)
                elif kind == "SetSpeedFraction":
                    f[wi, 0] = r.choice([0.0, 1.0, r.random()])
                elif kind == "LaunchMissile":
                    target[wi] = r.choice(w["ships"] + [4242])
                    n[wi] = r.choice(w["ships"])
                    f[wi, 0], f[wi, 1] = r.choice([0, 1]), r.choice([0, 1])
                elif kind == "CloakBall":
                    n[wi] = r.choice([1, 2])
                    f[wi, 0] = r.uniform(100, 3000)
                elif kind == "SetBallVelocity":
                    f[wi, :3] = torch.tensor(r.choice([[0.0, 0.0, 0.0], [r.uniform(-200, 200) for _ in range(3)]]),
                                             dtype=F64)
                elif kind == "Orbit":
                    target[wi] = r.choice(ids)
                    f[wi, 0] = r.uniform(300, 5000)
            per.append((kind, mask, ball, target, f, n, torch.zeros(W, dtype=I64)))
        inputs.append(per)
    return cs, [k for k, _ in sched], inputs


@CUDA
def test_a_graph_of_kernel_ticks_equals_eager_kernel_ticks():
    """The fixed schedule ticked eagerly on the kernel and replayed from a ServerGraph on the kernel (dynamical
    orientation: the combat park's tick waits for no host): every tensor of the parks and the server's state, the
    routes and flags, every tick."""
    cs, sched, inputs = _graph_case(1, 64, 10)
    eager, _ = KC.build(cs, "cuda", kernel="cuda")
    graphed, _ = KC.build(cs, "cuda", kernel="cuda")
    g = VS.ServerGraph(graphed, sched)
    launched = 0
    for t, per in enumerate(inputs):
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
        assert KC.tensors_differing(eager.a.park, graphed.a.park) == [], t
        assert eager.a.park.current_time == graphed.a.park.current_time, t
        for k in KC.SERVER_STATE:
            assert KC._bits_equal(getattr(eager, k), getattr(graphed, k)), (t, k)
        assert torch.equal(eager.batch.route, out.route), t
        assert torch.equal(eager.batch.has_character, out.has_character), t
        assert torch.equal(eager.batch.gets_bubble, out.gets_bubble), t
        launched += int((eager.batch.slots[2].ok & (eager.batch.slots[2].row >= 0)).sum())
    assert launched >= 100 and int((eager.a.park.mode == 5).sum()) > 0


def _bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


ANGLES = [0.0, -0.0, 5e-324, -5e-324, 1e-310, -2.2e-308, 1e-17, 3.0, -3.0, math.pi, -math.pi, 6.2831853071795862,
          -6.2831853071795862, 6.283185307179585, 6.283185307179587, 12.566370614359172, -12.566370614359172, 9.42,
          -9.42, 100.0, -1e5, 1e20, -1e20, 1e300, -1e300, 1.7976931348623157e308, float("inf"), float("-inf"),
          float("nan")]


def crafted_angles_against_torch(fault: str | None = None) -> list:
    """The yaw and pitch's reduction at SetBallVelocity's snap (no dynamical orientation): each ship's old yaw and pitch
    set to crafted values (zeros of both signs, subnormals, multiples of 2 pi and their neighbours, huge, infinite,
    NaN), then a SetBallVelocity (zero: the direction toward mGoto; or random), on the kernel and the torch path: the
    tensors that differ."""
    import random
    r = random.Random("crafted-angles")
    cs = KC.cases(5, len(ANGLES) * len(ANGLES) // 4, do=False, ticks=1, empty=False)
    servers = [KC.build(cs, "cuda")[0], KC.build(cs, "cuda", kernel="cuda", fault=fault)[0]]
    R = servers[0].R
    yaw = torch.tensor([ANGLES[k % len(ANGLES)] for k in range(R)], dtype=F64)
    pitch = torch.tensor([ANGLES[(k // len(ANGLES) + 3 * k) % len(ANGLES)] for k in range(R)], dtype=F64)
    W = len(cs)
    vel = torch.tensor([r.choice([[0.0, 0.0, 0.0], [r.uniform(-300, 300) for _ in range(3)]]) for _ in range(W)],
                       dtype=F64)
    for srv in servers:
        p = srv.a.park
        p.yaw, p.pitch = yaw.to(p.dev), pitch.to(p.dev)
        p.old_yaw, p.old_pitch = (yaw * 3.0).to(p.dev), (pitch - 1.0).to(p.dev)
        for ship in (1, 2, 3):
            f = torch.zeros(W, 4, dtype=F64)
            f[:, :3] = vel.roll(ship, 0)
            srv.push("SetBallVelocity", torch.ones(W, dtype=BOOL).cuda(), torch.full((W,), ship, dtype=I64).cuda(),
                     f=f.cuda())
        srv.pre_tick(snapshot=False)
    return KC.tensors_differing(servers[0].a.park, servers[1].a.park)


@CUDA
def test_the_yaw_and_pitch_snap_reduces_crafted_angles_as_the_torch_path():
    assert crafted_angles_against_torch() == []


@CUDA
def test_the_kernel_resources():
    """The kernel as loaded: registers, no local spill beyond its stack frame, the threads a block it takes."""
    e = SK.load_library()
    assert e["info"]["registers"] <= 128 and e["info"]["max_threads_per_block"] >= 128
    assert e["manifest"]["ptx_check"]["ok"]


@CUDA
@pytest.mark.parametrize("fault", sorted(SK.FAULTS))
def test_planted_faults_are_caught(fault):
    """Each planted fault, built into its own library (checks recorded, not enforced), makes the kernel differ from
    the torch path on the development streams or on the crafted angles."""
    bad, _ = KC.kernel_against_torch(KC.cases(1, 60, do=False, ticks=12), fault=fault)
    if not bad:
        bad = crafted_angles_against_torch(fault)
    assert bad, f"the planted fault {fault} went unnoticed"

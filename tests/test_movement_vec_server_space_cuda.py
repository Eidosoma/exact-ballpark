# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer over the bubble partition as one CUDA kernel a tick (exact_ballpark.vec_server_space_cuda) against its torch path (exact_ballpark.vec_server_space.SpaceServer, held to the scalar server layer by
tests/test_movement_vec_server_space.py) and against the scalar server layer directly, bit for bit, with no tolerance,
every tick, over worlds of several bubbles on VecModesPark (tests/server_space_cases.py: flights and jumps across top
faces, the re-filing setters, interactive changes, cloaks and uncloaks, follows and orbits refused across bubbles and
stopped by the clean-up, warps to far top boxes).

Compared every tick, kernel against torch path: every tensor of the park (doubles by their bits, two NaNs equal) and
of the partition; the server's state (the stamps, the bubble update's time, the clean-up's count); the SpaceBatch (the
stamps and clocks, each slot's kind, queued flags and ball, its rows, target rows and local argument where queued, the
routes, a BUBBLE route's bubble, the notices, the bubble update's lists per row, the send flags and narrowcasts); and
the decoded updates (every client's list: stamps, events, argument types, AddBallsToPark streams byte for byte,
PackagedActions, wait_for_bubble, counts). The torch path's clean-up slots are as many as the most a world has that
tick; the kernel's are fixed (the world's N), the rest never queued and routed nowhere: checked.

On the CPU (nvcc, no GPU): the source scan, the build's PTX and host checks, the layout of the structures, the
refusals. On the GPU (skipped without one): the kernel against the torch path (development seed 1, held-out seed 7;
with snapshots, and in one launch), against the scalar layer (held-out seed 7), block sizes, tiled worlds, a CUDA
graph of kernel ticks against eager kernel ticks, the kernel's resources, seven planted faults (each must be caught).
"""

from __future__ import annotations

import logging
import re

import pytest
import torch

import server_cases as S
import server_space_cases as SS
from exact_ballpark import vec_orient_cuda as K
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_space as VSS
from exact_ballpark import vec_server_space_cuda as Q

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
logging.getLogger("exact_ballpark.server").setLevel(logging.CRITICAL)
HELD_OUT = 7                                                # as tests/test_movement_vec_server_space.py


# ==================================================================================================== on the CPU

def spills(ptxas: list) -> int:
    """The spill stores and loads ptxas reports, summed over every entry and target."""
    return sum(int(x) for ln in ptxas for x in re.findall(r"(\d+) bytes spill (?:stores|loads)", ln))


def test_the_sources_call_no_cuda_math():
    """No function of CUDA's math library in the new source, with or without a planted fault (the included sources
    are scanned by their own tests and by every build)."""
    text = Q.SOURCE.read_text()
    for level in [0, *Q.FAULTS.values()]:
        assert K.scan_source(Q.active_text(text, level)) == [], level
    assert len({Q.active_text(text, level) for level in [0, *Q.FAULTS.values()]}) == 1 + len(Q.FAULTS)


def test_the_build_pins_every_rounding_and_the_layout(tmp_path):
    """The library built (sm_90 and the PTX): every floating-point instruction of the PTX pinned to nearest, no fused
    instruction in the partition's host code, no spill; the ctypes structures as compiled."""
    m = Q.build(tmp_path, variant="sm90")
    assert m["ptx_check"]["ok"], m["ptx_check"]
    assert m["host_check"]["ok"] and m["host_check"]["functions"] > 0, m["host_check"]
    assert m["source_scan"] == []
    assert spills(m["ptxas"]) == 0, m["ptxas"]
    assert any("server_space_kernel" in ln for ln in m["ptxas"])
    layout = Q.layout_check(m, tmp_path)
    assert all(a == b for a, b in layout.values()), layout


def test_refusals():
    """The kernel runs a VecModesPark's layer on a CUDA device only."""
    cs = SS.cases(1, 2, "orient", 1)
    server, _ = SS.build(cs, "orient", "cpu")
    with pytest.raises(ValueError):
        Q.CudaSpaceServer(server.a)
    if torch.cuda.is_available():
        cs = SS.cases(1, 2, "modes", 1)
        ts, _ = SS.build(cs, "modes", "cuda")
        with pytest.raises(ValueError):
            Q.CudaSpaceServer(ts.a, cleanup_slots=1)


# ==================================================================================================== on the GPU

def build_kernel(cs: list, fault: str | None = None, block: int | None = None,
                 cleanup_slots: int | None = None) -> tuple:
    """(CudaSpaceServer, index) on its own VecModesPark, built as server_space_cases.build builds the torch path's."""
    from exact_ballpark.vec_modes import VecModesPark
    rows = [(w, b) for w in cs for b in w["balls"]]
    p = VecModesPark(len(rows), device="cuda")
    p.load([{"id": b["id"], "world": w["id"], "time": 0, "mass": b["mass"], "radius": b["radius"],
             "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
             "speed_fraction": b["sf"], "free": True, "massive": b["massive"], "interactive": b["interactive"],
             "cloaked": 0, "bubble": -1, "moribund": False} for w, b in rows])
    a = VSS.SpaceModesAdapter(p)
    index = {(w["id"], b["id"]): i for i, (w, b) in enumerate(rows)}
    pilots = [-1] * a.R
    for w in cs:
        for bid, pp in w["pilots"].items():
            pilots[index[(w["id"], bid)]] = pp
    ks = Q.CudaSpaceServer(a, pilots, fault=fault, block=block, cleanup_slots=cleanup_slots)
    return ks, index


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit for bit; doubles by their bits, two NaNs equal."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (torch.float64, torch.float32):
        both_nan = torch.isnan(a) & torch.isnan(b)
        ia = a.contiguous().view(torch.int64 if a.dtype == torch.float64 else torch.int32)
        ib = b.contiguous().view(torch.int64 if b.dtype == torch.float64 else torch.int32)
        return bool(((ia == ib) | both_nan).all())
    return torch.equal(a, b)


def state_diff(ts, ks) -> list:
    """The park's, the partition's and the server's tensors that differ."""
    bad = []
    pt, pk = vars(ts.a.park), vars(ks.a.park)
    for name, v in pt.items():
        if torch.is_tensor(v) and not _same(v, pk[name]):
            bad.append(f"park.{name}")
    for name, v in ts.space.st.items():
        if torch.is_tensor(v) and not _same(v, ks.space.st[name]):
            bad.append(f"space.{name}")
    for name in ("stamp_for_system", "_ad_time"):
        if not _same(getattr(ts, name), getattr(ks, name)):
            bad.append(name)
    if ts.cleanup_stops != ks.cleanup_stops:
        bad.append("cleanup_stops")
    return bad


def batch_diff(ts, ks) -> list:
    """The SpaceBatch fields that differ (the kernel's extra clean-up slots: never queued, routed nowhere)."""
    a, b = ts.batch, ks.batch
    bad = []
    Jt, Jk = len(a.slots), len(b.slots)
    if Jk < Jt:
        return [f"slots {Jt} > {Jk}"]
    for name in ("stamp", "clock", "has_character", "gets_bubble", "narrow"):
        if not _same(getattr(a, name), getattr(b, name)):
            bad.append(name)
    if not torch.equal(a.route, b.route[:, :Jt]):
        bad.append("route")
    if Jk > Jt and (bool((b.route[:, Jt:] != VS.NONE).any()) or any(bool(s.ok.any()) for s in b.slots[Jt:])):
        bad.append("extra clean-up slots used")
    on = a.route == VS.BUBBLE
    if not torch.equal(torch.where(on, a.rbubble, -9), torch.where(on, b.rbubble[:, :Jt], -9)):
        bad.append("rbubble")
    for j in range(Jt):
        sa, sb = a.slots[j], b.slots[j]
        ok = sa.ok
        if sa.kind != sb.kind or not torch.equal(sa.ok, sb.ok) or not torch.equal(sa.ball, sb.ball):
            bad.append(f"slot{j}")
            continue
        for name in ("row", "trow", "loc"):
            if not torch.equal(torch.where(ok, getattr(sa, name), -9), torch.where(ok, getattr(sb, name), -9)):
                bad.append(f"slot{j}.{name}")
    if [j for j, _, _ in a.notices] != [j for j, _, _ in b.notices]:
        bad.append("notice slots")
    else:
        for (j, ra, _), (_, rb, _) in zip(a.notices, b.notices):
            if not torch.equal(ra, rb):
                bad.append(f"notice{j}")
    for name in ("kind", "adds", "addn", "dels", "deln"):
        if not torch.equal(a.ad[name], b.ad[name]):
            bad.append(f"ad.{name}")
    return bad


def decoded(server) -> list:
    return S.canon([S._listify(x) for x in VSS.decode(server)])


def kernel_against_torch(cs: list, snapshot: bool = True, fault: str | None = None, block: int | None = None,
                         ticks: int | None = None, stop_at_first: bool = False) -> tuple:
    """Both servers on their own VecModesPark, the same pushes every tick: ([(tick, what differs)], volumes)."""
    T = len(cs[0]["ticks"]) if ticks is None else ticks
    ts, _ = SS.build(cs, "modes", "cuda")
    ks, _ = build_kernel(cs, fault=fault, block=block)
    bad, vol = [], {"ticks": 0, "worlds": len(cs), "ok_slots": 0, "bubble_routes": 0, "updates": 0}
    for t in range(T):
        for kind, *data in S.pushes(cs, t, "cuda"):
            ts.push(kind, *data)
            ks.push(kind, *data)
        for srv in (ts, ks):
            srv.pre_tick(snapshot=snapshot)
            srv.evolve()
            srv.post_tick()
        diff = state_diff(ts, ks) + batch_diff(ts, ks)
        if snapshot:
            raw = VSS.decode(ts)
            vol["updates"] += sum(len(ups) for per in raw for ups in per.values())   # every client's updates
            if S.canon([S._listify(x) for x in raw]) != decoded(ks):
                diff.append("decode")
        if diff:
            bad.append((t, diff[:6]))
            if stop_at_first:
                break
        vol["ticks"] += 1
        vol["ok_slots"] += int(sum(int(s.ok.sum()) for s in ts.batch.slots))
        vol["bubble_routes"] += int((ts.batch.route == VS.BUBBLE).sum())
    vol["cleanup_stops"] = ts.cleanup_stops
    if any(bool(f) for f in ts.space.st["flags"].tolist()):
        bad.append(("flagged", [w for w, f in enumerate(ts.space.st["flags"].tolist()) if f][:8]))
    return bad, vol


@CUDA
@pytest.mark.parametrize("seed", [1, HELD_OUT])
def test_the_kernel_equals_the_torch_path(seed):
    """Seeds 1 (development) and 7 (held out), 120 worlds x 16 ticks, with snapshots: every tensor, every tick, and the
    decoded updates; the clean-up's Stops occur."""
    bad, vol = kernel_against_torch(SS.cases(seed, 120, "modes", 16))
    assert not bad, bad[:3]
    assert vol["cleanup_stops"] > 0 and vol["bubble_routes"] > 0 and vol["updates"] > 0, vol


@CUDA
def test_the_kernel_in_one_launch_equals_the_torch_path():
    """Without snapshots the pre-tick is one launch: every tensor, every tick."""
    bad, vol = kernel_against_torch(SS.cases(2, 120, "modes", 16), snapshot=False)
    assert not bad, bad[:3]


def run_kernel_against_scalar(cs: list) -> dict:
    """server_space_cases.run with the kernel's server: the client update lists and ship rows that differ."""
    T = len(cs[0]["ticks"])
    scal = [S.ScalarWorld(w) for w in cs]
    server, index = build_kernel(cs)
    bad_sends, bad_rows, sends, compared = [], [], 0, 0
    for t in range(T):
        want = [sw.tick(w["ticks"][t]) for sw, w in zip(scal, cs)]
        for kind, *data in S.pushes(cs, t, server.dev):
            server.push(kind, *data)
        server.pre_tick(snapshot=True)
        server.evolve()
        server.post_tick()
        got = VSS.decode(server)
        flags = server.space.st["flags"].cpu().tolist()
        bub_rows = server.bubbles_of(torch.arange(server.a.R, device=server.dev)).cpu().tolist()
        pos, vel = server.a.park.pos.cpu(), server.a.park.vel.cpu()
        for wi, w in enumerate(cs):
            if flags[wi]:
                continue
            if S.canon(S._listify(want[wi])) != S.canon(S._listify(got[wi])):
                bad_sends.append((t, w["id"], S._first_diff(want[wi], got[wi])))
            sends += sum(len(v) for v in want[wi].values())
            for b in scal[wi].park.lane.balls.values():
                r = index[(w["id"], b.id)]
                compared += 1
                mine = (bub_rows[r], tuple(SS._bits(x) for x in pos[r].tolist()),
                        tuple(SS._bits(x) for x in vel[r].tolist()))
                theirs = (b.bubble, tuple(SS._bits(x) for x in b.pos), tuple(SS._bits(x) for x in b.vel))
                if mine != theirs:
                    bad_rows.append((t, w["id"], b.id, theirs[0], mine[0]))
    flagged = [wi for wi, f in enumerate(server.space.st["flags"].cpu().tolist()) if f]
    return {"sends": bad_sends, "rows": bad_rows, "updates": sends, "compared": compared, "flagged": flagged,
            "cleanup_stops": server.cleanup_stops}


@CUDA
def test_the_kernel_equals_the_scalar_server_layer():
    """The kernel's server directly against the scalar server layer, held-out seed 7, 120 worlds x 16 ticks: no client
    update list and no ship's bubble, position or velocity differs."""
    res = run_kernel_against_scalar(SS.cases(HELD_OUT, 120, "modes", 16))
    assert not res["flagged"], res["flagged"][:10]
    assert not res["sends"], [str(d)[:600] for d in res["sends"][:3]]
    assert not res["rows"], res["rows"][:5]
    assert res["updates"] > 1000 and res["cleanup_stops"] > 0, res


@CUDA
@pytest.mark.parametrize("block", [32, 96])
def test_block_sizes(block):
    bad, _ = kernel_against_torch(SS.cases(1, 70, "modes", 8), snapshot=False, block=block)
    assert not bad, bad[:3]


@CUDA
def test_a_block_past_the_kernels_limit_is_refused():
    cs = SS.cases(1, 4, "modes", 1)
    ks, _ = build_kernel(cs)
    limit = ks._lib["info"]["max_threads_per_block"]
    ks.block = limit + 32
    with pytest.raises(ValueError):
        ks.pre_tick(snapshot=False)


@CUDA
def test_tiled_worlds_each_equal_their_original():
    """40 worlds tiled to 1,000 in one park: after 10 ticks every copy's park rows, partition state and send flags
    equal its original's, and the originals equal the torch path's."""
    base = SS.cases(3, 40, "modes", 10)
    reps = 25
    cs = []
    for k in range(reps):
        for w in base:
            cs.append(dict(w, id=k * len(base) + w["id"]))
    ks, _ = build_kernel(cs)
    for t in range(10):
        for kind, *data in S.pushes(cs, t, "cuda"):
            ks.push(kind, *data)
        ks.pre_tick(snapshot=False)
        ks.evolve()
        ks.post_tick()
    n = len(base)
    rows = ks.a.rows_of_world
    p = ks.a.park
    for name in ("pos", "vel", "mode", "bubble", "cloaked", "interactive", "massive", "sensor"):
        t = getattr(p, name)
        orig = t[rows[:n].clamp(min=0)]
        for k in range(1, reps):
            assert _same(t[rows[k * n:(k + 1) * n].clamp(min=0)], orig), (name, k)
    for name, v in ks.space.st.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == len(cs) and name not in ("ev",):
            for k in range(1, reps):
                assert torch.equal(v[k * n:(k + 1) * n], v[:n]), (name, k)
    for name in ("has_character", "gets_bubble", "narrow"):
        t = getattr(ks.batch, name)
        orig = t[rows[:n].clamp(min=0)]
        for k in range(1, reps):
            assert torch.equal(t[rows[k * n:(k + 1) * n].clamp(min=0)], orig), (name, k)
    bad, _ = kernel_against_torch(base, snapshot=False)
    assert not bad, bad[:3]


def _schedule_inputs(cs: list, t: int, W: int, dev) -> list:
    """A fixed schedule's arguments for tick t: each world's pilot 0 flies GotoDirection and SetSpeedFraction; ship 1
    stops every third tick; the special ship cloaks at tick 2 and uncloaks at tick 5; ship 0 jumps across its row's
    top face (SetBallPosition) at tick 4; ship 1 orbits ship 0 at tick 6."""
    ball0 = torch.tensor([w["balls"][0]["id"] for w in cs], dtype=torch.int64)
    ball1 = torch.tensor([w["balls"][1]["id"] for w in cs], dtype=torch.int64)
    special = torch.tensor([w["special"] for w in cs], dtype=torch.int64)
    on = torch.ones(W, dtype=torch.bool)
    z = torch.zeros(W, dtype=torch.int64)
    zf = torch.zeros(W, 4, dtype=torch.float64)
    ang = torch.arange(W, dtype=torch.float64) * 0.37 + 0.11 * t
    goto = zf.clone()
    goto[:, 0], goto[:, 1], goto[:, 2] = torch.cos(ang), torch.sin(ang), 0.25
    sf = zf.clone()
    sf[:, 0] = 0.25 + 0.5 * ((t + torch.arange(W)) % 3).to(torch.float64) / 2.0
    jump = zf.clone()
    jump[:, 0] = SS.TOP + 3000.0 * (1 + (torch.arange(W) % 4))
    jump[:, 1] = 0.3 * SS.TOP
    jump[:, 2] = -0.4 * SS.TOP
    cloak = zf.clone()
    cloak[:, 0] = 5000.0
    orbit = zf.clone()
    orbit[:, 0] = 2500.0
    out = [dict(mask=on, ball=ball0, target=z, f=goto, n=z, loc=z),
           dict(mask=on, ball=ball0, target=z, f=sf, n=z, loc=z),
           dict(mask=on & (t % 3 == 0), ball=ball1, target=z, f=zf, n=z, loc=z),
           dict(mask=on & (t == 2), ball=special, target=z, f=cloak, n=z + 1, loc=z),
           dict(mask=on & (t == 5), ball=special, target=z, f=zf, n=z, loc=z),
           dict(mask=on & (t == 4), ball=ball0, target=z, f=jump, n=z, loc=z),
           dict(mask=on & (t == 6), ball=ball1, target=ball0, f=orbit, n=z, loc=z)]
    return [{k: v.to(dev) for k, v in d.items()} for d in out]


SCHEDULE = ["GotoDirection", "SetSpeedFraction", "Stop", "CloakBall", "UncloakBall", "SetBallPosition", "Orbit"]


@CUDA
def test_a_graph_of_kernel_ticks_equals_eager_kernel_ticks():
    """A fixed schedule (SCHEDULE) over 64 worlds, 10 ticks: the captured tick's every tensor (the park, the partition,
    the server, the batch) equals eager kernel ticks with the same pushes."""
    cs = SS.cases(4, 64, "modes", 1)
    eager, _ = build_kernel(cs)
    graphed, _ = build_kernel(cs)
    g = Q.SpaceServerGraph(graphed, SCHEDULE)
    W, dev = eager.W, eager.dev
    for t in range(10):
        args = _schedule_inputs(cs, t, W, dev)
        for k, a in zip(SCHEDULE, args):
            eager.push(k, a["mask"], a["ball"], a["target"], a["f"], a["n"], a["loc"])
        eager.pre_tick(snapshot=False)
        eager.evolve()
        eager.post_tick()
        for inp, a in zip(g.inputs, args):
            for k in inp:
                inp[k].copy_(a[k])
        b = g.tick()
        diff = [n for n in ("pos", "vel", "mode", "bubble", "cloaked", "massive", "sensor", "follow_id")
                if not _same(getattr(eager.a.park, n), getattr(graphed.a.park, n))]
        diff += [n for n, v in eager.space.st.items() if torch.is_tensor(v) and not _same(v, graphed.space.st[n])]
        diff += [n for n in ("stamp_for_system", "_ad_time", "cleanup_total")
                 if not _same(getattr(eager, n), getattr(graphed, n))]
        e = eager.batch
        for n in ("stamp", "clock", "route", "rbubble", "has_character", "gets_bubble", "narrow"):
            if not _same(getattr(e, n), getattr(b, n)):
                diff.append(f"batch.{n}")
        for n in ("kind", "adds", "addn", "dels", "deln"):
            if not torch.equal(e.ad[n], b.ad[n]):
                diff.append(f"ad.{n}")
        for j, (sa, sb) in enumerate(zip(e.slots, b.slots)):
            if not (torch.equal(sa.ok, sb.ok) and torch.equal(sa.row, sb.row) and torch.equal(sa.ball, sb.ball)):
                diff.append(f"slot{j}")
        assert not diff, (t, diff[:6])
    assert int(eager.batch.route.ne(VS.NONE).sum()) > 0


@CUDA
def test_the_kernel_resources():
    """The kernel's registers, local memory and threads a block on this card (recorded), and its build without
    spills."""
    with torch.cuda.device(0):
        e = Q.load_library()
    info = e["info"]
    assert info["max_threads_per_block"] >= 128 and info["registers"] > 0, info
    assert spills(e["manifest"]["ptxas"]) == 0, e["manifest"]["ptxas"]
    print({"info": info, "library": e["manifest"]["library"]})


@CUDA
@pytest.mark.parametrize("fault", sorted(Q.FAULTS))
def test_planted_faults_are_caught(fault):
    """Each planted fault's library differs from the torch path somewhere in seed 1's worlds (the unplanted kernel
    equals it there: test_the_kernel_equals_the_torch_path)."""
    bad, _ = kernel_against_torch(SS.cases(1, 120, "modes", 16), fault=fault, stop_at_first=True)
    assert bad, fault

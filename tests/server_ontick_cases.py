# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""OnTick's per-world intervals in Destiny's one-bubble server layers (after the
tests/ontick_cases.py and the tests/parks_ontick_cases.py): each world driven by its own simulation clock,
generated from a seed, and the two runners compared bit for bit after every call: the scalar server layer (the
tests/server_cases.py ScalarWorld, or the tests/server_combat_cases.py ScalarWorld with its formation: Actions,
Ticker, BubbleUpdater and ParkUpdateBatcher on a master DestinyPark over FullBallpark), its park's OnTick
(exact_ballpark.full.FullBallpark.on_tick, Ballpark.cpp:219-330, TIME-702 to TIME-705) calling the Ticker's tick at
DoPreTick and send_batch at DoPostTick, as Destiny's server hangs them (_baseticker.py:23-26,
_parkupdatebatcher.py:16-18), against exact_ballpark.vec_server.VecServer.on_tick (one call for every world, each
world's clock its own) over VecOrientPark, VecCollidePark or VecModesPark, and the combat server's
(exact_ballpark.vec_server_combat.CombatServer) over VecCombatPark, on their torch paths and kernels. Compared after every
call: each world's sends in order (one list a DoPostTick, the canonical form), every ball's compared fields, and
the clocks (mTime, mFirstTime, mCurrentTime). And the server kernels against the torch path under OnTick, every tensor
after every call (`kernel_against_torch`).

**The clocks**: a world starts at 0, under one interval, one, or 2 to 20 intervals (its first call
holding an interval evolves once, without the tick events and with its queue kept); each call advances it by nothing,
under one interval (too early: no tick, the queue kept), one interval (four times as often) and a little more (the
remainder carried), two or three intervals, or more than five (the master's resynchronisation). **The calls** between
two OnTicks are the earlier generators' (the tests/server_cases.py for the three parks, the
tests/server_combat_cases.py and the tests/server_combat_kernel_cases.py for the combat park), one tick's calls a
call; at a call after one with calls, the previous call's last call is now and then pushed again first (TIME-710's
duplicate test across a queue that a world without a tick kept). A world is compared until it leaves the layer's
domain (a ball out of bubble 0, the combat park's flags, a ball removed by BringOutTheDeads): from that call on it is
counted, not compared."""

from __future__ import annotations

import random
from collections import Counter

import torch

import server_cases as S
import server_combat_cases as SC
import server_combat_kernel_cases as SK
import server_space_cases as SS
from exact_ballpark import server as srv
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_space as VSS

I64 = torch.int64
INTERVAL = 1000 * 10000                                  # the tick interval (ms) x 10,000: 100 ns units (Be::Time)
STEPS = (0.0, 0.4, 0.9999, 1.0, 1.0, 1.0, 1.0, 1.25, 1.6, 2.0, 2.3, 3.0, 4.2, 5.5, 6.0, 7.4)
REPEAT = 0.3                                             # a call's chance to push the previous call's last call first


# =========================================================================================== the generators

def _step(r: random.Random) -> int:
    """One call's advance of a world's clock (100 ns units)."""
    k = r.choice(STEPS)
    return int(k * INTERVAL) + (r.randint(0, 999) if k else 0)


def _clock(r: random.Random, calls: int) -> list:
    """A world's simulation time at each of its calls: a start, then each call's own advance."""
    k = r.random()
    if k < 0.4:
        t = 0
    elif k < 0.6:
        t = r.randint(1, INTERVAL - 1)
    elif k < 0.75:
        t = INTERVAL
    else:
        t = r.randint(2, 20) * INTERVAL + r.randint(0, 999)
    out = []
    for _ in range(calls):
        t += _step(r)
        out.append(t)
    return out


def _repeatable(c: tuple, w: dict) -> bool:
    """Whether a call may be pushed again at the next call: not LaunchMissile (each missile launched once), no NaN
    argument (one NaN object twice is Python's identity shortcut), and not a call the generators keep from a special
    ship that may have warped (tests/server_cases.py's _INTERRUPTS)."""
    name, args = c
    if name == "launch_missile":
        return False
    if any(isinstance(v, float) and v != v for v in args):
        return False
    return not (args and args[0] == w.get("special") and name in S._INTERRUPTS)


def _repeat(r: random.Random, w: dict, calls: list) -> None:
    prev = w["ticks"][-1] if w["ticks"] else []
    if prev and r.random() < REPEAT and _repeatable(prev[-1], w):
        calls.insert(0, (prev[-1][0], tuple(prev[-1][1])))
        w["repeats"].append(len(w["ticks"]))


def server_cases(seed: int, n: int, park: str, do: bool = False, calls: int = 14) -> list:
    """n worlds for a park ('modes', 'orient', 'collide'; the generator), each with `calls` OnTick calls: the
    Actions calls before each (`ticks`) and the simulation time it passes (`clock`)."""
    r = random.Random(f"server-ontick-{park}-{int(do)}-{seed}")
    out = []
    for wid in range(n):
        w = S._world(r, wid, park, do)
        w["ticks"], w["repeats"] = [], []
        for _ in range(calls):
            cl = S._tick_calls(r, w, park, do)
            _repeat(r, w, cl)
            w["ticks"].append(cl)
        w.pop("warped", None)
        w["clock"] = _clock(r, calls)
        out.append(w)
    return out


def combat_cases(seed: int, n: int, device: str = "cpu", calls: int = 14) -> list:
    """n combat worlds (the generator: the scalar comparisons' worlds), each with `calls` OnTick calls."""
    r = random.Random(f"server-ontick-combat-{device}-{seed}")
    out = []
    for wid in range(n):
        w = SC._world(r, wid, device)
        w["launched"], w["repeats"] = set(), []
        for _ in range(calls):
            cl = []
            for _ in range(r.choice([0, 1, 1, 2, 3, 4])):
                c = SC._call(r, w, device)
                cl.append(c)
                if r.random() < 0.08:
                    cl.append((c[0], tuple(c[1])))           # the same call twice in a row: the second dropped
            _repeat(r, w, cl)
            w["ticks"].append(cl)
        w.pop("launched")
        w["clock"] = _clock(r, calls)
        out.append(w)
    return out


def combat_kernel_cases(seed: int, n: int, do: bool = False, calls: int = 14) -> list:
    """n combat worlds of the kernel generator (empty rows, every call kind; outside the park's domain too, as
    the kernel against the torch path holds there), each with `calls` OnTick calls."""
    r = random.Random(f"server-ontick-combat-kernel-{int(do)}-{seed}")
    out = []
    for wid in range(n):
        w = SK._world(r, wid, do, True)
        w["repeats"] = []
        for _ in range(calls):
            cl = []
            for _ in range(r.choice([0, 1, 2, 2, 3, 4, 5])):
                c = SK._call(r, w, do)
                cl.append(c)
                if r.random() < 0.08:
                    cl.append((c[0], tuple(c[1])))
            _repeat(r, w, cl)
            w["ticks"].append(cl)
        w["clock"] = _clock(r, calls)
        out.append(w)
    return out


def space_cases(seed: int, n: int, park: str = "orient", calls: int = 14) -> list:
    """n worlds of several bubbles (the tests/server_space_cases.py generator) for SpaceServer over VecOrientPark
    (the CPU) or VecModesPark (a CUDA device; follows, orbits and warps too), each with `calls` OnTick calls."""
    r = random.Random(f"server-ontick-space-{park}-{seed}")
    out = []
    for wid in range(n):
        w = SS._world(r, wid, park)
        w["ticks"], w["repeats"] = [], []
        for _ in range(calls):
            cl = SS._tick_calls(r, w, park)
            _repeat(r, w, cl)
            w["ticks"].append(cl)
        w.pop("warped", None)
        w["clock"] = _clock(r, calls)
        out.append(w)
    return out


# ============================================================================================ the scalar side

class _OnTick:
    """The scalar server world driven through its park's OnTick: the Ticker at DoPreTick, send_batch at DoPostTick;
    each DoPostTick's sends kept apart (the OnTickWorld)."""

    def _hook(self) -> None:
        lane = self.park.lane
        self.sends: list = []
        lane.pre_tick_handlers.append(lambda time: self.ticker.tick())
        lane.post_tick_handlers.append(self._post)

    def _post(self, time) -> None:
        self.batcher.send_batch()
        self.sends.append(S.per_client(self.network.take()))

    def call(self, calls: list, sim_time: int) -> list:
        for name, args in calls:
            getattr(self.actions, name)(*args)
        self.sends = []
        self.park.lane.on_tick(sim_time)
        return self.sends


class ServerWorld(_OnTick, S.ScalarWorld):
    def __init__(self, w: dict):
        S.ScalarWorld.__init__(self, w)
        self._hook()


class CombatWorld(_OnTick, SC.ScalarWorld):
    def __init__(self, w: dict):
        SC.ScalarWorld.__init__(self, w)
        self._hook()


# =========================================================================================== the comparisons

def build(cs: list, park: str, device: str, do: bool = False, kernel: str = "torch", server_kernel: str = "torch",
          contact_slots: int | None = None, fault: str | None = None) -> tuple:
    """(server, index) as the earlier runners build them: tests/server_cases.build for the three
    parks, tests/server_combat_kernel_cases.build for the combat park."""
    if park == "combat":
        return SK.build(cs, device, kernel=server_kernel, fault=fault)
    return S.build(cs, park, device, do=do, kernel=kernel, contact_slots=contact_slots, server_kernel=server_kernel)


def _pushes(cs: list, t: int, park: str, device) -> list:
    return SC.pushes(cs, t, device) if park == "combat" else S.pushes(cs, t, device)


def sim_times(cs: list, t: int) -> torch.Tensor:
    return torch.tensor([w["clock"][t] for w in cs], dtype=I64)


def run(cs: list, park: str, device: str, do: bool = False, kernel: str = "torch", server_kernel: str = "torch",
        contact_slots: int | None = None, calls: int | None = None, server=None) -> dict:
    """Both sides on the worlds cs, call by call (`server`: a server built by the caller, e.g. with a planted fault):
    {"sends": [(call, world, scalar DoPostTicks, batched)] whose sends differ, "rows": [(call, world, ball, fields)],
    "clocks": [(call, world, scalar (mTime, mFirstTime, mCurrentTime), batched)], "compared", "updates", "out":
    {world: the call it left the domain}, "coverage", "server"}."""
    T = len(cs[0]["clock"]) if calls is None else int(calls)
    old = srv.is_dynamical_orientation_enabled()
    try:
        scal = [CombatWorld(w) if park == "combat" else ServerWorld(w) for w in cs]
        if server is None:
            server, index = build(cs, park, device, do=do, kernel=kernel, server_kernel=server_kernel,
                                  contact_slots=contact_slots)
        else:
            server, index = server
        A = server.a
        bad_sends, bad_rows, bad_clocks, out = [], [], [], {}
        compared = updates = 0
        cov = Counter()
        for t in range(T):
            want = [sw.call(w["ticks"][t], w["clock"][t]) for sw, w in zip(scal, cs)]
            for kind, *data in _pushes(cs, t, park, server.dev):
                server.push(kind, *data)
            batches = server.on_tick(sim_times(cs, t))
            decoded = [(b.ticked.cpu().tolist(), VS.decode(server, b)) for b in batches]
            for k, v in server.tick_counts.items():
                cov[k] += v
            for ticked, _ in decoded:
                if not all(ticked):
                    cov["masked rounds"] += 1
                    cov["worlds left out of a masked round"] += ticked.count(False)
                else:
                    cov["rounds of every world"] += 1
            flags = A.park.world_flags() if park == "combat" else [0] * len(cs)
            removed = server.removed.cpu().tolist() if getattr(server, "removed", None) is not None else [0] * len(cs)
            if park == "combat":
                p = A.park
                cpu = type("P", (), {k: v.cpu() for k, v in vars(p).items() if torch.is_tensor(v)})
            else:
                got_rows = S.batched_rows(server, cs, index, park, do)
            clk, tm, ft = (A.clock().cpu().tolist(), server.time.cpu().tolist(), server.first_time.cpu().tolist())
            for wi, w in enumerate(cs):
                lane = scal[wi].park.lane
                if wi not in out and (flags[wi] or removed[wi] or any(b.bubble != 0 for b in lane.balls.values())):
                    out[wi] = t
                if wi in out:
                    continue
                got = [d[wi] for ticked, d in decoded if ticked[wi]]
                if any(d[wi] for ticked, d in decoded if not ticked[wi]):
                    bad_sends.append((t, w["id"], "sent without a tick"))
                if len(want[wi]) != len(got) or any(S.canon(S._listify(a)) != S.canon(S._listify(b))
                                                    for a, b in zip(want[wi], got)):
                    bad_sends.append((t, w["id"], len(want[wi]), len(got)))
                updates += sum(sum(len(v) for v in x.values()) for x in want[wi])
                if w["ticks"][t] and not want[wi]:
                    cov["calls kept by a world without a tick"] += len(w["ticks"][t])
                if t in w["repeats"]:
                    cov["repeats of the last call"] += 1
                    if kept_before(scal[wi]):
                        cov["repeats across a kept queue"] += 1
                if park == "combat":
                    for b in lane.balls.values():
                        r = index[(w["id"], b.id)]
                        compared += 1
                        mine, theirs = SC.ball_state(cpu, r), SC.scalar_ball_state(b)
                        if mine != theirs:
                            bad_rows.append((t, w["id"], b.id, "state"))
                else:
                    for bid, row in S.scalar_rows(scal[wi], park, do).items():
                        compared += 1
                        if not S.same_rows(row, got_rows[(w["id"], bid)]):
                            bad_rows.append((t, w["id"], bid, "fields"))
                theirs = (lane.time, bool(lane.first_time), lane.current_time)
                mine = (tm[wi], bool(ft[wi]), clk[wi])
                if theirs != mine:
                    bad_clocks.append((t, w["id"], theirs, mine))
            for wi, sw in enumerate(scal):
                sw.ticked_last = bool(want[wi])
        return {"sends": bad_sends, "rows": bad_rows, "clocks": bad_clocks, "compared": compared, "updates": updates,
                "out": out, "coverage": dict(cov), "calls": T, "worlds": len(cs), "server": server}
    finally:
        if old:
            srv.enable_dynamical_orientation()
        else:
            srv.disable_dynamical_orientation()


def kept_before(sw) -> bool:
    """Whether the world's previous call ran no pre-tick (its queue, holding the call pushed again, was kept)."""
    return not getattr(sw, "ticked_last", True)


def run_space(cs: list, park: str, device: str, calls: int | None = None, server=None) -> dict:
    """The partition layer's server (exact_ballpark.vec_server_space.SpaceServer, the tests/server_space_cases.build:
    VecOrientPark or VecModesPark under vec_space's layer) against the scalar server layer, call by call, as `run`:
    each world's sends in order (vec_server_space.decode), every ship's bubble, position, velocity and mode, and the
    clocks; a world flagged by the layer is counted from that call on, not compared."""
    T = len(cs[0]["clock"]) if calls is None else int(calls)
    old = srv.is_dynamical_orientation_enabled()
    try:
        scal = [ServerWorld(w) for w in cs]
        server, index = SS.build(cs, park, device) if server is None else server
        A = server.a
        bad_sends, bad_rows, bad_clocks, out = [], [], [], {}
        compared = updates = 0
        cov = Counter()
        for t in range(T):
            want = [sw.call(w["ticks"][t], w["clock"][t]) for sw, w in zip(scal, cs)]
            for kind, *data in S.pushes(cs, t, server.dev):
                server.push(kind, *data)
            batches = server.on_tick(sim_times(cs, t))
            decoded = [(b.ticked.cpu().tolist(), VSS.decode(server, b)) for b in batches]
            for k, v in server.tick_counts.items():
                cov[k] += v
            for ticked, _ in decoded:
                if not all(ticked):
                    cov["masked rounds"] += 1
                    cov["worlds left out of a masked round"] += ticked.count(False)
                else:
                    cov["rounds of every world"] += 1
            flags = server.space.st["flags"].cpu().tolist()
            bub = server.bubbles_of(torch.arange(A.R, device=server.dev)).cpu().tolist()
            pos, vel, mode = A.park.pos.cpu(), A.park.vel.cpu(), A.park.mode.cpu().tolist()
            clk, tm, ft = (A.clock().cpu().tolist(), server.time.cpu().tolist(), server.first_time.cpu().tolist())
            for wi, w in enumerate(cs):
                lane = scal[wi].park.lane
                if wi not in out and flags[wi]:
                    out[wi] = t
                if wi in out:
                    continue
                got = [d[wi] for ticked, d in decoded if ticked[wi]]
                if any(d[wi] for ticked, d in decoded if not ticked[wi]):
                    bad_sends.append((t, w["id"], "sent without a tick"))
                if len(want[wi]) != len(got) or any(S.canon(S._listify(a)) != S.canon(S._listify(b))
                                                    for a, b in zip(want[wi], got)):
                    bad_sends.append((t, w["id"], len(want[wi]), len(got)))
                updates += sum(sum(len(v) for v in x.values()) for x in want[wi])
                for x in want[wi]:
                    SS.coverage(x, cov)
                if w["ticks"][t] and not want[wi]:
                    cov["calls kept by a world without a tick"] += len(w["ticks"][t])
                if t in w["repeats"]:
                    cov["repeats of the last call"] += 1
                    if kept_before(scal[wi]):
                        cov["repeats across a kept queue"] += 1
                for b in lane.balls.values():
                    r = index[(w["id"], b.id)]
                    compared += 1
                    mine = (bub[r], tuple(SS._bits(x) for x in pos[r].tolist()),
                            tuple(SS._bits(x) for x in vel[r].tolist()), int(mode[r]))
                    theirs = (b.bubble, tuple(SS._bits(x) for x in b.pos), tuple(SS._bits(x) for x in b.vel),
                              int(b.mode))
                    if mine != theirs:
                        bad_rows.append((t, w["id"], b.id, "bubble/pos/vel/mode"))
                if len({b.bubble for b in lane.balls.values()}) > 1:
                    cov["world-calls with several bubbles"] += 1
                theirs = (lane.time, bool(lane.first_time), lane.current_time)
                mine = (tm[wi], bool(ft[wi]), clk[wi])
                if theirs != mine:
                    bad_clocks.append((t, w["id"], theirs, mine))
            for wi, sw in enumerate(scal):
                sw.ticked_last = bool(want[wi])
        cov["follower clean-up stops"] = server.cleanup_stops
        return {"sends": bad_sends, "rows": bad_rows, "clocks": bad_clocks, "compared": compared, "updates": updates,
                "out": out, "coverage": dict(cov), "calls": T, "worlds": len(cs), "server": server}
    finally:
        if old:
            srv.enable_dynamical_orientation()
        else:
            srv.disable_dynamical_orientation()


# ================================================================================ the kernels against torch

SERVER_STATE = ("stamp_for_system", "settled", "dead", "_bubble_time", "_fresh", "_listed", "pilot", "time",
                "first_time")


def _batch_view(b) -> dict:
    """A Batch on the host as Destiny's queue holds it: per world, its queued calls in queue order (kind, ball, target,
    float and int arguments, row, target's row, local argument, route, an UncloakBall's notice recipients); the
    stamps, clocks, send flags, the bubble update's lists and the ticked worlds. (Under OnTick's mask a kernel path
    keeps a world's pushed calls untested until that world's pre-tick, so its batch can hold slots the torch path
    dropped at the push as duplicates: never queued, they are not in this view; their slot indices are a form.)"""
    W = int(b.stamp.shape[0])
    col = lambda attr: [getattr(s, attr).tolist() for s in b.slots]            # noqa: E731
    ok, ball, target, f, n = col("ok"), col("ball"), col("target"), col("f"), col("n")
    row, trow, loc = col("row"), col("trow"), col("loc")
    route = b.route.tolist()
    notice = {j: tuple(i for i, x in enumerate(rec.tolist()) if x) for j, rec, _ in b.notices}
    queued = []
    for w in range(W):
        queued.append([(s.kind, ball[j][w], target[j][w], tuple(SK.bits(x) for x in f[j][w]), n[j][w], row[j][w],
                        trow[j][w], loc[j][w], route[w][j], notice.get(j))
                       for j, s in enumerate(b.slots) if ok[j][w]])
    return {"stamp": b.stamp.tolist(), "clock": b.clock.tolist(), "has": b.has_character.tolist(),
            "gets": b.gets_bubble.tolist(), "fresh": b.additions[0].tolist(), "listed": b.additions[1].tolist(),
            "ticked": b.ticked.tolist(), "queued": queued}


def kernel_against_torch(cs: list, park: str, do: bool = False, calls: int | None = None) -> tuple:
    """Both servers on their own park (CUDA), the server layer's kernel (`server_kernel="cuda"`: for
    VecModesPark and for VecCombatPark) against its torch path, the same pushes and the same OnTick calls:
    ([(call, what differs)], volumes). Compared after every call: every tensor of the two parks and adapters, the
    park's time, the servers' state (the OnTick clocks included), the queue each kept (each call's queued flag, as
    Destiny's Actions holds it), the batches' count and every field of each, the decoded updates."""
    T = len(cs[0]["clock"]) if calls is None else int(calls)
    old = srv.is_dynamical_orientation_enabled()
    try:
        ts, _ = build(cs, park, "cuda", do=do)
        ks, _ = build(cs, park, "cuda", do=do, server_kernel="cuda")
        bad, vol = [], Counter()
        for t in range(T):
            for kind, *data in _pushes(cs, t, park, "cuda"):
                ts.push(kind, *data)
                ks.push(kind, *data)
            bt = ts.on_tick(sim_times(cs, t))
            bk = ks.on_tick(sim_times(cs, t))
            for name in SK.tensors_differing(ts.a.park, ks.a.park):
                bad.append((t, "park", name))
            for name in SK.tensors_differing(ts.a, ks.a):
                bad.append((t, "adapter", name))
            if ts.a.park.current_time != ks.a.park.current_time:
                bad.append((t, "current_time"))
            for k in SERVER_STATE:
                if not SK._bits_equal(getattr(ts, k), getattr(ks, k)):
                    bad.append((t, "server", k))
            qa = [(s.kind, q.tolist()) for s, q in zip(ts.slots, ts._queued())]
            qb = [(s.kind, q.tolist()) for s, q in zip(ks.slots, ks._queued())]
            if [x for x in qa if any(x[1])] != [x for x in qb if any(x[1])]:
                bad.append((t, "kept queue"))
            if len(bt) != len(bk):
                bad.append((t, "batches", len(bt), len(bk)))
            for i, (a, b) in enumerate(zip(bt, bk)):
                va, vb = _batch_view(a), _batch_view(b)
                diff = [k for k in sorted(set(va) | set(vb)) if va.get(k) != vb.get(k)]
                if diff:
                    bad.append((t, "batch", i, diff[:5]))
                da, db = VS.decode(ts, a), VS.decode(ks, b)
                if S.canon([S._listify(x) for x in da]) != S.canon([S._listify(x) for x in db]):
                    bad.append((t, "decode", i))
                vol["decoded updates"] += sum(len(v) for per in da for v in per.values())
                if not all(a.ticked.tolist()):
                    vol["masked rounds"] += 1
            for k, v in ts.tick_counts.items():
                vol[k] += v
            vol["kept calls"] += sum(int(q.sum()) for q in ts._queued())
            vol["calls"] += 1
        return bad, dict(vol)
    finally:
        if old:
            srv.enable_dynamical_orientation()
        else:
            srv.disable_dynamical_orientation()

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The server-layer scenarios: the earlier worlds and command streams (the one-bubble worlds,
tests/server_cases.py; the and the combat worlds, tests/server_combat_cases.py and
tests/server_combat_kernel_cases.py; the worlds of several bubbles, tests/server_space_cases.py), drawn with their
own generators under new seeds, with Destiny's SetBallTroll and SetBallFree Actions calls (_actions.py:133-148, 525-538)
put among each tick's calls (the delay below 1 now and then, the flag 0, 1 or 2, the same call twice in a row), and,
on the worlds, members the Python SetBoid takes out of a follow mode between ticks (`boids`: set on the scalar park
and on the batched one alike, as a game calls PySetBoid between server ticks). Never on a world's anchor ship (its
bubble kept alive: a petrified troll is not interactive) for a troll, never on the ship that warps (its warp is never
interrupted, as the earlier generators keep it).

The runners compare the batched layer with the scalar one (exact_ballpark.server over FullBallpark) tick by tick: every
client's update list, every ball's state (the fields with the free flag, and each ball's members by Destiny's own
test: follow_ptr and the follow id name it, the reference's set less its stale entries), and between the calls and the pre-tick the pending queries has_pending_set_free and has_pending_set_not_free
(the answer and its type: Destiny returns the int 0 when nothing is pending); and a kernel path with its torch path,
every tensor of the park and the adapter, the server's state, the Batch and the decoded updates. Frozen before any
held-out seed is drawn.
"""

from __future__ import annotations

import random
from collections import Counter

import torch

import server_cases as sc
from exact_ballpark import server as srv
from exact_ballpark import vec_server as VS

I64, BOOL = torch.int64, torch.bool
FI = sc.FIELDS_MODES.index("followers")


# ================================================================================================= the scenarios

def inject(cs: list, tag: str, seed: int, anchor_of, special_of, ids_of, troll: bool = True, p: float = 0.6) -> list:
    """SetBallTroll and SetBallFree calls put among each tick's calls of the worlds cs (in place; returned)."""
    r = random.Random(f"compose3-{tag}-{seed}")
    for w in cs:
        anchor, special = anchor_of(w), special_of(w)
        ids = [i for i in ids_of(w) if i != special]
        for t, calls in enumerate(w["ticks"]):
            if not ids or r.random() > p:
                continue
            new = list(calls)
            for _ in range(r.choice([1, 1, 2])):
                s = r.choice(ids)
                if troll and s != anchor and r.random() < 0.45:
                    c = ("set_ball_troll", (s, r.choice([-3, 0, 1, 2, 3, 5, 12])))
                else:
                    c = ("set_ball_free", (s, r.choice([0, 0, 1, 2])))
                k = r.randint(0, len(new))
                new.insert(k, c)
                if r.random() < 0.15:
                    new.insert(k + 1, (c[0], tuple(c[1])))    # the same call twice in a row: the second dropped
            w["ticks"][t] = new
    return cs


def modes_cases(seed: int, n: int = 40, do: bool = False, ticks: int = 14, boids: bool = True) -> list:
    """the one-bubble worlds (server_cases.cases, park "modes", under seed 100 + seed) with the calls and, with
    `boids`, ships set to BOID between ticks (any ship but the one that warps)."""
    cs = sc.cases(100 + seed, n, "modes", do=do, ticks=ticks)
    inject(cs, f"modes-{int(do)}", seed, lambda w: w["anchor"], lambda w: w["special"],
           lambda w: [b["id"] for b in w["balls"]])
    _boids(cs, f"modes-{int(do)}", seed, lambda w: [b["id"] for b in w["balls"] if b["id"] != w["special"]]
           if boids else [])
    return cs


def combat_cases(seed: int, n: int = 30, ticks: int = 12, device: str = "cuda", kernel_worlds: bool = False) -> list:
    """the combat worlds (server_combat_cases.cases for `device`, under seed 100 + seed), or the (with
    empty rows: server_combat_kernel_cases.cases), with the calls on their ships (ship 1 the anchor)."""
    if kernel_worlds:
        import server_combat_kernel_cases as KC
        cs = KC.cases(100 + seed, n, ticks=ticks)
    else:
        import server_combat_cases as SC
        cs = SC.cases(100 + seed, n, device, ticks=ticks)
    inject(cs, f"combat-{device}-{int(kernel_worlds)}", seed, lambda w: 1, lambda w: w["special"],
           lambda w: list(w["ships"]))
    _boids(cs, f"combat-{int(kernel_worlds)}", seed, lambda w: [i for i in w["ships"] if i != w["special"]])
    return cs


def _boids(cs: list, tag: str, seed: int, pick_of) -> None:
    """Each world's SetBoid(on) between ticks (`boids`: tick -> ids), drawn apart from the calls."""
    r = random.Random(f"compose3-boids-{tag}-{seed}")
    for w in cs:
        w["boids"] = {}
        pick = pick_of(w)
        for t in range(len(w["ticks"])):
            if pick and r.random() < 0.3:
                w["boids"][t] = r.sample(pick, k=r.randint(1, min(2, len(pick))))


def collide_world(wid: int = 0, ticks: int = 8) -> dict:
    """A server-layer world over VecCollidePark (server_cases' layout) whose two massive ships meet: 600 m apart, closing at
    300 m/s, each on a GotoPoint at the other's start, a third ship near them; the finding 2 (the own
    collide worlds never collide) made this port's helper's reading of the collision lists never matter."""
    def ship(bid, x, vx, inter):
        return {"id": bid, "mass": 1.0e7, "radius": 100.0, "max_vel": 200.0, "agility": 1.0, "sf": 1.0,
                "pos": [x, 0.0, 0.0], "vel": [vx, 0.0, 0.0], "interactive": inter, "massive": True,
                "max_ang_vel": 1.0, "rot_agility": 0.9}
    balls = [ship(101, 0.0, 150.0, True), ship(102, 600.0, -150.0, True), ship(103, 300.0, 0.0, False)]
    balls[2]["pos"] = [300.0, 260.0, 0.0]
    calls = [[("go_to_point", (101, 600.0, 0.0, 0.0)), ("go_to_point", (102, 0.0, 0.0, 0.0))]] + \
        [[] for _ in range(ticks - 1)]
    return {"id": wid, "park": "collide", "do": False, "balls": balls, "anchor": 101, "special": 102,
            "pilots": {101: 0, 102: 1}, "ticks": calls}


def space_cases(seed: int, n: int = 40, ticks: int = 14) -> list:
    """the worlds of several bubbles over VecModesPark (server_space_cases.cases, park "modes", under seed
    100 + seed) with SetBallFree calls (SetBallTroll is not this layer's: SpaceModesAdapter)."""
    import server_space_cases as SS
    cs = SS.cases(100 + seed, n, "modes", ticks=ticks)
    return inject(cs, "space", seed, lambda w: w["balls"][0]["id"], lambda w: w["special"],
                  lambda w: [b["id"] for b in w["balls"]], troll=False)


# =================================================================================================== the runners

def boid_hook(cs: list, t: int, parks: list, index: dict, scal: list | None = None) -> int:
    """Tick t's SetBoid(on) of each world, between server ticks: on the scalar parks (PySetBoid) and on each batched park
    in `parks` (VecModesPark.set_boid or VecCombatPark's); the number set."""
    rows, k = [], 0
    for wi, w in enumerate(cs):
        for bid in w.get("boids", {}).get(t, ()):
            if scal is not None:
                scal[wi].park.SetBoid(bid, 1)
            rows.append(index[(w["id"], bid)])
            k += 1
    if rows:
        for p in parks:
            m = torch.zeros(p.n, dtype=BOOL, device=p.dev)
            m[torch.tensor(rows, dtype=I64, device=p.dev)] = True
            p.set_boid(m, torch.ones(p.n, dtype=I64, device=p.dev))
    return k


def scalar_rows(sw, do: bool) -> dict:
    """the scalar rows, the members by Destiny's test in place of the set, and the free flag."""
    lane = sw.park.lane
    out = {}
    for bid, row in sc.scalar_rows(sw, "modes", do).items():
        row = list(row)
        row[FI] = tuple(sorted(f.id for f in lane.balls.values() if f.follow_ptr == bid and f.follow_id == bid))
        out[bid] = tuple(row) + (int(lane.balls[bid].is_free),)
    return out


def batched_rows(server, cs: list, index: dict, do: bool) -> dict:
    """the batched rows, the members by Destiny's test (VecModesPark.followers) and the free flag."""
    p = server.a.park
    free = p.free.cpu().tolist()
    out = {}
    for key, row in sc.batched_rows(server, cs, index, "modes", do).items():
        row = list(row)
        row[FI] = tuple(p.followers(index[key]))
        out[key] = tuple(row) + (int(free[index[key]]),)
    return out


def ask(cs: list, scal: list, server, out: Counter, bad: list, t: int) -> None:
    """has_pending_set_free and has_pending_set_not_free for three balls a world (the third a ball not in the park),
    both sides, between the calls and the pre-tick: the answer and its type (Destiny's int 0 where nothing is pending)."""
    W = len(cs)
    mask = torch.ones(W, dtype=BOOL, device=server.dev)
    for probe in range(3):
        ids = [w["balls"][probe]["id"] if probe < 2 and probe < len(w["balls"]) else 4242 for w in cs]
        ball = torch.tensor(ids, dtype=I64, device=server.dev)
        a1, f1 = server.has_pending_set_free(mask, ball)
        a0, f0 = server.has_pending_set_not_free(mask, ball)
        a1, f1, a0, f0 = (x.tolist() for x in (a1, f1, a0, f0))
        for wi, sw in enumerate(scal):
            for want, a, f, name in ((sw.actions.has_pending_set_free(ids[wi]), a1[wi], f1[wi], "free"),
                                     (sw.actions.has_pending_set_not_free(ids[wi]), a0[wi], f0[wi], "not_free")):
                got = bool(a) if f else 0
                out["asked"] += 1
                out["pending"] += int(bool(f))
                out["true"] += int(want is True)
                if type(want) is not type(got) or want != got:
                    bad.append((t, cs[wi]["id"], ids[wi], name, want, got))


def run_modes(cs: list, device: str, do: bool = False, evolve: bool = True, server_kernel: str = "torch",
              boids: bool = True, queries: bool = True) -> tuple:
    """The scalar server layer and the batched one over VecModesPark, tick by tick, the BOID hook between ticks and
    the pending queries before each pre-tick: (counts, [(tick, world, what differs)])."""
    T = len(cs[0]["ticks"])
    old = srv.is_dynamical_orientation_enabled()
    try:
        scal = [sc.ScalarWorld(w) for w in cs]
        server, index = sc.build(cs, "modes", device, do=do, server_kernel=server_kernel)
        out, bad = Counter(), []
        for t in range(T):
            if boids:
                out["boids"] += boid_hook(cs, t, [server.a.park], index, scal)
            for sw, w in zip(scal, cs):
                for name, args in w["ticks"][t]:
                    getattr(sw.actions, name)(*args)
                    out["call " + name] += 1
            for kind, *data in sc.pushes(cs, t, server.dev):
                server.push(kind, *data)
            if queries:
                ask(cs, scal, server, out, bad, t)
            want_sends = []
            for sw in scal:
                sw.ticker.tick()
                if evolve:
                    sw.park.Evolve()
                sw.batcher.send_batch()
                want_sends.append(sc.per_client(sw.network.take()))
            server.pre_tick(snapshot=True)
            if evolve:
                server.evolve()
            server.post_tick()
            got_sends = VS.decode(server)
            got = batched_rows(server, cs, index, do)
            for wi, w in enumerate(cs):
                if sc.canon(sc._listify(want_sends[wi])) != sc.canon(sc._listify(got_sends[wi])):
                    bad.append((t, w["id"], "sends", sc._first_diff(want_sends[wi], got_sends[wi])))
                out["updates"] += sum(len(v) for v in want_sends[wi].values())
                for bid, row in scalar_rows(scal[wi], do).items():
                    out["rows"] += 1
                    g = got[(w["id"], bid)]
                    if not sc.same_rows(row, g):
                        names = sc.FIELDS_MODES + ("free",)
                        bad.append((t, w["id"], bid, [f for f, x, y in zip(names, row, g) if not sc.fc.same(x, y)]))
                lane = scal[wi].park.lane
                out["trolls"] += sum(1 for b in lane.balls.values() if b.mode == 8)
                out["rigid"] += sum(1 for b in lane.balls.values() if b.mode == 11)
                out["fixed"] += sum(1 for b in lane.balls.values() if not b.is_free)
                out["stale_members_kept"] += sum(1 for b in lane.balls.values() if b.mode not in (1, 4)
                                                 and b.follow_ptr is not None and b.follow_id == b.follow_ptr)
        if evolve:
            server.check()
        return out, bad
    finally:
        if old:
            srv.enable_dynamical_orientation()
        else:
            srv.disable_dynamical_orientation()


def lockstep(ts, ks, cs: list, index: dict, pushes, boids: bool = True) -> tuple:
    """Two servers (a torch path ts and a kernel path ks, each on its own park on the GPU), the same BOID hook and
    pushes every tick: every tensor of the two parks and adapters, the park's time, the servers' state, every field of
    the Batch and the decoded updates, compared (server_combat_kernel_cases' comparisons): (counts, differences)."""
    import server_combat_kernel_cases as KC
    T = len(cs[0]["ticks"])
    out, bad = Counter(), []
    for t in range(T):
        if boids:
            out["boids"] += boid_hook(cs, t, [ts.a.park, ks.a.park], index)
        for kind, *data in pushes(cs, t, "cuda"):
            ts.push(kind, *data)
            ks.push(kind, *data)
        for s in (ts, ks):
            s.pre_tick(snapshot=True)
            s.evolve()
            s.post_tick()
        for name in KC.tensors_differing(ts.a.park, ks.a.park):
            bad.append((t, "park", name))
        for name in KC.tensors_differing(ts.a, ks.a):
            bad.append((t, "adapter", name))
        if ts.a.park.current_time != ks.a.park.current_time:
            bad.append((t, "current_time"))
        for k in KC.SERVER_STATE:
            if hasattr(ts, k) and not KC._bits_equal(getattr(ts, k), getattr(ks, k)):
                bad.append((t, "server", k))
        va, vb = KC.batch_view(ts), KC.batch_view(ks)
        diff = [k for k in sorted(set(va) | set(vb)) if va.get(k) != vb.get(k)]
        if diff:
            bad.append((t, "batch", diff[:5]))
        da, db = VS.decode(ts), VS.decode(ks)
        if sc.canon([sc._listify(x) for x in da]) != sc.canon([sc._listify(x) for x in db]):
            bad.append((t, "decode"))
        out["ticks"] += 1
        for s in ts.batch.slots:
            out["queued " + VS.KINDS[s.kind]] += int(s.ok.sum())
        p = ts.a.park
        out["trolls"] += int((p.mode == 8).sum())
        out["rigid"] += int((p.mode == 11).sum())
        out["fixed"] += int((~p.free).sum())
    return out, bad

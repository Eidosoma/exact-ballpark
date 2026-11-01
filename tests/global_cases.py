# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Global balls on Destiny's master, generated from a seed, and the runners the tests compare
bit for bit, at the three levels of tests/space_collide_cases.py (the worlds, which this module draws
from and does not change):

- **the partition layer over the collision park**: the master FullBallpark (one park a world) against
  exact_ballpark.vec_space.VecSpace on its CollideView over exact_ballpark.vec_collide.VecCollidePark made with
  `global_balls` (the park's torch path on the CPU or a CUDA device, the kernel, its statics build, on a CUDA
  device), compared after every step: tests/space_batch.py's whole state comparison (bubbles, filings, sets,
  dictionaries, sensors, events, query answers), each ball's collision state (tests/space_collide_cases.py's) and its
  global flag;
- **the server layer**: the scalar server layer (tests/server_cases.py's ScalarWorld: Actions, Ticker, BubbleUpdater,
  ParkUpdateBatcher on a master DestinyPark over FullBallpark) against exact_ballpark.vec_server_space.SpaceServer over
  SpaceCollideAdapter, with make_ball_global and make_ball_local among the Actions calls, compared after every tick:
  every client's update list (the canonical form: AddBallsToPark streams byte for byte, the global flag in them)
  and every ship's row, extra fields and bubble;
- **Destiny's compiled code** (the conformance kit's space harness, tests/space_cases.py's GLOBAL_OP and AddBall's global flag):
  the worlds as space_cases scenarios, the master reference against the harness on every row kind, and the batched
  layer's ball rows (the global flag among them) against the harness's directly.

**The worlds** are the (a row of three top boxes with two massive ships in contact across the face between two
bubbles; a face, a corner, a split world), drawn from this module's own random stream; each ball is global at its
AddBall with probability 0.25, and in a row world one of the two ships across the face is global (the other not) with
probability 0.6: GetNearbyBalls' asymmetric rule then makes the other ship collide with it while it does not collide
with the other ship (Partition.cpp:344: a candidate in another bubble is skipped unless it is global).
**The layer's records** are an earlier generator's, and SetBallGlobal ("global", id, flag; 8 % of the records, any ball).
**The server's calls** are an earlier generator's, and make_ball_global and make_ball_local (9 % each, any ship of the world).
"""

from __future__ import annotations

import random
from collections import Counter

import torch

import server_cases as S
import space_batch as SB
import space_collide_cases as C
from exact_ballpark import server as srv
from exact_ballpark import vec_server_space as VSS
from exact_ballpark import vec_space as SP

F64, I64, BOOL = torch.float64, torch.int64, torch.bool


# ============================================================================================== the generator

def _globalise(r: random.Random, w: dict) -> None:
    """Each ball global at AddBall with probability 0.25; in a row world one of the two ships across the face global
    and the other not, with probability 0.6."""
    for b in w["balls"]:
        b["global"] = r.random() < 0.25
    pair = w.get("pair")
    if pair and r.random() < 0.6:
        on = r.choice(pair)
        for b in w["balls"]:
            if b["id"] in pair:
                b["global"] = b["id"] == on


def _record(r: random.Random, w: dict) -> tuple:
    if r.random() < 0.08:
        return ("global", r.choice(w["balls"])["id"], r.choice([0, 1]))
    return C._record(r, w)


def cases(seed: int, n: int, steps: int = 16) -> list:
    """n worlds, each with `steps` steps of layer records (0 to 4 a world a step)."""
    r = random.Random(f"global-balls-{seed}")
    out = []
    for wid in range(n):
        w = C._world(r, wid)
        _globalise(r, w)
        for _ in range(steps):
            w["steps"].append([_record(r, w) for _ in range(r.choice([0, 1, 1, 2, 2, 3, 4]))])
        out.append(w)
    return out


def _call(r: random.Random, w: dict) -> tuple:
    k = r.random()
    if k < 0.18:
        return (r.choice(["make_ball_global", "make_ball_local"]), (r.choice(w["balls"])["id"],))
    return C._call(r, w)


def _tick_calls(r: random.Random, w: dict) -> list:
    calls = []
    for _ in range(r.choice([0, 1, 1, 2, 2, 3, 4])):
        c = _call(r, w)
        calls.append(c)
        x = r.random()
        nan = any(isinstance(v, float) and v != v for v in c[1])
        if x < 0.10 and not nan:
            calls.append((c[0], tuple(c[1])))               # the same call twice in a row: the second dropped
        elif x < 0.16:
            calls.append(("stop", (c[1][0],)))
    return calls


def server_cases(seed: int, n: int, ticks: int = 14) -> list:
    """n worlds with `ticks` ticks of Actions calls each."""
    r = random.Random(f"server-global-balls-{seed}")
    out = []
    for wid in range(n):
        w = C._world(r, wid)
        for b in w["balls"]:
            b["free"] = True                                # the server's AddBall: every ship free
        _globalise(r, w)
        w["ticks"] = [_tick_calls(r, w) for _ in range(ticks)]
        out.append(w)
    return out


# ================================================================== the layer's side: VecSpace over VecCollidePark

class ScalarWorld(SB.ScalarWorld):
    """tests/space_batch.py's master world with AddBall's global flag and SetBallGlobal ("global" records)."""

    def __init__(self, w: dict, friction: float = 1.0e6, tick_ms: int = 1000):
        from collections import Counter as _C
        from exact_ballpark.full import FullBallpark
        self.park = p = FullBallpark(friction, tick_ms, is_master=True)
        p.trace = _C()
        for b in w["balls"]:
            p.add_ball(b["id"], b["mass"], b["radius"], b["max_vel"], is_free=b["free"],
                       is_global=bool(b.get("global", False)), is_massive=b["massive"],
                       is_interactive=b["interactive"], x=b["pos"][0], y=b["pos"][1], z=b["pos"][2],
                       vx=b["vel"][0], vy=b["vel"][1], vz=b["vel"][2], agility=b["agility"],
                       speed_fraction=b["sf"])
        self.answers: list = []

    def apply(self, rec: tuple) -> None:
        if rec[0] == "global":
            self.park.set_global(rec[1], bool(rec[2]))
            return
        super().apply(rec)


class GlobalWorlds(C.CollideWorlds):
    """tests/space_collide_cases.py's CollideWorlds on a VecCollidePark made with `global_balls`: AddBall's global
    flag at the load, SetBallGlobal ("global" records) through the park's set_global, which the layer reads."""

    def __init__(self, cs: list, device: str, kernel: str = "torch", contact_slots: int | None = None,
                 iterative: bool = False, hits: bool = True, lib: dict | None = None, friction: float = 1.0e6,
                 tick_ms: int = 1000):
        from exact_ballpark.vec_collide import VecCollidePark
        self.cs = cs
        self.dev = torch.device(device)
        self.kind = "collide"
        N = max(len(w["balls"]) for w in cs)
        W = len(cs)
        self.real = p = VecCollidePark(W, N, friction=friction, tick_interval_ms=tick_ms, device=self.dev,
                                       contact_slots=contact_slots, kernel=kernel, iterative=iterative,
                                       global_balls=True)
        p.load([[{"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
                  "agility": b["agility"], "speed_fraction": b["sf"], "pos": b["pos"], "vel": b["vel"],
                  "free": b["free"], "massive": b["massive"], "global": bool(b.get("global", False))}
                 for b in w["balls"]] for w in cs])
        rows, index, inter = [], {}, [False] * (W * N)
        for wi, w in enumerate(cs):
            order = sorted(b["id"] for b in w["balls"])                 # the park's slots
            rr = []
            for j, b in enumerate(w["balls"]):                          # the layer's slots: AddBall's order
                r = wi * N + order.index(b["id"])
                index[(wi, b["id"])] = (r, j)
                inter[r] = bool(b["interactive"])
                rr.append(r)
            rows.append(rr + [-1] * (N - len(rr)))
        self.index = index
        self.flat = [(r // N, None) for r in range(W * N)]
        self.view = SP.CollideView(p, interactive=inter)
        self.park = C._FlatPark(p)
        self.space = SP.VecSpace(self.view, rows, hits=hits, lib=lib)
        self.space.load()
        self.answers = [[] for _ in cs]
        self.W = W

    def _apply(self, recs: list) -> None:
        p = self.real
        mask = torch.zeros(p.W * p.N, dtype=BOOL)
        flag = torch.zeros(p.W * p.N, dtype=BOOL)
        rest = []
        for wi, rec in enumerate(recs):
            if rec is not None and rec[0] == "global":
                if self.alive(wi, rec[1]):
                    r = self.index[(wi, rec[1])][0]
                    mask[r], flag[r] = True, bool(rec[2])
                rest.append(None)
            else:
                rest.append(rec)
        if bool(mask.any()):                                # SetBallGlobal at once (the park's flag, in place)
            p.set_global(mask.reshape(p.W, p.N).to(self.dev), flag.reshape(p.W, p.N).to(self.dev))
        super()._apply(rest)


def _cross_bubble(sw: ScalarWorld) -> tuple:
    """(collision-list entries naming a ball of another bubble, sensor-list entries of another bubble's ball) of a
    scalar world now: the asymmetric rule's reach."""
    p = sw.park
    col = sen = 0
    for b in p.balls.values():
        for c in b.collisions:
            o = p.balls.get(c)
            if o is not None and o.bubble != b.bubble:
                col += 1
        for oid in b.sensor_balls.keys():
            o = p.balls.get(oid)
            if o is not None and o.bubble != b.bubble:
                sen += 1
    return col, sen


def compare_layer(cs: list, device: str, kernel: str = "torch", steps: int | None = None, hits: bool = True,
                  lib: dict | None = None, contact_slots: int | None = None, iterative: bool = False,
                  graph: bool = False) -> dict:
    """Both sides on the worlds cs, step by step (tests/space_collide_cases.py's compare_layer with the global flags):
    the differences, the volumes compared, the flagged worlds and what was reached. `graph`: the layer's tick replayed
    as a CUDA graph (SpaceGraph, captured after the first step's eager tick)."""
    T = len(cs[0]["steps"]) if steps is None else int(steps)
    scal = [ScalarWorld(w) for w in cs]
    if iterative:
        for sw in scal:
            sw.park.iterative_collision = True
    bw = GlobalWorlds(cs, device, kernel=kernel, hits=hits, lib=lib, contact_slots=contact_slots,
                      iterative=iterative)
    g = None
    diffs, compared, reach = [], Counter(), Counter()
    for sw in scal:
        SB.scalar_events(sw)
    for t in range(T):
        for wi, w in enumerate(cs):
            for rec in w["steps"][t]:
                scal[wi].apply(rec)
        for wi in range(len(cs)):
            scal[wi].evolve()
        bw.space.clear_events()
        width = max((len(w["steps"][t]) for w in cs), default=0)
        for j in range(width):
            bw._apply([w["steps"][t][j] if j < len(w["steps"][t]) else None for w in cs])
        if graph and t >= 1:
            if g is None:
                g = SP.SpaceGraph(bw.space, bw.real)
            g.tick()
        else:
            bw._evolve()
        flags = bw.space.st["flags"].cpu().tolist()
        glob = bw.real.glob.reshape(-1).cpu().tolist()
        for wi in range(len(cs)):
            if flags[wi]:
                continue
            want, got = SB.scalar_state(scal[wi]), SB.batched_state(bw, wi)
            if set(want["balls"]) != set(got["balls"]):
                diffs.append((t, wi, "balls", sorted(set(want["balls"]) ^ set(got["balls"]))))
            for bid in want["balls"]:
                if bid not in got["balls"]:
                    continue
                for key, val in want["balls"][bid].items():
                    compared["ball_fields"] += 1
                    if got["balls"][bid][key] != val:
                        diffs.append((t, wi, bid, key, val, got["balls"][bid][key]))
            for key in ("dicts", "interactives", "keep_alives", "bubble_id", "tops", "proximity"):
                compared["world_fields"] += 1
                if want["world"][key] != got["world"][key]:
                    diffs.append((t, wi, key, want["world"][key], got["world"][key]))
            cw, cg = C.scalar_collisions(scal[wi]), C.batched_collisions(bw, wi)
            for bid, val in cw.items():
                compared["collision_fields"] += 4
                compared["collision_list_entries"] += len(val[2])
                if cg.get(bid) != val:
                    diffs.append((t, wi, bid, "collision", val, cg.get(bid)))
            for b in scal[wi].park.balls.values():          # the global flags
                compared["global_flags"] += 1
                compared["global_ball_steps"] += int(b.is_global)
                mine = bool(glob[bw.index[(wi, b.id)][0]])
                if mine != bool(b.is_global):
                    diffs.append((t, wi, b.id, "global", bool(b.is_global), mine))
            ev_w = SB.scalar_events(scal[wi])
            ev_g = bw.space.events(wi)
            compared["events"] += len(ev_w)
            if ev_w != ev_g:
                diffs.append((t, wi, "events", ev_w, ev_g))
            aw, ag = scal[wi].answers, bw.answers[wi]
            compared["answers"] += len(aw)
            if aw != ag:
                diffs.append((t, wi, "answers", aw, ag))
            scal[wi].answers, bw.answers[wi] = [], []
            compared["ball_steps"] += len(want["balls"])
            compared["ball_steps_with_collisions"] += sum(1 for v in cw.values() if v[2])
            col, sen = _cross_bubble(scal[wi])
            reach["collision_entries_across_bubbles"] += col
            reach["sensor_entries_across_bubbles"] += sen
    flagged = [wi for wi, f in enumerate(bw.space.st["flags"].cpu().tolist()) if f]
    ref_hits = Counter()
    for sw in scal:
        ref_hits.update(sw.park.trace)
    reach["nearby.global_other_bubble"] = ref_hits.get("nearby.global_other_bubble", 0)
    reach["nearby.other_bubble"] = ref_hits.get("nearby.other_bubble", 0)
    return {"diffs": diffs, "compared": dict(compared), "flagged": flagged, "hits": bw.space.hit_counts(),
            "ref_hits": dict(ref_hits), "reach": dict(reach), "steps": T, "graph": g is not None}


# ============================================================================= the server's side: SpaceServer

class ServerWorld(S.ScalarWorld):
    """tests/server_cases.py's scalar server world with AddBall's isGlobal (the thunk's sixth argument)."""

    def __init__(self, w: dict, friction: float = 1.0e6, tick_ms: int = 1000):
        srv.disable_dynamical_orientation()
        self.park = park = srv.DestinyPark(is_master=True)
        assert (park.lane.friction, park.lane.tick_interval) == (friction, tick_ms), "the parks' defaults"
        for b in w["balls"]:
            park.AddBall(b["id"], b["mass"], b["radius"], b["max_vel"], 1, int(bool(b.get("global", False))),
                         int(b["massive"]), int(b["interactive"]), 0, *b["pos"], *b["vel"], b["agility"], b["sf"])
        self.interests = S.PilotInterests(w["pilots"])
        self.network = S.RecordingNetwork()
        self.actions = srv.Actions(park)
        self.batcher = srv.ParkUpdateBatcher(park, self.network, self.interests, self.interests,
                                             srv.BubbleUpdater(park))
        self.ticker = srv.Ticker(park, self.interests, self.actions, self.batcher)


def build_server(cs: list, device: str, kernel: str = "torch", contact_slots: int | None = None) -> tuple:
    """(SpaceServer over SpaceCollideAdapter on a VecCollidePark made with global balls, index {(world, ball id):
    row})."""
    from exact_ballpark.vec_collide import VecCollidePark
    slots = max(len(w["balls"]) for w in cs)
    p = VecCollidePark(len(cs), slots, device=device, contact_slots=contact_slots, kernel=kernel,
                       global_balls=True)
    p.load([[{"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
              "agility": b["agility"], "speed_fraction": b["sf"], "pos": b["pos"], "vel": b["vel"], "free": True,
              "massive": b["massive"], "global": bool(b.get("global", False))} for b in w["balls"]] for w in cs])
    index, inter = {}, []
    for wi, w in enumerate(cs):
        by_id = {b["id"]: b for b in w["balls"]}
        row_inter = [False] * slots
        for s, bid in enumerate(sorted(by_id)):
            index[(w["id"], bid)] = wi * slots + s
            row_inter[s] = bool(by_id[bid]["interactive"])
        inter += row_inter
    a = VSS.SpaceCollideAdapter(p, interactive=inter, added=[[b["id"] for b in w["balls"]] for w in cs])
    pilots = [-1] * a.R
    for w in cs:
        for bid, pp in w["pilots"].items():
            pilots[index[(w["id"], bid)]] = pp
    return VSS.SpaceServer(a, pilots), index


def _global_coverage(per: dict, cov: Counter) -> None:
    """The global notices a tick's sends hold: AddBallsToPark narrowcasts (and those whose stream flags a global
    ball), RemoveBall narrowcasts."""
    for c, ups in per.items():
        for kind, method, state, wait, count in ups:
            if kind != "narrow":
                continue
            for stamp, (name, args) in state:
                if name == "AddBallsToPark":
                    cov["global: AddBallsToPark narrowcast entries"] += 1
                    if len(args[0]) > 42 and args[0][42] & 0x02:    # the flags byte (header, id, mode, radius, pos)
                        cov["global: AddBallsToPark entries of a global ball"] += 1
                elif name == "RemoveBall":
                    cov["global: RemoveBall narrowcast entries"] += 1


def run_server(cs: list, device: str, kernel: str = "torch", contact_slots: int | None = None,
               ticks: int | None = None) -> dict:
    """Both sides on the worlds cs, tick by tick (tests/space_collide_cases.py's run_server with global balls): the
    client update lists that differ, the ship rows that differ (the collision fields, EXTRA, the bubble, the global
    flag), the volumes, the flagged worlds, what was reached."""
    T = len(cs[0]["ticks"]) if ticks is None else int(ticks)
    old = srv.is_dynamical_orientation_enabled()
    try:
        scal = [ServerWorld(w) for w in cs]
        server, index = build_server(cs, device, kernel, contact_slots)
        bad_sends, bad_rows, sends, compared, cov = [], [], 0, 0, Counter()
        for t in range(T):
            want = [sw.tick(w["ticks"][t]) for sw, w in zip(scal, cs)]
            for kind, *data in S.pushes(cs, t, server.dev):
                server.push(kind, *data)
            server.pre_tick(snapshot=True)
            server.evolve()
            server.post_tick()
            got = VSS.decode(server)
            flags = server.space.st["flags"].cpu().tolist()
            rows_b = C.collide_rows(server, index)
            p = server.a.park
            R = server.a.R
            ext = {k: getattr(p, k).reshape(R).cpu().tolist() for k in C.EXTRA}
            bub = server.bubbles_of(torch.arange(R, device=server.dev)).cpu().tolist()
            glob = p.glob.reshape(R).cpu().tolist()
            for wi, w in enumerate(cs):
                if flags[wi]:
                    continue
                if S.canon(S._listify(want[wi])) != S.canon(S._listify(got[wi])):
                    bad_sends.append((t, w["id"], S._first_diff(want[wi], got[wi])))
                sends += sum(len(v) for v in want[wi].values())
                C.SSC_coverage(want[wi], cov)
                _global_coverage(want[wi], cov)
                lane = scal[wi].park.lane
                srows = S.scalar_rows(scal[wi], "collide", False)
                for b in lane.balls.values():
                    r = index[(w["id"], b.id)]
                    compared += 1
                    mine = (rows_b[(w["id"], b.id)], tuple(ext[k][r] for k in C.EXTRA), bub[r], bool(glob[r]))
                    theirs = (srows[b.id], tuple(getattr(b, k) for k in C.EXTRA), b.bubble, bool(b.is_global))
                    if not (S.same_rows(mine[0], theirs[0]) and S.same_rows(mine[1], theirs[1])
                            and mine[2:] == theirs[2:]):
                        diff = [f for f, x, y in zip(S.FIELDS_COLLIDE, theirs[0], mine[0]) if not S.fc.same(x, y)]
                        diff += [f for f, x, y in zip(C.EXTRA, theirs[1], mine[1]) if not S.fc.same(x, y)]
                        diff += [f for f, x, y in zip(("bubble", "global"), theirs[2:], mine[2:]) if x != y]
                        bad_rows.append((t, w["id"], b.id, diff))
                    if b.collisions:
                        cov["ship_ticks_with_collisions"] += 1
                    cov["global_ship_ticks"] += int(b.is_global)
                    for c in b.collisions:
                        o = lane.balls.get(c)
                        if o is not None and o.bubble != b.bubble:
                            cov["collision_entries_across_bubbles"] += 1
        flagged = [wi for wi, f in enumerate(server.space.st["flags"].cpu().tolist()) if f]
        cov["follower_clean_up_stops"] = server.cleanup_stops
        return {"sends": bad_sends, "rows": bad_rows, "updates": sends, "compared": compared, "flagged": flagged,
                "ticks": T, "coverage": dict(cov), "server": server}
    finally:
        if old:
            srv.enable_dynamical_orientation()
        else:
            srv.disable_dynamical_orientation()


# ========================================================== the park's kernel against its torch path, in lockstep

def _state(bw: GlobalWorlds) -> dict:
    out = C._park_state(bw)
    out["glob"] = bw.real.glob.cpu()
    return out


def lockstep(cs: list, device: str, kernel_a: str = "torch", kernel_b: str = "cuda", steps: int | None = None,
             contact_slots: int | None = None) -> dict:
    """The worlds cs on two GlobalWorlds (the park's `kernel_a` and `kernel_b`), the same records, every compared
    tensor of the park (the global flags among them) and the layer equal after every step (float64 by their bits):
    {"diffs": [(step, name)], "tensors": count compared, "steps"}."""
    T = len(cs[0]["steps"]) if steps is None else int(steps)
    a = GlobalWorlds(cs, device, kernel=kernel_a, contact_slots=contact_slots)
    b = GlobalWorlds(cs, device, kernel=kernel_b, contact_slots=contact_slots)
    diffs, n = [], 0
    for t in range(T):
        for bw in (a, b):
            bw.space.clear_events()
            bw.step(t)
        sa, sb = _state(a), _state(b)
        for k in sa:
            n += 1
            if not C._same_tensor(sa[k], sb[k]):
                diffs.append((t, k))
    return {"diffs": diffs, "tensors": n, "steps": T}


# ================================================== Destiny's compiled code (the conformance kit's space harness), directly

def scenarios(cs: list) -> list:
    """The worlds as tests/space_cases.py's scenarios (tests/space_collide_cases.py's, with AddBall's global flag and
    SetBallGlobal as GLOBAL_OP)."""
    import space_cases as sc
    out = C.scenarios([{**w, "steps": [[x for x in st if x[0] != "global"] for st in w["steps"]]} for w in cs])
    for i, w in enumerate(cs):
        recs = out[i]["records"]
        by_id = {b["id"]: b for b in w["balls"]}
        for rec in recs:
            if rec[0] == "B":
                rec[2]["global"] = bool(by_id[rec[2]["id"]].get("global", False))
        rebuilt = [x for x in recs if x[0] == "B"]
        for t, step in enumerate(w["steps"]):
            for rec in step:
                if rec[0] == "global":
                    rebuilt.append(sc._k(t + 1, sc.GLOBAL_OP, rec[1], w=int(rec[2])))
                else:
                    k = C._as_record(sc, t + 1, rec)
                    if k is not None:
                        rebuilt.append(k)
        out[i]["records"] = rebuilt
    return out


def compare_harness(cs: list, device: str, kernel: str = "torch", want: dict | None = None) -> dict:
    """The batched layer over the collision park against Destiny's compiled code directly (tests/space_collide_cases
    .py's compare_harness on GlobalWorlds): every ball's HARNESS_FIELDS and global flag after every step against the
    harness's rows (doubles by their bits); the worlds the layer flags left out."""
    import space_cases as sc
    want = sc.oracle(scenarios(cs)) if want is None else want
    idx = {f: i for i, f in enumerate(sc.S_FIELDS)}
    rows = {(r[0], r[1], r[2]): r for r in want["S"]}
    bw = GlobalWorlds(cs, device, kernel=kernel)
    p = bw.real
    diffs, n, nrows, missing = [], 0, 0, []
    for t in range(len(cs[0]["steps"])):
        bw.space.clear_events()
        bw.step(t)
        flags = bw.space.st["flags"].cpu().tolist()
        tv = {k: v.cpu() for k, v in bw.view.tensors().items()}
        st = {k: v.cpu() for k, v in bw.space.st.items() if v is not None}
        lc, lcol = p.last_c.reshape(-1, 3).cpu(), p.last_collision.reshape(-1).cpu()
        hits, hn = p.hits.reshape(p.W * p.R, -1).cpu().tolist(), p.hits_n.reshape(-1).cpu().tolist()
        glob = p.glob.reshape(-1).cpu().tolist()
        for (wi, bid), (r, j) in bw.index.items():
            if flags[wi] or not st["in_park"][wi, j]:
                continue
            want_row = rows.get((wi + 1, t + 1, bid))
            if want_row is None:
                missing.append((t, wi, bid))
                continue
            s = r - wi * p.N
            hr = wi * p.R + s
            mine = {"bubble": int(st["bubble"][wi, j]), "old_bubble": int(st["old_bubble"][wi, j]),
                    "box_level": int(st["lvl"][wi, j]), "radius": float(tv["radius"][r]),
                    "last_collision": float(lcol[r]), "interactive": int(tv["interactive"][r]),
                    "cloaked": int(tv["cloaked"][r]), "global": int(glob[r]),
                    "collisions": tuple(int(o) for o in hits[hr][:hn[hr]]) if s < p.R else ()}
            for i, ax in enumerate("xyz"):
                mine["box_i" + ax] = int(st["key"][wi, j, i])
                mine["mod_" + ax] = int(st["mod"][wi, j, i])
                mine["p" + ax], mine["v" + ax] = float(tv["pos"][r, i]), float(tv["vel"][r, i])
                mine["c" + ax] = float(lc[r, i])
            nrows += 1
            for f in C.HARNESS_FIELDS + ("global",):
                n += 1
                if not sc.same(want_row[idx[f]], mine[f]):
                    diffs.append((t, wi, bid, f, want_row[idx[f]], mine[f]))
    return {"diffs": diffs, "fields": n, "rows": nrows, "missing": missing}

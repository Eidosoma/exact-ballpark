# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's pending-action queries and the undo of a pending cloak on the batched server layers:
command streams of the frozen generators (the tests/server_cases.py, the tests/server_space_cases.py,
the tests/server_combat_cases.py; their streams untouched) with the game's queries and undos added between the
Actions calls, and the runners the tests compare bit for bit: the scalar server layer (exact_ballpark.server's Actions:
has_pending_cloak, has_pending_uncloak, get_pending_cloak_mode, get_pending_interactive_state,
get_pending_speed_fraction, undo_pending_cloak, _actions.py:686-766, held to Destiny's own tests by
tests/test_destiny_server_layer.py) and the batched layers (exact_ballpark.vec_server.VecServer and its subclasses, on
the torch path and on the kernels).

Added to each tick of each world, by this module's own stream: queries of every kind at any place among the calls (of
the special ship, of another ship, of a missing ball), undos (of the special ship, of a ship with nothing pending),
the special ship's CloakBall (its modes as the world's generator draws them) twice in a row, then undone; a cloak
undone and pushed again with another mode; and the prune's edge, a call X, a CloakBall, X again, the undo: the two X
then stand side by side in Destiny's queue and both are applied (the second X was queued while the CloakBall stood
between them; the duplicate test compares a call with the queue's last action only), which a kernel that tested the
calls again at the pre-tick would drop.
"""

from __future__ import annotations

import random

import torch

import server_cases as S
from exact_ballpark import vec_server as VS

F64, I64 = torch.float64, torch.int64
QUERIES = ("has_pending_cloak", "has_pending_uncloak", "get_pending_cloak_mode", "get_pending_interactive_state",
           "get_pending_speed_fraction")
UNDO = "undo_pending_cloak"
MISSING = 987654321
MODES = {"server": {"collide": (2, 3, 0, 200), "modes": (1, 2, 3, 0), "orient": (1, 2, 3, 0)},
         "space": (1, 1, 2, 3, 0, 200), "combat": (1, 1, 2, 0)}


def _ships(w: dict) -> list:
    return [b["id"] for b in w["balls"]] if "balls" in w else list(w["ships"])


def augment(cs: list, family: str, tag: str, modes: tuple) -> list:
    """Queries, undos and cloaks added into every world's ticks (in place; returns cs). `tag`: this module's stream."""
    r = random.Random(f"pending-{family}-{tag}")
    for w in cs:
        sp = w["special"]
        ships = _ships(w)
        others = [s for s in ships if s != sp]
        for t, calls in enumerate(w["ticks"]):
            out = list(calls)

            def anywhere(call):
                out.insert(r.randrange(len(out) + 1), call)

            for _ in range(r.choice((1, 2, 2, 3))):
                ball = r.choice([sp, sp, r.choice(ships), MISSING])
                anywhere((r.choice(QUERIES), (ball,)))
            k = r.random()
            cloak = ("cloak_ball", (sp, r.choice(modes), float(r.choice((500.0, 1500.0, 2500.0)))))
            if k < 0.25:                                    # twice in a row, then undone: nothing left
                out += [cloak, cloak, ("has_pending_cloak", (sp,)), (UNDO, (sp,)), ("has_pending_cloak", (sp,)),
                        ("get_pending_cloak_mode", (sp,))]
            elif k < 0.45:                                  # undone, then pushed again with another mode
                again = ("cloak_ball", (sp, r.choice(modes), cloak[1][2]))
                out += [cloak, ("get_pending_cloak_mode", (sp,)), (UNDO, (sp,)), again,
                        ("get_pending_cloak_mode", (sp,))]
            elif k < 0.65 and others:                       # the prune's edge: X, CloakBall, X, undo -> X X
                o = r.choice(others)
                x = r.choice((("set_speed_fraction", (o, r.choice((0.25, 0.5, 1.0)))), ("set_max_speed",
                                                                                       (o, 150.0))))
                out += [x, cloak, x, (UNDO, (sp,)), ("get_pending_speed_fraction", (o,)), x]
            elif k < 0.75:                                  # an undo with nothing of the ball pending
                anywhere((UNDO, (r.choice(others or ships),)))
            elif k < 0.85:
                anywhere((UNDO, (sp,)))
            w["ticks"][t] = out
    return cs


# ============================================================================================ the scalar side

def scalar_tick(sw, calls: list, evolve: bool = True) -> tuple:
    """server_cases.ScalarWorld.tick, the queries' answers kept: (sends per client, [(query, ball, answer)])."""
    answers = []
    for name, args in calls:
        res = getattr(sw.actions, name)(*args)
        if name in QUERIES:
            answers.append((name, args[0], res))
    sw.ticker.tick()
    if evolve:
        sw.park.Evolve()
    sw.batcher.send_batch()
    return S.per_client(sw.network.take()), answers


def canon_answer(name: str, x):
    """An answer as compared: None apart; flags as bools; the cloak mode as an int; the speed fraction by its double's
    bits (the batched layer holds the pushed value as a double)."""
    if x is None:
        return None
    if name.startswith("has_") or name == "get_pending_interactive_state":
        return ("b", bool(x))
    if name == "get_pending_cloak_mode":
        return ("i", int(x))
    return ("f", float(x).hex())


# =========================================================================================== the batched side

def steps(cs: list, t: int, device, pushes=S.pushes) -> list:
    """Tick t of every world as the batched layer runs it: the j-th calls of all worlds together, each world's in its
    own order, grouped by what they are: ("push", kind, mask, ball, target, f, n, loc) (by the world's generator's
    `pushes`), ("query", name, mask, ball) and ("undo", mask, ball)."""
    W = len(cs)
    out = []
    width = max((len(w["ticks"][t]) for w in cs), default=0)
    for j in range(width):
        plain = [dict(w, ticks=[[]] * len(w["ticks"])) for w in cs]
        groups: dict = {}
        for wi, w in enumerate(cs):
            calls = w["ticks"][t]
            if j >= len(calls):
                continue
            name, args = calls[j]
            if name in QUERIES or name == UNDO:
                groups.setdefault(name, []).append((wi, int(args[0])))
            else:
                plain[wi]["ticks"] = [[] for _ in w["ticks"]]
                plain[wi]["ticks"][t] = [calls[j]]
        for p in pushes(plain, t, device):
            out.append(("push", *p))
        for name, items in groups.items():
            mask = torch.zeros(W, dtype=torch.bool)
            ball = torch.zeros(W, dtype=I64)
            for wi, b in items:
                mask[wi], ball[wi] = True, b
            mask, ball = mask.to(device), ball.to(device)
            out.append(("undo", mask, ball) if name == UNDO else ("query", name, mask, ball))
    return out


def batched_answers(server, name: str, mask: torch.Tensor, ball: torch.Tensor) -> list:
    """One query of every world in mask, as Destiny's answers: [answer or None (outside mask: absent)]."""
    res = getattr(server, name)(mask, ball)
    if name.startswith("has_"):
        vals, known = res.cpu().tolist(), [True] * server.W
    else:
        vals, known = res[0].cpu().tolist(), res[1].cpu().tolist()
    m = mask.cpu().tolist()
    return [(vals[w] if known[w] else None) if m[w] else "absent" for w in range(server.W)]


def drive(server, cs: list, t: int, answers: list | None = None, pushes=S.pushes) -> None:
    """Tick t's pushes, queries and undos on a batched server, in order; each world's answers appended to
    answers[wi] as (query, ball, answer)."""
    for st in steps(cs, t, server.dev, pushes):
        if st[0] == "push":
            server.push(*st[1:])
        elif st[0] == "undo":
            server.undo_pending_cloak(st[1], st[2])
        else:
            name, mask, ball = st[1:]
            got = batched_answers(server, name, mask, ball)
            if answers is not None:
                bl = ball.cpu().tolist()
                for wi, a in enumerate(got):
                    if a != "absent":
                        answers[wi].append((name, bl[wi], a))


def run(cs: list, park: str, device: str, server, index: dict, scal: list, rows: str = "server",
        pushes=S.pushes) -> dict:
    """The batched server (built by the caller over cs) against the scalar worlds `scal` tick by tick: the answers
    that differ, the client update lists that differ, the ship rows that differ (rows="server": the fields;
    "space": the bubble, position and velocity, an earlier generator's; "combat": the ball state), and the volumes."""
    import server_combat_cases as SC
    import server_space_cases as SS
    from exact_ballpark import vec_server_space as VSS
    T = len(cs[0]["ticks"])
    bad_answers, bad_sends, bad_rows = [], [], []
    vol = {"answers": 0, "undos": 0, "updates": 0, "rows": 0, "kinds": {}}
    for t in range(T):
        want = [scalar_tick(sw, w["ticks"][t]) for sw, w in zip(scal, cs)]
        got_answers = [[] for _ in cs]
        drive(server, cs, t, got_answers, pushes)
        server.pre_tick(snapshot=True)
        server.evolve()
        server.post_tick()
        got = (VSS.decode(server) if rows == "space" else VS.decode(server))
        for wi, w in enumerate(cs):
            vol["undos"] += sum(1 for name, _ in w["ticks"][t] if name == UNDO)
            wa = [(n, b, canon_answer(n, a)) for n, b, a in want[wi][1]]
            ga = [(n, b, canon_answer(n, a)) for n, b, a in got_answers[wi]]
            vol["answers"] += len(wa)
            for n, b, a in wa:
                key = f"{n}:{'none' if a is None else a[0]}"
                vol["kinds"][key] = vol["kinds"].get(key, 0) + 1
            if wa != ga:
                bad_answers.append((t, w["id"], wa, ga))
            if S.canon(S._listify(want[wi][0])) != S.canon(S._listify(got[wi])):
                bad_sends.append((t, w["id"], S._first_diff(want[wi][0], got[wi])))
            vol["updates"] += sum(len(v) for v in want[wi][0].values())
        if rows == "space":
            bub = server.bubbles_of(torch.arange(server.a.R, device=server.dev)).cpu().tolist()
            pos, vel = server.a.park.pos.cpu(), server.a.park.vel.cpu()
            for wi, w in enumerate(cs):
                for b in scal[wi].park.lane.balls.values():
                    r = index[(w["id"], b.id)]
                    vol["rows"] += 1
                    mine = (bub[r], tuple(SS._bits(x) for x in pos[r].tolist()),
                            tuple(SS._bits(x) for x in vel[r].tolist()))
                    theirs = (b.bubble, tuple(SS._bits(x) for x in b.pos), tuple(SS._bits(x) for x in b.vel))
                    if mine != theirs:
                        bad_rows.append((t, w["id"], b.id))
        elif rows == "combat":
            p = server.a.park
            cpu = type("P", (), {k: v.cpu() for k, v in vars(p).items() if torch.is_tensor(v)})
            for wi, w in enumerate(cs):
                for b in scal[wi].park.lane.balls.values():
                    r = index[(w["id"], b.id)]
                    vol["rows"] += 1
                    if SC.ball_state(cpu, r) != SC.scalar_ball_state(b):
                        bad_rows.append((t, w["id"], b.id))
        else:
            g = S.batched_rows(server, cs, index, park, False)
            for wi, w in enumerate(cs):
                for bid, row in S.scalar_rows(scal[wi], park, False).items():
                    vol["rows"] += 1
                    if not S.same_rows(row, g[(w["id"], bid)]):
                        bad_rows.append((t, w["id"], bid))
    flagged = []
    if hasattr(server, "space"):
        flagged = [wi for wi, f in enumerate(server.space.st["flags"].cpu().tolist()) if f]
    elif rows == "combat":
        flagged = [wi for wi, f in enumerate(server.a.park.world_flags()) if f]
    return {"answers": bad_answers, "sends": bad_sends, "rows": bad_rows, "flagged": flagged, "volumes": vol}


# ====================================================================================== kernel against torch

def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit for bit: doubles by their bits, two NaNs equal."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == F64:
        return bool(((a.view(I64) == b.view(I64)) | (torch.isnan(a) & torch.isnan(b))).all())
    return torch.equal(a, b)


def tensors_diff(ts, ks) -> list:
    """Every tensor of the two servers' parks, partitions (if any) and server states that differs."""
    bad = []
    for name, v in vars(ts.a.park).items():
        if torch.is_tensor(v) and not same(v, vars(ks.a.park)[name]):
            bad.append(f"park.{name}")
    if hasattr(ts, "space"):
        for name, v in ts.space.st.items():
            if torch.is_tensor(v) and not same(v, ks.space.st[name]):
                bad.append(f"space.{name}")
    for name in ("stamp_for_system", "settled", "_fresh", "_listed", "_bubble_time"):
        if torch.is_tensor(getattr(ts, name, None)) and not same(getattr(ts, name), getattr(ks, name)):
            bad.append(name)
    return bad


def batch_diff(ts, ks) -> list:
    """The two batches' fields that differ: stamps, routes, flags, notices; each slot's kind, queued flags, balls and,
    where queued, rows, target rows and local arguments (a kernel's extra clean-up slots: never queued)."""
    a, b = ts.batch, ks.batch
    bad = []
    Jt = len(a.slots)
    if len(b.slots) < Jt:
        return ["slots"]
    for name in ("stamp", "clock", "has_character", "gets_bubble"):
        if not same(getattr(a, name), getattr(b, name)):
            bad.append(name)
    if not torch.equal(a.route, b.route[:, :Jt]):
        bad.append("route")
    if any(bool(s.ok.any()) for s in b.slots[Jt:]):
        bad.append("extra slots queued")
    for j in range(Jt):
        sa, sb = a.slots[j], b.slots[j]
        if sa.kind != sb.kind or not torch.equal(sa.ok, sb.ok) or not torch.equal(sa.ball, sb.ball):
            bad.append(f"slot{j}")
            continue
        for name in ("row", "trow", "loc"):
            if not torch.equal(torch.where(sa.ok, getattr(sa, name), -9), torch.where(sa.ok, getattr(sb, name), -9)):
                bad.append(f"slot{j}.{name}")
    if [j for j, _, _ in a.notices] != [j for j, _, _ in b.notices] or \
            any(not torch.equal(x, y) for (_, x, _), (_, y, _) in zip(a.notices, b.notices)):
        bad.append("notices")
    return bad


def kernel_against_torch(cs: list, ts, ks, decode, ticks: int | None = None, pushes=S.pushes) -> tuple:
    """Two servers over their own parks, the same pushes, queries and undos every tick: ([(tick, what differs)],
    volumes). Before each pre-tick the kernel path's answers (its calls tested by the queries' own test) equal the
    torch path's (tested at each push), and after it every tensor and the batch."""
    T = len(cs[0]["ticks"]) if ticks is None else ticks
    bad, vol = [], {"ticks": 0, "answers": 0, "undos": 0, "ok_slots": 0, "resolved_slots": 0, "updates": 0}
    for t in range(T):
        at, ak = [[] for _ in cs], [[] for _ in cs]
        for st in steps(cs, t, ts.dev, pushes):
            for srv, ans in ((ts, at), (ks, ak)):
                if st[0] == "push":
                    srv.push(*st[1:])
                elif st[0] == "undo":
                    srv.undo_pending_cloak(st[1], st[2])
                else:
                    got = batched_answers(srv, st[1], st[2], st[3])
                    for wi, a in enumerate(got):
                        if a != "absent":
                            ans[wi].append((st[1], canon_answer(st[1], a)))
            if st[0] == "undo":
                vol["undos"] += int(st[1].sum())
        vol["resolved_slots"] += sum(1 for s in ks.slots if getattr(s, "resolved", False))
        if at != ak:
            bad.append((t, "answers"))
        vol["answers"] += sum(len(x) for x in at)
        for srv in (ts, ks):
            srv.pre_tick(snapshot=True)
            srv.evolve()
            srv.post_tick()
        diff = tensors_diff(ts, ks) + batch_diff(ts, ks)
        da, db = decode(ts), decode(ks)
        vol["updates"] += sum(len(u) for per in da for u in per.values())
        if S.canon([S._listify(x) for x in da]) != S.canon([S._listify(x) for x in db]):
            diff.append("decode")
        if diff:
            bad.append((t, diff[:6]))
        vol["ticks"] += 1
        vol["ok_slots"] += int(sum(int(s.ok.sum()) for s in ts.batch.slots))
    return bad, vol

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The composed park (exact_ballpark.vec_composed.VecComposedPark: flight modes, missiles and
ship-to-ship collisions in one park, on tensors) against the scalar reference (exact_ballpark.full.FullBallpark), bit
for bit, every field of every ball at every step (tests/destiny_combat_cases.py's fields with the collision
acceleration, the collision list and the partition's filing), which balls exist, the ReserveFormationSlot answers and
the DoCollision events, on generated worlds (tests/park_cases.py): on the CPU and, when a CUDA device is visible, on
the GPU. The reference is held to Destiny's compiled code by tests/test_movement_full_oracle.py and
tests/test_movement_space_oracle.py, and on a sample of these very worlds here (the conformance kit's full harness), so this is
exact against Destiny. Development worlds are seed 1, held-out seed 2
(first compared after the generator was frozen). Worlds outside the reference's domain (more than one bubble on the
server, a result Destiny leaves undefined) and worlds the batch refuses (its domain flags) are counted, never compared
as equal. Planted faults, one rule of the reference each, must each be caught.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

import destiny_combat_cases as dc
import park_cases as pc
from exact_ballpark import vec_combat as vc
from exact_ballpark import vec_composed as vp
from exact_ballpark.modes import MISSILE, WARP
from exact_ballpark.ref import STOP

CUDA = torch.cuda.is_available()
NB, NW = 16, 16                                   # each seed: 16 batches of 16 worlds
_REF: dict = {}


def references(seed: int, nb: int = NB, nw: int = NW, gen=pc) -> list:
    key = (gen.__name__, seed, nb, nw)
    if key not in _REF:
        _REF[key] = [[gen.reference(cfg, w) for w in cfg["worlds"]] for cfg in gen.batches(seed, nb, nw)]
    return _REF[key]


def run(seed: int, device: str = "cpu", park_cls=None, nb: int = NB, nw: int = NW, batches=None, gen=pc,
        **kw) -> Counter:
    """Every batch of a seed on the composed park: counts and the first differences."""
    refs = references(seed, nb, nw, gen)
    out = Counter()
    examples = []
    for bi, cfg in enumerate(gen.batches(seed, nb, nw)):
        if batches is not None and bi not in batches:
            continue
        got, park = pc.batched(cfg, device, park_cls, **kw)
        flags = park.world_flags()
        for wi, want in enumerate(refs[bi]):
            if not dc.in_domain(want):
                out["outside_reference_domain"] += 1
                continue
            if flags[wi]:
                out["refused"] += 1
                out[f"refused_flag_{flags[wi]}"] += 1
                continue
            out["worlds"] += 1
            out["ball_steps"] += sum(len(v) for v in want["S"].values())
            out["events"] += len(want["E"])
            bad = dc.differences(want, got[wi])
            if bad:
                out["differ"] += 1
                out["values"] += len(bad)
                if len(examples) < 4:
                    examples.append((bi, wi, bad[:3]))
    out["examples"] = examples
    return out


def _clean(out: Counter, worlds: int, allowed: int = 0) -> None:
    """No difference, at least `worlds` compared, and no world refused but for the flags in `allowed`."""
    assert out["differ"] == 0, out["examples"]
    assert out["worlds"] >= worlds, dict(out)
    refused = sum(v for k, v in out.items() if k.startswith("refused_flag_") and int(k[13:]) & ~allowed)
    assert refused == 0, dict(out)


@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_composed_park_matches_fullballpark_on_the_cpu(seed):
    _clean(run(seed), 240)


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_composed_park_matches_fullballpark_on_the_gpu(seed):
    """The same algorithm on the card's tensors (the first half of each seed's batches: the login card's minutes are
    rationed), against the reference."""
    _clean(run(seed, "cuda", batches=range(NB // 2)), 115)


def test_the_scenarios_reach_every_branch():
    """The development worlds put every flight mode and missiles into contact (the reference's own counters and
    rows), drop balls out of warp among the others (seen by higher ids only, at rest re-filed), and fly missiles as
    plain balls."""
    trace, rows = Counter(), Counter()
    for cfg in pc.batches(1, NB, NW):
        for w in cfg["worlds"]:
            t = Counter()
            want = pc.reference(cfg, w, trace=t)
            if not dc.in_domain(want):
                continue
            trace.update(t)
            S = want["S"]
            for s in range(1, cfg["steps"] + 1):
                for bid, row in S[s].items():
                    prev = S[s - 1].get(bid)
                    if prev is not None and prev["mode"] == WARP and prev["effect_stamp"] >= 0 and row["mode"] == STOP:
                        rows["drop_out"] += 1
                    if not row["collisions"]:
                        continue
                    rows[f"contact_{prev['mode'] if prev is not None else row['mode']}"] += 1
                    if bid in pc.POOL and row["mode"] != MISSILE:
                        rows["missile_ball_contact"] += 1
                    if len(set(row["collisions"])) > 1:
                        rows["two_or_more"] += 1
                    for oid in set(row["collisions"]):
                        o0, o1 = S[s - 1].get(oid), S[s].get(oid)
                        if o0 and o1 and o0["mode"] == WARP and o0["effect_stamp"] >= 0 and o1["mode"] == STOP:
                            rows["met_a_drop_out"] += 1
    from exact_ballpark.modes import FOLLOW, FORMATION, ORBIT
    from exact_ballpark.ref import GOTO
    floors = {f"contact_{m}": 10 for m in (GOTO, STOP, FOLLOW, ORBIT, MISSILE, FORMATION)}
    floors[f"contact_{WARP}"] = 5
    floors.update(drop_out=50, missile_ball_contact=10, two_or_more=200, met_a_drop_out=10)
    for k, v in floors.items():
        assert rows[k] >= v, (k, rows[k])
    for k in ("potential.free", "potential.fixed", "potential.recurse", "potential.light", "potential.early_return",
              "potential.equal_time", "potential.earlier_kept", "potential.coincident", "warp.drop_out",
              "warp.overrun", "warp.drop_out_insert", "missile.collision_event", "missile.self_collision",
              "candidates.missile_target", "release.goto_direction", "release.stop", "orbit_old.tangent",
              "orbit_new.factor_log", "orbit_new.heading_normal", "orbit_new.default_normal", "gtf.orbit_thrust",
              "formation.slot", "launch.straight", "warpin.warp", "warp.align_ok", "insert.corner"):
        assert trace[k] >= 3, (k, trace[k])


def _drop_world(mv: float) -> dict:
    """A ship (id 2) warping in onto two others: id 1 on one side, id 3 on the other, neither touching the other."""
    centre = [1000.0, -2000.0, 500.0]
    common = {"free": True, "massive": True, "agility": 1.0, "speed_fraction": 1.0, "max_ang_vel": 1.0,
              "rot_agility": 0.9, "vel": [0.0, 0.0, 0.0], "max_vel": 150.0}
    ships = [{**common, "id": 1, "mass": 1e7, "radius": 60.0, "interactive": True,
              "pos": [centre[0] - 50.0, centre[1], centre[2]]},
             {**common, "id": 2, "mass": 2e7, "radius": 50.0, "interactive": False, "max_vel": mv,
              "pos": [centre[0] + 1.5e5, centre[1], centre[2]]},
             {**common, "id": 3, "mass": 1e7, "radius": 40.0, "interactive": False,
              "pos": [centre[0] + 70.0, centre[1], centre[2]]}]
    world = {"ships": ships, "pool": list(pc.POOL), "records": [(0, "warpin", 2, *centre, 6000)], "start": 100}
    return {"do": False, "do_off": False, "master": False, "tick": 1000, "friction": 1.0e6, "new_orbit": False,
            "steps": 12, "worlds": [world]}


@pytest.mark.parametrize("mv", [0.0, 150.0], ids=["at_rest", "moving"])
def test_a_drop_out_is_met_by_the_higher_ids_only(mv):
    """WarpDistance drops a ball out at its own turn in phase 3: ship 3 (a higher id) meets it at its drop-out point
    in that tick, ship 1 (a lower id) does not, though both overlap it. At rest the drop-out re-files it there; moving
    it keeps its last filing, from its last warp position, which ship 3's boxes do not relate to."""
    cfg = _drop_world(mv)
    want = pc.reference(cfg, cfg["worlds"][0])
    S = want["S"]
    drop = next(s for s in range(1, cfg["steps"] + 1) if S[s - 1][2]["mode"] == WARP and S[s][2]["mode"] == STOP)
    assert 2 not in S[drop][1]["collisions"]
    assert (2 in S[drop][3]["collisions"]) == (mv == 0.0)
    got, park = pc.batched(cfg)
    assert park.world_flags() == [0]
    assert dc.differences(want, got[0]) == []


def _overrun_world() -> dict:
    """_drop_world's ships with a long tick: EntityWarpIn begins five ticks back, so the short warp is past its
    deceleration at its first tick and drops out at once, at rest on the goto point, far from its last filing."""
    cfg = _drop_world(150.0)
    cfg["tick"] = 2000
    return cfg


def test_an_overrun_drop_out_is_filed_where_it_rests():
    """The drop-out past the deceleration (WarpDistance's last branch: the goto point, at rest) re-files the ball
    there at its own turn (modes.py:711), so ship 3, after it in the tick, meets it at once; ship 1 does not."""
    cfg = _overrun_world()
    want = pc.reference(cfg, cfg["worlds"][0])
    S = want["S"]
    assert S[0][2]["mode"] == WARP and S[1][2]["mode"] == STOP and S[1][2]["vel"] == (0.0, 0.0, 0.0)
    assert 2 in S[1][3]["collisions"] and 2 not in S[1][1]["collisions"]
    assert S[0][2]["key"] != S[1][2]["key"]
    got, park = pc.batched(cfg)
    assert park.world_flags() == [0]
    assert dc.differences(want, got[0]) == []


def test_a_worlds_result_does_not_depend_on_its_batch():
    """A world alone, among others and in a reversed batch gives the same bits."""
    cfg = pc.batches(1, 2, NW)[1]
    full, _ = pc.batched(cfg)
    rev = {**cfg, "worlds": cfg["worlds"][::-1]}
    back, _ = pc.batched(rev)
    for wi in range(len(cfg["worlds"])):
        assert dc.differences(full[wi], back[len(cfg["worlds"]) - 1 - wi]) == []
    for wi in (0, 5):
        alone, _ = pc.batched({**cfg, "worlds": [cfg["worlds"][wi]]})
        assert dc.differences(full[wi], alone[0]) == []


def test_the_combat_worlds_with_collisions_on():
    """tests/destiny_combat_cases.py's worlds (missiles, formations, cloaks, warps; at most one
    massive ball a world, so few contacts but a missile's) with collisions on: the composed park equals the reference;
    a normal cloak on the master is refused (its sensor's re-filing)."""
    out = run(1, nb=6, nw=24, gen=dc)
    _clean(out, 110, allowed=vp.FLAG_SENSOR)       # a normal cloak on the master: its sensor's re-filing


def test_collisions_off_is_vec_combat_parks_tick():
    """With collisions off the composed park is VecCombatPark's tick (its two-massive-balls refusal kept), on the
    worlds VecCombatPark's own tensor tick holds and on its warps (the kernels' worlds): the reference's bits."""
    out = run(1, nb=6, nw=24, gen=dc, collisions=False)
    _clean(out, 135, allowed=vc.FLAG_MASSIVE)


def test_the_domain_is_refused_not_guessed():
    """A massive ball in warp proper (SetBallMassive during a warp) and, on the master, a normal cloak's proximity
    sensor are flagged and raised by check(); the kernel path (`evolve` with collisions on) needs a CUDA device
    (tests/test_movement_vec_composed_cuda.py)."""
    cfg = _drop_world(0.0)
    rows, where = dc.rows_for(cfg)
    p = vp.VecComposedPark(len(rows), cfg["friction"], cfg["tick"], "cpu")
    p.load(rows)
    n = len(rows)
    one = torch.zeros(n, dtype=torch.bool)
    one[where[(0, 2)]] = True
    p.entity_warp_in(one, torch.tensor([[1000.0, -2000.0, 500.0]] * n, dtype=torch.float64),
                     torch.full((n,), 6000, dtype=torch.int64))
    p.evolve_torch()
    p.massive = torch.where(one, True, p.massive)
    p.evolve_torch()
    assert p.world_flags()[0] & vp.FLAG_WARP_MASSIVE
    with pytest.raises(RuntimeError, match="warp proper"):
        p.check()
    with pytest.raises(ValueError, match="CUDA device"):
        p.evolve()
    m = vp.VecComposedPark(len(rows), cfg["friction"], cfg["tick"], "cpu", master=True)
    m.load([{**r, "bubble": 0} for r in rows])
    m.cloak(one, torch.ones(n, dtype=torch.int64))
    assert m.world_flags()[0] & vp.FLAG_SENSOR


def _harness_case(cfg: dict, w: dict, cid: int) -> dict:
    """A world as the conformance kit's full harness takes it (tests/full_cases.py's records): tests/test_movement_vec_combat.py's
    adapter (ReserveFormationSlot-then-FormationFollow as the query and a FormationFollow with the slot the reference
    answered; LaunchMissile's aimed and massive as its bits), with this generator's commands added (FollowBall, Orbit,
    EntityWarpIn and the setters)."""
    import full_cases as fc
    char = lambda n: ((int(n) + 128) & 0xFF) - 128
    answers = iter(pc.reference(cfg, w)["A"])

    def ball(b, step):
        return ("B", step, {"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
                            "free": b["free"], "global": False, "massive": b["massive"],
                            "interactive": b["interactive"], "junk": False, "pos": list(b["pos"]),
                            "vel": list(b["vel"]), "agility": b["agility"], "sf": b["speed_fraction"],
                            "max_ang_vel": b["max_ang_vel"], "rot_agility": b["rot_agility"]})
    recs = [ball(s, 0) for s in w["ships"]]
    flat = [float(len(dc.FORMATIONS))]
    for vectors in dc.FORMATIONS:
        flat.append(float(len(vectors)))
        for v in vectors:
            flat.extend(float(c) for c in v)
    k = fc._k
    recs.append(k(0, fc.FORMATIONS_OP, d=flat))
    simple = {"goto": fc.GOTO_OP, "dir": fc.DIR_OP, "vel": fc.VEL_OP}
    scalar = {"radius": fc.RADIUS_OP, "maxvel": fc.MAXVEL_OP, "sf": fc.SF_OP, "mass": fc.MASS_OP}
    for rec in w["records"]:
        s, op, a = rec[0], rec[1], rec[2:]
        if op in simple:
            recs.append(k(s, simple[op], a[0], d=a[1:4]))
        elif op in scalar:
            recs.append(k(s, scalar[op], a[0], d=(0.0, 0.0, 0.0, a[1])))
        elif op in ("follow", "orbit"):
            recs.append(k(s, fc.FOLLOW_OP if op == "follow" else fc.ORBIT_OP, a[0], a[1], d=(0.0, 0.0, 0.0, a[2])))
        elif op == "warpin":
            recs.append(k(s, fc.WARPIN_OP, a[0], w=a[4], d=a[1:4]))
        elif op == "stop":
            recs.append(k(s, fc.STOP_OP, a[0]))
        elif op == "warp":
            recs.append(k(s, fc.WARP_OP, a[0], w=a[5], d=(*a[1:4], a[4])))
        elif op == "massive":
            recs.append(k(s, fc.MASSIVE_OP, a[0], w=a[1]))
        elif op == "add":
            recs.append(ball(a[0], s))
        elif op == "launch":
            recs.append(k(s, fc.LAUNCH_OP, a[0], a[1], a[2], w=(char(a[3]) == 1) | ((char(a[4]) != 0) << 1)))
        elif op == "mfollow":
            recs.append(k(s, fc.MISSILE_FOLLOW_OP, a[0], a[1], a[2]))
        elif op == "cloak":
            recs.append(k(s, fc.CLOAK_OP, a[0], w=a[1]))
        elif op == "uncloak":
            recs.append(k(s, fc.UNCLOAK_OP, a[0]))
        elif op == "remove":
            recs.append(k(s, fc.REMOVE_OP, a[0], w=0))
        elif op == "setform":
            recs.append(k(s, fc.SET_FORMATION_OP, a[0], w=a[1]))
        elif op == "formfollow":
            recs.append(k(s, fc.FORMATION_FOLLOW_OP, a[0], a[1], w=a[2]))
        elif op == "reserve_follow":
            slot = next(answers, -1)
            recs.append(k(s, fc.RESERVE_SLOT_Q, a[1]))
            if slot >= 0:
                recs.append(k(s, fc.FORMATION_FOLLOW_OP, a[0], a[1], w=slot))
        elif op == "reserve":
            next(answers, -1)
            recs.append(k(s, fc.RESERVE_SLOT_Q, a[0]))
        elif op == "free":
            recs.append(k(s, fc.FREE_SLOT_OP, a[0], w=a[1]))
        else:
            raise ValueError(op)
    return {"id": cid, "kind": "park1", "friction": cfg["friction"], "tick": cfg["tick"], "steps": cfg["steps"],
            "new_orbit": cfg["new_orbit"], "iterative": False, "do": cfg["do"], "do_missiles_off": cfg["do_off"],
            "manual": False, "master": cfg["master"], "seed": 1, "start": w["start"], "records": recs}


def test_the_scenarios_are_destinys_own():
    """The first four development batches (64 worlds) run directly on Destiny's own code compiled verbatim (the conformance kit's full harness) and on the reference: every row of every step equal (balls, boxes, bubbles, events, answers), the
    chain's first link on these very worlds."""
    import full_cases as fc
    if not fc.HARNESS.is_file():
        pytest.skip("Destiny full harness not built (conformance/build_full.py)")
    cs, cid = [], 0
    for cfg in pc.batches(1, NB, NW)[:4]:
        for w in cfg["worlds"]:
            cid += 1
            cs.append(_harness_case(cfg, w, cid))
    rows, undefined = fc.run_reference(cs)
    want = fc.oracle([c for c in cs if c["id"] not in undefined])
    assert not undefined, undefined
    assert not fc.differences(want, rows, undefined)
    assert len(rows["S"]) > 5000 and len(rows["E"]) > 500, {k: len(v) for k, v in rows.items()}


# ---- planted faults: one rule each, every one must change the comparison ----

class _SeenByAll(vp.VecComposedPark):
    """A ball dropping out of warp a candidate of every responder (its id ignored)."""
    @staticmethod
    def _dropped_seen(dropped_v, idr, idn):
        return dropped_v[:, None, :] & (idn != idr)


class _SeenByNone(vp.VecComposedPark):
    """A ball dropping out of warp no candidate in its drop-out tick."""
    @staticmethod
    def _dropped_seen(dropped_v, idr, idn):
        return torch.zeros_like(dropped_v[:, None, :] & (idn < idr))


class _NotAtRest(vp.VecComposedPark):
    """The drop-out's re-filing at rest left out."""
    def _at_rest(self, dropped, wv):
        return torch.zeros_like(dropped)


class _MissilesSeen(vp.VecComposedPark):
    """Missiles not left out of the other balls' candidates."""
    @staticmethod
    def _left_out(missile):
        return torch.zeros_like(missile)


class _ExemptView(vp._GradientView):
    def _exempt(self, other):
        return torch.zeros_like(other, dtype=torch.bool)


class _NoExemption(vp.VecComposedPark):
    """COL-711's missile exception left out (a missile keeps its response to its own target)."""
    view_cls = _ExemptView


class _ReversedView(vp._GradientView):
    def _order_keys(self, pair):
        return -super()._order_keys(pair)


class _Reversed(vp.VecComposedPark):
    """Each ball's candidates in the reverse of the partition's order."""
    view_cls = _ReversedView


class _LaunchNotFiled(vp.VecComposedPark):
    """LaunchMissile's SetBallPosition without its re-filing."""
    def launch_missile(self, mask, target, owner, aimed, massive):
        vc.VecCombatPark.launch_missile(self, mask, target, owner, aimed, massive)


class _AllBefore(vp.VecComposedPark):
    """Every released follower released before its own turn, whatever the ids."""
    def _release_turns(self, hit, ts):
        return hit, torch.zeros_like(hit)


class _YprBlind(vp.VecComposedPark):
    """CalculateYawPitchRoll blind to an aligning warper's collisions."""
    def _collided(self):
        return torch.zeros_like(self.hits_n, dtype=torch.bool)


FAULTS = {"drop_seen_by_all": _SeenByAll, "drop_seen_by_none": _SeenByNone, "drop_not_filed_at_rest": _NotAtRest,
          "missiles_seen_by_others": _MissilesSeen, "no_missile_exemption": _NoExemption,
          "reversed_order": _Reversed, "launch_not_filed": _LaunchNotFiled, "releases_all_before": _AllBefore,
          "ypr_blind_to_collisions": _YprBlind}


# the development batches where each fault shows;
# the drop-out's re-filing at rest changes no generated world (their overruns rest within a metre of their last
# filing), so the hand-built overrun catches it
FAULT_BATCHES = {"drop_seen_by_all": [0, 1], "drop_seen_by_none": [0, 1], "drop_not_filed_at_rest": [],
                 "missiles_seen_by_others": [0], "no_missile_exemption": [0], "reversed_order": [0],
                 "launch_not_filed": [7, 11], "releases_all_before": [1, 2], "ypr_blind_to_collisions": [8, 10]}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault):
    cls = FAULTS[fault]
    out = run(1, park_cls=cls, batches=FAULT_BATCHES[fault]) if FAULT_BATCHES[fault] else Counter()
    hand = 0
    for cfg in (_drop_world(0.0), _drop_world(150.0), _overrun_world()):
        got, _ = pc.batched(cfg, "cpu", cls)
        hand += len(dc.differences(pc.reference(cfg, cfg["worlds"][0]), got[0]))
    if FAULT_BATCHES[fault]:
        assert out["differ"] >= 1, (fault, dict(out))
    else:
        assert hand >= 1, fault

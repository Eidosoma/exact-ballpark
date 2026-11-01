# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's collisions batched (exact_ballpark.vec_collide) against the scalar reference (exact_ballpark.space, a client's
view), bit for bit, every tick, on the CPU and (when one is visible) the GPU; a sample against Destiny's own code
compiled verbatim; the contact order and every branch of both methods exercised and held by planted faults.

Generated worlds (tests/collide_cases.py, records as tests/space_cases.py's): approaching, glancing and crossing pairs
and groups, overlapping and coincident balls, light balls, fixed spheres as walls and as obstacle fields, several
contacts of one ball in a tick, near misses by a millimetre to metres, chains, fast balls, balls on and near partition
box faces at levels 3 to 7 (the order of a ball's candidates), space junk, cloaked and non-massive neighbours, and
racers flying GOTO through obstacle fields with throttle and stop commands; 2 to 8 balls a world, ticks of 0.25 to 2 s,
several frictions. After every tick each ball's position, velocity, old position and velocity, goto point, mLastG,
mLastC, mLastCollision, speed fraction, mode, partition filing (level, key, side, boxes) and collision list are
compared (doubles by their bits). Development worlds are seed 1; held-out worlds are seed 2.

The gradient method also runs without host syncs (a fixed contact_slots, for CUDA graphs), equal to the pruned run, and
an exceeded contact_slots is an overflow event. The server's view with every ball in one live bubble decides the same
as the client's view the park models. Inactive worlds (evolve(active)) keep their state bit for bit. The iterative
method's worlds where Destiny's result is undefined (its CalculateEnd under a reallocating push_back) are refused by
the reference and marked by the park, the same worlds. With collisions off the park is exact_ballpark.vec.VecBallpark's
tick, unchanged.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

import collide_cases as cc
import space_cases as sc
import exact_ballpark.vec_collide as VC
from movement_cases import cases as movement_cases
from exact_ballpark import vec_exp
from exact_ballpark.vec import VecBallpark

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                                reason="no visible GPU"))]
_REF: dict = {}


def reference(seed: int, n: int, iterative: bool) -> tuple:
    """(scenarios, the reference's rows, the undefined scenarios), cached."""
    key = (seed, n, iterative)
    if key not in _REF:
        cs = cc.cases(seed, n=n, iterative=iterative)
        rows, undefined = cc.reference_rows(cs)
        _REF[key] = (cs, rows, undefined)
    return _REF[key]


def compare(seed: int, n: int, iterative: bool, device: str, **kw) -> list:
    cs, want, und_ref = reference(seed, n, iterative)
    und: set = set()
    got = cc.batched(cs, device, undefined=und, **kw)
    assert und == und_ref, f"undefined worlds: reference {sorted(und_ref)}, park {sorted(und)}"
    return cc.differences(want, {k: v for k, v in got.items() if k[0] not in und})


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [1, 2])
def test_gradient_matches_reference_bit_for_bit(device, seed):
    bad = compare(seed, 240, False, device)
    assert not bad, f"{len(bad)} values differ, first {bad[:5]}"


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_without_host_syncs_is_the_same(device):
    """A fixed contact_slots (no host sync in the tick, capturable in a CUDA graph) gives the same bits; one too small
    is an overflow event at check()."""
    assert not compare(1, 240, False, device, contact_slots=7)
    cs, _, _ = reference(1, 240, False)
    with pytest.raises(RuntimeError, match="contact_slots"):
        cc.batched(cs, device, contact_slots=1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [1, 2])
def test_iterative_matches_reference_bit_for_bit(device, seed):
    bad = compare(seed, 40, True, device)
    assert not bad, f"{len(bad)} values differ, first {bad[:5]}"


@pytest.mark.parametrize("device", DEVICES)
def test_the_impact_branch_on_its_pairs_without_host_syncs_is_the_same(device):
    """with contact_slots and impact_slots the impact branch runs only on the gathered pairs that reach
    it (static shapes, no host sync), with or without rows (one row: no recursion computed): the same bits; too few
    impact slots is an overflow event at check()."""
    assert not compare(1, 240, False, device, contact_slots=7, impact_slots=4096)
    assert not compare(1, 240, False, device, contact_slots=7, impact_slots=4096, rows="free")
    cs, _, _ = reference(1, 240, False)
    with pytest.raises(RuntimeError, match="impact_slots"):
        cc.batched(cs, device, contact_slots=7, impact_slots=1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [1, 2])
def test_the_rows_lever_matches_reference_bit_for_bit(device, seed):
    """The lever (the section 5.5): each world's free balls in the park's first rows, the pairs over the
    rows only (R x N, not N x N): the same bits in both host modes and in the iterative method; a world with more free
    balls than rows is refused."""
    for slots in (None, 7):
        bad = compare(seed, 240, False, device, rows="free", contact_slots=slots)
        assert not bad, f"{len(bad)} values differ, first {bad[:5]}"
    if seed == 1:
        bad = compare(seed, 40, True, device, rows="free")
        assert not bad, f"{len(bad)} values differ, first {bad[:5]}"
        park = VC.VecCollidePark(1, 3, rows=1, device=device)
        ball = {"id": 1, "mass": 1e6, "radius": 10.0, "max_vel": 100.0, "agility": 1.0, "speed_fraction": 1.0,
                "pos": (0.0, 0.0, 0.0), "vel": (0.0, 0.0, 0.0), "free": True, "massive": True}
        with pytest.raises(ValueError, match="rows"):
            park.load([[ball, dict(ball, id=2, pos=(100.0, 0.0, 0.0))]])


def test_the_worlds_exercise_both_methods():
    """The reference's own branch counts on the development worlds: the gradient's and the iterative method's paths,
    several contacts of one ball in one tick, and the partition's adjacent and corner filings."""
    for iterative, n, need in ((False, 240, ("potential.free", "potential.fixed", "potential.recurse",
                                             "potential.light", "potential.early_return", "potential.equal_time",
                                             "potential.earlier_kept", "potential.coincident", "gradient.skip_junk")),
                               (True, 40, ("iterative.free", "iterative.fixed", "iterative.free_contact",
                                           "iterative.static_contact", "iterative.no_impulse", "iterative.ignored",
                                           "iterative.substeps", "iterative.invalidated", "iterative.limited",
                                           "iterative.cap", "iterative.coincident"))):
        tr: Counter = Counter()
        multi = 0
        for c in cc.cases(1, n=n, iterative=iterative):
            rows = sc.reference(c, tr)
            col = sc.S_FIELDS.index("collisions")
            multi += sum(1 for row in rows["S"] if len(set(row[col])) >= 2)
        missing = [k for k in need if tr[k] == 0]
        assert not missing, f"branches never taken: {missing}"
        assert tr["insert.adjacent"] > 0 and tr["insert.corner"] > 0
        if not iterative:
            assert multi > 100


@pytest.mark.skipif(not sc.HARNESS.is_file(), reason="Destiny space harness not built")
@pytest.mark.parametrize("iterative,n", [(False, 80), (True, 24)])
def test_against_compiled_destiny(iterative, n):
    """Straight against the compiled oracle (held-out seed 3), and the reference with it."""
    cs = cc.cases(3, n=n, iterative=iterative)
    ref, und = cc.reference_rows(cs)
    cs = [c for c in cs if c["id"] not in und]
    want = cc.oracle_rows(cs)
    assert want and not cc.differences(want, ref)
    bad = cc.differences(want, cc.batched(cs, "cpu"))
    assert not bad, f"{len(bad)} values differ from compiled Destiny, first {bad[:5]}"


@pytest.mark.parametrize("iterative,n", [(False, 240), (True, 12)])
def test_the_servers_view_in_one_live_bubble_is_the_same(iterative, n):
    """The park models a client's view; on the server (is_master), with every ball in one bubble that is alive (an
    interactive ball in it), the reference decides the same, so the park holds there too."""
    cs = cc.cases(1, n=n, iterative=iterative)
    want, und = cc.reference_rows([dict(c, master=True) for c in cs])
    got = cc.batched(cs, "cpu")
    assert not cc.differences(want, {k: v for k, v in got.items() if k[0] not in und})


def test_undefined_calculate_end_is_marked_where_the_reference_refuses():
    """Destiny's undefined CalculateEnd (a push_back under a held reference, when a path's points fill its vector's
    capacity) occurs in no generated world, nor in this port's 840 iterative scenarios; with every capacity lowered to 3
    points after the first tick (an artificial state, in both), the reference refuses exactly the worlds the park
    marks, and the others still agree bit for bit."""
    from collections import defaultdict
    from exact_ballpark.space import DestinyUndefined, SpaceBallpark
    cs = [c for c in cc.cases(1, n=40, iterative=True) if c["friction"] == 1.0e6 and c["tick"] == 1000]
    refused, want = set(), {}
    for c in cs:
        park = SpaceBallpark(friction=c["friction"], tick_interval_ms=c["tick"], is_master=False, seed=c["seed"],
                             iterative=True)
        out = defaultdict(list)
        by_step = defaultdict(list)
        for rec in c["records"]:
            by_step[rec[1]].append(rec)
        try:
            for step in range(c["steps"] + 1):
                for rec in by_step.get(step, ()):
                    sc._apply(park, rec, c["id"], step, out)
                if step > 0:
                    park.evolve()
                if step == 1:
                    for prop in park.collision_locations.values():
                        prop.capacity = 3
                sc._state(out, c["id"], step, park)
        except DestinyUndefined:
            refused.add(c["id"])
            continue
        want.update(cc.rows_of(out["S"]))
    marked: set = set()
    got = cc.batched(cs, "cpu", undefined=marked, after_tick=_lower_capacity)
    assert refused and refused == marked
    assert not cc.differences(want, {k: v for k, v in got.items() if k[0] not in marked})


def _lower_capacity(park, step: int) -> None:
    if step == 1:
        park.cap = torch.where(park.has_prop, torch.full_like(park.cap, 3), park.cap)


# ---- planted faults: each changes what the reference decides, and the comparison must see it ----

class _Reversed(VC.VecCollidePark):                     # a ball's candidates in reverse order
    def _order_keys(self, pair):
        o = super()._order_keys(pair)
        return torch.where(o == VC.NOT_A_CANDIDATE, o, (1 << 50) - o)


class _ById(VC.VecCollidePark):                         # the candidates by id alone, the partition ignored
    def _order_keys(self, pair):
        o = super()._order_keys(pair)
        return torch.where(o == VC.NOT_A_CANDIDATE, o, torch.zeros_like(o))


class _NoLight(VC.VecCollidePark):                      # COL-710's light-ball rule dropped
    def _light(self, mask, dist):
        pass


class _NoEarly(VC.VecCollidePark):                      # the overlap branch's early return dropped
    def _potential_pairs(self, pair, sync):
        P = super()._potential_pairs(pair, sync)
        P["early"] = torch.zeros_like(P["early"])
        return P


class _Strict(VC.VecCollidePark):                       # '>' for COL-711's '>='
    def _update(self, mask, s, a1, other):
        super()._update(mask & (s != self.last_collision), s, a1, other)


class _NoRecursion(VC.VecCollidePark):                  # Potential's recursion dropped
    def _potential_pairs(self, pair, sync):
        P = super()._potential_pairs(pair, sync)
        P["s1"] = torch.full_like(P["s1"], -1.0)
        return P


class _AnchorMass(VC.VecCollidePark):                   # a fixed ball's own mass in the response, not 1e34
    def _potential_pairs(self, pair, sync):
        free = self.free
        self.free = torch.ones_like(free)
        try:
            return super()._potential_pairs(pair, sync)
        finally:
            self.free = free


class _NoStaleStart(VC.VecCollidePark):                 # each sub-step starts at the last, ignored or not
    KEEP_START_AFTER_IGNORED = False


class _NoCascade(VC.VecCollidePark):                    # RemoveImpactFromFree never called
    def _push(self, mask, slots, times, colliders):
        super()._push(torch.zeros_like(mask), slots, times, colliders)


class _NoPhantom(VC.VecCollidePark):                    # a contact without impulse not counted as found
    def _add_impact(self, jj, mask, time, collider, dv):
        super()._add_impact(jj, mask & (collider != 0), time, collider, dv)


@pytest.mark.parametrize("cls", [_Reversed, _ById, _NoLight, _NoEarly, _Strict, _NoRecursion, _AnchorMass],
                         ids=lambda c: c.__name__)
def test_planted_faults_in_the_gradient_are_caught(cls):
    cs, want, _ = reference(1, 240, False)
    assert cc.differences(want, cc.batched(cs, "cpu", park_cls=cls))


def test_planted_fault_unfused_exp_is_caught(monkeypatch):
    """The gradient's impact branch integrates over s dt with glibc's exp; evaluated without its fusions it differs."""
    cs, want, _ = reference(1, 240, False)
    exp, fma = vec_exp.exp, vec_exp.fma

    def unfused(x):
        vec_exp.fma = lambda a, b, c, b_split=None: a * b + c
        try:
            return exp(x)
        finally:
            vec_exp.fma = fma
    monkeypatch.setattr(vec_exp, "exp", unfused)
    assert cc.differences(want, cc.batched(cs, "cpu"))


# development worlds (seed 1 of 240) where these faults show: the cascade 17, the phantom impact 17, the velocity limit
# 8 and 27, the stale sub-step start 90 and 207
ITERATIVE_FAULT_WORLDS = (8, 17, 27, 90, 207)


@pytest.mark.parametrize("fault", ["no_cascade", "no_phantom", "no_stale_start", "no_velocity_limit"])
def test_planted_faults_in_the_iterative_method_are_caught(monkeypatch, fault):
    cs = [c for c in cc.cases(1, n=240, iterative=True) if c["id"] in ITERATIVE_FAULT_WORLDS]
    want, und = cc.reference_rows(cs)
    cls = {"no_cascade": _NoCascade, "no_phantom": _NoPhantom, "no_stale_start": _NoStaleStart}.get(fault)
    if fault == "no_velocity_limit":
        monkeypatch.setattr(VC, "VELOCITY_LIMIT", 1e12)
    got = cc.batched(cs, "cpu", park_cls=cls)
    assert cc.differences(want, {k: v for k, v in got.items() if k[0] not in und})


# ---- invariances, the switch off, refusals ----

def test_batch_composition_does_not_change_a_world():
    """A world alone, among others and in reversed order gives the same bits (padding, per-world masks)."""
    for iterative, n in ((False, 60), (True, 12)):
        cs = cc.cases(1, n=n, iterative=iterative)
        together = cc.batched(cs, "cpu")
        reverse = cc.batched(list(reversed(cs)), "cpu")
        alone = {}
        for c in cs[:6]:
            alone.update(cc.batched([c], "cpu"))
        assert not cc.differences(together, reverse)
        assert not cc.differences({k: together[k] for k in alone}, alone)


STATE = ("pos", "vel", "old_pos", "old_vel", "goto", "last_g", "last_c", "last_collision", "mode", "speed_fraction",
         "level", "key", "mod", "hits", "hits_n", "cap", "has_prop")


@pytest.mark.parametrize("iterative", [False, True])
def test_inactive_worlds_keep_their_state(iterative):
    """evolve(active): an inactive world's state is unchanged bit for bit; the active worlds tick as a park of their
    own would."""
    worlds = [[cc._ball_dict(r[2]) for r in c["records"] if r[0] == "B"] for c in cc.cases(1, n=12,
                                                                                            iterative=iterative)]
    n = max(len(w) for w in worlds)

    def park(ws):
        p = VC.VecCollidePark(len(ws), n, iterative=iterative)
        p.load(ws)
        p.goto_point(p.exists & p.free, torch.zeros(len(ws), n, 3, dtype=torch.float64))
        return p
    mixed, alone = park(worlds), park(worlds[::2])
    active = torch.arange(len(worlds)) % 2 == 0
    before = {k: getattr(mixed, k).clone() for k in STATE}
    for _ in range(4):
        mixed.evolve(active)
        alone.evolve()
    bits = lambda t: t.view(torch.int64) if t.is_floating_point() else t
    for k in STATE:
        assert torch.equal(bits(getattr(mixed, k)[~active]), bits(before[k][~active])), k
        assert torch.equal(bits(getattr(mixed, k)[active]), bits(getattr(alone, k))), k
    assert not torch.equal(mixed.pos[active], before["pos"][active])


@pytest.mark.parametrize("seed", [1, 2])
def test_collisions_off_is_vecballpark_unchanged(seed):
    """With collisions off, one ball a world: exact_ballpark.vec.VecBallpark's tick bit for bit."""
    groups: dict = {}
    for c in movement_cases(seed):
        groups.setdefault((c[1], c[2]), []).append(c)
    for (friction, tick), g in groups.items():
        rows = [{"mass": c[5], "agility": c[6], "max_vel": c[7], "speed_fraction": c[8], "pos": c[9:12],
                 "vel": c[12:15], "goto": c[15:18], "mode": c[4]} for c in g]
        ref = VecBallpark(len(g), friction=friction, tick_interval_ms=tick)
        ref.load(rows)
        park = VC.VecCollidePark(len(g), 1, friction=friction, tick_interval_ms=tick, collisions=False)
        park.load([[{"id": 1, "mass": r["mass"], "radius": 0.0, "max_vel": r["max_vel"], "agility": r["agility"],
                     "speed_fraction": r["speed_fraction"], "pos": r["pos"], "vel": r["vel"], "free": True,
                     "massive": True}] for r in rows])
        park.agility = ref.agility[:, None].clone()        # VecBallpark's load keeps agility 1.0 for 0 (as AddBall)
        park.speed_fraction = ref.speed_fraction[:, None].clone()
        park.time_factor = ref.time_factor[:, None].clone()
        park.goto = ref.goto[:, None].clone()
        park.mode = ref.mode[:, None].clone()
        for _ in range(max(c[3] for c in g)):
            ref.evolve()
            park.evolve()
            assert torch.equal(park.pos[:, 0].view(torch.int64), ref.pos.view(torch.int64))
            assert torch.equal(park.vel[:, 0].view(torch.int64), ref.vel.view(torch.int64))


def test_refusals():
    park = VC.VecCollidePark(1, 2)
    ball = {"id": 1, "mass": 1e6, "radius": 10.0, "max_vel": 100.0, "agility": 1.0, "speed_fraction": 1.0,
            "pos": (0.0, 0.0, 0.0), "vel": (0.0, 0.0, 0.0), "free": True, "massive": True}
    with pytest.raises(ValueError, match="distinct"):
        park.load([[ball, dict(ball)]])
    with pytest.raises(ValueError, match="positive"):
        park.load([[dict(ball, id=0)]])
    with pytest.raises(ValueError, match="mass"):
        park.load([[dict(ball, mass=0.0)]])
    with pytest.raises(ValueError, match="edge"):
        park.load([[dict(ball, pos=(3.2985348833e13, 0.0, 0.0))]])
    with pytest.raises(ValueError, match="slots"):
        park.load([[ball, dict(ball, id=2), dict(ball, id=3)]])

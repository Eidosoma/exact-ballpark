# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""OnTick's intervals in the four batched parks: VecOrientPark, VecModesPark,
VecCombatPark and VecComposedPark driven by `on_tick` (every world at its own simulation clock; the worlds with an
interval left ticking together under OnTick's per-world mask, `evolve(active=...)`, and the CUDA graphs' masked
captures) against FullBallpark.on_tick (Ballpark.cpp:219-330, TIME-702 to TIME-705), one park a world, bit for bit after
every call: every field the batch holds of every ball, which balls exist, the DoCollision events, the answers, and the
clocks (mTime, mFirstTime, mCurrentTime). The reference on these OnTick scenarios is held to Destiny's compiled code
(the full harness), and the batch to it directly. A masked tick leaves every per-row tensor of the rows outside the
mask as it was and gives the others the unmasked tick's values. Development seeds 1 and 7; held-out seed 19, first run
after the generator (tests/parks_ontick_cases.py) was frozen. Planted faults must each be caught."""

from __future__ import annotations

import pytest
import torch

import full_cases as fc
import modes_batch as mb
import orient_batch as ob
import parks_ontick_cases as T
import exact_ballpark.vec_combat as vc
import exact_ballpark.vec_orient as vo

CUDA = torch.cuda.is_available()
_REF: dict = {}


def _refs(family: str, seed: int, nb: int, nw: int) -> tuple:
    key = (family, seed, nb, nw)
    if key not in _REF:
        cfgs = T.worlds(family, seed, nb, nw)
        _REF[key] = (cfgs, [[T.reference_world(family, cfg, w) for w in cfg["worlds"]] for cfg in cfgs])
    return _REF[key]


def _full_refs(which: str, seed: int, n: int) -> tuple:
    key = (which, seed, n)
    if key not in _REF:
        cs = (T.flight if which == "flight" else T.turning)(seed, n)
        fields = mb.FIELDS_FULL if which == "flight" else ob.FIELDS
        _REF[key] = (cs, *T.references_full(cs, fields))
    return _REF[key]


def _check_worlds(family: str, seed: int, device: str, path: str = "torch", graph: bool = False, nb: int = 4,
                  nw: int = 8) -> dict:
    cfgs, refs = _refs(family, seed, nb, nw)
    r = T.compare_worlds(family, cfgs, refs, device, path, graph)
    c = r["counts"]
    assert r["n_bad"] == 0, (r["n_bad"], r["bad"])
    assert c.get("worlds", 0) >= 8, c
    assert c.get("masked rows out", 0) > 0 and c.get("tick.first", 0) > 0 and c.get("tick.too_early", 0) > 0, c
    return c


def _check_orient(seed: int, device: str = "cpu", **kw) -> tuple:
    cs, want, wclk, hits = _full_refs("turn", seed, 40)
    got, gclk, seen = T.batched_orient(cs, device, **kw)
    bad = ob.differences(want, got)
    assert not bad, bad[:5]
    cbad = [k for k in wclk if wclk[k][:2] != gclk[k][:2]]
    assert not cbad, [(k, wclk[k], gclk[k]) for k in cbad[:5]]
    assert seen[1] > 0 and hits.get("tick.first", 0) and hits.get("tick.too_early", 0) and hits.get("tick.catch_up")
    return seen, hits


def _check_modes(seed: int, graph: bool = False, device: str = "cuda") -> tuple:
    cs, want, wclk, hits = _full_refs("flight", seed, 40)
    got, gclk, seen = T.batched_modes(cs, device, graph=graph)
    bad = mb.differences(want, got, mb.FIELDS_FULL)
    assert not bad, bad[:5]
    cbad = [k for k in wclk if wclk[k] != gclk[k]]
    assert not cbad, [(k, wclk[k], gclk[k]) for k in cbad[:5]]
    assert seen[1] > 0 and hits.get("tick.first", 0) and hits.get("tick.too_early", 0) and hits.get("tick.catch_up")
    return seen, hits


# ============================================================================================== on the CPU

@pytest.mark.parametrize("seed", [1, 7], ids=["development_1", "development_7"])
def test_orient_on_tick_matches_fullballpark(seed):
    _check_orient(seed)


@pytest.mark.parametrize("family", T.FAMILIES)
@pytest.mark.parametrize("seed", [1, 7], ids=["development_1", "development_7"])
def test_worlds_on_tick_match_fullballpark_cpu(family, seed):
    c = _check_worlds(family, seed, "cpu")
    if family != "leftovers":
        assert c.get("tick.resync", 0) > 0, c                     # the master's batches resynchronise


@pytest.mark.parametrize("seed", [1, 7, 19], ids=["development_1", "development_7", "held_out_19"])
def test_modes_on_tick_matches_fullballpark_on_the_cpu(seed):
    """VecModesPark's masked tick on tensors (OnTick's per-world mask) on the CPU."""
    _check_modes(seed, device="cpu")


@pytest.mark.parametrize("family", ("orient",) + T.FAMILIES)
def test_on_tick_held_out_cpu(family):
    if family == "orient":
        _check_orient(19)
    else:
        _check_worlds(family, 19, "cpu")


def _row_state(park) -> dict:
    return {k: getattr(park, k).clone() for k in park.row_tensors()}


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype == torch.float64:
        return torch.equal(a.contiguous().view(torch.int64), b.contiguous().view(torch.int64))
    return torch.equal(a, b)


def _twins(kind: str):
    """Two identical parks of a kind, loaded with one world a few rows apart (CPU), and their tick on tensors."""
    if kind.startswith("orient"):
        cs = T.turning(1, 16)
        cs = [c for c in cs if bool(c["do"]) == (kind == "orient_do")][:8]
        parks = []
        for _ in range(2):
            got = {}
            members = [(c, rec[2]) for c in cs for rec in c["records"] if rec[0] == "B"]
            p = vo.VecOrientPark(len(members), friction=cs[0]["friction"], tick_interval_ms=cs[0]["tick"],
                                 dynamical_orientation=kind == "orient_do")
            p.load([{"mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"], "pos": b["pos"],
                     "vel": b["vel"], "agility": b["agility"], "speed_fraction": b["sf"],
                     "max_ang_vel": b["max_ang_vel"], "rot_agility": b["rot_agility"]} for _, b in members])
            goal = torch.tensor([[1e4 * (i % 3 - 1), 5e3 * (i % 2), -7e3] for i in range(p.n)], dtype=torch.float64)
            p.goto_point(torch.ones(p.n, dtype=torch.bool), goal)
            parks.append((p, got))
        return parks[0][0], parks[1][0], lambda p, m: p.evolve(active=m)
    family = "composed" if kind == "composed" else "combat"
    cfg = T.worlds(family, 1, 1, 6)[0]
    if family == "combat":
        cfg = {**cfg, "worlds": [w for w in cfg["worlds"] if T.torch_domain(family, w)][:4]}
    pa = T.batched_worlds(family, {**cfg, "steps": 3}, "cpu")[1]
    pb = T.batched_worlds(family, {**cfg, "steps": 3}, "cpu")[1]
    return pa, pb, lambda p, m: p.evolve_torch(active=m)


@pytest.mark.parametrize("kind", ["orient", "orient_do", "combat", "composed"])
def test_a_masked_tick_puts_back_every_row_outside_its_mask(kind):
    """Twin parks: one ticks under a mask, the other unmasked; inside the mask every per-row tensor is the unmasked
    tick's, outside it every per-row tensor is as it was before the tick (bit for bit); a mask of every row is the
    unmasked tick; an empty mask changes no row."""
    pa, pb, tick = _twins(kind)
    if hasattr(pa, "world"):
        w = pa.world
        m = (w % 2 == 0)
    else:
        m = torch.arange(pa.n) % 3 != 0
    assert bool(m.any()) and not bool(m.all())
    before = _row_state(pa)
    tick(pa, m)
    tick(pb, None)
    names = pa.row_tensors()
    assert names == pb.row_tensors()
    changed = [k for k in names if not _same(getattr(pb, k), before[k])]
    assert len(changed) >= 4, changed
    for k in names:
        assert _same(getattr(pa, k)[m], getattr(pb, k)[m]), k
        assert _same(getattr(pa, k)[~m], before[k][~m]), k
    qa, qb, _ = _twins(kind)                                      # fresh twins: a mask of every row
    tick(qa, torch.ones(qa.n, dtype=torch.bool))
    tick(qb, None)
    for k in names:
        assert _same(getattr(qa, k), getattr(qb, k)), k
    before = _row_state(pa)
    tick(pa, torch.zeros(pa.n, dtype=torch.bool))
    for k in names:
        assert _same(getattr(pa, k), before[k]), k


def test_a_worlds_rows_must_share_its_time():
    cfg = T.worlds("combat", 1, 1, 3)[0]
    p = T.batched_worlds("combat", {**cfg, "steps": 0}, "cpu")[1]
    sim = torch.full((p.n,), 10 ** 8, dtype=torch.int64)
    sim[1] += 1
    with pytest.raises(ValueError, match="different simulation times"):
        p.on_tick(sim)


def test_a_park_without_on_tick_holds_the_tensors_it_held():
    """A park never driven by on_tick holds no OnTick clock, and its tick is the unmasked one (no `_ticked`)."""
    cfg = T.worlds("combat", 1, 1, 3)[0]
    p = T.batched_worlds("combat", {**cfg, "steps": 0}, "cpu")[1]
    p.evolve_torch()
    assert not hasattr(p, "time") and not hasattr(p, "first_time") and not hasattr(p, "_ticked")
    assert "time" not in p.row_tensors() and "clock" in p.row_tensors()


# ---- compiled Destiny (the full harness) on the OnTick scenarios ----

@pytest.mark.skipif(not fc.HARNESS.exists(), reason="the compiled full harness is not built")
def test_the_reference_and_the_orient_batch_match_destinys_compiled_code():
    """On OnTick's scenarios (turn_cases' batch kinds and flight_cases', seed 1): the reference's ball rows and its
    clock (mCurrentTime) after every call equal compiled Destiny's, and VecOrientPark's rows equal the harness's
    directly."""
    for which in ("turn", "flight"):
        cs, want, wclk, _ = _full_refs(which, 1, 40)
        fields = mb.FIELDS_FULL if which == "flight" else ob.FIELDS
        idx = [fc.S_FIELDS.index(f) for f in fields]
        orc = fc.oracle(cs)
        theirs = {(row[0], row[1], row[2]): tuple(row[i] for i in idx) for row in orc["S"]}
        assert not mb.differences(want, theirs, fields), which
        clocks = {(row[0], row[1]): row[2] for row in orc["P"]}
        assert clocks and all(clocks[k] == wclk[k][2] for k in wclk), which
        if which == "turn":
            got, _, _ = T.batched_orient(cs, "cpu")
            assert not ob.differences(theirs, got)


# ---- planted faults (each must be caught) ----

def _no_put_back(park, active, tick, in_place):
    tick()


def _clock_kept_out(monkeypatch):
    monkeypatch.setattr(vc.VecCombatPark, "_ROW_EXCLUDED", vc.VecCombatPark._ROW_EXCLUDED + ("clock",))


def _remainder_dropped(monkeypatch):
    orig = vo.VecOrientPark.on_tick

    def on_tick(self, sim_time, tick=None, after=None):
        old = getattr(self, "time", None)
        old = None if old is None else old.clone()
        out = orig(self, sim_time, tick, after)
        sim = torch.as_tensor(sim_time, dtype=torch.int64).to(self.dev)
        self.time = torch.where(self.time != (old if old is not None else torch.zeros_like(self.time)), sim,
                                self.time)
        return out
    monkeypatch.setattr(vo.VecOrientPark, "on_tick", on_tick)


def _first_like_the_rest(monkeypatch):
    orig = vo.VecOrientPark.on_tick

    def on_tick(self, sim_time, tick=None, after=None):
        if getattr(self, "time", None) is None:
            self.time = torch.zeros(self.n, dtype=torch.int64, device=self.dev)
            self.first_time = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        return orig(self, sim_time, tick, after)
    monkeypatch.setattr(vo.VecOrientPark, "on_tick", on_tick)


FAULTS = {
    "the rows outside the mask ticked": ("combat", lambda mp: mp.setattr(vc, "masked_tick", _no_put_back)),
    "a world without a tick keeps no clock": ("combat", _clock_kept_out),
    "no resynchronisation on the master": ("combat", lambda mp: mp.setattr(vc.VecCombatPark, "_on_tick_master",
                                                                         lambda self: False)),
    "the remainder dropped": ("combat", _remainder_dropped),
    "the first call runs every interval": ("combat", _first_like_the_rest),
    "BringOutTheDeads not called": ("leftovers", lambda mp: mp.setattr(vc.VecCombatPark, "_bring_out_on_tick",
                                                                      lambda self: None)),
    "a masked tick's events not limited to its rows": ("combat", lambda mp: mp.setattr(
        vc.VecCombatPark, "_ticked_rows", lambda self, sel: sel)),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    family, plant = FAULTS[fault]
    nb, nw = (6, 12) if family == "leftovers" else (4, 8)
    cfgs, refs = _refs(family, 1, nb, nw)
    plant(monkeypatch)
    r = T.compare_worlds(family, cfgs, refs, "cpu", "torch")
    assert r["n_bad"] > 0, (fault, r["counts"])


# ============================================================================================== on the GPU

@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("seed", [1, 7], ids=["development_1", "development_7"])
def test_modes_on_tick_matches_fullballpark_on_the_gpu(seed, graph):
    _check_modes(seed, graph)


@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_modes_on_tick_held_out_on_the_gpu(graph):
    _check_modes(19, graph)


@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("seed", [1, 19], ids=["development_1", "held_out_19"])
def test_orient_kernels_on_tick_on_the_gpu(seed, graph):
    """VecOrientPark on the card: dynamical orientation by its kernel (kernel="cuda"), the yaw, pitch and roll on the
    device (ypr="cuda"); eager masked ticks and TickGraph's masked capture."""
    _check_orient(seed, "cuda", kernel="cuda", ypr="cuda", graph=graph)


@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize("graph", [False, True], ids=["kernels", "graph"])
@pytest.mark.parametrize("family", T.FAMILIES)
@pytest.mark.parametrize("seed", [1, 19], ids=["development_1", "held_out_19"])
def test_worlds_on_tick_on_the_gpu(family, seed, graph):
    _check_worlds(family, seed, "cuda", "kernel", graph, nb=4 if not graph else 6, nw=8)


@pytest.mark.skipif(not (CUDA and fc.HARNESS.exists()), reason="needs a CUDA GPU and the compiled full harness")
def test_the_modes_batch_matches_destinys_compiled_code_on_the_gpu():
    cs, _, wclk, _ = _full_refs("flight", 1, 40)
    idx = [fc.S_FIELDS.index(f) for f in mb.FIELDS_FULL]
    orc = fc.oracle(cs)
    theirs = {(row[0], row[1], row[2]): tuple(row[i] for i in idx) for row in orc["S"]}
    got, gclk, _ = T.batched_modes(cs, "cuda")
    assert not mb.differences(theirs, got, mb.FIELDS_FULL)
    clocks = {(row[0], row[1]): row[2] for row in orc["P"]}
    assert all(clocks[k] == gclk[k][2] for k in gclk)


@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
def test_a_graphs_put_back_left_out_is_caught(monkeypatch):
    """The planted fault of a masked graph capture without its put-back (the rows outside the mask tick in every
    replay): caught on VecModesPark's graph."""
    monkeypatch.setattr(vo.GraphMask, "put_back", lambda self: None)
    cs, want, wclk, _ = _full_refs("flight", 1, 40)
    got, gclk, _ = T.batched_modes(cs, "cuda", graph=True)
    assert mb.differences(want, got, mb.FIELDS_FULL) or any(wclk[k] != gclk[k] for k in wclk)


@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
def test_a_masked_graph_of_every_row_is_the_unmasked_graph():
    """VecModesPark's masked graph replayed with every row in the mask equals the unmasked graph, every state tensor
    every tick; with half the worlds, the rows outside keep every state tensor."""
    from exact_ballpark import vec_modes_cuda as K
    cs = T.flight(7, 24)
    parks = []
    for masked in (False, True):
        members = [(c, rec[2]) for c in cs for rec in c["records"] if rec[0] == "B"]
        from exact_ballpark.vec_modes import VecModesPark
        p = VecModesPark(len(members), device="cuda", dynamical_orientation=True)
        p.load([{"id": b["id"], "world": c["id"], "time": c["start"], "mass": b["mass"], "radius": b["radius"],
                 "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
                 "speed_fraction": b["sf"], "free": b["free"], "massive": b["massive"],
                 "interactive": b["interactive"], "max_ang_vel": b["max_ang_vel"],
                 "rot_agility": b["rot_agility"]} for c, b in members])
        goal = torch.tensor([[2e4 * (i % 5 - 2), 3e3 * (i % 2), -9e3] for i in range(p.n)], dtype=torch.float64)
        p.goto_point(torch.ones(p.n, dtype=torch.bool, device="cuda"), goal.cuda())
        parks.append((p, K.ModesGraph(p, masked=masked)))
    (pa, ga), (pb, gb) = parks
    for _ in range(6):
        ga.tick()
        gb.tick(None)
    for name, _, _ in K.STATE_TENSORS:
        assert _same(getattr(pa, name), getattr(pb, name)), name
    m = (pb.world % 2 == 0)
    before = {name: getattr(pb, name).clone() for name, _, _ in K.STATE_TENSORS}
    gb.tick(m)
    ga.tick()
    for name, _, _ in K.STATE_TENSORS:
        assert _same(getattr(pb, name)[~m], before[name][~m]), name
        assert _same(getattr(pb, name)[m], getattr(pa, name)[m]), name

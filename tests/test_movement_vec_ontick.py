# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""OnTick's intervals on Destiny's master server (Ballpark.cpp:219-330, TIME-702 to TIME-705):
exact_ballpark.vec_server_space.SpaceServer.on_tick over SpaceCollideAdapter, every world at its own simulation clock, bit
for bit against the scalar server layer driven through its park's OnTick (exact_ballpark.full.FullBallpark.on_tick with
the Ticker at DoPreTick and send_batch at DoPostTick) on generated worlds and clocks (tests/ontick_cases.py:
development seeds 1 and 7, held-out seed 19, first run after the generator's freeze): each world's sends, call by call,
and every ship's row and clock. The per-world mask with every world in it is the unmasked tick, bit for bit; the
planted faults must be caught. The CPU cases run everywhere; the rest need a GPU."""

import pytest
import torch

import global_cases as G
import ontick_cases as O
import server_cases as S
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_space as VSS

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
DEV, DEV2, HELD_OUT = 1, 7, 19
_RUNS: dict = {}

# what development seed 1 reached (60 worlds x 14 calls; the minimums below are about half of it)
MINIMUM = {"worlds too early": 50, "first calls": 60, "resynchronisations": 50, "intervals run": 500,
           "remainders carried": 250}


def run(seed: int, device: str = "cpu", n: int = 60, calls: int = 14, **kw) -> dict:
    k = (seed, device, n, calls, tuple(sorted(kw.items())))
    if k not in _RUNS:
        res = O.run(O.cases(seed, n, calls), device, **kw)
        res.pop("server")
        _RUNS[k] = res
    return _RUNS[k]


def assert_exact(res: dict, minimum: bool = False) -> None:
    assert res["sends"] == [] and res["rows"] == [], (res["sends"][:3], res["rows"][:3])
    assert res["flagged"] == [] and res["updates"] > 0
    cov = res["coverage"]
    for key in ("worlds too early", "first calls", "resynchronisations", "remainders carried"):
        assert cov.get(key, 0) > 0, key
    assert all(cov.get(f"ticks in a call: {k}", 0) > 0 for k in (0, 1, 2, 3, 4, 5)), cov
    if minimum:
        for key, low in MINIMUM.items():
            assert cov.get(key, 0) >= low, (key, cov.get(key, 0))


def test_on_tick_matches_the_reference_development_worlds():
    assert_exact(run(DEV), minimum=True)


def test_on_tick_matches_the_reference_second_development_worlds():
    assert_exact(run(DEV2))


def test_held_out_on_tick_matches_the_reference():
    """Held-out seed 19 (its first run after the generator's freeze)."""
    assert_exact(run(HELD_OUT))


def test_a_mask_of_every_world_is_the_unmasked_tick_bit_for_bit():
    """pre_tick, evolve and post_tick with every world in the mask give the batches, sends and park state the unmasked
    tick gives (tests/global_cases.py's worlds and calls, development seed 1)."""
    cs = G.server_cases(DEV, 30, 10)
    a, _ = G.build_server(cs, "cpu")
    b, _ = G.build_server(cs, "cpu")
    every = torch.ones(len(cs), dtype=torch.bool)
    names = ("pos", "vel", "last_c", "last_collision", "mode", "level", "key", "mod", "bubble", "glob", "hits_n")
    for t in range(10):
        for srv in (a, b):
            for kind, *data in S.pushes(cs, t, srv.dev):
                srv.push(kind, *data)
        a.tick()
        b.pre_tick(mask=every)
        b.evolve(mask=every)
        b.post_tick(mask=every)
        assert S.canon([S._listify(x) for x in VSS.decode(a)]) == S.canon([S._listify(x) for x in VSS.decode(b)])
        for k in names:
            x, y = getattr(a.a.park, k), getattr(b.a.park, k)
            assert torch.equal(x.view(torch.int64) if x.dtype == torch.float64 else x,
                               y.view(torch.int64) if y.dtype == torch.float64 else y), (t, k)
        for k, v in a.space.st.items():
            if v is not None:
                assert torch.equal(v, b.space.st[k]), (t, k)


def test_refusals():
    import server_space_cases as SSC
    from exact_ballpark import vec_space as SP
    srv, _ = SSC.build(SSC.cases(1, 2, "orient", ticks=1), "orient", "cpu")       # VecOrientPark's adapter
    # SpaceServer.on_tick, its masked evolve and the layer's masked tick run over VecOrientPark and
    # VecModesPark now (tests/test_movement_vec_server_ontick.py); the park's tick by worlds stays VecCollidePark's
    # (the other parks take OnTick's mask as rows)
    with pytest.raises(ValueError, match="VecCollidePark only"):
        SP.evolve_skipping(srv.a.park, torch.zeros(srv.a.R, dtype=torch.bool), worlds=torch.tensor([True, False]))


# ---- planted faults (each must make the comparison differ, development seed 1) ----

def _no_resynchronisation(monkeypatch):
    monkeypatch.setattr(VSS, "MAX_CATCH_UP_TICKS", 10 ** 9)


def _remainder_dropped(monkeypatch):
    on_tick = VSS.SpaceServer.on_tick

    def faulty(self, sim_time):
        before = self.time.clone()
        out = on_tick(self, sim_time)
        sim = torch.as_tensor(sim_time, dtype=torch.int64).to(self.dev)
        self.time = torch.where(self.time != before, sim, self.time)
        return out
    monkeypatch.setattr(VSS.SpaceServer, "on_tick", faulty)


def _first_call_with_events(monkeypatch):
    on_tick = VSS.SpaceServer.on_tick

    def faulty(self, sim_time):
        self.first_time = torch.zeros_like(self.first_time)
        return on_tick(self, sim_time)
    monkeypatch.setattr(VSS.SpaceServer, "on_tick", faulty)


def _queue_lost_without_a_tick(monkeypatch):
    pre = VSS.SpaceServer.pre_tick

    def faulty(self, snapshot=None, mask=None):
        pre(self, snapshot, mask)
        self.slots = []
    monkeypatch.setattr(VSS.SpaceServer, "pre_tick", faulty)


def _park_ticks_every_world(monkeypatch):
    evolve = VSS.SpaceServer.evolve

    def faulty(self, mask=None):
        if mask is None:
            return evolve(self)
        self.space.tick(self.a.park)
        self.a.time.add_(mask.to(torch.int64).to(self.dev))
    monkeypatch.setattr(VSS.SpaceServer, "evolve", faulty)


def _stream_bytes_ignored(monkeypatch):
    push = VSS.SpaceServer._push_with_loc

    def faulty(self, k, mask, ball, target, f, n, loc):
        return push(self, k, mask, ball, target, f, None if k == VS.GLOBAL_ON else n, loc)
    monkeypatch.setattr(VSS.SpaceServer, "_push_with_loc", faulty)


FAULTS = {"no_resynchronisation": _no_resynchronisation, "remainder_dropped": _remainder_dropped,
          "stream_bytes_ignored_by_the_duplicate_test": _stream_bytes_ignored,
          "first_call_with_events": _first_call_with_events, "queue_lost_without_a_tick": _queue_lost_without_a_tick,
          "park_ticks_every_world": _park_ticks_every_world}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_a_planted_fault_is_caught(fault, monkeypatch):
    """The master's resynchronisation dropped, the remainder dropped, AddBallsToPark's stream left out of the duplicate
    test (one ball's stream changes within one queue across a world's first OnTick), the first call run with the tick
    events, a world without a tick losing its queue, the park ticking every world: each is seen."""
    FAULTS[fault](monkeypatch)
    res = O.run(O.cases(DEV, 60, 14), "cpu")
    assert len(res["sends"]) + len(res["rows"]) > 0


# ---- on the card: the park's torch path and the kernel ----

@CUDA
@pytest.mark.parametrize("kernel", ["torch", "cuda"])
def test_on_tick_on_the_card_matches_the_reference(kernel):
    assert_exact(run(DEV, "cuda", kernel=kernel))


@CUDA
def test_held_out_on_tick_on_the_card_matches_the_reference():
    """Held-out seed 19 on the card (the kernel; its first card run after the generator's freeze)."""
    assert_exact(run(HELD_OUT, "cuda", kernel="cuda"))

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's delayed removal on the master batched (exact_ballpark.vec_space_removal: RemoveBall with a
delay and BringOutTheDeads with its 7-ball buffer, TIME-713) against the scalar reference
(exact_ballpark.full.FullBallpark, master), bit for bit, every step, over generated worlds (tests/space_removal_cases.py):
every ball's partition state, flight fields, moribund and massive flags and removal tick, each world's moribundBalls,
the events in order (the DoFinalCleanups of a bring-out in ascending id), the queries' answers.

The order is the reference's, ascending id, never Destiny's (heap addresses: TIME-713 cannot pass by construction).

On the CPU (VecOrientPark): development seed 1 and held-out seed 5 (first run after the generator was frozen), the
coverage (delayed and immediate removals, the buffer, the kill count binding), a constructed bring-out of 9 balls
(the 7 lowest ids first), five planted faults (each must be caught). On the GPU (skipped without one): VecModesPark
(follows, orbits and warps given to waiting balls: removal ticks moved by commands) and VecOrientPark, a CUDA graph of
the bring-out with the layer's tick against eager ticks, and a sixth planted fault (the removal tick kept apart from
mEffectStamp, which only VecModesPark's commands can move).
"""

from __future__ import annotations

import pytest
import torch

import space_removal_cases as SRC
from exact_ballpark import vec_space as SP
from exact_ballpark import vec_space_removal as VR

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
HELD_OUT = 5                                                # first run after the generator was frozen
_RUNS: dict = {}


def run(seed: int, n: int = 200, park: str = "orient", steps: int = 16, device: str = "cpu") -> dict:
    k = (seed, n, park, steps, device)
    if k not in _RUNS:
        _RUNS[k] = SRC.compare(SRC.cases(seed, n, park, steps), device, park)
    return _RUNS[k]


# VecOrientPark has no follow or warp: no command moves a waiting ball's removal tick there (no "stamp_moved" minimum)
MINIMUM = {"remove.delayed": 300, "remove.now": 100, "remove.buffer": 100, "limit_binds": 5}


def assert_exact(res: dict, minimum: dict, n: int = 200) -> None:
    """No difference; at most 4 % of the worlds refused by the layer (they leave its domain, as its own `check()`
    says: all of them by the live-box bound, FLAG_BOXES, in grown worlds) and so excluded; the coverage."""
    assert len(res["flagged"]) <= n // 25, res["flagged"][:10]
    assert not res["diffs"], [str(d)[:400] for d in res["diffs"][:3]]
    for key, low in minimum.items():
        assert res["coverage"].get(key, 0) >= low, (key, res["coverage"])
    assert res["compared"]["final_cleanups"] > 100 and res["compared"]["waiting_ball_steps"] > 300, res["compared"]


# ==================================================================================================== on the CPU

def test_matches_the_reference_development_worlds():
    assert_exact(run(1), MINIMUM)


def test_matches_the_reference_held_out_worlds():
    assert_exact(run(HELD_OUT), MINIMUM)


def test_a_bring_out_takes_seven_from_the_buffer_in_ascending_id():
    """A constructed world: 9 balls removed with a delay of 2 at time 0; at time 1 all 9 are within 2 ticks of their
    removal tick and none has passed it: the bring-out takes the 7 lowest ids, their DoFinalCleanups in ascending id;
    the other 2 go at time 2 (within the buffer, the count fresh). Destiny would take 7 in heap order (TIME-713)."""
    cs = SRC.cases(1, 1, "orient", 1)
    w = cs[0]
    w["balls"] = [dict(w["balls"][0], id=bid, pos=[float(k) * 2e4, 0.0, 0.0], occluder=False, massive=False,
                       free=True, interactive=k == 0) for k, bid in enumerate([907, 13, 512, 44, 3, 88, 610, 21, 77])]
    ids = [b["id"] for b in w["balls"]]
    w["steps"] = [[("remove_later", bid, 2) for bid in ids if bid != 907], [], [], []]
    w["steps"][0].append(("remove_later", 907, 2))
    res = SRC.compare(cs, "cpu", "orient")
    assert not res["diffs"], res["diffs"][:3]
    bw = SRC.BatchedWorlds(cs, "cpu", "orient", hits=False)
    events = []
    for t in range(3):
        bw.space.clear_events()
        bw.step(t)
        events.append([e[1] for e in bw.space.events(0) if e[0] == "DoFinalCleanup"])
    assert events[0] == [], events
    assert events[1] == sorted(ids)[:7], events
    assert events[2] == sorted(ids)[7:], events
    assert res["coverage"]["limit_binds"] == 1 and res["coverage"]["remove.buffer"] == 9, res["coverage"]


class _Descending(VR.DelayedRemoval):
    """Planted: moribundBalls walked in descending id (which 7 the buffer takes, the removals' order)."""

    def bring_out_the_deads(self):
        sp = self.space
        saved = sp.by_id
        n = (saved >= 0).sum(dim=1, keepdim=True)
        k = torch.arange(sp.N, device=saved.device)[None, :]
        sp.by_id = torch.where(k < n, torch.gather(saved, 1, (n - 1 - k).clamp(min=0)), -1).contiguous()
        try:
            super().bring_out_the_deads()
        finally:
            sp.by_id = saved


class _NoKillCount(VR.DelayedRemoval):
    """Planted: every ball within the buffer taken (no kill count)."""

    def due(self):
        old = VR.KILL_COUNT
        VR.KILL_COUNT = 1 << 20
        try:
            return super().due()
        finally:
            VR.KILL_COUNT = old


class _OneTickBuffer(VR.DelayedRemoval):
    """Planted: a buffer of one tick (stamp - time - 1 < 0)."""

    def due(self):
        old = VR.BUFFER_TICKS
        VR.BUFFER_TICKS = 1
        try:
            return super().due()
        finally:
            VR.BUFFER_TICKS = old


class _OwnRemovalTick(VR.DelayedRemoval):
    """Planted: the removal tick kept apart from mEffectStamp (a command no longer moves it)."""

    def __init__(self, space):
        super().__init__(space)
        self.own = self._get("effect_stamp").clone()

    def remove_ball(self, mask, ball, delay):
        before = self._get("effect_stamp").clone()
        super().remove_ball(mask, ball, delay)
        after = self._get("effect_stamp")
        self.own = torch.where(after != before, after, self.own)

    def due(self):
        cur = self._get("effect_stamp").clone()
        self._put("effect_stamp", self.own)
        try:
            return super().due()
        finally:
            self._put("effect_stamp", cur)


class _KeepsMassive(VR.DelayedRemoval):
    """Planted: a ball removed with a delay stays massive."""

    def remove_ball(self, mask, ball, delay):
        before = self._get("massive").clone()
        super().remove_ball(mask, ball, delay)
        self._put("massive", before | self._get("massive"))


class _NoStop(VR.DelayedRemoval):
    """Planted: RemoveBall neither stops the ball nor releases its followers."""

    def _stop_and_release(self, m):
        pass


FAULTS = {"descending_id": _Descending, "no_kill_count": _NoKillCount, "one_tick_buffer": _OneTickBuffer,
          "keeps_massive": _KeepsMassive, "no_stop": _NoStop}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_a_planted_fault_is_caught(fault):
    res = SRC.compare(SRC.cases(1, 120, "orient", 12), "cpu", "orient", removal_cls=FAULTS[fault])
    assert res["diffs"], fault


# ==================================================================================================== on the GPU

@CUDA
def test_matches_the_reference_on_vec_modes_park():
    """VecModesPark on the card (follows, orbits and warps; commands given to waiting balls), seeds 1 and 5."""
    for seed in (1, HELD_OUT):
        res = run(seed, 150, "modes", 16, "cuda")
        assert_exact(res, {"remove.delayed": 200, "remove.now": 50, "remove.buffer": 50, "limit_binds": 3,
                           "stamp_moved": 10}, 150)


def test_matches_the_reference_on_vec_modes_park_on_the_cpu():
    """the same worlds, VecModesPark's tick on tensors on the CPU."""
    for seed in (1, HELD_OUT):
        res = run(seed, 150, "modes", 16, "cpu")
        assert_exact(res, {"remove.delayed": 200, "remove.now": 50, "remove.buffer": 50, "limit_binds": 3,
                           "stamp_moved": 10}, 150)


@CUDA
def test_matches_the_reference_on_vec_orient_park_on_the_card():
    assert_exact(run(1, 200, "orient", 16, "cuda"), MINIMUM)


@CUDA
def test_a_cuda_graph_of_the_bring_out_and_the_tick_equals_eager_ticks():
    """Two copies of the same worlds (VecModesPark): their delayed removals pushed eagerly; then 8 ticks, one copy
    eager (BringOutTheDeads, then the layer's tick), the other the same captured once as a CUDA graph and replayed;
    every tensor of the park, the partition and moribundBalls equal after every tick."""
    cs = SRC.cases(2, 80, "modes", 6)
    sides = []
    for _ in range(2):
        bw = SRC.BatchedWorlds(cs, "cuda", "modes", hits=False)
        for t in range(6):                                  # the records, the bring-outs and ticks: eager
            bw.step(t)
        sides.append(bw)
    eager, graphed = sides
    for bw in sides:                                        # a fresh wave: every world's first ball with a delay of 2
        mask = torch.ones(bw.W, dtype=torch.bool)
        ball = torch.tensor([w["balls"][0]["id"] for w in cs], dtype=torch.int64)
        bw.removal.remove_ball(mask, ball, 2)
    park = graphed.park
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    t0 = park.current_time
    with torch.cuda.graph(g):
        graphed.removal.bring_out_the_deads()
        graphed.space.tick(park)
    park.current_time = t0
    ptrs = {k: v.data_ptr() for k, v in vars(park).items() if torch.is_tensor(v)}
    for t in range(8):
        eager.removal.bring_out_the_deads()
        eager.space.tick(eager.park)
        g.replay()
        park.current_time += 1
        for k, v in vars(eager.park).items():
            if torch.is_tensor(v):
                a, b = v, getattr(park, k)
                same = torch.equal(a, b) if a.dtype not in (torch.float64,) else bool(
                    ((a.view(torch.int64) == b.view(torch.int64)) | (torch.isnan(a) & torch.isnan(b))).all())
                assert same, (t, k)
        for k, v in eager.space.st.items():
            if torch.is_tensor(v):
                assert torch.equal(v, graphed.space.st[k]), (t, k)
        assert torch.equal(eager.removal.pending, graphed.removal.pending), t
        assert {k: v.data_ptr() for k, v in vars(park).items() if torch.is_tensor(v)} == ptrs
    assert int(eager.space.st["in_park"].sum()) < sum(len(w["balls"]) for w in cs)


@CUDA
def test_the_gpu_planted_fault_is_caught():
    """The removal tick kept apart from mEffectStamp: on VecModesPark a command given to a waiting ball (a Stop of a
    follower, a WarpTo) moves its tick in the reference; the fault keeps it; caught."""
    res = SRC.compare(SRC.cases(1, 150, "modes", 16), "cuda", "modes", removal_cls=_OwnRemovalTick)
    assert res["diffs"]

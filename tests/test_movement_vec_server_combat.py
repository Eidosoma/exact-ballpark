# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer over the combat park (exact_ballpark.vec_server_combat: the LaunchMissile
action, and the missile and formation branches of the actions' Stop and StopAllFollowers on the master) against the
scalar server layer (exact_ballpark.server on a master DestinyPark over FullBallpark, held to Destiny's published server
tests), bit for bit, every tick: every client's update list (stamps, events, argument types, AddBallsToPark streams
byte for byte with the missiles' and formation followers' tails, PackagedActions, wait_for_bubble, counts) and every
ball's position, velocity, mode, massive flag, follow and owner ids, formation id and slots
(tests/server_combat_cases.py).

On the CPU (VecCombatPark's torch tick): development seed 1 and held-out seed 5 (first run after the generator was
frozen), the kinds and modes reached, the missiles released by a cloak and the formation slots freed by a Stop,
the refusals, five planted faults (each must be caught). On the GPU (skipped without one; the park's kernels, FollowBall
and Orbit too): seeds 1 and 5.
"""

from __future__ import annotations

import logging

import pytest
import torch

import server_combat_cases as SC
from exact_ballpark import vec_combat as VCB
from exact_ballpark import vec_modes as VMO
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_combat as VSC

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
logging.getLogger("exact_ballpark.server").setLevel(logging.CRITICAL)
HELD_OUT = 5                                                # first run after the generator was frozen
_RUNS: dict = {}


def run(seed: int, n: int = 200, device: str = "cpu", ticks: int = 12) -> dict:
    k = (seed, n, device, ticks)
    if k not in _RUNS:
        _RUNS[k] = SC.run(SC.cases(seed, n, device, ticks), device)
    return _RUNS[k]


MINIMUM = {"LaunchMissile": 300, "CloakBall": 300, "Stop": 300, "AddBallsToPark": 200, "ticks_in_MISSILE": 500,
           "ticks_in_FORMATION": 300, "missiles_released": 20, "formation_slots_freed": 20}


def assert_exact(res: dict, minimum: dict) -> None:
    assert not res["flagged"], res["flagged"][:10]
    assert not res["sends"], [str(d)[:600] for d in res["sends"][:3]]
    assert not res["rows"], res["rows"][:5]
    for key, low in minimum.items():
        assert res["reached"].get(key, 0) >= low, (key, res["reached"])


# ==================================================================================================== on the CPU

def test_matches_the_scalar_server_layer_development_worlds():
    assert_exact(run(1), MINIMUM)


def test_matches_the_scalar_server_layer_held_out_worlds():
    assert_exact(run(HELD_OUT), MINIMUM)


def test_launch_missile_is_a_kind_of_the_combat_park_only():
    """LaunchMissile is queued for VecCombatPark's adapter; VecModesPark's adapter refuses it (no missiles), and its
    kinds are as before."""
    cs = SC.cases(1, 2, "cpu", 1)
    server, _ = SC.build(cs, "cpu")
    assert VS.LAUNCH in server.a.kinds
    p = VMO.VecModesPark(2, device="cpu")
    p.load([{"id": i + 1, "world": 0, "time": 0, "mass": 1e6, "radius": 50.0, "max_vel": 100.0, "pos": [0.0, 0.0, 0.0],
             "vel": [0.0, 0.0, 0.0], "agility": 1.0, "speed_fraction": 1.0, "free": True, "interactive": True}
            for i in range(2)])
    a = VS.ModesAdapter(p)
    assert VS.LAUNCH not in a.kinds
    assert a.kinds == (frozenset(range(1, VS.LAUNCH)) | {VS.TROLL, VS.FREE}) - VS.DYNAMICAL_ONLY
    with pytest.raises(ValueError):
        VS.VecServer(a).push("LaunchMissile", torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.int64))
    with pytest.raises(ValueError):
        VSC.CombatAdapter(VCB.VecCombatPark(2, device="cpu", master=False))   # the master's view only here
    with pytest.raises(ValueError):
        VSC.CombatServer(a)


_PACK = VS._pack_ball


def _no_missile_tail(fl, r, out):
    mode = int(fl["mode"][r]) & 0xFF
    if mode == 5:
        fl = dict(fl, mode=fl["mode"].clone())
        fl["mode"][r] = 2                                   # written as STOP: no tail
    _PACK(fl, r, out)


def _no_formation_id(fl, r, out):
    _PACK({k: v for k, v in fl.items() if k != "formation_id"}, r, out)


_ROWS = VSC.CombatServer._rows


def _target_as_row(self, slot):                             # LaunchMissile given the target's row lookup (-1)
    return VS.VecServer._rows(self, slot)


_RELEASE = VCB.VecCombatPark._release_followers


def _release_without_missiles(self, rows):                  # StopAllFollowers as VecModesPark's: missiles stopped
    safe = torch.where(self.follow_ptr >= 0, self.follow_ptr, 0)
    f = self._following() & (self.follow_ptr >= 0) & rows[safe] & self.exists
    orbiter = f & self.interactive & (self.mode == VCB.ORBIT)
    self._stop_rows(f & ~orbiter)
    self.goto_direction(orbiter, self.vel.clone())


_STOP_ROWS = VCB.VecCombatPark._stop_rows


def _stop_without_slot_free(self, mask):                    # the master's slot free left out of Stop
    follower = mask & self._following()
    self.owner_ptr = torch.where(follower, -1, self.owner_ptr)
    VMO.VecModesPark._stop_rows(self, mask)


FAULTS = {"missile_tail_dropped": ("_pack_ball", VS, _no_missile_tail),
          "formation_id_dropped": ("_pack_ball", VS, _no_formation_id),
          "launch_target_as_row": ("_rows", VSC.CombatServer, _target_as_row),
          "release_without_missiles": ("_release_followers", VCB.VecCombatPark, _release_without_missiles),
          "stop_without_slot_free": ("_stop_rows", VCB.VecCombatPark, _stop_without_slot_free)}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_a_planted_fault_is_caught(fault, monkeypatch):
    name, owner, fn = FAULTS[fault]
    monkeypatch.setattr(owner, name, fn)
    res = SC.run(SC.cases(1, 120, "cpu", 12), "cpu")
    assert res["sends"] or res["rows"], fault


# ==================================================================================================== on the GPU

@CUDA
def test_matches_the_scalar_server_layer_on_the_park_kernels():
    """VecCombatPark's kernels (FollowBall and Orbit of ships too), seeds 1 and 5."""
    for seed in (1, HELD_OUT):
        assert_exact(run(seed, 200, "cuda"), dict(MINIMUM, FollowBall=50, Orbit=50))

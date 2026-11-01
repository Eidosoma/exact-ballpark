# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's pending-action queries and the undo of a pending cloak on the batched server layers: VecServer's has_pending_cloak, has_pending_uncloak,
get_pending_cloak_mode, get_pending_interactive_state, get_pending_speed_fraction and undo_pending_cloak (and so
SpaceServer's, CudaSpaceServer's and CombatServer's), against the scalar server layer (exact_ballpark.server's Actions,
_actions.py:686-766, held to Destiny's own tests by tests/test_destiny_server_layer.py), bit for bit: every answer, every
client's update list, every ship's state, every tick (tests/server_pending_cases.py, over the earlier frozen streams with the queries, undos and cloaks added).

On the CPU: the kernels' mask byte unchanged without an undo; the torch paths of VecServer over VecOrientPark,
SpaceServer over VecOrientPark and CombatServer over VecCombatPark against the scalar layer on development seeds 101,
102 and 109 (109 was to be held out; it ran before the freeze was recorded, so it counts as development) and held-out
seed 117 (first run after the generator's freeze); three planted faults caught (the undo ignored, the queue's last action kept after the
prune, the calls dropped as duplicates counted). On the GPU (skipped without one): the three kernels (VecServer's over
VecModesPark, CudaSpaceServer, CombatServer's) against their torch paths, every answer, every tensor and the batch,
every tick (seed 101 and held-out seed 117), and against the scalar layer; the planted fault of each kernel (an undo's tested calls tested again)
caught.
"""

from __future__ import annotations

import logging

import pytest
import torch

import server_cases as S
import server_combat_cases as SC
import server_pending_cases as PC
import server_space_cases as SS
from exact_ballpark import vec_server as VS

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
DEV = (101, 102, 109)
HELD_OUT = 117


@pytest.fixture(autouse=True)
def _quiet():
    """The scalar layer logs every event with a stale stamp (a quirk of Destiny's test order, server.py's docstring)."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def scalar_run(family: str, seed: int, n: int = 40, ticks: int = 12) -> dict:
    if family == "server":
        cs = PC.augment(S.cases(seed, n, "orient", ticks=ticks), family, f"orient-{seed}",
                        PC.MODES["server"]["orient"])
        server, index = S.build(cs, "orient", "cpu")
        return PC.run(cs, "orient", "cpu", server, index, [S.ScalarWorld(w) for w in cs])
    if family == "space":
        cs = PC.augment(SS.cases(seed, n, "orient", ticks=ticks), family, f"orient-{seed}", PC.MODES["space"])
        server, index = SS.build(cs, "orient", "cpu")
        return PC.run(cs, "orient", "cpu", server, index, [S.ScalarWorld(w) for w in cs], rows="space")
    cs = PC.augment(SC.cases(seed, n, "cpu", ticks=ticks), family, f"cpu-{seed}", PC.MODES["combat"])
    server, index = SC.build(cs, "cpu")
    return PC.run(cs, "combat", "cpu", server, index, [SC.ScalarWorld(w) for w in cs], rows="combat",
                  pushes=SC.pushes)


def assert_exact(res: dict, answers: int = 800, undos: int = 200) -> None:
    assert not res["flagged"], res["flagged"][:8]
    assert not res["answers"], [str(d)[:300] for d in res["answers"][:3]]
    assert not res["sends"], [str(d)[:300] for d in res["sends"][:3]]
    assert not res["rows"], res["rows"][:5]
    vol = res["volumes"]
    assert vol["answers"] >= answers and vol["undos"] >= undos, vol
    for kind in ("has_pending_cloak:b", "has_pending_uncloak:b", "get_pending_cloak_mode:i",
                 "get_pending_cloak_mode:none", "get_pending_interactive_state:b",
                 "get_pending_interactive_state:none", "get_pending_speed_fraction:f",
                 "get_pending_speed_fraction:none"):
        assert vol["kinds"].get(kind, 0) > 0, (kind, vol["kinds"])


# ==================================================================================================== on the CPU

def test_the_mask_bytes_are_the_masks_without_an_undo():
    """Without an undo the kernels read the masks' bytes 0 and 1 as before; an undo's tested calls carry bit 1."""
    W = 5
    m = [torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool), torch.tensor([0, 0, 1, 0, 1], dtype=torch.bool)]
    slots = [VS.Slot(VS.CLOAK, None, torch.zeros(W, dtype=torch.int64), None, None, None, None, None, None)
             for _ in m]
    for s, x in zip(slots, m):
        s.mask = x
    b = VS.mask_bytes(slots, W, "cpu")
    assert b.dtype == torch.uint8 and torch.equal(b, torch.stack(m).to(torch.uint8))
    slots[0].resolved = True
    b = VS.mask_bytes(slots, W, "cpu")
    assert b[0].tolist() == [3, 2, 3, 3, 2] and b[1].tolist() == [0, 0, 1, 0, 1]
    assert VS.mask_bytes([], W, "cpu").shape == (0, W)


@pytest.mark.parametrize("family", ["server", "space", "combat"])
@pytest.mark.parametrize("seed", DEV)
def test_the_torch_paths_match_the_scalar_server_layer(family, seed):
    """VecServer over VecOrientPark, SpaceServer over VecOrientPark (several bubbles), CombatServer over VecCombatPark
    (missiles and formations): every answer, update and ship, every tick, development seeds."""
    assert_exact(scalar_run(family, seed))


@pytest.mark.parametrize("family", ["server", "space", "combat"])
def test_the_torch_paths_match_the_scalar_server_layer_held_out(family):
    assert_exact(scalar_run(family, HELD_OUT))


def _undo_ignored(self, mask, ball):
    return None


def _last_kept(self, mask, ball):
    saved = {k: v.clone() for k, v in self.last.items()}
    _UNDO(self, mask, ball)
    self.last = saved


def _duplicates_counted(self):
    return [torch.ones_like(s.ok) for s in self.slots]


_UNDO = VS.VecServer.undo_pending_cloak
PLANTED = {"undo_ignored": ("undo_pending_cloak", _undo_ignored),
           "last_kept_after_the_prune": ("undo_pending_cloak", _last_kept),
           "duplicates_counted_as_queued": ("_queued", _duplicates_counted)}


@pytest.mark.parametrize("fault", sorted(PLANTED))
def test_a_planted_fault_is_caught(fault, monkeypatch):
    """Each fault planted into VecServer (monkeypatch) makes the torch path differ from the scalar layer on development
    seed 101 (the worlds); the true code there differs nowhere (test above)."""
    name, fn = PLANTED[fault]
    monkeypatch.setattr(VS.VecServer, name, fn)
    res = scalar_run("server", 101)
    assert res["answers"] or res["sends"] or res["rows"], fault


# ==================================================================================================== on the GPU

def _servers(family: str, cs: list, fault: str | None = None) -> tuple:
    """(torch path, kernel path, decode, pushes) over their own parks on the card."""
    if family == "server":
        ts, _ = S.build(cs, "modes", "cuda")
        if fault is None:
            ks, _ = S.build(cs, "modes", "cuda", server_kernel="cuda")
        else:
            from exact_ballpark import vec_server_cuda as V
            ks, _ = S.build(cs, "modes", "cuda", server_kernel="cuda")
            with torch.cuda.device(ks.dev):
                ks._lib = V.load_library(variant="sm90", fault=fault, checks=False)
        return ts, ks, VS.decode, S.pushes
    if family == "space":
        from exact_ballpark import vec_server_space as VSS
        from exact_ballpark import vec_server_space_cuda as Q
        from exact_ballpark.vec_modes import VecModesPark
        ts, _ = SS.build(cs, "modes", "cuda")
        rows = [(w, b) for w in cs for b in w["balls"]]
        p = VecModesPark(len(rows), device="cuda")
        p.load([{"id": b["id"], "world": w["id"], "time": 0, "mass": b["mass"], "radius": b["radius"],
                 "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
                 "speed_fraction": b["sf"], "free": True, "massive": b["massive"], "interactive": b["interactive"],
                 "cloaked": 0, "bubble": -1, "moribund": False} for w, b in rows])
        a = VSS.SpaceModesAdapter(p)
        index = {(w["id"], b["id"]): i for i, (w, b) in enumerate(rows)}
        pilots = [-1] * a.R
        for w in cs:
            for bid, pp in w["pilots"].items():
                pilots[index[(w["id"], bid)]] = pp
        ks = Q.CudaSpaceServer(a, pilots, fault=fault)
        return ts, ks, VSS.decode, S.pushes
    from exact_ballpark import vec_server_combat as VC
    ts, _ = SC.build(cs, "cuda")
    ks, _ = SC.build(cs, "cuda")
    ks = VC.CombatServer(ks.a, ks.pilot, kernel="cuda", fault=fault)
    return ts, ks, VS.decode, SC.pushes


def _cases(family: str, seed: int, n: int = 40, ticks: int = 12) -> list:
    if family == "server":
        return PC.augment(S.cases(seed, n, "modes", ticks=ticks), family, f"modes-{seed}", PC.MODES["server"]["modes"])
    if family == "space":
        return PC.augment(SS.cases(seed, n, "modes", ticks=ticks), family, f"modes-{seed}", PC.MODES["space"])
    return PC.augment(SC.cases(seed, n, "cuda", ticks=ticks), family, f"cuda-{seed}", PC.MODES["combat"])


@CUDA
@pytest.mark.parametrize("family", ["server", "space", "combat"])
def test_the_kernels_equal_their_torch_paths(family):
    """Each kernel against its torch path over its own park, the same pushes, queries and undos: every answer before
    the pre-tick, then every tensor of the park (the partition), the server and the batch, every tick; development
    seed 101 and held-out seed 117."""
    for seed in (101, HELD_OUT):
        cs = _cases(family, seed)
        ts, ks, decode, pushes = _servers(family, cs)
        bad, vol = PC.kernel_against_torch(cs, ts, ks, decode, pushes=pushes)
        assert not bad, bad[:3]
        assert vol["answers"] >= 800 and vol["undos"] >= 200 and vol["resolved_slots"] >= 300, vol


@CUDA
@pytest.mark.parametrize("family", ["server", "space", "combat"])
def test_the_kernels_equal_the_scalar_server_layer(family):
    """Each kernel path against the scalar layer directly: answers, updates, ships; development seed 102."""
    cs = _cases(family, 102)
    ts, ks, decode, pushes = _servers(family, cs)
    del ts
    if family == "server":
        index = {(w["id"], b["id"]): i for i, (w, b) in enumerate((w, b) for w in cs for b in w["balls"])}
        res = PC.run(cs, "modes", "cuda", ks, index, [S.ScalarWorld(w) for w in cs])
    elif family == "space":
        index = {(w["id"], b["id"]): i for i, (w, b) in enumerate((w, b) for w in cs for b in w["balls"])}
        res = PC.run(cs, "modes", "cuda", ks, index, [S.ScalarWorld(w) for w in cs], rows="space")
    else:
        _, index = SC.build(cs, "cuda")
        res = PC.run(cs, "combat", "cuda", ks, index, [SC.ScalarWorld(w) for w in cs], rows="combat",
                     pushes=SC.pushes)
    assert_exact(res)


@CUDA
@pytest.mark.parametrize("family", ["server", "space", "combat"])
def test_the_kernels_planted_fault_is_caught(family):
    """Each kernel built with its pending fault (an undo's tested calls tested again at the pre-tick: the prune's
    edge, X CloakBall X undone, loses its second X) differs from its torch path on development seed 101."""
    cs = _cases(family, 101)
    ts, ks, decode, pushes = _servers(family, cs, fault="undo_retested")
    bad, _ = PC.kernel_against_torch(cs, ts, ks, decode, pushes=pushes)
    assert bad, family

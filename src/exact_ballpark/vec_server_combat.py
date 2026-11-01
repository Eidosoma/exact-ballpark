# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer over the combat park: `exact_ballpark.vec_server.VecServer` (the batched
server timing: the queue, stamps, duplicate test, pre-tick order, routes and update batches, unchanged) over
`exact_ballpark.vec_combat.VecCombatPark` (missiles, formations, cloaking at the park), with the one
queued Action among the missile and formation commands, LaunchMissile (_actions.py:653-675), applied at the pre-tick
in queue order like every other action, and the missile and formation branches of the actions that release or stop
balls taken from the park's own methods: a Stop of a FORMATION follower frees its slot in its leader on the master
(Ballpark.cpp:4629-4633), and CloakBall's StopAllFollowers makes a missile non-massive and flies it on along its
velocity, stops a formation follower (freeing its slot) and every other follower (MOV-730), as VecCombatPark's
`_stop_rows` and `_release_followers` do.

It must reproduce `exact_ballpark.server` (Actions, Ticker, BubbleUpdater, ParkUpdateBatcher on a master DestinyPark over
FullBallpark: this port's LaunchMissile thunk, PyLaunchMissile, with its parse and the negative-target refusal) bit for
bit; checked by tests/test_movement_vec_server_combat.py on generated command streams (tests/server_combat_cases.py)
on the CPU (VecCombatPark's torch tick: GOTO, STOP, MISSILE and FORMATION) and the GPU (its kernels: FOLLOW, ORBIT and
WARP too): every client's update list and every ball's state.

**The worlds** are VecServer's (one bubble a world, static, kept alive by an interactive ball), on VecCombatPark with
`master=True`: the missile balls are in the park from the load (in STOP until LaunchMissile), as the server's queue
needs them (an AddBall mid-run would need the bubble update's later additions, which the one-bubble layer does not
send: `exact_ballpark.vec_server_space` does, over VecModesPark only); formations are loaded and set by the caller on the
park (they are not Actions). A LaunchMissile event is routed as any other: to the bubble, or to the missile's character
while it is cloaked; AddBallsToPark streams carry a missile's and a formation follower's tails (vec_server's
`_pack_ball`, as WriteBallToStream).

**Exactness.** Every arithmetic step is VecCombatPark's own (held to the scalar reference, and through it to Destiny's
compiled code, by its own tests); this module adds index work only.

**The kernel**: `CombatServer(..., kernel="cuda")` over VecCombatPark itself on a CUDA device runs
the pre-tick and the send flags as one CUDA kernel a tick, one thread a world (exact_ballpark.vec_server_combat_cuda:
LaunchMissile, the missile and formation branches of Stop and StopAllFollowers, the lookups of the balls in the park,
SetBallVelocity's snap of the rotation or of the yaw, pitch and roll), held to this module's torch path bit for bit by
tests/test_movement_vec_server_combat_cuda.py; VecServer's own kernel (vec_server_cuda: VecModesPark's) refuses this
adapter.

OnTick's intervals: VecServer.on_tick, inherited (CombatAdapter.evolve takes the per-world mask:
VecCombatPark's evolve(active) on the kernels, evolve_torch(active) on the CPU; BringOutTheDeads at each call, a world
that loses a ball flagged), on the torch path and the kernel alike.

Not here: the partition (SpaceServer runs over VecModesPark and VecOrientPark), AddBall and RemoveBall mid-run through
the server layer, defender missiles (VecCombatPark refuses them).
"""

from __future__ import annotations

import torch

from . import vec_server as VS

I64, BOOL = torch.int64, torch.bool


class CombatAdapter(VS.ModesAdapter):
    """ModesAdapter over VecCombatPark: every kind ModesAdapter applies and LaunchMissile; `exists` is the park's (an
    empty row is no ball); its stream fields carry the formation id. On the CPU the park's tick is its torch path
    (`evolve_torch`: GOTO, STOP, MISSILE and FORMATION), on a CUDA device its kernels."""

    def __init__(self, park, master: bool = True):
        from .vec_combat import VecCombatPark
        if not isinstance(park, VecCombatPark):
            raise ValueError("CombatAdapter: a VecCombatPark")
        if bool(master) != bool(park.master):
            raise ValueError("CombatAdapter: the park's view (master) must be the adapter's")
        super().__init__(park, master)
        self.kinds = frozenset(self.kinds | {VS.LAUNCH})

    exists = property(lambda s: s.park.exists, lambda s, v: None)

    @property
    def ids_of_world(self) -> torch.Tensor:
        """[W, N] the Destiny ids of each world's rows, 0 for an empty row or padding."""
        safe = self.rows_of_world.clamp(min=0)
        ok = (self.rows_of_world >= 0) & self.park.exists[safe]
        return torch.where(ok, self.park.ids[safe], 0)

    @ids_of_world.setter
    def ids_of_world(self, value) -> None:                   # _index_worlds sets it at construction
        pass

    def lookup(self, ball: torch.Tensor) -> torch.Tensor:
        """Each world's row of the ball with Destiny id ball[w] (in the park: `exists`), or -1."""
        ids_w = self.ids_of_world
        safe = self.rows_of_world.clamp(min=0)
        hit = (ids_w == ball.to(I64)[:, None]) & (self.rows_of_world >= 0) & self.park.exists[safe]
        idx = hit.to(I64).argmax(dim=1)
        return torch.where(hit.any(dim=1), self.rows_of_world.gather(1, idx[:, None])[:, 0], -1)

    def apply(self, kind: int, m: torch.Tensor, trow: torch.Tensor, f: torch.Tensor, n: torch.Tensor) -> None:
        """ModesAdapter.apply; LaunchMissile (here `trow` holds the target's Destiny id, `n` the owner's) through
        VecCombatPark.launch_missile (PyLaunchMissile from Thunkers.cpp:878: the negative-target refusal included)."""
        if kind == VS.LAUNCH:
            self.park.launch_missile(m, trow, n, f[:, 0].to(I64), f[:, 1].to(I64))
            return
        super().apply(kind, m, trow, f, n)

    def evolve(self, worlds: torch.Tensor | None = None) -> None:
        """The park's tick (its kernels on a CUDA device, its torch tick on the CPU); `worlds` [W] (OnTick's mask): those worlds' rows alone, every other row kept bit for bit (VecCombatPark's `active`)."""
        rows = None if worlds is None else worlds[self.world]
        if self.dev.type == "cuda":
            self.park.evolve(active=rows)
        else:
            self.park.evolve_torch(active=rows)

    def stream_fields(self) -> dict:
        out = super().stream_fields()
        out["formation_id"] = self.park.formation_id.detach().cpu()
        return out


class CombatServer(VS.VecServer):
    """VecServer over a CombatAdapter: LaunchMissile's target is passed to the park as its Destiny id (the park finds
    the row itself, as PyLaunchMissile does). `kernel="cuda"` (a VecCombatPark, not a subclass, on a CUDA device) runs
    the pre-tick and the send flags as one CUDA kernel, one thread a world (exact_ballpark.vec_server_combat_cuda);
    "torch", the default, is the exactness reference. `block`: the kernel's threads a block (by default the module's
    BLOCK); `fault`: a planted fault's library (the tests only)."""

    def __init__(self, adapter: CombatAdapter, pilots=None, kernel: str = "torch", block: int | None = None,
                 fault: str | None = None):
        if not isinstance(adapter, CombatAdapter):
            raise ValueError("CombatServer: a CombatAdapter")
        if kernel not in ("torch", "cuda"):
            raise ValueError(f"kernel: 'torch' or 'cuda', not {kernel!r}")
        if kernel == "cuda":
            from .vec_combat import VecCombatPark
            if type(adapter.park) is not VecCombatPark or adapter.dev.type != "cuda":
                raise ValueError('kernel="cuda" runs the server layer of a VecCombatPark (not a subclass) on a CUDA '
                                 'device')
        super().__init__(adapter, pilots, kernel="torch")
        self.block = None if block is None else int(block)
        if kernel == "cuda":
            from . import vec_server_combat_cuda as SK
            self.kernel, self._cuda = "cuda", SK
            with torch.cuda.device(self.dev):
                self._lib = (SK.load_library() if fault is None else
                             SK.load_library(variant="sm90", fault=fault, checks=False))

    def _rows(self, slot: VS.Slot) -> tuple:
        m, trow, f, n = super()._rows(slot)
        if slot.kind == VS.LAUNCH:
            trow = slot.target[self.a.world]
        return m, trow, f, n

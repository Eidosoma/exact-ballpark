# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's delayed removal on the master, batched: RemoveBall with a delay and BringOutTheDeads with
its 7-ball buffer (TIME-713), for the worlds of a `exact_ballpark.vec_space.VecSpace` over its park (VecModesPark or
VecOrientPark, on the CPU or a CUDA device), on the park's device, with no host synchronisation (so the
bring-out and a tick of the layer can be captured together as a CUDA graph).

It must reproduce the scalar reference bit for bit: `exact_ballpark.full.FullBallpark` with is_master=True (space.py's
`remove_ball` and `bring_out_the_deads`, full.py's `remove_ball`), checked by tests/test_movement_vec_space_removal.py
on generated worlds (tests/space_removal_cases.py), on the CPU and the GPU. Rules (Ballpark.cpp:5471, 5699; MOV-730):

- `remove_ball(mask, ball, delay)`, RemoveBall(ball, delay) outside Evolve for every world in mask: a ball not in the
  park is ignored; the ball's Stop and StopAllFollowers (MOV-730: VecModesPark's release, an interactive orbiter flying
  on along its velocity, any other follower stopped; VecOrientPark has no followers), then it is moribund. With a
  delay above 0 it is made non-massive, its removal tick is the park's time plus the delay, kept in mEffectStamp (the
  field the warp and the modes use too: a command given to the ball before it goes moves its removal tick, as in
  Destiny, DESTINY_FINDINGS section 1, item 8), and it joins moribundBalls (`pending`). With none it leaves at once:
  DoFinalCleanup, out of moribundBalls, out of its boxes and its bubble (TIME-711; vec_space's REMOVE_BALL).
- `bring_out_the_deads()`, as OnTick calls it before the tick: each ball of moribundBalls goes whose moribund flag was
  cleared, or whose removal tick has passed (tick - time < 0), or, while fewer than 7 such have gone, which is within 2
  ticks of it (tick - time - 2 < 0); each that goes and is moribund is removed at once (its Stop and StopAllFollowers,
  then the partition's REMOVE_BALL with its DoFinalCleanup), in that order; each leaves moribundBalls.

**Exact against the reference, never against Destiny (TIME-713).** Destiny's BringOutDeadBalls walks an unordered_set
of Ball pointers (Ballpark.cpp:5716): the order of one call's removals and of their DoFinalCleanup events, and which
balls the kill count of 7 takes when more than 7 are within the buffer, follow heap addresses, which no transcription
can reproduce (DESTINY_FINDINGS section 1, item 7). The reference walks moribundBalls in ascending id, and so does
this module (the layer's `by_id`): it equals the reference bit for bit, and it equals Destiny only where a call
removes at most one ball and the buffer holds at most 7.

**Exactness.** No float arithmetic: index work, integer comparisons (the removal tick against the park's time), the
park's own Stop and release (held to the reference by their own tests), and the partition's REMOVE_BALL (vec_space's
algorithm, host and device). The bring-out's park part is applied to every ball that goes before the partition's
removals in ascending id; the reference interleaves them ball by ball, which is the same, as the Stop and the release
change only the ball and its followers' flight fields, which the partition's removal does not read (and a moribund
ball can gain no follower: FollowBall and Orbit refuse a moribund target).

Not here: removals inside Evolve (missiles' contacts: VecCombatPark), the capsules' and the minis' removals (no
static collidables in these parks), AddBall of a moribund ball (which clears the flag: the `not moribund` branch is
transcribed but cannot arise), the server layer's view of a removed ball (VecServer's lookups know every loaded row).
"""

from __future__ import annotations

import torch

from . import vec_space as SP

I64, BOOL = torch.int64, torch.bool
KILL_COUNT = 7                                      # Ballpark.cpp:5716 (killCount)
BUFFER_TICKS = 2                                    # stamp - time - 2 < 0


class DelayedRemoval:
    """RemoveBall with a delay and BringOutTheDeads for the worlds of `space` (a VecSpace, loaded) over its park.
    `pending` [W, N]: moribundBalls by slot."""

    def __init__(self, space: SP.VecSpace):
        from .vec_modes import VecModesPark
        self.space = space
        self.view = space.view
        self.park = space.view.park
        self.modes = isinstance(self.park, VecModesPark)
        if not self.modes and not isinstance(self.view, SP.OrientView):
            raise ValueError("DelayedRemoval: a VecSpace over VecModesPark (ModesView) or VecOrientPark (OrientView)")
        self.pending = torch.zeros(space.W, space.N, dtype=BOOL, device=space.dev)

    # ---- the park's fields, by park ----

    def _get(self, name: str) -> torch.Tensor:
        if self.modes:
            return getattr(self.park, name)
        if name == "massive":
            return self.view.flag("massive")
        return getattr(self.view, name)             # moribund, effect_stamp: the view's own

    def _put(self, name: str, value: torch.Tensor) -> None:
        """Written in place (the tensor keeps its storage, so a CUDA graph's captured launches still read it)."""
        self._get(name).copy_(value)

    def clock(self) -> torch.Tensor:
        """[W]: each world's mCurrentTime (VecModesPark: its first slot's clock; VecOrientPark: the park's one)."""
        sp = self.space
        if self.modes:
            return self.park.clock[sp.rows[:, 0].clamp(min=0)]
        return torch.full((sp.W,), int(self.park.current_time), dtype=I64, device=sp.dev)

    def _row_clock(self) -> torch.Tensor:
        if self.modes:
            return self.park.clock
        return torch.full((self.view.R,), int(self.park.current_time), dtype=I64, device=self.space.dev)

    def _rows_of(self, by_slot: torch.Tensor) -> torch.Tensor:
        """[R]: the rows of the slots marked in by_slot [W, N] (no host synchronisation)."""
        sp = self.space
        out = torch.zeros(self.view.R + 1, dtype=BOOL, device=sp.dev)
        out.scatter_(0, sp._flat, by_slot.reshape(-1))
        return out[:self.view.R]

    def _stop_and_release(self, m: torch.Tensor) -> None:
        """The park's part of RemoveBall: Stop(Ball*), then StopAllFollowers (MOV-730), by the park's own commands;
        every tensor they rebind is copied back into its storage (in place, as a captured graph needs)."""
        before = {k: v for k, v in vars(self.park).items() if torch.is_tensor(v)}
        self.park.stop(m)
        if self.modes:
            self.park._release_followers(m)
        for k, old in before.items():
            new = getattr(self.park, k)
            if new is not old:
                old.copy_(new)
                setattr(self.park, k, old)

    # ---- RemoveBall ----

    def remove_ball(self, mask: torch.Tensor, ball: torch.Tensor, delay) -> None:
        """RemoveBall(ball, delay) outside Evolve for every world in mask [W] (ball [W] Destiny ids, delay [W] or one
        int: 0 or less removes at once)."""
        sp = self.space
        W, N, d = sp.W, sp.N, sp.dev
        mask = torch.as_tensor(mask, dtype=BOOL).to(d).reshape(W)
        ball = torch.as_tensor(ball, dtype=I64).to(d).reshape(W)
        delay = torch.as_tensor(delay, dtype=I64).to(d)
        delay = delay.expand(W) if delay.dim() == 0 else delay.reshape(W)
        rows = sp.rows
        ids = self.view.tensors()["ids"]
        ids_w = torch.where(rows >= 0, ids[rows.clamp(min=0)], 0)
        hit = (ids_w == ball[:, None]) & (rows >= 0) & (sp.st["in_park"] != 0) & mask[:, None]   # [W, N]
        later = hit & (delay > 0)[:, None]
        now = hit & (delay <= 0)[:, None]
        m = self._rows_of(hit)
        self._stop_and_release(m)
        self._put("moribund", self._get("moribund") | m)
        ml = self._rows_of(later)
        self._put("massive", self._get("massive") & ~ml)
        row_delay = torch.zeros(self.view.R + 1, dtype=I64, device=d)
        row_delay.scatter_(0, sp._flat, delay[:, None].expand(W, N).reshape(-1))
        stamp = self._row_clock() + row_delay[:self.view.R]
        self._put("effect_stamp", torch.where(ml, stamp, self._get("effect_stamp")))
        self.pending |= later                               # in place (a captured graph reads its storage)
        self.pending &= ~now
        code = torch.where(now, SP.OP["REMOVE_BALL"], SP.OP["NONE"]).to(I64)
        slots = torch.arange(N, dtype=I64, device=d).expand(W, N).contiguous()
        sp.tape(code, slots)

    # ---- BringOutTheDeads ----

    def due(self) -> tuple:
        """(due [W, N], removed [W, N]) in ascending-id order (the layer's by_id): the balls the next bring-out takes,
        and those of them it removes (moribund)."""
        sp = self.space
        by = sp.by_id
        valid = by >= 0
        slot = by.clamp(min=0)
        pend = torch.gather(self.pending, 1, slot) & valid
        rr = torch.gather(sp.rows, 1, slot).clamp(min=0)
        mor = self._get("moribund")[rr]
        left = self._get("effect_stamp")[rr] - self.clock()[:, None]        # the removal tick minus the time
        passed = left < 0
        now = pend & (~mor | passed)
        cand = pend & mor & ~passed & (left - BUFFER_TICKS < 0)
        buffered = cand & (torch.cumsum(cand.to(I64), dim=1) <= KILL_COUNT)
        due = now | buffered
        return due, due & mor

    def bring_out_the_deads(self) -> None:
        """BringOutTheDeads (Ballpark.cpp:5699) for every world: the balls due, in ascending id, removed at once."""
        sp = self.space
        W, N, d = sp.W, sp.N, sp.dev
        due, rem = self.due()
        by = sp.by_id
        valid = by >= 0
        slot_or_pad = torch.where(valid, by, N)
        rem_slot = torch.zeros(W, N + 1, dtype=BOOL, device=d)
        rem_slot.scatter_(1, slot_or_pad, rem & valid)
        due_slot = torch.zeros(W, N + 1, dtype=BOOL, device=d)
        due_slot.scatter_(1, slot_or_pad, due & valid)
        self._stop_and_release(self._rows_of(rem_slot[:, :N]))
        code = torch.where(rem, SP.OP["REMOVE_BALL"], SP.OP["NONE"]).to(I64)
        sp.tape(code, by.clamp(min=0))                      # the partition's removals, in ascending id
        self.pending &= ~due_slot[:, :N]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's combat pieces batched: missiles (their launch, flight, contact with the
target, collision events, release and end) and formations (the slots and formation follow), with cloaking at the
park, for many independent worlds, on the flight park of step 3 (`exact_ballpark.vec_modes.VecModesPark`, whose
GOTO, STOP, FOLLOW, ORBIT, WARP, setters and dynamical orientation are kept unchanged); the tick is four CUDA launches
(exact_ballpark.vec_combat_cuda; `evolve`), or the same algorithm on tensors for every mode on the CPU or the GPU
(`evolve_torch`: VecComposedPark's tick with collisions off, which held FOLLOW,
ORBIT and WARP, moved here; `evolve` takes it on any device but a CUDA one); the commands are torch operations on
either device.

It must reproduce `exact_ballpark.full.FullBallpark` bit for bit (the gradient method, Destiny's default; on a client
or, with `master=True`, on the master with one live bubble a world), which is held bit for bit to Destiny's own code
compiled verbatim (tests/test_movement_full_oracle.py), so it is exact against Destiny (Carbon's `destiny` at 114e89fa,
MIT, (c) CCP Games). Checked by tests/test_movement_vec_combat.py on generated
scenarios. Rules, by rule id:

- MOV-723: MISSILE mode (`EvolveMissile`, Ballpark.cpp:972): straight on toward mGoto while (time - launch) x tick is
  at most 800 ms, then the lead point at full thrust (GotoThrust's missile flag: no taper).
- MOV-724: `MissileFollow` (Ballpark.cpp:3798; the Python thunk's negative-target refusal): refused for coinciding ids,
  a missing missile or target, a moribund target; otherwise the missile is stopped, and then refused for a target in
  another bubble or cloaked; else the follow fields, the owner, the launch stamp, mGoto 1e16 m along the velocity,
  speed fraction 1, MISSILE.
- MOV-725: `PyLaunchMissile` (Thunkers.cpp:857): at the launcher (found by |owner|), aimed (1 m/s toward an uncloaked
  target) or not (the launcher's velocity plus max(150, its maximum speed) along its velocity, or along its rotated
  (0, 0, -1) when it is at rest: by the rotation with dynamical orientation, by RotationQuaternion(yaw, pitch, roll)
  without), made massive if asked, its velocity set (the orientation snapped), then MissileFollow, or GotoDirection
  for no target, or nothing more for a cloaked one.
- COL-703 and COL-711 (a normal missile's part): its one collision candidate is its target in its bubble (unfiltered);
  Gradient's skips; Potential with its overlap recursion; the response is not kept for its own target, so the target's
  id is appended to its collision list and neither ball's motion changes (the target does not see missiles).
- COL-719: `Ball::DispatchCollisions` at the commit for a missile that moved at least sqrt(0.1) m: DoCollision with
  each collider, then on the master with itself when its owner is gone, cloaked or in another bubble (`events()`).
- MOV-730 for missiles and formations: StopAllFollowers (when the followed ball starts warp proper, cloaks or is
  removed): a missile is made non-massive and flies on by GotoDirection along its velocity, an interactive orbiter
  flies on, every other follower stops.
- MOV-726: formations (`LoadFormations`, `SetBallFormation`, `FormationFollow`, `ReserveFormationSlot`,
  `FreeFormationSlot`, `EvolveFormation`): the leader's slot offset scaled to both radii plus its length, turned by the
  leader's rotation (with dynamical orientation its quaternion; without it its yaw, pitch and roll through DirectXMath's
  RotationQuaternion), GotoThrust toward it; at most 16 slots, the first free reserved; a smaller formation stops the
  followers in the slots it lacks, its own loop never freeing their bits; on the master every Stop of a FORMATION
  follower frees its slot in the leader (Ballpark.cpp:4629-4633: commands here, a leader's warp in the kernels).
- SEN-711 at the park (`cloak`, `uncloak`, as FullBallpark.cloak_ball and uncloak_ball): the mode a signed char,
  refused unless positive; on the master a normal cloak sets the sensor flag (the sensor itself is
  exact_ballpark.vec_space's); the followers released; cloaked, non-massive; uncloaking clears the normal cloak's flag and
  makes the ball massive unless in warp proper.
- MOV-729 where these modes read it (without dynamical orientation): each ball's yaw, pitch and roll, by
  `Ball::CalculateYawPitchRoll` for every mode (the warp's and the missile's branches included), after each tick and
  at the snaps (AddBall, SetBallVelocity), with glibc's double atan2, asin and sin on the host (Python's math is the C
  library's: exact by construction, two round trips a tick), the rest on the device; or, with `ypr="cuda"` (the switch), atan2, asin and sin on the device too (exact_ballpark.vec_ypr_cuda).
- MOV-711, TIME-711's immediate removal: `add_balls` (the Python AddBall of new balls into empty rows of their world)
  and `remove` (RemoveBall outside Evolve without a delay: the ball stopped, its followers released, its row emptied).
- MOV-732, MOV-734, MOV-736 (the commands, VecModesPark's): SetBallFree,
  SetBallRigid, the Python SetBoid, SetBallTroll and SetBallInteractive in this park's tick. The follower sets are
  StopAllFollowers' test (follow_ptr and follow id name the target: `followers`, `_release_followers`, the kernels'
  `member`), so a member SetBoid took out of FOLLOW, ORBIT, MISSILE or FORMATION is released too (and stopped:
  Stop(Ball*) clears the follow fields only in a follow mode); PetrifyTroll after phase 4 (the fifth launch,
  vec_combat_cuda.after), its SetBallVelocity's yaw, pitch and roll snap on the host without dynamical orientation.
- MOV-734's MUSHROOM: `add_mushroom` (PyAddMushroom, AddMushroom: a fixed massive ball of mass 1e20f,
  radius 0, at its owner; on a client id 0, on the master the world's next local id, two taken with dynamical
  orientation); a free mushroom grows at its turn in phase 4, range x powf(fraction, 0.25f) (glibc's powf: the C
  library's on the host in `evolve_torch`, exact_ballpark.dlibm_powf's transcription in the kernel), without the swap
  (its integrated state stays in the old members), and when grown is removed there (RemoveBall in Evolve: stopped, its
  followers released, before their own turn or after it by ids, moribund and non-massive until `bring_out_the_deads`,
  BringOutTheDeads between ticks, which empties its row).

**The worlds.** As VecModesPark: each row a ball of one world (`world`), Destiny's ids in `ids`, a world's rows sharing
one clock; here also empty rows (`exists` false: moribund, not evolved) that `add_balls` fills, so a world can launch
missiles mid-run. A follower's `follow_ptr` is its target's row; a missile's `owner_ptr` is its owner's row (or -1),
refreshed when a ball is added or removed. Formations are one table for the park (`load_formations`; a LoadFormations
mid-run reloads it for every world, as PyLoadFormations clears and reloads Destiny's table, Thunkers.cpp:1921).

**The domain** (each refusal a flag, raised by `check()`): the gradient method; no two balls meet but a normal missile
and its target (there is no other collision code: as VecModesPark); no defender missiles (a negative owner sees every
ball); on the master one bubble a world (the `bubble` ids equal), with an interactive ball (no dead bubbles); no
undefined call (a negative formation slot or index read, a formation index past the end); no delayed removal (a
mushroom's removal in a tick is the one moribund state, its stamp always passed at the next BringOutTheDeads); an
AddMushroom whose ball's id is in its world already (a client's second mushroom, id 0: AddBall's "not created" path) is
FLAG_POOL.

**Exactness.** The tick's kernels run the reference's operations in its order with every rounding pinned
(vec_combat_cuda); `evolve_torch` and the commands are the reference's on tensors: correctly rounded operations in its
order, tensor divisors, float operations as doubles rounded to float, the float functions from exact_ballpark.vec_libm,
dynamical orientation VecOrientPark's own torch code. Without dynamical orientation the yaw, pitch and roll take two
host round trips a tick (with `ypr="host"`, the default), so only a park with it is captured as a CUDA graph
(vec_combat_cuda.CombatGraph, which captures anew after a LoadFormations).

Not here: defender missiles, the iterative collision method, missiles beside VecCollidePark (ships colliding while
missiles fly), RemoveBall with a delay (moribund until BringOutTheDeads), the master's proximity sensors
(exact_ballpark.vec_space), the server layer's actions for these commands (exact_ballpark.vec_server's kinds).
"""

from __future__ import annotations

import math

import torch

from . import vec_combat_cuda as KC
from . import vec_libm as L
from .full import _MUSHROOM_MASS
from .full import _powf as _C_POWF                   # the C library's powf (ctypes), as the reference calls it
from .full import DISPATCH_MIN_MOVE_SQ as FU_DISPATCH_MIN_MOVE_SQ
from .full import FORMATION_MAX_SLOTS, HALF_PI, MISSILE_LAUNCH_SPEED, ROLL_ACCELERATION, ROLL_DECAY, \
    ROLL_SPEED_ACCELERATION, ROLL_SPEED_DECAY
from .full import MISSILE_STRAIGHT_MS as FU_MISSILE_STRAIGHT_MS
from .modes import FOLLOW, FORMATION, MISSILE, ORBIT, RIGID, TROLL, WARP
from .modes import MINIBALL as MINIBALL_MODE
from .modes import MUSHROOM
from .modes import MUSHROOM as MUSHROOM_MODE
from .ref import GOTO, STOP                           # noqa: F401 (GOTO: the tests read vc.GOTO)
from .vec_modes import VecModesPark, q_rotation_ypr  # noqa: F401 (q_rotation_ypr moved there)
from .vec_modes import q_rotation_ypr as _q_rotation_ypr  # (the name the moved _formation_thrust reads)
from .vec_orient import _cols, _stack, m_rotation_q, masked_tick, transform_coord

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
f32 = L.f32
DSTNORMALCLOAK = 1                                  # IDstConstants.h:77
MISSILE_GOTO = 1.e16                                # MissileFollow: mGoto 1e16 m along the velocity (Ballpark.cpp:3867)
SEPARATING_SPEED2 = KC.COMBAT_LITERALS["C_SEPARATING_SPEED2"][1]   # Potential: |v1 - v2|^2 > 0.0001 returns early
FIXED_MASS = KC.COMBAT_LITERALS["C_FIXED_MASS"][1]                 # Potential: a fixed ball's mass, 1.0e34
FOLLOW_MODES = (FOLLOW, ORBIT, MISSILE, FORMATION)  # followModes (Ballpark.cpp:68)
FLAG_DEFENDER, FLAG_UNDEFINED, FLAG_BUBBLES, FLAG_DEAD, FLAG_POOL, FLAG_MASSIVE = (KC.FLAG_DEFENDER, KC.FLAG_UNDEFINED,
                                                                       KC.FLAG_BUBBLES, KC.FLAG_DEAD, KC.FLAG_POOL,
                                                                       KC.FLAG_MASSIVE)
FLAG_NAMES = {FLAG_DEFENDER: "a defender missile", FLAG_UNDEFINED: "a call whose result Destiny leaves undefined",
              FLAG_BUBBLES: "more than one bubble in a world", FLAG_DEAD: "a world without an interactive ball",
              FLAG_POOL: "an AddBall the rows cannot take",
              FLAG_MASSIVE: "two massive balls outside MISSILE mode in a world (no collision code for them here)"}


def _char(n: torch.Tensor) -> torch.Tensor:
    """A conversion to (signed) char: 128 to 255 become negative."""
    return ((n.to(I64) + 128) & 0xFF) - 128


def _fmax(a, b):
    """std::max: (a < b) ? b : a."""
    return torch.where(a < b, b, a)


def _fmin(a, b):
    """std::min: (b < a) ? b : a."""
    return torch.where(b < a, b, a)


class VecCombatPark(VecModesPark):
    """VecModesPark with MISSILE and FORMATION, the missiles' contacts and collision events, cloaking at the park,
    AddBall and RemoveBall mid-run; the tick four CUDA launches (vec_combat_cuda), or on tensors (`evolve_torch`)."""

    def __init__(self, n: int, friction: float = 1.0e6, tick_interval_ms: int = 1000, device="cuda",
                 dynamical_orientation: bool = False, use_new_orbit: bool = False, block: int = 128,
                 master: bool = False, disable_do_for_missiles: bool = False, ypr: str = "host"):
        super().__init__(n, friction, tick_interval_ms, device, dynamical_orientation, use_new_orbit, block, ypr)
        self.tick_ms = int(tick_interval_ms)
        self.master = bool(master)
        self.disable_do_for_missiles = bool(disable_do_for_missiles)
        d = self.dev
        zi = lambda v=0: torch.full((n,), v, dtype=I64, device=d)
        self.exists = torch.ones(n, dtype=BOOL, device=d)
        self.formation_id, self.formation_slots, self.owner_ptr = zi(-1), zi(), zi(-1)
        self.hits_n, self.ev_n = zi(), zi()
        self.ev_self = torch.zeros(n, dtype=BOOL, device=d)
        self.flags = zi()
        self.dying = torch.zeros(n, dtype=torch.uint8, device=d)      # a tick's records (the kernels')
        self.petrified = torch.zeros(n, dtype=torch.uint8, device=d)
        self.local_cnt = zi(-1)                                       # the master's last local id, a world's (space.py)
        self.load_formations([])

    # ---- loading ----

    def load(self, rows: list[dict]) -> None:
        """VecModesPark.load, every row: a ball (as the Python AddBall adds it) or, with "exists": False, an empty row
        of its world (inert until `add_balls`); without dynamical orientation the free balls' yaw, pitch and roll
        snapped (AddOldStyleOrientedBall). Then the world index (rows by world, in row order)."""
        rows = [dict(r) for r in rows]
        for r in rows:
            if not r.get("exists", True):
                r.update(moribund=True, free=False, massive=False, interactive=False, mode=STOP)
        super().load(rows)
        d = self.dev
        self.exists = torch.tensor([bool(r.get("exists", True)) for r in rows], dtype=BOOL, device=d)
        n = self.n
        self.formation_id = torch.full((n,), -1, dtype=I64, device=d)
        self.formation_slots = torch.zeros(n, dtype=I64, device=d)
        self.owner_ptr = torch.full((n,), -1, dtype=I64, device=d)
        self.hits_n, self.ev_n = torch.zeros(n, dtype=I64, device=d), torch.zeros(n, dtype=I64, device=d)
        self.ev_self = torch.zeros(n, dtype=BOOL, device=d)
        self.flags = torch.zeros(n, dtype=I64, device=d)
        self.dying = torch.zeros(n, dtype=torch.uint8, device=d)
        self.petrified = torch.zeros(n, dtype=torch.uint8, device=d)
        self.local_cnt = torch.tensor([int(r.get("local_cnt", -1)) for r in rows], dtype=I64, device=d)
        self._index_worlds()
        if not self.dynamical_orientation:
            z = torch.zeros(n, dtype=F64, device=d)
            for k in ("yaw", "old_yaw", "pitch", "old_pitch", "roll", "old_roll", "roll_speed", "old_roll_speed",
                      "yaw_delta"):
                setattr(self, k, z.clone())
            self._ypr(self.exists & self.free, snap=True)

    def _index_worlds(self) -> None:
        """rows_of_world [W, N] (-1 padded, in row order) and each row's world index (index work only)."""
        labels = self.world.tolist()
        order, groups = [], {}
        for r, w in enumerate(labels):
            if w not in groups:
                groups[w] = []
                order.append(w)
            groups[w].append(r)
        self.labels = order
        width = max((len(g) for g in groups.values()), default=0)
        self.rows_of_world = torch.tensor([g + [-1] * (width - len(g)) for g in (groups[w] for w in order)],
                                          dtype=I64, device=self.dev).reshape(len(order), width)
        index = {w: k for k, w in enumerate(order)}
        self.world_index = torch.tensor([index[w] for w in labels], dtype=I64, device=self.dev)

    def lookup(self, rows: torch.Tensor, ball: torch.Tensor) -> torch.Tensor:
        """For each row, the row of its own world holding the ball with Destiny id ball[i] (`balls.get`), or -1."""
        cand = self.rows_of_world[self.world_index[rows.to(I64)]]                  # [R, N]
        safe = cand.clamp(min=0)
        hit = (cand >= 0) & self.exists[safe] & (self.ids[safe] == ball.to(I64)[:, None])
        idx = hit.to(I64).argmax(dim=1)
        return torch.where(hit.any(dim=1), cand.gather(1, idx[:, None])[:, 0], -1)

    def add_balls(self, mask: torch.Tensor, rows: list) -> None:
        """The Python AddBall of new balls (PyAddBall, Thunkers.cpp:150; MOV-711) into the empty rows in `mask` (one
        dict per masked row, in row order, the keys of `load`; "id" a new id of that world): AddBall's clamps and
        defaults, the decay factor, STOP; the orientation snapped (with dynamical orientation the rotation, without it
        the yaw, pitch and roll of a free ball). A missile whose owner is the new ball finds it again."""
        idx = mask.nonzero().flatten().tolist()
        if len(idx) != len(rows):
            raise ValueError(f"{len(rows)} balls for {len(idx)} rows")
        if not idx:
            return
        t_idx = torch.tensor(idx, dtype=I64, device=self.dev)
        taken = self.exists[t_idx] | (self.lookup(t_idx, torch.tensor([int(r["id"]) for r in rows], dtype=I64,
                                                                    device=self.dev)) >= 0)
        if bool(taken.any()):
            self.flags[t_idx] |= torch.where(taken, FLAG_POOL, 0)
            raise ValueError("add_balls: a row is not empty or the id is already in its world")
        sub = VecModesPark(len(idx), self.friction, self.tick_ms, self.dev, self.dynamical_orientation,
                           self.use_new_orbit, self.block)
        sub.load([{**r, "world": 0, "time": 0} for r in rows])
        names = ("mass", "radius", "max_vel", "agility", "speed_fraction", "pos", "vel", "old_pos", "old_vel", "goto",
                 "last_g", "time_factor", "rot", "old_rot", "ang_vel", "old_ang_vel", "torque", "rot_roll",
                 "max_ang_vel", "rot_agility", "ids", "mode", "free", "massive", "interactive", "moribund", "cloaked",
                 "sensor", "effect_stamp", "follow_id", "owner_id", "follow_ptr", "follow_range", "last_collision")
        if all("bubble" in r for r in rows):                # otherwise the world's bubble the empty row holds
            names = names + ("bubble",)
        for name in names:
            t = getattr(self, name).clone()
            t[t_idx] = getattr(sub, name)
            setattr(self, name, t)
        for name, v in (("exists", True), ("formation_id", -1), ("formation_slots", 0), ("owner_ptr", -1)):
            t = getattr(self, name).clone()
            t[t_idx] = v
            setattr(self, name, t)
        if not self.dynamical_orientation:
            new = torch.zeros(self.n, dtype=BOOL, device=self.dev)
            new[t_idx] = True
            for k in ("yaw", "old_yaw", "pitch", "old_pitch", "roll", "old_roll", "roll_speed", "old_roll_speed",
                      "yaw_delta"):
                setattr(self, k, torch.where(new, 0.0, getattr(self, k)))
            self._ypr(new & self.free, snap=True)
        self._refresh_owners()

    def _refresh_owners(self) -> None:
        """Each missile's owner row (`balls.get(ABS(mOwnerId))`, Ballpark.cpp:633): -1 when it is not in the park."""
        rows = torch.arange(self.n, device=self.dev)
        found = self.lookup(rows, self.owner_id.abs())
        self.owner_ptr = torch.where(self.exists & (self.mode == MISSILE), found, -1)

    # ---- the follower bookkeeping (followModes: FOLLOW, ORBIT, MISSILE, FORMATION) ----

    def _following(self) -> torch.Tensor:
        return ((self.mode == FOLLOW) | (self.mode == ORBIT) | (self.mode == MISSILE)) | (self.mode == FORMATION)

    def _stop_rows(self, mask: torch.Tensor) -> None:
        """Stop(Ball*) (Ballpark.cpp:4578): on the master a FORMATION follower's slot freed in its leader first
        (4629-4633); a follower leaves its target's (implicit) follower set and its follow fields are cleared (the
        owner's row with them); then STOP."""
        if self.master:
            self._free_slots(mask & (self.mode == FORMATION) & (self.follow_ptr >= 0))
        follower = mask & self._following()
        self.owner_ptr = torch.where(follower, -1, self.owner_ptr)
        super()._stop_rows(mask)

    def _free_slots(self, rows: torch.Tensor) -> None:
        """Ball::FreeFormationSlot((size_t) mEffectStamp) (Ball.cpp:2324) on the leader of each row: nothing without
        a loaded formation or for a slot past its end (unsigned comparisons), a slot of 16 or more flagged (Destiny's
        bitset<16>::reset throws: the reference refuses it), the bits cleared per leader (integer steps only)."""
        if not bool(rows.any()):
            return
        lead = self.follow_ptr.clamp(min=0)
        fid = self.formation_id[lead]
        ok = rows & (fid != -1) & ~(fid > self.n_formations - 1)
        bad = ok & (fid < 0)
        sign = torch.iinfo(torch.int64).min
        slot = self.effect_stamp
        past = (slot ^ sign) > ((self._formation_len(fid) - 1) ^ sign)     # size_t slot > numberOfSlots - 1
        go = ok & ~bad & ~past
        wide = go & ((slot < 0) | (slot >= FORMATION_MAX_SLOTS))
        self.flags = self.flags | torch.where(bad | wide, FLAG_UNDEFINED, 0)
        clear = go & ~wide
        bits = torch.zeros(self.n, dtype=I64, device=self.dev)
        for b in range(FORMATION_MAX_SLOTS):
            hit = torch.zeros(self.n, dtype=I64, device=self.dev).index_add_(0, lead, (clear & (slot == b)).to(I64))
            bits = bits | torch.where(hit > 0, 1 << b, 0)
        self.formation_slots = self.formation_slots & ~bits

    def followers(self, row: int) -> list:
        """The follower set of the ball in `row` as StopAllFollowers acts on it (VecModesPark's: follow_ptr names the
        row and the follow id is its id), ascending ids."""
        sel = (self.follow_ptr == row) & (self.follow_id == self.ids[row]) & self.exists
        return sorted(self.ids[sel].tolist())

    def _g11(self, name: str) -> None:
        """the commands and AddMushroom are batched for this park's own tick (its kernels and
        `evolve_torch`); a subclass with its own tick (VecComposedPark) refuses them."""
        if type(self).evolve is not VecCombatPark.evolve or type(self).evolve_torch is not VecCombatPark.evolve_torch:
            raise NotImplementedError(f"{name} is batched for VecCombatPark's own tick only, not {type(self).__name__}")

    def _release_followers(self, rows: torch.Tensor) -> None:
        """StopAllFollowers (Ballpark.cpp:5416; MOV-730) of the balls in `rows`: each member (Destiny's test: `_members`) released (`_release_members`)."""
        self._release_members(self._members(rows) & self.exists)

    def _release_members(self, f: torch.Tensor) -> None:
        """StopAllFollowers on the members in `f`: a missile is made non-massive and, as an interactive orbiter, flies on
        by GotoDirection along its velocity; every other member stops (Stop(Ball*): a member out of the follow modes
        keeps its follow fields)."""
        missile = f & (self.mode == MISSILE)
        flies = missile | (f & self.interactive & (self.mode == ORBIT))
        self.massive = torch.where(missile, False, self.massive)
        self._stop_rows(f & ~flies)
        self.goto_direction(flies, self.vel.clone())

    # ---- missiles (MOV-723 to MOV-725) ----

    def missile_follow(self, mask: torch.Tensor, target: torch.Tensor, owner: torch.Tensor) -> None:
        """PyMissileFollow and MissileFollow (MOV-724) for the missiles in `mask`: target is each row's target ball's
        Destiny id, owner the owner's id (negative: a defender)."""
        rows = torch.arange(self.n, device=self.dev)
        dst, owner = target.to(I64), owner.to(I64)
        trow = self.lookup(rows, dst)
        tsafe = trow.clamp(min=0)
        src = self.ids
        ok = (mask & self.exists & (dst >= 0) & (src != dst) & (src != owner) & (dst != owner) & (trow >= 0)
              & ~self.moribund[tsafe])
        self._stop_rows(ok)
        ok = ok & (self.bubble == self.bubble[tsafe]) & (self.cloaked[tsafe] == 0)
        self.follow_id = torch.where(ok, dst, self.follow_id)
        self.follow_ptr = torch.where(ok, trow, self.follow_ptr)
        self.follow_range = torch.where(ok, -f32(self.radius + self.radius[tsafe]), self.follow_range)
        self.owner_id = torch.where(ok, owner, self.owner_id)
        self.owner_ptr = torch.where(ok, self.lookup(rows, owner.abs()), self.owner_ptr)
        self.effect_stamp = torch.where(ok, self.clock, self.effect_stamp)
        self.goto = torch.where(ok[:, None], self._unit(self.vel) * MISSILE_GOTO, self.goto)
        self.speed_fraction = torch.where(ok, 1.0, self.speed_fraction)
        self.mode = torch.where(ok, MISSILE, self.mode)
        self.flags |= torch.where(ok & (owner < 0), FLAG_DEFENDER, 0)

    def launch_missile(self, mask: torch.Tensor, target: torch.Tensor, owner: torch.Tensor, aimed: torch.Tensor,
                       massive: torch.Tensor) -> None:
        """PyLaunchMissile (Thunkers.cpp:857; MOV-725) for the missile balls in `mask`: target and owner are Destiny
        ids (a negative owner: a defender, its launcher found by |owner|), aimed and massive the thunk's chars."""
        rows = torch.arange(self.n, device=self.dev)
        dst, owner = target.to(I64), owner.to(I64)
        aimed, massive = _char(aimed), _char(massive)
        go = mask & (dst >= 0)
        lrow = self.lookup(rows, owner.abs())
        go = go & (lrow >= 0)
        lsafe = lrow.clamp(min=0)
        trow = torch.where(dst != owner, self.lookup(rows, dst), -1)
        has_t = trow >= 0
        tsafe = trow.clamp(min=0)
        ps = self.pos[lsafe]
        aim = has_t & (aimed == 1) & (self.cloaked[tsafe] == 0)
        v_aim = self._unit(self.pos[tsafe] - ps)
        vs = self.vel[lsafe]
        max_velocity = torch.where(aim, 0.0, self.max_vel[lsafe])
        direction = self._unit(vs)
        still = (direction[:, 0] * direction[:, 0] + direction[:, 1] * direction[:, 1]
                 + direction[:, 2] * direction[:, 2]) == 0.0
        if bool((go & ~aim & still).any()):
            back = self.rotated_vector(lsafe, (0.0, 0.0, -1.0))
            direction = torch.where(still[:, None], back, direction)
        speed = _fmax(torch.full_like(max_velocity, MISSILE_LAUNCH_SPEED), max_velocity)
        v0 = torch.where(aim[:, None], v_aim, vs + direction * speed[:, None])
        here = go & self.exists
        self.massive = torch.where(here & (massive != 0), True, self.massive)
        self.pos = torch.where(here[:, None], ps, self.pos)                  # SetBallPosition (no partition here)
        self.old_pos = torch.where(here[:, None], ps, self.old_pos)
        self.set_velocity(here, v0)
        cloaked_t = has_t & (self.cloaked[tsafe] != 0)
        self.goto_direction(here & ~has_t & (max_velocity > 0.0), v0)
        self.missile_follow(go & has_t & ~cloaked_t, dst, owner)

    # ---- formations (MOV-726) ----

    def load_formations(self, formations: list) -> None:
        """PyLoadFormations and AddFormation: the slot offsets (doubles), one table for the park."""
        f = [[[float(c) for c in v] for v in vectors] for vectors in formations]
        self.formations = f
        self.n_formations = len(f)
        self.max_slots = max([len(v) for v in f] + [1])
        table = torch.zeros(max(1, len(f)), self.max_slots, 3, dtype=F64, device=self.dev)
        for i, vectors in enumerate(f):
            if vectors:
                table[i, :len(vectors)] = torch.tensor(vectors, dtype=F64, device=self.dev)
        self.formation_table = table.contiguous()
        self.formation_lengths = torch.tensor([len(v) for v in f] or [0], dtype=I64, device=self.dev)

    def _formation_len(self, fid: torch.Tensor) -> torch.Tensor:
        safe = fid.clamp(0, max(0, self.n_formations - 1))
        return self.formation_lengths[safe]

    def set_ball_formation(self, mask: torch.Tensor, formation: torch.Tensor) -> None:
        """PySetBallFormation and SetBallFormation (Ballpark.cpp:6673): leaving a formation releases the followers
        and frees every slot; a smaller formation stops the FORMATION followers in the slots it lacks."""
        fid = _char(formation)
        F = self.n_formations
        ok = mask & self.exists & ~(fid > F - 1)
        cur = self.formation_id
        inside = ok & (cur >= 0)
        leave = inside & (fid == -1)
        self._release_followers(leave)
        self.formation_slots = torch.where(leave, 0, self.formation_slots)
        change = inside & (fid != -1) & (fid != cur)
        bad = change & ((fid < 0) | (cur > F - 1))
        self.flags |= torch.where(bad, FLAG_UNDEFINED, 0)
        change = change & ~bad
        new_len = self._formation_len(fid)
        shrink = change & (self._formation_len(cur) > new_len)
        safe = torch.where(self.follow_ptr >= 0, self.follow_ptr, 0)
        f = (self.exists & (self.mode == FORMATION) & (self.follow_ptr >= 0) & shrink[safe]
             & (self.effect_stamp > new_len[safe] - 1))
        self._stop_rows(f)
        self.formation_id = torch.where(ok & ~bad, fid, self.formation_id)

    def formation_follow(self, mask: torch.Tensor, leader: torch.Tensor, slot: torch.Tensor) -> None:
        """PyFormationFollow and FormationFollow (Ballpark.cpp:3947) for the balls in `mask`: leader is each row's
        leader's Destiny id, slot the thunk's char."""
        rows = torch.arange(self.n, device=self.dev)
        slot, dst = _char(slot), leader.to(I64)
        lrow = self.lookup(rows, dst)
        lsafe = lrow.clamp(min=0)
        fid = self.formation_id[lsafe]
        ok = (mask & self.exists & (self.ids != dst) & (lrow >= 0) & (fid >= 0) & ~(fid > self.n_formations - 1)
              & ~(slot > self._formation_len(fid) - 1))
        self._stop_rows(ok)
        ok = ok & (self.bubble == self.bubble[lsafe]) & (self.cloaked[lsafe] == 0)
        self.follow_id = torch.where(ok, dst, self.follow_id)
        self.follow_ptr = torch.where(ok, lrow, self.follow_ptr)
        self.effect_stamp = torch.where(ok, slot, self.effect_stamp)
        self.mode = torch.where(ok, FORMATION, self.mode)
        self.flags |= torch.where(ok & (slot < 0), FLAG_UNDEFINED, 0)   # EvolveFormation reads a negative slot

    def reserve_formation_slot(self, row: int) -> int:
        """Ball::ReserveFormationSlot (Ball.cpp:2275): the first free slot of at most 16, or -1 (one ball, on the
        host: a rare call)."""
        fid = int(self.formation_id[row])
        if fid == -1 or fid > self.n_formations - 1:
            return -1
        if fid < 0:
            self.flags[row] |= FLAG_UNDEFINED
            return -1
        n = len(self.formations[fid])
        if n > FORMATION_MAX_SLOTS:
            return -1
        bits = int(self.formation_slots[row])
        for i in range(n):
            if bits >> i & 1:
                continue
            self.formation_slots[row] = bits | (1 << i)
            return i
        return -1

    def free_formation_slot(self, row: int, slot: int) -> None:
        """Ball::FreeFormationSlot (Ball.cpp:2324), with a size_t slot (one ball, on the host)."""
        fid = int(self.formation_id[row])
        if fid == -1 or fid > self.n_formations - 1:
            return
        if fid < 0:
            self.flags[row] |= FLAG_UNDEFINED
            return
        n = len(self.formations[fid])
        slot = int(slot) & 0xFFFFFFFFFFFFFFFF
        if slot > ((n - 1) & 0xFFFFFFFFFFFFFFFF):
            return
        if slot >= FORMATION_MAX_SLOTS:
            self.flags[row] |= FLAG_UNDEFINED
            return
        self.formation_slots[row] = int(self.formation_slots[row]) & ~(1 << slot)

    # ---- cloaking (SEN-711) and removal ----

    def cloak(self, mask: torch.Tensor, mode: torch.Tensor, range_: torch.Tensor | None = None) -> None:
        """CloakBall (FullBallpark.cloak_ball; Ballpark.cpp:5223): refused unless the (signed char) mode is positive;
        on the master a normal cloak sets the sensor flag (AddProximitySensor's refusal of float(radius) +
        float(range) < 0 kept; the sensor is vec_space's); the followers released; cloaked with the mode's byte,
        non-massive."""
        mode = _char(mode)
        ok = mask & self.exists & (mode > 0)
        if self.master:
            rng = f32(torch.full((self.n,), 2000.0, dtype=F64, device=self.dev) if range_ is None
                      else range_.to(F64))
            self.sensor = self.sensor | (ok & (mode == DSTNORMALCLOAK) & ~(f32(self.radius + rng) < 0.0))
        self._release_followers(ok)
        self.cloaked = torch.where(ok, mode & 0xFF, self.cloaked)
        self.massive = torch.where(ok, False, self.massive)

    def uncloak(self, mask: torch.Tensor) -> None:
        """UncloakBall (FullBallpark.uncloak_ball; Ballpark.cpp:5257): the normal cloak's sensor flag cleared (master);
        uncloaked; massive again unless in warp proper."""
        ok = mask & self.exists
        if self.master:
            self.sensor = self.sensor & ~(ok & (self.cloaked == DSTNORMALCLOAK))
        self.cloaked = torch.where(ok, 0, self.cloaked)
        warping = (self.mode == WARP) & (self.effect_stamp >= 0)
        self.massive = torch.where(ok & ~warping, True, self.massive)

    def remove(self, mask: torch.Tensor) -> None:
        """RemoveBall outside Evolve without a delay (Ballpark.cpp:5471; TIME-711): the ball stopped, its followers
        released (a missile flies off), its row emptied (a missile whose owner it was is orphaned)."""
        ok = mask & self.exists
        self._stop_rows(ok)
        self._release_followers(ok)
        # a member out of the follow modes keeps its fields, a pointer to a ball now gone: in no follower set again
        stale = self._members(ok) & self.exists
        self.follow_ptr = torch.where(stale | ok, -1, self.follow_ptr)
        self.exists = self.exists & ~ok
        self.moribund = self.moribund | ok
        self.massive = torch.where(ok, False, self.massive)
        self.interactive = torch.where(ok, False, self.interactive)
        self.sensor = torch.where(ok, False, self.sensor)
        self.formation_id = torch.where(ok, -1, self.formation_id)
        self.formation_slots = torch.where(ok, 0, self.formation_slots)
        self._refresh_owners()

    # ---- mushrooms (MOV-734) ----

    def add_mushroom(self, mask: torch.Tensor, owner: torch.Tensor, range_: torch.Tensor, life: torch.Tensor) -> None:
        """PyAddMushroom ("Lfd": the owner's id, the range a float, the lifetime a double) and AddMushroom
        (Ballpark.cpp:3719) into the empty rows in `mask`, each in its owner's world: refused (the row left empty) for a
        range or a lifetime not above 0 (`range <= 0.0 || time <= 0`) or an owner not in the world (mBalls[ownerId]);
        else a ball as AddBall adds it (MOV-711: mass 1e20f, radius 0, maximum speed 0, fixed, massive, not interactive,
        at the owner's position, at rest, agility and speed fraction 1; with dynamical orientation at rest, maximum
        angular speed 0, rotation agility 1), its id 0 on a client and the world's next local id on the master
        (GetIDForCollisionObject, taken twice with dynamical orientation: AddDynamicallyOrientedBall takes one before
        AddBall), then its range (follow_range), lifetime (goto's x), owner's id (owner_id), the clock (effect_stamp),
        MUSHROOM. An id already in the world (AddBall's "not created" path) is not batched: FLAG_POOL. Index work on the
        host (a rare command)."""
        self._g11("AddMushroom")
        idx = mask.nonzero().flatten().tolist()
        if not idx:
            return
        if bool((self.exists[mask]).any()):
            raise ValueError("add_mushroom: a row is not empty")
        t_idx = torch.tensor(idx, dtype=I64, device=self.dev)
        orow = self.lookup(t_idx, owner.to(I64)[t_idx]).tolist()
        rng = f32(range_.to(device=self.dev, dtype=F64)[t_idx]).tolist()
        tl = life.to(device=self.dev, dtype=F64)[t_idx].tolist()
        own = owner.to(I64)[t_idx].tolist()
        widx, cnt, clock = self.world_index.tolist(), self.local_cnt.tolist(), self.clock.tolist()
        pos = self.pos.tolist()
        add, fields = [], []
        for r, o, rg, t, oid in zip(idx, orow, rng, tl, own):
            if rg <= 0.0 or t <= 0 or o < 0:
                continue
            mid = 0
            if self.master:
                mid = cnt[r] - (2 if self.dynamical_orientation else 1)
                for q in range(self.n):
                    if widx[q] == widx[r]:
                        cnt[q] = mid
            hit = self.lookup(torch.tensor([r], dtype=I64, device=self.dev),
                              torch.tensor([mid], dtype=I64, device=self.dev))
            if int(hit[0]) >= 0:
                self.flags = self.flags | torch.where(self.world_index == widx[r], FLAG_POOL, 0)
                continue
            add.append((r, {"id": mid, "mass": _MUSHROOM_MASS, "radius": 0.0, "max_vel": 0.0, "free": False,
                            "massive": True, "interactive": False, "pos": pos[o], "vel": [0.0, 0.0, 0.0],
                            "agility": 1.0, "speed_fraction": 1.0, "max_ang_vel": 0.0, "rot_agility": 1.0}))
            fields.append((r, rg, t, oid, clock[r]))
        self.local_cnt = torch.tensor(cnt, dtype=I64, device=self.dev)
        for r, row in add:                                  # one at a time: ids are checked against the world's
            one = torch.zeros(self.n, dtype=BOOL, device=self.dev)
            one[r] = True
            self.add_balls(one, [row])
        if fields:
            rows = torch.tensor([f[0] for f in fields], dtype=I64, device=self.dev)
            hit = torch.zeros(self.n, dtype=BOOL, device=self.dev)
            hit[rows] = True
            put = lambda name, vals, dt: getattr(self, name).index_put((rows,), torch.tensor(vals, dtype=dt,
                                                                                               device=self.dev))
            self.follow_range = put("follow_range", [f[1] for f in fields], F64)
            g = self.goto.clone()
            g[rows, 0] = torch.tensor([f[2] for f in fields], dtype=F64, device=self.dev)
            self.goto = g
            self.owner_id = put("owner_id", [f[3] for f in fields], I64)
            self.effect_stamp = put("effect_stamp", [f[4] for f in fields], I64)
            self.mode = torch.where(hit, MUSHROOM_MODE, self.mode)

    def _mushroom_fraction(self) -> torch.Tensor:
        """Each row's `(mCurrentTime - startTime)*mTickInterval*0.001f/(float)(mGoto[0])` (Ballpark.cpp:566) in float:
        the integer product made a float (to double, then float, as the reference: one rounding below 2^53 ms), times
        0.001f, over the lifetime made a float; meaningful for MUSHROOM rows."""
        elapsed = f32(((self.clock - self.effect_stamp) * self.tick_ms).to(F64))
        a = f32(elapsed * f32(torch.full_like(elapsed, 0.001)))
        return f32(a / f32(self.goto[:, 0]))

    def bring_out_the_deads(self, mask: torch.Tensor | None = None) -> None:
        """BringOutTheDeads (Ballpark.cpp:5699; space.py's), between ticks as OnTick calls it, in the worlds of the rows
        in `mask` (default every row): each moribund ball whose stamp has passed is removed (`remove`: stopped, its
        followers released, its row emptied). Here the one moribund state is a mushroom removed in a tick (its stamp
        that tick's time, so always passed: no ball is kept by the buffer of up to seven within two ticks, which only
        RemoveBall with a delay reaches)."""
        due = self.exists & self.moribund & (self.effect_stamp - self.clock < 0)
        self.remove(due if mask is None else due & mask)

    # ---- OnTick's per-world mask (VecOrientPark.on_tick) ----

    _ROW_EXCLUDED = VecModesPark._ROW_EXCLUDED + ("rows_of_world", "formation_table", "formation_lengths")

    def _on_tick_master(self) -> bool:
        """On the master (`master`) OnTick cuts a backlog of more than five intervals to one (TIME-703)."""
        return self.master

    def _bring_out_on_tick(self) -> None:
        """OnTick's BringOutTheDeads before the interval check, every world (Ballpark.cpp:231; TIME-713)."""
        self.bring_out_the_deads()

    def _ticked_rows(self, sel: torch.Tensor) -> torch.Tensor:
        """sel, less the rows outside the last tick's mask when that tick was a masked one (vec_orient.masked_tick's
        `_ticked`): a world without a tick posts no event."""
        last = getattr(self, "_ticked", None)
        if last is not None and last[0] == self.current_time:
            return sel & last[1]
        return sel

    # ---- the setters that snap the orientation ----

    def set_velocity(self, mask: torch.Tensor, v: torch.Tensor) -> None:
        """SetBallVelocity (MOV-713): VecModesPark's, and without dynamical orientation the yaw, pitch and roll
        snapped (FullBallpark.set_velocity)."""
        super().set_velocity(mask, v)
        if not self.dynamical_orientation:
            self._ypr(mask & self.exists, snap=True)

    # ---- the orientation without dynamical orientation (MOV-729), every mode ----

    def _ypr(self, mask: torch.Tensor, snap: bool, time: torch.Tensor | None = None) -> None:
        """Ball::CalculateYawPitchRoll (Ball.cpp:335; FullBallpark._yaw_pitch_roll) for the rows in mask: atan2, asin
        and sin on the host (Python's math is the C library's) or, with ypr="cuda", on the device
        (exact_ballpark.vec_ypr_cuda: glibc's own algorithms, the same bits, no host round trip), the rest in the
        reference's order on tensors. `time` is each row's mCurrentTime (the tick's, before the clock moves)."""
        device = self.ypr == "cuda"
        if not device and not bool(mask.any()):         # (on the device no host sync: rows outside mask are kept)
            return
        time = self.clock if time is None else time
        old_pitch, old_yaw, old_roll, old_roll_speed = self.pitch, self.yaw, self.roll, self.roll_speed
        mm = self.mass / torch.full_like(self.mass, self.friction)
        mm = torch.where(mm != 0.0, torch.full_like(mm, 100.0) / torch.where(mm != 0.0, mm, 1.0), 100.0)
        v = self.vel
        vel2 = v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2]
        a = self.goto - self.pos
        warp = self.mode == WARP
        aligning = warp & (self.effect_stamp < 0)
        proper = warp & ~(self.effect_stamp < 0)
        aim = torch.where(self._collided()[:, None], v, a)      # an aligning ball with collisions: its velocity
        direction = torch.where(warp[:, None], torch.where(aligning[:, None], aim, torch.zeros_like(v)), v)
        span = (time - self.effect_stamp) * 2
        inner = torch.where(span > 15, 15.0, span.to(F64))       # 15.0 if 15.0 < inner else float(inner)
        x = mm * inner
        moved = proper & (vel2 != 0.0) & (mm < 1.0) & (time != self.effect_stamp)
        mm = torch.where(moved, torch.where(x > 1.0, 1.0, x), mm)
        if not snap:
            norm = torch.sqrt(a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1] + a[:, 2] * a[:, 2])
            inv = torch.ones_like(norm) / torch.where(norm == 0.0, 1.0, norm)
            offset = torch.where((norm == 0.0)[:, None], a, a * inv[:, None])
            mixed = direction + offset * (torch.sqrt(vel2) * 0.3)[:, None]
            direction = torch.where((self.mode != MISSILE)[:, None], mixed, direction)
        direction = torch.where((vel2 == 0.0)[:, None], a, direction)
        r = torch.sqrt(direction[:, 0] * direction[:, 0] + direction[:, 1] * direction[:, 1]
                       + direction[:, 2] * direction[:, 2])
        sine = direction[:, 1] / torch.where(r > 0.0, r, 1.0)
        sine = _fmin(torch.ones_like(sine), _fmax(-torch.ones_like(sine), sine))
        if device:
            from . import vec_ypr_cuda as Y
            yaw = Y.atan2(direction[:, 0].contiguous(), direction[:, 2].contiguous())
            pitch = -Y.asin(sine)                       # sine is clamped to [-1, 1] or NaN: asin's NaN either way
        else:
            host = torch.stack((direction[:, 0], direction[:, 2], sine), dim=1).cpu().tolist()
            yaw = torch.tensor([math.atan2(d0, d2) for d0, d2, _ in host], dtype=F64, device=self.dev)
            pitch = torch.tensor([-math.asin(s) if -1.0 <= s <= 1.0 else math.nan for _, _, s in host], dtype=F64,
                                 device=self.dev)
        pitch = torch.where(r > 0.0, pitch, 0.0)
        old_yaw_m, yaw = self._modulo2pi(old_yaw, yaw)
        old_pitch_m, pitch = self._modulo2pi(old_pitch, pitch)
        if snap:
            z = torch.zeros_like(yaw)
            new = dict(yaw=yaw, pitch=pitch, old_yaw=old_yaw_m, old_pitch=old_pitch_m, roll=z, old_roll=z,
                       roll_speed=z, old_roll_speed=z, yaw_delta=z)
        else:
            tmp = torch.ones_like(mm) / (1.0 + mm * 0.5)
            yaw = (old_yaw_m + mm * 0.5 * yaw) * tmp
            pitch = (old_pitch_m + mm * 0.5 * pitch) * tmp
            if device:                                  # glibc's sin of an infinity or NaN is NaN, as the host's guard
                yaw_delta = Y.asin_sin(yaw - old_yaw_m)
            else:
                change = (yaw - old_yaw_m).cpu().tolist()
                yaw_delta = torch.tensor([math.asin(math.sin(c)) if math.isfinite(c) else math.nan for c in change],
                                         dtype=F64, device=self.dev)
            roll_speed = self.roll_speed + yaw_delta * ROLL_SPEED_ACCELERATION
            roll_speed = roll_speed + old_roll_speed * ROLL_SPEED_DECAY * self.dt
            roll_speed = _fmax(torch.full_like(mm, -HALF_PI), _fmin(torch.full_like(mm, HALF_PI), roll_speed))
            roll = self.roll + roll_speed * ROLL_ACCELERATION * self.dt
            roll = roll + old_roll * ROLL_DECAY * self.dt
            roll = _fmax(torch.full_like(mm, -HALF_PI), _fmin(torch.full_like(mm, HALF_PI), roll))
            new = dict(yaw=yaw, pitch=pitch, old_yaw=old_yaw_m, old_pitch=old_pitch_m, roll=roll, old_roll=old_roll,
                       roll_speed=roll_speed, old_roll_speed=old_roll_speed, yaw_delta=yaw_delta)
        for k, val in new.items():
            setattr(self, k, torch.where(mask, val, getattr(self, k)))

    def _collided(self) -> torch.Tensor:
        """Whether each row's collision list (Ball::mCollisions) is non-empty, as CalculateYawPitchRoll reads it for an
        aligning warper (`b.collisions`): never here, where only a missile collides (the composed park,
        exact_ballpark.vec_composed, holds the lists)."""
        return torch.zeros(self.n, dtype=BOOL, device=self.dev)

    # ---- one tick (Ballpark::Evolve, TIME-706, TIME-707) ----

    def evolve(self, lib: dict | None = None, active: torch.Tensor | None = None) -> None:
        """One tick of every world, in place. On any device but a CUDA one, the tick on tensors (`evolve_torch`;
        `lib`, a kernel library, is refused there). On a CUDA device: the four launches
        (vec_combat_cuda.tick), then without dynamical orientation the yaw, pitch and roll of every live ball
        (phase 4's CalculateYawPitchRoll, on the host), then the
        fifth launch (vec_combat_cuda.after: a removed mushroom's members whose turns came first; PetrifyTroll) and,
        without dynamical orientation, the petrified trolls' yaw, pitch and roll snapped (SetBallVelocity's).
        `active` [n] (OnTick's per-world mask, a world's rows together): only those rows tick, every
        other row keeps every per-row tensor bit for bit (vec_orient.masked_tick, in place; `events()` then reports
        the rows that ticked)."""
        if self.dev.type != "cuda":                 # the tick on tensors off a CUDA device
            if lib is not None:
                raise ValueError("lib is a kernel library: a park on the CPU ticks on tensors (evolve_torch)")
            self.evolve_torch(active)
            return
        if active is not None:
            masked_tick(self, active, lambda: VecCombatPark.evolve(self, lib), in_place=True)
            return
        self.watch()
        KC.tick(self, self.block, lib)
        self.after_tick()
        KC.after(self, self.block, lib)
        if not self.dynamical_orientation:
            self._ypr(self.petrified.bool() & self.exists, snap=True, time=self.clock - 1)
        self.watch()
        self.current_time += 1

    # ---- the tick on tensors (the CPU path, and a cross-check of the kernels on the GPU) ----

    @staticmethod
    def _spheres(p0, p1, q0, q1, cr) -> torch.Tensor:
        """CollideTwoSpheres (Collision.cpp:108; COL-705) per row: 0, the earlier root in [0, 1], or -1."""
        d = p0 - q0
        p0q0_2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
        e = (p1 - p0) - (q1 - q0)
        a = e[:, 0] * e[:, 0] + e[:, 1] * e[:, 1] + e[:, 2] * e[:, 2]
        b = 2.0 * (d[:, 0] * e[:, 0] + d[:, 1] * e[:, 1] + d[:, 2] * e[:, 2])
        c = p0q0_2 - cr * cr
        det = b * b - 4.0 * a * c
        root = ((-b - torch.sqrt(det)) * 0.5) / a              # IEEE: a zero a gives an infinity or NaN, refused
        s = torch.where((root >= 0.0) & (root <= 1.0), root, -1.0)
        s = torch.where(det < 0.0, -1.0, s)
        s = torch.where((a == 0.0) & (b == 0.0), -1.0, s)
        return torch.where(p0q0_2 <= cr * cr, 0.0, s)

    def _contacts(self, live: torch.Tensor) -> torch.Tensor:
        """Each massive normal missile's Potential with its target (the gradient's recursion at depths 0 to 2): how
        many times the target's id is appended (vec_combat_cuda.cu's `contacts`, on tensors)."""
        t = self.follow_ptr
        ok_t = (t >= 0) & (t < self.n)
        ts = torch.where(ok_t, t, 0)
        go = (live & self.massive & (self.mode == MISSILE) & (self.owner_id >= 0) & ok_t
              & (self.bubble == self.bubble[ts]))
        mt = self.mode[ts]
        skip = ((self.ids[ts] == self.owner_id) | ((mt == MINIBALL_MODE) & (self.owner_id[ts] == self.owner_id))
                | (((mt == MISSILE) | (mt == MUSHROOM_MODE)) & (self.ids == self.owner_id[ts]))
                | ((mt == MISSILE) & (self.ids == self.follow_id[ts])))
        go = go & ~skip
        m1 = (self.mass * self.agility)[:, None]
        free_o = self.free[ts]
        m2 = torch.where(free_o, self.mass[ts] * self.agility[ts], FIXED_MASS)[:, None]
        vme, vo, gme, go_ = self.vel, self.vel[ts], self.last_g, self.last_g[ts]
        tf1, tf2 = self.time_factor[:, None], self.time_factor[ts][:, None]
        cr = self.radius + self.radius[ts]
        idm, ido = self.ids, self.ids[ts]
        sum_m = m1 + m2

        def s_of(P, Q):
            p1, _ = self._integrate(P, vme, gme, m1, tf1)
            q1, _ = self._integrate(Q, vo, go_, m2, tf2)
            return self._spheres(P, p1, Q, q1, cr)

        def push(P, Q):
            d = Q - P
            d2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
            sign = torch.where(idm > ido, 1.0, -1.0)
            axis = torch.stack((sign, torch.zeros_like(sign), torch.zeros_like(sign)), dim=1)
            normal = torch.where((d2 == 0.0)[:, None], axis, self._unit(d))
            dist = (cr - torch.sqrt(d2)) + 1.0
            d1 = (m2 / sum_m)[:, 0] * dist
            dd2 = (m1 / sum_m)[:, 0] * dist
            return P - normal * d1[:, None], Q + normal * dd2[:, None]

        P0, Q0 = self.pos, self.pos[ts]
        s0 = s_of(P0, Q0)
        P1, Q1 = push(P0, Q0)
        s1 = s_of(P1, Q1)
        P2, Q2 = push(P1, Q1)
        s2 = s_of(P2, Q2)
        rel = vme - vo
        early = (rel[:, 0] * rel[:, 0] + rel[:, 1] * rel[:, 1] + rel[:, 2] * rel[:, 2]) > SEPARATING_SPEED2
        one, zero = torch.ones_like(idm), torch.zeros_like(idm)
        back = torch.where(early, zero, one)
        c2 = torch.where(s2 == -1.0, zero, one)
        c1 = torch.where(s1 == -1.0, zero, torch.where(s1 > 0.0, one, torch.where(~free_o, one, c2 + back)))
        c0 = torch.where(s0 == -1.0, zero, torch.where(s0 > 0.0, one, torch.where(~free_o, one, c1 + back)))
        return torch.where(go, c0, zero)

    def _orphans(self) -> torch.Tensor:
        """The master's test for a missile's self-collision (Ballpark.cpp:614-638; COL-719): its owner (the row of
        |mOwnerId|) gone, in another bubble while the missile has one, or cloaked."""
        o = self.owner_ptr
        osafe = o.clamp(min=0)
        gone = (o < 0) | ~self.exists[osafe]
        return gone | ((self.bubble[osafe] != self.bubble) & (self.bubble != -1)) | (self.cloaked[osafe] != 0)

    def evolve_torch(self, active: torch.Tensor | None = None) -> None:
        """One tick on tensors (the CPU path; on a CUDA device a cross-check of the kernels) for every mode
        (VecComposedPark's tick moved here, its collisions hooks): phase 1
        (`_phase1`: the releases, every mode's behaviour, dynamical orientation), the warp paths, the contacts
        (`_contacts_phase`: a normal missile's with its target here; VecComposedPark's gradient), phases 3 and 4
        (the warp path or the integrator with mLastG + mLastC, the drop-out's extra step, the rotation a tick on or
        the disabled missile's snap, the commit, the dispatch counts), the clock, then `after_tick`; a free
        mushroom grows at its phase-4 turn (glibc's powf, no swap) or is removed there (its members released
        before their own commit when their ids are higher, after phase 4 otherwise), and PetrifyTroll after phase
        4 (`_after_phase4`), as the kernels. VecComposedPark's re-filings and collision lists are
        its hooks (`_warp_massive`, `_refile`, `_last_c`, `_list_self_collisions`). `active` [n]:
        OnTick's per-world mask, as `evolve`'s."""
        if active is not None:
            masked_tick(self, active, lambda: VecCombatPark.evolve_torch(self), in_place=True)
            return
        self.watch()
        live = self.free & ~self.moribund & self.exists
        do = self.dynamical_orientation
        frac = self._mushroom_fraction()                # the mushrooms that go this tick (from their own fields)
        mush = live & (self.mode == MUSHROOM)
        dying = mush & ~(frac < 1.0)
        self.dying = dying.to(torch.uint8)
        self._phase1(live)
        # the gradient's mLastCollision; the warp paths (a warper's own state only); the gradient
        warping = live & (self.mode == WARP) & (self.effect_stamp >= 0)
        self._warp_massive(warping)
        self.last_collision = torch.where(live & self.massive, -1.0, self.last_collision)
        wp, wv, dropped = self._warp_path(warping)
        rest = self._at_rest(dropped, wv)
        self._contacts_phase(live, wp, wv, dropped, rest)
        # the drop-out: massive and sensing again, Stop, its state the drop-out point's, re-filed at rest
        self.massive = torch.where(dropped, True, self.massive)
        self.sensor = torch.where(dropped, True, self.sensor)
        self._stop_rows(dropped)
        self.pos = torch.where(dropped[:, None], wp, self.pos)
        self.vel = torch.where(dropped[:, None], wv, self.vel)
        self._refile(rest)
        # phase 3: the end-of-tick states; the rotation a tick on, by the mode then
        a = self.last_g + self._last_c()
        pn, vn = self._integrate_rows(self.pos, self.vel, a)
        coast = warping & ~dropped
        pn = torch.where(coast[:, None], wp, pn)
        vn = torch.where(coast[:, None], wv, vn)
        if do:
            off = self.disable_do_for_missiles & (self.mode == MISSILE)
            rot1, w1 = self._apply_torque(_cols(f32(self.torque)), f32(self.rot_roll), self._tau())
            rot1 = torch.where(off[:, None], self.old_rot, _stack(rot1))
            w1 = torch.where(off[:, None], self.old_ang_vel, _stack(w1))
        # a dying mushroom's members (StopAllFollowers at its turn): those whose turns come after it released before
        # their commit, from their state before the swap; the others after phase 4 (`_after_phase4`)
        tsafe = torch.where(self.follow_ptr >= 0, self.follow_ptr, 0)
        members = self._members(dying) & self.exists
        self._release_rows(members & (self.ids[tsafe] < self.ids))
        later = members & (self.ids[tsafe] > self.ids)
        commit = live & ~mush                           # a mushroom skips the swap: its integrated state stays old
        sel, msel = commit[:, None], mush[:, None]
        before = self.pos
        self.old_pos = torch.where(sel, self.pos, torch.where(msel, pn, self.old_pos))
        self.old_vel = torch.where(sel, self.vel, torch.where(msel, vn, self.old_vel))
        self.pos = torch.where(sel, pn, self.pos)
        self.vel = torch.where(sel, vn, self.vel)
        if do:
            off = self.disable_do_for_missiles & (self.mode == MISSILE)      # phase 4's mode (after a release)
            q = rot1
            rot_ok = torch.isfinite(f32(f32(f32(f32(q[:, 0] * q[:, 0]) + f32(q[:, 1] * q[:, 1]))
                                            + f32(q[:, 2] * q[:, 2])) + f32(q[:, 3] * q[:, 3])))
            w_ok = torch.isfinite(f32(f32(f32(w1[:, 0] * w1[:, 0]) + f32(w1[:, 1] * w1[:, 1]))
                                      + f32(w1[:, 2] * w1[:, 2])))
            r0, w0 = self.rot, self.ang_vel
            rn = torch.where((sel & rot_ok[:, None]), rot1, r0)
            ro = torch.where(sel & rot_ok[:, None], r0, torch.where(sel | msel, rot1, self.old_rot))
            wn = torch.where(sel & w_ok[:, None], w1, w0)
            wo = torch.where(sel & w_ok[:, None], w0, torch.where(sel | msel, w1, self.old_ang_vel))
            self.rot, self.old_rot, self.ang_vel, self.old_ang_vel = rn, ro, wn, wo
            self._snap_orientation(commit & off)
        # a mushroom's turn: mRadius = range*pow(timeFraction,0.25f) (the C library's powf), re-filed with its radius
        # and its state at the tick's start (InsertBallInBoxes, Ballpark.cpp:571); or RemoveBall in Evolve
        grow = mush & ~dying
        if bool(grow.any()):
            rows = grow.nonzero().flatten()
            fr = frac[rows].tolist()
            p = torch.tensor([_C_POWF(x, 0.25) for x in fr], dtype=F64, device=self.dev)
            r = self.radius.clone()
            r[rows] = f32(self.follow_range[rows] * p)
            self.radius = r
            self._refile(grow)
        self._stop_rows(dying)                          # RemoveBall in Evolve: Stop(Ball*), moribund, non-massive
        self.moribund = self.moribund | dying
        self.massive = torch.where(dying, False, self.massive)
        self.effect_stamp = torch.where(dying, self.clock, self.effect_stamp)
        # phase 4: the re-filing of a ball that moved, the missiles' dispatch
        dd = self.pos - before
        moved = commit & ~((dd[:, 0] * dd[:, 0] + dd[:, 1] * dd[:, 1] + dd[:, 2] * dd[:, 2]) < FU_DISPATCH_MIN_MOVE_SQ)
        self._refile(moved)
        dispatch = moved & (self.mode == MISSILE)
        self.ev_n = torch.where(dispatch, self.hits_n, 0)
        self.ev_self = dispatch & self._orphans() if self.master else torch.zeros_like(dispatch)
        self._list_self_collisions()
        self.clock = self.clock + 1
        self.after_tick()
        self._after_phase4(later)
        self.watch()
        self.current_time += 1

    def _missile_thrust(self, missile: torch.Tensor, ts: torch.Tensor, pos0, vel0, vy_seen) -> torch.Tensor:
        """EvolveMissile (MOV-723): VecCombatPark.evolve_torch's, the target's velocity as seen at the turn."""
        a = torch.zeros_like(self.pos)
        straight = missile & ((self.clock - self.effect_stamp) * self.tick_ms <= FU_MISSILE_STRAIGHT_MS)
        a = torch.where(straight[:, None], self._thrust_to(self.goto, torch.ones_like(missile)), a)
        pursue = missile & ~straight
        if bool(pursue.any()):
            op = pos0[ts]
            ov = vel0[ts].clone()
            ov[:, 1] = vy_seen
            delta = self.pos - op
            dist = torch.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1] + delta[:, 2] * delta[:, 2])
            ct = dist / self.max_vel
            target = op + ov * ct[:, None]
            r = (self.follow_range + self.radius) + self.radius[ts]
            on = torch.stack((op[:, 0] + 1.0 * r, op[:, 1] + 0.0 * r, op[:, 2] + 0.0 * r), dim=1)
            inv = torch.ones_like(dist) / torch.where(dist == 0.0, 1.0, dist)
            off = target + (delta * r[:, None]) * inv[:, None]
            target = torch.where((dist == 0.0)[:, None], on, off)
            self.goto = torch.where(pursue[:, None], target, self.goto)
            a = torch.where(pursue[:, None], self._thrust_to(target, torch.zeros_like(missile)), a)
        return a

    def _formation_thrust(self, form: torch.Tensor, ts: torch.Tensor, pos0) -> torch.Tensor:
        """EvolveFormation (MOV-726): VecCombatPark.evolve_torch's."""
        a = torch.zeros_like(self.pos)
        if not bool(form.any()):
            return a
        do = self.dynamical_orientation
        fid = self.formation_id[ts]
        F = self.n_formations
        fsafe = fid.clamp(0, max(0, F - 1))
        slot = _char(self.effect_stamp)
        length_f = self.formation_lengths[fsafe]
        ok = form & (fid >= 0) & ~(fid > F - 1) & ~(slot > length_f - 1) & (slot >= 0)
        offset = self.formation_table[fsafe, slot.clamp(0, self.max_slots - 1)]
        length = torch.sqrt(offset[:, 0] * offset[:, 0] + offset[:, 1] * offset[:, 1] + offset[:, 2] * offset[:, 2])
        length = torch.where(length == 0.0, 1.0, length)
        k = (self.radius + self.radius[ts]) + length
        inv = torch.ones_like(length) / length
        vec = _cols(f32((offset * k[:, None]) * inv[:, None]))
        if do:
            q = _cols(self.rot[ts])
        else:
            q = _q_rotation_ypr(f32(self.yaw[ts]), f32(self.pitch[ts]), f32(self.roll[ts]))
        rv = _stack(transform_coord(vec, m_rotation_q(q)))
        g = pos0[ts] + rv
        self.goto = torch.where(ok[:, None], g, self.goto)
        return torch.where(ok[:, None], self._thrust_to(g, torch.ones_like(form)), a)

    # ---- the hooks in the tick on tensors (VecModesPark._phase1 and the tick above): this park's parts,
    # the statements VecComposedPark's tick held at those points less its collisions (VecComposedPark's override) ----

    def _existing(self) -> torch.Tensor:
        """The rows holding a ball (`_phase1`'s StopAllFollowers members): `exists`."""
        return self.exists

    def _phase1_combat(self, a, live, mode, t_ok, ts, pos0, vel0, vy_seen) -> torch.Tensor:
        """Phase 1's MISSILE and FORMATION thrusts (EvolveMissile, EvolveFormation), at their place in `_phase1`."""
        missile = live & (mode == MISSILE) & t_ok
        if bool(missile.any()):
            a = torch.where(missile[:, None], self._missile_thrust(missile, ts, pos0, vel0, vy_seen), a)
        form = live & (mode == FORMATION) & t_ok
        if bool(form.any()):
            a = torch.where(form[:, None], self._formation_thrust(form, ts, pos0), a)
        return a

    def _clear_contacts(self, live: torch.Tensor) -> None:
        """Phase 1's clearing of the collision lists (VecComposedPark's): none here (`_contacts_phase` sets every
        row's count)."""

    def _warp_massive(self, warping: torch.Tensor) -> None:
        """A massive ball in warp proper beside collisions (VecComposedPark's FLAG_WARP_MASSIVE): nothing here."""

    def _contacts_phase(self, live: torch.Tensor, wp, wv, dropped: torch.Tensor, rest: torch.Tensor) -> None:
        """The contacts at phase 3: each massive normal missile's Potential with its target, counted (`_contacts`)."""
        self.hits_n = self._contacts(live)

    def _refile(self, which: torch.Tensor) -> None:
        """A re-filing in the partition (VecComposedPark's): no partition here."""

    def _last_c(self):
        """mLastC at phase 3 (VecComposedPark's collision responses): zero here, `last_g + 0.0` (-0 becomes +0)."""
        return 0.0

    def _list_self_collisions(self) -> None:
        """The master's self-collision appended to a missile's collision list (VecComposedPark's): no lists here (the
        dispatch's counts, `ev_n` and `ev_self`, are the events)."""

    def _after_phase4(self, later: torch.Tensor) -> None:
        """After phase 4 and its yaw, pitch and roll (vec_combat_cuda.after's launch on tensors): the
        members in `later` (of a mushroom removed this tick, their turns before its own) released from their committed
        state; then PetrifyTroll of each troll due, TrollReady checked again, as the reference does it: SetBallFree
        (false), SetBallInteractive(false), RIGID (`petrified` marks them)."""
        self._release_members(later)
        due = self.free & ~self.moribund & self.exists & (self.mode == TROLL) & ~(self.effect_stamp > self.clock - 1)
        self.petrified = due.to(torch.uint8)
        self.set_free(due, torch.zeros_like(due))
        self.set_interactive(due, torch.zeros_like(due))
        self.mode = torch.where(due, RIGID, self.mode)

    def after_tick(self) -> None:
        """What a tick leaves to the host: without dynamical orientation, phase 4's CalculateYawPitchRoll of every
        live ball at the tick's time (the clock less one)."""
        if not self.dynamical_orientation:            # a mushroom's turn skips it
            self._ypr(self.free & ~self.moribund & self.exists & (self.mode != MUSHROOM_MODE), snap=False,
                      time=self.clock - 1)

    def events(self) -> list:
        """The tick's DoCollision events (COL-719) in Destiny's order, from the dispatch's counts: per world (its label,
        in first-appearance order), the missiles by ascending id, each with its target's id ev_n times, then itself on
        the master when orphaned: (world label, time, missile id, collider id). Call after a tick (one host sync).
        After a tick under OnTick's mask, the rows that ticked only."""
        sel = self._ticked_rows((self.ev_n > 0) | self.ev_self)
        if not bool(sel.any()):
            return []
        rows = sel.nonzero().flatten().tolist()
        world, ids = self.world.tolist(), self.ids.tolist()
        ev_n, ev_self, clock = self.ev_n.tolist(), self.ev_self.tolist(), self.clock.tolist()
        follow = self.follow_id.tolist()
        order = {w: k for k, w in enumerate(self.labels)}
        out = []
        for r in sorted(rows, key=lambda r: (order[world[r]], ids[r])):
            for _ in range(ev_n[r]):
                out.append((world[r], clock[r] - 1, ids[r], follow[r]))
            if ev_self[r]:
                out.append((world[r], clock[r] - 1, ids[r], ids[r]))
        return out

    # ---- the domain ----

    def check(self) -> None:
        """Raise if any world left the domain (the module's docstring): the flags set by the commands, and on the
        master a world with balls in more than one bubble or without an interactive ball."""
        flags = self._or_flags()
        if self.master:
            live = self.exists
            rw = self.rows_of_world
            safe = rw.clamp(min=0)
            ok = (rw >= 0) & live[safe]
            b = torch.where(ok, self.bubble[safe], self.bubble[safe[:, :1]].expand_as(safe))
            first = torch.where(ok, b, torch.iinfo(torch.int64).max).min(dim=1).values
            if bool((ok & (b != first[:, None])).any()):
                flags |= FLAG_BUBBLES
            inter = (ok & self.interactive[safe]).any(dim=1)
            if bool((ok.any(dim=1) & ~inter).any()):
                flags |= FLAG_DEAD
        if flags:
            names = [v for k, v in FLAG_NAMES.items() if flags & k]
            raise RuntimeError(f"outside the batched domain: {', '.join(names)}")

    def watch(self) -> None:
        """The domain's collision condition, at each tick's start and end: a world holding two massive balls outside
        MISSILE mode (free or fixed: a fixed one is a static sphere to the free) is flagged on all its rows, since this
        park computes only a normal missile's contact with its target (index work; one small reduction)."""
        m = self.exists & ~self.moribund & self.massive & (self.mode != MISSILE)
        rw = self.rows_of_world
        safe = rw.clamp(min=0)
        two = (m[safe] & (rw >= 0)).sum(dim=1) > 1
        self.flags = self.flags | torch.where(two[self.world_index], FLAG_MASSIVE, 0)

    def world_flags(self) -> list:
        """Each world's flags (the OR of its rows'), by world label order (`labels`)."""
        rw = self.rows_of_world
        safe = rw.clamp(min=0)
        f = torch.where(rw >= 0, self.flags[safe], 0).tolist()
        out = []
        for row in f:
            v = 0
            for x in row:
                v |= int(x)
            out.append(v)
        return out

    def _or_flags(self) -> int:
        out = 0
        for v in set(self.flags.tolist()):
            out |= int(v)
        return out

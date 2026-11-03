# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer batched: many independent worlds, each one bubble with its pilots; each
world's queued commands applied in Destiny's order at the point of the tick the server applies them, through a batched
park (exact_ballpark.vec_modes, vec_orient or vec_collide); each world's update batches; and what each pilot's client
holds, as tensors on the CPU or a GPU.

It must reproduce `exact_ballpark.server` (Destiny's net/server transcribed: Actions, Ticker, BubbleUpdater,
ParkUpdateBatcher on a master DestinyPark over FullBallpark) bit for bit, driven as Destiny's own test helper drives it
(test_ticker.py:172-182: the Ticker's tick, the park's Evolve, the batcher's send_batch), and the client view must
reproduce `exact_ballpark.client.Session`'s client parks. Checked by tests/test_movement_vec_server.py on generated command
streams, on the CPU and the GPU. Rules, by rule id:

- TIME-709, TIME-710: `push` is an Actions call for every masked world at once: the action is queued with the world's
  stamp_for_system (Actions' first stamp 0, then the clock + 1 set at each pre-tick, _actions.py:31-71,
  _baseticker.py:42-46, _ticker.py:64-66), and dropped when it equals the world's last queued action, same ball and
  event, compared as Python compares the queued tuples: floats by value (0.0 equals -0.0; a NaN equals nothing; Python's
  identity shortcut, which makes one NaN *object* queued twice equal, has no counterpart on tensors), ints exactly.
- TIME-712, the pre-tick (_baseticker.py:23-32, _ticker.py:56-122): the follower clean-up (none can arise in a world of
  one bubble: checked, not run); the stamp moves; the queue is flushed; every queued action but CloakBall is applied
  through the park in queue order (one push a slot: the worlds' j-th calls together, each world's in its own order);
  the bubble's additions (BubbleUpdater, AdditionsAndDeletions: a user ball seen for the first time gets every uncloaked
  ball of its bubble); then the history in queue order: CloakBall applied there (after the bubble update) and its event
  sent to the bubble it was queued in, UncloakBall's event to the ball's own character and an AddBallsToPark of it in
  front of every other interactive ball's characters, a local SetBallPosition sent nowhere, every other event to the
  bubble, or to the ball's character while it is cloaked, or nowhere for a ball not in the park or a dead bubble.
- SEN-712 and the send (_parkupdatebatcher.py:52-119): per world, the bubble's narrowcast to the pilots of its
  interactive balls (count TWO for those with a character update this tick), each pilot's singlecast (wait_for_bubble
  and TWO when it also gets the bubble's): 0 to 2 updates a pilot a tick, the singlecast first (`post_tick`'s flags;
  `decode` builds Destiny's tuples on the host).
- MOV-730's cloak trigger, SEN-711: CloakBall (on the master a normal cloak adds a 2 s proximity sensor, the followers
  are released, the ball cloaked and non-massive) and UncloakBall (the sensor removed for a normal cloak, massive again
  unless in warp proper), as FullBallpark.cloak_ball and uncloak_ball.
- The pending queries and the undo (_actions.py:686-766): has_pending_cloak, has_pending_uncloak,
  get_pending_cloak_mode, get_pending_interactive_state, get_pending_speed_fraction read the world's queue as Destiny's
  Actions holds it between ticks (the last such call's argument, else the ball's current value, else None);
  undo_pending_cloak prunes the ball's queued CloakBalls, and the calls pushed after it are tested against the queue's
  last action as it then stands. On a kernel path, which tests the calls at the pre-tick, the queries test them here
  by the same rule, and an undo hands the kernel the tested calls (bit 1 of the kernels' mask byte, `mask_bytes`), so
  the prune's edge (X, CloakBall, X, undone: both X queued, side by side) comes out as Destiny's.

**The worlds.** A world is one bubble (id 0, as the master files the first ball of a park; bubbles are static: the
batched parks have no partition, so no ball changes bubble and no follower is ever remote). Its balls are the park's
rows of that world, in the order they were added (the bubble's dictionary order). A pilot flies one ship: character and
client `pilot` (an index within the world; -1 for a ball without a pilot, whose character is None), interested in that
ship only (exact_ballpark.client.PilotInterests). Every world's bubble must stay live (an interactive ball in it): on the
master a dead bubble's balls are not evolved (TIME-714), which the batched parks do not model; `check()` refuses it.

**Three parks** (`ModesAdapter`, `OrientAdapter`, `CollideAdapter`), each applying the command kinds it has:
VecModesPark every flight mode, the setters and turning (its tick a CUDA kernel: on the CPU the pre-tick alone);
VecOrientPark GOTO, STOP, the setters and turning (CPU or GPU); VecCollidePark GOTO, STOP and the speed fraction with
Destiny's collisions (CPU or GPU), its cloaks without the normal cloak's sensor (a sensor re-files its ball, which the
park's filing does not model). The flags the layer routes by (interactive, cloaked, massive, the sensor) are the park's
own where it has them.

**Exactness.** Every arithmetic step is the park's own command or setter (held to the scalar reference by its own
tests); this module adds only index work (masks, gathers, comparisons, integer steps), no float arithmetic but the
cloak's float sum radius + range (float32, as the C++). A tick is capturable as a CUDA graph (`ServerGraph`) when its
pushes are a fixed schedule and the park's tick never waits for the host: VecModesPark's kernel; VecOrientPark with
dynamical orientation on its kernel (without it, its yaw, pitch and roll are computed on the host each tick; with it on
the torch path, its torque search asks the host); VecCollidePark with contact_slots or on its kernel.

**The client's view** (`ClientView`): a pilot's client applies each update at its stamp to its own park, which settles
`down` ticks behind the server's when updates take `down` ticks to arrive (exact_ballpark.client.Session); with the same
Destiny code and the same events its park then holds the server's state at time t - down. ClientView keeps the last
`down` + 1 server states of the fields a learner observes and gives that state, with another pilot's cloaked ship absent.

- SetBallTroll and SetBallFree (_actions.py:133-148, 525-538): two kinds appended to KINDS (25, 26),
  the int argument in `n`, applied by the parks with the commands (`set_ball_troll`, `set_free`: ModesAdapter
  and CombatAdapter) and by the server kernels; a SetBallTroll makes its ball interactive, so a tick that queues one
  records the streams; PetrifyTroll is the park's tick's. has_pending_set_free and has_pending_set_not_free read the
  queue as the other pending queries do. The streams write a troll's stamp and leave out a rigid ball's mass block, as
  WriteBallToStream does. The StopAllFollowers of CloakBall and EntityWarpIn take Destiny's members on every path
  (follow_ptr and the follow id: a ball the Python SetBoid took out of a follow mode between ticks too).

- Global balls (_actions.py:328-362): make_ball_global and make_ball_local are two kinds appended to
  KINDS (27 AddBallsToPark, 28 BallNotGlobal, by the events they queue), applied by the partition layer's server only
  (exact_ballpark.vec_server_space.SpaceServer over a VecCollidePark made with global balls): no adapter here takes them.
  Their routes are EVERY (to every bubble with an interactive ball; BallNotGlobal's RemoveBall to every such bubble but
  the one it names) and the streams write the ball's global flag (DSTBALL_ISGLOBAL) where the fields hold one.

- OnTick's intervals (Ballpark.cpp:219-330, TIME-702 to TIME-705; the
  SpaceServer.on_tick for this layer): `on_tick(sim_time)` drives every world by its own simulation clock as
  FullBallpark.on_tick does with the Ticker's tick at DoPreTick and send_batch at DoPostTick: BringOutTheDeads first
  (the park's: VecCombatPark's; a world that loses a ball is flagged in `removed`, which `check()` refuses, as this
  layer sends no RemoveBalls), too early: nothing; the master's backlog of more than five intervals cut to one; a
  world's first call evolves once without a pre-tick (its queue kept); the other worlds' whole intervals, the
  remainder carried. The worlds tick together under a per-world mask (`pre_tick`, `evolve` and `post_tick` take
  `mask`; a round of every world runs the unmasked code): a world without a tick keeps its queue (each queued call
  split by its worlds, in order, with its last action for the duplicate test), its stamp, its bubble update's lists
  and time, its balls and clock (the parks' `evolve(active)`), and sends nothing (`Batch.ticked`). On
  the kernel paths (vec_server_cuda, vec_server_combat_cuda) the kernels are unchanged: the queue is split on the
  host, and the server state a kernel writes for a world with no pushed call (its stamp, its bubble update; every
  park write is a queued call's) is put back, its flags cleared.

Not here: bubbles other than one per world and their transitions, global balls (but SpaceServer's), removals and
additions after the load (a removal by OnTick's BringOutTheDeads is flagged), missiles, harmonics and force fields,
formations, the clean-up's remote followers (none can arise), the signals and logging, the game's SetState
(`send_full_state_update`), the master's sensor events, OnTick's mask in a CUDA graph (ServerGraph: the queue split
and a world's first pre-tick are host decisions).
"""

from __future__ import annotations

import copy
import struct
from dataclasses import dataclass, field

import torch

from . import vec_libm as L
from .modes import WARP
from .ref import STOP

F64, I64, BOOL = torch.float64, torch.int64, torch.bool

# the queued actions, by the event name the server sends (index = the kind's code; 0 is no action)
KINDS = ("", "GotoPoint", "GotoDirection", "Stop", "FollowBall", "Orbit", "WarpTo", "EntityWarpIn", "SetSpeedFraction",
         "SetMaxSpeed", "SetBallMass", "SetBallAgility", "SetBallRadius", "SetBallVelocity", "SetBallMassive",
         "SetBallInteractive", "SetBallPosition", "CloakBall", "UncloakBall", "SetMaxAngularSpeed",
         "SetBallAngularVelocity", "SetBallRotation", "SetBallAngularAgility", "SetMaxAngularVelocity",
         "LaunchMissile", "SetBallTroll", "SetBallFree",      # the last two appended (25, 26)
         "AddBallsToPark", "BallNotGlobal")                   # (27, 28): make_ball_global, _local
CODE = {name: k for k, name in enumerate(KINDS) if name}
# Actions' methods (_actions.py:81-675) and the kinds they queue
ACTION_KINDS = {"go_to_point": "GotoPoint", "go_to_direction": "GotoDirection", "stop": "Stop",
                "follow_ball": "FollowBall", "orbit": "Orbit", "warp_to": "WarpTo", "entity_warp_in": "EntityWarpIn",
                "set_speed_fraction": "SetSpeedFraction", "set_max_speed": "SetMaxSpeed",
                "set_ball_mass": "SetBallMass", "set_ball_agility": "SetBallAgility",
                "set_ball_radius": "SetBallRadius", "set_ball_velocity": "SetBallVelocity",
                "set_ball_massive": "SetBallMassive", "set_ball_interactive": "SetBallInteractive",
                "set_ball_position": "SetBallPosition", "cloak_ball": "CloakBall", "uncloak_ball": "UncloakBall",
                "set_max_angular_speed": "SetMaxAngularSpeed", "set_ball_angular_velocity": "SetBallAngularVelocity",
                "set_ball_rotation": "SetBallRotation", "set_ball_angular_agility": "SetBallAngularAgility",
                "set_max_angular_velocity": "SetMaxAngularVelocity", "launch_missile": "LaunchMissile",
                "set_ball_troll": "SetBallTroll", "set_ball_free": "SetBallFree",
                "make_ball_global": "AddBallsToPark", "make_ball_local": "BallNotGlobal"}
# the kinds Actions refuses to queue without dynamical orientation (_actions.py:169-309, 584-598)
DYNAMICAL_ONLY = frozenset(CODE[k] for k in ("SetBallAngularVelocity", "SetMaxAngularVelocity", "SetBallRotation",
                                             "SetBallAngularAgility", "SetMaxAngularSpeed"))
# the event's arguments after the ball id: (from f, the count) or the int arguments, per kind (decode)
_XYZ = ("GotoPoint", "GotoDirection", "SetBallVelocity", "SetBallAngularVelocity", "SetMaxAngularVelocity",
        "SetBallPosition")
_ONE = ("SetSpeedFraction", "SetMaxSpeed", "SetBallMass", "SetBallAgility", "SetBallRadius", "SetMaxAngularSpeed",
        "SetBallAngularAgility")
NONE, BUBBLE, CHARACTER = 0, 1, 2                   # where an event goes (the routes)
EVERY = 3                                           # every bubble with an interactive ball (_ticker.py:
                                                    # 173-196), less the route's bubble where one is named
DSTNORMALCLOAK = 1                                  # IDstConstants.h:77
CLOAK, UNCLOAK, SETPOS, INTERACTIVE = CODE["CloakBall"], CODE["UncloakBall"], CODE["SetBallPosition"], \
    CODE["SetBallInteractive"]
FOLLOWS = (CODE["FollowBall"], CODE["Orbit"])
# LaunchMissile (_actions.py:653-675): the missile's id as the ball, the target's as `target`, the
# owner's as `n`, the thunk's chars aimed and massive as f[0] and f[1]; applied by a park with missiles only
# (exact_ballpark.vec_server_combat's CombatAdapter over VecCombatPark)
LAUNCH = CODE["LaunchMissile"]
# SetBallTroll and SetBallFree (_actions.py:133-148, 525-538): the ball's id, the int argument (the
# delay in ticks, the flag) as `n`, applied through the park's set_ball_troll and set_free (PySetBallTroll and
# PySetBallFree, "Li": a C int each), by the parks with the commands (VecModesPark, VecCombatPark)
TROLL, FREE = CODE["SetBallTroll"], CODE["SetBallFree"]
# make_ball_global and make_ball_local (_actions.py:328-362): SetBallGlobal applied at the call, then
# AddBallsToPark (the ball's stream as written at the call, its global flag set) or BallNotGlobal (the ball's bubble at
# the call, in `loc`) queued; the partition layer's server (SpaceServer) alone takes them
GLOBAL_ON, GLOBAL_OFF = CODE["AddBallsToPark"], CODE["BallNotGlobal"]
GLOBAL_KINDS = frozenset((GLOBAL_ON, GLOBAL_OFF))


def mask_bytes(slots: list, W: int, d) -> torch.Tensor:
    """The server kernels' mask byte [J, W] (uint8) of a tick's pushes: bit 0, the call pushed for the world; bit 1,
    the call already tested by an undo (`resolved`), whose bit 0 the kernel then takes as its queued flag without the
    duplicate test. Without an undo the bytes are the masks' 0 and 1, as the kernels read them before."""
    if not slots:
        return torch.zeros(0, W, dtype=torch.uint8, device=d)
    m = torch.stack([s.mask.to(torch.uint8) for s in slots])
    res = [j for j, s in enumerate(slots) if getattr(s, "resolved", False)]
    if res:
        m[res] = m[res] | 2
    return m.contiguous()


def _with_flag(s, flag: str, value: torch.Tensor):
    """A copy of a queued call (a Slot) with its flag `flag` [W] replaced: `ok` on the torch path, a kernel path's
    pushed `mask` (every other attribute kept, an undo's `resolved` among them, after the
    vec_server_space._with_ok)."""
    t = copy.copy(s)
    setattr(t, flag, value)
    return t


def _char(n: torch.Tensor) -> torch.Tensor:
    """A conversion to (signed) char (full.py's _char): 128 to 255 become negative."""
    return ((n + 128) & 0xFF) - 128


# =============================================================================================== the parks

class _Adapter:
    """What the server layer needs of a batched park. Subclasses set: park, R (rows), dev, world [R] (0 .. W-1), ids
    [R] (Destiny's ball ids), exists [R], rows_of_world [W, N] (-1 padded, in the order the balls were added),
    ids_of_world [W, N] (0 padded), the flags' storage, `kinds` (the codes it applies) and `cloak_modes`."""

    master = True

    def _index_worlds(self, labels: list) -> None:
        """Group the rows by their world label (first appearance order), rows in their own order."""
        order, groups = [], {}
        for r, w in enumerate(labels):
            if w not in groups:
                groups[w] = []
                order.append(w)
            groups[w].append(r)
        self.labels = order
        self.W = len(order)
        n = max((len(g) for g in groups.values()), default=0)
        rows = [[*groups[w], *([-1] * (n - len(groups[w])))] for w in order]
        self.rows_of_world = torch.tensor(rows, dtype=I64, device=self.dev).reshape(self.W, n)
        index = {w: k for k, w in enumerate(order)}
        self.world = torch.tensor([index[w] for w in labels], dtype=I64, device=self.dev)
        safe = self.rows_of_world.clamp(min=0)
        self.ids_of_world = torch.where(self.rows_of_world >= 0, self.ids[safe], 0)
        self.arange = torch.arange(self.R, device=self.dev)

    def lookup(self, ball: torch.Tensor) -> torch.Tensor:
        """Each world's row of the ball with Destiny id ball[w], or -1 (index work only)."""
        hit = self.ids_of_world == ball.to(I64)[:, None]
        idx = hit.to(I64).argmax(dim=1)
        return torch.where(hit.any(dim=1), self.rows_of_world.gather(1, idx[:, None])[:, 0], -1)

    def any_in_world(self, flag: torch.Tensor) -> torch.Tensor:
        """[W]: whether any row of each world has `flag`."""
        safe = self.rows_of_world.clamp(min=0)
        return (flag[safe] & (self.rows_of_world >= 0)).any(dim=1)

    # the flags: subclasses map them onto the park's tensors where it has them
    def cloak(self, m: torch.Tensor, mode: torch.Tensor, rng: torch.Tensor) -> None:
        """CloakBall (FullBallpark.cloak_ball; Ballpark.cpp:5223): refused unless the (signed char) mode is positive; on
        the master a normal cloak adds a proximity sensor (unless float(radius) + float(range) < 0, AddProximitySensor's
        guard); the followers released; cloaked with the mode's byte, non-massive."""
        mode = _char(mode.to(I64))
        ok = m & self.exists & (mode > 0)
        if self.master:
            rng32 = L.f32(rng.to(F64))
            sense = ok & (mode == DSTNORMALCLOAK) & ~(L.f32(self.radius() + rng32) < 0.0)
            self.set_sensor(sense)
        self.release_followers(ok)
        self.cloaked = torch.where(ok, mode & 0xFF, self.cloaked)
        self.massive = torch.where(ok, False, self.massive)

    def uncloak(self, m: torch.Tensor) -> None:
        """UncloakBall (FullBallpark.uncloak_ball; Ballpark.cpp:5257): a normal cloak's sensor removed (master);
        uncloaked; massive again unless in warp proper."""
        ok = m & self.exists
        if self.master:
            self.clear_sensor(ok & (self.cloaked == DSTNORMALCLOAK))
        self.cloaked = torch.where(ok, 0, self.cloaked)
        self.massive = torch.where(ok & ~self.warping(), True, self.massive)

    def set_sensor(self, m: torch.Tensor) -> None:
        pass

    def clear_sensor(self, m: torch.Tensor) -> None:
        pass

    def release_followers(self, m: torch.Tensor) -> None:
        pass

    def warping(self) -> torch.Tensor:
        return torch.zeros(self.R, dtype=BOOL, device=self.dev)


class ModesAdapter(_Adapter):
    """exact_ballpark.vec_modes.VecModesPark: its rows, worlds (`world`), ids and flags; every kind; its tick the park's
    `evolve` (the CUDA kernel on a CUDA device, the tick on tensors elsewhere)."""

    def __init__(self, park, master: bool = True):
        self.park, self.R, self.dev, self.master = park, park.n, park.dev, bool(master)
        self.ids = park.ids
        self.exists = torch.ones(self.R, dtype=BOOL, device=self.dev)
        self._index_worlds(park.world.tolist())
        do = park.dynamical_orientation
        self.kinds = frozenset(k for k in range(1, len(KINDS)) if (do or k not in DYNAMICAL_ONLY) and k != LAUNCH
                               and k not in GLOBAL_KINDS)          # (SpaceServer's alone)
        self.cloak_modes = frozenset(range(256))

    interactive = property(lambda s: s.park.interactive, lambda s, v: setattr(s.park, "interactive", v))
    cloaked = property(lambda s: s.park.cloaked, lambda s, v: setattr(s.park, "cloaked", v))
    massive = property(lambda s: s.park.massive, lambda s, v: setattr(s.park, "massive", v))

    def radius(self) -> torch.Tensor:
        return self.park.radius

    def clock(self) -> torch.Tensor:
        """[W]: each world's mCurrentTime (its rows share one clock)."""
        return self.park.clock[self.rows_of_world[:, 0]]

    def set_sensor(self, m: torch.Tensor) -> None:
        self.park.sensor = self.park.sensor | m

    def clear_sensor(self, m: torch.Tensor) -> None:
        self.park.sensor = self.park.sensor & ~m

    def release_followers(self, m: torch.Tensor) -> None:
        self.park._release_followers(m)

    def warping(self) -> torch.Tensor:
        return (self.park.mode == WARP) & (self.park.effect_stamp >= 0)

    def apply(self, kind: int, m: torch.Tensor, trow: torch.Tensor, f: torch.Tensor, n: torch.Tensor) -> None:
        """One command kind on the rows in m (the args per row): the park's own command or setter."""
        p, name = self.park, KINDS[kind]
        xyz = f[:, :3].contiguous()
        if name == "GotoPoint":
            p.goto_point(m, xyz)
        elif name == "GotoDirection":
            p.goto_direction(m, xyz)
        elif name == "Stop":
            p.stop(m)
        elif name == "FollowBall":
            p.follow(m, trow, f[:, 0].contiguous())
        elif name == "Orbit":
            p.orbit(m, trow, f[:, 0].contiguous())
        elif name == "WarpTo":
            p.warp_to(m, xyz, f[:, 3].contiguous(), n)
        elif name == "EntityWarpIn":
            p.entity_warp_in(m, xyz, n)
        elif name == "SetSpeedFraction":
            p.set_speed_fraction(m, f[:, 0].contiguous())
        elif name == "SetMaxSpeed":
            p.set_max_velocity(m, f[:, 0].contiguous())
        elif name == "SetBallMass":
            p.set_mass(m, f[:, 0].contiguous())
        elif name == "SetBallAgility":
            p.set_agility(m, f[:, 0].contiguous())
        elif name == "SetBallRadius":
            p.set_radius(m, f[:, 0].contiguous())
        elif name == "SetBallVelocity":
            p.set_velocity(m, xyz)
        elif name == "SetBallMassive":
            p.massive = torch.where(m, n != 0, p.massive)
        elif name == "SetBallInteractive":
            p.interactive = torch.where(m, n != 0, p.interactive)
        elif name == "SetBallPosition":                 # SetBallPosition: the new and old position (then re-filed)
            p.pos = torch.where(m[:, None], xyz, p.pos)
            p.old_pos = torch.where(m[:, None], xyz, p.old_pos)
        elif name == "UncloakBall":
            self.uncloak(m)
        elif name == "SetMaxAngularSpeed":
            p.set_max_angular_speed(m, f[:, 0].contiguous())
        elif name == "SetBallAngularVelocity":
            p.set_angular_velocity(m, xyz)
        elif name == "SetBallRotation":
            p.set_rotation(m, f)
        elif name == "SetBallAngularAgility":
            p.set_rot_agility(m, f[:, 0].contiguous())
        elif name == "SetMaxAngularVelocity":
            pass                                        # no such park method: never applied (_ticker.py:115-116)
        elif name == "SetBallTroll":                    # PySetBallTroll: the park's set_ball_troll
            p.set_ball_troll(m, n)
        elif name == "SetBallFree":                     # PySetBallFree: the park's set_free (flag != 0: freed)
            p.set_free(m, n)
        else:
            raise ValueError(f"{name} is not applied by {type(self).__name__}")

    def evolve(self, worlds: torch.Tensor | None = None) -> None:
        """The park's tick; `worlds` [W] (OnTick's mask): those worlds' rows alone, every other row kept
        bit for bit, its clock included (VecModesPark.evolve(active))."""
        if worlds is None:
            self.park.evolve()
            return
        self.park.evolve(active=worlds[self.world])

    def stream_fields(self) -> dict:
        """The fields WriteBallToStream writes (Thunkers.cpp:3180-3377), every row, on the host."""
        p = self.park
        out = {k: getattr(p, k).detach().cpu() for k in ("ids", "mode", "radius", "pos", "free", "massive",
                                                          "interactive", "mass", "cloaked", "max_vel", "vel",
                                                          "agility", "speed_fraction", "goto", "follow_id",
                                                          "follow_range", "effect_stamp", "last_collision",
                                                          "owner_id", "rot", "max_ang_vel", "ang_vel",
                                                          "rot_agility")}
        out["dynamical"] = bool(p.dynamical_orientation)
        return out


class OrientAdapter(_Adapter):
    """exact_ballpark.vec_orient.VecOrientPark (B balls, GOTO and STOP, the setters and turning): the worlds, ids and the
    flags it lacks (interactive, cloaked, massive) are given here and kept by the adapter."""

    def __init__(self, park, world: list | None = None, ids: list | None = None, interactive: list | None = None,
                 massive: list | None = None, master: bool = True):
        self.park, self.R, self.dev, self.master = park, park.n, park.dev, bool(master)
        R, d = self.R, self.dev
        self.ids = torch.tensor(list(range(1, R + 1)) if ids is None else [int(i) for i in ids], dtype=I64, device=d)
        self.exists = torch.ones(R, dtype=BOOL, device=d)
        self._index_worlds(list(range(R)) if world is None else list(world))
        self.interactive = torch.tensor([True] * R if interactive is None else [bool(x) for x in interactive],
                                        dtype=BOOL, device=d)
        self.massive = torch.tensor([False] * R if massive is None else [bool(x) for x in massive], dtype=BOOL,
                                    device=d)
        self.cloaked = torch.zeros(R, dtype=I64, device=d)
        self.sensor = torch.zeros(R, dtype=BOOL, device=d)
        self.time = torch.full((self.W,), int(park.current_time), dtype=I64, device=d)
        do = park.dynamical_orientation
        names = ("GotoPoint", "GotoDirection", "Stop", "SetSpeedFraction", "SetMaxSpeed", "SetBallMass",
                 "SetBallAgility", "SetBallRadius", "SetBallVelocity", "SetBallMassive", "SetBallInteractive",
                 "SetBallPosition", "CloakBall", "UncloakBall") + (("SetMaxAngularSpeed", "SetBallAngularVelocity",
                                                                     "SetBallRotation", "SetBallAngularAgility",
                                                                     "SetMaxAngularVelocity") if do else ())
        self.kinds = frozenset(CODE[k] for k in names)
        self.cloak_modes = frozenset(range(256))

    def radius(self) -> torch.Tensor:
        return self.park.radius

    def clock(self) -> torch.Tensor:
        """[W]: the park's mCurrentTime (one clock; a device tensor, so a CUDA graph reads it live)."""
        return self.time

    def set_sensor(self, m: torch.Tensor) -> None:
        self.sensor = self.sensor | m

    def clear_sensor(self, m: torch.Tensor) -> None:
        self.sensor = self.sensor & ~m

    def apply(self, kind: int, m: torch.Tensor, trow: torch.Tensor, f: torch.Tensor, n: torch.Tensor) -> None:
        p, name = self.park, KINDS[kind]
        xyz = f[:, :3].contiguous()
        if name == "GotoPoint":
            p.goto_point(m, xyz)
        elif name == "GotoDirection":
            p.goto_direction(m, xyz)
        elif name == "Stop":
            p.stop(m)
        elif name == "SetSpeedFraction":
            p.set_speed_fraction(m, f[:, 0].contiguous())
        elif name == "SetMaxSpeed":
            p.set_max_velocity(m, f[:, 0].contiguous())
        elif name == "SetBallMass":
            p.set_mass(m, f[:, 0].contiguous())
        elif name == "SetBallAgility":
            p.set_agility(m, f[:, 0].contiguous())
        elif name == "SetBallRadius":
            p.set_radius(m, f[:, 0].contiguous())
        elif name == "SetBallVelocity":
            p.set_velocity(m, xyz)
        elif name == "SetBallMassive":
            self.massive = torch.where(m, n != 0, self.massive)
        elif name == "SetBallInteractive":
            self.interactive = torch.where(m, n != 0, self.interactive)
        elif name == "SetBallPosition":
            p.pos = torch.where(m[:, None], xyz, p.pos)
            p.old_pos = torch.where(m[:, None], xyz, p.old_pos)
        elif name == "UncloakBall":
            self.uncloak(m)
        elif name == "SetMaxAngularSpeed":
            p.set_max_angular_speed(m, f[:, 0].contiguous())
        elif name == "SetBallAngularVelocity":
            p.set_angular_velocity(m, xyz)
        elif name == "SetBallRotation":
            p.set_rotation(m, f)
        elif name == "SetBallAngularAgility":
            p.set_rot_agility(m, f[:, 0].contiguous())
        elif name == "SetMaxAngularVelocity":
            pass
        else:
            raise ValueError(f"{name} is not applied by {type(self).__name__}")

    def evolve(self, worlds: torch.Tensor | None = None) -> None:
        """The park's tick and each world's clock; `worlds` [W] (OnTick's mask): those worlds' rows
        alone, every other row kept bit for bit (VecOrientPark.evolve(active)), their clocks kept."""
        if worlds is None:
            self.park.evolve()
            self.time.add_(1)
            return
        self.park.evolve(active=worlds[self.world])
        self.time.add_(worlds.to(I64))

    def stream_fields(self) -> dict:
        p, R = self.park, self.R
        z = torch.zeros(R, dtype=I64)
        out = {k: getattr(p, k).detach().cpu() for k in ("mode", "radius", "pos", "mass", "max_vel", "vel", "agility",
                                                          "speed_fraction", "goto", "rot", "max_ang_vel", "ang_vel",
                                                          "rot_agility")}
        out.update(ids=self.ids.cpu(), free=torch.ones(R, dtype=BOOL), massive=self.massive.cpu(),
                   interactive=self.interactive.cpu(), cloaked=self.cloaked.cpu(), follow_id=z, owner_id=z,
                   effect_stamp=z, follow_range=torch.zeros(R, dtype=F64), last_collision=torch.zeros(R, dtype=F64),
                   dynamical=bool(p.dynamical_orientation))
        return out


class CollideAdapter(_Adapter):
    """exact_ballpark.vec_collide.VecCollidePark (W worlds of N slots, GOTO and STOP with Destiny's collisions): row
    w x N + slot; its own exists, ids, massive and cloaked (a flag there: the mode is kept here); cloak modes other than
    the normal one (whose sensor re-files its ball on the master, which the park does not model)."""

    def __init__(self, park, interactive: list | None = None, master: bool = True):
        self.park, self.dev, self.master = park, park.dev, bool(master)
        W, N = park.W, park.N
        self.R = W * N
        self.ids = park.id.reshape(-1)
        self.exists = park.exists.reshape(-1)
        labels = [r // N for r in range(self.R)]
        self._index_worlds(labels)
        ex = park.exists.reshape(-1)
        self.rows_of_world = torch.where(ex.reshape(W, N), self.rows_of_world, -1)
        safe = self.rows_of_world.clamp(min=0)
        self.ids_of_world = torch.where(self.rows_of_world >= 0, self.ids[safe], 0)
        self.interactive = torch.tensor([True] * self.R if interactive is None else [bool(x) for x in interactive],
                                        dtype=BOOL, device=self.dev) & ex
        self.cloaked = torch.where(park.cloaked.reshape(-1), 1, 0).to(I64)
        self.time = torch.full((self.W,), int(park.current_time), dtype=I64, device=self.dev)
        self.kinds = frozenset(CODE[k] for k in ("GotoPoint", "GotoDirection", "Stop", "SetSpeedFraction",
                                                  "SetBallMassive", "SetBallInteractive", "CloakBall", "UncloakBall"))
        self.cloak_modes = frozenset(m for m in range(256) if ((m + 128) & 0xFF) - 128 != DSTNORMALCLOAK)

    @property
    def massive(self):
        return self.park.massive.reshape(-1)

    @massive.setter
    def massive(self, v):
        self.park.massive = v.reshape(self.park.W, self.park.N)

    @property
    def cloaked(self):
        return self._cloaked

    @cloaked.setter
    def cloaked(self, v):
        self._cloaked = v
        self.park.cloaked = (v != 0).reshape(self.park.W, self.park.N)

    def radius(self) -> torch.Tensor:
        return self.park.radius.reshape(-1)

    def clock(self) -> torch.Tensor:
        return self.time

    def apply(self, kind: int, m: torch.Tensor, trow: torch.Tensor, f: torch.Tensor, n: torch.Tensor) -> None:
        p, name, W, N = self.park, KINDS[kind], self.park.W, self.park.N
        mm = m.reshape(W, N)
        xyz = f[:, :3].reshape(W, N, 3)
        if name == "GotoPoint":
            p.goto_point(mm, xyz)
        elif name == "GotoDirection":
            p.goto_direction(mm, xyz)
        elif name == "Stop":
            p.stop(mm)
        elif name == "SetSpeedFraction":
            p.set_speed_fraction(mm, f[:, 0].reshape(W, N))
        elif name == "SetBallMassive":
            self.massive = torch.where(m, n != 0, self.massive)
        elif name == "SetBallInteractive":
            self.interactive = torch.where(m & self.exists, n != 0, self.interactive)
        elif name == "UncloakBall":
            self.uncloak(m)
        else:
            raise ValueError(f"{name} is not applied by {type(self).__name__}")

    def evolve(self, worlds: torch.Tensor | None = None) -> None:
        """The park's tick and each world's clock; `worlds` [W] (OnTick's mask): those worlds alone, the
        others kept bit for bit (VecCollidePark's `active`), their clocks kept."""
        if worlds is None:
            self.park.evolve()
            self.time.add_(1)
            return
        self.park.evolve(active=worlds)
        self.time.add_(worlds.to(I64))

    def stream_fields(self) -> dict:
        p, R = self.park, self.R
        flat = lambda t: t.detach().reshape(R, *t.shape[2:]).cpu()       # noqa: E731
        z = torch.zeros(R, dtype=I64)
        out = {k: flat(getattr(p, k)) for k in ("mode", "radius", "pos", "free", "mass", "max_vel", "vel", "agility",
                                                 "speed_fraction", "goto")}
        out.update(ids=self.ids.cpu(), massive=self.massive.cpu(), interactive=self.interactive.cpu(),
                   cloaked=self.cloaked.cpu(), follow_id=z, owner_id=z, effect_stamp=z,
                   follow_range=torch.zeros(R, dtype=F64), last_collision=flat(p.last_collision), dynamical=False)
        return out


# ===================================================================================== the queue and the tick

@dataclass
class Slot:
    """One push: one Actions call for every masked world (its kind, and per world the data)."""
    kind: int
    ok: torch.Tensor                    # [W] queued (masked and not a duplicate of the world's last action)
    ball: torch.Tensor                  # [W] Destiny id
    row: torch.Tensor                   # [W] the ball's row, or -1
    target: torch.Tensor                # [W] the target's Destiny id (FollowBall, Orbit)
    trow: torch.Tensor                  # [W] the target's row, or -1
    f: torch.Tensor                     # [W, 4] the float arguments
    n: torch.Tensor                     # [W] the int argument
    loc: torch.Tensor                   # [W] the local argument (SetBallPosition: is local; CloakBall: its bubble)


@dataclass
class Batch:
    """One tick's update batches, every world: the stamps, the queued actions (`slots`), where each went (`route`), the
    uncloaks' AddBallsToPark notices, the bubble additions, and the post-tick's delivery flags per row (the pilot of
    that row's ship). `decode` turns it into Destiny's tuples on the host."""
    stamp: torch.Tensor                 # [W] the actions' stamp
    clock: torch.Tensor                 # [W] the park's time at the pre-tick (the packaged additions' stamp)
    slots: list
    route: torch.Tensor                 # [W, J] NONE, BUBBLE or CHARACTER
    notices: list = field(default_factory=list)     # (slot j, recipients [R] bool, the stream's fields)
    additions: tuple | None = None      # (fresh [R], listed [R], the stream's fields)
    gets_bubble: torch.Tensor | None = None     # [R] the row's pilot gets the bubble's narrowcast
    has_character: torch.Tensor | None = None   # [R] the row's pilot has a character update
    ticked: torch.Tensor | None = None  # [W] the worlds this tick ran for (OnTick's mask; None: every
                                        # world, a tick outside on_tick)


class VecServer:
    """The server layer over a batched park (an adapter): `push` queues, `tick` is the pre-tick, the park's tick and
    the post-tick (Destiny's test helper's order), `batch` the last tick's update batches. `kernel="cuda"` (a
    VecModesPark on a CUDA device) runs the pre-tick and the send flags as one CUDA kernel, one thread a world
    (exact_ballpark.vec_server_cuda); "torch", the default, is the exactness reference."""

    def __init__(self, adapter: _Adapter, pilots: list | torch.Tensor | None = None, kernel: str = "torch"):
        if kernel not in ("torch", "cuda"):
            raise ValueError(f"kernel: 'torch' or 'cuda', not {kernel!r}")
        if kernel == "cuda" and not (isinstance(adapter, ModesAdapter) and adapter.dev.type == "cuda"):
            raise ValueError('kernel="cuda" runs the server layer of a VecModesPark on a CUDA device')
        if kernel == "cuda" and LAUNCH in adapter.kinds:      # this kernel has no missile or formation branch
            raise ValueError('kernel="cuda" here is VecModesPark\'s kernel: a combat park\'s server layer runs on '
                             'vec_server_combat.CombatServer(..., kernel="cuda")')
        self.kernel = kernel
        self.a = A = adapter
        self.W, self.R, self.dev = A.W, A.R, A.dev
        self.pilot = (torch.full((self.R,), -1, dtype=I64, device=self.dev) if pilots is None
                      else torch.as_tensor(pilots, dtype=I64).to(self.dev))
        clock = A.clock()
        # Actions' first stamp: 0 at time 0 (_actions.py:31-34); Destiny reads park._currentTime otherwise, which the
        # C++ park has not (an AttributeError): here the next tick's, as the Ticker's own first stamp
        self.stamp_for_system = torch.where(clock == 0, 0, clock + 1)
        self.settled = torch.zeros(self.R, dtype=BOOL, device=self.dev)     # the ball's old bubble is its bubble
        self.dead = torch.zeros(self.W, dtype=BOOL, device=self.dev)        # a dead bubble at an evolve (refused)
        self._bubble_time = torch.full((self.W,), -1, dtype=I64, device=self.dev)   # BubbleUpdater's _tick
        self._fresh = torch.zeros(self.R, dtype=BOOL, device=self.dev)
        self._listed = torch.zeros(self.R, dtype=BOOL, device=self.dev)
        self.slots: list = []
        self._reset_last()
        self.batch: Batch | None = None
        if kernel == "cuda":
            from . import vec_server_cuda
            self._cuda = vec_server_cuda
            with torch.cuda.device(self.dev):
                self._lib = vec_server_cuda.load_library()

    def _reset_last(self) -> None:
        W, d = self.W, self.dev
        self.last = {"kind": torch.zeros(W, dtype=I64, device=d), "ball": torch.zeros(W, dtype=I64, device=d),
                     "target": torch.zeros(W, dtype=I64, device=d), "f": torch.zeros(W, 4, dtype=F64, device=d),
                     "n": torch.zeros(W, dtype=I64, device=d), "loc": torch.zeros(W, dtype=I64, device=d)}

    # ---- Actions (TIME-709, TIME-710) ----

    def push(self, kind, mask: torch.Tensor, ball: torch.Tensor, target: torch.Tensor | None = None,
             f: torch.Tensor | None = None, n: torch.Tensor | None = None, loc: torch.Tensor | None = None) -> Slot:
        """An Actions call (kind: a code or an event name) for every world in mask [W]: ball [W] Destiny ids, target
        [W] ids (FollowBall, Orbit), f [W, k<=4] floats, n [W] the int argument (warp factor, cloak mode, flag), loc [W]
        SetBallPosition's is_local_ball (1 for True). Unused arguments are zero, so the duplicate test compares
        them alike. CloakBall captures the ball's bubble (0, or -1 while its old bubble is not yet its bubble,
        _actions.py:628-638) and is not queued for a ball not in the park (it returns False there)."""
        k = CODE[kind] if isinstance(kind, str) else int(kind)
        if k not in self.a.kinds:
            raise ValueError(f"{KINDS[k] or kind} is not applied by the park ({type(self.a).__name__})")
        W, d = self.W, self.dev
        z = lambda dt: torch.zeros(W, dtype=dt, device=d)                    # noqa: E731
        mask = mask.to(BOOL).to(d)
        ball = ball.to(I64).to(d)
        target = z(I64) if target is None else target.to(I64).to(d)
        ff = torch.zeros(W, 4, dtype=F64, device=d)
        if f is not None:
            f = f.to(F64).to(d)
            ff[:, :f.shape[1]] = f.reshape(W, -1)
        n = z(I64) if n is None else n.to(I64).to(d)
        loc = z(I64) if loc is None else loc.to(I64).to(d)
        if self.kernel == "cuda":                           # the kernel tests and looks up at the pre-tick
            slot = Slot(k, None, ball, None, target, None, ff, n, loc)
            slot.mask = mask
            self.slots.append(slot)
            return slot
        row = self.a.lookup(ball)
        trow = self.a.lookup(target) if k in FOLLOWS else z(I64) - 1
        if k == CLOAK:
            safe = row.clamp(min=0)
            loc = torch.where(self.settled[safe], 0, -1)
            mask = mask & (row >= 0)                    # not queued for a ball not in the park
        last = self.last
        dup = ((last["kind"] == k) & (last["ball"] == ball) & (last["target"] == target)
               & (last["f"] == ff).all(dim=1) & (last["n"] == n) & (last["loc"] == loc))
        ok = mask & ~dup
        for key, val in (("kind", torch.full((W,), k, dtype=I64, device=d)), ("ball", ball), ("target", target),
                         ("n", n), ("loc", loc)):
            last[key] = torch.where(ok, val, last[key])
        last["f"] = torch.where(ok[:, None], ff, last["f"])
        slot = Slot(k, ok, ball, row, target, trow, ff, n, loc)
        self.slots.append(slot)
        return slot

    # ---- the pending queries and the undo (_actions.py:686-766) ----
    #
    # Destiny's game code asks the Actions queue between ticks whether a cloak is pending, which cloak mode a ship will
    # have, and undoes a pending cloak. The queue is this tick's slots: on the torch path each call was tested at its
    # push (`ok`); on a kernel path (`ok` None until the pre-tick) the same test runs here, in queue order, and an
    # undo hands the kernel the tested calls (`resolved`: their mask is their queued flag, bit 1 of the kernel's mask
    # byte, so the kernel does not test them again) before it prunes them. Index work only, no float arithmetic.

    def _cloak_loc(self, row: torch.Tensor, pushed: torch.Tensor) -> torch.Tensor:
        """CloakBall's local argument as the queue holds it (_actions.py:628-638): 0, or -1 while the ball's old bubble
        is not yet its bubble or the ball is not in the park (the kernels' rule, which the torch path's push shares
        where the ball is in the park; a ball not in the park is not queued)."""
        return torch.where((row >= 0) & self.settled[row.clamp(min=0)], 0, -1)

    def _deferred(self) -> bool:
        """Whether the pending calls are tested at the pre-tick (a kernel path) rather than at each push."""
        return any(s.ok is None for s in self.slots)

    def _fresh_last(self) -> dict:
        W, d = self.W, self.dev
        return {"kind": torch.zeros(W, dtype=I64, device=d), "ball": torch.zeros(W, dtype=I64, device=d),
                "target": torch.zeros(W, dtype=I64, device=d), "f": torch.zeros(W, 4, dtype=F64, device=d),
                "n": torch.zeros(W, dtype=I64, device=d), "loc": torch.zeros(W, dtype=I64, device=d)}

    @staticmethod
    def _follow_last(last: dict, s: Slot, q: torch.Tensor, loc: torch.Tensor) -> None:
        """The world's last queued action becomes the call `s` where it is queued (q [W])."""
        for key, val in (("kind", torch.full_like(last["kind"], s.kind)), ("ball", s.ball), ("target", s.target),
                         ("n", s.n), ("loc", loc)):
            last[key] = torch.where(q, val, last[key])
        last["f"] = torch.where(q[:, None], s.f, last["f"])

    def _resolve(self, commit: bool = False) -> list:
        """A kernel path's pending calls tested now as the kernel tests them at the pre-tick (and as the torch path's
        push does at once): TIME-710's duplicate test against the world's last queued action, CloakBall's local
        argument, CloakBall not queued for a ball not in the park; in queue order, the calls an undo already tested
        (`resolved`, a prefix) taken as they are. Returns each call's queued flag [W]. `commit` (an undo): every call
        becomes resolved, its mask its queued flag and CloakBall's local argument the one tested."""
        last = getattr(self, "_klast", None) or self._fresh_last()
        last = {k: v.clone() for k, v in last.items()}
        out = []
        for s in self.slots:
            if getattr(s, "resolved", False):
                out.append(s.mask)
                continue
            row = self.a.lookup(s.ball)
            mask, loc = s.mask.to(BOOL), s.loc
            if s.kind == CLOAK:
                loc = self._cloak_loc(row, s.loc)
                mask = mask & (row >= 0)
            dup = ((last["kind"] == s.kind) & (last["ball"] == s.ball) & (last["target"] == s.target)
                   & (last["f"] == s.f).all(dim=1) & (last["n"] == s.n) & (last["loc"] == loc))
            ok = mask & ~dup
            self._follow_last(last, s, ok, loc)
            out.append(ok)
            if commit:
                s.mask, s.loc, s.resolved = ok, loc, True
        if commit:
            self._klast = last
        return out

    def _queued(self) -> list:
        """Each pending call's queued flag [W], as Destiny's queue holds it now."""
        return self._resolve() if self._deferred() else [s.ok for s in self.slots]

    def _last_pending(self, kind: int, mask: torch.Tensor, ball: torch.Tensor) -> tuple:
        """Actions._get_pending_actions (_actions.py:709-728) for one kind, every world in mask: (found [W], the last
        such call's slot index [W], -1 where none)."""
        mask, ball = mask.to(BOOL).to(self.dev), ball.to(I64).to(self.dev)
        found = torch.zeros(self.W, dtype=BOOL, device=self.dev)
        at = torch.full((self.W,), -1, dtype=I64, device=self.dev)
        for j, (s, q) in enumerate(zip(self.slots, self._queued())):
            if s.kind != kind:
                continue
            hit = mask & q & (s.ball == ball)
            found = found | hit
            at = torch.where(hit, j, at)
        return found, at

    def _pending_arg(self, at: torch.Tensor, attr: str, col: int | None = None) -> torch.Tensor:
        """The argument `attr` (f's column `col`) of the call at slot index at[w] of each world (0 where at < 0)."""
        if not self.slots:
            return torch.zeros(self.W, dtype=F64 if col is not None else I64, device=self.dev)
        vals = torch.stack([getattr(s, attr) if col is None else getattr(s, attr)[:, col] for s in self.slots])
        return vals.gather(0, at.clamp(min=0)[None, :])[0]

    def has_pending_cloak(self, mask: torch.Tensor, ball: torch.Tensor) -> torch.Tensor:
        """Actions.has_pending_cloak (_actions.py:730-731), every world in mask: [W] bool (False outside mask)."""
        return self._last_pending(CLOAK, mask, ball)[0]

    def has_pending_uncloak(self, mask: torch.Tensor, ball: torch.Tensor) -> torch.Tensor:
        """Actions.has_pending_uncloak (_actions.py:733-734), every world in mask: [W] bool."""
        return self._last_pending(UNCLOAK, mask, ball)[0]

    def _pending_or_current(self, kind: int, mask, ball, attr: str, col, current) -> tuple:
        found, at = self._last_pending(kind, mask, ball)
        row = self.a.lookup(ball.to(I64).to(self.dev))
        in_park = row >= 0
        pend = self._pending_arg(at, attr, col)
        cur = current[row.clamp(min=0)].to(pend.dtype)
        known = mask.to(BOOL).to(self.dev) & (found | in_park)
        return torch.where(found, pend, torch.where(in_park, cur, torch.zeros_like(pend))), known

    def get_pending_cloak_mode(self, mask: torch.Tensor, ball: torch.Tensor) -> tuple:
        """Actions.get_pending_cloak_mode (_actions.py:736-742), every world in mask: the cloak mode of the ball's last
        queued CloakBall as pushed, else the ball's isCloaked (the park's cloak byte), else None: (value [W] int64,
        known [W] bool, False for None and outside mask)."""
        return self._pending_or_current(CLOAK, mask, ball, "n", None, self.a.cloaked)

    def get_pending_interactive_state(self, mask: torch.Tensor, ball: torch.Tensor) -> tuple:
        """Actions.get_pending_interactive_state (_actions.py:752-758): bool(the last queued SetBallInteractive's
        flag), else the ball's isInteractive, else None: (value [W] bool, known [W] bool)."""
        v, known = self._pending_or_current(INTERACTIVE, mask, ball, "n", None, self.a.interactive.to(I64))
        return v != 0, known

    def get_pending_speed_fraction(self, mask: torch.Tensor, ball: torch.Tensor) -> tuple:
        """Actions.get_pending_speed_fraction (_actions.py:760-766): the last queued SetSpeedFraction's value as
        pushed, else the ball's speedFraction (the park's), else None: (value [W] float64, known [W] bool)."""
        return self._pending_or_current(CODE["SetSpeedFraction"], mask, ball, "f", 0, self.a.park.speed_fraction)

    def _pending_free(self, mask: torch.Tensor, ball: torch.Tensor, value: int) -> tuple:
        found, at = self._last_pending(FREE, mask, ball)
        return found & (self._pending_arg(at, "n") == value), found

    def has_pending_set_free(self, mask: torch.Tensor, ball: torch.Tensor) -> tuple:
        """Actions.has_pending_set_free (_actions.py:744-746), every world in mask: whether the ball's
        last queued SetBallFree's flag, as pushed, equals 1: (answer [W] bool, pending [W] bool); where nothing is
        pending Destiny returns the int 0, not False (`len(events) and ...`): `pending` False."""
        return self._pending_free(mask, ball, 1)

    def has_pending_set_not_free(self, mask: torch.Tensor, ball: torch.Tensor) -> tuple:
        """Actions.has_pending_set_not_free (_actions.py:748-750): the last queued SetBallFree's flag,
        as pushed, equals 0: (answer [W] bool, pending [W] bool; `pending` False for Destiny's int 0)."""
        return self._pending_free(mask, ball, 0)

    def undo_pending_cloak(self, mask: torch.Tensor, ball: torch.Tensor) -> None:
        """Actions.undo_pending_cloak (_actions.py:686-707), every world in mask: every queued CloakBall of ball[w]
        pruned from the world's queue; the calls pushed after it are tested against the queue's last action as it
        then stands (Destiny compares with `_system_history[-1]`)."""
        mask, ball = mask.to(BOOL).to(self.dev), ball.to(I64).to(self.dev)
        deferred = self._deferred()
        if deferred:
            self._resolve(commit=True)
        for s in self.slots:
            if s.kind != CLOAK:
                continue
            prune = mask & (s.ball == ball)
            if deferred:
                s.mask = s.mask & ~prune
            else:
                s.ok = s.ok & ~prune
        last = self._fresh_last()
        for s in self.slots:
            self._follow_last(last, s, s.mask if deferred else s.ok, s.loc)
        if deferred:
            self._klast = last
        else:
            self.last = last

    # ---- the tick ----

    def _rows(self, slot: Slot) -> tuple:
        """The slot as per-row masks and arguments: row r is commanded iff its world's call is queued for it."""
        A = self.a
        w = A.world
        m = slot.ok[w] & (slot.row[w] == A.arange)
        return m, slot.trow[w], slot.f[w], slot.n[w]

    def pre_tick(self, snapshot: bool | None = None, mask: torch.Tensor | None = None) -> None:
        """DoPreTick's Ticker.tick (TIME-712): the stamp moves, the queue is flushed and applied (CloakBall after the
        bubble update), the events routed. `snapshot`: record the streams the bubble additions and the uncloaks
        send (for `decode`); by default when any can happen (the first pre-tick, an uncloak or a SetBallInteractive
        queued), decided on the host from the queued kinds. `mask` [W] (OnTick's): the worlds that
        tick; every other world keeps its queue (each queued call split by its worlds), its stamp, its last action and
        its bubble update's lists, and routes nothing (`Batch.ticked`)."""
        if self.kernel == "cuda":
            return self._pre_tick_cuda(snapshot, mask)
        A = self.a
        on = None if mask is None else mask.to(BOOL).to(self.dev).reshape(self.W)
        clock = A.clock().clone()                           # (the park's clock moves in place at its tick)
        stamp = self.stamp_for_system
        if on is None:
            self.stamp_for_system = clock + 1               # _increment_timestamp, after the (empty) clean-up
            slots, self.slots = self.slots, []
            self._reset_last()
            kinds = {s.kind for s in slots}
            if snapshot is None:
                snapshot = (UNCLOAK in kinds) or (INTERACTIVE in kinds) or (TROLL in kinds) or not self._seen_first
        else:                                               # a world without a tick keeps its queue
            self.stamp_for_system = torch.where(on, clock + 1, stamp)
            slots = self._split_queue(on, "ok")
            self.last = self._keep_last(self.last, on)
            kinds = {s.kind for s in slots}
            first = self._first_pre_tick(on)
            if snapshot is None:
                snapshot = (UNCLOAK in kinds) or (INTERACTIVE in kinds) or (TROLL in kinds) or first
        for s in slots:                                     # _apply_park_parent_methods_except_cloak_ball
            if s.kind == CLOAK:
                continue
            A.apply(s.kind, *self._rows(s))
        # BubbleUpdater (_bubbleupdater.py:19-54): AdditionsAndDeletions and CopyBubbles once per park time; a second
        # pre-tick at the same time (no Evolve between) reuses the lists, the streams written anew
        user = A.interactive & A.exists
        fresh = user & ~self.settled                        # a user ball's old bubble is not yet its bubble
        listed = A.exists & (A.cloaked == 0)                # every uncloaked ball of its bubble, itself included
        fresh = fresh & A.any_in_world(listed)[A.world]     # an empty list is not sent
        again = (clock == self._bubble_time)[A.world]
        if on is not None:                                  # a world without a tick keeps its lists
            again = again | ~on[A.world]
        fresh = torch.where(again, self._fresh, fresh)
        listed = torch.where(again, self._listed, listed)
        self.settled = torch.where(again, self.settled, self.settled | user)
        if on is None:
            self._bubble_time, self._fresh, self._listed = clock, fresh, listed
        else:
            self._bubble_time, self._fresh, self._listed = torch.where(on, clock, self._bubble_time), fresh, listed
        if on is not None:                                  # the batch's additions are the ticked
            fresh = fresh & on[A.world]                     # worlds' (the state, self._fresh, kept)
        additions = (fresh, listed, A.stream_fields() if snapshot else None)
        if on is None:
            self._all_seen()
        live = A.any_in_world(A.interactive & A.exists)
        route = torch.zeros(self.W, len(slots), dtype=torch.int8, device=self.dev)
        notices = []
        for j, s in enumerate(slots):                       # _apply_history
            valid = s.ok & (s.row >= 0)
            safe = s.row.clamp(min=0)
            if s.kind == CLOAK:                             # to the bubble queued in; then applied
                r = torch.where(s.ok & (s.loc == 0), BUBBLE, NONE)
                m, _, f, n = self._rows(s)
                A.cloak(m, n, f[:, 0])
            elif s.kind == UNCLOAK:                         # the event to its own character; AddBallsToPark to others
                own = valid & A.interactive[safe] & (self.pilot[safe] >= 0) & live
                r = torch.where(own, CHARACTER, NONE)
                rec_world = valid & live
                rec = (rec_world[A.world] & A.interactive & A.exists & (A.arange != s.row[A.world])
                       & (self.pilot >= 0))
                notices.append((j, rec, A.stream_fields() if snapshot else None))
            elif s.kind == SETPOS:                          # a local ball's move is sent nowhere
                go = valid & (s.loc == 0) & live
                r = torch.where(go, torch.where(A.cloaked[safe] == 0, BUBBLE,
                                                torch.where(self.pilot[safe] >= 0, CHARACTER, NONE)), NONE)
            else:
                go = valid & live & (s.ball > 0)
                r = torch.where(go, torch.where(A.cloaked[safe] == 0, BUBBLE,
                                                torch.where(self.pilot[safe] >= 0, CHARACTER, NONE)), NONE)
            route[:, j] = r.to(torch.int8)
        self.batch = Batch(stamp=stamp, clock=clock, slots=slots, route=route, notices=notices, additions=additions,
                           ticked=on)

    _seen_first = False

    def _pre_tick_cuda(self, snapshot: bool | None, mask: torch.Tensor | None = None) -> None:
        """The pre-tick as the kernel (vec_server_cuda): one launch, or, with snapshots (for `decode`), the bubble
        update, then the history split at each UncloakBall (the streams it sends are written there), then the flags.
        `mask` [W] (OnTick's): the queue split by worlds (the calls of a world without a tick kept,
        their mask bits cleared for the launch); the kernel's code is unchanged and a world with no pushed call gets
        no park write from it (every park write is a queued call's), only its stamp, its bubble update and its flags,
        which are put back (`_put_back_cuda`)."""
        A, W, R, d = self.a, self.W, self.R, self.dev
        on = None if mask is None else mask.to(BOOL).to(d).reshape(W)
        if on is None:
            slots, self.slots = self.slots, []
            self._klast = None                              # the next tick's queue starts empty (the undo's state)
        else:                                               # a world without a tick keeps its queue
            slots = self._split_queue(on, "mask")
            if getattr(self, "_klast", None) is not None:
                self._klast = self._keep_last(self._klast, on)
        J = len(slots)
        kinds = [s.kind for s in slots]
        if on is None:
            if snapshot is None:
                snapshot = (UNCLOAK in kinds) or (INTERACTIVE in kinds) or (TROLL in kinds) or not self._seen_first
            self._all_seen()
        else:
            first = self._first_pre_tick(on)
            if snapshot is None:
                snapshot = (UNCLOAK in kinds) or (INTERACTIVE in kinds) or (TROLL in kinds) or first
            saved = self._server_state()
        st = lambda attr: (torch.stack([getattr(s, attr) for s in slots]) if J else      # noqa: E731
                           torch.zeros(0, W, dtype=I64, device=d))
        q = {"kinds": kinds, "mask": mask_bytes(slots, W, d),
             "ball": st("ball"), "target": st("target"),
             "f": torch.stack([s.f for s in slots]) if J else torch.zeros(0, W, 4, dtype=F64, device=d),
             "n": st("n"), "loc_in": st("loc"),
             "ok": torch.zeros(J, W, dtype=BOOL, device=d), "row": torch.zeros(J, W, dtype=I64, device=d),
             "trow": torch.zeros(J, W, dtype=I64, device=d), "loc": torch.zeros(J, W, dtype=I64, device=d),
             "route": torch.zeros(W, J, dtype=torch.int8, device=d),
             "notice": torch.zeros(J, R, dtype=BOOL, device=d), "stamp": torch.zeros(W, dtype=I64, device=d),
             "clock": torch.zeros(W, dtype=I64, device=d), "has_character": torch.zeros(R, dtype=BOOL, device=d),
             "gets_bubble": torch.zeros(R, dtype=BOOL, device=d)}
        launch = self._cuda.launch
        notices, fields = [], None
        if not snapshot:
            launch(self, q, 0)
        else:
            launch(self, q, 1)
            fields = A.stream_fields()
            j0 = 0
            for j in [j for j, k in enumerate(kinds) if k == UNCLOAK]:
                launch(self, q, 2, j0, j)
                notices.append((j, q["notice"][j], A.stream_fields()))
                j0 = j
            launch(self, q, 2, j0, J)
            launch(self, q, 3)
        if on is not None:                                  # the worlds without a tick as they were
            self._put_back_cuda(on, saved, q)
        if not snapshot:
            notices = [(j, q["notice"][j], None) for j, k in enumerate(kinds) if k == UNCLOAK]
        for j, s in enumerate(slots):                       # the slots as the torch path holds them
            s.ok, s.row, s.trow, s.loc = q["ok"][j], q["row"][j], q["trow"][j], q["loc"][j]
        self._q = q                                         # for the flags again after the park's tick (post_tick)
        self.batch = Batch(stamp=q["stamp"], clock=q["clock"], slots=slots, route=q["route"], notices=notices,
                           additions=(self._fresh.clone() if on is None else self._fresh & on[A.world],
                                      self._listed.clone(), fields),
                           gets_bubble=q["gets_bubble"], has_character=q["has_character"], ticked=on)

    def evolve(self, mask: torch.Tensor | None = None) -> None:
        """The park's tick (Ballpark::Evolve); a world whose bubble is dead is refused (TIME-714: `check`). `mask` [W]
        (OnTick's): those worlds alone, every other world's balls and clock kept bit for bit (the
        adapter's `evolve(worlds)`)."""
        A = self.a
        if mask is None:
            self.dead = self.dead | ~A.any_in_world(A.interactive & A.exists)
            A.evolve()
            return
        on = mask.to(BOOL).to(self.dev).reshape(self.W)
        self.dead = self.dead | (on & ~A.any_in_world(A.interactive & A.exists))
        A.evolve(on)

    def post_tick(self, mask: torch.Tensor | None = None) -> None:
        """DoPostTick's send_batch: per row (its pilot), whether it gets the bubble's narrowcast and a character
        update (_parkupdatebatcher.py:52-119). `mask` [W] (OnTick's; by default the pre-tick's,
        `Batch.ticked`): the worlds that ticked; the others send nothing."""
        on = mask if mask is not None else (self.batch.ticked if self.batch is not None else None)
        if self.kernel == "cuda":                           # the kernel computed the flags at the pre-tick;
            q = getattr(self, "_q", None)                   # they are computed again after the park's tick,
            if q is not None:                               # which can change a row's interactive flag (PetrifyTroll's
                self._cuda.launch(self, q, 3)               # SetBallInteractive(false)): the flags' phase alone, in
                if on is not None:                          # place in the Batch, as send_batch reads them after Evolve
                    ron = on.to(BOOL).to(self.dev).reshape(self.W)[self.a.world]    # (the worlds
                    q["has_character"].logical_and_(ron)                            # without a tick send nothing)
                    q["gets_bubble"].logical_and_(ron)
            return
        A, b = self.a, self.batch
        bubble = (b.route == BUBBLE).any(dim=1) if b.route.shape[1] else torch.zeros(self.W, dtype=BOOL,
                                                                                     device=self.dev)
        has = torch.zeros(self.R, dtype=BOOL, device=self.dev)
        for j, s in enumerate(b.slots):
            m = s.ok[A.world] & (s.row[A.world] == A.arange) & (b.route[:, j] == CHARACTER)[A.world]
            has = has | m
        for j, rec, _ in b.notices:
            has = has | rec
        has = has | b.additions[0]
        pilot = self.pilot >= 0
        if on is not None:                                  # the worlds without a tick send nothing
            pilot = pilot & on.to(BOOL).to(self.dev).reshape(self.W)[A.world]
        b.has_character = has & pilot
        b.gets_bubble = bubble[A.world] & A.interactive & A.exists & pilot

    def tick(self) -> None:
        self.pre_tick()
        self.evolve()
        self.post_tick()

    def check(self) -> None:
        """Raise on a world whose bubble was dead at an evolve, or one in which OnTick's BringOutTheDeads
        removed a ball (`removed`: the bubble update's deletions, RemoveBalls, are not sent by this layer)."""
        if bool(self.dead.any()):
            bad = torch.nonzero(self.dead).flatten().tolist()
            raise RuntimeError(f"worlds {bad[:8]} had no interactive ball at an evolve: on the master their balls would "
                               f"not be evolved (TIME-714), which the batched park does not model")
        removed = getattr(self, "removed", None)
        if removed is not None and bool(removed.any()):
            bad = torch.nonzero(removed).flatten().tolist()
            raise RuntimeError(f"worlds {bad[:8]} had a ball removed by OnTick's BringOutTheDeads: the bubble update's "
                               f"deletions (RemoveBalls) are not sent by the one-bubble layer")

    # ---- OnTick (Ballpark.cpp:219-330, TIME-702 to TIME-705; the SpaceServer.on_tick) ----

    def on_tick(self, sim_time) -> list:
        """Ballpark::OnTick on the master for every world at once, each at its own simulation time (sim_time [W], 100 ns
        units; exact_ballpark.full.FullBallpark.on_tick, the Ticker's tick at DoPreTick and send_batch at DoPostTick, as
        Destiny's server hangs them): BringOutTheDeads first (the park's, every world: VecCombatPark's; a world with a
        removal is flagged, `removed`); less than a whole interval since the world's time: nothing (the time kept, the
        queue kept); on the master a backlog of more than five intervals cut to one; a world's first call evolves once
        without the tick events (its queue kept); every other world runs its whole intervals, each a pre-tick, the
        park's tick and a post-tick, the worlds with an interval left ticking together under a mask (a round of every
        world runs the unmasked tick), the remainder carried in its time. `time` [W] (Be::Time mTime) and `first_time`
        [W] (mFirstTime) are made at the first call. Returns the batches of the ticks run, in order (a world's sends
        are in those whose `ticked` holds it); `tick_counts` holds the call's counts. Host synchronisations: the
        counts, a round's all-worlds test, the masked pre-tick's (the kept calls, a world's first pre-tick)."""
        from .full import MAX_CATCH_UP_TICKS
        A, W, d = self.a, self.W, self.dev
        if getattr(self, "time", None) is None:
            self.time = torch.zeros(W, dtype=I64, device=d)
            self.first_time = torch.ones(W, dtype=BOOL, device=d)
        sim = torch.as_tensor(sim_time, dtype=I64).to(d).reshape(W)
        interval = int(round(A.park.dt * 1000.0)) * 10000              # mTickInterval x 10,000 (dt = ms x 0.001)
        self._bring_out_on_tick()
        last = sim - self.time
        go = ~(last < interval)                                         # too early: nothing (Ballpark.cpp:235)
        n = torch.where(go, torch.div(last, interval, rounding_mode="floor"), 0)
        resync = go & (n > MAX_CATCH_UP_TICKS) & bool(A.master)         # the master's resynchronisation (246-257)
        last = torch.where(resync, interval, last)
        n = torch.where(resync, 1, n)
        first = go & self.first_time                                    # the first call: Evolve once, no events
        rest = go & ~first
        runs = torch.where(rest, n, 0)
        counts = torch.stack(((~go).sum(), first.sum(), first.all().to(I64), resync.sum(), runs.sum(),
                              (rest & (last - n * interval > 0)).sum(), runs.max() if W else runs.sum())).tolist()
        self.tick_counts = {"worlds too early": counts[0], "first calls": counts[1], "resynchronisations": counts[3],
                            "intervals run": counts[4], "remainders carried": counts[5]}
        if counts[1]:
            self.evolve(mask=None if counts[2] else first)
        self.first_time = self.first_time & ~first
        batches = []
        for k in range(counts[6]):
            m = rest & (n > k)
            mm = None if bool(m.all()) else m                           # a round of every world: the unmasked tick
            self.pre_tick(mask=mm)
            self.evolve(mask=mm)
            self.post_tick(mask=mm)
            self.batch.ticked = m
            batches.append(self.batch)
        self.time = torch.where(go, sim - torch.where(rest, last - n * interval, 0), self.time)
        return batches

    def _bring_out_on_tick(self) -> None:
        """OnTick's BringOutTheDeads before the interval check, every world (Ballpark.cpp:231; TIME-713): the park's
        (VecCombatPark's moribund balls due; no other park removes a ball); a world that lost a ball is flagged in
        `removed` (`check`)."""
        park = self.a.park
        if not hasattr(park, "bring_out_the_deads"):
            return
        before = self.a.exists.clone()
        park._bring_out_on_tick()
        gone = self.a.any_in_world(before & ~self.a.exists)
        removed = getattr(self, "removed", None)
        self.removed = gone if removed is None else removed | gone

    def _split_queue(self, on: torch.Tensor, flag: str) -> list:
        """OnTick's queue split (the rule): each queued call's flag [W] (`ok` on the torch path,
        tested at the push; a kernel path's pushed `mask`, with an undo's `resolved` kept) split by the worlds that
        tick, `on` [W]; returns the calls for them (the pre-tick's queue) and keeps the rest, in queue order, as the
        queue of the worlds without a tick (the calls no world keeps dropped: one host synchronisation)."""
        slots, self.slots = self.slots, []
        if not slots:
            return []
        kept = [_with_flag(s, flag, getattr(s, flag) & ~on) for s in slots]
        hold = torch.stack([getattr(s, flag) for s in kept]).any(dim=1).tolist()
        self.slots = [s for s, h in zip(kept, hold) if h]
        return [_with_flag(s, flag, getattr(s, flag) & on) for s in slots]

    def _keep_last(self, last: dict, on: torch.Tensor) -> dict:
        """The duplicate test's last queued action (`last`, or a kernel path's undo state) after a masked pre-tick:
        empty for the worlds that ticked (their queue flushed), kept for the others."""
        fresh = self._fresh_last()
        return {k: torch.where(on.reshape(self.W, *([1] * (v.dim() - 1))), fresh[k], v) for k, v in last.items()}

    def _first_pre_tick(self, on: torch.Tensor) -> bool:
        """Whether a world in `on` holds its first pre-tick (the bubble's additions are then sent: their streams are
        recorded); `_seen_first` becomes true once every world has had one (one host synchronisation)."""
        seen = getattr(self, "_seen_w", None)
        if seen is None:
            seen = torch.full((self.W,), bool(self._seen_first), dtype=BOOL, device=self.dev)
        first, every = torch.stack(((on & ~seen).any(), (seen | on).all())).tolist()
        self._seen_w = seen | on
        self._seen_first = bool(every)
        return bool(first)

    def _all_seen(self) -> None:
        """An unmasked pre-tick: every world has had one."""
        self._seen_first = True
        if getattr(self, "_seen_w", None) is not None:      # (only after a masked pre-tick)
            self._seen_w = torch.ones_like(self._seen_w)

    def _server_state(self) -> tuple:
        """A kernel path's server state its pre-tick writes for a world with no queued call: the stamp,
        the bubble update's time and lists (copies)."""
        return tuple(t.clone() for t in (self.stamp_for_system, self._bubble_time, self.settled, self._fresh,
                                          self._listed))

    def _put_back_cuda(self, on: torch.Tensor, saved: tuple, q: dict) -> None:
        """After a masked kernel pre-tick: the worlds outside `on` given back the server state they had (in place, the
        kernels' buffers), their flags cleared."""
        ron = on[self.a.world]
        for cur, old, sel in zip((self.stamp_for_system, self._bubble_time, self.settled, self._fresh, self._listed),
                                 saved, (on, on, ron, ron, ron)):
            cur.copy_(torch.where(sel, cur, old))
        q["has_character"].logical_and_(ron)
        q["gets_bubble"].logical_and_(ron)


# ===================================================================================== a tick as a CUDA graph

class ServerGraph:
    """A VecServer's tick with a fixed schedule of pushes, captured once as a CUDA graph (exact by construction; held
    to eager ticks bit for bit by the tests). `schedule` is the pushes' kinds in their order; before each replay the
    caller writes each push's arguments into `inputs[j]` (mask, ball, target, f, n, loc) and `tick()` replays the
    pushes, the pre-tick, the park's tick and the post-tick, and returns the graph's Batch (its tensors refilled; no
    snapshots: `decode` needs an eager tick). The state (every tensor of the park, the adapter and the server but the
    constants) lives in fixed buffers: the capture ends by copying each state tensor back into its buffer, and `tick()`
    first copies any tensor rebound since (an eager setter between ticks) into its buffer. The park's tick must not
    wait for the host (VecModesPark's kernel; VecOrientPark with dynamical orientation on its kernel, kernel="cuda" (its
    torch path's torque search asks the host whether any ball still searches); VecCollidePark with contact_slots or
    its kernel)."""

    CONSTANT = frozenset(("world", "ids", "exists", "rows_of_world", "ids_of_world", "arange", "pilot"))

    def __init__(self, server: VecServer, schedule: list, warm: int = 2):
        if server.dev.type != "cuda":
            raise ValueError("a CUDA graph needs a CUDA device")
        if server.slots:
            raise ValueError("pushes are queued: tick them eagerly first")
        self.server, self.schedule = server, [CODE[k] if isinstance(k, str) else int(k) for k in schedule]
        W, d = server.W, server.dev
        self.inputs = [{"mask": torch.zeros(W, dtype=BOOL, device=d), "ball": torch.zeros(W, dtype=I64, device=d),
                        "target": torch.zeros(W, dtype=I64, device=d), "f": torch.zeros(W, 4, dtype=F64, device=d),
                        "n": torch.zeros(W, dtype=I64, device=d), "loc": torch.zeros(W, dtype=I64, device=d)}
                       for _ in self.schedule]
        self.state = []                                     # (object, attribute, buffer)
        bufs = {}
        for obj in (server.a.park, server.a, server):
            for name, v in list(vars(obj).items()):
                if not torch.is_tensor(v) or v.device.type != d.type or name in self.CONSTANT:
                    continue                                # (a device type: "cuda" and "cuda:0" alike)
                if id(v) not in bufs:
                    bufs[id(v)] = v.contiguous().clone()
                setattr(obj, name, bufs[id(v)])
                self.state.append((obj, name, bufs[id(v)]))
        self._buf_ids = {id(b) for _, _, b in self.state}
        saved = [b.clone() for _, _, b in self.state]
        time0 = server.a.park.current_time
        side = torch.cuda.Stream(device=d)
        side.wait_stream(torch.cuda.current_stream(d))
        with torch.cuda.stream(side):                       # warm-up: libraries loaded, the allocator primed
            for _ in range(max(1, int(warm))):
                self._body()
        torch.cuda.current_stream(d).wait_stream(side)
        torch.cuda.synchronize(d)
        for (obj, name, b), v in zip(self.state, saved):    # back to the state before the warm-up
            b.copy_(v)
            setattr(obj, name, b)
        server.a.park.current_time = time0
        server.slots = []
        server._reset_last()
        torch.cuda.synchronize(d)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()
        server.a.park.current_time = time0                  # the capture ran no kernel: the state is unchanged
        self.out = server.batch

    def _copy_back(self) -> None:
        """Every state attribute's current tensor into its buffer (a current tensor that is another attribute's buffer
        is cloned first: an evolve's swap leaves old_pos holding pos's buffer), then rebound to it."""
        pairs = []
        for obj, name, b in self.state:
            cur = getattr(obj, name)
            if cur is b:
                continue
            pairs.append((obj, name, b, cur.clone() if id(cur) in self._buf_ids else cur))
        for obj, name, b, cur in pairs:
            b.copy_(cur)
        for obj, name, b in self.state:
            setattr(obj, name, b)

    def _body(self) -> None:
        srv = self.server
        for k, inp in zip(self.schedule, self.inputs):
            srv.push(k, inp["mask"], inp["ball"], inp["target"], inp["f"], inp["n"], inp["loc"])
        srv.pre_tick(snapshot=False)
        srv.evolve()
        srv.post_tick()
        self._copy_back()

    def tick(self) -> Batch:
        if self.server.slots:
            raise RuntimeError("eager pushes before a graph tick: the graph's tick pushes its own schedule")
        self._copy_back()
        self.graph.replay()
        self.server.a.park.current_time += 1
        return self.out


# ===================================================================================== what each client holds

class ClientView:
    """What each pilot's client holds after each tick: the server's state `down` ticks back (exact_ballpark.client.Session:
    a client applies each update at its stamp and its park settles `down` ticks behind), as the last `down` + 1
    recorded states of the fields a learner observes; another pilot's cloaked ship is absent from a client's park (its
    CloakBall event removed it), its own is there. `record()` after each server tick; `view()` the state each client
    holds and its park time."""

    def __init__(self, server: VecServer, down: int, fields: tuple = ("pos", "vel", "mode")):
        if down < 0:
            raise ValueError("down: whole ticks, at least 0")
        self.server, self.down, self.fields = server, int(down), tuple(fields)
        self.depth = self.down + 1
        self.ring: dict = {}
        self.times = torch.full((self.depth,), -1, dtype=I64)
        self.k = 0

    def _get(self, name: str) -> torch.Tensor:
        a = self.server.a
        if name == "cloaked":
            return a.cloaked
        t = getattr(a.park, name)
        return t.reshape(a.R, *t.shape[2:]) if isinstance(a, CollideAdapter) else t

    def record(self, time: int) -> None:
        """Keep the server's state at park time `time` (the time the park has reached)."""
        slot = self.k % self.depth
        for name in self.fields + ("cloaked",):
            t = self._get(name)
            if name not in self.ring:
                self.ring[name] = torch.zeros((self.depth, *t.shape), dtype=t.dtype, device=t.device)
            self.ring[name][slot].copy_(t)
        self.times[slot] = int(time)
        self.k += 1

    def view(self) -> tuple:
        """(time, {field: [R, ...]}, present [R, R] (row x pilot's ship row: the ball is in that pilot's park)): the
        state at the latest recorded time minus `down`; `time` is -1 before that state exists."""
        if self.k <= self.down:
            return -1, {}, None
        slot = (self.k - 1 - self.down) % self.depth
        out = {name: self.ring[name][slot] for name in self.fields}
        cl = self.ring["cloaked"][slot]
        a = self.server.a
        same_world = a.world[:, None] == a.world[None, :]
        present = same_world & ((cl == 0)[:, None] | torch.eye(a.R, dtype=BOOL, device=cl.device))
        return int(self.times[slot]), out, present


# ================================================================================ decoding (the host, the tests)

def _pack_ball(fl: dict, r: int, out: list) -> None:
    """Ballpark::WriteBallToStream (Thunkers.cpp:3180-3377; exact_ballpark.server's transcription) from a row's fields."""
    mode = int(fl["mode"][r]) & 0xFF
    glob = "global" in fl and bool(fl["global"][r])     # DSTBALL_ISGLOBAL where the fields hold it
    flags = ((0x01 if bool(fl["free"][r]) else 0) | (0x02 if glob else 0) | (0x04 if bool(fl["massive"][r]) else 0)
             | (0x08 if bool(fl["interactive"][r]) else 0))
    pos = [float(x) for x in fl["pos"][r].tolist()]
    out.append(struct.pack("<qBf3dB", int(fl["ids"][r]), mode, float(fl["radius"][r]), *pos, flags))
    cl = int(fl["cloaked"][r]) & 0xFF
    if mode != 11:                                      # RIGID writes no mass block (a petrified troll)
        out.append(struct.pack("<dbqii", float(fl["mass"][r]), cl - 256 if cl >= 128 else cl, -1, -1, -1))
    if flags & 0x01:
        vel = [float(x) for x in fl["vel"][r].tolist()]
        out.append(struct.pack("<f3dff", float(fl["max_vel"][r]), *vel, float(fl["agility"][r]),
                               float(fl["speed_fraction"][r])))
        if fl["dynamical"]:
            out.append(struct.pack("<4ff3ff", *[float(x) for x in fl["rot"][r].tolist()], float(fl["max_ang_vel"][r]),
                                   *[float(x) for x in fl["ang_vel"][r].tolist()], float(fl["rot_agility"][r])))
    fid = int(fl["formation_id"][r]) if "formation_id" in fl else -1      # mFormationId (VecCombatPark), else -1
    out.append(struct.pack("<b", ((fid + 128) & 0xFF) - 128))
    stamp = int(fl["effect_stamp"][r]) & 0xFFFFFFFF
    stamp = stamp - (1 << 32) if stamp >= (1 << 31) else stamp
    goto = [float(x) for x in fl["goto"][r].tolist()]
    if mode in (1, 4):                                  # FOLLOW, ORBIT
        out.append(struct.pack("<qf", int(fl["follow_id"][r]), float(fl["follow_range"][r])))
    elif mode == 0:                                     # GOTO
        out.append(struct.pack("<3d", *goto))
    elif mode == 3:                                     # WARP
        out.append(struct.pack("<3didqq", *goto, stamp, float(fl["last_collision"][r]), int(fl["follow_id"][r]),
                               int(fl["owner_id"][r])))
    elif mode == 5:                                     # MISSILE (VecCombatPark)
        out.append(struct.pack("<qfqi3d", int(fl["follow_id"][r]), float(fl["follow_range"][r]),
                               int(fl["owner_id"][r]), stamp, *goto))
    elif mode == 12:                                    # FORMATION (VecCombatPark)
        out.append(struct.pack("<qfi", int(fl["follow_id"][r]), float(fl["follow_range"][r]), stamp))
    elif mode == 6:                                     # MUSHROOM: its range, lifetime, start and owner
        out.append(struct.pack("<fdiq", float(fl["follow_range"][r]), float(fl["goto"][r][0]), stamp,
                               int(fl["owner_id"][r])))
    elif mode == 8:                                     # TROLL: when it is due
        out.append(struct.pack("<i", stamp))


def ball_stream(fl: dict, rows: list, time: int) -> bytes:
    """WriteBallsToStream (Thunkers.cpp:2202-2257): DESTINY_BALLS, the time as int32, then each listed ball."""
    t = int(time) & 0xFFFFFFFF
    out = [struct.pack("<Bi", 1, t - (1 << 32) if t >= (1 << 31) else t)]
    for r in rows:
        _pack_ball(fl, r, out)
    return b"".join(out)


def event_of(slot: Slot, w: int, ids: list) -> tuple:
    """The event (name, args) Destiny queued for world w's call in this slot (local arguments stripped), with the
    caller's argument types: floats for the float arguments, ints for ids and the int arguments."""
    name = KINDS[slot.kind]
    b = int(slot.ball[w])
    f = [float(x) for x in slot.f[w].tolist()]
    n = int(slot.n[w])
    if name in _XYZ:
        return name, (b, f[0], f[1], f[2])
    if name in _ONE:
        return name, (b, f[0])
    if name in ("Stop", "UncloakBall"):
        return name, (b,)
    if name in ("FollowBall", "Orbit"):
        return name, (b, int(slot.target[w]), f[0])
    if name == "WarpTo":
        return name, (b, f[0], f[1], f[2], f[3], n)
    if name == "EntityWarpIn":
        return name, (b, f[0], f[1], f[2], n)
    if name in ("SetBallMassive", "SetBallInteractive", "SetBallTroll", "SetBallFree"):
        return name, (b, n)
    if name == "CloakBall":
        return name, (b, n, f[0])
    if name == "SetBallRotation":
        return name, (b, f[0], f[1], f[2], f[3])
    if name == "LaunchMissile":
        return name, (b, int(slot.target[w]), n, int(f[0]), int(f[1]))
    raise ValueError(name)


def decode(server: VecServer, batch: Batch | None = None, client_base: int = 10000) -> list:
    """The batch as Destiny's network sends it, per world: {client id: [(kind, method, state, wait_for_bubble,
    count)]} in the order each client receives them (its singlecast, then the bubble's narrowcast); a state is
    [(stamp, (name, args))] with PackagedAction payloads as exact_ballpark.server.Marshalled. Host only."""
    from .server import ClientUpdateCountThisTick, Marshalled
    A, b = server.a, batch or server.batch
    world = A.world.tolist()
    pilot = server.pilot.tolist()
    ids = A.ids.tolist()
    rows_of = [[r for r in rw if r >= 0] for rw in A.rows_of_world.tolist()]
    route = b.route.cpu().tolist()
    stamp, clock = b.stamp.tolist(), b.clock.tolist()
    oks = [s.ok.tolist() for s in b.slots]
    has, gets = b.has_character.tolist(), b.gets_bubble.tolist()
    out = []
    for w in range(A.W):
        bubble, chars = [], {}
        for j, s in enumerate(b.slots):
            if not oks[j][w]:
                continue
            ev = event_of(s, w, ids)
            if route[w][j] == BUBBLE:
                bubble.append((stamp[w], ev))
            elif route[w][j] == CHARACTER:
                r = int(s.row[w])
                chars.setdefault(pilot[r], []).append((stamp[w], ev))
        front = {}
        for j, rec, fl in b.notices:                # each uncloak's AddBallsToPark, inserted in front as processed
            s = b.slots[j]
            if not oks[j][w] or int(s.row[w]) < 0 or not any(bool(rec[r]) for r in rows_of[w]):
                continue
            if fl is None:
                raise ValueError("this tick ran without snapshots: its uncloaks' streams were not recorded")
            payload = ("AddBallsToPark", (ball_stream(fl, [int(s.row[w])], clock[w]),))
            for r in rows_of[w]:
                if rec[r]:
                    front.setdefault(pilot[r], []).insert(0, (stamp[w], payload))
        adds = {}
        fresh, listed, fl = b.additions
        listing = [r for r in rows_of[w] if bool(listed[r])]
        for r in rows_of[w]:
            if bool(fresh[r]) and pilot[r] >= 0 and listing:
                if fl is None:
                    raise ValueError("this tick ran without snapshots: its additions' streams were not recorded")
                blob = Marshalled([(clock[w], ("AddBallsToPark", (ball_stream(fl, listing, clock[w]),)))])
                adds[pilot[r]] = [(clock[w], ("PackagedAction", blob))]
        per = {}
        for r in rows_of[w]:
            p = pilot[r]
            if p < 0:
                continue
            c = client_base + p
            state = front.get(p, []) + adds.get(p, []) + chars.get(p, [])
            got = []
            if has[r]:
                got.append(("single", "DoDestinyUpdate", state, bool(gets[r]),
                            ClientUpdateCountThisTick.TWO if gets[r] else ClientUpdateCountThisTick.ONE))
            if gets[r]:
                got.append(("narrow", "DoDestinyUpdate", bubble, False,
                            ClientUpdateCountThisTick.TWO if has[r] else ClientUpdateCountThisTick.ONE))
            if got:
                per[c] = got
        out.append(per)
    return out

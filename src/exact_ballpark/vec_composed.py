# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's flight modes, missiles and ship-to-ship collisions in one batched park (the GPU parity
report's items G1 and G2): FOLLOW, ORBIT (both laws) and WARP ships, GOTO and STOP ships, missiles and formations,
colliding by the gradient method (Destiny's default) in Destiny's contact order, for many independent worlds, as
float64 tensors on the CPU or the GPU (`VecComposedPark.evolve_torch`: the tick
moved to VecCombatPark.evolve_torch and its phase 1, warp path and helpers to VecModesPark, verbatim, this park's
collision parts left here as their hooks: `_clear_contacts`, `_warp_massive`, `_contacts_phase`, `_refile`,
`_last_c`, `_list_self_collisions`; `evolve` takes it on any device but a CUDA one).

It must reproduce `exact_ballpark.full.FullBallpark` bit for bit (the gradient method; a client's view, or the master
with one live bubble a world), which is held bit for bit to Destiny's own code compiled verbatim
(tests/test_movement_full_oracle.py, tests/test_movement_space_oracle.py), so it is exact against Destiny (Carbon's
`destiny` at 114e89fa, MIT, (c) CCP Games). Checked by
tests/test_movement_vec_composed.py on generated worlds (tests/park_cases.py). It composes three parks without
changing what they compute: `exact_ballpark.vec_combat.VecCombatPark` (the rows, the commands,
missiles, formations, cloaking, and through it the VecModesPark and the VecOrientPark; one
hook added there, `_collided`, false in VecCombatPark), and the gradient of `exact_ballpark.vec_collide.VecCollidePark`
(the partition's filings, the contact order, Potential with its recursion), whose torch code
runs here unchanged on a view of the worlds. Rules, by rule id:

- TIME-706, TIME-707: the whole tick (`Ballpark::Evolve`, Ballpark.cpp:421): every free ball's behaviour in ascending
  id (phase 1), then for each free ball in ascending id its gradient (a massive ball) and its end-of-tick state
  (phase 3), then the commit with the re-filing and the missiles' collision events (phase 4).
- MOV-714 to MOV-722, MOV-730, TRV-705 to TRV-714 (the kernel, on tensors here): FOLLOW, both orbits, the
  warp's alignment, activation (RealWarp: the ball non-massive) and path, the drop-out (massive again) with its extra
  step, and the release of a warper's followers before or after their own turn by their ids.
- MOV-723, MOV-726, COL-719: MISSILE and FORMATION in phase 1; a missile's collision events.
- COL-701 to COL-711, SEN-705, SEN-706: the gradient for every massive free ball; its candidates
  are GetNearbyBalls' collisionCase1 (Partition.cpp:302, 344): another existing, massive, uncloaked ball that is not a
  missile, in a related partition box, ordered by the partition's walk; Potential with its recursion; the latest
  impact kept (COL-711); each ball re-filed when it moved at least sqrt(0.1) m, and at the drop-out when it is at
  rest (modes.py:711, full.py's evolve); a launched missile re-filed at its launcher (SetBallPosition, space.py's
  set_position).
- COL-703, COL-711 for missiles (the report's G2; the section 9, item 1): a normal missile's one candidate is
  its target, unfiltered and outside the box relation, in its bubble (Partition::GetCollisionCandidates,
  Partition.cpp:301); Gradient's skips (a missile and its owner, a missile and the missile it targets, space junk);
  COL-711's exception (a missile's response to its own target is not kept, but the target's id is appended); every
  other ball's candidates leave missiles out; a missile's collision list is dispatched as DoCollision events.
- COL-703 for defender missiles (the report's G9; the section 9, item 2): a missile with a
  negative owner sees GetNearbyBalls with missiles kept (Partition.cpp:301-305: exclude_missiles false), every other
  existing, massive (or dropping out of warp, by ids), uncloaked ball in a related box, missiles included, in the
  partition's walk order (its target has no first place); Gradient's skips compare the signed owner id, so its
  launcher (|owner|) is not skipped; COL-711's exception for its own target; its response to every other ball is
  kept; its collision list, whatever it met, is dispatched (`events`); the master's self-collision reads |owner|.

**What changes during phase 3.** Each ball's gradient reads the others' state at its own turn. Only a drop-out changes
another ball's state then (WarpDistance, Ballpark.cpp:4260, at the warper's turn: the ball massive again, its
position and velocity the drop-out point's, re-filed when at rest). A ball dropping out this tick was non-massive
before its turn, so it is never a candidate of a ball with a lower id, and is one, at its drop-out state, for every
ball with a higher id. The warp path depends only on the warper's own state, so it is computed before the gradient,
and the dropped ball's candidacy is a per-pair rule on ids. A massive ball in warp proper (SetBallMassive during a
warp) would be seen at two states in one tick and its warp length is the gradient's: refused (FLAG_WARP_MASSIVE).

**The view.** Each tick the rows are gathered into a [W, S] view, a world's rows in ascending id (`slot_rows`), on
which VecCollidePark's `_gradient` runs with the composed candidates (`_GradientView._pairs`), the missile's target
first in its order, and COL-711's missile exception (`_GradientView._update`); the results are gathered back to the
rows. The partition's filings (level, key, adjacent side) are row state, filed by VecCollidePark's `_filing` on the
view: at AddBall (`load`, `add_balls`), SetBallRadius and SetMaxSpeed, the drop-out at rest and the commit.

**Exactness.** Only correctly rounded IEEE operations in the reference's order: every division has a tensor divisor
and a tensor dividend (torch's `scalar / tensor` is a reciprocal times the scalar, and CUDA's `tensor / scalar` a
product by the scalar's reciprocal); sums left to right; no fused multiply-add, no reduction of floats. exp is glibc's
(exact_ballpark.vec_exp); the double sin, cos and log of the orbits and the warp run on the host through the C library
(exact by construction: the reference calls the same functions), on the rows that need them: the orbits' sine and cosine
of one argument from its sincos (exact_ballpark.modes._sincos, as compiled Destiny takes them: the patch), the rest through Python's math; the float functions are vec_libm's (dynamical
orientation, VecOrientPark's code). The torch path therefore synchronises with the host a few times a tick: it is the
exactness reference and the CPU path.

**The kernel path** (`evolve`, exact_ballpark.vec_composed_cuda): the same tick as thirteen launches on
one stream, no allocation and no host synchronisation (a tick with dynamical orientation can be captured in a CUDA
graph, vec_composed_cuda.ComposedGraph): the prepass and phase-1 kernels, the clearing and the warp paths,
the filings at rest, the view's gather, the gradient kernel built with its composed switch (these candidates and
COL-711's exception), the scatter, the commit, the re-filing; held to `evolve_torch` bit for bit, every tensor
every tick (tests/test_movement_vec_composed_cuda.py).

**The commands and the mushrooms**: SetBallFree,
SetBallRigid, the Python SetBoid, SetBallTroll and SetBallInteractive, AddMushroom and BringOutTheDeads are batched for
this park's tick as for VecCombatPark's (its `_g11` lets them through; a subclass with yet another tick refuses them),
with this park's filings: SetBallFree(false) re-files its ball (full.py's set_free: InsertBallInBoxes after Stop and
SetBallVelocity(0, 0, 0)), so PetrifyTroll, after phase 4, re-files each troll it fixes; a free mushroom grows at its
phase-4 turn (glibc's powf, no swap) and is re-filed there with its new radius and its state at the tick's start
(Ballpark.cpp:571), or is removed there (moribund and non-massive, still filed until BringOutTheDeads), its members
released before their own commit when their ids are higher and after phase 4 otherwise (the order); a
mushroom is a massive responder and candidate in the gradient (its owner skips it: Gradient's skips). StopAllFollowers'
members are Destiny's (follow_ptr and the follow id), in phase 1's release by a target starting warp proper too. The
kernel path adds, after the host's yaw, pitch and roll, `vec_composed_cuda.after`'s four launches (three of this
library's and one of the collision kernel's).

**The domain** (refusals are flags on the world's rows, read by `world_flags()` and raised by `check()`): the gradient
method; VecCombatPark's domain less its two-massive-balls refusal and, with collisions, its refusal of defender missiles
(FLAG_DEFENDER: both are computed here; with collisions off both stand, as VecCombatPark's); no massive ball in
warp proper (FLAG_WARP_MASSIVE); no ball at the partition's edge or wider than a top box (FLAG_PARTITION, as
VecCollidePark refuses at its load); on the master no normal cloak, whose proximity sensor re-files its ball at its
checks (FLAG_SENSOR: the sensors are exact_ballpark.vec_space's).

Not here: the iterative method, static collidables, force fields, space junk (no input for it), defender missiles with
collisions off (no partition: VecCombatPark's refusal), proximity sensors, bubbles beyond one a world, the delayed
removal (a RemoveBall with a delay; a mushroom removed in a tick is held), a client's second mushroom while the first is
in the park (VecCombatPark's FLAG_POOL).
"""

from __future__ import annotations

import torch

from . import vec_libm as L
from .modes import MINIBALL, MISSILE, MUSHROOM
from .ref import STOP
from .space import LEVEL_WIDTH, RESPONSE_SCALE
from .vec_collide import VecCollidePark
from .vec_combat import FLAG_DEFENDER, FLAG_NAMES, VecCombatPark
# the flight tick's constants and tensor helpers moved to exact_ballpark.vec_modes; REST_SQ is read as
# this module's by vec_composed_cuda and the tests
from .vec_modes import REST_SQ, _dot  # noqa: F401
from .vec_orient import masked_tick

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
f32 = L.f32

FLAG_WARP_MASSIVE = 64          # a massive ball in warp proper beside collisions (its state changes mid-tick)
FLAG_PARTITION = 128            # a ball at the partition's edge or wider than a top box (VecCollidePark's refusal)
FLAG_SENSOR = 256               # a proximity sensor on the master (a normal cloak's): its re-filing is not modelled
COMPOSED_FLAG_NAMES = {**FLAG_NAMES,
                       FLAG_WARP_MASSIVE: "a massive ball in warp proper beside collisions",
                       FLAG_PARTITION: "a ball at the partition's edge or wider than a top box",
                       FLAG_SENSOR: "a proximity sensor on the master (its re-filing is exact_ballpark.vec_space's)"}


# ================================================================================================= the view

class _GradientView(VecCollidePark):
    """VecCollidePark over a composed park's worlds: W worlds of S slots, each world's rows in ascending id. Its
    `_gradient` (the torch path, unchanged arithmetic) runs with the composed park's candidates (`pair_mask`, set
    by the park each tick), a missile's one candidate first in its order, and COL-711's missile exception."""

    def __init__(self, worlds: int, slots: int, friction: float, tick_interval_ms: int, device):
        super().__init__(worlds, slots, friction=friction, tick_interval_ms=tick_interval_ms, device=device)
        z = lambda dt: torch.zeros(self.W, self.N, dtype=dt, device=self.dev)
        self.missile, self.follow_v = z(BOOL), z(I64)        # mode MISSILE; a missile's target id (Destiny's)
        self.missile_pair = torch.zeros(self.W, self.N, self.N, dtype=BOOL, device=self.dev)
        self.pair_mask = torch.zeros(self.W, self.N, self.N, dtype=BOOL, device=self.dev)

    def _pairs(self, responder: torch.Tensor) -> torch.Tensor:
        """The (ball, other) pairs that call Potential: the park's composed candidates for the responders."""
        return self.pair_mask & responder[:, :, None]

    def _order_keys(self, pair: torch.Tensor) -> torch.Tensor:
        """VecCollidePark's walk order; a missile's one candidate (its target, outside the box relation) first."""
        keys = super()._order_keys(pair)
        return torch.where(self.missile_pair & pair, torch.zeros_like(keys), keys)

    def _update(self, mask, s, a1, other) -> None:
        """Potential's end (COL-711, Ballpark.cpp:2983-2995): VecCollidePark._update, except that a missile's response
        to its own target is not kept (`me.mode != MISSILE or other.id != me.follow_id`); the id is appended."""
        R = self.R
        exempt = self._exempt(other)
        ge = mask & ~exempt & (s >= self.last_collision)
        eq = s == self.last_collision
        lc = a1 * RESPONSE_SCALE
        stronger = _dot(lc, lc) > _dot(self.last_c, self.last_c)
        take = ge & (~eq | stronger)
        self.last_c = torch.where(take[..., None], lc, self.last_c)
        self.last_collision = torch.where(ge & ~eq, s, self.last_collision)
        self.hits = self._append(self.hits, self.hits_n, mask, other)
        self.hits_n = self.hits_n + mask.to(I64)

    def _exempt(self, other: torch.Tensor) -> torch.Tensor:
        """COL-711's exception [W, R]: a missile responder meeting its own target (`other`: the candidate's id)."""
        R = self.R
        return self.missile[:, :R] & (other == self.follow_v[:, :R])

    def filing_bad(self, which: torch.Tensor) -> torch.Tensor:
        """VecCollidePark._check_filing's test per slot [W, S] (a flag here, not an exception)."""
        n = self.grid_t[self.level][..., None]
        side = self.key + self.mod
        bad = (which[..., None] & ((self.key <= 0) | (self.key >= n - 1) | (side < 0) | (side >= n))).any(-1)
        return bad | (which & (self.level == 0) & (self.radius * 2 > LEVEL_WIDTH[0]))


# ================================================================================================= the park

class VecComposedPark(VecCombatPark):
    """VecCombatPark (rows, worlds, every command) with FOLLOW, ORBIT and WARP on tensors and the gradient method's
    collisions between every massive ball, missiles included (the module's docstring); `evolve_torch` is the tick.
    With `collisions=False` the tick is VecCombatPark's (no ball meets another but a missile its target; the
    two-massive-balls refusal kept; no partition filed and no refusal of this park's own)."""

    view_cls = _GradientView                # the gradient's view of the worlds
    _ROW_EXCLUDED = VecCombatPark._ROW_EXCLUDED + ("slot_rows", "slot_ok")     # [world, slot] tables

    def __init__(self, n: int, friction: float = 1.0e6, tick_interval_ms: int = 1000, device="cpu",
                 dynamical_orientation: bool = False, use_new_orbit: bool = False, block: int = 128,
                 master: bool = False, disable_do_for_missiles: bool = False, collisions: bool = True):
        super().__init__(n, friction, tick_interval_ms, device, dynamical_orientation, use_new_orbit, block, master,
                         disable_do_for_missiles)
        self.collisions = bool(collisions)
        d = self.dev
        self.last_c = torch.zeros(n, 3, dtype=F64, device=d)
        self.level = torch.zeros(n, dtype=I64, device=d)
        self.key = torch.zeros(n, 3, dtype=I64, device=d)
        self.mod = torch.zeros(n, 3, dtype=I64, device=d)
        self.hits = torch.full((n, 1), -1, dtype=I64, device=d)
        self.view = None
        self._kbuf = None                       # the kernel path's scratch and view (vec_composed_cuda.buffers)

    # ---- loading, adding, the setters that re-file ----

    def load(self, rows: list[dict]) -> None:
        """VecCombatPark.load, then the view's slots (each world's rows by ascending id) and every ball's first filing
        (AddBall files it: Ballpark.cpp:3303, `_insert_ball_in_boxes`)."""
        super().load(rows)
        d = self.dev
        self.last_c = torch.zeros(self.n, 3, dtype=F64, device=d)
        ids, worlds = self.ids.tolist(), self.world_index.tolist()
        groups: dict = {}
        for r, w in enumerate(worlds):
            groups.setdefault(w, []).append(r)
        W = len(self.labels)
        S = max((len(g) for g in groups.values()), default=1)
        table = []
        for w in range(W):
            g = sorted(groups.get(w, []), key=lambda r: ids[r])
            table.append(g + [-1] * (S - len(g)))
        self.slot_rows = torch.tensor(table, dtype=I64, device=d).reshape(W, S)
        flat = self.slot_rows.reshape(-1).tolist()
        row_slot = [0] * self.n
        for k, r in enumerate(flat):
            if r >= 0:
                row_slot[r] = k
        self.row_slot = torch.tensor(row_slot, dtype=I64, device=d)
        self.slot_ok = self.slot_rows >= 0
        self.view = self.view_cls(W, S, self.friction, self.tick_ms, d)
        # the collision lists: three appends a candidate at most (the recursion), and the master's self-collision
        self.hits = torch.full((self.n, max(1, 3 * (S - 1)) + 1), -1, dtype=I64, device=d)
        self.hits_n = torch.zeros(self.n, dtype=I64, device=d)
        self.level = torch.zeros(self.n, dtype=I64, device=d)
        self.key = torch.zeros(self.n, 3, dtype=I64, device=d)
        self.mod = torch.zeros(self.n, 3, dtype=I64, device=d)
        self._kbuf = None
        self._file(self.exists & ~self.moribund)

    def add_balls(self, mask: torch.Tensor, rows: list) -> None:
        """VecCombatPark.add_balls, then each new ball filed (AddBall)."""
        super().add_balls(mask, rows)
        new = mask & self.exists
        self.last_c = torch.where(new[:, None], 0.0, self.last_c)
        self.hits = torch.where(new[:, None], -1, self.hits)
        self.hits_n = torch.where(new, 0, self.hits_n)
        self._file(new)

    def set_radius(self, mask: torch.Tensor, r: torch.Tensor) -> None:
        """SetBallRadius: VecOrientPark's, then the ball re-filed (space.py's _set_ball_radius)."""
        ok = mask & ~(f32(r.to(F64)) < 0.0) & self.exists
        super().set_radius(mask, r)
        self._file(ok)

    def set_max_velocity(self, mask: torch.Tensor, v: torch.Tensor) -> None:
        """SetMaxSpeed: VecOrientPark's, then the ball re-filed (space.py's set_max_velocity)."""
        ok = mask & ~(f32(v.to(F64)) < 0.0) & self.exists
        super().set_max_velocity(mask, v)
        self._file(ok)

    def launch_missile(self, mask: torch.Tensor, target: torch.Tensor, owner: torch.Tensor, aimed: torch.Tensor,
                       massive: torch.Tensor) -> None:
        """PyLaunchMissile (FullBallpark.launch_missile): VecCombatPark's, with SetBallPosition's re-filing at the
        launcher's position (space.py's set_position), made before SetBallVelocity and MissileFollow change the
        velocity and the speed fraction the filing reads."""
        rows = torch.arange(self.n, device=self.dev)
        go = mask & (target.to(I64) >= 0) & (self.lookup(rows, owner.to(I64).abs()) >= 0) & self.exists
        vel0, sf0 = self.vel.clone(), self.speed_fraction.clone()
        super().launch_missile(mask, target, owner, aimed, massive)
        self._file(go, vel=vel0, sf=sf0)

    def missile_follow(self, mask: torch.Tensor, target: torch.Tensor, owner: torch.Tensor) -> None:
        """MissileFollow (VecCombatPark's; LaunchMissile's too). With collisions a defender missile (a negative owner)
        is held here, its candidates the gradient's (`_pair_mask`), so VecCombatPark's refusal of it (FLAG_DEFENDER)
        is lifted; with collisions off it stands (no partition: no candidates but a normal missile's target)."""
        before = self.flags.clone()
        super().missile_follow(mask, target, owner)
        if self.collisions:
            self.flags = (self.flags & ~FLAG_DEFENDER) | (before & FLAG_DEFENDER)

    def set_free(self, mask: torch.Tensor, flag: torch.Tensor) -> None:
        """SetBallFree (Ballpark.cpp:4887; MOV-732): VecCombatPark's (VecModesPark's, the yaw, pitch and roll snapped
        without dynamical orientation), then a ball fixed re-filed (InsertBallInBoxes after Stop and SetBallVelocity(0, 0,
        0), out of the free set: full.py's set_free), from its state then. PetrifyTroll's runs here."""
        fix = mask & ((flag.to(self.dev) != 0) != self.free) & (flag.to(self.dev) == 0)
        super().set_free(mask, flag)
        self._file(fix & self.exists)

    def _g11(self, name: str) -> None:
        """the commands and AddMushroom are batched for this park's own tick too (its
        torch path and its kernels); a subclass with yet another tick refuses them."""
        if type(self).evolve is not VecComposedPark.evolve or type(self).evolve_torch is not VecComposedPark.evolve_torch:
            raise NotImplementedError(f"{name} is batched for VecComposedPark's own tick only, not {type(self).__name__}")

    def cloak(self, mask: torch.Tensor, mode: torch.Tensor, range_: torch.Tensor | None = None) -> None:
        """CloakBall: VecCombatPark's; on the master a normal cloak's proximity sensor re-files its ball at its checks
        (space.py's _check_sensor), which this park does not model (exact_ballpark.vec_space holds the sensors): such a
        world is flagged (FLAG_SENSOR)."""
        before = self.sensor
        super().cloak(mask, mode, range_)
        if self.master and self.collisions:
            self.flags = self.flags | torch.where(self.sensor & ~before, FLAG_SENSOR, 0)

    # ---- the view ----

    def _to_view(self, x: torch.Tensor, fill=0) -> torch.Tensor:
        """A row tensor [n, ...] as [W, S, ...] (empty slots `fill`)."""
        v = x[self.slot_rows.clamp(min=0)]
        ok = self.slot_ok.reshape(*self.slot_ok.shape, *([1] * (x.dim() - 1)))
        return torch.where(ok, v, torch.full_like(v, fill))

    def _to_rows(self, v: torch.Tensor) -> torch.Tensor:
        """A view tensor [W, S, ...] back as rows [n, ...] (every row has exactly one slot)."""
        return v.reshape(-1, *v.shape[2:])[self.row_slot]

    def _fill_view(self, pos=None, vel=None, sf=None) -> None:
        """The view's per-slot state from the rows (pos, vel and the speed fraction may be given per row)."""
        V = self._to_view
        v = self.view
        v.exists = V(self.exists & ~self.moribund, False)
        v.free = V(self.free, False)
        v.massive = V(self.massive, False)
        v.cloaked = V(self.cloaked != 0, False)
        v.junk = torch.zeros_like(v.exists)
        v.id = V(self.ids)
        v.mass, v.agility, v.radius = V(self.mass), V(self.agility), V(self.radius)
        v.max_vel, v.time_factor = V(self.max_vel), V(self.time_factor)
        v.speed_fraction = V(self.speed_fraction if sf is None else sf)
        v.pos = V(self.pos if pos is None else pos)
        v.vel = V(self.vel if vel is None else vel)
        v.last_g = V(self.last_g)
        v.level, v.key, v.mod = V(self.level), V(self.key), V(self.mod)

    def _file(self, which: torch.Tensor, pos=None, vel=None, sf=None) -> None:
        """Ball::InsertInBoxes's outcome (level, key, adjacent side) for the rows in `which`, by VecCollidePark's
        `_filing` on the view, from each row's state (or the given pos, vel and speed fraction); a filing
        VecCollidePark refuses is flagged (FLAG_PARTITION)."""
        if self.view is None or not self.collisions or not bool(which.any()):
            return
        self._fill_view(pos, vel, sf)
        w = self._to_view(which, False)
        v = self.view
        v.level, v.key, v.mod = v._filing(w)
        bad = self._to_rows(v.filing_bad(w))
        self.level = torch.where(which, self._to_rows(v.level), self.level)
        self.key = torch.where(which[:, None], self._to_rows(v.key), self.key)
        self.mod = torch.where(which[:, None], self._to_rows(v.mod), self.mod)
        self.flags = self.flags | torch.where(bad, FLAG_PARTITION, 0)

    # ---- the domain ----

    def watch(self) -> None:
        """VecCombatPark's two-massive-balls refusal only with collisions off (with them, they collide here)."""
        if not self.collisions:
            super().watch()

    def check(self) -> None:
        """Raise if any world left the domain: this park's flags, then VecCombatPark's."""
        flags = self._or_flags()
        extra = flags & (FLAG_WARP_MASSIVE | FLAG_PARTITION | FLAG_SENSOR)
        if extra:
            names = [v for k, v in COMPOSED_FLAG_NAMES.items() if extra & k]
            raise RuntimeError(f"outside the batched domain: {', '.join(names)}")
        super().check()

    def evolve(self, lib: dict | None = None, active: torch.Tensor | None = None) -> None:
        """On any device but a CUDA one the tick on tensors (`evolve_torch`; `lib`, a kernel library,
        is refused there). The kernel path (a CUDA device): with collisions the composed tick, thirteen launches
        (exact_ballpark.vec_composed_cuda.tick; `lib` its libraries, vec_composed_cuda.libraries()), then without
        dynamical orientation the yaw, pitch and roll on the host, then `after`'s four launches
        (vec_composed_cuda.after, three of this library's and one of the collision kernel's: a removed mushroom's members whose turns
        came first, PetrifyTroll, the petrified trolls re-filed) and, without dynamical orientation, their yaw, pitch and roll snapped (SetBallVelocity's); with
        collisions off VecCombatPark's five launches (`lib` vec_combat_cuda's), then the collision lists made from their
        counts as the tick on tensors keeps them (`_lists_from_counts`).
        `active` [n]: OnTick's per-world mask (vec_orient.masked_tick, in place: a collision's
        candidates are rows of its world)."""
        if self.dev.type != "cuda":                 # the tick on tensors off a CUDA device
            if lib is not None:
                raise ValueError("lib is a kernel library: a park on the CPU ticks on tensors (evolve_torch)")
            self.evolve_torch(active)
            return
        if active is not None:
            masked_tick(self, active, lambda: VecComposedPark.evolve(self, lib), in_place=True)
            return
        if not self.collisions:
            follow = self.follow_id.clone()         # the ids phase 3 appends (`_lists_from_counts`)
            super().evolve(lib)
            self._lists_from_counts(follow)
            return
        from . import vec_composed_cuda as KP
        self.watch()
        KP.tick(self, KP.BLOCK, lib)
        self.after_tick()
        KP.after(self, KP.BLOCK, lib)
        if not self.dynamical_orientation:
            self._ypr(self.petrified.bool() & self.exists, snap=True, time=self.clock - 1)
        self.watch()
        self.current_time += 1

    def kernel_buffers(self) -> dict:
        """The kernel path's own tensors (its scratch rows and the [world, slot] view), made once a load."""
        from . import vec_composed_cuda as KP
        shape = (self.n, *(int(x) for x in self.slot_rows.shape), int(self.hits.shape[1]))
        if self._kbuf is None or self._kbuf["shape"] != shape:
            self._kbuf = KP.buffers(self)
        return self._kbuf

    def _pair_mask(self, responder_v: torch.Tensor, dropped_v: torch.Tensor) -> torch.Tensor:
        """The composed candidates [W, S, S] (the module's docstring), on the filled view."""
        v = self.view
        S = v.N
        idr, idn = v.id[:, :, None], v.id[:, None, :]
        eye = torch.eye(S, dtype=BOOL, device=self.dev)[None]
        mode = self._to_view(self.mode, STOP)
        owner = self._to_view(self.owner_id)
        follow = self._to_view(self.follow_id)
        bubble = self._to_view(self.bubble)
        missile = v.exists & (mode == MISSILE)
        defender = self._defenders(missile, owner)
        seen = v.massive[:, None, :] | self._dropped_seen(dropped_v, idr, idn)
        # GetNearbyBalls without missiles left out (a defender's), then collisionCase1's (every other ball's)
        nearby = v.exists[:, None, :] & seen & ~v.cloaked[:, None, :] & ~eye & v._related()
        cand = nearby & ~self._left_out(missile)[:, None, :]
        tslot = self._to_view(torch.where(self.follow_ptr >= 0, self.row_slot[self.follow_ptr.clamp(min=0)], -1), -1)
        S_ar = torch.arange(S, device=self.dev)
        w_of_slot = (tslot // S)
        same_world = w_of_slot == torch.arange(v.W, device=self.dev)[:, None]
        tgt = ((S_ar[None, None, :] == (tslot % S)[:, :, None]) & (tslot >= 0)[:, :, None] & same_world[:, :, None]
               & v.exists[:, None, :] & (bubble[:, :, None] == bubble[:, None, :]))
        own = torch.where(defender[:, :, None], self._defender_candidates(nearby, cand), tgt)
        pair = torch.where(missile[:, :, None], own, cand)
        mn = mode[:, None, :]
        skip = ((missile[:, :, None] & ((idn == owner[:, :, None])
                                         | ((mn == MINIBALL) & (owner[:, None, :] == owner[:, :, None]))))
                | (((mn == MISSILE) | (mn == MUSHROOM)) & (idr == owner[:, None, :]))
                | ((mn == MISSILE) & (idr == follow[:, None, :]))
                | (v.junk[:, None, :] & ~v.junk[:, :, None]))
        v.missile = missile
        v.follow_v = follow
        v.owner_v = owner
        v.missile_pair = self._target_first(missile, defender)[:, :, None] & tgt
        return pair & ~skip & responder_v[:, :, None]

    @staticmethod
    def _defenders(missile: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
        """The defender missiles [W, S] (Partition::GetCollisionCandidates, Partition.cpp:301: mOwnerID < 0)."""
        return missile & (owner < 0)

    @staticmethod
    def _defender_candidates(nearby: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """A defender's candidates [W, S, S]: GetNearbyBalls with exclude_missiles false (every massive, uncloaked ball
        in a related box, missiles included), not collisionCase1's `cand`."""
        return nearby

    @staticmethod
    def _target_first(missile: torch.Tensor, defender: torch.Tensor) -> torch.Tensor:
        """The responders whose one candidate, their target, is first in order (key 0): the normal missiles only; a
        defender's candidates keep the partition's walk order."""
        return missile & ~defender

    @staticmethod
    def _dropped_seen(dropped_v: torch.Tensor, idr: torch.Tensor, idn: torch.Tensor) -> torch.Tensor:
        """A ball dropping out of warp this tick, as a candidate [W, S, S]: massive from its own turn on, so seen by
        the responders with higher ids only."""
        return dropped_v[:, None, :] & (idn < idr)

    @staticmethod
    def _left_out(missile: torch.Tensor) -> torch.Tensor:
        """The balls no responder but a missile's own sees [W, S]: missiles (collisionCase1's exclude_missiles)."""
        return missile

    def _gradient_rows(self, responder: torch.Tensor, wp, wv, dropped: torch.Tensor, refile: torch.Tensor) -> None:
        """Gradient (Ballpark.cpp:2745) for every responder (live, massive at phase 3), on the view: last_c,
        last_collision and the collision list back on the rows. A ball dropping out this tick is in the view at its
        drop-out state (its filing re-made when at rest), a candidate only of responders with higher ids."""
        pos = torch.where(dropped[:, None], wp, self.pos)
        vel = torch.where(dropped[:, None], wv, self.vel)
        self._fill_view(pos, vel)
        v = self.view
        if bool(refile.any()):
            lv, kv, mv = v._filing(self._to_view(refile, False))
            v.level, v.key, v.mod = lv, kv, mv
        dropped_v = self._to_view(dropped, False)
        resp_v = self._to_view(responder, False)
        v.pair_mask = self._pair_mask(resp_v, dropped_v)
        S = v.N
        v.last_c = torch.zeros(v.W, S, 3, dtype=F64, device=self.dev)
        v.last_collision = torch.full((v.W, S), -1.0, dtype=F64, device=self.dev)
        v.hits = torch.full((v.W, S, max(1, 3 * (S - 1))), -1, dtype=I64, device=self.dev)
        v.hits_n = torch.zeros(v.W, S, dtype=I64, device=self.dev)
        v.imp_slot.fill_(-1)
        v.imp_n.zero_()
        v._gradient(resp_v)
        self.last_c = torch.where(responder[:, None], self._to_rows(v.last_c), self.last_c)
        self.last_collision = torch.where(responder, self._to_rows(v.last_collision), self.last_collision)
        vh = torch.cat([v.hits, torch.full((v.W, S, 1), -1, dtype=I64, device=self.dev)], -1)
        self.hits = torch.where(responder[:, None], self._to_rows(vh), self.hits)
        self.hits_n = torch.where(responder, self._to_rows(v.hits_n), self.hits_n)

    # ---- the hooks in the tick on tensors (VecCombatPark.evolve_torch and VecModesPark._phase1, moved
    # there from this class): this park's collision parts, each the statements that stood at its point ----

    def _clear_contacts(self, live: torch.Tensor) -> None:
        """Phase 1: the live balls' collision responses and lists cleared."""
        self.last_c = torch.where(live[:, None], 0.0, self.last_c)
        self.hits = torch.where(live[:, None], -1, self.hits)
        self.hits_n = torch.where(live, 0, self.hits_n)

    def _warp_massive(self, warping: torch.Tensor) -> None:
        """With collisions, a massive ball in warp proper is flagged (FLAG_WARP_MASSIVE)."""
        if self.collisions:
            self.flags = self.flags | torch.where(warping & self.massive, FLAG_WARP_MASSIVE, 0)

    def _contacts_phase(self, live: torch.Tensor, wp, wv, dropped: torch.Tensor, rest: torch.Tensor) -> None:
        """The gradient for every live massive ball (collisions on), or VecCombatPark's missiles' contacts (off)."""
        if self.collisions:
            responder = live & self.massive
            if bool(responder.any()):
                self._gradient_rows(responder, wp, wv, dropped & ~self.massive, rest)
        else:
            self._legacy_contacts(live)

    def _refile(self, which: torch.Tensor) -> None:
        """Ball::InsertInBoxes for the rows in `which`, with collisions (`_file`)."""
        if self.collisions:
            self._file(which)

    def _last_c(self) -> torch.Tensor:
        """mLastC at phase 3: each row's collision response."""
        return self.last_c

    def _list_self_collisions(self) -> None:
        """The master's self-collision is appended to the missile's list (FullBallpark.evolve's phase 4)."""
        at = torch.arange(self.hits.shape[1], device=self.dev)
        put = (at[None, :] == self.hits_n.clamp(max=self.hits.shape[1] - 1)[:, None]) & self.ev_self[:, None]
        self.hits = torch.where(put, self.ids[:, None], self.hits)
        self.hits_n = self.hits_n + self.ev_self.to(I64)

    def _legacy_contacts(self, live: torch.Tensor) -> None:
        """With collisions off: VecCombatPark's missiles' contacts (each massive normal missile's Potential with its
        target, counted), the collision list its target's id that many times."""
        self.hits_n = self._contacts(live)
        width = self.hits.shape[1]
        ar = torch.arange(width, device=self.dev)
        self.hits = torch.where(ar[None, :] < self.hits_n[:, None], self.follow_id[:, None], -1)

    def _lists_from_counts(self, follow: torch.Tensor) -> None:
        """With collisions off, after VecCombatPark's kernels
        (`evolve` on a CUDA device), which leave each row's contacts with its target counted in `hits_n` and the
        master's self-collision in `ev_self` but keep no list, the collision lists as the tick on tensors keeps them
        (`_legacy_contacts`, then `_list_self_collisions`): the target's id once a contact, then the missile's own id
        on a self-collision; every row's list made anew, as the tick on tensors makes it. `follow` holds the follow
        ids at the tick's start, the ids phase 3 appends: a row counts a contact only as a massive MISSILE at phase 3,
        and within a tick a missile's follow id changes only by StopAllFollowers' release (Stop(Ball*)), which in
        phase 1 makes it non-massive (no contact) and otherwise comes after phase 3 (a removed mushroom's members), so
        each row with a contact has the same follow id at the tick's start and at phase 3. Device tensor work only, no
        host synchronisation."""
        width = self.hits.shape[1]
        ar = torch.arange(width, device=self.dev)
        self.hits = torch.where(ar[None, :] < self.hits_n[:, None], follow[:, None], -1)
        self._list_self_collisions()

    def _collided(self) -> torch.Tensor:
        """Each row's collision list non-empty (CalculateYawPitchRoll's aligning warper with collisions turns along
        its velocity: FullBallpark._yaw_pitch_roll)."""
        return self.hits_n > 0

    # ---- reading ----

    def collision_lists(self) -> list:
        """Each row's collision list (Ball::mCollisions: the ids appended by Potential, in order)."""
        hits, n = self.hits.tolist(), self.hits_n.tolist()
        return [tuple(h[:k]) for h, k in zip(hits, n)]

    def events(self) -> list:
        """The tick's DoCollision events (COL-719, Ball::DispatchCollisions) in Destiny's order: per world (its label,
        in first-appearance order) the missiles that dispatched by ascending id, each with its collision list in order
        (ev_n colliders Potential appended: a normal missile's target, a defender's any candidates) and then itself on
        the master when orphaned: (world label, time, missile id, collider id). With collisions off, VecCombatPark's
        (its kernels count the target's contacts and keep no list). Call after a tick (one host sync)."""
        if not self.collisions:
            return super().events()
        sel = self._ticked_rows((self.ev_n > 0) | self.ev_self)           # a masked tick's rows
        if not bool(sel.any()):
            return []
        rows = sel.nonzero().flatten().tolist()
        world, ids = self.world.tolist(), self.ids.tolist()
        ev_n, ev_self, clock = self.ev_n.tolist(), self.ev_self.tolist(), self.clock.tolist()
        hits = self.hits[sel].tolist()
        lists = dict(zip(rows, hits))
        order = {w: k for k, w in enumerate(self.labels)}
        out = []
        for r in sorted(rows, key=lambda r: (order[world[r]], ids[r])):
            for k in range(ev_n[r]):
                out.append((world[r], clock[r] - 1, ids[r], lists[r][k]))
            if ev_self[r]:
                out.append((world[r], clock[r] - 1, ids[r], ids[r]))
        return out


__all__ = ["VecComposedPark", "FLAG_WARP_MASSIVE", "FLAG_PARTITION", "FLAG_SENSOR", "COMPOSED_FLAG_NAMES"]

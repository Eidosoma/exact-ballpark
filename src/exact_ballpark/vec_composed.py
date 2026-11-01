# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's flight modes, missiles and ship-to-ship collisions in one batched park (the GPU parity
report's items G1 and G2): FOLLOW, ORBIT (both laws) and WARP ships, GOTO and STOP ships, missiles and formations,
colliding by the gradient method (Destiny's default) in Destiny's contact order, for many independent worlds, as
float64 tensors on the CPU or the GPU (`VecComposedPark.evolve_torch`).

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

import math

import torch

from . import modes as MO
from . import vec_exp
from . import vec_libm as L
from .modes import (AU, FOLLOW, FORMATION, MAX_ALIGN_TICKS, MINIBALL, MISSILE, MUSHROOM, ORBIT, ORBITAL_PRECESSION,
                    WARP, WARP_FACTOR_TO_ACCELERATION, WARP_FACTOR_TO_AU_PER_SECOND, WARP_FACTOR_TO_DECELERATION)
from .ref import GOTO, STOP, STOP_DAMP
from .full import _powf as _C_POWF                   # the C library's powf (ctypes), as the reference calls it
from .space import LEVEL_WIDTH, RESPONSE_SCALE
from .vec_collide import VecCollidePark, _to_int64
from .vec_combat import FLAG_DEFENDER, FLAG_NAMES, FU_DISPATCH_MIN_MOVE_SQ, FU_MISSILE_STRAIGHT_MS, VecCombatPark, _char
from .vec_combat import q_rotation_ypr as _q_rotation_ypr
from .vec_orient import (_cols, _fdiv, _fmin, _stack, _where, m_rotation_q, masked_tick, transform_coord,
                         v3_dot, v3_mul)

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
f32 = L.f32

FLAG_WARP_MASSIVE = 64          # a massive ball in warp proper beside collisions (its state changes mid-tick)
FLAG_PARTITION = 128            # a ball at the partition's edge or wider than a top box (VecCollidePark's refusal)
FLAG_SENSOR = 256               # a proximity sensor on the master (a normal cloak's): its re-filing is not modelled
COMPOSED_FLAG_NAMES = {**FLAG_NAMES,
                       FLAG_WARP_MASSIVE: "a massive ball in warp proper beside collisions",
                       FLAG_PARTITION: "a ball at the partition's edge or wider than a top box",
                       FLAG_SENSOR: "a proximity sensor on the master (its re-filing is exact_ballpark.vec_space's)"}
TRUNC7 = 10000000               # _trunc7's scale (modes.py:91)
RF_SCALE = 40000.0              # EvolveOldStyleOrbit's radial falloff, exp(-(r - dist)^2 / 40000) (modes.py:449)
GOTO_FAR = 10.0 * AU            # the old orbit's goto point, 10 AU along the thrust (modes.py:460)
NORMAL_MIN = 1e-12              # GetOrbitalNormal's parallel test (modes.py:484)
ANGLE_SCALE = 0.1               # the new orbit's angle and distance floor (modes.py:500, 507)
ANGLE_ZERO = 1e-10              # the new orbit's circle and linear tests (modes.py:511, 520)
FACTOR_FLOOR = -0.999           # the new orbit's factor floor (modes.py:517)
LOG_FLOOR = 0.001               # math.log(0.001) below the floor (modes.py:518)
SLOW_TARGET = 5e-2              # GotoThrustFollow's slow target (modes.py:541)
THRUST_OFF = 0.1                # OrbitThrust's cut (modes.py:581)
ALIGN_COS = 0.01                # IsAlignedForWarp (modes.py:331)
ALIGN_SPEED2 = 0.5625           # 75 % of maxVel, squared (modes.py:332)
DROP_SPEED = 100.0              # the drop-out speed's cap (modes.py:632)
REST_SQ = 0.1                   # the drop-out's re-filing at rest, and the commit's (modes.py:711, 723)
DIR_DIST = 1.0e17               # GotoDirection's point (modes.py:241)


# ---- tensor helpers: the reference's vector arithmetic, components left to right ----

def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _len(a: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_dot(a, a))


def _unit(v: torch.Tensor) -> torch.Tensor:
    """ref.py's _normalized: times the reciprocal of the length; a zero vector stays as it is."""
    n = _len(v)
    inv = torch.ones_like(n) / torch.where(n == 0.0, 1.0, n)
    return torch.where((n == 0.0)[..., None], v, v * inv[..., None])


def _cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vector3d::Cross (modes.py's _cross): components in Destiny's order."""
    return torch.stack((u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1], u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2],
                        u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]), dim=-1)


def _div(a, b) -> torch.Tensor:
    """IEEE division a / b with tensors on both sides (a Python number becomes a tensor first)."""
    if not isinstance(b, torch.Tensor):
        b = torch.full_like(a, float(b))
    if not isinstance(a, torch.Tensor):
        a = torch.full_like(b, float(a))
    return a / b


def _c_call(fn, x: float) -> float:
    """The C library's double function at x through Python's math (which calls it), with C's results where Python
    raises (outside the reference's domain: the reference itself raises there)."""
    try:
        return fn(x)
    except ValueError:
        if fn is math.log and x == 0.0:
            return -math.inf
        return math.nan
    except OverflowError:
        return math.inf


def _host(fn, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """fn (math.sin, math.cos or math.log: glibc's double functions) on the host for the elements in mask; NaN
    elsewhere. One host round trip."""
    out = torch.full_like(x, math.nan)
    if bool(mask.any()):
        vals = [_c_call(fn, v) for v in x[mask].tolist()]
        out[mask] = torch.tensor(vals, dtype=F64, device=x.device)
    return out


def _host_sincos(x: torch.Tensor, mask: torch.Tensor) -> tuple:
    """glibc's sincos (exact_ballpark.modes._sincos: the sine and cosine of one argument as compiled Destiny takes them,
    the finding F3) on the host for the elements in mask; NaN elsewhere. One host round trip."""
    s, c = torch.full_like(x, math.nan), torch.full_like(x, math.nan)
    if bool(mask.any()):
        vals = [MO._sincos(v) for v in x[mask].tolist()]
        s[mask] = torch.tensor([v[0] for v in vals], dtype=F64, device=x.device)
        c[mask] = torch.tensor([v[1] for v in vals], dtype=F64, device=x.device)
    return s, c


def _trunc7(x: torch.Tensor) -> torch.Tensor:
    """`(double)((int64_t)(x * 10000000)) / 10000000` (modes.py's _trunc7; the integer conversion makes a value in
    (-1e-7, 0] a +0)."""
    return _div(_to_int64(x * float(TRUNC7)).to(F64), float(TRUNC7))


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
        """The kernel path (a CUDA device): with collisions the composed tick, thirteen launches
        (exact_ballpark.vec_composed_cuda.tick; `lib` its libraries, vec_composed_cuda.libraries()), then without
        dynamical orientation the yaw, pitch and roll on the host, then `after`'s four launches
        (vec_composed_cuda.after, three of this library's and one of the collision kernel's: a removed mushroom's members whose turns
        came first, PetrifyTroll, the petrified trolls re-filed) and, without dynamical orientation, their yaw, pitch and roll snapped (SetBallVelocity's); with
        collisions off VecCombatPark's five launches (`lib` vec_combat_cuda's). `active` [n]: OnTick's
        per-world mask (vec_orient.masked_tick, in place: a collision's candidates are rows of its world)."""
        if active is not None:
            masked_tick(self, active, lambda: VecComposedPark.evolve(self, lib), in_place=True)
            return
        if not self.collisions:
            super().evolve(lib)
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

    # ---- phase 1 on tensors: every mode (vec_combat_cuda's prepass and phase-1 kernels, as tensors) ----

    def _activates(self, live: torch.Tensor) -> torch.Tensor:
        """Starts warp proper at its own turn this tick (IsAlignedForWarp of an aligning ball, TRV-706, on its start
        state, which no other ball changes before its turn)."""
        d = _unit(self.goto - self.pos)
        u = _unit(self.vel)
        cos = u[:, 0] * d[:, 0] + u[:, 1] * d[:, 1] + u[:, 2] * d[:, 2]
        fast = _dot(self.vel, self.vel) > ALIGN_SPEED2 * self.max_vel * self.max_vel
        aligned = ((1.0 - cos).abs() < ALIGN_COS) & fast
        forced = self.effect_stamp.abs() > MAX_ALIGN_TICKS
        return live & (self.mode == WARP) & (self.effect_stamp < 0) & (aligned | forced)

    def _released_mode(self) -> torch.Tensor:
        """The mode StopAllFollowers leaves each row in (combat_release with apply false): a missile or an interactive
        orbiter GOTO, unless its velocity or its point is not finite (then its mode stays); any other follower STOP."""
        flies = (self.mode == MISSILE) | (self.interactive & (self.mode == ORBIT))
        point = self.pos + _unit(self.vel) * DIR_DIST
        ok = torch.isfinite(self.vel).all(dim=1) & torch.isfinite(point).all(dim=1)
        return torch.where(flies, torch.where(ok, GOTO, self.mode), STOP)

    def _release_rows(self, f: torch.Tensor) -> None:
        """StopAllFollowers on the follower rows in f (MOV-730): a missile made non-massive, a missile or an
        interactive orbiter flies on by GotoDirection along its velocity, any other follower stops (on the master a
        formation follower's slot freed: VecCombatPark._stop_rows)."""
        if not bool(f.any()):
            return
        missile = f & (self.mode == MISSILE)
        flies = missile | (f & self.interactive & (self.mode == ORBIT))
        self.massive = torch.where(missile, False, self.massive)
        self._stop_rows(f & ~flies)
        self.goto_direction(flies, self.vel.clone())

    def _release_turns(self, hit: torch.Tensor, ts: torch.Tensor) -> tuple:
        """The followers released this tick (`hit`: their target starts warp proper) split by turn (MOV-730): before
        their own turn when the target's id is lower, after it otherwise."""
        return hit & (self.ids[ts] < self.ids), hit & (self.ids[ts] > self.ids)

    def _precessing_normal(self, rows: torch.Tensor) -> torch.Tensor:
        """DefaultOrbitalNormal (Ballpark.cpp:735): the precessing w vector of MOV-717 truncated to 7 decimals, not
        normalised; cos and sin on the host for `rows`."""
        t = self.clock.to(F64)
        phi1 = t * ORBITAL_PRECESSION
        phi2 = (self.ids & 0x000000000000FFFF).to(F64) + t * ORBITAL_PRECESSION
        (s1, c1), (s2, c2) = _host_sincos(phi1, rows), _host_sincos(phi2, rows)
        return torch.stack((_trunc7(c1 * c2), _trunc7(s2), _trunc7(s1 * c2)), dim=1)

    def _old_orbit(self, rows: torch.Tensor, op: torch.Tensor, orad: torch.Tensor) -> tuple:
        """EvolveOldStyleOrbit (Ballpark.cpp:1246; MOV-717, MOV-718): (thrust, goto point)."""
        cruise = self.speed_fraction * self.max_vel
        max_thrust = (self.friction * cruise) / (self.mass * self.agility)
        r = (self.follow_range + self.radius) + orad
        to = op - self.pos
        dist = _len(to)
        to = _unit(to)
        w = _unit(_cross(self._precessing_normal(rows), to))
        to_comp = dist * dist - r * r
        tangent = to_comp >= 0.0
        rad_comp = (r * torch.sqrt(torch.where(tangent, to_comp, 0.0))) / dist
        s = to_comp / dist
        to = torch.where(tangent[:, None], _unit(to * s[:, None] + w * rad_comp[:, None]), to)
        dr = r - dist
        rf = _trunc7(vec_exp.exp(_div((-dr) * dr, RF_SCALE)))
        phi = ((-to[:, 0]) * w[:, 0] + (-to[:, 1]) * w[:, 1]) + (-to[:, 2]) * w[:, 2]
        tf = 1.0 + (rf * rf) * (phi * phi - 1.0)
        tf = torch.where(tf > 0.0, rf * phi + torch.sqrt(torch.where(tf > 0.0, tf, 0.0)), rf * phi)
        tf = tf * torch.where(dist - r >= 0.0, 1.0, -1.0)
        a = (w * rf[:, None] + to * tf[:, None]) * max_thrust[:, None]
        return a, self.pos + a * GOTO_FAR

    def _orbit_thrust(self, target: torch.Tensor) -> torch.Tensor:
        """OrbitThrust (Ballpark.cpp:1428; MOV-722): brake toward the point; zero below a tenth of max acceleration."""
        fom = _div(self.friction, self.mass * self.agility)
        max_accel = (fom * self.speed_fraction) * self.max_vel
        a = (target - self.pos) * fom[:, None] - self.vel
        tacc = _len(a) / max_accel
        inv = torch.ones_like(tacc) / tacc
        a = torch.where((tacc > 1.0)[:, None], a * inv[:, None], a)
        a = torch.where((tacc < THRUST_OFF)[:, None], 0.0, a)
        return torch.where((max_accel <= 0.0)[:, None], 0.0, a)

    def _goto_thrust_follow(self, target: torch.Tensor, tv: torch.Tensor) -> torch.Tensor:
        """GotoThrustFollow (Ballpark.cpp:1463; MOV-721) with weight 1, for the new orbit (its only caller): reach
        the point and match the velocity, capped; OrbitThrust for a slow target."""
        speed = _len(tv)
        slow = speed < (SLOW_TARGET * self.max_vel) * self.speed_fraction
        fom = _div(self.friction, self.mass * self.agility)
        max_accel = (fom * self.speed_fraction) * self.max_vel
        one_minus = 1.0 - self.time_factor
        inv = torch.ones_like(fom) / (self.dt * fom - one_minus)
        a = (((target - self.pos) * fom[:, None] - self.vel * one_minus[:, None]) * fom[:, None]) * inv[:, None]
        inv = torch.ones_like(fom) / (1.0 - self.time_factor)
        avel = ((tv - self.vel * self.time_factor[:, None]) * fom[:, None]) * inv[:, None]
        weight = 1.0
        inv_w = 1.0 / (1.0 + weight)                    # a Python constant: 0.5, exactly
        a = (a + avel * weight) * inv_w
        accel = _dot(a, a)
        capped = accel > max_accel * max_accel
        s = max_accel / torch.sqrt(accel)
        a = torch.where(capped[:, None], a * s[:, None], a)
        a = torch.where((max_accel <= 0.0)[:, None], 0.0, a)
        return torch.where(slow[:, None], self._orbit_thrust(target), a)

    def _new_orbit(self, rows: torch.Tensor, op: torch.Tensor, ov: torch.Tensor, orad: torch.Tensor) -> tuple:
        """EvolveNewStyleOrbit (Ballpark.cpp:1134; MOV-720) with GetOrbitalNormal (MOV-719; with dynamical orientation
        the heading's fallback, FullBallpark._orbital_normal): (thrust, goto point). ov is the target's velocity as
        the ball sees it at its own turn."""
        do = self.dynamical_orientation
        cruise = self.speed_fraction * self.max_vel
        tau = _div(self.mass * self.agility, self.friction)
        max_thrust = cruise / tau
        r = (self.follow_range + self.radius) + orad
        to = op - self.pos
        to = to + ov * self.dt
        dist = _len(to)
        on = dist > 0.0
        inv = torch.ones_like(dist) / torch.where(on, dist, 1.0)
        to = torch.where(on[:, None], to * inv[:, None], torch.tensor([0.0, 0.0, 1.0], dtype=F64, device=self.dev))
        o = _cross(to, self.vel)
        length2 = _dot(o, o)
        par = length2 > NORMAL_MIN
        inv = torch.ones_like(length2) / torch.sqrt(torch.where(par, length2, 1.0))
        normal = o * inv[:, None]
        if do:
            heading = self.rotated_vector(torch.arange(self.n, device=self.dev), (0.0, 0.0, 1.0))
            o2 = _cross(to, heading)
            l2 = _dot(o2, o2)
            use_h = ~par & ~(l2 <= NORMAL_MIN)
            inv2 = torch.ones_like(l2) / torch.sqrt(torch.where(use_h, l2, 1.0))
            normal = torch.where(use_h[:, None], o2 * inv2[:, None], normal)
            default = ~par & ~use_h
        else:
            default = ~par
        default = default & rows
        if bool(default.any()):
            normal = torch.where(default[:, None], self._precessing_normal(default), normal)
        radial = _unit(_cross(normal, to))
        speed = _len(self.vel)
        max_speed = torch.where(cruise < speed, speed, cruise)
        angle = _div(ANGLE_SCALE * (r - dist), max_speed)
        angle = torch.where(angle < -1.0, -1.0, angle)
        angle = torch.where(angle > 1.0, 1.0, angle)
        dist = torch.where(dist < ANGLE_SCALE, ANGLE_SCALE, dist)
        sq = torch.sqrt(dist * max_thrust)
        est_v = torch.where(sq < max_speed, sq, max_speed)
        est_d = est_v * self.dt
        circle = angle.abs() < ANGLE_ZERO
        # the circle: t = est_d / dist
        t_c = est_d / dist
        # the spiral: t from the factor's three branches
        factor = ((est_d * angle) / dist) / torch.sqrt(angle * angle + 1.0)
        floor = ~circle & (factor < FACTOR_FLOOR)
        logb = ~circle & ~floor & (factor.abs() > ANGLE_ZERO)
        lin = ~circle & ~floor & ~logb
        log_floor = _c_call(math.log, LOG_FLOOR)
        lg = _host(math.log, 1.0 + factor, logb & rows)
        t_s = torch.where(floor, _div(torch.full_like(angle, log_floor), angle),
                          torch.where(logb, lg / angle, factor / angle))
        t = torch.where(circle, t_c, t_s)
        need = rows & (circle | floor | logb | lin)
        st, ct = _host_sincos(t, need)
        gp_c = to * (dist * (1.0 - ct))[:, None] + radial * (dist * st)[:, None]
        gv_c = radial * (ct * est_v)[:, None] + to * (st * est_v)[:, None]
        dexp = dist * vec_exp.exp(angle * t)
        gp_s = to * (dist - dexp * ct)[:, None] + radial * (dexp * st)[:, None]
        gv_s = _unit(to * ((-angle) * ct + st)[:, None] + radial * (angle * st + ct)[:, None]) * est_v[:, None]
        gp = torch.where(circle[:, None], gp_c, gp_s)
        gv = torch.where(circle[:, None], gv_c, gv_s)
        goto = self.pos + gp
        gv = gv + ov
        return self._goto_thrust_follow(goto, gv), goto

    def _real_warp(self, act: torch.Tensor) -> None:
        """RealWarp (Ballpark.cpp:4143; TRV-707), the warper's own part (its followers are released by their rows):
        the exit point min range short of the goto point, warp proper from this tick, non-massive, sensor off."""
        if not bool(act.any()):
            return
        dst = self.goto
        delta = _unit(self.pos - dst)
        mr = self.follow_id.contiguous().view(F64)
        dst = dst + delta * mr[:, None]
        delta = dst - self.pos
        self.goto = torch.where(act[:, None], dst, self.goto)
        self.effect_stamp = torch.where(act, self.clock, self.effect_stamp)
        self.last_collision = torch.where(act, _len(delta), self.last_collision)
        self.massive = torch.where(act, False, self.massive)
        self.sensor = torch.where(act, False, self.sensor)

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

    def _orient_composed(self, a: torch.Tensor, steer: torch.Tensor) -> torch.Tensor:
        """EvolveBehaviorForBall's dynamical-orientation branch (Ballpark.cpp:840-890; MOV-728; FullBallpark._orient)
        for the rows in `steer`: VecOrientPark._orient, with a ball in warp proper that coasts turning toward its exit
        point (the torque searched toward mGoto - pos, as floats)."""
        ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
        accel = f32(torch.sqrt(ax * ax + ay * ay + az * az))
        max_accel = _fdiv(f32(f32(torch.full_like(ax, self.ffriction) * self.speed_fraction) * self.max_vel),
                          f32(f32(self.mass) * self.agility))
        thrust = accel > 1e-2 * max_accel
        warping = (self.mode == WARP) & (self.effect_stamp >= 0)
        coast_warp = ~thrust & warping
        t_scale = self._t_scale()
        tf = L.expf(f32(-self.fdt * t_scale))
        kap = self._kappas(tf)
        a32 = _cols(f32(a))
        g32 = _cols(f32(self.goto - self.pos))
        direction = _where(thrust, a32, g32)
        search = steer & (thrust | coast_warp)
        torque = self._goto_torque_breaks(direction, tf, kap, search)
        torque = _where(search, torque, (torch.zeros_like(ax),) * 3)
        torque0, roll0 = self.torque, self.rot_roll
        self.torque = torch.where(steer[:, None], _stack(torque), torque0)
        roll = self._goto_roll_breaks(torque, tf, kap)
        self.rot_roll = torch.where(steer, roll, roll0)
        rotation, angvel = self._apply_torque(torque, roll, self._tau())
        heading = transform_coord((0.0, 0.0, 1.0), m_rotation_q(rotation))
        cosangle = _fmin(torch.ones_like(accel), _fdiv(v3_dot(heading, a32), accel))
        h = _stack(v3_mul(heading, f32(f32(cosangle * cosangle) * accel)))
        steered = torch.where((cosangle <= 0.0)[:, None], 0.0, h)
        return torch.where((steer & thrust)[:, None], steered, a)

    def _phase1(self, live: torch.Tensor) -> None:
        """Phase 1 (EvolveBehaviorForBall, Ballpark.cpp:789) for every live ball in ascending id, as tensors:
        vec_combat_cuda's prepass and phase-1 kernels. A member of a ball that starts warp proper this tick (Destiny's
        test, follow_ptr and the follow id: a ball SetBoid took out of a follow mode too) is released
        before its own turn when the target's id is lower (it then behaves in its new mode), after it otherwise; a ball
        reading its target's velocity sees it after the target's STOP damping when the target's turn came first."""
        do = self.dynamical_orientation
        n = self.n
        t = self.follow_ptr
        t_ok = (t >= 0) & (t < n)
        ts = torch.where(t_ok, t, 0)
        act = self._activates(live)
        pos0, vel0 = self.pos.clone(), self.vel.clone()
        hit = self._members(act) & self.exists      # StopAllFollowers' members (Destiny's test)
        before, after = self._release_turns(hit, ts)
        # the prepass: the vertical velocity each ball shows after its own turn (its mode at its turn)
        m_turn = torch.where(before, self._released_mode(), self.mode)
        vy = vel0[:, 1]
        damp = live & (m_turn == STOP) & (not do)
        vy_after = torch.where(damp, (vy - 0.07 * vy) * STOP_DAMP, vy)
        vy_seen = torch.where(self.ids[ts] < self.ids, vy_after[ts], vel0[ts, 1])
        # the releases before the follower's own turn
        self._release_rows(before)
        # the behaviours
        mode = self.mode
        t = self.follow_ptr
        t_ok = (t >= 0) & (t < n)
        ts = torch.where(t_ok, t, 0)
        a = torch.zeros_like(self.pos)
        warp = live & (mode == WARP)
        proper = warp & (self.effect_stamp >= 0)
        activate = warp & ~proper & act
        align = warp & ~proper & ~act
        self.effect_stamp = torch.where(align, self.effect_stamp - 1, self.effect_stamp)
        self._real_warp(activate)
        go = live & (mode == GOTO)
        a = torch.where((go | align)[:, None], self._thrust_to(self.goto, torch.ones_like(go)), a)
        follow = live & (mode == FOLLOW) & t_ok
        if bool(follow.any()):
            op = pos0[ts]
            delta = self.pos - op
            dist = _len(delta)
            r = (self.follow_range + self.radius) + self.radius[ts]
            inv = torch.ones_like(dist) / torch.where(dist == 0.0, 1.0, dist)
            on = torch.stack((op[:, 0] + 1.0 * r, op[:, 1] + 0.0 * r, op[:, 2] + 0.0 * r), dim=1)
            target = torch.where((dist == 0.0)[:, None], on, op + (delta * r[:, None]) * inv[:, None])
            self.goto = torch.where(follow[:, None], target, self.goto)
            a = torch.where(follow[:, None], self._thrust_to(self.goto, torch.ones_like(go)), a)
        orbit = live & (mode == ORBIT) & t_ok
        if bool(orbit.any()):
            op = pos0[ts]
            if self.use_new_orbit:
                ov = vel0[ts].clone()
                ov[:, 1] = vy_seen
                th, gp = self._new_orbit(orbit, op, ov, self.radius[ts])
            else:
                th, gp = self._old_orbit(orbit, op, self.radius[ts])
            self.goto = torch.where(orbit[:, None], gp, self.goto)
            a = torch.where(orbit[:, None], th, a)
        missile = live & (mode == MISSILE) & t_ok
        if bool(missile.any()):
            a = torch.where(missile[:, None], self._missile_thrust(missile, ts, pos0, vel0, vy_seen), a)
        form = live & (mode == FORMATION) & t_ok
        if bool(form.any()):
            a = torch.where(form[:, None], self._formation_thrust(form, ts, pos0), a)
        stop = live & (mode == STOP)
        if not do:
            v = self.vel.clone()
            v[:, 1] = torch.where(stop, (vy - 0.07 * vy) * STOP_DAMP, v[:, 1])
            self.vel = v
        else:
            steer = live & ~(self.disable_do_for_missiles & (mode == MISSILE))
            a = self._orient_composed(a, steer)
        self.last_g = torch.where(live[:, None], a, self.last_g)
        self.last_c = torch.where(live[:, None], 0.0, self.last_c)
        self.hits = torch.where(live[:, None], -1, self.hits)
        self.hits_n = torch.where(live, 0, self.hits_n)
        # the releases after the follower's own turn (its thrust stands)
        self._release_rows(after)

    # ---- phase 3 ----

    def _warp_path(self, rows: torch.Tensor) -> tuple:
        """WarpDistance (Ballpark.cpp:4260; TRV-708 to TRV-713) with SetupWarpConstants, for the rows in warp proper:
        (position, velocity, dropped out). The logs on the host; the exps glibc's (vec_exp)."""
        wf = self.owner_id.to(F64)
        wd = self.last_collision
        s = (wf * WARP_FACTOR_TO_AU_PER_SECOND) * AU
        acc = wf * WARP_FACTOR_TO_ACCELERATION
        d0 = wf * WARP_FACTOR_TO_DECELERATION
        dec = torch.where(2.0 < d0, 2.0, d0)
        cap = (((wd + 1.0) * acc) * dec) / (acc + dec)
        s = torch.where(cap < s, cap, s)
        t_a = _host(math.log, s / acc, rows) / acc
        t_c = (((wd / s) - (torch.ones_like(acc) / acc)) - (torch.ones_like(dec) / dec)) + (torch.ones_like(s) / s)
        t_d = _host(math.log, s / dec, rows) / dec
        d_a, d_c, d_d = s / acc, s * t_c, s / dec - 1.0
        tt = (self.clock - self.effect_stamp).to(F64) * self.dt
        p, v = self.pos, self.vel
        d = _unit(self.goto - p)
        total = (d_a + d_c) + d_d
        b1 = tt < t_a
        b2 = ~b1 & ((tt - t_a) < t_c)
        b3 = ~b1 & ~b2 & (((tt - t_c) - t_a) < (t_d + 1.0))
        b4 = ~b1 & ~b2 & ~b3
        e1 = vec_exp.exp(acc * tt)
        speed1 = acc * e1
        cur = _len(v)
        speed1 = torch.where(cur > speed1, cur, speed1)
        v1 = d * speed1[:, None]
        p1 = self.goto - d * (total - e1)[:, None]
        v2 = d * s[:, None]
        p2 = self.goto - d * (total - (s * (tt - t_a) + d_a))[:, None]
        e3 = vec_exp.exp((-dec) * ((tt - t_c) - t_a))
        speed3 = s * e3
        v3 = d * speed3[:, None]
        sd = s / dec
        p3 = self.goto - d * (total - (((sd - sd * e3) + d_a) + d_c))[:, None]
        half = _div(self.max_vel, 2.0)
        drop3 = speed3 < torch.where(DROP_SPEED < half, DROP_SPEED, half)
        pw = torch.where(b1[:, None], p1, torch.where(b2[:, None], p2, torch.where(b3[:, None], p3, self.goto)))
        vw = torch.where(b1[:, None], v1, torch.where(b2[:, None], v2, torch.where(b3[:, None], v3, 0.0)))
        drop = rows & ((b3 & drop3) | b4)
        return pw, vw, drop

    # ---- the tick ----

    def _integrate_rows(self, p, v, a) -> tuple:
        m = (self.mass * self.agility)[:, None]
        return self._integrate(p, v, a, m, self.time_factor[:, None])

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

    def _at_rest(self, dropped: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
        """The drop-outs re-filed at their drop-out point: at rest, |v|^2 < 0.1 (modes.py:711)."""
        return dropped & ~(_dot(wv, wv) >= REST_SQ)

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

    def evolve_torch(self, active: torch.Tensor | None = None) -> None:
        """One tick on tensors (CPU or GPU) for every mode: phase 1 (`_phase1`), the warp paths, the gradient
        (collisions, missiles' contacts), phases 3 and 4 (the warp path or the integrator with mLastG + mLastC, the
        drop-out's extra step, the rotation a tick on or the disabled missile's snap, the commit, the re-filing, the
        dispatch counts), the clock, then `after_tick`. a free mushroom grows at its phase-4 turn
        (glibc's powf, no swap, re-filed with its new radius) or is removed there (its members released before their
        own commit when their ids are higher, after phase 4 otherwise), and PetrifyTroll after phase 4 (`_after_phase4`:
        SetBallFree(false) re-files the troll), as VecCombatPark's tick with this park's filings.
        `active` [n]: OnTick's per-world mask, as `evolve`'s."""
        if active is not None:
            masked_tick(self, active, lambda: VecComposedPark.evolve_torch(self), in_place=True)
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
        if self.collisions:
            self.flags = self.flags | torch.where(warping & self.massive, FLAG_WARP_MASSIVE, 0)
        self.last_collision = torch.where(live & self.massive, -1.0, self.last_collision)
        wp, wv, dropped = self._warp_path(warping)
        rest = self._at_rest(dropped, wv)
        if self.collisions:
            responder = live & self.massive
            if bool(responder.any()):
                self._gradient_rows(responder, wp, wv, dropped & ~self.massive, rest)
        else:
            self._legacy_contacts(live)
        # the drop-out: massive and sensing again, Stop, its state the drop-out point's, re-filed at rest
        self.massive = torch.where(dropped, True, self.massive)
        self.sensor = torch.where(dropped, True, self.sensor)
        self._stop_rows(dropped)
        self.pos = torch.where(dropped[:, None], wp, self.pos)
        self.vel = torch.where(dropped[:, None], wv, self.vel)
        if self.collisions:
            self._file(rest)
        # phase 3: the end-of-tick states; the rotation a tick on, by the mode then
        a = self.last_g + self.last_c
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
            if self.collisions:
                self._file(grow)
        self._stop_rows(dying)                          # RemoveBall in Evolve: Stop(Ball*), moribund, non-massive
        self.moribund = self.moribund | dying
        self.massive = torch.where(dying, False, self.massive)
        self.effect_stamp = torch.where(dying, self.clock, self.effect_stamp)
        # phase 4: the re-filing of a ball that moved, the missiles' dispatch
        dd = self.pos - before
        moved = commit & ~((dd[:, 0] * dd[:, 0] + dd[:, 1] * dd[:, 1] + dd[:, 2] * dd[:, 2]) < FU_DISPATCH_MIN_MOVE_SQ)
        if self.collisions:
            self._file(moved)
        dispatch = moved & (self.mode == MISSILE)
        self.ev_n = torch.where(dispatch, self.hits_n, 0)
        self.ev_self = dispatch & self._orphans() if self.master else torch.zeros_like(dispatch)
        # the master's self-collision is appended to the missile's list (FullBallpark.evolve's phase 4)
        at = torch.arange(self.hits.shape[1], device=self.dev)
        put = (at[None, :] == self.hits_n.clamp(max=self.hits.shape[1] - 1)[:, None]) & self.ev_self[:, None]
        self.hits = torch.where(put, self.ids[:, None], self.hits)
        self.hits_n = self.hits_n + self.ev_self.to(I64)
        self.clock = self.clock + 1
        self.after_tick()
        self._after_phase4(later)
        self.watch()
        self.current_time += 1

    def _legacy_contacts(self, live: torch.Tensor) -> None:
        """With collisions off: VecCombatPark's missiles' contacts (each massive normal missile's Potential with its
        target, counted), the collision list its target's id that many times."""
        self.hits_n = self._contacts(live)
        width = self.hits.shape[1]
        ar = torch.arange(width, device=self.dev)
        self.hits = torch.where(ar[None, :] < self.hits_n[:, None], self.follow_id[:, None], -1)

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

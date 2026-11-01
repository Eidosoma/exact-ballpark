# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's collisions batched: the gradient method (Destiny's default) and the iterative method (behind its switch)
for W independent small worlds of up to N balls each, with GOTO and STOP, in Destiny's tick order, as float64 tensors
on the CPU or the GPU.

It must reproduce `exact_ballpark.space.SpaceBallpark` bit for bit (a client's view, `is_master` false), which is itself
held bit for bit to Destiny's own code compiled verbatim (tests/test_movement_space_oracle.py), so it is exact against
Destiny (Carbon's `destiny` at 114e89fa, MIT, (c) CCP Games). Checked by
tests/test_movement_vec_collide.py on generated worlds, on the CPU and the GPU, and on a sample against the compiled
oracle directly. Rules, by rule id:

- TIME-706, TIME-707: the tick (`Ballpark::Evolve`, Ballpark.cpp:421): every free ball's behaviour (GOTO's thrust or
  STOP's vertical damping, mLastG; mLastC and the collision list cleared), then (iterative method only) the collision
  responses, then each free ball's end-of-tick state (gradient method: Gradient first, for massive balls), then the
  commit, each ball re-filed in the partition when it moved at least sqrt(0.1) m.
- SEN-705, SEN-706: the partition, as far as collisions read it: each ball's filing (the level of its inflated radius,
  its box's key and the adjacent boxes its inflated sphere reaches, `Ball::InsertInBoxes`, Ball.cpp:2354) is state,
  re-filed by the reference's rule. Boxes are not stored: a box exists while a ball is filed in it or below it, so the
  candidates of `Partition::GetNearbyBalls` (Partition.cpp:344) and their order follow from the filings alone (below).
- COL-701 to COL-711, the gradient method: `Gradient` (Ballpark.cpp:2745), `Potential` with its overlap recursion
  (2789), `CollideTwoSpheres` and `Quadratic` (Collision.cpp:108, 137): candidates are other massive, uncloaked balls,
  free or fixed (a fixed ball is a static sphere, mass 1e34 in the response), space junk one-sided; the response
  reflects off a fixed ball, exchanges momentum with a free one, pushes overlapping balls apart, keeps 0.85 of it, the
  later impact replacing an earlier one.
- COL-714 to COL-718, the iterative method (`CalculateIterativeCollisionResponses`, Ballpark.cpp:332, and its helpers):
  see `VecCollidePark._iterative`.

**The contact order.** A ball's candidates come in the order GetNearbyBalls walks them: its boxes by key; for each, its
non-empty ancestors bottom-up (unless an earlier box of the ball shares them), its descendants with balls in post-order
(children by key), then the box itself; within a box the balls by id; each ball at its first box. A ball's boxes are
the product of {key} or {key, key + mod} on each axis, so whether two balls are related is a per-axis test
(`_related`) and the walk's order is a sort key per pair (`_order_keys`): the index of the first of the ball's boxes the
other ball relates to, the part (ancestor, below, the box itself), the ancestor's level or the box's post-order code
(its path digits from the ball's level, padded past its own level with a digit that sorts last), then the other ball's
id. The gradient's per-pair work (the time of impact and every branch of Potential at its three recursion depths) is
computed for all pairs at once; only the order-dependent updates of mLastC and mLastCollision (and the collision list)
run sequentially, over each ball's contacts in that order.

**Two ways to run the gradient.** With `contact_slots=None` (the default) the host prunes: a tick with no candidate
pair does nothing more, depths 1 and 2 run only when a pair overlaps, the impact branch (two exps) only on its pairs,
and the sort keys only when a ball has two or more contacts; a few host syncs a tick. With `contact_slots=k` nothing
waits for the host (for capture in a CUDA graph): every branch is computed for every pair, and a ball with more than k
contacts in a tick is an overflow event, raised by `check()`. With `impact_slots=m` as well, the impact
branch (its exps) runs only on the first m (depth, ball, other) pairs that reach it, gathered with static shapes, every
input the same element as the dense call's (more than m in a tick is an overflow event); and with one row (below) no
other ball is free, so the recursion's depths 1 and 2, which only two free balls reach, are not computed.

**Rows** (the section 5.5 lever): only free balls respond, so with `rows=R` a world's free balls take
its first R slots (ascending id), its fixed balls the rest (ascending id), and the per-pair work runs over the R x N
(ball, other) pairs instead of N x N: K ships among M obstacles cost K x (K + M), not (K + M)^2. The arithmetic and
its order are unchanged (the same expressions on the responders' rows; the free balls keep their id order, which the
iterative method walks); `rows=None` (the default) is every slot, as before. A world with more than R free balls is
refused at the load. The collision list (`hits`) and the impact record are kept on the rows: fixed balls collect
neither.

**The impact record**: each impact of the gradient's s > 0 branch at depth 0 is kept per ball in candidate
order: the other ball's slot, the time of impact and the closing speed along the normal
at it (`imp_slot`, `imp_s`, `imp_closing`, `imp_n`), for a damage rule of the caller's. It changes nothing else.

**The kernel**: with `kernel="cuda"` (a CUDA device, the gradient method) the gradient's collision step
is one CUDA kernel, one thread a responder, running this module's algorithm with every operation's rounding pinned,
and the tick's other parts are three more launches (the behaviours, the end of the tick with its re-filing, and the
filing itself), so a tick is four launches (exact_ballpark.vec_collide_cuda; the same bits,
tests/test_movement_vec_collide_cuda.py); `kernel="torch"` (the default) is the torch path below, unchanged, the
kernel's exactness reference. The kernels update the park's tensors in place, and the collision kernel computes the
impact branch for every pair that reaches it (impact_slots does not apply); contact_slots applies as here. The
iterative method stays on the torch path.

**Exactness.** Only correctly rounded IEEE operations, in the reference's order: +, -, *, sqrt, division with a tensor
divisor (CUDA divides by a host scalar through its reciprocal), comparisons and integer steps; vector sums left to
right; no fused multiply-add, no compilation, no float atomics or reductions. exp (an integration over part of a tick)
is glibc's, from `exact_ballpark.vec_exp`, exact on every device.

**The master's bubbles**. With `bubble` set ([W, N] int64, each slot's bubble; the
partition layer exact_ballpark.vec_space keeps it through its CollideView, as Destiny's master files its balls), a
candidate in another bubble is skipped, on the torch path (the gradient's pairs and the iterative method's candidate
lists) and on the kernel, as the master's GetNearbyBalls skips it (Partition.cpp:344; SEN-708; there are no global
balls here); `evolve(skip=...)` leaves the master's dead bubbles' balls out of the tick (TIME-714). With `bubble` None
(the default) the park is a client's view, as before, every operation unchanged.

**Global balls** (SEN-708). A park made with `global_balls=True` holds each ball's global flag
(`glob` [W, N], AddBall's isGlobal at the load as a ball's `global`, SetBallGlobal's `set_global`, applied at once as
Destiny's park applies it); on the master's view (`bubble` set) the bubble filter is GetNearbyBalls' asymmetric rule
(Partition.cpp:344; exact_ballpark.full's _nearby): a candidate in another bubble is skipped unless it is global, so
every ball near a global ball in another bubble takes it as a candidate, while a global ball's own candidates are its
bubble's balls and the other bubbles' global balls; on the torch path (the gradient and the iterative method) and on
the kernel (the statics build, which a park made with global balls runs, so the default and composed builds compile to
what they compiled to before). A park made without the option holds exactly the tensors it held before.

**Static collidables** (COL-712): capsules and oriented boxes, in the gradient method, on the
torch path and on the kernel (its statics build). A park made with `statics` slots a world takes them at the load
(`load(worlds, statics)`, AddCapsule's and AddOrientedBox's arguments, `static_record`): each a record made once on the
host in the reference's double arithmetic (exact_ballpark.full's Capsule, OrientedBox, BoxShape, RectangleShape, AABB;
they never move) with its filing, InsertInBoxes' boxes at one level (a capsule's along its segment every 2r, a box's
around its centre, all 27 top boxes for one wider than a top box). Gradient walks a responder's static collidables
after its balls (Ballpark.cpp:2778-2783), in the order GetNearbyBalls collects them: the position of the static
collidable's first box in the ball's walk (`_static_order_keys`, the balls' sort key over its boxes), then its parent's
id, then its own (a box's std::set). Each is tested from the tick's start, its straight path predicted with mLastG
(CollideWithBall: the capsule's cylinder and caps, `_capsule_contact`; the box's AABB exclusion, its inside test with
the extrication normal, and its sides' triangles, `_box_contact`), and reacts as ReactToCollision does (a capsule
ignored when mLastG points out of it at an impact within the tick; the reflection off it, two exps, or the ejection by
the overlap plus a metre); the response replaces mLastC and mLastCollision except at an equal time, where the stronger
is kept (no later-impact test), and the static collidable's id joins the collision list (`_static_update`, the only
order-dependent step, in that order). With `statics` 0 (the default) every tensor and operation is as before (the
kernel's default build is the same machine code). Refused with them: the iterative method, the master's bubbles,
keys at the grid's edge, a capsule whose walk through the boxes never ends (Destiny's result undefined).

**Minis and force fields** (COL-713, MOV-737), in the gradient method, on the torch path and on the
kernel (the statics build, which a park with static slots, minis or fields runs). A park made with `minis=True` takes
local (negative) ids. `make_minis` makes a world's minis as a client's Destiny makes them for a fixed parent at the
load (Ball::AddMiniBall, PyAddMiniCapsule and AddMiniBox through the Python thunkers, then Ballpark::AddMiniball,
AddCapsule(-1, ...) and AddOrientedBox(-1, ...)): each takes the next local id (GetIDForCollisionObject(-1) on a
client: LOCAL_BALLS - 1, - 2, ... in the order of the calls); a miniball is a fixed, massive ball in mode MINIBALL at
the parent's position plus its offset, with the parent's mass, owned by the parent (`owner`); a minicapsule or a
minibox is a capsule or an oriented box (the `statics` slots above) with the parent as its parent. Within a box
GetNearbyBalls takes the balls in BallSortComparer's order (Ball.cpp:2528: two local ids by their owner first,
otherwise by id), which the candidates' sort follows through `sort_id` (each ball's rank in that order, made at the
load). A park made with `fields=True` holds each ball's harmonic, corporation and alliance (-1 at the load, AddBall's)
and takes SetBallHarmonic (`set_harmonic`: a force field is stopped and set to FIELD; clearing a field stops it); the
gradient's candidates leave out a ball in mode FIELD when the responder's harmonic is -2 (the pass), or its harmonic,
corporation or alliance (not -1) is the responder's, as GetNearbyBalls does in the collision searches
(Partition.cpp:344; the field's whole physical effect: the proximity sensors pass exclude_harmonics). A FIELD ball is
inert in the tick (no thrust and no damping: MOV-734), as the behaviours already treat every mode but GOTO and STOP.
Without them (the defaults) every tensor and operation is as before. Refused with them: the iterative method, the
master's bubbles.

Not here (the load takes no such input, or refuses it): modes other than GOTO, STOP, FIELD (SetBallHarmonic's) and
MINIBALL (a miniball's), missiles, warps, static collidables, minis and force fields in the iterative method or on the
master's view, their removal, the minis' making and removal when a parent is fixed or freed during a run (SetBallFree,
MOV-732), proximity sensors (a sensor's check re-files its ball: the layer's, over its CollideView), the bubbles
themselves (the layer's), moribund balls, local (negative) ids outside a park made with `minis`, balls whose filing
would clamp at the grid's edge or span the 27 top boxes; dynamical orientation (off, as Destiny's default). On the
server, with every ball in one live bubble, the collisions are the same.
"""

from __future__ import annotations

import math

import torch

from . import vec_exp
from .modes import FIELD, MINIBALL
from .ref import GOTO, STOP, STOP_DAMP, f32
from .space import (COLLISION_MAX_ITERATIONS, FUZZY_DELTA, LEVEL_GRID, LEVEL_WIDTH, LIGHT_BALL_RATIO, LOCAL_BALLS,
                    LOCATIONS_RESERVED, NUM_LEVELS, RESPONSE_SCALE, VELOCITY_LIMIT)

F64, I64, BOOL = torch.float64, torch.int64, torch.bool
HALF = tuple(0.5 * LEVEL_WIDTH[i] * LEVEL_GRID[i] for i in range(NUM_LEVELS))      # key_for's offset (one value)
CENTER = tuple(LEVEL_WIDTH[i] * LEVEL_GRID[i] * 0.5 for i in range(NUM_LEVELS))    # Partition::GetBox's centre
FAC = tuple(1.0 / LEVEL_WIDTH[i] for i in range(NUM_LEVELS))
CODE_BASE = 65                       # a post-order digit: 0..63 for a child box, 64 past the box's own level
CODE_SPAN = CODE_BASE ** (NUM_LEVELS - 1)
NOT_A_CANDIDATE = 1 << 62
SMALL_SORT = 16                      # _sorted by comparisons up to this many slots
INT64_MAX, INT64_MIN = 9223372036854775807, -9223372036854775808


def _float32(x: float) -> float:
    try:
        return f32(x)
    except OverflowError:
        return math.copysign(math.inf, x)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector3d's dot product, x first (Vector3d.h:49)."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _normalized(v: torch.Tensor) -> torch.Tensor:
    """Vector3d::Normalize: multiply by the reciprocal of the length; a zero vector stays zero."""
    n = torch.sqrt(_dot(v, v))
    zero = n == 0.0
    inv = 1.0 / torch.where(zero, torch.ones_like(n), n)
    return torch.where(zero[..., None], v, v * inv[..., None])


def _to_int64(x: torch.Tensor) -> torch.Tensor:
    """`int64_t(x)` as the compiled code does it on aarch64 (fcvtzs): toward zero, saturating, NaN to 0."""
    ok = (x >= -9223372036854775808.0) & (x < 9223372036854775808.0)
    out = torch.where(ok, x, torch.zeros_like(x)).to(I64)
    out = torch.where(x >= 9223372036854775808.0, torch.full_like(out, INT64_MAX), out)
    return torch.where(x < -9223372036854775808.0, torch.full_like(out, INT64_MIN), out)


# ============================================== static collidables: their record, made once on the host
# A capsule or an oriented box as AddCapsule and AddOrientedBox make it (exact_ballpark.full's Capsule, OrientedBox,
# BoxShape, RectangleShape, AABB; Capsule.cpp, OrientedBox.cpp, BoxShape.cpp), with its filing in the partition
# (Capsule::InsertInBoxes, OrientedBox::InsertInBoxes), computed once at the load in the reference's double arithmetic
# and order (a static collidable never moves); the tick's tests read these values on tensors. A record is G_SIZE
# doubles (the layout below) and I_SIZE integers, with its boxes' keys beside them.

STATIC_CAPSULE, STATIC_BOX = 0, 1
EJECT_MARGIN = 1.0                                  # Capsule.cpp:427, OrientedBox.cpp:106 (COL-712)
DBL_MAX = 1.7976931348623157e308
STATIC_TINY, STATIC_FLOOR = 1e-10, 1e-5             # BoxShape::CollideWithSphere's thresholds (BoxShape.cpp:200)
SIDE_FRONT, SIDE_BACK, SIDE_LEFT, SIDE_RIGHT, SIDE_TOP, SIDE_BOTTOM = range(6)
CONNECTED_SIDES = ((SIDE_BACK, SIDE_LEFT, SIDE_BOTTOM), (SIDE_BACK, SIDE_RIGHT, SIDE_BOTTOM),      # BoxShape.cpp:189
                   (SIDE_BACK, SIDE_LEFT, SIDE_TOP), (SIDE_FRONT, SIDE_LEFT, SIDE_BOTTOM),
                   (SIDE_BACK, SIDE_RIGHT, SIDE_TOP), (SIDE_FRONT, SIDE_LEFT, SIDE_TOP),
                   (SIDE_FRONT, SIDE_RIGHT, SIDE_BOTTOM), (SIDE_FRONT, SIDE_RIGHT, SIDE_TOP))
SIDE_MAPPING = ((SIDE_LEFT, SIDE_RIGHT), (SIDE_BOTTOM, SIDE_TOP), (SIDE_BACK, SIDE_FRONT))           # BoxShape.cpp:137
# the doubles of a record: a capsule's A, B, B - A and radius (a float); a box's centre, its AABB's low and high
# corners, its edge vectors lx, ly, lz, each normalised, half their lengths and their squared lengths, the corners A to
# H, each side's corners a, b, c, d, each side's normal ((b - a) x (c - a), normalised: the triangle (a, b, c)'s) and
# the normal of its triangle (a, c, d)
G_A, G_B, G_AB, G_RAD = 0, 3, 6, 9
G_CENTER, G_LO, G_HI = 10, 13, 16
G_AXES, G_UNIT, G_HALF, G_LENSQ = 19, 28, 37, 40
G_CORNERS, G_RECT, G_RNORM, G_TNORM = 43, 67, 139, 157
G_SIZE = 175
I_KIND, I_ID, I_PARENT, I_LEVEL, I_NBOX = range(5)      # the integers: kind, id, parent id, level, how many boxes
I_SIZE = 5


def _hadd(u, v) -> list:
    return [u[0] + v[0], u[1] + v[1], u[2] + v[2]]


def _hscale(u, c: float) -> list:
    return [u[0] * c, u[1] * c, u[2] * c]


def _static_level(radius: float) -> int:
    """Partition::GetBox(radius, pos)'s level (space.SpaceBallpark.level_for): the finest at least 3x the radius."""
    level = 0
    for i in range(NUM_LEVELS):
        if 3.0 * radius > LEVEL_WIDTH[i]:
            break
        level = i
    return level


def _static_key(level: int, pos) -> tuple:
    """GetBox(radius, pos)'s key at a level before the clamp (space.SpaceBallpark.key_for)."""
    from .space import _to_int64 as int64_of
    w = LEVEL_WIDTH[level]
    fac, half = 1.0 / w, 0.5 * w * LEVEL_GRID[level]
    return tuple(int64_of((pos[i] + half) * fac) for i in range(3))


def _static_bounds(level: int, key) -> tuple:
    """A box's low and high corners (space.SpaceBallpark._get_box, for a key inside the grid)."""
    w, n = LEVEL_WIDTH[level], LEVEL_GRID[level]
    center = w * n * 0.5
    return [key[i] * w - center for i in range(3)], [(key[i] + 1) * w - center for i in range(3)]


def _with_sides(key, mod) -> set:
    """The box and the adjacent boxes an InsertInBoxes reaches with these sides (-1, 0, +1 an axis)."""
    out = set()
    for c in range(8):
        bits = ((c >> 2) & 1, (c >> 1) & 1, c & 1)
        if all(mod[q] or not bits[q] for q in range(3)):
            out.add(tuple(key[q] + mod[q] * bits[q] for q in range(3)))
    return out


def static_record(s: dict) -> tuple:
    """One static collidable as the park keeps it: (doubles [G_SIZE], integers [I_SIZE], its boxes' keys, sorted).
    `s` is {"kind": "capsule", "id", "parent", "a", "b", "radius"} (AddCapsule's arguments: the radius a float) or
    {"kind": "box", "id", "parent", "corner", "lx", "ly", "lz"} (AddOrientedBox's), the id as the park would hold it
    (Ballpark::GetIDForCollisionObject's result). Its filing is InsertInBoxes' at its first insert: a capsule's boxes at
    its radius's level along the segment, sampled every 2r from A while the float length lasts, each with the adjacent
    boxes its radius reaches, none for a radius <= 0 (Capsule.cpp:449); a box's at half its AABB's longest width, with
    the adjacent boxes its AABB reaches, all 26 around it at the top level when its bounding diameter exceeds the top
    width (OrientedBox.cpp:127). Refused (ValueError): an unknown kind, a key at the grid's edge (where
    Partition::GetBox clamps), and a capsule whose walk through the boxes never ends (Capsule.cpp:499, 578: Destiny's
    result undefined)."""
    from .modes import _cross, _lensq, _length
    from .ref import _normalized
    from .space import _sub
    geo, ints = [0.0] * G_SIZE, [0] * I_SIZE
    ints[I_ID], ints[I_PARENT] = int(s["id"]), int(s["parent"])
    kind = s["kind"]
    if kind == "capsule":
        ints[I_KIND] = STATIC_CAPSULE
        a, b, radius = [float(x) for x in s["a"]], [float(x) for x in s["b"]], _float32(s["radius"])
        ab = _sub(b, a)
        geo[G_A:G_A + 3], geo[G_B:G_B + 3], geo[G_AB:G_AB + 3], geo[G_RAD] = a, b, ab, radius
        boxes: set = set()
        level = _static_level(radius)
        if radius > 0.0:                            # Capsule::InsertInBoxes (no volume: no box)
            interp = list(a)
            length = _float32(_length(ab))
            direction = [x * float(radius) * 2.0 for x in _normalized(ab)]
            step = _float32(radius * 2.0)
            while length >= 0.0:
                key = _static_key(level, interp)
                lo, hi = _static_bounds(level, key)
                mod = [0, 0, 0]
                for q in range(3):
                    if lo[q] + float(radius) - interp[q] > 0.0:
                        mod[q] = -1
                    elif hi[q] - float(radius) - interp[q] < 0.0:
                        mod[q] = 1
                boxes |= _with_sides(key, mod)
                interp = _hadd(interp, direction)
                shorter = _float32(length - step)
                if shorter == length:
                    raise ValueError("Capsule::InsertInBoxes never ends (twice the radius is below half a float ulp of "
                                     "the length left; Capsule.cpp:499, 578): Destiny's result is undefined")
                length = shorter
    elif kind == "box":
        ints[I_KIND] = STATIC_BOX
        A = [float(x) for x in s["corner"]]
        lx, ly, lz = ([float(x) for x in s[k]] for k in ("lx", "ly", "lz"))
        B, C, D = _hadd(A, lx), _hadd(A, ly), _hadd(A, lz)               # BoxShape::BoxShape (BoxShape.cpp:34)
        E, F, G = _hadd(B, ly), _hadd(C, lz), _hadd(B, lz)
        H = _hadd(E, lz)
        corners = (A, B, C, D, E, F, G, H)
        rects = [None] * 6
        rects[SIDE_FRONT], rects[SIDE_BACK], rects[SIDE_LEFT] = (H, F, D, G), (C, E, B, A), (F, C, A, D)
        rects[SIDE_RIGHT], rects[SIDE_TOP], rects[SIDE_BOTTOM] = (E, H, G, B), (C, F, H, E), (B, G, D, A)
        center = _hadd(A, _hscale(_sub(H, A), 1.0 / 2.0))                  # d_div(v, 2.0): v times 1 / 2
        low, high = list(A), list(A)                                        # AABB(A, A), then each other corner
        for v in corners[1:]:
            for q in range(3):
                if v[q] > high[q]:
                    high[q] = v[q]
                elif v[q] < low[q]:
                    low[q] = v[q]
        geo[G_CENTER:G_CENTER + 3], geo[G_LO:G_LO + 3], geo[G_HI:G_HI + 3] = center, low, high
        for i, axis in enumerate((lx, ly, lz)):
            geo[G_AXES + 3 * i:G_AXES + 3 * i + 3] = axis
            geo[G_UNIT + 3 * i:G_UNIT + 3 * i + 3] = _normalized(axis)
            geo[G_HALF + i], geo[G_LENSQ + i] = _length(axis) / 2.0, _lensq(axis)
        for i, v in enumerate(corners):
            geo[G_CORNERS + 3 * i:G_CORNERS + 3 * i + 3] = v
        for i, (ra, rb, rc, rd) in enumerate(rects):
            for j, v in enumerate((ra, rb, rc, rd)):
                geo[G_RECT + 12 * i + 3 * j:G_RECT + 12 * i + 3 * j + 3] = v
            geo[G_RNORM + 3 * i:G_RNORM + 3 * i + 3] = _normalized(_cross(_sub(rb, ra), _sub(rc, ra)))
            geo[G_TNORM + 3 * i:G_TNORM + 3 * i + 3] = _normalized(_cross(_sub(rc, ra), _sub(rd, ra)))
        dx, dy, dz = high[0] - low[0], high[1] - low[1], high[2] - low[2]
        inner = dz if dy < dz else dy                                       # AABB::LongestWidth: _fmax(dy, dz)
        r = (inner if dx < inner else dx) / 2.0                             # _fmax(dx, ..) / 2
        level = _static_level(r)
        key = _static_key(level, center)
        lo, hi = _static_bounds(level, key)
        mod = [0, 0, 0]
        for q in range(3):
            if lo[q] - low[q] > 0.0:
                mod[q] = -1
            elif hi[q] - high[q] < 0.0:
                mod[q] = 1
        bounding = _float32(_length(_sub(H, A)) / 2.0)
        if level == 0 and _float32(bounding * 2.0) > LEVEL_WIDTH[0]:      # the 26 top boxes around it as well
            boxes = {(key[0] + x, key[1] + y, key[2] + z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)}
        else:
            boxes = _with_sides(key, mod)
    else:
        raise ValueError(f"a static collidable's kind is 'capsule' or 'box', not {kind!r}")
    n = LEVEL_GRID[level]
    keys = sorted(boxes)
    if any(not 0 < k[q] < n - 1 for k in keys for q in range(3)):
        raise ValueError(f"static collidable {ints[I_ID]}: a box at the partition's edge (not in this module)")
    ints[I_LEVEL], ints[I_NBOX] = level, len(keys)
    return geo, ints, keys


# ======================================================= minis and force fields (COL-713, MOV-737)

FIELD_MODE, MINIBALL_MODE = FIELD, MINIBALL         # DSTBALLMODE's FIELD and MINIBALL (IDstConstants.h), from modes
HARMONIC_PASS = -2                                  # GetNearbyBalls: a ball of harmonic -2 passes every force field
NO_GROUP = -1                                       # no harmonic, corporation or alliance (AddBall's): never a match
MINI_KINDS = ("ball", "capsule", "box")


def make_minis(balls: list[dict], minis: list[dict], local: int = LOCAL_BALLS) -> tuple:
    """A world's minis as a client's Destiny makes them for a fixed parent (COL-713): Ball::
    AddMiniBall, PyAddMiniCapsule and AddMiniBox through the Python thunkers, each made at once when its parent is fixed
    (Ballpark::AddMiniball; AddCapsule(-1, ...), AddOrientedBox(-1, ...)), taking the next local id
    (GetIDForCollisionObject(-1) on a client: the counter, `local` before the calls, LOCAL_BALLS in a new park, less 1).
    `balls` are the world's balls as `VecCollidePark.load` takes them (the parents among them, by id) and `minis` the
    calls in order: {"kind": "ball", "parent", "pos" (the offset), "radius"}, {"kind": "capsule", "parent", "a", "b"
    (offsets), "radius"} or {"kind": "box", "parent", "corner" (an offset), "lx", "ly", "lz"}. As exact_ballpark.full's
    add_mini_*: a call for a missing parent does nothing; a minicapsule whose float radius is <= 0 is the thunker's
    ValueError (cleared by its caller: nothing made, no id); a free parent keeps the record unmade (made when it is
    fixed: SetBallFree, not in this module) and takes no id. A miniball: fixed, massive, non-interactive, mode MINIBALL,
    the parent's mass, the float radius (0 if negative), at the parent's position plus the offset, at rest, its maximum
    speed, agility and speed fraction 0 (AddMiniball's, no AddBall clamps), owned by the parent; a minicapsule's ends
    and a minibox's corner are the parent's position plus the offsets, the parent their parent. Returns (the miniballs
    as `load` takes them, the minicapsules and miniboxes as `load_statics` takes them, the counter after the calls)."""
    by_id = {int(b["id"]): b for b in balls}
    made_balls: list[dict] = []
    made_statics: list[dict] = []
    for m in minis:
        kind = m["kind"]
        if kind not in MINI_KINDS:
            raise ValueError(f"a mini's kind is 'ball', 'capsule' or 'box', not {kind!r}")
        p = by_id.get(int(m["parent"]))
        if p is None:                                   # add_mini_*: no such ball, nothing
            continue
        if kind == "capsule" and _float32(m["radius"]) <= 0.0:
            continue                                    # "Radius must be positive" (Ball.cpp:2156-2160)
        if bool(p["free"]):                             # kept by a free parent, unmade
            continue
        local -= 1                                      # GetIDForCollisionObject(-1): mLocalHiCnt
        pos = [float(c) for c in p["pos"]]
        pid = int(p["id"])
        if kind == "ball":
            off = [float(c) for c in m["pos"]]
            radius = _float32(m["radius"])
            mass = float(p["mass"])
            mass = 0.0 if mass < 0.0 else mass          # the parent's (AddBall's clamp), then AddMiniball's
            made_balls.append({"id": local, "owner": pid, "miniball": True, "mass": mass,
                               "radius": 0.0 if radius < 0.0 else radius, "max_vel": 0.0, "agility": 0.0,
                               "speed_fraction": 0.0, "pos": [pos[i] + off[i] for i in range(3)],
                               "vel": [0.0, 0.0, 0.0], "free": False, "massive": True})
        elif kind == "capsule":
            a, b = [float(c) for c in m["a"]], [float(c) for c in m["b"]]
            made_statics.append({"kind": "capsule", "id": local, "parent": pid, "a": [pos[i] + a[i] for i in range(3)],
                                 "b": [pos[i] + b[i] for i in range(3)], "radius": _float32(m["radius"])})
        else:
            corner = [float(c) for c in m["corner"]]
            made_statics.append({"kind": "box", "id": local, "parent": pid,
                                 "corner": [corner[i] + pos[i] for i in range(3)],
                                 "lx": [float(c) for c in m["lx"]], "ly": [float(c) for c in m["ly"]],
                                 "lz": [float(c) for c in m["lz"]]})
    return made_balls, made_statics, local


def ball_sort_key(bid: int, owner: int) -> tuple:
    """BallSortComparer (Ball.cpp:2528; exact_ballpark.full._ball_key): two local (negative) ids by their owner first,
    otherwise by id."""
    return (0, owner, bid) if bid < 0 else (1, 0, bid)


class VecCollidePark:
    """W worlds of N ball slots (a slot holds a ball or nothing); in each world the balls fill the first slots in
    ascending id, as Destiny walks them (with `rows`, the free balls first: the module's docstring)."""

    # GetNextCollisionFree's `continue` after an ignored contact skips the step's `b0, o0 = b1, o1`, so the next
    # sub-step starts where the ignored one did (as coded; Ballpark.cpp:2180)
    KEEP_START_AFTER_IGNORED = True

    def __init__(self, worlds: int, slots: int, friction: float = 1.0e6, tick_interval_ms: int = 1000,
                 iterative: bool = False, collisions: bool = True, contact_slots: int | None = None,
                 locations: int = 48, device="cpu", rows: int | None = None, impact_slots: int | None = None,
                 kernel: str = "torch", statics: int = 0, minis: bool = False, fields: bool = False,
                 global_balls: bool = False):
        self.W, self.N, self.dev = int(worlds), int(slots), torch.device(device)
        self.S = int(statics)       # static collidable slots a world; 0: none, every tensor as before
        if self.S < 0:
            raise ValueError(f"statics={statics}: not negative")
        self.minis = bool(minis)    # local (negative) ids, miniballs among them; False: as before
        self.fields = bool(fields)  # force fields: harmonic, corporation, alliance; False: as before
        self.global_balls = bool(global_balls)      # each ball's global flag; False: as before
        if bool(iterative) and (self.minis or self.fields):
            raise ValueError("minis and force fields in the iterative method are not in this module")
        self.R = self.N if rows is None else int(rows)     # the slots that may hold a free ball (the responders' rows)
        if not 1 <= self.R <= self.N:
            raise ValueError(f"rows={rows}: from 1 to the {self.N} slots")
        self.friction, self.dt = float(friction), tick_interval_ms * 0.001
        self.iterative = bool(iterative)            # g_useIterativeCollision (COL-701)
        self.collisions = bool(collisions)          # off: GOTO and STOP alone, as exact_ballpark.vec.VecBallpark
        self.contact_slots = contact_slots          # None: each tick loops over the most contacts any ball has
        self.impact_slots = impact_slots            # with contact_slots: the impact branch on at most this many pairs
        self.k_t = torch.tensor(self.friction, dtype=F64, device=self.dev)        # a device divisor
        self.neg_k_t = torch.tensor(-self.friction, dtype=F64, device=self.dev)
        self.current_time = 0
        t = lambda xs, dt=F64: torch.tensor(xs, dtype=dt, device=self.dev)
        self.width_t, self.half_t, self.center_t, self.fac_t = t(LEVEL_WIDTH), t(HALF), t(CENTER), t(FAC)
        self.grid_t = t(LEVEL_GRID, I64)
        self._levels = t(list(range(1, NUM_LEVELS)), I64)                     # the post-order code's levels
        self._powers = t([CODE_BASE ** (NUM_LEVELS - 1 - lv) for lv in range(1, NUM_LEVELS)], I64)
        self._power_list = [CODE_BASE ** (NUM_LEVELS - 1 - lv) for lv in range(1, NUM_LEVELS)]
        self._combos = [t([(c >> 2) & 1, (c >> 1) & 1, c & 1], I64) for c in range(8)]
        ar = torch.arange(self.N, device=self.dev)
        self._slot_lt = ar[None, :] < ar[:, None]                              # [i, j]: j < i (_sorted_small)
        W, N = self.W, self.N
        z = lambda *s: torch.zeros(W, N, *s, dtype=F64, device=self.dev)
        zb = lambda: torch.zeros(W, N, dtype=BOOL, device=self.dev)
        self.exists, self.free, self.massive, self.junk, self.cloaked = zb(), zb(), zb(), zb(), zb()
        self.id = torch.zeros(W, N, dtype=I64, device=self.dev)
        self.mode = torch.full((W, N), STOP, dtype=I64, device=self.dev)
        self.mass, self.radius, self.max_vel, self.agility, self.speed_fraction = z(), z(), z(), z(), z()
        self.time_factor, self.last_collision = z(), torch.full((W, N), -1.0, dtype=F64, device=self.dev)
        self.pos, self.vel, self.goto, self.last_g, self.last_c = z(3), z(3), z(3), z(3), z(3)
        self.old_pos, self.old_vel = z(3), z(3)
        self.level = torch.zeros(W, N, dtype=I64, device=self.dev)
        self.key = torch.zeros(W, N, 3, dtype=I64, device=self.dev)
        self.mod = torch.zeros(W, N, 3, dtype=I64, device=self.dev)
        R = self.R                                      # Ball::mCollisions, on the rows (only free balls collect)
        # (with static collidables one more entry a static slot: each appends its id at most once a tick)
        self.hits = torch.full((W, R, max(1, 3 * (N - 1)) + self.S), -1, dtype=I64, device=self.dev)
        self.hits_n = torch.zeros(W, R, dtype=I64, device=self.dev)
        # the impact record (the gradient's impacts at depth 0, in candidate order): the other's slot, s, closing speed
        self.imp_slot = torch.full((W, R, max(1, N - 1)), -1, dtype=I64, device=self.dev)
        self.imp_s = torch.zeros(W, R, max(1, N - 1), dtype=F64, device=self.dev)
        self.imp_closing = torch.zeros(W, R, max(1, N - 1), dtype=F64, device=self.dev)
        self.imp_n = torch.zeros(W, R, dtype=I64, device=self.dev)
        self.overflow = torch.zeros(W, dtype=BOOL, device=self.dev)     # a fixed contact_slots was too small
        self.impact_overflow = torch.zeros((), dtype=BOOL, device=self.dev)     # impact_slots was too small
        # the iterative method: each ball's collision path (CollisionBallProperties) as padded points, its vector's
        # capacity (kept from tick to tick, as Destiny keeps the object), and where Destiny's result is undefined
        self.L = int(locations)
        self.cap = torch.full((W, N), LOCATIONS_RESERVED, dtype=I64, device=self.dev)
        self.has_prop = torch.zeros(W, N, dtype=BOOL, device=self.dev)
        self.undefined = torch.zeros(W, dtype=BOOL, device=self.dev)
        self.loc_overflow = torch.zeros(W, dtype=BOOL, device=self.dev)
        self.stack_overflow = torch.zeros(W, dtype=BOOL, device=self.dev)
        self.passes = 0
        # the master's bubbles: [W, N] int64, each slot's bubble id as the partition layer keeps it
        # (exact_ballpark.vec_space's CollideView sets it and the layer writes it in place, so it is never rebound
        # here); None (the default) is a client's view, every candidate whatever its bubble, as before
        self.bubble = None
        # the static slots' tensors exist only in a park made with some, so a park without them holds
        # exactly the tensors it held before (snapshots and graph pins enumerate them)
        self.has_statics = False                # a host flag: no read of the device in the tick
        if self.S:
            self._empty_statics(1)              # none filled until a load gives some
        # as the static slots', these tensors exist only in a park made with them
        if self.minis:                          # mOwnerId (a miniball's parent, else 0) and BallSortComparer's rank
            self.owner = torch.zeros(W, N, dtype=I64, device=self.dev)
            self.sort_id = torch.arange(N, dtype=I64, device=self.dev).expand(W, N).clone()
        if self.fields:                         # SetBallHarmonic's values, AddBall's -1 until it is called
            self.harmonic = torch.full((W, N), NO_GROUP, dtype=I64, device=self.dev)
            self.corporation = torch.full((W, N), NO_GROUP, dtype=I64, device=self.dev)
            self.alliance = torch.full((W, N), NO_GROUP, dtype=I64, device=self.dev)
        if self.global_balls:                   # AddBall's isGlobal, SetBallGlobal's flag (in place)
            self.glob = torch.zeros(W, N, dtype=BOOL, device=self.dev)
        self.kernel = "torch"
        self.use_kernel(kernel)

    def use_kernel(self, kernel: str) -> None:
        """The tick by the torch path ("torch", the exactness reference) or on CUDA kernels ("cuda": the
        collision step as one kernel, the behaviours, the end of the tick and the filing one launch each; a CUDA device
        and the gradient method; the library built and its module loaded here, never during a CUDA graph's capture)."""
        if kernel not in ("torch", "cuda"):
            raise ValueError(f"kernel: 'torch' or 'cuda', not {kernel!r}")
        if kernel == "cuda":
            if self.dev.type != "cuda":
                raise ValueError(f"kernel='cuda' needs a CUDA device, not the {self.dev.type}")
            if self.iterative:
                raise ValueError("kernel='cuda' runs the gradient method; the iterative method is the torch path's")
            from . import vec_collide_cuda
            with torch.cuda.device(self.dev):
                vec_collide_cuda.load_library()
                if self.S or self.minis or self.fields or self.global_balls:
                    # the statics build as well
                    vec_collide_cuda.load_library(statics=True)
        self.kernel = kernel

    # ================================================================================================= loading

    def load(self, worlds: list[list[dict]], statics: list[list[dict]] | None = None) -> None:
        """Each world a list of balls (AddBall's arguments, `Ballpark::AddBall`, Ballpark.cpp:3303, as the Python
        thunker passes them): id, mass, radius, max_vel, agility, speed_fraction, pos, vel, free, massive, and optionally
        junk and cloaked. AddBall's float32 properties and clamps, the time factor and the first filing are computed on
        the host; every ball starts in STOP with its goto point at the origin. `statics` (a park made
        with `statics` slots): each world's static collidables, as `load_statics` takes them. In a park made with
        `minis` an id may be local (negative; never 0 or -1, which GetIDForCollisionObject never
        gives), a ball may name its `owner`, and a ball marked `miniball` (make_minis') is AddMiniball's: fixed, in mode
        MINIBALL, its agility kept (no AddBall clamp); each ball's rank in BallSortComparer's order is its `sort_id`. In
        a park made with `fields` every ball's harmonic, corporation and alliance start at -1 (AddBall's). In a park
        made with `global_balls` a ball's `global` is AddBall's isGlobal (False when absent)."""
        if len(worlds) != self.W:
            raise ValueError(f"{len(worlds)} worlds for a park of {self.W}")
        if statics is not None and len(statics) != self.W:
            raise ValueError(f"{len(statics)} worlds of static collidables for a park of {self.W}")
        cols = {k: [[0.0] * self.N for _ in range(self.W)] for k in ("mass", "radius", "max_vel", "agility", "sf",
                                                                      "tf")}
        vec = {k: [[[0.0] * 3 for _ in range(self.N)] for _ in range(self.W)] for k in ("pos", "vel", "old")}
        flags = {k: [[False] * self.N for _ in range(self.W)] for k in ("exists", "free", "massive", "junk",
                                                                        "cloaked")}
        ids = [[0] * self.N for _ in range(self.W)]
        owners = [[0] * self.N for _ in range(self.W)]          # (a park made with minis)
        minis = [[False] * self.N for _ in range(self.W)]
        ranks = [list(range(self.N)) for _ in range(self.W)]
        globs = [[False] * self.N for _ in range(self.W)]       # (a park made with global_balls)
        for w, balls in enumerate(worlds):
            if len(balls) > self.N:
                raise ValueError(f"world {w} has {len(balls)} balls for {self.N} slots")
            seen = [int(b["id"]) for b in balls]
            if not self.global_balls and any(bool(b.get("global", False)) for b in balls):
                raise ValueError(f"world {w}: a global ball needs a park made with global_balls=True")
            if not self.minis:
                if any("owner" in b or "miniball" in b for b in balls):
                    raise ValueError(f"world {w}: owners and miniballs need a park made with minis=True")
                if len(set(seen)) != len(seen) or min(seen, default=1) <= 0:
                    raise ValueError(f"world {w}: ball ids must be positive and distinct")
            elif len(set(seen)) != len(seen) or any(i in (0, -1) for i in seen):
                raise ValueError(f"world {w}: ball ids must be distinct, and neither 0 nor -1")
            if self.R == self.N:                                          # ascending id
                order = sorted(balls, key=lambda b: int(b["id"]))
            else:                                                         # the free balls first, each part by id
                order = sorted(balls, key=lambda b: (not bool(b["free"]), int(b["id"])))
                if sum(1 for b in balls if bool(b["free"])) > self.R:
                    raise ValueError(f"world {w} has more free balls than the park's {self.R} rows")
            for j, b in enumerate(order):
                radius, max_vel = _float32(b["radius"]), _float32(b["max_vel"])
                agility, sf = _float32(b.get("agility", 0.0)), _float32(b.get("speed_fraction", 0.0))
                mass = float(b["mass"])
                mass = 0.0 if mass < 0.0 else mass
                mini = bool(b.get("miniball", False))   # AddMiniball's ball, no AddBall clamp
                if not mini:
                    agility = 1.0 if agility <= 0.0 else agility
                free = bool(b["free"])
                if mini and free:
                    raise ValueError(f"world {w}, ball {b['id']}: a miniball is fixed")
                if free and not mass * agility > 0.0:
                    raise ValueError(f"world {w}, ball {b['id']}: a free ball needs a positive mass x agility")
                m = mass * agility
                cols["mass"][w][j], cols["radius"][w][j] = mass, 0.0 if radius < 0.0 else radius
                cols["max_vel"][w][j], cols["agility"][w][j] = 0.0 if max_vel < 0.0 else max_vel, agility
                cols["sf"][w][j] = 0.0 if sf < 0.0 else sf
                cols["tf"][w][j] = math.exp(-self.friction * self.dt / m) if m > 0 else 0.0
                p, v = [float(c) for c in b["pos"]], [float(c) for c in b.get("vel", (0.0, 0.0, 0.0))]
                vec["pos"][w][j], vec["vel"][w][j] = p, v
                vec["old"][w][j] = [p[i] - v[i] * self.dt for i in range(3)]
                flags["exists"][w][j], flags["free"][w][j] = True, free
                flags["massive"][w][j], flags["junk"][w][j] = bool(b["massive"]), bool(b.get("junk", False))
                flags["cloaked"][w][j] = bool(int(b.get("cloaked", 0)) & 0xFF)
                ids[w][j] = int(b["id"])
                owners[w][j], minis[w][j] = int(b.get("owner", 0)), mini
                globs[w][j] = bool(b.get("global", False))
            if self.minis:                              # BallSortComparer's rank of each ball (empty slots after)
                key = lambda j: ball_sort_key(ids[w][j], owners[w][j])      # noqa: E731
                for rank, j in enumerate(sorted(range(len(order)), key=key)):
                    ranks[w][j] = rank
                for j in range(len(order), self.N):
                    ranks[w][j] = j
        t = lambda xs, dt=F64: torch.tensor(xs, dtype=dt, device=self.dev)
        self.mass, self.radius, self.max_vel = t(cols["mass"]), t(cols["radius"]), t(cols["max_vel"])
        self.agility, self.speed_fraction, self.time_factor = t(cols["agility"]), t(cols["sf"]), t(cols["tf"])
        self.pos, self.vel, self.old_pos = t(vec["pos"]), t(vec["vel"]), t(vec["old"])
        self.old_vel = self.vel.clone()
        for k in flags:
            setattr(self, k, t(flags[k], BOOL))
        self.id = t(ids, I64)
        self.mode = torch.full((self.W, self.N), STOP, dtype=I64, device=self.dev)
        if self.minis:                                  # the miniballs' mode, owners and ranks
            self.mode = torch.where(t(minis, BOOL), torch.full_like(self.mode, MINIBALL_MODE), self.mode)
            self.owner, self.sort_id = t(owners, I64), t(ranks, I64)
        if self.fields:                                 # AddBall's -1
            for name in ("harmonic", "corporation", "alliance"):
                setattr(self, name, torch.full((self.W, self.N), NO_GROUP, dtype=I64, device=self.dev))
        if self.global_balls:                           # in place (a view of the layer reads it)
            self.glob.copy_(t(globs, BOOL))
        z3 = torch.zeros(self.W, self.N, 3, dtype=F64, device=self.dev)
        self.goto, self.last_g, self.last_c = z3.clone(), z3.clone(), z3.clone()
        self.last_collision = torch.full((self.W, self.N), -1.0, dtype=F64, device=self.dev)
        self.hits.fill_(-1)
        self.hits_n.zero_()
        self.imp_slot.fill_(-1)
        self.imp_s.zero_()
        self.imp_closing.zero_()
        self.imp_n.zero_()
        self.overflow.zero_()
        self.impact_overflow.zero_()
        self.cap.fill_(LOCATIONS_RESERVED)
        self.has_prop.zero_()
        self.undefined.zero_()
        self.loc_overflow.zero_()
        self.stack_overflow.zero_()
        self.level, self.key, self.mod = self._filing(self.exists)
        self._check_filing(self.exists)
        if statics is not None:
            self.load_statics(statics)
        elif self.S:
            self._empty_statics(1)

    # ================================================================================================= commands

    def goto_point(self, mask: torch.Tensor, point: torch.Tensor) -> None:
        """GotoPoint (Ballpark.cpp:4529) for the balls in `mask` [W, N]: ignored for a non-finite point; else STOP,
        the point, a speed fraction of 1 if it was 0, GOTO."""
        ok = mask & self.exists & torch.isfinite(point).all(-1)
        self.goto = torch.where(ok[..., None], point.to(F64), self.goto)
        sf0 = ok & (self.speed_fraction == 0.0)
        self.speed_fraction = torch.where(sf0, torch.ones_like(self.speed_fraction), self.speed_fraction)
        self.mode = torch.where(ok, torch.full_like(self.mode, GOTO), self.mode)

    def goto_direction(self, mask: torch.Tensor, direction: torch.Tensor) -> None:
        """GotoDirection (Ballpark.cpp:4483): a point 1e17 m along the unit direction."""
        ok = mask & torch.isfinite(direction).all(-1)
        self.goto_point(ok, self.pos + _normalized(direction.to(F64)) * 1.0e17)

    def stop(self, mask: torch.Tensor) -> None:
        """Stop (Ballpark.cpp:4556): GOTO and STOP have no follow fields to clear."""
        self.mode = torch.where(mask & self.exists, torch.full_like(self.mode, STOP), self.mode)

    def set_speed_fraction(self, mask: torch.Tensor, s: torch.Tensor) -> None:
        """SetSpeedFraction: a float, finite only, clamped to [0, 1]."""
        s = s.to(F64).to(torch.float32).to(F64)
        ok = mask & self.exists & torch.isfinite(s)
        s = torch.where(s < 0.0, torch.zeros_like(s), torch.where(s > 1.0, torch.ones_like(s), s))
        self.speed_fraction = torch.where(ok, s, self.speed_fraction)

    # the setters the master's server layer applies (exact_ballpark.vec_server_space's
    # SpaceCollideAdapter): as exact_ballpark.space's, each on the balls in `mask` [W, N]; the re-filing that Destiny
    # does after SetBallRadius and SetMaxSpeed (when accepted) and SetBallPosition is the partition layer's (its
    # REFILE), which the park's filing then follows (exact_ballpark.vec_space.CollideView.sync_filing)

    def _time_factor(self) -> torch.Tensor:
        """SetBallTimeFactor (Ballpark.cpp:4968): 0 if mass x agility <= 0, else exp(-k dt / m) (glibc's exp,
        exact_ballpark.vec_exp; a NaN m gives NaN), as ref.py's _refresh_time_factor."""
        m = self.mass * self.agility
        return torch.where(m <= 0.0, torch.zeros_like(m), vec_exp.exp(torch.full_like(m, -self.friction * self.dt) / m))

    def set_mass(self, mask: torch.Tensor, mass: torch.Tensor) -> None:
        """SetBallMass: ignored unless positive (a NaN is stored, as coded); the time factor refreshed."""
        m = mass.to(F64)
        ok = mask & self.exists & ~(m <= 0.0)
        self.mass = torch.where(ok, m, self.mass)
        self.time_factor = torch.where(ok, self._time_factor(), self.time_factor)

    def set_agility(self, mask: torch.Tensor, agility: torch.Tensor) -> None:
        """SetBallAgility: the float ignored unless positive; the time factor refreshed."""
        a = agility.to(F64).to(torch.float32).to(F64)
        ok = mask & self.exists & ~(a <= 0.0)
        self.agility = torch.where(ok, a, self.agility)
        self.time_factor = torch.where(ok, self._time_factor(), self.time_factor)

    def set_radius(self, mask: torch.Tensor, r: torch.Tensor) -> None:
        """SetBallRadius: the float ignored if negative (then re-filed by the partition layer)."""
        r = r.to(F64).to(torch.float32).to(F64)
        ok = mask & self.exists & ~(r < 0.0)
        self.radius = torch.where(ok, r, self.radius)

    def set_max_velocity(self, mask: torch.Tensor, v: torch.Tensor) -> None:
        """SetMaxSpeed: the float ignored if negative (then re-filed by the partition layer; no time-factor
        refresh)."""
        v = v.to(F64).to(torch.float32).to(F64)
        ok = mask & self.exists & ~(v < 0.0)
        self.max_vel = torch.where(ok, v, self.max_vel)

    def set_velocity(self, mask: torch.Tensor, v: torch.Tensor) -> None:
        """SetBallVelocity: the new and the old velocity (no orientation here)."""
        ok = (mask & self.exists)[..., None]
        v = v.to(F64)
        self.vel = torch.where(ok, v, self.vel)
        self.old_vel = torch.where(ok, v, self.old_vel)

    def set_global(self, mask: torch.Tensor, flag) -> None:
        """SetBallGlobal (exact_ballpark.space's set_global; Destiny's park applies it at once) for the
        balls in `mask` [W, N], in a park made with `global_balls`: the global flag set to `flag` (a bool or a bool
        tensor [W, N]), in place."""
        if not self.global_balls:
            raise ValueError("SetBallGlobal needs a park made with global_balls=True")
        flag = torch.as_tensor(flag, dtype=BOOL, device=self.dev).expand(self.W, self.N)
        self.glob.copy_(torch.where(mask.to(BOOL) & self.exists, flag, self.glob))

    def set_harmonic(self, mask: torch.Tensor, harmonic: torch.Tensor, corporation: torch.Tensor,
                     alliance: torch.Tensor, is_field: torch.Tensor) -> None:
        """SetBallHarmonic (MOV-737; exact_ballpark.space's set_harmonic) for the balls in `mask`
        [W, N], in a park made with `fields`: the harmonic, corporation and alliance (integers [W, N]) set; where
        `is_field` [W, N] the ball is stopped and set to FIELD (a force field); otherwise a FIELD ball is stopped (Stop
        has no follow fields to clear in GOTO, STOP, FIELD or MINIBALL)."""
        if not self.fields:
            raise ValueError("SetBallHarmonic needs a park made with fields=True")
        ok = mask & self.exists
        self.harmonic = torch.where(ok, harmonic.to(I64), self.harmonic)
        self.corporation = torch.where(ok, corporation.to(I64), self.corporation)
        self.alliance = torch.where(ok, alliance.to(I64), self.alliance)
        on = ok & is_field
        off = ok & ~is_field & (self.mode == FIELD_MODE)
        self.mode = torch.where(on, torch.full_like(self.mode, FIELD_MODE),
                                torch.where(off, torch.full_like(self.mode, STOP), self.mode))

    # ================================================================================================ partition

    def _inflated_radius(self) -> torch.Tensor:
        """Ball::GetInflatedRadius (Ball.cpp:838; SEN-706): a free ball's radius plus a tick at full thrust."""
        m = self.mass * self.agility
        a = (self.friction * self.speed_fraction) * self.max_vel / m
        v = torch.sqrt(_dot(self.vel, self.vel))
        ma = m * a
        mps = (ma - (ma - v * self.friction) * self.time_factor) / self.k_t
        return torch.where(self.free, self.radius + self.dt * mps, self.radius)

    def _filing(self, which: torch.Tensor) -> tuple:
        """Ball::InsertInBoxes's outcome for the balls in `which` (the others keep theirs): the level, the key of the
        box holding the centre, and the side (-1, 0, +1) on each axis where the inflated sphere reaches the next box."""
        if self.kernel == "cuda":                                           # one launch, the same bits
            from .vec_collide_cuda import filing
            return filing(self, which)
        infl = self._inflated_radius()
        broke = (3.0 * infl)[..., None] > self.width_t                      # level_for's break test, per level
        level = ((~broke).sum(-1) - 1).clamp(min=0)
        u = (self.pos + self.half_t[level][..., None]) * self.fac_t[level][..., None]
        key = _to_int64(u)
        n = self.grid_t[level][..., None]
        key = torch.where(key < 0, torch.zeros_like(key), torch.where(key >= n, n - 1, key))
        w, c = self.width_t[level][..., None], self.center_t[level][..., None]
        lo = key.to(F64) * w - c
        hi = (key + 1).to(F64) * w - c
        down = (lo + infl[..., None]) - self.pos > 0.0
        up = (hi - infl[..., None]) - self.pos < 0.0
        mod = torch.where(down, -torch.ones_like(key), torch.where(up, torch.ones_like(key), torch.zeros_like(key)))
        sel = which[..., None]
        return (torch.where(which, level, self.level), torch.where(sel, key, self.key),
                torch.where(sel, mod, self.mod))

    def _check_filing(self, which: torch.Tensor) -> None:
        """Refuse what this module leaves out of the partition: a key clamped at the grid's edge (its own or an adjacent
        box's) and a ball spanning the 27 top boxes."""
        n = self.grid_t[self.level][..., None]
        side = self.key + self.mod
        bad = which[..., None] & ((self.key <= 0) | (self.key >= n - 1) | (side < 0) | (side >= n))
        huge = which & (self.level == 0) & (self.radius * 2 > LEVEL_WIDTH[0])
        if bool(bad.any()) or bool(huge.any()):
            raise ValueError("a ball at the partition's edge or wider than a top box: not in this module")

    def _related(self) -> torch.Tensor:
        """Whether a box of the other ball (column) is one of the ball's (row) boxes, an ancestor of one or below one
        [W, R, N]. A ball's boxes are a product of per-axis options ({key} or {key, key + mod}) and a box's ancestor is
        its key shifted right by 2 per level, so the test is per axis: some option of the finer ball, shifted to the
        coarser level, equals some option of the coarser."""
        R = self.R
        Lb, Ln = self.level[:, :R, None], self.level[:, None, :]
        Kb, Kn = self.key[:, :R, None, :], self.key[:, None, :, :]
        Mb, Mn = self.mod[:, :R, None, :], self.mod[:, None, :, :]
        lo_b, hi_b, lo_n, hi_n = Kb + Mb.clamp(max=0), Kb + Mb.clamp(min=0), Kn + Mn.clamp(max=0), Kn + Mn.clamp(min=0)
        s = (2 * (Lb - Ln)).abs()[..., None]
        coarse = (Ln <= Lb)[..., None]
        x_lo, x_hi = torch.where(coarse, lo_b, lo_n) >> s, torch.where(coarse, hi_b, hi_n) >> s
        y_lo, y_hi = torch.where(coarse, lo_n, lo_b), torch.where(coarse, hi_n, hi_b)
        return ((x_lo == y_lo) | (x_lo == y_hi) | (x_hi == y_lo) | (x_hi == y_hi)).all(-1)

    def _order_keys(self, pair: torch.Tensor) -> torch.Tensor:
        """The position of each other ball's first box in the ball's walk (Partition::GetNearbyBalls), as a sort key
        [W, R, N] (then the other's id): (segment x 3 + part) x 65^7 + sub, where the segment is the index of the first
        of the ball's boxes (sorted by key) the other's box relates to, the part 0 for an ancestor (sub: levels up),
        1 below (sub: the post-order code) and 2 for the box itself; NOT_A_CANDIDATE outside `pair`."""
        R = self.R
        Lb, Ln = self.level[:, :R, None], self.level[:, None, :]
        Kb, Kn = self.key[:, :R, None, :], self.key[:, None, :, :]
        Mb, Mn = self.mod[:, :R, None, :], self.mod[:, None, :, :]
        cnt_b = 1 + (Mb != 0).to(I64)                                        # the ball's boxes per axis: 1 or 2
        lo_b = Kb + Mb.clamp(max=0)                                          # its options per axis, ascending
        hi_b = Kb + Mb.clamp(min=0)
        # an ancestor or the same box: the first of the ball's boxes whose ancestor at Ln holds the other ball
        s_up = (2 * (Lb - Ln)).clamp(min=0)[..., None]

        def holds(v):                                                        # v (at Ln) is one of the other's boxes
            return (v == Kn) | ((Mn != 0) & (v == Kn + Mn))
        a0 = holds(lo_b >> s_up)
        idx = torch.where(a0, torch.zeros_like(Kb), torch.ones_like(Kb))
        seg_up = (idx[..., 0] * cnt_b[..., 1] + idx[..., 1]) * cnt_b[..., 2] + idx[..., 2]
        # below: each of the other ball's boxes (up to 2 x 2 x 2), its ancestor at Lb if that is one of the ball's
        # boxes, and its post-order code there (path digits from Lb + 1, 64 past its own level, which sorts last)
        s_dn = (2 * (Ln - Lb)).clamp(min=0)
        lvl = self._levels
        shift = (2 * (Ln[..., None] - lvl)).clamp(min=0)                    # [W, N, N, 7]
        past, above = lvl > Ln[..., None], lvl <= Lb[..., None]
        best = torch.full(Lb.shape, NOT_A_CANDIDATE, dtype=I64, device=self.dev)
        cnt_n = 1 + (Mn != 0).to(I64)
        for bit in self._combos:
            valid = ((bit == 0) | (cnt_n == 2)).all(-1)
            box = Kn + Mn * bit                                              # one of the other's boxes (level Ln)
            anc = box >> s_dn[..., None]
            in0, in1 = anc == lo_b, (anc == hi_b) & (cnt_b == 2)
            own = (in0 | in1).all(-1) & valid
            j = torch.where(in0, torch.zeros_like(anc), torch.ones_like(anc))
            seg = (j[..., 0] * cnt_b[..., 1] + j[..., 1]) * cnt_b[..., 2] + j[..., 2]
            code = None                             # the post-order code level by level (an exact integer sum:
            for li, power in enumerate(self._power_list):   # no [W, R, N, 7, 3] temporary)
                d = (box >> shift[..., li, None]) & 3
                digit = d[..., 0] * 16 + d[..., 1] * 4 + d[..., 2]
                digit = torch.where(past[..., li], torch.full_like(digit, CODE_BASE - 1), digit)
                digit = torch.where(above[..., li], torch.zeros_like(digit), digit)
                code = digit * power if code is None else code + digit * power
            key = (seg * 3 + 1) * CODE_SPAN + code
            best = torch.where(own & (key < best), key, best)
        up_key = (seg_up * 3) * CODE_SPAN + (Lb - Ln)
        same_key = (seg_up * 3 + 2) * CODE_SPAN
        order = torch.where(Ln < Lb, up_key, torch.where(Ln == Lb, same_key, best))
        return torch.where(pair, order, torch.full_like(order, NOT_A_CANDIDATE))

    def _sorted(self, active: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        """The permutation [W, R, N] putting each ball's active candidates first, in GetNearbyBalls' order (the box's
        position, then id; in a park made with `minis`, BallSortComparer's rank)."""
        order = torch.where(active, order, torch.full_like(order, NOT_A_CANDIDATE))
        idn = self.id[:, None, :].expand_as(order)
        if self.minis:                              # BallSortComparer's order (local ids by owner)
            idn = self.sort_id[:, None, :].expand_as(order)
        if self.N <= SMALL_SORT:                    # by ranks: the same permutation as the two stable sorts below
            return self._sorted_small(order, idn)
        perm = torch.sort(idn, dim=-1, stable=True).indices
        return torch.gather(perm, -1, torch.sort(torch.gather(order, -1, perm), dim=-1, stable=True).indices)

    def _sorted_small(self, order: torch.Tensor, idn: torch.Tensor) -> torch.Tensor:
        """_sorted's permutation for few slots, by comparisons (a segmented sort of a few elements a row
        costs a radix sort's kernels over the batch): the two stable sorts order a row lexicographically by (order key,
        id, slot), so each slot's rank is the number of slots before it in that order, and the permutation puts each
        slot at its rank; integers only."""
        N = order.shape[-1]
        o_i, o_j = order[..., :, None], order[..., None, :]
        d_i, d_j = idn[..., :, None], idn[..., None, :]
        j_lt_i = self._slot_lt                                                # [N, N]: j < i
        before = (o_j < o_i) | ((o_j == o_i) & ((d_j < d_i) | ((d_j == d_i) & j_lt_i)))
        rank = before.to(I64).sum(-1)                                         # each slot's place
        at = rank[..., None, :] == torch.arange(N, device=self.dev)[:, None]  # [.., place, slot]
        return (at.to(I64) * torch.arange(N, device=self.dev)).sum(-1)

    # ============================================================================================== integration

    def _integrate_dt(self, p, v, a, m, tf):
        """Ballpark::Integrate (Ballpark.cpp:751) over one tick, with the cached decay factor (MOV-701 to MOV-704)."""
        k, t = self.friction, self.dt
        k2 = k * k
        ook2, ook = 1.0 / k2, 1.0 / k
        mm, tt = m[..., None], tf[..., None]
        exact = (mm * (mm * a * (tt - 1.0) + k * (a * t + v - tt * v)) + p * k2) * ook2
        taylor = (mm * (a * (-k * t) + k * (a * t + v * k * (1.0 / mm) * t)) + p * k2) * ook2
        pn = torch.where((k < 1e-10 * m * t)[..., None], taylor, exact)
        return pn, (mm * a - (mm * a - v * k) * tt) * ook

    def _decay(self, m, t) -> torch.Tensor:
        """The argument of Integrate's exp(-k/m t) for a time t that is not the tick."""
        return (self.neg_k_t / m) * t

    def _integrate_t(self, p, v, a, m, tf, t, tf_t):
        """Ballpark::Integrate over a time t [..]: nothing for t = 0, the cached factor for t = dt, else exp's value
        tf_t (vec_exp.exp of `_decay(m, t)`)."""
        k = self.friction
        k2 = k * k
        ook2, ook = 1.0 / k2, 1.0 / k
        f = torch.where(t != self.dt, tf_t, tf)
        mm, ff, tt = m[..., None], f[..., None], t[..., None]
        exact = (mm * (mm * a * (ff - 1.0) + k * (a * tt + v - ff * v)) + p * k2) * ook2
        taylor = (mm * (a * (-k * tt) + k * (a * tt + v * k * (1.0 / mm) * tt)) + p * k2) * ook2
        pn = torch.where((k < 1e-10 * m * t)[..., None], taylor, exact)
        vn = (mm * a - (mm * a - v * k) * ff) * ook
        zero = (t == 0.0)[..., None]
        return torch.where(zero, p, pn), torch.where(zero, v, vn)

    @staticmethod
    def _collide_two_spheres(p0, p1, q0, q1, cr) -> torch.Tensor:
        """CollideTwoSpheres (Collision.cpp:108; COL-705): 0 if already touching, else the earlier root in [0, 1],
        else -1 (Quadratic's linear branch is dead code, Collision.cpp:146-156)."""
        p0q0 = p0 - q0
        p0q0_2 = _dot(p0q0, p0q0)
        cr2 = cr * cr
        dpq = (p1 - p0) - (q1 - q0)
        a = _dot(dpq, dpq)
        b = 2.0 * _dot(p0q0, dpq)
        c = p0q0_2 - cr2
        det = b * b - 4.0 * a * c
        root = torch.sqrt(torch.where(det < 0.0, torch.zeros_like(det), det))
        s2 = ((-b - root) * 0.5) / a
        s = torch.where((s2 >= 0.0) & (s2 <= 1.0), s2, torch.full_like(s2, -1.0))
        s = torch.where((det < 0.0) | ((a == 0.0) & (b == 0.0)), torch.full_like(s, -1.0), s)
        return torch.where(p0q0_2 <= cr2, torch.zeros_like(s), s)

    # ================================================================================================= the tick

    def _behaviour(self, live: torch.Tensor) -> None:
        """Phase 1 (EvolveBehaviorForBall, Ballpark.cpp:789) for the free balls: mLastG from GOTO's thrust (as
        exact_ballpark.vec), STOP's vertical damping, mLastC and the collision list cleared."""
        if self.kernel == "cuda":                                           # one launch, the same bits
            from .vec_collide_cuda import behaviour
            behaviour(self, live)
            return
        goto, stop = live & (self.mode == GOTO), live & (self.mode == STOP)
        self.last_g = torch.where(goto[..., None], self._goto_thrust(), torch.where(live[..., None],
                                                                                   torch.zeros_like(self.pos),
                                                                                   self.last_g))
        vy = self.vel[..., 1]
        vel = self.vel.clone()
        vel[..., 1] = torch.where(stop, (vy - 0.07 * vy) * STOP_DAMP, vy)
        self.vel = vel
        self.last_c = torch.where(live[..., None], torch.zeros_like(self.last_c), self.last_c)
        rl = live[:, :self.R]                                               # the collision lists are the rows'
        self.hits = torch.where(rl[..., None], torch.full_like(self.hits, -1), self.hits)
        self.hits_n = torch.where(rl, torch.zeros_like(self.hits_n), self.hits_n)
        self.imp_slot = torch.where(rl[..., None], torch.full_like(self.imp_slot, -1), self.imp_slot)
        self.imp_n = torch.where(rl, torch.zeros_like(self.imp_n), self.imp_n)

    def _goto_thrust(self) -> torch.Tensor:
        """GotoThrust (Ballpark.cpp:1397; MOV-705 to MOV-707), as exact_ballpark.vec.VecBallpark._goto_thrust."""
        a = self.goto - self.pos
        length2 = a[..., 0] * a[..., 0] + a[..., 1] * a[..., 1] + a[..., 2] * a[..., 2]
        dist = self.speed_fraction * self.max_vel * self.dt
        max_thrust = self.friction * self.speed_fraction * self.max_vel / (self.mass * self.agility)
        norm = torch.sqrt(length2)
        inv = torch.where(norm == 0.0, torch.ones_like(norm), 1.0 / torch.where(norm == 0.0, torch.ones_like(norm),
                                                                                  norm))
        unit = torch.where((norm == 0.0)[..., None], a, a * inv[..., None])
        dd = dist * dist
        coff = length2 / torch.where(dd == 0.0, torch.ones_like(dd), dd)
        scale = torch.where(length2 < dd, max_thrust * coff * coff, max_thrust)
        return unit * scale[..., None]

    def evolve(self, active: torch.Tensor | None = None, skip: torch.Tensor | None = None) -> None:
        """One tick (TIME-706, TIME-707) in every world, or in the worlds where `active` [W] (the others keep their
        state bit for bit); the balls in `skip` [W, N] (the master's dead bubbles, TIME-714) are left out of the whole tick as Destiny's Evolve leaves a ball whose
        mInDeadBubble is set (no behaviour, response, end state, commit or re-filing; they stay candidates of the
        others, which the master's bubble filter, `bubble`, keeps from meeting them)."""
        if (self.minis or self.fields) and self.bubble is not None:
            raise ValueError("minis and force fields on the master's view (bubble set) are not in this module")
        live = self.exists & self.free
        if active is not None:
            live = live & active[:, None]
        if skip is not None:
            live = live & ~skip
        self._behaviour(live)                                               # (1)
        if self.collisions and self.iterative:                              # (2)
            self._iterative(live)
        if self.collisions and not self.iterative:                          # (3) the gradient for massive balls
            responder = live & self.massive
            self.last_collision = torch.where(responder, torch.full_like(self.last_collision, -1.0),
                                              self.last_collision)
            if self.kernel == "cuda":                                       # one kernel, the same bits
                from .vec_collide_cuda import gradient
                gradient(self, responder)                                   # (with statics, the statics build's)
            else:
                self._gradient(responder)
                if self.S:                          # then the static collidables, in the walk's order
                    self._gradient_statics(responder)
        if self.kernel == "cuda":                                           # the end states, (4): one launch
            from .vec_collide_cuda import commit
            commit(self, live)
            self.current_time += 1
            return
        m = self.mass * self.agility
        if self.collisions and self.iterative:
            pn, vn = self._iterative_end(live)
        elif self.collisions:
            pn, vn = self._integrate_dt(self.pos, self.vel, self.last_g + self.last_c, m, self.time_factor)
        else:                                                               # VecBallpark's tick, unchanged
            pn, vn = self._integrate_dt(self.pos, self.vel, self.last_g, m, self.time_factor)
        old = self.pos                                                      # (4) commit, then re-file
        sel = live[..., None]
        self.old_pos = torch.where(sel, self.pos, self.old_pos)
        self.old_vel = torch.where(sel, self.vel, self.old_vel)
        self.pos = torch.where(sel, pn, self.pos)
        self.vel = torch.where(sel, vn, self.vel)
        d = self.pos - old
        moved = live & ~(_dot(d, d) < 0.1)
        self.level, self.key, self.mod = self._filing(moved)
        self.current_time += 1

    # ============================================================================================ the gradient

    def _pairs(self, responder: torch.Tensor) -> torch.Tensor:
        """(ball, other) pairs that call Potential [W, R, N]: GetNearbyBalls' collision candidates (collisionCase1:
        other massive, uncloaked balls in a related box; on the master, `bubble` set, in the ball's bubble; in a park
        made with `fields`, less the force fields the ball passes) less Gradient's space-junk skip."""
        R = self.R
        candidate = self.exists & self.massive & ~self.cloaked
        eye = torch.eye(self.N, dtype=BOOL, device=self.dev)[None, :R]
        junk = self.junk[:, None, :] & ~self.junk[:, :R, None]
        pair = responder[:, :R, None] & candidate[:, None, :] & ~eye & ~junk & self._related()
        if self.fields:                             # the force fields' filter (MOV-737)
            pair = pair & ~self._field_passed()
        return pair if self.bubble is None else pair & self._same_bubble()

    def _field_passed(self) -> torch.Tensor:
        """GetNearbyBalls' force-field filter in the collision searches (Partition.cpp:344, exclude_harmonics false;
        exact_ballpark.full's _nearby) [W, R, N]: the other ball is in mode FIELD and the ball's
        harmonic is -2 (the pass), or the other's harmonic, corporation or alliance is set (not -1) and is the
        ball's."""
        R = self.R
        field = (self.mode == FIELD_MODE)[:, None, :]
        same = lambda x: (x[:, None, :] != NO_GROUP) & (x[:, None, :] == x[:, :R, None])      # noqa: E731
        passed = (self.harmonic[:, :R, None] == HARMONIC_PASS) | same(self.harmonic) | same(self.corporation)
        return field & (passed | same(self.alliance))

    def _same_bubble(self) -> torch.Tensor:
        """The master's bubble filter of GetNearbyBalls (Partition.cpp:344; SEN-708) [W, R, N]: on the
        master a candidate in another bubble is skipped unless it is global (in a park made with `global_balls`: the candidate's own flag, so the rule is asymmetric); applied only where `bubble` is set (a
        client's view, None, is unchanged)."""
        same = self.bubble[:, :self.R, None] == self.bubble[:, None, :]
        if self.global_balls:
            same = same | self.glob[:, None, :]
        return same

    def _gradient(self, responder: torch.Tensor) -> None:
        """Gradient (Ballpark.cpp:2745) for every responder: Potential with each candidate in GetNearbyBalls' order.
        With contact_slots None the host prunes (a sync or a few a tick: no contact, no work); with a number it does
        not (every step on the device, for capture in a CUDA graph), and more contacts than that is an overflow."""
        sync = self.contact_slots is None
        pair = self._pairs(responder)
        if sync and not bool(pair.any()):
            return
        P = self._potential_pairs(pair, sync)
        active = pair & (P["s0"] != -1.0)
        counts = active.sum(-1)
        if sync:
            slots = int(counts.max())
            if slots == 0:
                return
            need_order = bool((counts > 1).any())
        else:
            slots = min(int(self.contact_slots), self.N - 1)
            self.overflow |= (counts > slots).any(-1)
            need_order = True
        order = self._order_keys(pair) if need_order else torch.zeros_like(counts)[..., None].expand_as(pair)
        perm = self._sorted(active, order)[:, :, :slots]
        used = P if self.R > 1 else {k: P[k] for k in ("s0", "ov0", "ap0", "cl0")}    # one row: depth 0's alone
        G = {k: torch.gather(v, 2, perm if v.dim() == 3 else perm[..., None].expand(-1, -1, -1, v.shape[-1]))
             for k, v in used.items()}
        G["active"] = torch.gather(active, 2, perm)
        G["other"] = torch.gather(self.id[:, None, :].expand_as(pair), 2, perm)
        G["slot"] = perm
        R, N = self.R, self.N
        if R < N:                                   # the updates on the rows' views (the fixed balls' rows unchanged)
            full_c, full_col = self.last_c, self.last_collision
            self.last_c, self.last_collision = full_c[:, :R], full_col[:, :R]
        for j in range(slots):
            self._apply_contact({k: v[:, :, j] for k, v in G.items()})
        if R < N:
            self.last_c = torch.cat([self.last_c, full_c[:, R:]], 1)
            self.last_collision = torch.cat([self.last_collision, full_col[:, R:]], 1)

    def _overlap(self, pb, pn, cr, m1, m2, g1, v1v, id1, id2) -> tuple:
        """Potential's overlap branch (COL-709, COL-710): the normal (along the centres, or +-x for coincident
        centres by id), the overlap plus one metre, the two pushes and the acceleration that pushes the ball out
        within the tick against mLastG's normal part."""
        k, dt = self.friction, self.dt
        p0q0 = pn - pb
        p0q0_2 = _dot(p0q0, p0q0)
        xs = torch.where(id1 > id2, 1.0, -1.0).to(F64).expand(p0q0_2.shape)
        axis = torch.stack([xs, torch.zeros_like(xs), torch.zeros_like(xs)], -1)
        normal = torch.where((p0q0_2 == 0.0)[..., None], axis, _normalized(p0q0))
        dist = (cr - torch.sqrt(p0q0_2)) + 1.0
        d1 = (m2 / (m1 + m2)) * dist
        d2 = (m1 / (m1 + m2)) * dist
        tmp = 1.0 / (m1 + dt * k)
        c = (-d1) / ((dt * tmp) * m1)
        a1 = (normal * c[..., None] - v1v) * (1.0 / dt) - normal * _dot(g1, normal)[..., None]
        return normal, dist, d1, d2, a1

    def _approach(self, pb, pn, s, v1v, v2v, g1, g2, m1, m2, tf1, tf2, mm1, mm2, free2) -> tuple:
        """Potential's impact branch (s > 0; COL-707, COL-708): both balls integrated to the impact (exp over s dt),
        the normal velocity reflected off a fixed ball or exchanged with a free one (base masses), the rest of the
        tick integrated (exp over (1 - s) dt), and the acceleration that reaches that end from the tick's start; with
        the closing speed along the normal at the impact (the impact record)."""
        k, dt = self.friction, self.dt
        ts, tr = s * dt, (1.0 - s) * dt
        ex = vec_exp.exp(torch.stack([self._decay(m1, ts), self._decay(m2, ts), self._decay(m1, tr)]))
        p1, vp1 = self._integrate_t(pb, v1v, g1, m1, tf1, ts, ex[0])
        q1, vq1 = self._integrate_t(pn, v2v, g2, m2, tf2, ts, ex[1])
        normal = _normalized(q1 - p1)
        u1, u2 = _dot(vp1, normal), _dot(vq1, normal)
        reflect = vp1 - normal * (2.0 * u1)[..., None]
        u1p = ((mm1 * u1 - mm2 * u1) + (2.0 * mm2) * u2) / (mm1 + mm2)
        exchange = vp1 + normal * (u1p - u1)[..., None]
        vp1 = torch.where(free2[..., None], exchange, reflect)
        _, vp2 = self._integrate_t(p1, vp1, g1, m1, tf1, tr, ex[2])
        mn, ft = m1[..., None], tf1[..., None]
        x = (((g1 * -mn) + g1 * (ft * mn)) - (v1v * ft) * k) + vp2 * k
        return (-x) * (1.0 / m1)[..., None] * (1.0 / (tf1 - 1.0))[..., None], u1 - u2

    def _potential_pairs(self, pair: torch.Tensor, sync: bool) -> dict:
        """Potential (Ballpark.cpp:2789; COL-705 to COL-711) for every (ball, other) pair at once: at each recursion
        depth the time of impact and both branches' accelerations; the order-dependent updates come after. With the
        host pruning (sync), depths 1 and 2 are computed only if a pair overlaps, and the impact branch (its exp) only
        for the pairs that reach it."""
        shape, R = pair.shape, self.R
        me = lambda x: x[:, :R, None].expand(*shape, *x.shape[2:])
        ot = lambda x: x[:, None, :].expand(*shape, *x.shape[2:])
        m1 = me(self.mass * self.agility)
        m2 = ot(torch.where(self.free, self.mass * self.agility, torch.full_like(self.mass, 1.0e34)))
        tf1, tf2, g1, g2 = me(self.time_factor), ot(self.time_factor), me(self.last_g), ot(self.last_g)
        v1v, v2v, pb, pn = me(self.vel), ot(self.vel), me(self.pos), ot(self.pos)
        cr = self.radius[:, :R, None] + self.radius[:, None, :]
        free2, id1, id2 = ot(self.free), self.id[:, :R, None], self.id[:, None, :]
        dv = v1v - v2v
        out = {"free2": free2, "early": _dot(dv, dv) > 0.0001, "light": (m2 / m1) > LIGHT_BALL_RATIO}
        reach = pair
        none = torch.full(shape, -1.0, dtype=F64, device=self.dev)
        zero3 = torch.zeros(*shape, 3, dtype=F64, device=self.dev)
        sp, pbs, pns, reached = [], [], [], []
        mf = self.mass * self.agility
        ends_free, _ = self._integrate_dt(self.pos, self.vel, self.last_g, mf, self.time_factor)
        ends_fixed, _ = self._integrate_dt(self.pos, self.vel, self.last_g, torch.full_like(mf, 1.0e34),
                                           self.time_factor)
        depths = 3 if self.R > 1 else 1         # one row: no other ball is free, so Potential never recurses
        for depth in range(3):
            if depth >= depths or (sync and depth > 0 and not bool(reach.any())):
                s, ov, dist = none, zero3, none
            else:
                if depth == 0:                                              # one prediction a ball, not a pair
                    p1, q1 = me(ends_free), ot(torch.where(self.free[..., None], ends_free, ends_fixed))
                else:
                    p1, _ = self._integrate_dt(pb, v1v, g1, m1, tf1)
                    q1, _ = self._integrate_dt(pn, v2v, g2, m2, tf2)
                s = torch.where(reach, self._collide_two_spheres(pb, p1, pn, q1, cr), none)
                if sync and not bool((s == 0.0).any()):
                    ov, dist, d1, d2, normal = zero3, none, None, None, None
                else:
                    normal, dist, d1, d2, ov = self._overlap(pb, pn, cr, m1, m2, g1, v1v, id1, id2)
            out[f"s{depth}"], out[f"ov{depth}"], out[f"dist{depth}"] = s, ov, dist
            sp.append(s)
            pbs.append(pb)
            pns.append(pn)
            reached.append(reach)
            reach = reach & (s == 0.0) & free2
            if depth < depths - 1 and (not sync or bool(reach.any())):
                pb = pb - normal * d1[..., None]
                pn = pn + normal * d2[..., None]
        need = [r & (x > 0.0) for r, x in zip(reached, sp)]
        rest = (v1v, v2v, g1, g2, m1, m2, tf1, tf2, me(self.mass), ot(self.mass), free2)
        zero = torch.zeros(shape, dtype=F64, device=self.dev)
        if sync:                                    # the impact branch on its pairs only, the three depths in one call
            idx = [n.nonzero(as_tuple=True) for n in need]
            counts = [int(i[0].numel()) for i in idx]
            for depth in range(3):
                out[f"ap{depth}"] = zero3
            out["cl0"] = zero
            if sum(counts):
                cat = lambda xs: torch.cat([x[i] for x, i in zip(xs, idx)])
                a1, cl = self._approach(cat(pbs), cat(pns), cat(sp), *(cat([x] * 3) for x in rest))
                for depth, part in enumerate(torch.split(a1, counts)):
                    if counts[depth]:
                        ap = zero3.clone()
                        ap[idx[depth]] = part
                        out[f"ap{depth}"] = ap
                if counts[0]:                       # the closing speed of depth 0's impacts (the impact record)
                    c = zero.clone()
                    c[idx[0]] = cl[:counts[0]]
                    out["cl0"] = c
        elif self.impact_slots is not None:         # the impact branch on the first impact_slots pairs that reach it
            a1, cl = self._approach_some(need[:depths], pbs[:depths], pns[:depths], sp[:depths], shape)
            for depth in range(3):
                out[f"ap{depth}"] = a1[depth] if depth < depths else zero3
            out["cl0"] = cl[0]
        else:
            e3 = lambda x: x.expand(depths, *x.shape)
            a1, cl = self._approach(torch.stack(pbs[:depths]), torch.stack(pns[:depths]), torch.stack(sp[:depths]),
                                    *(e3(x) for x in rest))
            for depth in range(3):
                out[f"ap{depth}"] = torch.where(need[depth][..., None], a1[depth], zero3) if depth < depths else zero3
            out["cl0"] = torch.where(need[0], cl[0], zero)
        return out

    def _approach_some(self, need, pbs, pns, sp, shape) -> tuple:
        """_approach on the first impact_slots (depth, ball, other) pairs that need it, in index order, by gathers with
        no host read (static shapes, for a CUDA graph), every input the same element as the dense call's; its results
        put back by gathers (zero elsewhere, as the dense call's selection). More pairs than that is an overflow event,
        raised by check()."""
        cap, (W, R, N), D = int(self.impact_slots), shape, len(need)
        flat = torch.stack(need).reshape(-1)
        cs = torch.cumsum(flat.to(I64), 0)
        pos = torch.searchsorted(cs, torch.arange(1, cap + 1, dtype=I64, device=self.dev)).clamp(max=flat.numel() - 1)
        self.impact_overflow |= cs[-1] > cap
        M1 = W * R * N
        d, q = torch.div(pos, M1, rounding_mode="floor"), torch.remainder(pos, M1)
        w, rn = torch.div(q, R * N, rounding_mode="floor"), torch.remainder(q, R * N)
        r, n = torch.div(rn, N, rounding_mode="floor"), torch.remainder(rn, N)
        mf = self.mass * self.agility
        m2f = torch.where(self.free, mf, torch.full_like(mf, 1.0e34))
        if D == 1:                                  # depth 0's own positions: the balls' states
            pb, pn, s = self.pos[w, r], self.pos[w, n], sp[0][w, r, n]
        else:
            pb, pn, s = torch.stack(pbs)[d, w, r, n], torch.stack(pns)[d, w, r, n], torch.stack(sp)[d, w, r, n]
        a1c, clc = self._approach(pb, pn, s, self.vel[w, r], self.vel[w, n], self.last_g[w, r], self.last_g[w, n],
                                  mf[w, r], m2f[w, n], self.time_factor[w, r], self.time_factor[w, n],
                                  self.mass[w, r], self.mass[w, n], self.free[w, n])
        keep = flat & (cs <= cap)
        rank = (cs - 1).clamp(min=0, max=cap - 1)
        a1 = torch.where(keep[:, None], a1c[rank], torch.zeros_like(a1c[rank])).reshape(D, W, R, N, 3)
        cl = torch.where(keep, clc[rank], torch.zeros_like(clc[rank])).reshape(D, W, R, N)
        return a1, cl

    def _apply_contact(self, c: dict) -> None:
        """One candidate of every responder, in order: Potential's updates of mLastC and mLastCollision and its
        appends to the collision list, through the recursion (deepest first), as the scalar code meets them."""
        act, s0 = c["active"], c["s0"]
        if self.R == 1:                         # one row: no other ball is free, so depth 0 alone (reach1 is false)
            self._update(act, s0, torch.where((s0 > 0.0)[..., None], c["ap0"], c["ov0"]), c["other"])
            self._record(act & (s0 > 0.0), c["slot"], s0, c["cl0"])
            return
        free2, early, s1, s2 = c["free2"], c["early"], c["s1"], c["s2"]
        reach1 = act & (s0 == 0.0) & free2                                   # depth 1 called
        reach2 = reach1 & (s1 == 0.0)                                        # depth 2 called
        a1 = {d: torch.where((c[f"s{d}"] > 0.0)[..., None], c[f"ap{d}"], c[f"ov{d}"]) for d in range(3)}
        self._update(reach2 & (s2 != -1.0), s2, a1[2], c["other"])          # depth 2 (no recursion past it)
        self._light(reach2 & c["light"], c["dist1"])                         # depth 1 after its recursion
        self._update(reach1 & (s1 != -1.0) & ~(reach2 & early), s1, a1[1], c["other"])
        self._light(reach1 & c["light"], c["dist0"])                         # depth 0 after its recursion
        self._update(act & ~(reach1 & early), s0, a1[0], c["other"])
        self._record(act & (s0 > 0.0), c["slot"], s0, c["cl0"])

    def _record(self, mask, slot, s, closing) -> None:
        """The impact record (kept where Potential computes the impact at depth 0): the other
        ball's slot, the time of impact and the closing speed along the normal, appended in candidate order."""
        for name, val in (("imp_slot", slot), ("imp_s", s), ("imp_closing", closing)):
            setattr(self, name, self._append(getattr(self, name), self.imp_n, mask, val))
        self.imp_n = self.imp_n + mask.to(I64)

    def _light(self, mask, dist) -> None:
        """Potential's light-ball rule (COL-710): mLastC renormalised to the overlap's length."""
        self.last_c = torch.where(mask[..., None], _normalized(self.last_c) * dist[..., None], self.last_c)

    def _update(self, mask, s, a1, other) -> None:
        """Potential's end (COL-711): 0.85 of the acceleration if the impact is not earlier than the one held (the
        stronger at an equal time), then the other ball on the collision list."""
        ge = mask & (s >= self.last_collision)
        eq = s == self.last_collision
        lc = a1 * RESPONSE_SCALE
        stronger = _dot(lc, lc) > _dot(self.last_c, self.last_c)
        take = ge & (~eq | stronger)
        self.last_c = torch.where(take[..., None], lc, self.last_c)
        self.last_collision = torch.where(ge & ~eq, s, self.last_collision)
        self.hits = self._append(self.hits, self.hits_n, mask, other)
        self.hits_n = self.hits_n + mask.to(I64)

    def _append(self, lst: torch.Tensor, n: torch.Tensor, mask: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        """The lists `lst` [.., C] with `val` at position n (the last position if past it) where mask: a selection, no
        scatter (a deterministic scatter on CUDA sorts its indices, dozens of kernels at every append)."""
        at = n.clamp(max=lst.shape[-1] - 1)
        put = (torch.arange(lst.shape[-1], device=lst.device) == at[..., None]) & mask[..., None]
        return torch.where(put, val[..., None], lst)

    # ================================================================== static collidables (COL-712)

    def _empty_statics(self, boxes: int) -> None:
        """The park's static collidables, none: [W, S] slots, each a record (static_record's doubles and integers) and
        its boxes' keys [W, S, boxes, 3] (the first I_NBOX of them hold)."""
        W, S = self.W, self.S
        self.B = max(1, int(boxes))
        self.s_exists = torch.zeros(W, S, dtype=BOOL, device=self.dev)
        self.s_geo = torch.zeros(W, S, G_SIZE, dtype=F64, device=self.dev)
        self.s_int = torch.zeros(W, S, I_SIZE, dtype=I64, device=self.dev)
        self.s_keys = torch.zeros(W, S, self.B, 3, dtype=I64, device=self.dev)
        self.has_statics = False                # a host flag: no read of the device in the tick
        # BoxShape's tables on the device, made here (never inside a tick a CUDA graph records)
        self._sides_t = torch.tensor(CONNECTED_SIDES, dtype=I64, device=self.dev)
        self._mapping_t = torch.tensor(SIDE_MAPPING, dtype=I64, device=self.dev)

    def load_statics(self, statics: list[list[dict]]) -> None:
        """Each world's static collidables (capsules and oriented boxes, `static_record`'s dicts; distinct ids within a
        world), in the park's static slots: as AddCapsule and AddOrientedBox add them to a park (a client's view: the
        master's bubbles, `bubble`, are refused with them), filed once. They take part in the gradient method only (the
        iterative method's CheckCollision is not in this module: refused)."""
        if len(statics) != self.W:
            raise ValueError(f"{len(statics)} worlds of static collidables for a park of {self.W}")
        recs = []
        for w, items in enumerate(statics):
            if len(items) > self.S:
                raise ValueError(f"world {w} has {len(items)} static collidables for the park's {self.S} slots")
            ids = [int(s["id"]) for s in items]
            if len(set(ids)) != len(ids):
                raise ValueError(f"world {w}: static collidable ids must be distinct")
            recs.append([static_record(s) for s in items])
        if self.iterative and any(recs):
            raise ValueError("static collidables in the iterative method are not in this module")
        if not self.S:                              # none given to a park without slots: nothing to hold
            return
        self._empty_statics(max([len(r[2]) for rs in recs for r in rs], default=1))
        geo = [[[0.0] * G_SIZE for _ in range(self.S)] for _ in range(self.W)]
        ints = [[[0] * I_SIZE for _ in range(self.S)] for _ in range(self.W)]
        keys = [[[[0, 0, 0] for _ in range(self.B)] for _ in range(self.S)] for _ in range(self.W)]
        exists = [[False] * self.S for _ in range(self.W)]
        for w, rs in enumerate(recs):
            for j, (g, i, k) in enumerate(rs):
                geo[w][j], ints[w][j], exists[w][j] = g, i, True
                for b, key in enumerate(k):
                    keys[w][j][b] = list(key)
        t = lambda xs, dt: torch.tensor(xs, dtype=dt, device=self.dev)
        self.s_geo, self.s_int, self.s_keys, self.s_exists = t(geo, F64), t(ints, I64), t(keys, I64), t(exists, BOOL)
        self.has_statics = any(exists[w][j] for w in range(self.W) for j in range(self.S))

    def _static_order_keys(self) -> torch.Tensor:
        """The position of each static collidable's first box in each responder's walk (Partition::GetNearbyBalls with
        its static collidables, Partition.cpp:344) [W, R, S], as _order_keys's key: a static collidable's boxes are all
        at one level (its InsertInBoxes'), each related to the ball's boxes as an ancestor, the box itself or a box
        below; the smallest key over its boxes; NOT_A_CANDIDATE where none relates. Within a box the walk takes them by
        their parent's id, then their own (_static_sorted)."""
        R = self.R
        Lb = self.level[:, :R, None, None]                                   # [W, R, 1, 1]
        Ln = self.s_int[:, None, :, None, I_LEVEL]                           # [W, 1, S, 1]
        Kn = self.s_keys[:, None]                                            # [W, 1, S, B, 3]
        held = (torch.arange(self.B, device=self.dev) < self.s_int[..., I_NBOX, None])[:, None]     # [W, 1, S, B]
        Kb, Mb = self.key[:, :R, None, None, :], self.mod[:, :R, None, None, :]               # [W, R, 1, 1, 3]
        cnt_b = 1 + (Mb != 0).to(I64)
        lo_b, hi_b = Kb + Mb.clamp(max=0), Kb + Mb.clamp(min=0)
        s_up = (2 * (Lb - Ln)).clamp(min=0)[..., None]                       # [W, R, S, 1, 1]
        s_dn = (2 * (Ln - Lb)).clamp(min=0)[..., None]
        code = None                                 # each box's post-order code below the ball's level
        for li, power in enumerate(self._power_list):
            lvl = li + 1
            d = (Kn >> (2 * (Ln - lvl)).clamp(min=0)[..., None]) & 3
            digit = d[..., 0] * 16 + d[..., 1] * 4 + d[..., 2]               # [W, 1, S, B]
            digit = torch.where(Ln < lvl, torch.full_like(digit, CODE_BASE - 1), digit)
            digit = torch.where(Lb >= lvl, torch.zeros_like(digit), digit)  # [W, R, S, B]
            code = digit * power if code is None else code + digit * power
        best = torch.full(code.shape, NOT_A_CANDIDATE, dtype=I64, device=self.dev)
        na = torch.full_like(best, NOT_A_CANDIDATE)
        for bit in self._combos:                    # each of the ball's boxes, in its walk's order (by key)
            valid = ((bit == 0) | (cnt_b == 2)).all(-1)                      # [W, R, 1, 1]
            box = torch.where(bit == 1, hi_b, lo_b)                          # [W, R, 1, 1, 3]
            seg = (bit[0] * cnt_b[..., 0, 1] + bit[1]) * cnt_b[..., 0, 2] + bit[2]         # [W, R, 1]
            seg = seg[..., None]
            up = ((box >> s_up) == Kn).all(-1)                               # an ancestor of the ball's box
            same = (box == Kn).all(-1)                                       # the ball's box itself
            down = (box == (Kn >> s_dn)).all(-1)                             # below the ball's box
            k = torch.where(Ln < Lb, torch.where(up, (seg * 3) * CODE_SPAN + (Lb - Ln), na),
                            torch.where(Ln == Lb, torch.where(same, (seg * 3 + 2) * CODE_SPAN, na),
                                        torch.where(down, (seg * 3 + 1) * CODE_SPAN + code, na)))
            best = torch.where(valid & held & (k < best), k, best)
        return best.min(-1).values

    def _static_sorted(self, active: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        """The permutation [W, R, S] putting each responder's active static collidables first, in the walk's order: the
        first box's position, then (as Box::mStaticCollidables, a std::set) the parent's id, then the id (then the
        slot); by ranks, integers only."""
        S = self.S
        order = torch.where(active, order, torch.full_like(order, NOT_A_CANDIDATE))
        par = self.s_int[:, None, :, I_PARENT].expand_as(order)
        ids = self.s_int[:, None, :, I_ID].expand_as(order)
        o_i, o_j = order[..., :, None], order[..., None, :]
        p_i, p_j = par[..., :, None], par[..., None, :]
        d_i, d_j = ids[..., :, None], ids[..., None, :]
        ar = torch.arange(S, device=self.dev)
        j_lt_i = ar[None, :] < ar[:, None]
        before = (o_j < o_i) | ((o_j == o_i) & ((p_j < p_i) | ((p_j == p_i) & ((d_j < d_i) | ((d_j == d_i) & j_lt_i)))))
        rank = before.to(I64).sum(-1)
        at = rank[..., None, :] == ar[:, None]
        return (at.to(I64) * ar).sum(-1)

    @staticmethod
    def _pick(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """table[..., idx, :] for each leading index: table [*L, K, *T], idx [*L] (int64) -> [*L, *T]."""
        lead = idx.shape
        flat = table.reshape(-1, *table.shape[len(lead):])
        picked = flat[torch.arange(flat.shape[0], device=table.device), idx.reshape(-1)]
        return picked.reshape(*lead, *table.shape[len(lead) + 1:])

    @staticmethod
    def _cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Vector3d::Cross (exact_ballpark.modes._cross): u x v, components in Destiny's order."""
        return torch.stack([u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1],
                            u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2],
                            u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]], -1)

    def _capsule_contact(self, p0, p1, vp1, brad, geo) -> tuple:
        """Capsule::HandleCollisionNonIteratively (Capsule.cpp:206; full.py's _capsule_non_iterative) for every pair:
        (toi, the normal before its normalisation, whether the reaction starts from the tick's end, as coded when the
        end velocity is parallel to the axis); toi -1 where it does not react."""
        ca, cb, ab, crad = geo[..., G_A:G_A + 3], geo[..., G_B:G_B + 3], geo[..., G_AB:G_AB + 3], geo[..., G_RAD]
        rad = brad + crad
        ao = p0 - ca
        abab = _dot(ab, ab)
        m = _dot(ab, vp1) / abab
        n = _dot(ab, ao) / abab
        q = vp1 - ab * m[..., None]
        r = ao - ab * n[..., None]
        a = _dot(q, q)
        bq = 2.0 * _dot(q, r)
        cq = _dot(r, r) - rad * rad
        rad_sq = rad * rad
        minus = torch.full_like(a, -1.0)
        zero3 = torch.zeros_like(ao)
        sa = self._collide_two_spheres(p0, p1, ca, ca, rad)                 # the caps (spheres at A and B)
        sb = self._collide_two_spheres(p0, p1, cb, cb, rad)
        na, nb = p0 - ca, p0 - cb
        take_a = sa != -1.0                                                  # a = 0: the caps alone
        toi_p = torch.where(take_a, sa, minus)
        take_b = (sb != -1.0) & ((toi_p == -1.0) | (sb < toi_p))
        toi_p = torch.where(take_b, sb, toi_p)
        nrm_p = torch.where(take_b[..., None], nb, torch.where(take_a[..., None], na, zero3))
        det = bq * bq - 4.0 * a * cq                                         # Quadratic (Collision.cpp:137)
        found = ~(det < 0.0)
        root = torch.sqrt(torch.where(found, det, torch.zeros_like(det)))
        s1 = ((-bq + root) * 0.5) / a
        s2 = ((-bq - root) * 0.5) / a
        s = torch.where((s1 >= 0.0) & (s1 <= 1.0), s1, minus)
        second = (s2 >= 0.0) & (s2 <= 1.0) & (s2 < s1)
        s = torch.where(second, s2, s)
        t = _dot(ao, ab) / abab
        dist = p0 - (ca + ab * t[..., None])
        inside = ~second & (s == -1.0) & (t >= 0.0) & (t <= 1.0) & (_dot(dist, dist) < rad_sq)
        mid_n = ao - ab * t[..., None]
        mid_t = torch.where(_dot(mid_n, mid_n) < rad_sq, torch.zeros_like(s), s)
        toi, nrm = minus, zero3
        for tk, far in (((s1 * m) + n, p0), ((s2 * m) + n, p1)):          # each root's end: a cap or the cylinder
            low, high = tk < 0.0, tk > 1.0
            ta = low & (sa != -1.0) & ((toi == -1.0) | (sa < toi))
            tb = ~low & high & (sb != -1.0) & ((toi == -1.0) | (sb < toi))
            tm = ~low & ~high & (s != -1.0) & ((toi == -1.0) | (s < toi))
            toi = torch.where(ta, sa, torch.where(tb, sb, torch.where(tm, mid_t, toi)))
            nrm = torch.where(ta[..., None], na, torch.where(tb[..., None], far - cb,
                                                            torch.where(tm[..., None], mid_n, nrm)))
        parallel = a == 0.0
        toi = torch.where(parallel, toi_p, torch.where(~found, minus, torch.where(inside, torch.zeros_like(toi), toi)))
        nrm = torch.where(parallel[..., None], nrm_p, torch.where(inside[..., None], dist, nrm))
        return toi, nrm, parallel

    def _triangle_closest(self, A, B, C, P) -> torch.Tensor:
        """Triangle::ClosestPoint (Triangle.cpp; full.py's Triangle.closest_point), its regions in order."""
        ab, ac = B - A, C - A
        ap = P - A
        d1, d2 = _dot(ab, ap), _dot(ac, ap)
        bp = P - B
        d3, d4 = _dot(ab, bp), _dot(ac, bp)
        vc = d1 * d4 - d3 * d2
        cp = P - C
        d5, d6 = _dot(ab, cp), _dot(ac, cp)
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4
        e43, e56 = d4 - d3, d5 - d6
        denom = 1.0 / (va + vb + vc)
        inner = (A + ab * (vb * denom)[..., None]) + ac * (vc * denom)[..., None]
        out = torch.where(((va <= 0.0) & (e43 >= 0.0) & (e56 >= 0.0))[..., None],
                          B + (C - B) * (e43 / (e43 + e56))[..., None], inner)
        out = torch.where(((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0))[..., None], A + ac * (d2 / (d2 - d6))[..., None],
                          out)
        out = torch.where(((d6 > 0.0) & (d5 <= d6))[..., None], C, out)
        out = torch.where(((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0))[..., None], A + ab * (d1 / (d1 - d3))[..., None],
                          out)
        out = torch.where(((d3 >= 0.0) & (d4 <= d3))[..., None], B, out)
        return torch.where(((d1 <= 0.0) & (d2 <= 0.0))[..., None], A, out)

    def _rect_contact(self, p, radius, vel, side, rect, rnorm, tnorm) -> tuple:
        """RectangleShape::CollideWithSphere (BoxShape.cpp:8) on one side of each pair: its triangle (a, c, d), or
        (a, b, c) when the velocity does not turn away from (c - p) x (a - p), then IntersectSphereTriangle
        (Collision.cpp:67): the crossing of the plane inside the triangle, else the sphere swept back from the triangle's
        closest point (IntersectSegmentSphere, Collision.cpp:29). (hit, normal, time)."""
        corners = self._pick(rect, side)                                     # [.., 4, 3]
        a, b, c, d = corners[..., 0, :], corners[..., 1, :], corners[..., 2, :], corners[..., 3, :]
        m = self._cross(c - p, a - p)
        abc = _dot(vel, m) <= 0.0
        TA = a
        TB = torch.where(abc[..., None], b, c)
        TC = torch.where(abc[..., None], c, d)
        n = torch.where(abc[..., None], self._pick(rnorm, side), self._pick(tnorm, side))
        p0 = p - n * radius[..., None]                                       # GetPointOnBallThatFirstIntersectsPlane
        p0p1 = (p0 + vel) - p0
        ndab = _dot(n, p0p1)                                                 # IntersectSegmentPlane
        t = (_dot(TA, n) - _dot(n, p0)) / ndab
        plane = (ndab != 0.0) & (t >= 0.0) & (t <= 1.0)
        point = p0 + p0p1 * t[..., None]
        v0, v1, v2 = TC - TA, TB - TA, point - TA                           # Triangle::Contains
        dot00, dot01, dot02 = _dot(v0, v0), _dot(v0, v1), _dot(v0, v2)
        dot11, dot12 = _dot(v1, v1), _dot(v1, v2)
        inv = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        within = (u >= 0) & (v >= 0) & (u + v < 1)
        q0 = self._triangle_closest(TA, TB, TC, point)
        direction = (q0 - vel) - q0                                          # IntersectSegmentSphere(q0, q0 - vel)
        nd = _normalized(direction)
        mm = q0 - p
        bb = _dot(mm, nd)
        cc = _dot(mm, mm) - radius * radius
        disc = bb * bb - cc
        tt = -bb - torch.sqrt(torch.where(disc < 0.0, torch.zeros_like(disc), disc))
        ts = torch.where(tt < 0.0, torch.zeros_like(tt), tt / torch.sqrt(_dot(direction, direction)))
        swept = ~((cc > 0.0) & (bb > 0.0)) & ~(disc < 0.0) & ~(ts > 1.0)
        impact = q0 + direction * ts[..., None]
        hit = plane & (within | swept)
        return hit, torch.where(within[..., None], n, _normalized(p - impact)), torch.where(within, t, ts)

    def _box_contact(self, p0, p1, brad, geo) -> tuple:
        """OrientedBox::CollideWithBall (OrientedBox.cpp:44) for every pair: the AABB's exclusion of the straight move,
        then BoxShape::CollideWithSphere (BoxShape.cpp:200): the closest corner; the centre inside its three planes
        (the extrication normal: out along the shortest way, the ray's exit side, or the closest point's direction);
        else its three sides' triangles, the first struck. (toi, the normal before its normalisation); toi -1 where it
        does not react."""
        lead = p0.shape[:-1]
        lo, hi = geo[..., G_LO:G_LO + 3], geo[..., G_HI:G_HI + 3]
        r = brad
        excl = torch.zeros(lead, dtype=BOOL, device=self.dev)               # AABB::CanExclude
        for q in range(3):
            excl = excl | (((p0[..., q] - r) > hi[..., q]) & ((p1[..., q] - r) > hi[..., q])) | (
                ((p0[..., q] + r) < lo[..., q]) & ((p1[..., q] + r) < lo[..., q]))
        vel = p1 - p0
        p, radius = p0, brad
        corners = geo[..., G_CORNERS:G_CORNERS + 24].reshape(*lead, 8, 3)
        rect = geo[..., G_RECT:G_RECT + 72].reshape(*lead, 6, 4, 3)
        rnorm = geo[..., G_RNORM:G_RNORM + 18].reshape(*lead, 6, 3)
        tnorm = geo[..., G_TNORM:G_TNORM + 18].reshape(*lead, 6, 3)
        cs_t, sm_t = self._sides_t, self._mapping_t
        dd = corners[..., 0, :] - p                                          # ClosestCorner: the first nearest
        min_d = _dot(dd, dd)
        closest = torch.zeros(lead, dtype=I64, device=self.dev)
        for i in range(1, 8):
            dd = corners[..., i, :] - p
            di = _dot(dd, dd)
            nearer = di < min_d
            min_d = torch.where(nearer, di, min_d)
            closest = torch.where(nearer, torch.full_like(closest, i), closest)
        sides = cs_t[closest]                                                # [.., 3]
        alive = torch.ones(lead, dtype=BOOL, device=self.dev)               # could_be_inside
        shortest = torch.full(lead, -1.0, dtype=F64, device=self.dev)
        half = torch.zeros(lead, dtype=I64, device=self.dev)
        for i in range(3):
            side = sides[..., i]
            n_s = self._pick(rnorm, side)
            point = p - n_s * radius[..., None]
            hs = _dot(n_s, self._pick(rect, side)[..., 0, :] - point)
            ok = hs >= 0.0
            take = alive & ok & ((hs < shortest) | (shortest == -1.0))
            shortest = torch.where(take, hs, shortest)
            half = torch.where(take, side, half)
            alive = alive & ok
        q = geo[..., G_CENTER:G_CENTER + 3]                                  # BoxShape::ClosestPoint
        d = p - q
        for i in range(3):
            u = geo[..., G_UNIT + 3 * i:G_UNIT + 3 * i + 3]
            hl = geo[..., G_HALF + i]
            dist = _dot(d, u)
            dist = torch.where(dist > hl, hl, torch.where(dist < -hl, -hl, dist))
            q = q + u * dist[..., None]
        normal = p - q
        n_len = torch.sqrt(_dot(normal, normal))
        speed = torch.sqrt(_dot(vel, vel))
        o = p - geo[..., G_CENTER:G_CENTER + 3]                              # BoxShape::RayIntersect, the slabs
        tmin, tmax, rev = [], [], []
        for i in range(3):
            arm = geo[..., G_AXES + 3 * i:G_AXES + 3 * i + 3]
            als, a_o, a_r = geo[..., G_LENSQ + i], _dot(arm, o), _dot(arm, vel)
            neg, pos = a_r < 0, a_r > 0
            tmin.append(torch.where(neg, (0.5 * als - a_o) / a_r, torch.where(pos, (-0.5 * als - a_o) / a_r,
                                                                              torch.full_like(a_r, -DBL_MAX))))
            tmax.append(torch.where(neg, (-0.5 * als - a_o) / a_r, torch.where(pos, (0.5 * als - a_o) / a_r,
                                                                               torch.full_like(a_r, DBL_MAX))))
            rev.append(neg.to(I64))
        i_min, v_min = torch.zeros(lead, dtype=I64, device=self.dev), tmin[0]
        i_max, v_max = torch.zeros(lead, dtype=I64, device=self.dev), tmax[0]
        for i in (1, 2):
            later = v_min < tmin[i]                                          # std::max_element: the first largest
            i_min, v_min = torch.where(later, torch.full_like(i_min, i), i_min), torch.where(later, tmin[i], v_min)
            sooner = tmax[i] < v_max                                         # std::min_element: the first smallest
            i_max, v_max = torch.where(sooner, torch.full_like(i_max, i), i_max), torch.where(sooner, tmax[i], v_max)
        rv = torch.stack(rev, -1).gather(-1, i_min[..., None])[..., 0]
        side1 = sm_t[i_min, rv]
        side2 = sm_t[i_min, 1 - rv]
        found = v_min < v_max
        fmax5 = lambda x: torch.where(x < STATIC_FLOOR, torch.full_like(x, STATIC_FLOOR), x)
        n_half = self._pick(rnorm, half)
        ray = torch.where((v_min.abs() < v_max.abs())[..., None],
                          self._pick(rnorm, side1) * fmax5(v_min.abs() * speed)[..., None],
                          self._pick(rnorm, side2) * fmax5(v_max.abs() * speed)[..., None])
        deep = torch.where((speed < STATIC_TINY)[..., None], n_half * fmax5(shortest)[..., None],
                           torch.where(found[..., None], ray, n_half * shortest[..., None]))
        n_in = torch.where((n_len < STATIC_TINY)[..., None], deep,
                           _normalized(normal) * (radius - n_len).abs()[..., None])
        inside = alive & (n_len < radius)
        hit, nrm, time = torch.zeros(lead, dtype=BOOL, device=self.dev), torch.zeros_like(p), torch.zeros_like(radius)
        for i in range(3):                                                   # else the sides' triangles, in order
            h_i, n_i, t_i = self._rect_contact(p, radius, vel, sides[..., i], rect, rnorm, tnorm)
            first = ~hit & h_i
            nrm = torch.where(first[..., None], n_i, nrm)
            time = torch.where(first, t_i, time)
            hit = hit | h_i
        hit = ~excl & (inside | hit)
        nrm = torch.where(inside[..., None], n_in, nrm)
        time = torch.where(inside, torch.zeros_like(time), time)
        return torch.where(hit, time, torch.full_like(time, -1.0)), nrm

    def _static_values(self) -> tuple:
        """CollideWithBall (Capsule.cpp:365, OrientedBox.cpp:44) of every (responder, static collidable) pair from the
        tick's start (its straight path predicted with mLastG), then ReactToCollision (Capsule.cpp:395,
        OrientedBox.cpp:76): ignored where it does not react, or (a capsule) where mLastG points out of it at an impact
        within the tick; the reflection off it (s > 0: integrated to the impact, the normal velocity reversed, the rest
        integrated; the acceleration that reaches that end) or the ejection (s = 0: out by the overlap plus a metre
        within the tick, against mLastG's normal part). (called [W, R, S], toi, 0.85 of the acceleration [W, R, S, 3])
        for _static_update, which alone depends on the order."""
        W, R, S = self.W, self.R, self.S
        me = lambda x: x[:, :R, None].expand(W, R, S, *x.shape[2:])
        it = lambda x: x[:, None].expand(W, R, S, *x.shape[2:])
        m = self.mass * self.agility
        pe, ve = self._integrate_dt(self.pos, self.vel, self.last_g, m, self.time_factor)
        p0, v0, g, p1, vp1 = me(self.pos), me(self.vel), me(self.last_g), me(pe), me(ve)
        m1, tf, brad = me(m), me(self.time_factor), me(self.radius)
        geo = it(self.s_geo)
        cap = it(self.s_int[..., I_KIND]) == STATIC_CAPSULE
        toi_c, nrm_c, end_c = self._capsule_contact(p0, p1, vp1, brad, geo)
        toi_b, nrm_b = self._box_contact(p0, p1, brad, geo)
        toi = torch.where(cap, toi_c, toi_b)
        normal = torch.where(cap[..., None], nrm_c, nrm_b)
        end = (cap & end_c)[..., None]
        pos, vel = torch.where(end, p1, p0), torch.where(end, vp1, v0)
        cl = torch.sqrt(_dot(normal, normal))
        nn = _normalized(normal)
        nc = _dot(g, nn)
        called = (toi != -1.0) & ~(cap & (toi > 0.0) & (nc > 0.0))
        k, dt = self.friction, self.dt
        t1, t2 = toi * dt, (1.0 - toi) * dt
        e = vec_exp.exp(torch.stack([self._decay(m1, t1), self._decay(m1, t2)]))
        pa, va = self._integrate_t(pos, vel, g, m1, tf, t1, e[0])
        vb = va - nn * (2.0 * _dot(va, nn))[..., None]
        _, vc = self._integrate_t(pa, vb, g, m1, tf, t2, e[1])
        mn, ft = m1[..., None], tf[..., None]
        x = (((g * -mn) + g * (ft * mn)) - (v0 * ft) * k) + vc * k
        reflect = (-x) * (1.0 / m1)[..., None] * (1.0 / (tf - 1.0))[..., None]
        dist = torch.where(cap, ((geo[..., G_RAD] + brad) - cl) + EJECT_MARGIN, cl + EJECT_MARGIN)
        tmp = 1.0 / (m1 + dt * k)
        c = dist / ((dt * tmp) * m1)
        eject = (nn * c[..., None] - v0) * (1.0 / dt) - nn * nc[..., None]
        acc = torch.where((toi > 0.0)[..., None], reflect, eject)
        return called, toi, acc * RESPONSE_SCALE

    def _static_update(self, mask, toi, lc, sid) -> None:
        """ReactToCollision's end (COL-711, COL-712): at an equal time the stronger response, else this one replaces
        mLastC and mLastCollision (no later-impact test); the static collidable's id on the collision list."""
        eq = toi == self.last_collision
        stronger = _dot(lc, lc) > _dot(self.last_c, self.last_c)
        self.last_c = torch.where((mask & (~eq | stronger))[..., None], lc, self.last_c)
        self.last_collision = torch.where(mask & ~eq, toi, self.last_collision)
        self.hits = self._append(self.hits, self.hits_n, mask, sid)
        self.hits_n = self.hits_n + mask.to(I64)

    def _gradient_statics(self, responder: torch.Tensor) -> None:
        """Gradient's static collidables (Ballpark.cpp:2778-2783; full.py's _gradient), after its balls: each candidate
        (filed in a box of the responder's walk) in the walk's order, through CollideWithBall. With contact_slots None
        the host prunes (no candidate, no work); with a number nothing waits for the host (every static slot is walked,
        for a CUDA graph)."""
        if not self.has_statics:
            return
        if self.bubble is not None:
            raise ValueError("static collidables on the master's view (bubble set) are not in this module")
        sync, R = self.contact_slots is None, self.R
        order = self._static_order_keys()
        pair = responder[:, :R, None] & self.s_exists[:, None, :] & (order < NOT_A_CANDIDATE)
        if sync and not bool(pair.any()):
            return
        called, toi, lc = self._static_values()
        active = pair & called
        slots = int(active.sum(-1).max()) if sync else self.S
        if slots == 0:
            return
        perm = self._static_sorted(active, order)[:, :, :slots]
        g_act, g_toi = torch.gather(active, 2, perm), torch.gather(toi, 2, perm)
        g_lc = torch.gather(lc, 2, perm[..., None].expand(-1, -1, -1, 3))
        g_id = torch.gather(self.s_int[:, None, :, I_ID].expand_as(active), 2, perm)
        N = self.N
        if R < N:                                   # the updates on the rows' views (the fixed balls' rows unchanged)
            full_c, full_col = self.last_c, self.last_collision
            self.last_c, self.last_collision = full_c[:, :R], full_col[:, :R]
        for j in range(slots):
            self._static_update(g_act[:, :, j], g_toi[:, :, j], g_lc[:, :, j], g_id[:, :, j])
        if R < N:
            self.last_c = torch.cat([self.last_c, full_c[:, R:]], 1)
            self.last_collision = torch.cat([self.last_collision, full_col[:, R:]], 1)

    # ============================================================================================ the iterative

    def _sel(self, x: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
        """x [W, N, ...] at one slot per world jj [W] -> [W, ...]."""
        idx = jj.view(self.W, 1, *([1] * (x.dim() - 2))).expand(self.W, 1, *x.shape[2:])
        return torch.gather(x, 1, idx).squeeze(1)

    def _put(self, x: torch.Tensor, jj: torch.Tensor, mask: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        """x with x[w, jj[w]] = val[w] where mask[w]."""
        cur = self._sel(x, jj)
        new = torch.where(mask.view(self.W, *([1] * (val.dim() - 1))), val, cur)
        idx = jj.view(self.W, 1, *([1] * (x.dim() - 2))).expand(self.W, 1, *x.shape[2:])
        return torch.scatter(x, 1, idx, new.unsqueeze(1))

    def _integrate_any(self, p, v, a, m, tf, t):
        """Ballpark::Integrate over any time t [..] (exp's value where t is not the tick)."""
        return self._integrate_t(p, v, a, m, tf, t, vec_exp.exp(self._decay(m, t)))

    def _position_at(self, lt, lp, lv, ln, m, tf, a, times) -> tuple:
        """CollisionBallProperties::CalculatePosition (FindIndex, then Integrate from that point), for paths with any
        leading dims: lt [.., L], lp and lv [.., L, 3], ln [..]; the ball's m, tf [..] and acceleration a [.., 3]
        (mLastG + mLastC as they stand); times [.., T] -> (p, v) [.., T, 3]."""
        lidx = torch.arange(lt.shape[-1], device=self.dev)
        valid = lidx < ln[..., None]
        before = (lt[..., None, :] + FUZZY_DELTA < times[..., :, None]) & valid[..., None, :]
        idx = (before.sum(-1) - 1).clamp(min=0)
        t0 = torch.gather(lt, -1, idx)
        i3 = idx[..., None].expand(*idx.shape, 3)
        p0, v0 = torch.gather(lp, -2, i3), torch.gather(lv, -2, i3)
        far = ((t0 - times).abs() >= FUZZY_DELTA)[..., None]
        mm, ff = m[..., None].expand_as(times), tf[..., None].expand_as(times)
        p, v = self._integrate_any(p0, v0, a[..., None, :].expand_as(p0), mm, ff, times - t0)
        return torch.where(far, p, p0), torch.where(far, v, v0)

    def _accel(self) -> torch.Tensor:
        return self.last_g + self.last_c

    def _props(self, jj) -> dict:
        """The collision path of one ball per world."""
        return {"t": self._sel(self.loc_t, jj), "c": self._sel(self.loc_c, jj), "p": self._sel(self.loc_p, jj),
                "v": self._sel(self.loc_v, jj), "n": self._sel(self.loc_n, jj), "ts": self._sel(self.t_start, jj),
                "te": self._sel(self.t_end, jj), "cap": self._sel(self.cap, jj)}

    def _set_props(self, jj, mask, d: dict) -> None:
        self.loc_t = self._put(self.loc_t, jj, mask, d["t"])
        self.loc_c = self._put(self.loc_c, jj, mask, d["c"])
        self.loc_p = self._put(self.loc_p, jj, mask, d["p"])
        self.loc_v = self._put(self.loc_v, jj, mask, d["v"])
        self.loc_n = self._put(self.loc_n, jj, mask, d["n"])
        self.t_start = self._put(self.t_start, jj, mask, d["ts"])
        self.t_end = self._put(self.t_end, jj, mask, d["te"])
        self.cap = self._put(self.cap, jj, mask, d["cap"])

    def _ball_terms(self, jj) -> tuple:
        """(m, tf, a) of one ball per world: mass x agility, the cached factor, mLastG + mLastC now."""
        return (self._sel(self.mass * self.agility, jj), self._sel(self.time_factor, jj),
                self._sel(self._accel(), jj))

    def _position(self, d: dict, jj, times: torch.Tensor) -> tuple:
        m, tf, a = self._ball_terms(jj)
        return self._position_at(d["t"], d["p"], d["v"], d["n"], m, tf, a, times)

    def _write_loc(self, d: dict, at, mask, t, c, p, v) -> None:
        """d's point at index `at` [W] set where mask (an index past the tensor's room is an overflow)."""
        self.loc_overflow |= mask & (at >= self.L)
        at = at.clamp(max=self.L - 1)
        sel = torch.nn.functional.one_hot(at, self.L).to(BOOL) & mask[:, None]
        d["t"] = torch.where(sel, t[:, None], d["t"])
        d["c"] = torch.where(sel, c[:, None], d["c"])
        d["p"] = torch.where(sel[..., None], p[:, None, :], d["p"])
        d["v"] = torch.where(sel[..., None], v[:, None, :], d["v"])

    def _init_props(self, resp: torch.Tensor) -> None:
        """CollisionBallProperties::Initialize for every responding ball: its start and its end over the tick."""
        L = self.L
        self.loc_t = torch.full((self.W, self.N, L), -1.0, dtype=F64, device=self.dev)
        self.loc_c = torch.full((self.W, self.N, L), -1, dtype=I64, device=self.dev)
        self.loc_p = torch.zeros(self.W, self.N, L, 3, dtype=F64, device=self.dev)
        self.loc_v = torch.zeros(self.W, self.N, L, 3, dtype=F64, device=self.dev)
        pe, ve = self._integrate_dt(self.pos, self.vel, self._accel(), self.mass * self.agility, self.time_factor)
        self.loc_t[..., 0], self.loc_t[..., 1] = 0.0, self.dt
        self.loc_p[..., 0, :], self.loc_v[..., 0, :] = self.pos, self.vel
        self.loc_p[..., 1, :], self.loc_v[..., 1, :] = pe, ve
        self.loc_n = torch.where(resp, 2, 0).to(I64)
        self.t_start = torch.zeros(self.W, self.N, dtype=F64, device=self.dev)
        self.t_end = torch.full((self.W, self.N), self.dt, dtype=F64, device=self.dev)
        self.cap = torch.where(resp & (self.cap < 2), torch.full_like(self.cap, 2), self.cap)

    def _calculate_end(self, jj, mask) -> None:
        """CollisionBallProperties::CalculateEnd: a last point at the tick's end, integrated from the path's last; the
        push_back under a held reference is undefined when the vector is full (DestinyUndefined in the reference)."""
        d = self._props(jj)
        last = (d["n"] - 1).clamp(min=0)
        lt = torch.gather(d["t"], 1, last[:, None])[:, 0]
        go = mask & (self.dt > lt)
        if not bool(go.any()):
            return
        self.undefined |= go & (d["n"] == d["cap"])
        i3 = last[:, None, None].expand(-1, 1, 3)
        lp, lv = torch.gather(d["p"], 1, i3)[:, 0], torch.gather(d["v"], 1, i3)[:, 0]
        m, tf, a = self._ball_terms(jj)
        p, v = self._integrate_any(lp, lv, a, m, tf, self.dt - lt)
        self._write_loc(d, d["n"], go, torch.full_like(lt, self.dt), torch.full_like(d["n"], -1), p, v)
        d["n"] = torch.where(go, d["n"] + 1, d["n"])
        d["te"] = torch.where(go, torch.full_like(lt, self.dt), d["te"])
        d["ts"] = torch.where(go, lt, d["ts"])
        self._set_props(jj, go, d)

    def _ignore_collision(self, d: dict, collider, toi) -> torch.Tensor:
        """CollisionBallProperties::IgnoreCollision for the ball's path d and colliders/times [W, ..]: a contact at the
        path's latest point (within the fuzzy delta) with a collider that already split the path there."""
        n = d["n"]
        lidx = torch.arange(self.L, device=self.dev)
        lower = torch.gather(d["t"], 1, (n - 2).clamp(min=0)[:, None])[:, 0] - FUZZY_DELTA
        near = (lidx[None, :] <= (n - 2)[:, None]) & (d["t"] >= lower[:, None])
        extra = collider.dim() - 1
        near = near.view(self.W, *([1] * extra), self.L)
        cc = d["c"].view(self.W, *([1] * extra), self.L)
        hit = (near & (cc == collider[..., None])).any(-1)
        return (n.view(self.W, *([1] * extra)) >= 2) & ~(toi > FUZZY_DELTA) & hit

    def _add_impact_path(self, jj, mask, time, collider, dv) -> tuple:
        """CollisionBallProperties::AddImpact: the impact's point replaces every later one; the replaced points' impacts
        are returned (newest first) as (times, colliders, valid) [W, L]."""
        d = self._props(jj)
        go = mask & ~(d["te"] <= time)
        lidx = torch.arange(self.L, device=self.dev)
        valid = lidx[None, :] < d["n"][:, None]
        index = ((d["t"] <= time[:, None]) & valid).sum(-1)
        inv = go[:, None] & valid & (lidx[None, :] >= index[:, None]) & (d["c"] != -1)
        out = (torch.flip(d["t"], [1]), torch.flip(d["c"], [1]), torch.flip(inv, [1]))
        newn = index + 1
        grow = go & (newn > d["cap"])
        d["cap"] = torch.where(grow, d["n"] + torch.maximum(d["n"], newn - d["n"]), d["cap"])
        prev = (index - 1).clamp(min=0)
        pt = torch.gather(d["t"], 1, prev[:, None])[:, 0]
        i3 = prev[:, None, None].expand(-1, 1, 3)
        pp, pv = torch.gather(d["p"], 1, i3)[:, 0], torch.gather(d["v"], 1, i3)[:, 0]
        m, tf, a = self._ball_terms(jj)
        p, v = self._integrate_any(pp, pv, a, m, tf, time - pt)
        first = (index == 0)[:, None]
        p = torch.where(first, self._sel(self.pos, jj), p)
        v = torch.where(first, self._sel(self.vel, jj), v) + dv
        self._write_loc(d, index, go, time, collider, p, v)
        d["n"] = torch.where(go, newn, d["n"])
        d["te"] = torch.where(go, time, d["te"])
        self._set_props(jj, go, d)
        return out

    def _remove_impact_path(self, jj, mask, time, collider) -> tuple:
        """CollisionBallProperties::RemoveImpact: the path cut at the impact (or, not found, back to its start with
        the ball's own velocity: `found` starts true), then CalculateEnd; the cut points' impacts returned in order."""
        d = self._props(jj)
        lidx = torch.arange(self.L, device=self.dev)
        valid = lidx[None, :] < d["n"][:, None]
        match = valid & (d["c"] == collider[:, None]) & ((d["t"] - time[:, None]).abs() < FUZZY_DELTA)
        index = torch.where(match.any(-1), match.to(I64).argmax(-1), torch.zeros_like(d["n"]))
        z = mask & (index == 0)
        d["c"] = torch.where(z[:, None] & (lidx[None, :] == 0), -1, d["c"])
        vel = self._sel(self.vel, jj)
        d["v"] = torch.where((z[:, None] & (lidx[None, :] == 0))[..., None], vel[:, None, :], d["v"])
        inv = mask[:, None] & valid & (lidx[None, :] > index[:, None]) & (d["c"] != -1)
        out = (d["t"].clone(), d["c"].clone(), inv)
        d["n"] = torch.where(mask, torch.where(index > 0, index, torch.ones_like(index)), d["n"])
        self._set_props(jj, mask, d)
        self._calculate_end(jj, mask)
        return out

    def _free_slot(self, ids: torch.Tensor) -> tuple:
        """The slot of each id [W, K] among the world's free balls, and whether there is one (in mFreeBalls)."""
        hit = (self.id[:, None, :] == ids[..., None]) & (self.exists & self.free)[:, None, :]
        return hit.to(I64).argmax(-1), hit.any(-1)

    def _push(self, mask, slots, times, colliders) -> None:
        """Push one list per world [W, K] (in processing order, where mask) so that it pops in that order."""
        k = mask.sum(-1)
        rank = torch.cumsum(mask.to(I64), -1) - 1
        at = self.sp[:, None] + (k[:, None] - 1 - rank)
        self.stack_overflow |= (mask & (at >= self.S)).any(-1)
        at = torch.where(mask, at.clamp(max=self.S - 1), torch.full_like(at, self.S))
        self.st_slot = torch.scatter(self.st_slot, 1, at, slots)
        self.st_t = torch.scatter(self.st_t, 1, at, times)
        self.st_c = torch.scatter(self.st_c, 1, at, colliders.expand_as(slots))
        self.sp = self.sp + k

    def _add_impact(self, jj, mask, time, collider, dv) -> None:
        """Ballpark::AddImpact (Ballpark.cpp:2577; COL-718): the velocity change limited to 5 maxVel, then the path's
        AddImpact, then RemoveImpactFromFree (2647) for each impact it replaced whose collider is free, recursively,
        depth first in the scalar code's order (an explicit stack per world)."""
        d = self._props(jj)
        go = mask & ~(d["te"] <= time)
        if not bool(go.any()):
            return
        mv = self._sel(self.max_vel, jj)
        limit = (VELOCITY_LIMIT * mv).to(torch.float32).to(F64)          # 25 maxVel maxVel in float
        limit = (limit * mv).to(torch.float32).to(F64)
        frac = limit / _dot(dv, dv)
        dv = torch.where((go & (frac < 1))[:, None], dv * frac[:, None], dv)
        inv_t, inv_c, inv_m = self._add_impact_path(jj, go, time, collider, dv)
        slots, free = self._free_slot(inv_c)
        self._push(inv_m & free, slots, inv_t, self._sel(self.id, jj)[:, None])
        while bool((self.sp > 0).any()):
            live = self.sp > 0
            self.sp = self.sp - live.to(I64)
            top = self.sp.clamp(min=0)[:, None]
            slot = torch.gather(self.st_slot, 1, top)[:, 0]
            t = torch.gather(self.st_t, 1, top)[:, 0]
            c = torch.gather(self.st_c, 1, top)[:, 0]
            r_t, r_c, r_m = self._remove_impact_path(slot, live, t, c)
            slots, free = self._free_slot(r_c)
            self._push(r_m & free, slots, r_t, self._sel(self.id, slot)[:, None])

    def _candidate_lists(self, resp: torch.Tensor) -> tuple:
        """Each responder's candidates (GetNearbyBalls, collisionCase1) as slots in order [W, R, N - 1], -1 padded."""
        R = self.R
        candidate = self.exists & self.massive & ~self.cloaked
        eye = torch.eye(self.N, dtype=BOOL, device=self.dev)[None, :R]
        pair = resp[:, :R, None] & candidate[:, None, :] & ~eye & self._related()
        if self.bubble is not None:                                         # the master's filter
            pair = pair & self._same_bubble()
        perm = self._sorted(pair, self._order_keys(pair))
        ok = torch.gather(pair, -1, perm)
        n = ok.sum(-1)
        width = max(1, int(n.max())) if self.R < self.N else max(1, self.N - 1)    # with rows: the padding cut off
        return torch.where(ok, perm, torch.full_like(perm, -1))[..., :width], n

    def _response(self, j: int, act: torch.Tensor) -> torch.Tensor:
        """CalculateCollisionResponse (Ballpark.cpp:1988) for slot j in the worlds where act; True where an impact was
        found (a phantom impact, collider 0, included: COL-714's finding)."""
        W, dev = self.W, self.dev
        jj = torch.full((W,), j, dtype=I64, device=dev)
        d = self._props(jj)
        a_end = act & (d["ts"] >= d["te"])
        if bool(a_end.any()):
            self._calculate_end(jj, a_end)
        go = act & ~a_end & (d["c"][:, 0] == -1) & (self._ncand[:, j] > 0)
        found = torch.zeros(W, dtype=BOOL, device=dev)
        if not bool(go.any()):
            return found
        d = self._props(jj)
        ts, te = d["ts"], d["te"]
        P, _ = self._position(d, jj, torch.stack([ts, te], 1))
        p0, p1 = P[:, 0], P[:, 1]
        cand = self._cand[:, j, :]
        C = cand.shape[1]
        cvalid = cand >= 0
        cs = cand.clamp(min=0)
        g = lambda x: torch.gather(x, 1, cs if x.dim() == 2 else cs[..., None].expand(-1, -1, x.shape[-1]))
        free_n, junk_n, mass_n = g(self.free), g(self.junk), g(self.massive)
        pos_n, rad_n, id_n = g(self.pos), g(self.radius), g(self.id)
        rad_b, id_b, junk_b = self.radius[:, j], self.id[:, j], self.junk[:, j]
        ignore_n = (junk_n & ~junk_b[:, None]) | ~mass_n                    # CollisionBallNeighborIgnore
        # GetNextCollisionStatic (Ballpark.cpp:2080): fixed balls in order; its test compares the fraction with the
        # scaled time held, and it stops at a contact at the path's start
        toi = self._collide_two_spheres(p0[:, None], p1[:, None], pos_n, pos_n, rad_b[:, None] + rad_n)
        ign = self._ignore_collision(d, id_n, toi)
        tt = ts[:, None] + toi * (te - ts)[:, None]
        st_ok = cvalid & ~free_n & ~ignore_n & (toi >= 0.0) & ~ign
        item_t = torch.full((W,), -1.0, dtype=F64, device=dev)
        item_slot = torch.full((W,), -1, dtype=I64, device=dev)
        item_toi = torch.zeros(W, dtype=F64, device=dev)
        item_static = torch.zeros(W, dtype=BOOL, device=dev)
        brk = ~go
        for c in range(C):
            live = ~brk & cvalid[:, c]
            take = live & st_ok[:, c] & ((item_t < 0.0) | (toi[:, c] < item_t))
            item_t = torch.where(take, tt[:, c], item_t)
            item_slot = torch.where(take, cs[:, c], item_slot)
            item_toi = torch.where(take, toi[:, c], item_toi)
            item_static = item_static | take
            brk = brk | (live & (item_t == ts))
        # GetNextCollisionFree (Ballpark.cpp:2180): free balls of higher id, each sub-stepped at the other's points;
        # a candidate's first unignored contact t* (after an ignored one the next step starts where it did) is taken
        # if earlier than the item held, in order
        go_f = go & (item_t != ts)
        te_n, ln_n = g(self.t_end), g(self.loc_n)
        elig = go_f[:, None] & cvalid & free_n & (id_n > id_b[:, None]) & ~ignore_n
        end = torch.where(te[:, None] <= te_n, te[:, None], te_n)
        elig = elig & (ts[:, None] < end)
        if bool(elig.any()):
            Lr = int(self.loc_n.max())                                     # the room in use: the rest is padding
            lt_n = torch.gather(self.loc_t[..., :Lr], 1, cs[..., None].expand(-1, -1, Lr))
            lidx = torch.arange(Lr, device=dev)
            mid = (elig[..., None] & (lidx < ln_n[..., None]) & (lt_n > ts[:, None, None])
                   & (lt_n < end[..., None]))
            nmid = mid.sum(-1)
            T = int(nmid.max()) + 2                                         # the longest list of sub-step times
            times = torch.zeros(W, C, T, dtype=F64, device=dev)
            at = torch.where(mid, torch.cumsum(mid.to(I64), -1), torch.full_like(lt_n, T, dtype=I64))
            times = torch.cat([times, torch.zeros(W, C, 1, dtype=F64, device=dev)], -1)
            times = torch.scatter(times, 2, at, lt_n)[..., :T]
            times[..., 0] = ts[:, None]
            times = torch.scatter(times, 2, (nmid + 1)[..., None], end[..., None])
            nt = nmid + 2
            m_j, tf_j, a_j = self._ball_terms(jj)
            pj, _ = self._position_at(d["t"][:, :Lr], d["p"][:, :Lr], d["v"][:, :Lr], d["n"], m_j, tf_j, a_j,
                                      times.view(W, C * T))
            pj = pj.view(W, C, T, 3)
            dn = {k: torch.gather(v, 1, cs.view(W, C, *([1] * (v.dim() - 2))).expand(W, C, *v.shape[2:]))
                  for k, v in (("t", self.loc_t[..., :Lr]), ("p", self.loc_p[..., :Lr, :]),
                               ("v", self.loc_v[..., :Lr, :]), ("n", self.loc_n))}
            pn, _ = self._position_at(dn["t"], dn["p"], dn["v"], dn["n"], g(self.mass * self.agility),
                                      g(self.time_factor), g(self._accel()), times)
            b0, o0 = pj[:, :, 0], pn[:, :, 0]
            stopped = ~elig
            tstar = torch.full((W, C), -1.0, dtype=F64, device=dev)
            fnd = torch.zeros(W, C, dtype=BOOL, device=dev)
            cr = rad_b[:, None] + rad_n
            for ii in range(int(nt.max()) - 1):
                act_ii = ~stopped & (ii < nt - 1)
                b1, o1 = pj[:, :, ii + 1], pn[:, :, ii + 1]
                tq = self._collide_two_spheres(b0, b1, o0, o1, cr)
                t0, t1 = times[..., ii], times[..., ii + 1]
                tc = t0 + (t1 - t0) * tq
                contact = act_ii & (tq >= 0.0)
                ig = self._ignore_collision(d, id_n, tq)
                f = contact & ~ig
                tstar = torch.where(f, tc, tstar)
                fnd = fnd | f
                stopped = stopped | f
                upd = (act_ii & ~f & ~(contact & ig & self.KEEP_START_AFTER_IGNORED) & (nt > 2))[..., None]
                b0, o0 = torch.where(upd, b1, b0), torch.where(upd, o1, o0)
            brk2 = ~go_f
            for c in range(C):
                live = ~brk2 & elig[:, c]
                take = live & fnd[:, c] & ((item_t < 0.0) | (tstar[:, c] < item_t))
                item_t = torch.where(take, tstar[:, c], item_t)
                item_slot = torch.where(take, cs[:, c], item_slot)
                item_static = item_static & ~take
                brk2 = brk2 | (live & (item_t == ts))
        has = go & (item_slot >= 0)
        stat, fr = has & item_static, has & ~item_static
        osl = item_slot.clamp(min=0)
        k = self.friction
        vel_impact = torch.zeros(W, 3, dtype=F64, device=dev)
        col_b = torch.zeros(W, dtype=I64, device=dev)
        if bool(stat.any()):
            # CalculateCollisionImpactStatic (Ballpark.cpp:2370): touching at the start pushes away (mLastC) first
            q = p0 + (p1 - p0) * item_toi[:, None]
            normal = _normalized(q - self._sel(self.pos, osl))
            touch = stat & (ts == item_t)
            lg = self._sel(self.last_g, jj)
            max_acc = (self.max_vel[:, j] * k) / (self.mass[:, j] * self.agility[:, j]) - _dot(lg, normal)
            self.last_c = self._put(self.last_c, jj, touch, normal * max_acc[:, None])
            _, v = self._position(self._props(jj), jj, item_t[:, None])
            impulse = _dot(v[:, 0] * -2.0, normal)
            real = stat & ~(impulse < 0.0)
            vel_impact = torch.where(real[:, None], normal * impulse[:, None], vel_impact)
            col_b = torch.where(real, self._sel(self.id, osl), col_b)
        if bool(fr.any()):
            # CalculateCollisionImpactFree (Ballpark.cpp:2442): an elastic exchange with base masses
            do = self._props(osl)
            bp, bv = self._position(self._props(jj), jj, item_t[:, None])
            op, ov = self._position(do, osl, item_t[:, None])
            bp, bv, op, ov = bp[:, 0], bv[:, 0], op[:, 0], ov[:, 0]
            nrm = bp - op
            length = torch.sqrt(_dot(nrm, nrm))
            pos_len = length > 0.0
            cinv = 1.0 / torch.where(pos_len, length, torch.ones_like(length))
            up = torch.stack([nrm[:, 0], nrm[:, 1], torch.ones_like(length)], -1)
            normal = torch.where(pos_len[:, None], nrm * cinv[:, None], up)
            onormal = -normal
            touch0 = fr & (item_t == 0.0)
            lg_b, lg_o = self._sel(self.last_g, jj), self._sel(self.last_g, osl)
            acc_b = (self.max_vel[:, j] * k) / (self.mass[:, j] * self.agility[:, j]) - _dot(lg_b, normal)
            acc_o = ((self._sel(self.max_vel, osl) * k) / (self._sel(self.mass, osl) * self._sel(self.agility, osl))
                     - _dot(lg_o, onormal))
            self.last_c = self._put(self.last_c, jj, touch0, normal * acc_b[:, None])
            self.last_c = self._put(self.last_c, osl, touch0, onormal * acc_o[:, None])
            impulse = -2.0 * _dot(ov - bv, onormal)
            real = fr & ~(impulse < 0.0)
            m1, m2 = self.mass[:, j], self._sel(self.mass, osl)
            jv = (m1 * m2 * impulse) / (m1 + m2)
            c1, c2 = jv / m1, jv / m2
            dv_o = torch.where(real[:, None], onormal * c2[:, None], torch.zeros_like(normal))
            col_o = torch.where(real, id_b, torch.zeros_like(id_b))
            vel_impact = torch.where(real[:, None], normal * c1[:, None], vel_impact)
            col_b = torch.where(real, self._sel(self.id, osl), col_b)
            self._add_impact(osl, fr, item_t, col_o, dv_o)                  # the neighbour's impact first
        if bool(has.any()):
            self._add_impact(jj, has, item_t, col_b, vel_impact)
        self._calculate_end(jj, go)
        return has

    def _iterative(self, live: torch.Tensor) -> None:
        """CalculateIterativeCollisionResponses (Ballpark.cpp:332; COL-714): every free, massive, live ball once in
        ascending id; then, up to 20 passes in all, the balls with a new impact and their free candidates."""
        resp = live & self.massive
        self._cand, self._ncand = self._candidate_lists(resp)
        valid = self._cand >= 0                                             # padding goes to a spare column
        at = torch.where(valid, self._cand, torch.full_like(self._cand, self.N))
        free = torch.gather(self.free[:, None, :].expand(-1, self.R, -1), 2, self._cand.clamp(min=0))
        spare = torch.zeros(self.W, self.R, self.N + 1, dtype=BOOL, device=self.dev)
        self._free_cand = torch.scatter(spare, 2, at, valid & free)[..., :self.N]
        self._init_props(resp)
        self.has_prop = self.has_prop | resp
        self.S = self.N * self.L
        self.st_slot = torch.zeros(self.W, self.S + 1, dtype=I64, device=self.dev)
        self.st_t = torch.zeros(self.W, self.S + 1, dtype=F64, device=self.dev)
        self.st_c = torch.zeros(self.W, self.S + 1, dtype=I64, device=self.dev)
        self.sp = torch.zeros(self.W, dtype=I64, device=self.dev)
        current = live.clone()
        for _ in range(COLLISION_MAX_ITERATIONS):
            self.passes += 1                                                # passes run (any world), for probes
            following = torch.zeros_like(current)
            for j in range(self.R):                                         # the slots that may hold a free ball
                act = current[:, j] & resp[:, j]
                if not bool(act.any()):
                    continue
                found = self._response(j, act)
                following = following | (found[:, None] & self._free_cand[:, j, :])
                following[:, j] |= found
            if not bool(following.any()):
                break
            current = following

    def _iterative_end(self, live: torch.Tensor) -> tuple:
        """CalculateBallPositionVelocity (Ballpark.cpp:2715): from the ball's collision path if it has one."""
        m = self.mass * self.agility
        pn, vn = self._integrate_dt(self.pos, self.vel, self._accel(), m, self.time_factor)
        pp, pv = self._position_at(self.loc_t, self.loc_p, self.loc_v, self.loc_n, m, self.time_factor,
                                   self._accel(), torch.full((self.W, self.N, 1), self.dt, dtype=F64,
                                                             device=self.dev))
        use = (self.has_prop & live)[..., None]
        return torch.where(use, pp[:, :, 0], pn), torch.where(use, pv[:, :, 0], vn)

    # ================================================================================================= checks

    def check(self) -> None:
        """Raise if a fixed contact_slots or the iterative method's room was exceeded in any world since the load (an
        overflow is an event, never a silent drop)."""
        if bool(self.overflow.any()):
            raise RuntimeError(f"contact_slots={self.contact_slots} was exceeded in "
                               f"{int(self.overflow.sum())} worlds")
        if bool(self.impact_overflow):
            raise RuntimeError(f"impact_slots={self.impact_slots} was exceeded in a tick")
        room = (self.loc_overflow | self.stack_overflow) & ~self.undefined
        if bool(room.any()):
            raise RuntimeError(f"the iterative method's room (locations={self.L}) was exceeded in "
                               f"{int(room.sum())} worlds")

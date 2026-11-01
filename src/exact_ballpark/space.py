# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's space partition, bubbles, proximity sensors, visibility, range queries, collisions (the gradient and the
iterative method) and ball removal, added to the scalar movement reference (modes.py), one ball at a time.

A transcription of Carbon's `destiny` (github.com/carbonengine/destiny at 114e89fa, MIT, (c) CCP Games), as ref.py and modes.py. `SpaceBallpark` extends `ModeBallpark` and fills slice 1's tick hooks
with Destiny's code: the dead-bubble skip, the iterative collision responses or the collision gradient, the paths
through the tick's impacts, the partition re-insertions and HandleProximities, in `Ballpark::Evolve`'s order
(Ballpark.cpp:421). Every container whose order Destiny's results depend on keeps Destiny's order: the free balls by
id (a std::map), a box's balls and children and a ball's boxes by key (std::set with Destiny's comparers), the balls
and the bubble dictionaries in insertion order (Python dicts, as CPython's), and the proximity balls and each sensor's
list in libstdc++'s unordered_map order (`StdHashMap`, the order the compiled code walks). Rules implemented, by rule id:

- SEN-705: 8 levels of cubic boxes, 7,864,320 m down to 480 m, centred on the origin; a box's parent is at the indices
  shifted right by 2 (`Partition::Partition`, `GetBox`, `Box::Box`, `Box::SetParent`; Partition.cpp:19, Box.cpp:15, 95).
- SEN-706: a ball is filed at the finest level whose box width is at least 3x its inflated radius, plus the adjacent
  boxes its inflated sphere reaches (up to 8), or the 27 top-level boxes around it when its diameter exceeds the top
  width; boxes are created on demand and released when empty (`Ball::GetInflatedRadius`, `Ball::InsertInBoxes`,
  `Ballpark::InsertInBoxes`, `Box::RemoveBall`, `Box::RemoveChildren`; Ball.cpp:838, 2354, Ballpark.cpp:3115).
- SEN-707: bubbles are top-level boxes; on the server a ball entering a top box without a bubble takes a neighbour's
  or a new one; InitializeBubbles seed-fills from each ball's top box in ball order; the bubble dictionaries
  (additions +1, kept 0, removals -1) and the interactive and keep-alive sets follow every change (`AddToBubble`,
  `InitializeBubbles`, `UpdateBallBubble`, `CopyBubbles`, `Box::Traverse`, `Box::NearbyBubbles`; Ballpark.cpp:6429,
  6565, 5959, 6154, Box.cpp:216, 252).
- SEN-708: on the server, collision and proximity candidates share the ball's bubble unless global; follow and orbit
  refuse other-bubble targets (modes.py); followers of a ball in another bubble are stopped at the pre-tick
  (`clean_up_followers`, MOV-731); CheckVisibility between bubbles is -2.
- SEN-701 to SEN-704: proximity sensors (period, shuffle, candidates from the partition with the sensor's radius
  inflated, entering and leaving), target tracking and the notification range, run by HandleProximities at the end of
  the tick (`Ball::CheckForProximity*`, `Ballpark::HandleProximities`; Ball.cpp:494-775, Ballpark.cpp:5822).
- SEN-709: CheckVisibility (Ballpark.cpp:5299). SEN-710: GetBallIdsInRange and GetBallIdsAndDistInRange from a ball or
  a point (Thunkers.cpp:1344).
- COL-701 to COL-711: the collision switch (gradient by default, iterative per park), only free, massive, live balls
  responding, the candidates (no missiles, cloaked or non-massive balls; other bubbles on the server unless global;
  force fields letting matching balls through; space junk one-sided), and the gradient method (`Gradient`,
  `Potential`, `CollideTwoSpheres`, `Quadratic`; Ballpark.cpp:2745, 2789, Collision.cpp:108, 137).
- COL-714 to COL-718: the iterative method (`CalculateIterativeCollisionResponses`, `CalculateCollisionResponse`,
  `GetNextCollisionStatic`, `GetNextCollisionFree`, `CalculateCollisionImpactStatic`, `CalculateCollisionImpactFree`,
  `AddImpact`, `RemoveImpactFromFree`, `CollisionBallProperties`; Ballpark.cpp:332, 1988-2673), with balls and fixed
  balls (no static collidables) and dynamical orientation off.
- MOV-730's removal trigger and TIME-713 (`RemoveBall`, `BringOutTheDeads`; Ballpark.cpp:5471, 5699), MOV-731
  (`PyGetRemoteFollowers`, then a Stop for each, as `Actions.clean_up_followers`), MOV-732 (only the free set is
  evolved; SetBallFree), MOV-733 (the flags), TIME-706 and TIME-707 (the whole tick), TIME-714 (dead bubbles).

Not here: static collidables (capsules, oriented boxes, miniballs: COL-712, COL-713), missiles and their collision
events (COL-719), cloaking (SEN-711), dynamical orientation and the orientation itself. Degenerate inputs follow IEEE
where Destiny divides by zero (a massless free ball's inflated radius and collision response). Ball ids must be
positive: Destiny sorts negative (local) ids by their owner first.

Where Destiny's own result is undefined the reference refuses (DestinyUndefined) instead of choosing one: a released
partition box is read and written while a ball is re-filed into a top box without a bubble (Ballpark.cpp:3141-3185,
Ball.cpp:2365), defined while BoxAllocator keeps the box's memory (it keeps 256) and undefined once it has deleted it;
and CollisionBallProperties::CalculateEnd keeps a reference across a push_back that reallocates
(CollisionBallProperties.cpp:173). BringOutDeadBalls walks an unordered_set of pointers (heap order, Ballpark.cpp:5716):
the reference walks ascending id. That order decides the order of one call's removals and their events, and which
balls the kill count (7 a call) admits when more than seven are within the two-tick buffer; neither is reproducible.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from .modes import (FIELD, FOLLOW, FOLLOW_MODES, MINIBALL, MISSILE, MUSHROOM, ORBIT, WARP, ModeBall, ModeBallpark,
                    _float32, _ieee_div, _lensq)
from .ref import GOTO, STOP, _normalized, f32

NUM_LEVELS = 8                                              # Partition.cpp:20 (SEN-705)
GRID_UNIT = 480.0                                           # Partition.cpp:21
_BIGBOX = (1 << (2 * NUM_LEVELS)) * GRID_UNIT * 0.25        # Partition.cpp:26
LEVEL_WIDTH = tuple(_BIGBOX / (1 << 2 * i) for i in range(NUM_LEVELS))                 # Partition.cpp:32
LEVEL_GRID = tuple(1 << (2 * i - 2 * NUM_LEVELS + 39) for i in range(NUM_LEVELS))      # Partition.cpp:35
LOCAL_BALLS = -1073741824                                   # DSTLOCALBALLS (IDstConstants.h:76)
BOX_CHUNK_SIZE = 256                                        # BoxAllocator: released boxes kept (Partition.cpp:520)
LIGHT_BALL_RATIO = 25.0                                     # Ballpark.cpp:2948 (COL-710)
RESPONSE_SCALE = 0.85                                       # Ballpark.cpp:2985 (COL-711)
COLLISION_MAX_ITERATIONS = 20                               # g_collisionMaxIterationsDefault (Settings.h:9; COL-714)
FUZZY_DELTA = 1e-4                                          # CollisionBallProperties::m_fuzzyDelta (COL-715)
VELOCITY_LIMIT = 25.0                                       # (5 x maxVel)^2 in AddImpact (Ballpark.cpp:2600; COL-718)
LOCATIONS_RESERVED = COLLISION_MAX_ITERATIONS + 2           # m_locations.reserve (CollisionBallProperties.cpp:12)
_MASK = (1 << 64) - 1


class DestinyUndefined(Exception):
    """Destiny's result is undefined here: it reads or writes a box whose memory it has returned to the allocator."""


def _to_int64(x: float) -> int:
    """`int64_t(x)`: truncation toward zero; outside the range C++ leaves it undefined and aarch64 (fcvtzs) saturates,
    with NaN giving 0, which is what the compiled code does on this machine."""
    if x != x:
        return 0
    if x >= 9223372036854775808.0:
        return 9223372036854775807
    if x < -9223372036854775808.0:
        return -9223372036854775808
    return int(x)


def _clamp_key(ix: int, iy: int, iz: int, n: int) -> tuple:
    """Key::Clamp (Partition.h:87): each index into [0, n - 1]."""
    return tuple(0 if i < 0 else n - 1 if i >= n else i for i in (ix, iy, iz))


def _dot(u, v) -> float:
    """Vector3d's dot product, x first (Vector3d.h:49)."""
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _sub(u, v) -> list:
    return [u[0] - v[0], u[1] - v[1], u[2] - v[2]]


def _cdiv(a: int, b: int) -> int:
    """C's integer division (toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def key_hash(ix: int, iy: int, iz: int, n: int) -> int:
    """KeyHash (Partition.h:104): the boxes' hash in Destiny's maps (no result here depends on it)."""
    seed = 0
    for v in (ix, iy, iz, n):
        seed ^= ((v & _MASK) + 0x9E3779B9 + ((seed << 6) & _MASK) + (seed >> 2)) & _MASK
    return seed


def box_center(level: int, pos) -> list:
    """Partition::GetBoxCenter (Partition.cpp:152): the centre of the box at a level holding a point."""
    w = LEVEL_WIDTH[level]
    fac, n = 1.0 / w, LEVEL_GRID[level]
    return [_to_int64((pos[i] + 0.5 * w * n) * fac) * w + w * 0.5 - w * n * 0.5 for i in range(3)]


from .glibc_rand import GlibcRandom  # noqa: E402  (glibc's rand(): LGPL-2.1-or-later, its own module)


# ---------------------------------------------------------------------------- libstdc++'s unordered containers
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
# A clean-room implementation, written from a behavioural specification (the rules in words, the data tables) without
# the C++ library's code. Destiny keeps a sensor's balls, the proximity balls and a ball's followers in
# std::unordered_map<int64_t, ...> and std::unordered_set<int64_t> and walks them in their iteration order, so the
# reference reproduces three things of GCC's libstdc++ for any number of keys: the keys' iteration order, the bucket
# count and the growth threshold (the next resize). The order depends on the bucket count, and the bucket count on
# the container's growth history (a cleared container keeps its buckets).
#
# The bucket counts a container can take: the library's __prime_list (a 64-bit target), 304 primes in increasing
# order and then the last one again (a sentinel). Data, read out of the pinned library, not computed.
_PRIME_LIST = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 103, 109,
    113, 127, 137, 139, 149, 157, 167, 179, 193, 199, 211, 227, 241, 257, 277, 293, 313, 337, 359, 383, 409, 439, 467,
    503, 541, 577, 619, 661, 709, 761, 823, 887, 953, 1031, 1109, 1193, 1289, 1381, 1493, 1613, 1741, 1879, 2029, 2179,
    2357, 2549, 2753, 2971, 3209, 3469, 3739, 4027, 4349, 4703, 5087, 5503, 5953, 6427, 6949, 7517, 8123, 8783, 9497,
    10273, 11113, 12011, 12983, 14033, 15173, 16411, 17749, 19183, 20753, 22447, 24281, 26267, 28411, 30727, 33223,
    35933, 38873, 42043, 45481, 49201, 53201, 57557, 62233, 67307, 72817, 78779, 85229, 92203, 99733, 107897, 116731,
    126271, 136607, 147793, 159871, 172933, 187091, 202409, 218971, 236897, 256279, 277261, 299951, 324503, 351061,
    379787, 410857, 444487, 480881, 520241, 562841, 608903, 658753, 712697, 771049, 834181, 902483, 976369, 1056323,
    1142821, 1236397, 1337629, 1447153, 1565659, 1693859, 1832561, 1982627, 2144977, 2320627, 2510653, 2716249, 2938679,
    3179303, 3439651, 3721303, 4026031, 4355707, 4712381, 5098259, 5515729, 5967347, 6456007, 6984629, 7556579, 8175383,
    8844859, 9569143, 10352717, 11200489, 12117689, 13109983, 14183539, 15345007, 16601593, 17961079, 19431899,
    21023161, 22744717, 24607243, 26622317, 28802401, 31160981, 33712729, 36473443, 39460231, 42691603, 46187573,
    49969847, 54061849, 58488943, 63278561, 68460391, 74066549, 80131819, 86693767, 93793069, 101473717, 109783337,
    118773397, 128499677, 139022417, 150406843, 162723577, 176048909, 190465427, 206062531, 222936881, 241193053,
    260944219, 282312799, 305431229, 330442829, 357502601, 386778277, 418451333, 452718089, 489790921, 529899637,
    573292817, 620239453, 671030513, 725980837, 785430967, 849749479, 919334987, 994618837, 1076067617, 1164186217,
    1259520799, 1362662261, 1474249943, 1594975441, 1725587117, 1866894511, 2019773507, 2185171673, 2364114217,
    2557710269, 2767159799, 2993761039, 3238918481, 3504151727, 3791104843, 4101556399, 4294967291, 6442450933,
    8589934583, 12884901857, 17179869143, 25769803693, 34359738337, 51539607367, 68719476731, 103079215087,
    137438953447, 206158430123, 274877906899, 412316860387, 549755813881, 824633720731, 1099511627689, 1649267441579,
    2199023255531, 3298534883309, 4398046511093, 6597069766607, 8796093022151, 13194139533241, 17592186044399,
    26388279066581, 35184372088777, 52776558133177, 70368744177643, 105553116266399, 140737488355213, 211106232532861,
    281474976710597, 562949953421231, 1125899906842597, 2251799813685119, 4503599627370449, 9007199254740881,
    18014398509481951, 36028797018963913, 72057594037927931, 144115188075855859, 288230376151711717, 576460752303423433,
    1152921504606846883, 2305843009213693951, 4611686018427387847, 9223372036854775783, 18446744073709551557,
    18446744073709551557)

_SIZE_MAX = (1 << 64) - 1          # size_t's largest value: the next resize once the list is exhausted
_TWO64 = float(1 << 64)            # 2**64 as a double: the first value past size_t

# The library the data and the behaviour are held to (tests/test_movement_space_hashtable.py checks the probe's
# libstdc++.so.6 against the SHA-256).
LIBSTDCXX_PIN = {"gcc": "14.3.0", "library": "libstdc++.so.6.0.33",
                 "sha256": "ac709fc594acdcdf9b19868b3ad7d7275815018f82d39f028710e3706407022f"}


def _floor_to_size_t(x: float) -> int:
    """floor(x) of a non-negative double, as a size_t; from 2**64 up it saturates at 2**64 - 1 (aarch64's conversion;
    no count of the containers reaches it)."""
    return _SIZE_MAX if x >= _TWO64 else math.floor(x)


class StdHashMap:
    """std::unordered_map<int64_t, V> (std::hash: the identity) under the prime rehash policy, as far as the walk
    goes: `order` holds the keys in iteration order, `values` each key's value (never part of the order), and
    `bucket_count` and `next_resize` the policy's state. A fresh map has 1 bucket and next resize 0.

    An absent key is placed by the linking rule: immediately before the first key of the order that shares its bucket,
    or at the front when none does; before that the policy is asked for room for one more key, and when it grows the
    keys are re-linked one at a time in their old order. Erasures keep the others' order; nothing ever shrinks."""

    _FAST_BKT = (2, 2, 2, 3, 5, 5, 7, 7, 11, 11, 11, 11, 13, 13)   # the bucket count for 1 to 13 buckets, by n
    _SEARCH_FROM = 6               # the first index of _PRIME_LIST searched (its entry 17)
    _FIRST_MIN = 11                # the keys a never-grown map asks buckets for: its first insertion makes 13
    _MAX_LOAD = 1.0                # max_load_factor: fixed (1.0f, exact as a double)
    _GROWTH = 2                    # a growth asks for at least this multiple of the bucket count

    def __init__(self):
        self.order: list = []
        self.values: dict = {}
        self.bucket_count = 1
        self.next_resize = 0

    def __contains__(self, k) -> bool:
        return k in self.values

    def __len__(self) -> int:
        return len(self.order)

    def keys(self) -> list:
        """The keys in iteration order, as a new list (a walk over it survives changes to the map)."""
        return list(self.order)

    def copy(self) -> StdHashMap:
        """The copy constructor: the same keys in the same order, the same values, bucket count and next resize."""
        m = type(self)()
        m.order, m.values = list(self.order), dict(self.values)
        m.bucket_count, m.next_resize = self.bucket_count, self.next_resize
        return m

    def insert(self, k, v) -> bool:
        """insert(): a present key changes nothing (its value kept); an absent one is linked. True when inserted."""
        if k in self.values:
            return False
        self._insert_node(k)
        self.values[k] = v
        return True

    def set(self, k, v) -> None:
        """operator[] and an assignment: an absent key is linked as insert() links it; the value is replaced."""
        if k not in self.values:
            self._insert_node(k)
        self.values[k] = v

    def erase(self, k) -> bool:
        """erase(key): the other keys keep their order, the bucket count and next resize are kept. True when erased."""
        if k not in self.values:
            return False
        del self.values[k]
        self.order.remove(k)
        return True

    def clear(self) -> None:
        """clear(): every key removed; the bucket count and next resize are kept (a cleared map is not a fresh one)."""
        self.order.clear()
        self.values.clear()

    # -- the policy's steps, by these names (the batched layer reads the constants; the tests call and replace these)

    def _bucket(self, k) -> int:
        """A key's bucket: its 64-bit two's-complement pattern as an unsigned integer, modulo the bucket count."""
        return (k & _SIZE_MAX) % self.bucket_count

    def _next_bkt(self, n: int) -> int:
        """The bucket count for at least n buckets; sets next_resize (left as it is for n = 0)."""
        if n < len(self._FAST_BKT):
            if n == 0:
                return 1
            b = self._FAST_BKT[n]
            self.next_resize = _floor_to_size_t(float(b) * self._MAX_LOAD)
            return b
        end = len(_PRIME_LIST) - 2                          # the search stops before the last prime (the sentinel's)
        j = bisect.bisect_left(_PRIME_LIST, n, self._SEARCH_FROM, end)
        if j == end:                                        # every entry searched is below n: never grow again
            self.next_resize = _SIZE_MAX
            return _PRIME_LIST[end]
        b = _PRIME_LIST[j]
        self.next_resize = _floor_to_size_t(float(b) * self._MAX_LOAD)   # past 2**53: the double nearest the prime
        return b

    def _need_rehash(self, n_bkt: int, n_elt: int, n_ins: int) -> tuple:
        """With n_bkt buckets, n_elt keys present and n_ins to insert: (True, the new bucket count) to grow, else
        (False, 0). Reads next_resize and may set it. Counts are size_t (modulo 2**64); m and the wanted count's
        first term are IEEE doubles (round to nearest), so past 2**53 the rounding is the library's."""
        total = (n_elt + n_ins) & _SIZE_MAX
        if total <= self.next_resize:
            return False, 0
        floor_keys = self._FIRST_MIN if self.next_resize == 0 else 0
        m = float(max(total, floor_keys)) / self._MAX_LOAD
        if m >= float(n_bkt):
            want = max(_floor_to_size_t(float(math.floor(m)) + 1.0), (n_bkt * self._GROWTH) & _SIZE_MAX)
            return True, self._next_bkt(want)
        self.next_resize = _floor_to_size_t(float(n_bkt) * self._MAX_LOAD)
        return False, 0

    def _insert_node(self, k) -> None:
        """Link an absent key: the policy is asked for one more key; on a growth every key is re-linked first."""
        grow, n = self._need_rehash(self.bucket_count, len(self.order), 1)
        if grow:
            self._rehash(n)
        self._insert_bucket_begin(k)

    def _rehash(self, n: int) -> None:
        """n buckets: from an empty order, the keys taken in their old iteration order, each placed by the linking
        rule under the new bucket count."""
        self.bucket_count = n
        old, self.order = self.order, []
        for k in old:
            self._insert_bucket_begin(k)

    def _insert_bucket_begin(self, k) -> None:
        """The linking rule: k immediately before the first key of the order in k's bucket, else at the front."""
        bucket = self._bucket
        b = bucket(k)
        order = self.order
        for i, x in enumerate(order):
            if bucket(x) == b:
                order.insert(i, k)
                return
        order.insert(0, k)


class StdHashSet(set):
    """std::unordered_set<int64_t> (a ball's followers): a Python set of the keys whose `table`, a StdHashMap with its
    values ignored, keeps the library's order, bucket count and next resize. `ordered()` is the walk's order."""

    def __init__(self, items=()):
        super().__init__()
        self.table = StdHashMap()      # the module's name, looked up at creation (the tests replace it)
        for k in items:
            self.add(k)

    def add(self, k) -> None:
        set.add(self, k)
        self.table.insert(k, None)

    def discard(self, k) -> None:
        set.discard(self, k)
        self.table.erase(k)

    def remove(self, k) -> None:
        if k not in self:
            raise KeyError(k)
        self.discard(k)

    def clear(self) -> None:
        set.clear(self)
        self.table.clear()

    def ordered(self) -> list:
        """The keys in iteration order (a new list)."""
        return self.table.keys()

    @property
    def bucket_count(self) -> int:
        return self.table.bucket_count

    @property
    def next_resize(self) -> int:
        return self.table.next_resize


class BallLocation:
    """BallLocation (CollisionStructures.h:36): a point of a ball's piecewise path in the tick."""
    __slots__ = ("time", "collider", "p", "v")

    def __init__(self, time: float = -1.0, collider: int = -1, p=None, v=None):
        self.time, self.collider = time, collider
        self.p = [0.0, 0.0, 0.0] if p is None else p
        self.v = [0.0, 0.0, 0.0] if v is None else v


class CollisionImpact:
    """CollisionImpact, value-initialised as `new CollisionImpact()`: collider 0 and no velocity change until set."""
    __slots__ = ("time", "collider", "dv")

    def __init__(self, time: float):
        self.time, self.collider, self.dv = time, 0, [0.0, 0.0, 0.0]


class CollisionItem:
    """CollisionItem (CollisionStructures.cpp:6): no impact time (-1), no collider, miniball indices -2 (static)."""
    __slots__ = ("time", "normal", "collider", "mb1", "mb2", "impact")

    def __init__(self):
        self.time, self.normal, self.collider = -1.0, [0.0, 0.0, 0.0], None
        self.mb1, self.mb2, self.impact = -2, -2, None


class CollisionBallProperties:
    """CollisionBallProperties (CollisionBallProperties.cpp; COL-714, COL-715): a ball's path through the tick as
    points (time, collider, position, velocity after the impact), searched from m_tStart to m_tEnd."""

    def __init__(self, park):
        self.park = park
        self.ball = None
        self.locations: list = []
        self.capacity = LOCATIONS_RESERVED                  # the reserve survives the moves into the map
        self.max_time = 0.0
        self.t_start = 0.0
        self.t_end = 0.0

    def _integrate(self, p, v, t):
        b = self.ball
        a = [b.last_g[i] + b.last_c[i] for i in range(3)]
        return self.park.integrate(p, v, a, b.mass * b.agility, self.park.friction, b.time_factor, t)

    def _resize(self, n: int) -> None:
        """std::vector::resize: new points are BallLocation(); growing past the capacity doubles it."""
        if n > self.capacity:
            self.capacity = len(self.locations) + max(len(self.locations), n - len(self.locations))
        del self.locations[n:]
        while len(self.locations) < n:
            self.locations.append(BallLocation())

    def initialize(self, ball, t_end: float) -> None:
        self.ball, self.max_time, self.t_start, self.t_end = ball, t_end, 0.0, t_end
        self._resize(2)
        p, v = self._integrate(list(ball.pos), list(ball.vel), t_end)
        l0, l1 = self.locations
        l0.time, l0.collider, l0.p, l0.v = 0.0, -1, list(ball.pos), list(ball.vel)
        l1.time, l1.collider, l1.p, l1.v = t_end, -1, p, v

    def add_impact(self, impact: CollisionImpact) -> list:
        invalidated = []
        if self.t_end <= impact.time:
            return invalidated
        index = len(self.locations)
        while index > 0:
            if impact.time >= self.locations[index - 1].time:
                break
            if self.locations[index - 1].collider != -1:
                invalidated.append((self.locations[index - 1].time, self.locations[index - 1].collider))
            index -= 1
        self._resize(index + 1)
        loc = self.locations[index]
        loc.collider, loc.time = impact.collider, impact.time
        if index == 0:
            loc.p = list(self.ball.pos)
            loc.v = [self.ball.vel[i] + impact.dv[i] for i in range(3)]
        else:
            prev = self.locations[index - 1]
            p, v = self._integrate(list(prev.p), list(prev.v), loc.time - prev.time)
            loc.p, loc.v = p, [v[i] + impact.dv[i] for i in range(3)]
        self.t_end = impact.time
        return invalidated

    def remove_impact(self, time: float, collider: int) -> list:
        invalidated = []
        index = 0                                           # `found` starts true: a missing impact means index 0
        for i, loc in enumerate(self.locations):
            if loc.collider == collider and abs(loc.time - time) < FUZZY_DELTA:
                index = i
                break
        if index == 0:
            self.locations[0].collider = -1
            self.locations[0].v = list(self.ball.vel)
        for loc in self.locations[index + 1:]:
            if loc.collider != -1:
                invalidated.append((loc.time, loc.collider))
        self._resize(index if index > 0 else 1)
        self.calculate_end()
        return invalidated

    def calculate_end(self) -> None:
        if self.max_time > self.locations[-1].time:
            if len(self.locations) == self.capacity:        # push_back reallocates under a held reference
                raise DestinyUndefined("CalculateEnd reads a point of a reallocated vector "
                                       "(CollisionBallProperties.cpp:173)")
            prev = self.locations[-1]
            p, v = self._integrate(list(prev.p), list(prev.v), self.max_time - prev.time)
            self.locations.append(BallLocation(self.max_time, -1, p, v))
            self.t_end, self.t_start = self.max_time, prev.time

    def ignore_collision(self, collider: int, toi: float) -> bool:
        if len(self.locations) < 2 or toi > FUZZY_DELTA:
            return False
        i = len(self.locations) - 2
        lower = self.locations[i].time - FUZZY_DELTA
        while i >= 0:
            if self.locations[i].time < lower:
                break
            if self.locations[i].collider == collider:
                return True
            i -= 1
        return False

    def find_index(self, time: float) -> int:
        index = 0
        while index < len(self.locations) and self.locations[index].time + FUZZY_DELTA < time:
            index += 1
        return index - 1 if index != 0 else 0

    def position(self, time: float) -> tuple:
        """CalculatePosition: the point at or before the time, integrated on to it."""
        loc = self.locations[self.find_index(time)]
        p, v = list(loc.p), list(loc.v)
        if abs(loc.time - time) >= FUZZY_DELTA:
            p, v = self._integrate(p, v, time - loc.time)
        return p, v


class Box:
    """A partition box (Box.h): its level, clamped key, bounds, parent, children and balls (sorted by key and id when
    walked, as Destiny's std::sets), the top box's bubble and the search flag; and whether it has been released, and
    then whether BoxAllocator kept its memory or deleted it. `statics` holds its static collidables, which only
    exact_ballpark.full adds (keyed by StaticCollidableSortComparer's parent id and id)."""
    __slots__ = ("level", "key", "lo", "hi", "parent", "children", "balls", "bubble", "handled", "released", "deleted",
                 "statics")

    def __init__(self, level: int, key: tuple, lo: list, hi: list):
        self.level, self.key, self.lo, self.hi = level, key, lo, hi
        self.parent: Box | None = None
        self.children: dict = {}
        self.balls: dict = {}
        self.bubble = -1
        self.handled = False
        self.released = False
        self.deleted = False
        self.statics: dict = {}

    def sorted_children(self) -> list:
        return [self.children[k] for k in sorted(self.children)]

    def sorted_balls(self) -> list:
        """BallSortComparer's order (Ball.cpp:2528): two local (negative) ids by their owner first, otherwise by id;
        for this slice's positive ids, ascending id."""
        return sorted(self.balls.values(), key=lambda b: (0, b.owner_id, b.id) if b.id < 0 else (1, 0, b.id))

    def sorted_statics(self) -> list:
        return [self.statics[k] for k in sorted(self.statics)]


@dataclass
class SpaceBall(ModeBall):
    """modes.py's ball with Partitionable's and Ball.h's remaining members, at Ball::Ball's defaults."""
    old_bubble: int = -1                                    # Partitionable::mOldBubble
    is_global: bool = False
    is_space_junk: bool = False
    harmonic: int = -1
    alliance_id: int = -1
    corporation_id: int = -1
    tracking_ptr: object = None                             # mTrackingPtr: the ball itself (it can outlive removal)
    tracking_range: float = 0.0
    within_tracking_range: bool = False
    notification_range: float = 0.0
    within_range: bool = False
    has_proximity: bool = False
    in_dead_bubble: bool = False
    handled: bool = False                                   # Ball::mHandled
    old_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    old_vel: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    collisions: list = field(default_factory=list)          # mCollisions, in order
    boxes: dict = field(default_factory=dict)               # mBoxes, key -> Box (all at one level)
    my_box: Box | None = None
    mod: list = field(default_factory=lambda: [-2, -2, -2])
    active: list = field(default_factory=lambda: [False] * 27)
    sensor_range: float = 0.0                               # ProximitySensor (Ball.h:31): range is a float
    sensor_period: float = 2.0
    sensor_elapsed: float = 0.0
    sensor_only_interactives: bool = False
    sensor_balls: StdHashMap = field(default_factory=StdHashMap)
    followers: StdHashSet = field(default_factory=StdHashSet)

    def sorted_boxes(self) -> list:
        return [self.boxes[k] for k in sorted(self.boxes)]


class SpaceBallpark(ModeBallpark):
    """modes.py's ballpark with Destiny's partition, bubbles, sensors, visibility, range queries and collisions; a
    client's view unless told otherwise, as Destiny's Ballpark() (isMaster false)."""

    def __init__(self, friction: float = 1.0e6, tick_interval_ms: int = 1000, use_new_orbit: bool = False,
                 is_master: bool = False, seed: int = 1, iterative: bool = False):
        super().__init__(friction, tick_interval_ms, use_new_orbit, is_master)
        self.iterative_collision = iterative                # g_useIterativeCollision (COL-701)
        self.collision_locations: dict = {}                 # mCollisionLocations: kept from tick to tick
        self.potential_balls: dict = {}                     # mPotentialCollisionBalls
        self.levels: list[dict] = [{} for _ in range(NUM_LEVELS)]    # Partition::mLevels
        self.free_boxes = 0                                 # BoxAllocator's freeBoxes.size()
        self.free_balls: set = set()                        # mFreeBalls (walked in ascending id)
        self.proximity_balls = StdHashMap()                 # mProximityBalls
        self.globals: dict = {}                             # mGlobals
        self.bubbles: dict = {}                             # bubbles: bubble id -> {ball id: +1, 0 or -1}
        self.bubble_interactives: dict = {}                 # bubble id -> set of ball ids
        self.bubble_keep_alives: dict = {}
        self.bubble_keep_alive_balls: set = set()
        self.bubble_id = 0                                  # mBubbleId
        self.local_cnt, self.local_hi_cnt = -1, LOCAL_BALLS
        self.in_evolve = False
        self.moribund: set = set()                          # moribundBalls, by id
        self.rng = GlibcRandom(seed)                        # srand(seed): floatrand's rand()

    # ======================================================================================================== partition

    def _get_box(self, ix: int, iy: int, iz: int, level: int) -> Box | None:
        """Partition::GetBox(ix, iy, iz, level): the box at the clamped key, created with its parents if missing."""
        if level >= NUM_LEVELS or level < 0:
            return None
        key = _clamp_key(ix, iy, iz, LEVEL_GRID[level])
        box = self.levels[level].get(key)
        if box is None:                                     # BoxAllocator::GetNewBox, then Box::Box
            self._hit("box.create")
            if self.free_boxes > 0:
                self.free_boxes -= 1
            w, n = LEVEL_WIDTH[level], LEVEL_GRID[level]
            center = w * n * 0.5
            box = Box(level, key, [key[i] * w - center for i in range(3)],
                      [(key[i] + 1) * w - center for i in range(3)])
            if level != 0:                                  # Box::SetParent
                box.parent = self._get_box(key[0] >> 2, key[1] >> 2, key[2] >> 2, level - 1)
                box.parent.children[key] = box
            self.levels[level][key] = box
        return box

    def _check_box(self, ix: int, iy: int, iz: int, level: int) -> Box | None:
        """Partition::CheckBox: the box at the clamped key if it exists."""
        if level >= NUM_LEVELS or level < 0:
            return None
        return self.levels[level].get(_clamp_key(ix, iy, iz, LEVEL_GRID[level]))

    @staticmethod
    def level_for(radius: float) -> int:
        """GetBox(radius, pos)'s level (SEN-706): the finest whose width is at least 3x the radius."""
        level = 0
        for i in range(NUM_LEVELS):
            if 3.0 * radius > LEVEL_WIDTH[i]:
                break
            level = i
        return level

    @staticmethod
    def key_for(level: int, pos) -> tuple:
        """GetBox(radius, pos)'s indices at a level, before the clamp."""
        w = LEVEL_WIDTH[level]
        fac, half = 1.0 / w, 0.5 * w * LEVEL_GRID[level]
        return tuple(_to_int64((pos[i] + half) * fac) for i in range(3))

    def _get_box_r(self, radius: float, pos, create: bool = True) -> Box | None:
        """Partition::GetBox(radius, pos, create)."""
        level = self.level_for(radius)
        key = self.key_for(level, pos)
        return self._get_box(*key, level) if create else self._check_box(*key, level)

    def _get_existing_top_box(self, pos) -> Box | None:
        return self._get_box_r(10000000.0, pos, False)

    @staticmethod
    def _topmost(box: Box | None) -> Box | None:
        if box is None:
            return None
        while box.parent is not None:
            box = box.parent
        return box

    def _release(self, box: Box) -> None:
        """Box::ReleaseBox through BoxAllocator::ReleaseBox: the memory is kept if fewer than 256 boxes are kept (the
        test precedes the destructor), otherwise deleted; the destructor leaves the parent (which may release it in
        turn), then the map."""
        self._hit("box.release")
        keep = self.free_boxes < BOX_CHUNK_SIZE
        if box.parent is not None:                          # Box::RemoveChildren
            p = box.parent
            p.children.pop(box.key, None)
            if not p.children and not p.balls:
                self._release(p)
        del self.levels[box.level][box.key]
        box.released = True
        if keep:
            self.free_boxes += 1
        else:
            self._hit("box.deleted")
            box.deleted = True

    @staticmethod
    def _stale(box: Box, what: str) -> Box:
        """A box Destiny may still use after releasing it: defined while its memory is kept (the stale values are
        intact), undefined once deleted (glibc then owns its first bytes)."""
        if box.deleted:
            raise DestinyUndefined(f"{what} of a box released and deleted (Ballpark.cpp:3141-3185, Ball.cpp:2365)")
        return box

    def _box_remove_ball(self, box: Box, b: SpaceBall) -> None:
        """Box::RemoveBall."""
        box.balls.pop(b.id, None)
        if not box.children and not box.balls:
            self._release(box)

    @staticmethod
    def _clear_boxes(b: SpaceBall) -> None:
        """Ball::ClearBoxes."""
        b.boxes, b.my_box = {}, None
        b.active = [False] * 27
        b.mod = [-2, -2, -2]

    def _delete_from_boxes(self, b: SpaceBall) -> None:
        """Ball::DeleteFromBoxes: leave every box (in key order), then forget them."""
        for box in b.sorted_boxes():
            self._box_remove_ball(box, b)
        self._clear_boxes(b)

    @staticmethod
    def _insert_in_box(b: SpaceBall, box: Box | None) -> None:
        """Ball::InsertInBox."""
        if box is None:
            return
        box.balls[b.id] = b
        b.boxes[box.key] = box

    def inflated_radius(self, b: SpaceBall) -> float:
        """Ball::GetInflatedRadius (Ball.cpp:838; SEN-706): a free ball's radius plus a tick at full thrust."""
        if not b.is_free:
            return b.radius
        m = b.mass * b.agility
        a = _ieee_div(self.friction * b.speed_fraction * b.max_vel, m)
        v = math.sqrt(_lensq(b.vel))
        k = self.friction
        max_poss_speed = _ieee_div(m * a - (m * a - v * k) * b.time_factor, k)
        return b.radius + self.dt * max_poss_speed

    def _insert_ball_in_boxes(self, b: SpaceBall) -> None:
        """Ballpark::InsertBallInBoxes (Ballpark.cpp:3189): file the ball, then register it unless inside Evolve."""
        self._insert_in_boxes(b)
        if not self.in_evolve:
            self.balls[b.id] = b                            # mBalls.Add keeps an existing key's place
            if b.is_global:
                self.globals[b.id] = b

    def _insert_in_boxes(self, b: SpaceBall) -> None:
        """Ballpark::InsertInBoxes (Ballpark.cpp:3115; SEN-706, SEN-707)."""
        top, new_bubble = None, -1
        inflated = self.inflated_radius(b)
        box1 = self._get_box_r(inflated, b.pos)
        if self.is_master:
            top = self._topmost(box1)
            new_bubble = top.bubble
            if new_bubble == -1:
                self._hit("bubble.no_bubble")
                self._delete_from_boxes(b)                  # may release box1 and top: they are still used
                b.my_box = None
                new_bubble = self._nearby_bubbles(self._stale(top, "NearbyBubbles"))
                if new_bubble == -1:
                    self._hit("bubble.new")
                    new_bubble = self.bubble_id
                    self.bubble_id += 1
                else:
                    self._hit("bubble.neighbour")
                self._stale(top, "the bubble written into the top box").bubble = new_bubble
            if b.bubble != new_bubble:
                self._hit("bubble.changed")
                b.old_bubble = b.bubble
                b.bubble = new_bubble
                self._add_to_bubble(b)
        self._ball_insert_in_boxes(b, box1, top, new_bubble)

    def _ball_insert_in_boxes(self, b: SpaceBall, box1: Box, top: Box | None, new_bubble: int) -> None:
        """Ball::InsertInBoxes (Ball.cpp:2354; SEN-706). box1 may be a box released just before (the ball was its only
        occupant): its stale bounds are read."""
        inflated = self.inflated_radius(b)
        if box1.released:
            self._hit("insert.stale_box")
            self._stale(box1, "the bounds")
        mod = [0, 0, 0]
        for q in range(3):
            if box1.lo[q] + inflated - b.pos[q] > 0.0:
                mod[q] = -1
            elif box1.hi[q] - inflated - b.pos[q] < 0.0:
                mod[q] = 1
        if b.my_box is not None and box1.level == b.my_box.level and box1.key == b.my_box.key:
            if mod == b.mod:
                self._hit("insert.unchanged")
                return
        self._delete_from_boxes(b)
        b.mod = list(mod)
        box1 = self._get_box_r(inflated, b.pos)
        ix, iy, iz = box1.key
        level = box1.level
        self._insert_in_box(b, box1)
        if self.is_master:
            top = self._topmost(box1)
            top.bubble = new_bubble
        b.active[13] = True
        b.my_box = box1
        if level == 0 and b.radius * 2 > LEVEL_WIDTH[0]:    # mRadius * 2 (a float) against the top width
            self._hit("insert.huge")
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    for z in (-1, 0, 1):
                        if x == 0 and y == 0 and z == 0:
                            continue
                        self._insert_in_box(b, self._get_box(ix + x, iy + y, iz + z, 0))
                        b.active[13 + x + 3 * y + 9 * z] = True
            return
        if mod[0] or mod[1] or mod[2]:
            self._hit("insert.adjacent")
        if mod[0]:
            self._insert_in_box(b, self._get_box(ix + mod[0], iy, iz, level))
            b.active[13 + mod[0]] = True
        if mod[1]:
            self._insert_in_box(b, self._get_box(ix, iy + mod[1], iz, level))
            b.active[13 + 3 * mod[1]] = True
        if mod[2]:
            self._insert_in_box(b, self._get_box(ix, iy, iz + mod[2], level))
            b.active[13 + 9 * mod[2]] = True
        if mod[0] and mod[1]:
            self._insert_in_box(b, self._get_box(ix + mod[0], iy + mod[1], iz, level))
            b.active[13 + mod[0] + 3 * mod[1]] = True
        if mod[0] and mod[2]:
            self._insert_in_box(b, self._get_box(ix + mod[0], iy, iz + mod[2], level))
            b.active[13 + mod[0] + 9 * mod[2]] = True
        if mod[1] and mod[2]:
            self._insert_in_box(b, self._get_box(ix, iy + mod[1], iz + mod[2], level))
            b.active[13 + 3 * mod[1] + 9 * mod[2]] = True
        if mod[0] and mod[1] and mod[2]:
            self._hit("insert.corner")
            self._insert_in_box(b, self._get_box(ix + mod[0], iy + mod[1], iz + mod[2], level))
            b.active[13 + mod[0] + 3 * mod[1] + 9 * mod[2]] = True

    def _nearby_balls(self, b: SpaceBall, exclude_fixed: bool, exclude_inert: bool, exclude_missiles: bool,
                      exclude_cloaked: bool, exclude_harmonics: bool) -> list:
        """Partition::GetNearbyBalls (Partition.cpp:344): the ball's boxes, their non-empty ancestors and descendants,
        then the balls in them that pass the filter, each once, in box order and then id order."""
        if exclude_inert and self.interactive_cnt(b) == 0:
            self._hit("nearby.no_interactives")
            return []
        s_boxes: list = []
        for box in b.sorted_boxes():
            if box.handled:
                continue
            p = box
            while p.parent is not None:
                p = p.parent
                if p.handled:
                    break
                if p.balls:
                    s_boxes.append(p)
                    p.handled = True
            self._explode_children(box, s_boxes)
            if len(box.balls) > 1:
                s_boxes.append(box)
                box.handled = True
        uni: list = []
        for box in s_boxes:
            box.handled = False
            for other in box.sorted_balls():
                if other.handled:
                    continue
                if other is b:
                    continue
                if other.is_moribund:
                    self._hit("nearby.moribund")
                    continue
                if exclude_cloaked and (other.is_cloaked or not other.is_massive):
                    continue
                if exclude_inert:
                    if not other.is_interactive:
                        continue
                else:
                    if self.is_master and other.bubble != b.bubble and not other.is_global:
                        self._hit("nearby.other_bubble")
                        continue
                    if self.is_master and other.bubble != b.bubble:
                        self._hit("nearby.global_other_bubble")
                    if exclude_fixed and not other.is_free:
                        continue
                    if other.mode == FIELD and not exclude_harmonics:
                        if b.harmonic == -2:
                            self._hit("nearby.field_master_pass")
                            continue
                        if other.harmonic != -1 and other.harmonic == b.harmonic:
                            self._hit("nearby.field_harmonic")
                            continue
                        if other.corporation_id != -1 and other.corporation_id == b.corporation_id:
                            self._hit("nearby.field_corporation")
                            continue
                        if other.alliance_id != -1 and other.alliance_id == b.alliance_id:
                            self._hit("nearby.field_alliance")
                            continue
                    if exclude_missiles and other.mode == MISSILE:
                        continue
                other.handled = True
                uni.append(other)
        for other in uni:
            other.handled = False
        return uni

    def _explode_children(self, box: Box, out: list) -> None:
        """Partition::ExplodeChildren: every descendant with balls, children before their parent."""
        for child in box.sorted_children():
            self._explode_children(child, out)
            if child.balls:
                out.append(child)

    # ========================================================================================================== bubbles

    def _nearby_bubbles(self, box: Box) -> int:
        """Box::NearbyBubbles: the first of the 26 neighbours (in i, j, k order) that has a bubble."""
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    other = self._check_box(box.key[0] + i, box.key[1] + j, box.key[2] + k, box.level)
                    if other is not None and other.bubble != -1:
                        return other.bubble
        return -1

    def _traverse(self, box: Box, seed: int) -> None:
        """Box::Traverse: seed-fill the connected top boxes."""
        if box.bubble != -1:
            return
        box.bubble = seed
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    other = self._check_box(box.key[0] + i, box.key[1] + j, box.key[2] + k, box.level)
                    if other is not None:
                        self._traverse(other, seed)

    def _add_to_bubble(self, b: SpaceBall) -> None:
        """Ballpark::AddToBubble (Ballpark.cpp:6429): +1 when added, 0 when kept, -1 when removed (a just-added ball
        is dropped instead); then the interactive and keep-alive sets."""
        if b.bubble != -1:
            nb = self.bubbles.get(b.bubble)
            if nb is None:
                nb = self.bubbles[b.bubble] = {}
            value = nb.get(b.id)
            if value is None:
                nb[b.id] = 1
            elif value == -1:
                nb[b.id] = 0
        if b.old_bubble != -1:
            ob = self.bubbles.get(b.old_bubble)
            if ob is not None:
                value = ob.get(b.id)
                if value is not None:
                    if value == 0:
                        ob[b.id] = -1
                    elif value == 1:
                        del ob[b.id]
        if b.is_interactive:                                # UpdateInteractiveCnt
            self._increase_interactive(b, b.bubble)
            self._decrease_interactive(b, b.old_bubble)
        if b.id in self.bubble_keep_alive_balls:            # UpdateKeepAliveBalls
            if b.bubble != -1:
                self.bubble_keep_alives.setdefault(b.bubble, set()).add(b.id)
            if b.old_bubble != -1:
                s = self.bubble_keep_alives.get(b.old_bubble)
                if s is not None:
                    s.discard(b.id)
                    if not s:
                        del self.bubble_keep_alives[b.old_bubble]

    def _increase_interactive(self, b: SpaceBall, bubble: int) -> None:
        if b.is_interactive and bubble != -1:
            self.bubble_interactives.setdefault(bubble, set()).add(b.id)

    def _decrease_interactive(self, b: SpaceBall, bubble: int) -> None:
        if b.is_interactive and bubble != -1:
            s = self.bubble_interactives.get(bubble)
            if s is not None:
                s.discard(b.id)
                if not s:
                    del self.bubble_interactives[bubble]

    def interactive_cnt(self, b: SpaceBall) -> int:
        """Ballpark::GetInteractiveCnt: the server's count of interactive balls in the ball's bubble."""
        if not self.is_master:
            return 0
        return len(self.bubble_interactives.get(b.bubble, ()))

    def in_dead_bubble(self, b: SpaceBall) -> bool:
        """Ballpark::InDeadBubble (Ballpark.cpp:6216; TIME-714)."""
        if not self.is_master:
            return False
        if b.mode == WARP:
            return False
        if len(self.bubble_interactives.get(b.bubble, ())) != 0:
            return False
        if len(self.bubble_keep_alives.get(b.bubble, ())) != 0:
            return False
        return True

    def initialize_bubbles(self, src_id: int = 0) -> None:
        """PyInitializeBubbles (Thunkers.cpp:1123): InitializeBubbles, or UpdateBallBubble for one ball."""
        if src_id == 0:
            self._initialize_bubbles()
        else:
            self._update_ball_bubble(src_id)

    def _initialize_bubbles(self) -> None:
        """Ballpark::InitializeBubbles (Ballpark.cpp:6565; SEN-707)."""
        for box in self.levels[0].values():                 # ClearTraversalFlag
            box.bubble = -1
        self.bubble_id = 0
        self.bubbles.clear()                                # ClearBubbles
        self.bubble_interactives.clear()
        self.bubble_keep_alives.clear()
        self.bubble_keep_alive_balls.clear()
        for b in list(self.balls.values()):
            if b.my_box is None:
                b.old_bubble, b.bubble = b.bubble, -1
                continue
            top = self._topmost(b.my_box)
            if top.bubble != -1:
                self._hit("init.traversed")
                b.old_bubble, b.bubble = b.bubble, top.bubble
                self._add_to_bubble(b)
                continue
            self._traverse(top, self.bubble_id)
            b.old_bubble, b.bubble = b.bubble, self.bubble_id
            self._add_to_bubble(b)
            self.bubble_id += 1

    def _update_ball_bubble(self, src_id: int) -> None:
        """Ballpark::UpdateBallBubble (Ballpark.cpp:5959)."""
        ball = self.balls.get(src_id)
        if ball is None:
            return
        max_width = LEVEL_WIDTH[0] * LEVEL_WIDTH[0]
        hi_id = -1
        for b in self.balls.values():
            if b.bubble > hi_id:
                hi_id = b.bubble
            if b is ball or b.bubble == -1:
                continue
            if _lensq(_sub(ball.pos, b.pos)) < max_width:
                ball.bubble = b.bubble
                break
        if ball.bubble == -1:
            ball.bubble = hi_id + 1

    def copy_bubbles(self) -> None:
        """Ballpark::CopyBubbles (Ballpark.cpp:6154): additions become kept, removals are deleted, and empty bubbles
        go."""
        dead = []
        for bubble_id, bubble in self.bubbles.items():
            removals = []
            for ball_id, action in bubble.items():
                if action == 1:
                    bubble[ball_id] = 0
                elif action == -1:
                    removals.append(ball_id)
            for ball_id in removals:
                del bubble[ball_id]
            if not bubble:
                dead.append(bubble_id)
        for bubble_id in dead:
            del self.bubbles[bubble_id]

    def add_keep_alive_ball(self, bid: int) -> None:
        """The Python layer's park.bubbleKeepAliveBalls.add(id)."""
        self.bubble_keep_alive_balls.add(bid)

    def bubble_at(self, x: float, y: float, z: float):
        """PyGetBubbleAtCoordinates: the top box's bubble, or None where no top box exists."""
        top = self._get_existing_top_box([float(x), float(y), float(z)])
        return None if top is None else top.bubble

    def remote_followers(self) -> list:
        """PyGetRemoteFollowers (Thunkers.cpp:1193; MOV-731): free balls following a ball in another bubble."""
        out = []
        for bid in sorted(self.free_balls):
            src = self.balls[bid]
            if src.mode in FOLLOW_MODES:
                dst = self.balls.get(src.follow_id)
                if dst is not None and src.bubble == dst.bubble:
                    continue
                out.append(src.id)
        return out

    def clean_up_followers(self) -> list:
        """Actions.clean_up_followers (_actions.py:677): a Stop for each remote follower, at the pre-tick."""
        ids = self.remote_followers()
        for bid in ids:
            self.stop(bid)
        return ids

    # ======================================================================================================= ball setup

    def _id_for_collision_object(self, object_id: int) -> int:
        """Ballpark::GetIDForCollisionObject."""
        if self.is_master:
            if object_id <= 0:
                if object_id < 0:
                    self.local_hi_cnt -= 1
                    return self.local_hi_cnt
                self.local_cnt -= 1
                return self.local_cnt
        elif object_id == -1:
            self.local_hi_cnt -= 1
            return self.local_hi_cnt
        return object_id

    def add_ball(self, object_id: int, mass: float, radius: float, max_vel: float, is_free: bool = False,
                 is_global: bool = False, is_massive: bool = False, is_interactive: bool = False,
                 is_space_junk: bool = False, x: float = 0.0, y: float = 0.0, z: float = 0.0, vx: float = 0.0,
                 vy: float = 0.0, vz: float = 0.0, agility: float = 0.0, speed_fraction: float = 0.0) -> SpaceBall:
        """Ballpark::AddBall (Ballpark.cpp:3303), with the Python thunker's float parameters."""
        radius, max_vel = _float32(radius), _float32(max_vel)
        agility, speed_fraction = _float32(agility), _float32(speed_fraction)
        the_id = self._id_for_collision_object(int(object_id))
        if the_id <= 0:
            raise NotImplementedError("local (negative) ball ids sort by owner: not in this slice")
        ball = self.balls.get(the_id)
        created = ball is None
        if created:
            ball = SpaceBall(id=the_id, mass=0.0, radius=0.0, max_vel=0.0, agility=1.0, speed_fraction=1.0,
                             time_factor=1.0, pos=[1e10, 1e10, 1e10])
        else:
            self._hit("add.existing")
            self._delete_from_boxes(ball)
        ball.pos = [float(x), float(y), float(z)]
        ball.vel = [float(vx), float(vy), float(vz)]
        if created:
            ball.old_pos = [ball.pos[i] - ball.vel[i] * self.dt for i in range(3)]
            ball.old_vel = list(ball.vel)
        else:
            ball.effect_stamp = 0
            ball.owner_id = 0
            ball.follow_id = 0
            ball.follow_range = 10.0
            ball.corporation_id = -1
            ball.alliance_id = -1
            ball.harmonic = -1
            if ball.is_moribund:
                ball.is_moribund = False
                self.moribund.discard(the_id)
        ball.radius = 0.0 if radius < 0.0 else radius
        mass = float(mass)
        ball.mass = 0.0 if mass < 0.0 else mass
        ball.is_free, ball.is_global, ball.is_massive = bool(is_free), bool(is_global), bool(is_massive)
        ball.is_interactive, ball.is_space_junk = bool(is_interactive), bool(is_space_junk)
        ball.max_vel = 0.0 if max_vel < 0.0 else max_vel
        ball.agility = 1.0 if agility <= 0.0 else agility
        ball.speed_fraction = 0.0 if speed_fraction < 0.0 else speed_fraction
        self._refresh_time_factor(ball)
        if not created and not is_free:
            self.free_balls.discard(the_id)
        elif ball.is_free:
            self.free_balls.add(the_id)
        self._set_mode(ball, STOP)
        if not created and ball.followers:
            for fid in ball.followers.ordered():
                f = self.balls.get(fid)
                if f is None or f.follow_ptr != ball.id:
                    ball.followers.discard(fid)
        self._insert_ball_in_boxes(ball)
        return ball

    def set_mass(self, bid: int, mass: float) -> None:
        """SetBallMass: ignored unless positive; refreshes the time factor."""
        mass = float(mass)
        if mass <= 0.0:
            return
        b = self.balls.get(bid)
        if b is None:
            return
        b.mass = mass
        self._refresh_time_factor(b)

    def set_radius(self, bid: int, r: float) -> None:
        """SetBallRadius(id): then re-filed."""
        b = self.balls.get(bid)
        if b is not None:
            self._set_ball_radius(b, _float32(r))

    def _set_ball_radius(self, b: SpaceBall, r: float) -> None:
        """SetBallRadius(Ball*, float)."""
        if r < 0.0:
            return
        b.radius = r
        self._insert_ball_in_boxes(b)

    def set_max_velocity(self, bid: int, v: float) -> None:
        """SetMaxSpeed: ignored if negative; then re-filed (no time-factor refresh)."""
        v = _float32(v)
        if v < 0.0:
            return
        b = self.balls.get(bid)
        if b is None:
            return
        b.max_vel = v
        self._insert_ball_in_boxes(b)

    def set_agility(self, bid: int, agility: float) -> None:
        """SetBallAgility: ignored unless positive; refreshes the time factor."""
        agility = _float32(agility)
        b = self.balls.get(bid)
        if b is None or agility <= 0.0:
            return
        b.agility = agility
        self._refresh_time_factor(b)

    def set_speed_fraction(self, bid: int, s: float) -> None:
        """SetSpeedFraction: finite only, clamped to [0, 1]."""
        s = _float32(s)
        if not math.isfinite(s):
            return
        b = self.balls.get(bid)
        if b is None:
            return
        b.speed_fraction = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s

    def set_position(self, bid: int, x: float, y: float, z: float) -> None:
        """SetBallPosition: the new and the old position, then re-filed."""
        b = self.balls.get(bid)
        if b is None:
            return
        b.pos = [float(x), float(y), float(z)]
        b.old_pos = list(b.pos)
        self._insert_ball_in_boxes(b)

    def set_velocity(self, bid: int, x: float, y: float, z: float) -> None:
        """SetBallVelocity: the new and the old velocity (and the orientation, not kept here)."""
        b = self.balls.get(bid)
        if b is None:
            return
        b.vel = [float(x), float(y), float(z)]
        b.old_vel = list(b.vel)

    def set_free(self, bid: int, flag: bool) -> None:
        """SetBallFree (MOV-732): fixing a ball stops it, zeroes its velocity, re-files it and clears mLastG."""
        b = self.balls.get(bid)
        if b is None or bool(flag) == b.is_free:
            return
        b.is_free = bool(flag)
        if b.is_free:
            self.free_balls.add(bid)
        else:
            self._stop(b)
            self.set_velocity(bid, 0.0, 0.0, 0.0)
            self.free_balls.discard(bid)
            self._insert_ball_in_boxes(b)
            b.last_g = [0.0, 0.0, 0.0]

    def set_massive(self, bid: int, flag: bool) -> None:
        b = self.balls.get(bid)
        if b is not None:
            b.is_massive = bool(flag)

    def set_global(self, bid: int, flag: bool) -> None:
        b = self.balls.get(bid)
        if b is None:
            return
        if flag:
            self.globals[bid] = b
        elif b.is_global:
            self.globals.pop(bid, None)
        b.is_global = bool(flag)

    def set_interactive(self, bid: int, flag: bool) -> None:
        """SetBallInteractive: the bubble's interactive set follows."""
        b = self.balls.get(bid)
        if b is None:
            return
        if flag and not b.is_interactive:
            b.is_interactive = True
            self._increase_interactive(b, b.bubble)
        elif not flag and b.is_interactive:
            self._decrease_interactive(b, b.bubble)
            b.is_interactive = False

    def set_cloaked(self, bid: int, value: int) -> None:
        """The Python attribute ball.isCloaked = value (no setter runs; CloakBall, SEN-711, is a later package)."""
        b = self.balls.get(bid)
        if b is not None:
            b.is_cloaked = int(value) & 0xFF

    def set_harmonic(self, bid: int, harmonic: int, corporation: int, alliance: int, is_field: bool) -> None:
        """SetBallHarmonic: a force field is stopped and set to FIELD; clearing the field stops it."""
        b = self.balls.get(bid)
        if b is None:
            return
        b.harmonic, b.corporation_id, b.alliance_id = int(harmonic), int(corporation), int(alliance)
        if is_field:
            self._stop(b)
            self._set_mode(b, FIELD)
        elif b.mode == FIELD:
            self._stop(b)

    # ========================================================================================================= removal

    def remove_ball(self, bid: int, delay: int = 0) -> None:
        """Ballpark::RemoveBall (Ballpark.cpp:5471; TIME-713, MOV-730): stop it and release its followers; inside Evolve
        or with a delay it lingers moribund and non-massive until BringOutDeadBalls, otherwise it leaves the park, its
        boxes and its bubble at once."""
        b = self.balls.get(bid)
        if b is None:
            return
        self._stop(b)
        self._stop_all_followers(b)
        b.is_moribund = True
        if self.in_evolve or delay > 0:
            self._hit("remove.delayed")
            b.is_massive = False
            b.effect_stamp = self.current_time + delay
            self.moribund.add(bid)
            return
        self._hit("remove.now")
        self._event("DoFinalCleanup", b.id)
        self.moribund.discard(bid)
        if self.iterative_collision:
            self.collision_locations.pop(bid, None)
            self.potential_balls.pop(bid, None)
        self._delete_from_boxes(b)
        b.old_bubble, b.bubble = b.bubble, -1
        self._add_to_bubble(b)
        if b.is_free:
            self.free_balls.discard(bid)
        if b.has_proximity:
            self.proximity_balls.erase(bid)
        if b.is_global:
            self.globals.pop(bid, None)
        del self.balls[bid]

    def bring_out_the_deads(self) -> None:
        """BringOutTheDeads (Ballpark.cpp:5699; TIME-713), as OnTick calls it before evolving: every moribund ball whose
        stamp has passed, and up to 7 within 2 ticks of it. Destiny walks an unordered_set of pointers, whose order is
        the heap's: ascending id here. The order decides the order of the removals (and their events) when more than
        one ball goes in a call, and which balls go when more than 7 are within the buffer."""
        if self.in_evolve:
            return
        kill, due = 0, []
        for bid in sorted(self.moribund):
            b = self.balls[bid]
            if not b.is_moribund:
                due.append(b)
                continue
            if b.effect_stamp - self.current_time < 0:
                due.append(b)
            elif b.effect_stamp - self.current_time - 2 < 0 and kill < 7:
                self._hit("remove.buffer")
                due.append(b)
                kill += 1
        for b in due:
            if b.is_moribund:
                self.remove_ball(b.id)
            self.moribund.discard(b.id)

    # ======================================================================================================== sensors

    def _update_proximity_status(self, b: SpaceBall) -> None:
        """Ball::UpdateProximityStatus."""
        if b.tracking_range != 0.0 or b.notification_range != 0.0 or b.sensor_active:
            b.has_proximity = True
            self.proximity_balls.insert(b.id, b)
        else:
            b.has_proximity = False
            self.proximity_balls.erase(b.id)

    def add_proximity_sensor(self, bid: int, range_: float, period: float = 2.0, shuffle: int = 0,
                             only_interactives: bool = False) -> None:
        """AddProximitySensor (Ball.cpp:244; SEN-701)."""
        b = self.balls.get(bid)
        if b is None:
            return
        range_ = _float32(range_)
        if f32(b.radius + range_) < 0.0:                    # float + float
            return
        b.sensor_balls.clear()
        b.sensor_range = range_
        b.sensor_period = float(period)
        b.sensor_active = True
        b.sensor_only_interactives = bool(only_interactives)
        if shuffle:
            self._hit("sensor.shuffle")
            b.sensor_elapsed = 0.0 + self.rng.next() / 2147483647.0 * (b.sensor_period - 0.0)
        self._update_proximity_status(b)

    def remove_proximity_sensor(self, bid: int) -> None:
        b = self.balls.get(bid)
        if b is not None:
            b.sensor_active = False
            self._update_proximity_status(b)

    def set_notification_range(self, bid: int, n: float) -> None:
        """SetNotificationRange (SEN-704): ignored if negative; resets the in-range flag."""
        b = self.balls.get(bid)
        if b is None:
            return
        n = float(n)
        if n < 0.0:
            return
        b.notification_range = n
        self._update_proximity_status(b)
        b.within_range = False

    def set_target_tracking(self, bid: int, target: int, range_: float) -> None:
        """SetTargetTracking (SEN-703): range <= 0 or target 0 turns it off (the proximity status is not updated)."""
        b = self.balls.get(bid)
        if b is None:
            return
        range_ = float(range_)
        if range_ <= 0.0 or target == 0:
            b.tracking_ptr, b.tracking_range = None, 0.0
            return
        if target not in self.balls:
            return
        b.tracking_ptr, b.tracking_range = self.balls[target], range_
        self._update_proximity_status(b)
        b.within_tracking_range = False

    def _handle_proximities(self) -> None:
        """Ballpark::HandleProximities (Ballpark.cpp:5822): every proximity ball, in the map's order; a ball in a dead
        bubble gets one forced sensor check if its sensor holds balls, and its list is emptied (TIME-714)."""
        for bid in self.proximity_balls.keys():             # a copy: the walk survives changes to the map
            b = self.balls[bid]
            if b.is_moribund:
                continue
            if self.in_dead_bubble(b):
                if len(b.sensor_balls) > 0:
                    self._hit("sensor.dead_bubble_check")
                    b.sensor_elapsed = b.sensor_period
                    self._check_for_proximity(b)
                b.sensor_balls.clear()
                continue
            if not b.has_proximity:
                continue
            self._check_for_proximity(b)

    def _check_for_proximity(self, b: SpaceBall) -> None:
        """Ball::CheckForProximity."""
        if b.is_moribund:
            return
        self._check_notification_range(b)
        self._check_target_tracking(b)
        if not b.sensor_active:
            return
        self._check_sensor(b)

    def _check_notification_range(self, b: SpaceBall) -> None:
        """CheckForProximity_NotificationRange (Ball.cpp:520; SEN-704)."""
        send = False
        inside = False
        if b.notification_range != 0.0:
            n = b.notification_range
            if b.mode == GOTO:
                lim = n + b.radius
                inside = _lensq(_sub(b.pos, b.goto)) < lim * lim
            elif b.mode in (FOLLOW, MISSILE, ORBIT):
                other = self.balls[b.follow_ptr]
                len2 = _lensq(_sub(b.pos, other.pos))
                hi = other.radius + b.follow_range + n + b.radius
                lo = hi - 2.0 * n
                if lo < 0.0:
                    lo = 0.0
                inside = len2 < hi * hi and len2 > lo * lo
            if inside:
                if not b.within_range:
                    send, b.within_range = True, True
            elif b.within_range:
                send, b.within_range = True, False
        if send:
            self._hit("range.event")
            self._event("DoRange", b.id, int(b.within_range))

    def _check_target_tracking(self, b: SpaceBall) -> None:
        """CheckForProximity_TargetTracking (Ball.cpp:595; SEN-703)."""
        send = False
        if b.tracking_range >= 0.00001 and b.tracking_ptr is not None:
            target = b.tracking_ptr
            d2 = _lensq(_sub(b.pos, target.pos))
            lim = b.tracking_range + b.radius
            if d2 < lim * lim:
                if not b.within_tracking_range:
                    send, b.within_tracking_range = True, True
            elif b.within_tracking_range:
                send, b.within_tracking_range = True, False
        if send:
            self._hit("tracking.event")
            self._event("DoTargetTracking", b.id, int(b.within_tracking_range))

    def _check_sensor(self, b: SpaceBall) -> None:
        """CheckForProximity_Sensor (Ball.cpp:642; SEN-701, SEN-702): every period, the candidates found with the
        sensor's radius inflated (and the velocity zeroed); the ball is re-filed twice on the way."""
        b.sensor_elapsed += self.dt
        if b.sensor_elapsed < b.sensor_period:
            self._hit("sensor.wait")
            return
        self._hit("sensor.check")
        b.sensor_elapsed = 0.0
        radius, vel = b.radius, b.vel
        b.vel = [0.0, 0.0, 0.0]
        self._set_ball_radius(b, f32(b.radius + b.sensor_range))
        exclude_fixed = not b.is_cloaked
        uni = self._nearby_balls(b, exclude_fixed, b.sensor_only_interactives, False, False, True)
        b.vel = vel
        self._set_ball_radius(b, radius)
        if len(b.sensor_balls) > 0:
            nearby = {o.id for o in uni}
            for an_id in b.sensor_balls.keys():
                if an_id == b.id or an_id in nearby:
                    continue
                self._hit("sensor.vanished")
                b.sensor_balls.erase(an_id)
                self._event("DoProximity", b.id, an_id, 0)
                if not b.sensor_active:
                    break
        for o in uni:
            if b.is_moribund:
                break
            if o.id == b.id or o.is_moribund:
                continue
            in_sensor = o.id in b.sensor_balls
            d2 = _lensq(_sub(b.pos, o.pos))
            ck = f32(f32(b.sensor_range + o.radius) + b.radius)     # float + float + float
            if d2 > ck * ck:
                if in_sensor:
                    self._hit("sensor.left")
                    b.sensor_balls.erase(o.id)
                    self._event("DoProximity", b.id, o.id, 0)
            elif not in_sensor:
                self._hit("sensor.entered")
                b.sensor_balls.set(o.id, 0)
                self._event("DoProximity", b.id, o.id, 1)
            if not b.sensor_active:
                break

    # ================================================================================================ visibility, range

    def check_visibility(self, src_id: int, dst_id: int) -> int:
        """Ballpark::CheckVisibility (Ballpark.cpp:5299; SEN-709)."""
        src, dst = self.balls.get(src_id), self.balls.get(dst_id)
        if src is None or dst is None:
            return -2
        if src.bubble != dst.bubble:
            self._hit("visibility.other_bubble")
            return -2
        ray = _sub(dst.pos, src.pos)
        distance = math.sqrt(_lensq(ray))
        ray = _normalized(ray)
        for occ in self.balls.values():
            if occ.bubble != src.bubble or occ is src or occ is dst or not occ.is_massive or occ.is_cloaked:
                continue
            to_center = _sub(occ.pos, src.pos)
            v = _dot(ray, to_center)
            r = occ.radius
            dsq = r * r - (_dot(to_center, to_center) - v * v)
            if dsq < 0.0:
                continue
            d = math.sqrt(dsq)
            if v - d > 0.0 and v - d < distance:
                self._hit("visibility.occluded")
                return occ.id
        return 0

    @staticmethod
    def _parse(args: tuple, fmt: str):
        """PyArg_ParseTuple for L, d and b with '|': (ok, values with defaults None)."""
        units = [c for c in fmt if c != "|"]
        required = len(fmt.split("|")[0])
        if not required <= len(args) <= len(units):
            return False, None
        out = []
        for c, a in zip(units, args):
            is_int = isinstance(a, int) and not isinstance(a, float)
            if c in "Lb":
                if not is_int:
                    return False, None
                out.append(int(a))
            else:
                out.append(float(a))
        return True, out + [None] * (len(units) - len(out))

    def ball_ids_in_range(self, *args, include_dist: bool = False):
        """GetBallIdsInRange and GetBallIdsAndDistInRange (Thunkers.cpp:1344; SEN-710), with the Python arguments:
        (ball id, range[, includeCloaked]) or (x, y, z, range[, includeCloaked]). The balls are those listed in the
        bubble's dictionary, in its order; None where Destiny raises."""
        ok, vals = self._parse(args, "Ld|b")
        ref = None
        if ok:
            src_id, rng, cloaked = vals
            ref = self.balls.get(src_id)
            if ref is None:
                return None
            center, bubble_id = ref.pos, ref.bubble
        else:
            ok, vals = self._parse(args, "dddd|b")
            if not ok:
                return None
            x, y, z, rng, cloaked = vals
            center = [x, y, z]
            top = self._get_existing_top_box(center)
            bubble_id = top.bubble if top is not None else -1
        cloaked = cloaked or 0
        out: list = []
        if bubble_id == -1:
            return out
        bubble = self.bubbles.get(bubble_id)
        if bubble is None:
            return out
        for ball_id in list(bubble):
            other = self.balls.get(ball_id)
            if other is None or other is ref:
                continue
            if other.is_moribund:
                self._hit("range.moribund")
                continue
            if other.is_cloaked and not cloaked:
                continue
            d2 = _lensq(_sub(other.pos, center))
            lim = rng + other.radius
            if d2 <= lim * lim:
                out.append((d2, other.id) if include_dist else other.id)
        return out

    # ======================================================================================================= collisions

    def integrate(self, p, v, a, m, k, time_factor, t):
        """Ballpark::Integrate as ref.py, with C's division where m may be zero (a massless free ball)."""
        if t == 0.0:
            return p, v
        if t != self.dt:
            time_factor = math.exp(_ieee_div(-k, m) * t)
        k2 = k * k
        ook2 = 1.0 / k2
        ook = 1.0 / k
        if k < 1e-10 * m * t:
            inv_m = _ieee_div(1.0, m)
            pn = [(m * (a[i] * (-k * t) + k * (a[i] * t + v[i] * k * inv_m * t)) + p[i] * k2) * ook2 for i in range(3)]
        else:
            pn = [(m * (m * a[i] * (time_factor - 1.0) + k * (a[i] * t + v[i] - time_factor * v[i])) + p[i] * k2) * ook2
                  for i in range(3)]
        vn = [(m * a[i] - (m * a[i] - v[i] * k) * time_factor) * ook for i in range(3)]
        return pn, vn

    @staticmethod
    def collide_two_spheres(p0, p1, q0, q1, coll_radius: float) -> float:
        """CollideTwoSpheres (Collision.cpp:108; COL-705): 0 if already touching, else the earlier root in [0, 1]."""
        p0q0 = _sub(p0, q0)
        p0q0_2 = _dot(p0q0, p0q0)
        if p0q0_2 <= coll_radius * coll_radius:
            return 0.0
        dpq = [(p1[i] - p0[i]) - (q1[i] - q0[i]) for i in range(3)]
        a, b, c = _dot(dpq, dpq), 2.0 * _dot(p0q0, dpq), p0q0_2 - coll_radius * coll_radius
        if a == 0.0 and b == 0.0:                           # Quadratic (Collision.cpp:137)
            return -1.0
        det = b * b - 4.0 * a * c
        if det < 0.0:
            return -1.0
        det = math.sqrt(det)
        s2 = _ieee_div((-b - det) * 0.5, a)
        if s2 >= 0.0 and s2 <= 1.0:
            return s2
        return -1.0

    def _gradient(self, b: SpaceBall) -> None:
        """Ballpark::Gradient (Ballpark.cpp:2745; COL-702 to COL-704): Potential with each collision candidate."""
        if b.mode == MISSILE:
            raise NotImplementedError("missile collisions are not in this slice")
        for n in self._nearby_balls(b, False, False, True, True, False):        # collisionCase1
            if ((b.mode == MISSILE and (n.id == b.owner_id or (n.mode == MINIBALL and n.owner_id == b.owner_id)))
                    or ((n.mode == MISSILE or n.mode == MUSHROOM) and b.id == n.owner_id)
                    or (n.mode == MISSILE and b.id == n.follow_id)
                    or (n.is_space_junk and not b.is_space_junk)):
                self._hit("gradient.skip_junk")
                continue
            self._potential(b, n, 0)

    def _potential(self, me: SpaceBall, other: SpaceBall, depth: int) -> None:
        """Ballpark::Potential (Ballpark.cpp:2789; COL-705 to COL-711)."""
        coll_radius = me.radius + other.radius
        m1 = me.mass * me.agility if me.is_free else 1.0e34
        m2 = other.mass * other.agility if other.is_free else 1.0e34
        k, dt = self.friction, self.dt
        p0, q0 = list(me.pos), list(other.pos)
        p1, _ = self.integrate(list(me.pos), list(me.vel), me.last_g, m1, k, me.time_factor, dt)
        q1, _ = self.integrate(list(other.pos), list(other.vel), other.last_g, m2, k, other.time_factor, dt)
        s = self.collide_two_spheres(p0, p1, q0, q1, coll_radius)
        if s == -1.0:
            return
        p1, vp1 = list(me.pos), list(me.vel)
        q1, vq1 = list(other.pos), list(other.vel)
        tf = me.time_factor
        if s > 0.0:
            p1, vp1 = self.integrate(p1, vp1, me.last_g, m1, k, tf, s * dt)
            q1, vq1 = self.integrate(q1, vq1, other.last_g, m2, k, other.time_factor, s * dt)
            normal = _normalized(_sub(q1, p1))
            v1, v2 = _dot(vp1, normal), _dot(vq1, normal)
            if not other.is_free:                           # COL-707: a fixed ball reflects
                self._hit("potential.fixed")
                c = 2.0 * v1
                vp1 = [vp1[i] - normal[i] * c for i in range(3)]
            else:                                           # COL-708: two free balls exchange
                self._hit("potential.free")
                mm1, mm2 = me.mass, other.mass
                v1p = _ieee_div(mm1 * v1 - mm2 * v1 + 2.0 * mm2 * v2, mm1 + mm2)
                vp1 = [vp1[i] + normal[i] * (v1p - v1) for i in range(3)]
            p1, vp1 = self.integrate(p1, vp1, me.last_g, m1, k, tf, (1.0 - s) * dt)
            g, v0 = me.last_g, me.vel
            inv_m, inv_t = _ieee_div(1.0, m1), _ieee_div(1.0, tf - 1.0)
            a1 = [-((((g[i] * -m1) + g[i] * (tf * m1)) - (v0[i] * tf) * k) + vp1[i] * k) * inv_m * inv_t
                  for i in range(3)]
        else:                                               # COL-709: already overlapping
            p0q0 = _sub(q0, p0)
            p0q0_2 = _dot(p0q0, p0q0)
            if p0q0_2 == 0.0:
                self._hit("potential.coincident")
                normal = [1.0, 0.0, 0.0] if me.id > other.id else [-1.0, 0.0, 0.0]
            else:
                normal = _normalized(_sub(q1, p1))
            dist = coll_radius - math.sqrt(p0q0_2) + 1
            d1 = _ieee_div(m2, m1 + m2) * dist
            d2 = _ieee_div(m1, m1 + m2) * dist
            if me.is_free and other.is_free:                # COL-710
                if depth < 2:
                    self._hit("potential.recurse")
                    tmp_me, tmp_other = me.pos, other.pos
                    me.pos = [me.pos[i] - normal[i] * d1 for i in range(3)]
                    other.pos = [other.pos[i] + normal[i] * d2 for i in range(3)]
                    self._potential(me, other, depth + 1)
                    me.pos, other.pos = tmp_me, tmp_other
                    if _ieee_div(m2, m1) > LIGHT_BALL_RATIO:
                        self._hit("potential.light")
                        me.last_c = [c * dist for c in _normalized(me.last_c)]
                    if _lensq(_sub(me.vel, other.vel)) > 0.0001:
                        self._hit("potential.early_return")
                        return
                else:
                    self._hit("potential.depth_limit")
            normal_comp = _dot(me.last_g, normal)
            tmp = 1.0 / (m1 + dt * k)
            c = _ieee_div(-d1, dt * tmp * m1)
            inv_dt = 1.0 / dt
            a1 = [(normal[i] * c - me.vel[i]) * inv_dt - normal[i] * normal_comp for i in range(3)]
        if (me.mode != MISSILE or other.id != me.follow_id) and s >= me.last_collision:     # COL-711
            last_c = [x * RESPONSE_SCALE for x in a1]
            if s == me.last_collision:
                self._hit("potential.equal_time")
                if _lensq(last_c) > _lensq(me.last_c):
                    me.last_c = last_c
            else:
                me.last_c = last_c
                me.last_collision = s
        else:
            self._hit("potential.earlier_kept")
        me.collisions.append(other.id)

    # ================================================================================== the iterative collision method

    @staticmethod
    def _neighbor_ignore(b: SpaceBall, n: SpaceBall) -> bool:
        """CollisionBallNeighborIgnore (Ballpark.cpp:2675; COL-704): owners, targets, space junk, non-massive."""
        return ((b.mode == MISSILE and (n.id == b.owner_id or (n.mode == MINIBALL and n.owner_id == b.owner_id)))
                or ((n.mode == MISSILE or n.mode == MUSHROOM) and b.id == n.owner_id)
                or (n.mode == MISSILE and b.id == n.follow_id)
                or ((n.is_space_junk and not b.is_space_junk) or (not n.is_massive)))

    def _properties(self, b: SpaceBall) -> CollisionBallProperties:
        prop = self.collision_locations.get(b.id)
        if prop is None:
            prop = self.collision_locations[b.id] = CollisionBallProperties(self)
            prop.initialize(b, self.dt)
        return prop

    def _collision_responses(self) -> None:
        """CalculateIterativeCollisionResponses (Ballpark.cpp:332; COL-714): every free, massive, live ball once in
        ascending id; then, up to 20 passes in all, the balls with a new impact and their free candidates."""
        for prop in self.collision_locations.values():
            prop.initialize(prop.ball, self.dt)
        for candidates in self.potential_balls.values():
            candidates.clear()
        current = sorted(self.free_balls)
        iteration = 0
        while True:
            following: set = set()
            for bid in current:
                b = self.balls[bid]
                if b.is_moribund or b.in_dead_bubble or not b.is_massive:
                    continue
                if self._collision_response(b, iteration):
                    self._hit("iterative.found")
                    for n in self.potential_balls[bid]:
                        if n.is_free:
                            following.add(n.id)
                    following.add(bid)
            if not following:
                break
            iteration += 1
            if iteration >= COLLISION_MAX_ITERATIONS:
                self._hit("iterative.cap")
                break
            current = sorted(following)

    def _collision_response(self, b: SpaceBall, iteration: int) -> bool:
        """CalculateCollisionResponse (Ballpark.cpp:1988)."""
        prop = self._properties(b)
        if iteration == 0:
            self.potential_balls[b.id] = self._nearby_balls(b, False, False, True, True, False)
        if prop.t_start >= prop.t_end:
            prop.calculate_end()
            return False
        if prop.locations[0].collider != -1:
            return False
        balls = self.potential_balls.setdefault(b.id, [])
        if not balls:
            return False
        item, neighbour = CollisionItem(), CollisionItem()
        self._next_collision_static(prop, item, balls)
        if item.time != prop.t_start:
            self._next_collision_free(prop, item, balls)
        if item.collider is not None:
            if item.mb2 == -2:
                self._impact_static(prop, item)
            else:
                neighbour = self._impact_free(prop, self.collision_locations[item.collider.id], item)
            if neighbour.impact is not None and neighbour.impact.collider != -1:
                self._add_impact(neighbour.impact, neighbour.collider)
            if item.impact is not None and item.impact.collider != -1:
                self._add_impact(item.impact, b)
                prop.calculate_end()
                return True
        prop.calculate_end()
        return False

    def _next_collision_static(self, prop: CollisionBallProperties, item: CollisionItem, balls: list) -> None:
        """GetNextCollisionStatic (Ballpark.cpp:2080; COL-715): no static collidables here; then the fixed balls,
        whose test compares the fraction with the scaled time held (as coded)."""
        b = prop.ball
        p0, _ = prop.position(prop.t_start)
        p1, _ = prop.position(prop.t_end)
        if item.time != prop.t_start:
            for n in balls:
                if not n.is_free:
                    if self._neighbor_ignore(b, n):
                        continue
                    toi = self.collide_two_spheres(p0, p1, n.pos, n.pos, b.radius + n.radius)
                    if toi >= 0.0:
                        if prop.ignore_collision(n.id, toi):
                            self._hit("iterative.ignored")
                            continue
                        t = prop.t_start + toi * (prop.t_end - prop.t_start)
                        if item.time < 0.0 or toi < item.time:
                            self._hit("iterative.fixed")
                            position = [p0[i] + (p1[i] - p0[i]) * toi for i in range(3)]
                            item.normal = _normalized(_sub(position, n.pos))
                            item.collider, item.time, item.mb1, item.impact = n, t, -1, None
                if item.time == prop.t_start:
                    break

    def _next_collision_free(self, prop: CollisionBallProperties, item: CollisionItem, balls: list) -> None:
        """GetNextCollisionFree (Ballpark.cpp:2180; COL-715): free balls of higher id, sub-stepped at the other ball's
        own impact times."""
        b = prop.ball
        for other in balls:
            if other.is_free and (other.id > b.id or b.mode == MISSILE):
                if self._neighbor_ignore(b, other):
                    continue
                oprop = self._properties(other)
                end_time = min(prop.t_end, oprop.t_end)
                if prop.t_start >= end_time:
                    continue
                times = [prop.t_start] + [loc.time for loc in oprop.locations
                                          if prop.t_start < loc.time < end_time] + [end_time]
                if len(times) > 2:
                    self._hit("iterative.substeps")
                b0, _ = prop.position(times[0])
                o0, _ = oprop.position(times[0])
                for ii in range(len(times) - 1):
                    tstart, tend = times[ii], times[ii + 1]
                    b1, _ = prop.position(tend)
                    o1, _ = oprop.position(tend)
                    toi = self.collide_two_spheres(b0, b1, o0, o1, b.radius + other.radius)
                    if toi >= 0.0:
                        t = tstart + (tend - tstart) * toi
                        if item.time < 0 or t < item.time:
                            if prop.ignore_collision(other.id, toi):
                                self._hit("iterative.ignored")
                                continue
                            self._hit("iterative.free")
                            item.time, item.collider, item.mb1, item.mb2, item.impact = t, other, -1, -1, None
                            break
                    if len(times) > 2:
                        b0, o0 = b1, o1
                if item.time == prop.t_start:
                    break

    def _impact_static(self, prop: CollisionBallProperties, item: CollisionItem) -> None:
        """CalculateCollisionImpactStatic (Ballpark.cpp:2370; COL-716): a perfectly elastic reflection; a ball
        touching at the start is also pushed away (mLastC)."""
        if item.impact is not None or item.mb2 != -2:
            return
        b = prop.ball
        item.impact = CollisionImpact(item.time)
        if prop.t_start == item.time:
            self._hit("iterative.static_contact")
            anormal = _dot(b.last_g, item.normal)
            max_acc = _ieee_div(b.max_vel * self.friction, b.mass * b.agility) - anormal
            b.last_c = [item.normal[i] * max_acc for i in range(3)]
        _, v = prop.position(item.time)
        impulse = _dot([v[i] * -2.0 for i in range(3)], item.normal)
        if impulse < 0.0:
            self._hit("iterative.no_impulse")
            return
        item.impact.dv = [item.normal[i] * impulse for i in range(3)]
        item.impact.collider = item.collider.id

    def _impact_free(self, prop: CollisionBallProperties, oprop: CollisionBallProperties,
                     item: CollisionItem) -> CollisionItem:
        """CalculateCollisionImpactFree (Ballpark.cpp:2442; COL-717): an elastic exchange with base masses; balls
        touching at t = 0 are both pushed apart (mLastC)."""
        if item.impact is not None or item.mb2 == -2:
            return CollisionItem()
        other = CollisionItem()
        other.time, other.collider, other.mb1, other.mb2 = item.time, item.collider, item.mb2, item.mb1
        other.impact, item.impact = CollisionImpact(item.time), CollisionImpact(item.time)
        bp, bv = prop.position(item.time)
        op, ov = oprop.position(item.time)
        normal = _sub(bp, op)
        length = math.sqrt(_lensq(normal))
        if length > 0.0:
            c = 1.0 / length
            normal = [x * c for x in normal]
        else:
            self._hit("iterative.coincident")
            normal[2] = 1.0
        item.normal, other.normal = normal, [-x for x in normal]
        b, o = prop.ball, oprop.ball
        if 0.0 == item.time:
            self._hit("iterative.free_contact")
            anormal = _dot(b.last_g, item.normal)
            max_acc = _ieee_div(b.max_vel * self.friction, b.mass * b.agility) - anormal
            b.last_c = [item.normal[i] * max_acc for i in range(3)]
            anormal = _dot(o.last_g, other.normal)
            max_acc = _ieee_div(o.max_vel * self.friction, o.mass * o.agility) - anormal
            o.last_c = [other.normal[i] * max_acc for i in range(3)]
        impulse = -2.0 * _dot(_sub(ov, bv), other.normal)
        if impulse < 0.0:
            self._hit("iterative.no_impulse")
            return other
        m1, m2 = b.mass, o.mass
        j = _ieee_div(m1 * m2 * impulse, m1 + m2)
        c1, c2 = _ieee_div(j, m1), _ieee_div(j, m2)
        item.impact.dv = [item.normal[i] * c1 for i in range(3)]
        other.impact.dv = [other.normal[i] * c2 for i in range(3)]
        item.impact.collider, other.impact.collider = o.id, b.id
        return other

    def _add_impact(self, impact: CollisionImpact, b: SpaceBall) -> None:
        """Ballpark::AddImpact (Ballpark.cpp:2577; COL-718): a velocity change above 5 x maxVel is scaled by
        25 maxVel^2 / |dv|^2 (25 x maxVel x maxVel in float); later impacts it invalidates are removed in turn."""
        prop = self.collision_locations.get(b.id)
        if prop is None or prop.t_end <= impact.time:
            return
        fraction = _ieee_div(f32(f32(VELOCITY_LIMIT * b.max_vel) * b.max_vel), _lensq(impact.dv))
        if fraction < 1:
            self._hit("iterative.limited")
            impact.dv = [x * fraction for x in impact.dv]
        for t, cid in prop.add_impact(impact):
            if cid in self.free_balls:
                self._hit("iterative.invalidated")
                self._remove_impact_from_free(t, b.id, self.balls[cid])

    def _remove_impact_from_free(self, time: float, collider: int, b: SpaceBall) -> None:
        """RemoveImpactFromFree (Ballpark.cpp:2647)."""
        if b is None or not b.is_free:
            return
        prop = self.collision_locations.get(b.id)
        if prop is None:
            return
        for t, cid in prop.remove_impact(time, collider):
            if cid in self.free_balls:
                self._remove_impact_from_free(t, b.id, self.balls[cid])

    def _position_velocity(self, b: SpaceBall, p: list, v: list) -> tuple:
        """CalculateBallPositionVelocity (Ballpark.cpp:2715): from the ball's collision path if it has one."""
        prop = self.collision_locations.get(b.id)
        if prop is None:
            return super()._position_velocity(b, p, v)
        if self.dt == 0.0:
            return p, v
        return prop.position(self.dt)

    # ============================================================================================================ tick

    def _free_ids(self) -> list:
        return sorted(self.free_balls)

    def _follower_order(self, b: SpaceBall) -> list:
        """StopAllFollowers walks a copy of the unordered_set: libstdc++'s order."""
        return b.followers.ordered()

    def _in_dead_bubble(self, b: SpaceBall) -> bool:
        """Phase 1's cached mInDeadBubble (read by the iterative method), cleared again in phase 4."""
        if self.in_dead_bubble(b):
            self._hit("tick.dead_bubble")
            b.in_dead_bubble = True
            return True
        return False

    def _behaviour(self, b: SpaceBall) -> None:
        """EvolveBehaviorForBall also clears the collision list."""
        b.collisions = []
        super()._behaviour(b)

    def _commit(self, b: SpaceBall, p: list, v: list) -> None:
        """Phase 4's std::swap: the start-of-tick state becomes the old state."""
        b.old_pos, b.old_vel = b.pos, b.vel
        b.pos, b.vel = p, v

    def evolve(self) -> None:
        self.in_evolve = True
        super().evolve()
        for b in self.balls.values():                       # phase 4 clears the flags it set (nothing reads them after)
            b.in_dead_bubble = False
        self.in_evolve = False

    # ========================================================================================================= reports

    @staticmethod
    def box_hash(box: Box) -> int:
        """The harness's per-box checksum (FNV-1a over the box's canonical content)."""
        h = 14695981039346656037

        def mix(h: int, v: int) -> int:
            return ((h ^ (v & _MASK)) * 1099511628211) & _MASK
        for v in (box.level, *box.key, box.bubble, int(box.handled), len(box.balls), *sorted(box.balls),
                  len(box.children)):
            h = mix(h, v)
        for k in sorted(box.children):
            for c in k:
                h = mix(h, c)
        return mix(h, box.parent.level if box.parent is not None else 99)

    def partition_hash(self) -> int:
        return sum(self.box_hash(box) for level in self.levels for box in level.values()) & _MASK

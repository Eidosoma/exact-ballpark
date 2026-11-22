# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Check geometric certificates against the solver and the input to the combat square root."""
from fractions import Fraction
import math
import struct
from unittest.mock import patch

from exact_ballpark import full
from exact_ballpark.space import SpaceBallpark


def bits(value):
    return int.from_bytes(struct.pack(">d", value), "big")


def gradient():
    # Equal times with opposite transverse responses, equal times with a lighter response,
    # and an earlier strong contact followed by a later weaker contact.
    for layout, obstacles in [
        ("opposite", [(6., .75, False), (6., -.75, False)]),
        ("strength", [(6., 0., False), (6., 0., True)]),
        ("time", [(4., 0., False), (9., 0., True)]),
    ]:
        for order in (obstacles, obstacles[::-1]):
            park = full.FullBallpark(friction=1)
            b = park.add_ball(1, 16, .5, 16, True, is_massive=True, agility=1, x=100, y=100, z=100, vx=16)
            candidates = []
            for i, (dx, dy, free) in enumerate(order, 2):
                candidates.append(park.add_ball(i, 16, .5, 16, free, is_massive=True, agility=1,
                                                x=100+dx, y=100+dy, z=100))
            def reset():
                b.last_collision, b.last_c, b.collisions = -1., [0., 0., 0.], []
            hits = {}
            for other in candidates:
                reset()
                park._potential(b, other, 0)
                if b.last_collision <= 0:
                    raise SystemExit(f"Missing positive gradient hit against {other.id}")
                hits[other.id] = (b.last_collision, list(b.last_c))
            reset()
            park._gradient(b)
            if sorted(b.collisions) != sorted(hits):
                raise SystemExit(f"Gradient visited {b.collisions} instead of both candidates")
            def key(i):
                time, (x, y, z) = hits[i]
                return time, (x*x + y*y) + z*z
            winner = max(b.collisions, key=key)
            time, vector = hits[winner]
            if (b.last_collision, list(map(bits, b.last_c))) != (time, list(map(bits, vector))):
                raise SystemExit(f"Gradient chose a different response for {layout} in order {b.collisions}")
            left, right = map(key, (2, 3))
            if ((layout == "opposite" and (left != right or hits[2][1] == hits[3][1])) or
                    (layout == "strength" and (left[0] != right[0] or left[1] == right[1])) or
                    (layout == "time" and (left[0]-right[0])*(left[1]-right[1]) >= 0)):
                raise SystemExit(f"Gradient fixture no longer distinguishes {layout}")


def compare(run_lake):
    origin = (0., 0., 0.)
    sweep = SpaceBallpark.collide_two_spheres
    jobs, expected = [], []

    def submit(kind, args, result):
        jobs.append(",".join(map(str, [*kind, *map(bits, args)])))
        expected.append(result)

    # Strictly interior non-dyadic entry, a tangency, outside the tick, a
    # false rounded midpoint, and an exact endpoint that this checker declines.
    for p, q, radius, depth, entry in [
        ((3., 0., 0.), (-3., 0., 0.), 1., 24, Fraction(1, 3)),
        ((-3., 1., 0.), (3., 1., 0.), 1., 20, Fraction(1, 2)),
        ((4., 0., 0.), (math.nextafter(1., math.inf), 0., 0.), 1., 24, None),
        ((-2.**28, 2., 0.), (2.**28, 2., 0.), 1., 24, None),
        ((2., 0., 0.), (1., 0., 0.), 1., 20, None),
    ]:
        result = sweep(p, q, origin, origin, radius)
        submit(("hit", depth), (*p, *q, radius), ("hit", bits(result), entry, depth))

    # Only the endpoint-based certificate accounts for lost clearance in the quadratic coefficients.
    extent = 2.**28
    for x, u, y, radius, last_only, from_inputs in [
        (-3., 3., extent, extent, "negative", "uncertain"),
        (-extent, extent, 2., 1., "uncertain", "uncertain"),
        (-extent, extent, 5., 1., "negative", "negative"),
        (-3., 3., .5, 1., "positive", "positive"),
        (-3., 3., 1., 1., "uncertain", "uncertain"),
        (-3., 3., 2., 1., "negative", "negative"),
    ]:
        result = sweep((x, y, 0.), (u, y, 0.), origin, origin, radius)
        submit(("disc",), (x, u, y, radius), ("disc", bits(result), last_only, from_inputs))

    for v, r, safe in [
        ((0., 1., 0.), (1., 0., 0.), True),
        ((1., 2.**-12, 0.), (1., 0., 0.), True),
        ((1., 2.**-30, 0.), (1., 0., 0.), False),
        ((.03, .04, .12), (3., 4., 12.), False),
        ((-.12, .03, .04), (-12., 3., 4.), False),
        ((.12, -.04, .03), (12., -4., 3.), False),
        ((3., 4., 12.), (3., 4., 12.), False),
    ]:
        park = full.FullBallpark()
        park.add_ball(1, 1e6, 1, 100, True)
        park.add_ball(2, 1e6, 1, 100, True, x=r[0], y=r[1], z=r[2])
        park.set_velocity(2, *v)
        with patch.object(full, "_sqrt", wraps=full._sqrt) as sqrt:
            park.get_accuracy(1, 2, 1e30, 1., 1., 1., 1.)
        if sqrt.call_count != 1:
            raise SystemExit("Tracking fixture missed the squared angular speed input")
        speed = sqrt.call_args.args[0]
        vq, rq = tuple(map(Fraction, v)), tuple(map(Fraction, r))
        dot = sum(x*y for x, y in zip(vq, rq, strict=True))
        distance_sq = sum(x*x for x in rq)
        lateral = sum(x*x for x in vq) - dot*dot/distance_sq
        submit(("track",), (*v, *r), ("track", bits(speed), lateral, safe))
        plain, fused = [], []
        for i, j in ((1, 2), (2, 0), (0, 1)):
            rounded = v[j]*r[i]
            plain.append(bits(v[i]*r[j] - rounded))
            fused.append(bits(math.fma(v[i], r[j], -rounded) + math.fma(-v[j], r[i], rounded)))
        submit(("cross",), (*v, *r), ("cross", bits(speed), plain + fused, lateral / distance_sq))

    rows = run_lake("env", "lean", "--run", "scripts/GeometryProbe.lean", *jobs, capture=True).splitlines()
    for row, (kind, actual, exact, condition) in zip(rows, expected, strict=True):
        label, encoded, *bounds = row.split()
        if (label, int(encoded)) != (kind, actual):
            raise SystemExit(f"Geometric model differs from the scalar engine on {row}")
        if kind == "disc":
            valid = bounds == [exact, condition]
        elif kind == "cross":
            valid = len(bounds) == 7 and list(map(int, bounds[:6])) == exact and Fraction(bounds[6]) == condition
        elif kind == "hit" and exact is None:
            valid = bounds == ["uncertain"]
        elif len(bounds) == 2:
            a, b = map(Fraction, bounds)
            if kind == "hit":
                rounded, = struct.unpack(">d", actual.to_bytes(8, "big"))
                valid = a <= exact <= b and a <= rounded <= b and b-a == Fraction(1, 2**condition)
            else:
                valid = a == exact and b >= 0 and (b <= a) == condition
        else:
            valid = False
        if not valid:
            raise SystemExit(f"Unexpected {kind} certificate for CPU word {actual:016x}: {bounds}")
    gradient()

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Compare complete ordered candidate lists and packed addresses on the scalar and tensor CPU paths."""
import math
import random
import struct


def worlds():
    grid = [
        (1, (100, 100, 100)), (1, (101, 101, 101)), (1, (1, 100, 100)),
        (1, (math.nextafter(1., 0.), 100, 100)), (1, (0, 0, 100)), (1, (0, 0, 0)),
        (1, (479.5, 479.5, 479.5)), (160, (98, 100, 100)), (161, (100, 100, 100)),
        (641, (-1, -1, -1)), (3000, (100, 100, 100)), (1, (1e6, 1e6, 1e6)),
        (1, (500, 486, 482)), (1, (490, 980, 20)), (1, (1900, 320, 600)),
        (12000, (6, 0, 4)), (45000, (0, -2, 1)), (175000, (1, 1, 1)), (700000, (0, 1, -1)),
    ]
    for size in (1, 6, 12, 16, len(grid)):
        fixture = list(enumerate(grid[:size], 1))
        random.Random(129 + size).shuffle(fixture)
        yield [(103-i*3, r, p) for i, (r, p) in fixture]


def swept_pairs():
    """Reach a common interior point across a face, a corner, and two tree levels."""
    from exact_ballpark.full import FullBallpark
    from exact_ballpark.vec_collide import VecCollidePark

    for first, second, velocity, radius in [
        ((478., 100., 100.), (481., 100., 100.), (4., 0., 0.), 1.),
        ((479.75,)*3, (480.25,)*3, (0., 0., 0.), 1.),
        ((1918., 100., 100.), (1921., 100., 100.), (4., 0., 0.), 161.),
    ]:
        park = FullBallpark(friction=1)
        objects = []
        for i, pos, vel, r in [(1, first, velocity, 1.), (2, second, (0., 0., 0.), radius)]:
            park.add_ball(i, 16., r, 10., i == 1, is_massive=True, x=pos[0], y=pos[1], z=pos[2],
                          vx=vel[0], vy=vel[1], vz=vel[2], agility=1, speed_fraction=1.)
            objects.append(dict(id=i, radius=r, pos=pos, vel=vel, mass=16., max_vel=10., free=i == 1,
                                massive=True, agility=1, speed_fraction=1.))
        moving, fixed = (park.balls[i] for i in (1, 2))
        reached, _ = park.integrate(first, velocity, (0.,)*3, 16., 1., moving.time_factor, .5)
        if (math.dist(reached, second) >= 1+radius or
                math.dist(reached, first)+1 > park.inflated_radius(moving) or
                (moving.my_box.level != fixed.my_box.level) != (radius > 1) or
                moving.my_box == fixed.my_box):
            raise SystemExit("Sweep fixture missed the interior overlap or the grid boundary")
        batched = VecCollidePark(1, 2, friction=1, device="cpu")
        batched.load([objects])
        related = batched._related()[0]
        for i, j in ((0, 1), (1, 0)):
            b = park.balls[objects[i]["id"]]
            ids = [n.id for n in park._nearby_balls(b, False, False, True, True, False)]
            if objects[j]["id"] not in ids or not related[i, j]:
                raise SystemExit(f"Broad phase omitted the contacting pair {i+1}, {j+1}")


def compare(run_lake):
    import torch
    from exact_ballpark.full import FullBallpark
    from exact_ballpark.vec_collide import VecCollidePark, NOT_A_CANDIDATE

    inputs, expected, sizes, levels = [], [], set(), set()
    for fixture in worlds():
        scalar = FullBallpark(is_master=False)
        batched = VecCollidePark(1, len(fixture), device="cpu", kernel="torch")
        objects = []
        for i, radius, pos in fixture:
            free = i % 2 == 0
            objects.append(dict(id=i, radius=radius, pos=pos, vel=(0, 0, 0), mass=16, max_vel=0,
                                free=free, massive=True, agility=1, speed_fraction=0))
            scalar.add_ball(i, 16, radius, 0, free, is_massive=True, agility=1, x=pos[0], y=pos[1], z=pos[2])
        batched.load([objects])
        ids = batched.id[0].tolist()
        active = batched._related() & ~torch.eye(len(ids), dtype=torch.bool).unsqueeze(0)
        order = batched._order_keys(active)
        indices = batched._sorted(active, order)[0].tolist()
        footprints = []
        for row, i in enumerate(ids):
            b = scalar.balls[i]
            level, key, mod = batched.level[0,row].item(), batched.key[0,row].tolist(), batched.mod[0,row].tolist()
            if (b.my_box.level, list(b.my_box.key), b.mod) != (level, key, mod):
                raise SystemExit(f"Different scalar/tensor filing for ball {i}")
            levels.add(level)
            sizes.add(len(b.sorted_boxes()))
            bounds = zip(b.my_box.lo, b.my_box.hi, b.pos, strict=True)
            values = [scalar.inflated_radius(b), *(v for axis in bounds for v in axis)]
            words = [int.from_bytes(struct.pack(">d", v), "big") for v in values]
            footprints.append(",".join(map(str, [i, level, *key, *(d+1 for d in mod), *words])))
            eligible = [ids[j] for j in range(len(ids)) if active[0,row,j]]
            permutation = [j for j in indices[row] if active[0,row,j]]
            scalar_ids = [n.id for n in scalar._nearby(b, False, False, True, True, False)[0]]
            batch_ids = [ids[j] for j in permutation]
            keys = order[0,row].tolist()
            if (scalar_ids != batch_ids or set(eligible) != set(scalar_ids)
                    or any((key == NOT_A_CANDIDATE) != (j not in permutation) for j, key in enumerate(keys))):
                raise SystemExit(f"Different CPU candidates for ball {i}: scalar {scalar_ids}, tensor {batch_ids}")
            expected.append((i, eligible, scalar_ids, batch_ids, [keys[j] for j in permutation]))
        inputs.append(";".join(footprints))
    output = run_lake("env", "lean", "--run", "scripts/CandidateProbe.lean", *inputs, capture=True)
    for line, row in zip(output.splitlines(), expected, strict=True):
        i, *fields = line.split("|")
        parsed = (int(i), *([int(j) for j in field.split(",") if j] for field in fields))
        if parsed != row:
            raise SystemExit(f"Candidate model differs for {row[0]}: Lean {parsed}, CPU {row}")
    if sizes != {1, 2, 4, 8} or levels != set(range(8)):
        raise SystemExit(f"Candidate fixtures exercise cell counts {sizes} and tree levels {levels}")
    swept_pairs()
    return len(expected)

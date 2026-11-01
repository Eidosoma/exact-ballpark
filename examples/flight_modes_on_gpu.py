# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Flight modes on the GPU: a few thousand worlds of ships following, orbiting and warping, stepped as a CUDA graph.
A sample of the same worlds is stepped by the scalar reference, one ball at a time, and compared bit for bit.

    python3 examples/flight_modes_on_gpu.py [--worlds 2048] [--ticks 60] [--check 8] [--seed 1]

Each world holds four ships (ids 1 to 4): ship 1 flies to a point 2 to 50 km away; ship 2 follows ship 1 at 500 m;
ship 3 orbits ship 1 at 2 km (destiny's new orbit law); ship 4 warps to a point 100,000 km away. The batched park
(`exact_ballpark.vec_modes.VecModesPark`) steps every world at once, each tick one replay of a CUDA graph
(`exact_ballpark.vec_modes_cuda.ModesGraph`). The first `--check` worlds are also stepped by
`exact_ballpark.modes.ModeBallpark`, and every ship's position and velocity are compared by their bits after every
tick. The script prints the speed and the number of differences, and exits 1 if there is any.

It needs an NVIDIA GPU and nvcc (the kernel is built at first use; see README.md).
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
import time

import torch

from exact_ballpark.modes import ModeBallpark
from exact_ballpark.ref import STOP

F64, I64 = torch.float64, torch.int64
FOLLOW_RANGE, ORBIT_RANGE, WARP_MIN_RANGE = 500.0, 2000.0, 20000.0


def ships(worlds: int, seed: int) -> list:
    """Four ships a world (ids 1 to 4), each world around its own point; every flag set explicitly."""
    r = random.Random(seed)
    rows = []
    for w in range(worlds):
        centre = [r.uniform(-1e6, 1e6) for _ in range(3)]
        for bid in (1, 2, 3, 4):
            rows.append({"id": bid, "world": w, "time": 1000, "mass": 10 ** r.uniform(6.0, 9.0),
                         "radius": r.uniform(20.0, 300.0), "max_vel": r.uniform(50.0, 800.0),
                         "pos": [c + r.uniform(-5e3, 5e3) for c in centre],
                         "vel": [r.uniform(-50.0, 50.0) for _ in range(3)], "agility": r.uniform(0.1, 2.0),
                         "speed_fraction": 1.0, "free": True, "massive": False, "interactive": False, "cloaked": 0,
                         "bubble": -1, "moribund": False, "mode": STOP})
    return rows


def orders(rows: list, seed: int) -> list:
    """Each world's orders: ship 1's goal, ship 4's warp point and warp factor."""
    r = random.Random(seed + 1)
    out = []
    for w in range(len(rows) // 4):
        p, q = rows[4 * w]["pos"], rows[4 * w + 3]["pos"]
        d = [r.gauss(0.0, 1.0) for _ in range(3)]
        e = [r.gauss(0.0, 1.0) for _ in range(3)]
        nd, ne, dist = sum(x * x for x in d) ** 0.5, sum(x * x for x in e) ** 0.5, r.uniform(2e3, 5e4)
        out.append({"goal": [p[k] + d[k] / nd * dist for k in range(3)],
                    "warp": [q[k] + e[k] / ne * 1e8 for k in range(3)], "warp_factor": r.randint(3000, 10000)})
    return out


def command(park, todo: list) -> None:
    """The orders on the batched park, one command kind at a time over its rows (as the reference gives them)."""
    n = park.n
    kind = torch.arange(n) % 4
    xyz, target = torch.zeros(n, 3, dtype=F64), torch.full((n,), -1, dtype=I64)
    rng, wf = torch.zeros(n, dtype=F64), torch.zeros(n, dtype=I64)
    for w, o in enumerate(todo):
        xyz[4 * w] = torch.tensor(o["goal"], dtype=F64)
        xyz[4 * w + 3] = torch.tensor(o["warp"], dtype=F64)
        target[4 * w + 1] = target[4 * w + 2] = 4 * w
        rng[4 * w + 1], rng[4 * w + 2], rng[4 * w + 3] = FOLLOW_RANGE, ORBIT_RANGE, WARP_MIN_RANGE
        wf[4 * w + 3] = o["warp_factor"]
    dev = park.dev
    xyz, target, rng, wf = xyz.to(dev), target.to(dev), rng.to(dev), wf.to(dev)
    park.goto_point((kind == 0).to(dev), xyz)
    park.follow((kind == 1).to(dev), target, rng)
    park.orbit((kind == 2).to(dev), target, rng)
    park.warp_to((kind == 3).to(dev), xyz, rng, wf)


def reference(rows: list, todo: list, w: int) -> ModeBallpark:
    """World w on the scalar reference, its orders given."""
    park = ModeBallpark(friction=1.0e6, tick_interval_ms=1000, use_new_orbit=True, is_master=True)
    park.current_time = rows[4 * w]["time"]
    for b in rows[4 * w:4 * w + 4]:
        park.add_ball(b["id"], b["mass"], b["radius"], b["max_vel"], b["pos"], b["vel"], b["agility"],
                      b["speed_fraction"], is_free=b["free"], is_massive=b["massive"], is_interactive=b["interactive"],
                      is_cloaked=b["cloaked"], bubble=b["bubble"], is_moribund=b["moribund"], mode=b["mode"])
    o = todo[w]
    park.goto_point(1, *o["goal"])
    park.follow_ball(2, 1, FOLLOW_RANGE)
    park.orbit(3, 1, ORBIT_RANGE)
    park.warp_to(4, *o["warp"], WARP_MIN_RANGE, o["warp_factor"])
    return park


def bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--worlds", type=int, default=2048)
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--check", type=int, default=8, help="worlds also stepped by the scalar reference")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("this example needs an NVIDIA GPU and nvcc", file=sys.stderr)
        return 2
    from exact_ballpark.vec_modes import VecModesPark
    from exact_ballpark.vec_modes_cuda import ModesGraph

    check = min(a.check, a.worlds)
    rows = ships(a.worlds, a.seed)
    todo = orders(rows, a.seed)
    park = VecModesPark(len(rows), device="cuda", use_new_orbit=True)
    park.load(rows)
    command(park, todo)
    refs = [reference(rows, todo, w) for w in range(check)]
    graph = ModesGraph(park)
    differences, modes_seen = 0, set()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(a.ticks):
        graph.tick()
        for ref in refs:
            ref.evolve()
        pos, vel = park.pos[:4 * check].cpu().view(I64), park.vel[:4 * check].cpu().view(I64)
        modes_seen |= set(park.mode.cpu().tolist())
        for w, ref in enumerate(refs):
            for j, bid in enumerate((1, 2, 3, 4)):
                b = ref.balls[bid]
                if [bits(v) for v in b.pos] != pos[4 * w + j].tolist() or \
                        [bits(v) for v in b.vel] != vel[4 * w + j].tolist():
                    differences += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    print(f"{a.worlds} worlds ({len(rows)} ships), {a.ticks} ticks in {wall:.2f} s, including the reference's "
          f"{check} worlds and the comparisons every tick; modes seen: {sorted(modes_seen)}")
    print(f"{check} worlds compared with the scalar reference after every tick: {differences} ship-ticks differ")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())

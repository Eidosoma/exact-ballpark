# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Engineering probe: the batched collisions' speed (exact_ballpark.vec_collide) on race-like worlds, in
world-ticks a second, at a few widths and ball counts, on one device. Never a scientific result.

The units. A width is a number of worlds, each of `--ships` ships and `--obstacles` fixed spheres, which collide.
Each phase reports `worlds` (the width), `ships` (all of them: worlds x a world's ships), `ships_a_world` and
`obstacles` (a world's), `world_ticks_per_s` (worlds x ticks / wall), `ship_ticks_per_s` (ships x ticks / wall) and
`ms_per_tick` (the wall time of one batch tick, every world stepped once). Until 2 November 2026 `ships` was a world's.

A world: K ships (free, massive; radius 20 to 60 m, 1e6 to 1e8 kg, 200 to 800 m/s) and M obstacles (fixed spheres of
100 to 900 m) in a 12 km field; each ship flies GOTO to an obstacle's surface or a point past it, re-aimed every 20
ticks, so contacts (ship-obstacle and ship-ship) happen while most ticks have none. Per configuration: a warm-up, then
ticks for about `--seconds`, timed between device syncs; the phase's epoch window is written so the card's counters
(benchmarks/gpm_window.sh, or a wrapper's sampler) can be read per phase. Also the same worlds with collisions off
(GOTO and STOP alone, as exact_ballpark.vec.VecBallpark).

    python3 benchmarks/collide_speed.py OUT.json --device cuda --widths 1024,8192 --ships 2,4 --obstacles 6,12
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch

from exact_ballpark.vec_collide import VecCollidePark


def worlds(n: int, ships: int, obstacles: int, seed: int) -> list:
    r = random.Random(seed)
    out = []
    for _ in range(n):
        balls = []
        for j in range(obstacles):
            balls.append({"id": 1000 + j, "mass": 1e12, "radius": r.uniform(100.0, 900.0), "max_vel": 0.0,
                          "agility": 1.0, "speed_fraction": 0.0, "free": False, "massive": True,
                          "pos": [r.uniform(-6000.0, 6000.0) for _ in range(3)]})
        for j in range(ships):
            balls.append({"id": 1 + j, "mass": 10 ** r.uniform(6.0, 8.0), "radius": r.uniform(20.0, 60.0),
                          "max_vel": r.uniform(200.0, 800.0), "agility": r.uniform(0.3, 1.5), "speed_fraction": 1.0,
                          "free": True, "massive": True, "pos": [r.uniform(-9000.0, 9000.0) for _ in range(3)]})
        out.append(balls)
    return out


def aim(park: VecCollidePark, gen: torch.Generator) -> None:
    """Every ship to a random obstacle's centre (offset by up to its radius) or a random point in the field."""
    W, N, dev = park.W, park.N, park.dev
    obstacle = (park.exists & ~park.free).cpu()
    pick = torch.multinomial(obstacle.to(torch.float64) + 1e-30, N, replacement=True, generator=gen).to(dev)
    target = torch.gather(park.pos, 1, pick[..., None].expand(-1, -1, 3))
    off = (torch.rand(W, N, 3, generator=gen, dtype=torch.float64) * 2.0 - 1.0).to(dev) * 600.0
    point = (torch.rand(W, N, 3, generator=gen, dtype=torch.float64) * 12000.0 - 6000.0).to(dev)
    use_point = (torch.rand(W, N, generator=gen) < 0.3).to(dev)
    goal = torch.where(use_point[..., None], point, target + off)
    park.goto_point(park.exists & park.free, goal)


def run(device: str, width: int, ships: int, obstacles: int, seconds: float, collisions: bool, iterative: bool,
        contact_slots, seed: int) -> dict:
    dev = torch.device(device)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    park = VecCollidePark(width, ships + obstacles, collisions=collisions, iterative=iterative,
                          contact_slots=contact_slots, device=device)
    park.load(worlds(width, ships, obstacles, seed))
    gen = torch.Generator().manual_seed(seed)
    aim(park, gen)
    for t in range(3):                                   # warm-up (allocator, first kernels)
        park.evolve()
    sync()
    a = time.time()
    ticks, contacts, passes0 = 0, 0, park.passes
    while True:
        if ticks % 20 == 0:
            aim(park, gen)
        park.evolve()
        ticks += 1
        if ticks % 5 == 0:
            sync()
            if time.time() - a >= seconds:
                break
    contacts = int(park.hits_n.sum()) if not iterative else None
    passes = park.passes - passes0
    sync()
    b = time.time()
    park.check()
    out = {"device": device, "width": width, "worlds": width, "ships": width * ships, "ships_a_world": ships,
           "obstacles": obstacles, "collisions": collisions,
           "method": "iterative" if iterative else "gradient", "contact_slots": contact_slots, "ticks": ticks,
           "seconds": round(b - a, 3), "world_ticks_per_s": round(width * ticks / (b - a), 1),
           "ship_ticks_per_s": round(width * ships * ticks / (b - a), 1),
           "ms_per_tick": round(1000.0 * (b - a) / ticks, 3), "contacts_last_tick": contacts,
           "passes_per_tick": round(passes / ticks, 2) if iterative else None, "a": a, "b": b}
    if dev.type == "cuda":
        out["peak_mib"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--widths", default="1024,8192")
    ap.add_argument("--ships", default="2,4")
    ap.add_argument("--obstacles", default="6,12")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--iterative", action="store_true")
    ap.add_argument("--contact-slots", type=int, default=None)
    ap.add_argument("--off", action="store_true", help="also each configuration with collisions off")
    ap.add_argument("--gap", type=float, default=2.0, help="idle seconds between phases (for the counters)")
    a = ap.parse_args()
    phases = []
    for width in map(int, a.widths.split(",")):
        for ships, obstacles in zip(map(int, a.ships.split(",")), map(int, a.obstacles.split(","))):
            for collisions in ([True, False] if a.off else [True]):
                r = run(a.device, width, ships, obstacles, a.seconds, collisions, a.iterative, a.contact_slots, 1)
                r["name"] = f"w{width}_k{ships}_m{obstacles}_{'on' if collisions else 'off'}"
                phases.append(r)
                print(json.dumps({k: v for k, v in r.items() if k not in ("a", "b")}), flush=True)
                time.sleep(a.gap)
    json.dump({"phases": phases}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

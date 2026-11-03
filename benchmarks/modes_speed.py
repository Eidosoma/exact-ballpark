# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Engineering probe: the flight modes' speed on the GPU (exact_ballpark.vec_modes, one CUDA kernel a
tick), at a few widths, against flight alone (vec.py's VecBallpark, GOTO) and the turning kernel
(VecOrientPark, GOTO with dynamical orientation). Never a scientific result.

The units. A width is a number of ships. On the flight-modes park (modes "modes", "modes_graph", "modes_do" and
"modes_do_graph") the ships fly in worlds of four that interact (a follower and an orbiter of their world's leader), so
n ships are ceil(n / 4) worlds, counted from the park's own world labels; VecOrientPark ("turn2", "turn2_graph") and
VecBallpark ("none") ignore the world labels, every ship its own world, so n ships are n worlds. Each phase reports
`worlds`, `ships`, `world_ticks_per_s` (worlds x ticks / wall), `ship_ticks_per_s` (ships x ticks / wall) and
`ms_per_tick` (the wall time of one batch tick, every world stepped once). Until 2 November 2026 `world_ticks_per_s`
was ships x ticks / wall, four times the world rate on the flight-modes park.

The mixed ships, in worlds of four (ids 1 to 4): a leader flying GOTO to points 2 to 50 km away, re-aimed every 10
ticks; a follower of it (FOLLOW, range 500 m); an orbiter of it (ORBIT under the new law, range 2 km); and a ship
warping 1e8 to 1e9 m away at warp factors 3,000 to 10,000, sent again every 40 ticks (aligning, warp proper, dropping
out, and its followers' release in between); a mass and a maximum speed set again every 25 ticks. Per configuration: a
warm-up, then ticks for about `--seconds`, timed between device syncs; each phase's epoch window is written for the
card's counters (benchmarks/gpm_window.sh).

Modes: "modes" (the kernel, eager ticks), "modes_graph" (each tick one replay of vec_modes_cuda.ModesGraph), "modes_do"
and "modes_do_graph" (the same with dynamical orientation), "turn2" and "turn2_graph" (VecOrientPark with dynamical
orientation, kernel="cuda", GOTO alone, its TickGraph), "none" (vec.py alone, GOTO).

    python3 benchmarks/modes_speed.py OUT.json --widths 1024,8192,65536,262144 --seconds 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from exact_ballpark.vec import VecBallpark
from exact_ballpark.vec_modes import VecModesPark
from exact_ballpark.vec_orient import VecOrientPark

F64, I64 = torch.float64, torch.int64


def ships(n: int, seed: int) -> list:
    """n ships in worlds of four (the last world may be short), each world around its own point."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        if i % 4 == 0:
            centre = [r.uniform(-1e6, 1e6) for _ in range(3)]
        out.append({"id": i % 4 + 1, "world": i // 4, "time": 1000, "mass": 10 ** r.uniform(6.0, 9.0),
                    "radius": r.uniform(20.0, 300.0), "max_vel": r.uniform(50.0, 800.0),
                    "pos": [c + r.uniform(-5e3, 5e3) for c in centre], "vel": [r.uniform(-50.0, 50.0) for _ in range(3)],
                    "agility": r.uniform(0.1, 2.0), "speed_fraction": 1.0, "max_ang_vel": r.uniform(0.2, 2.0),
                    "rot_agility": r.uniform(0.3, 2.0), "mode": 2, "interactive": r.random() < 0.5})
    return out


def worlds_of(park) -> int:
    """The worlds a park steps: the flight-modes park's distinct world labels (worlds of four here); every row of
    VecOrientPark and VecBallpark is its own world."""
    if isinstance(park, VecModesPark):
        return int(torch.unique(park.world).numel())
    return int(park.n)


def roles(n: int, dev) -> dict:
    k = torch.arange(n, device=dev) % 4
    leader = torch.arange(n, device=dev) - k
    return {"leader": k == 0, "follower": k == 1, "orbiter": k == 2, "warper": k == 3,
            "target": torch.where(leader < n, leader, 0)}


def aim(park, gen: torch.Generator, mask) -> None:
    """The masked ships to points 2 to 50 km away in random directions."""
    n, dev = park.n, park.dev
    d = torch.randn(n, 3, generator=gen, dtype=F64)
    d = d / d.norm(dim=1, keepdim=True)
    dist = torch.rand(n, 1, generator=gen, dtype=F64) * 48000.0 + 2000.0
    goal = park.pos + (d * dist).to(dev)
    if isinstance(park, VecOrientPark):
        park.goto_point(mask, goal)
    else:
        park.goto, park.mode = torch.where(mask[:, None], goal, park.goto), torch.where(mask, 0, park.mode)


def warp(park, gen: torch.Generator, mask) -> None:
    n, dev = park.n, park.dev
    d = torch.randn(n, 3, generator=gen, dtype=F64)
    d = d / d.norm(dim=1, keepdim=True)
    dist = 10 ** (torch.rand(n, 1, generator=gen, dtype=F64) + 8.0)
    wf = (torch.rand(n, generator=gen, dtype=F64) * 7000.0 + 3000.0).to(I64)
    park.warp_to(mask, park.pos + (d * dist).to(dev), torch.zeros(n, dtype=F64, device=dev), wf.to(dev))


def run(device: str, width: int, mode: str, seconds: float, seed: int) -> dict:
    """One phase: `width` ships, in worlds of four on the flight-modes park and one ship a world elsewhere (the
    module's docstring). The graph modes need a CUDA device; the others also run on the CPU (the parks' tensor
    ticks)."""
    dev = torch.device(device)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    rows = ships(width, seed)
    every = torch.ones(width, dtype=torch.bool, device=dev)
    gen = torch.Generator().manual_seed(seed)
    if mode == "none":
        park = VecBallpark(width, device=device)
        park.load(rows)
        lead = every
    elif mode.startswith("turn2"):
        park = VecOrientPark(width, device=device, dynamical_orientation=True, kernel="cuda")
        park.load(rows)
        lead = every
    else:
        park = VecModesPark(width, device=device, dynamical_orientation=mode.startswith("modes_do"), use_new_orbit=True)
        park.load(rows)
        ro = roles(width, dev)
        lead = ro["leader"]
        park.follow(ro["follower"], ro["target"], torch.full((width,), 500.0, dtype=F64, device=dev))
        park.orbit(ro["orbiter"], ro["target"], torch.full((width,), 2000.0, dtype=F64, device=dev))
    for _ in range(3):                                      # warm-up
        aim(park, gen, lead)
        park.evolve()
    tick = park.evolve
    if mode.endswith("graph"):
        if mode.startswith("turn2"):
            from exact_ballpark.vec_orient_cuda import TickGraph
            tick = TickGraph(park).tick
        else:
            from exact_ballpark.vec_modes_cuda import ModesGraph
            tick = ModesGraph(park).tick
        tick()
    sync()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    ticks, t0, e0 = 0, time.perf_counter(), time.time()
    while True:
        if ticks % 10 == 0:
            aim(park, gen, lead)
        if ticks % 40 == 0 and isinstance(park, VecModesPark):
            warp(park, gen, ro["warper"])
        if ticks % 25 == 0 and mode != "none":
            park.set_mass(every, park.mass * 1.01)
            park.set_max_velocity(every, park.max_vel * 0.999)
        tick()
        ticks += 1
        if ticks % 5 == 0:
            sync()
            if time.perf_counter() - t0 >= seconds:
                break
    sync()
    wall, e1 = time.perf_counter() - t0, time.time()
    worlds = worlds_of(park)
    out = {"device": device, "width": width, "mode": mode, "worlds": worlds, "ships": width, "ticks": ticks,
           "wall_s": round(wall, 3), "world_ticks_per_s": round(worlds * ticks / wall, 1),
           "ship_ticks_per_s": round(width * ticks / wall, 1), "ms_per_tick": round(1000.0 * wall / ticks, 3),
           "epoch_start": round(e0, 3), "epoch_end": round(e1, 3), "pid": os.getpid()}
    if dev.type == "cuda":
        out["peak_mib"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1)
    if isinstance(park, VecModesPark):
        m = park.mode.cpu()
        out["modes_at_end"] = {k: int((m == v).sum()) for k, v in (("goto", 0), ("follow", 1), ("stop", 2),
                                                                     ("warp", 3), ("orbit", 4))}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--widths", default="1024,8192,65536,262144")
    ap.add_argument("--modes", default="modes,modes_graph,modes_do,modes_do_graph,turn2,turn2_graph,none")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    phases = []
    for mode in a.modes.split(","):
        for w in (int(x) for x in a.widths.split(",")):
            res = run(a.device, w, mode, a.seconds, a.seed)
            print(json.dumps(res), flush=True)          # worlds, ships and both rates (the module's docstring)
            phases.append(res)
    with open(a.out, "w") as f:
        json.dump({"phases": phases}, f, indent=1)


if __name__ == "__main__":
    main()

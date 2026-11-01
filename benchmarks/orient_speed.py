# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Engineering probe: the batched turning's speed (exact_ballpark.vec_orient), in world-ticks a second (a
world here is one ship), at a few widths, with and without dynamical orientation, on one device. Never a scientific
result.

The ships: 1e6 to 1e9 kg, agility 0.1 to 2, 50 to 800 m/s, maximum angular speed 0.2 to 2 rad/s, rotational agility
0.3 to 2, flying GOTO to points 2 to 50 km away in random directions, re-aimed every 10 ticks (so they keep turning,
often through more than 90 degrees), a mass and a maximum speed set again every 25 ticks (the setters in flight). Per
configuration: a warm-up, then ticks for about `--seconds`, timed between device syncs; each phase's epoch window is
written so the card's counters (benchmarks/gpm_window.sh) can be read per phase. Also the same ships in vec.py's
VecBallpark (GOTO and STOP alone, no orientation) for comparison.

Modes: "dynamical" (the torch path), "cuda" (the same ticks with dynamical orientation as one CUDA kernel,
exact_ballpark.vec_orient_cuda), "cuda_graph" (those ticks each captured as one CUDA graph,
vec_orient_cuda.TickGraph), "ypr" (yaw, pitch and roll), "none" (vec.py alone).

    python3 benchmarks/orient_speed.py OUT.json --device cuda --widths 1024,8192,65536 --seconds 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from exact_ballpark.vec import VecBallpark
from exact_ballpark.vec_orient import VecOrientPark


def ships(n: int, seed: int) -> list:
    r = random.Random(seed)
    return [{"mass": 10 ** r.uniform(6.0, 9.0), "radius": r.uniform(20.0, 300.0), "max_vel": r.uniform(50.0, 800.0),
             "pos": [r.uniform(-1e4, 1e4) for _ in range(3)], "vel": [r.uniform(-50.0, 50.0) for _ in range(3)],
             "agility": r.uniform(0.1, 2.0), "speed_fraction": 1.0, "max_ang_vel": r.uniform(0.2, 2.0),
             "rot_agility": r.uniform(0.3, 2.0), "mode": 2} for _ in range(n)]


def aim(park, gen: torch.Generator) -> None:
    """Every ship to a point 2 to 50 km away in a random direction."""
    n, dev = park.n, park.dev
    d = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    d = d / d.norm(dim=1, keepdim=True)
    dist = torch.rand(n, 1, generator=gen, dtype=torch.float64) * 48000.0 + 2000.0
    goal = park.pos + (d * dist).to(dev)
    every = torch.ones(n, dtype=torch.bool, device=dev)
    if isinstance(park, VecOrientPark):
        park.goto_point(every, goal)
    else:
        park.goto, park.mode = goal, torch.zeros_like(park.mode)


def run(device: str, width: int, do, seconds: float, seed: int, kernel: str = "torch", graph: bool = False) -> dict:
    """One phase: do is True, False, or None for vec.py's VecBallpark; kernel and graph as the modes say."""
    dev = torch.device(device)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    rows = ships(width, seed)
    if do is None:
        park = VecBallpark(width, device=device)
        park.load(rows)
    else:
        park = VecOrientPark(width, device=device, dynamical_orientation=do, kernel=kernel)
        park.load(rows)
    gen = torch.Generator().manual_seed(seed)
    every = torch.ones(width, dtype=torch.bool, device=dev)
    for _ in range(3):                                      # warm-up
        aim(park, gen)
        park.evolve()
    tick = park.evolve
    if graph:
        from exact_ballpark.vec_orient_cuda import TickGraph
        tick = TickGraph(park).tick
        tick()
    sync()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    ticks, t0, e0 = 0, time.perf_counter(), time.time()
    while True:
        if ticks % 10 == 0:
            aim(park, gen)
        if ticks % 25 == 0 and do is not None:
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
    name = {True: "dynamical", False: "yaw-pitch-roll", None: "none (vec.py)"}[do]
    name += {("torch", False): "", ("cuda", False): " (CUDA kernel)", ("cuda", True): " (CUDA kernel, graph)"}[kernel,
                                                                                                         graph]
    out = {"device": device, "width": width, "orientation": name,
           "ticks": ticks, "wall_s": round(wall, 3), "world_ticks_per_s": round(width * ticks / wall, 1),
           "ms_per_tick": round(1000.0 * wall / ticks, 3), "epoch_start": round(e0, 3), "epoch_end": round(e1, 3),
           "pid": os.getpid()}
    if dev.type == "cuda":
        out["peak_mib"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--widths", default="1024,8192,65536")
    ap.add_argument("--modes", default="dynamical,ypr,none")       # also cuda, cuda_graph
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    which = {"dynamical": (True, "torch", False), "cuda": (True, "cuda", False), "cuda_graph": (True, "cuda", True),
             "ypr": (False, "torch", False), "none": (None, "torch", False)}
    phases = []
    for mode in a.modes.split(","):
        for w in (int(x) for x in a.widths.split(",")):
            do, kernel, graph = which[mode]
            res = run(a.device, w, do, a.seconds, a.seed, kernel, graph)
            print(json.dumps(res), flush=True)
            phases.append(res)
    with open(a.out, "w") as f:
        json.dump({"phases": phases}, f, indent=1)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Speed and memory probe for exact_ballpark.vec_space with proximity maps and sensor lists of
any size (the default capacity: the world's slots) against the former fixed capacity of 13 (umap_cap=13, or an older source when run with that source on PYTHONPATH: --label names it).

Worlds: N ships in one level-1 box (radius 200 km: two boxes each), on VecModesPark, flying slowly under
GotoDirection, every ship a proximity sensor of range R and a 1 s period (checked every tick: re-filed twice and its
candidates walked), so the proximity map holds all N; R = 2,000 km fills each list with the N - 1 others, R = 0 keeps
the lists empty. The first ticks run eagerly (the lists fill), then per width: ms a tick as one replay of SpaceGraph
(the dead-bubble skip, the park's kernel, the re-filings, HandleProximities), timed between device syncs for about
`seconds`; the layer's state bytes a world; the largest bucket count reached. Engineering only: the login card is
shared, so a timing is provisional.

The units. A width (W) is a number of worlds, each of N ships. Each record reports `worlds` (W, also as `W`), `ships`
(W x N), `ms_per_tick` (the wall time of one batch tick, every world stepped once; also as `ms_tick`, rounded),
`world_ticks_per_s` and `ship_ticks_per_s`.

    python benchmarks/space_primes_speed.py --cases 4:5000,13:2e6,64:2e6 --widths 256,2048 [--cap 13] [--label X]
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from exact_ballpark import vec_space as V
from exact_ballpark.space import LEVEL_WIDTH
from exact_ballpark.vec_modes import VecModesPark


def build(W: int, N: int, R: float, device: str, cap: int | None) -> tuple:
    g = torch.Generator().manual_seed(N)
    w1 = LEVEL_WIDTH[1]
    rows = []
    for w in range(W):
        for k in range(N):
            off = (torch.rand(3, generator=g) - 0.5) * 8.0e5
            rows.append({"id": 101 + 13 * k + (k % 3), "world": w, "time": 0, "mass": 1e7, "radius": 2.0e5,
                         "max_vel": 30.0, "pos": [1.5 * w1 + float(off[q]) for q in range(3)], "vel": [0.0, 0.0, 0.0],
                         "agility": 0.5, "speed_fraction": 1.0, "free": True, "massive": False, "interactive": True,
                         "cloaked": 0, "bubble": -1, "moribund": False})
    p = VecModesPark(len(rows), device=device)
    p.load(rows)
    kw = {} if cap is None else {"umap_cap": cap}
    space = V.VecSpace(V.ModesView(p), torch.arange(len(rows)).reshape(W, N), events=N * N + 2 * N, **kw)
    space.load()
    d = (torch.rand(W * N, 3, generator=g) - 0.5).to(torch.float64).to(p.dev)
    p.goto_direction(torch.ones(p.n, dtype=torch.bool, device=p.dev), d)
    code = torch.full((W, N), V.OP["ADD_SENSOR"], dtype=torch.int64, device=p.dev)
    slots = torch.arange(N, dtype=torch.int64, device=p.dev).expand(W, N).contiguous()
    f = torch.zeros(W, N, 4, dtype=torch.float64, device=p.dev)
    f[:, :, 0], f[:, :, 1] = R, 1.0
    space.tape(code, slots, torch.zeros_like(code), f)
    return p, space


def units(W: int, N: int, ms: float) -> dict:
    """A record's units from its ms a batch tick (the module's docstring)."""
    return {"worlds": W, "ships": W * N, "ms_per_tick": ms, "world_ticks_per_s": round(W * 1e3 / ms, 1),
            "ship_ticks_per_s": round(W * N * 1e3 / ms, 1)}


def timed(fn, seconds: float, dev) -> float:
    fn()
    torch.cuda.synchronize(dev)
    n, t0 = 0, time.perf_counter()
    while True:
        fn()
        n += 1
        if n % 8 == 0:
            torch.cuda.synchronize(dev)
            if time.perf_counter() - t0 >= seconds:
                break
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t0) / n * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="4:5000,13:2e6,64:2e6")
    ap.add_argument("--widths", default="256,2048")
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = []
    for case in a.cases.split(","):
        N, R = int(case.split(":")[0]), float(case.split(":")[1])
        for W in [int(x) for x in a.widths.split(",")]:
            torch.cuda.reset_peak_memory_stats(dev)
            p, space = build(W, N, R, "cuda", a.cap)
            for _ in range(3):
                space.tick(p)
            space.clear_events()
            flags = int((space.st["flags"] != 0).sum())
            g = V.SpaceGraph(space, p)
            ms = timed(g.tick, a.seconds, dev)
            st_bytes = sum(v.numel() * v.element_size() for v in space.st.values() if v is not None)
            ev_bytes = space.st["ev"].numel() * space.st["ev"].element_size()
            rec = {"label": a.label, "N": N, "R": R, "W": W, **units(W, N, ms), "S": space.S,
                   "ms_tick": round(ms, 3), "flagged_worlds": flags, "longest_list": int(space.st["slen"].max()),
                   "map": int(space.st["pn"].max()),
                   "bucket_max": int(space.st["pbkt"].max()) if "pbkt" in space.st else None,
                   "state_bytes_a_world": round((st_bytes - ev_bytes) / W, 1),
                   "event_bytes_a_world": round(ev_bytes / W, 1),
                   "peak_allocated_MiB": round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1)}
            print(json.dumps(rec), flush=True)
            out.append(rec)
            del g, space, p
            torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

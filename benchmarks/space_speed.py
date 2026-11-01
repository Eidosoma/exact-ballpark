# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Speed probe for Destiny's space on the master batched (exact_ballpark.vec_space): worlds of four
ships flying under GotoDirection near and across a top box's face (one bubble a world: the two tops touch, so the
second takes the first's id), every ship re-filed every tick, each with a 5 km proximity sensor of a 1 s period
(checked, so re-filed twice more, every tick), on VecModesPark. Per width, ms a tick: the park's tick alone (its kernel), the park with the layer
eager (VecSpace.tick: the dead-bubble skip, the kernel, the re-filings and HandleProximities; three of the layer's
launches and the park's two), the same as one replay of a CUDA graph (SpaceGraph), and as a graph without the sensors
(the partition alone). Each mode a warm-up, then
ticks for about `seconds`, timed between device syncs. Engineering only: the login card is shared."""

from __future__ import annotations

import argparse
import json
import time

import torch

from exact_ballpark import vec_space as V
from exact_ballpark.space import LEVEL_WIDTH
from exact_ballpark.vec_modes import VecModesPark

TOP = LEVEL_WIDTH[0]


def build(W: int, device: str, sensors: bool = True) -> tuple:
    g = torch.Generator().manual_seed(1)
    rows = []
    for w in range(W):
        for k in range(4):
            off = (torch.rand(3, generator=g) - 0.5) * 8000.0
            pos = [TOP + float(off[0]), 0.3 * TOP + float(off[1]), -0.3 * TOP + float(off[2])]
            if k == 0:
                pos[0] -= 0.5 * TOP                         # one ship deep in the left top: its bubble first
            rows.append({"id": 101 + k, "world": w, "time": 0, "mass": 1e7, "radius": 50.0, "max_vel": 300.0,
                         "pos": pos, "vel": [0.0, 0.0, 0.0], "agility": 0.5, "speed_fraction": 1.0, "free": True,
                         "massive": False, "interactive": True, "cloaked": 0, "bubble": -1, "moribund": False})
    p = VecModesPark(len(rows), device=device)
    p.load(rows)
    space = V.VecSpace(V.ModesView(p), torch.arange(len(rows)).reshape(W, 4))
    space.load()
    d = torch.tensor([[1.0, 0.2, 0.1] if k % 2 else [-1.0, -0.1, 0.2] for _ in range(W) for k in range(4)],
                     dtype=torch.float64, device=p.dev)
    p.goto_direction(torch.ones(p.n, dtype=torch.bool, device=p.dev), d)
    if sensors:
        code = torch.full((W, 4), V.OP["ADD_SENSOR"], dtype=torch.int64, device=p.dev)
        slots = torch.arange(4, dtype=torch.int64, device=p.dev).expand(W, 4).contiguous()
        f = torch.zeros(W, 4, 4, dtype=torch.float64, device=p.dev)
        f[:, :, 0], f[:, :, 1] = 5000.0, 1.0
        space.tape(code, slots, torch.zeros_like(code), f)
    return p, space


def build_server(W: int, device: str) -> tuple:
    """SpaceServer over VecModesPark: the same worlds, every ship piloted, each pilot sending GotoDirection and
    SetSpeedFraction every tick (the probe's shape: 8 Actions calls a world a tick)."""
    from exact_ballpark import vec_server_space as VSS
    p, _ = build(W, device)
    a = VSS.SpaceModesAdapter(p)
    server = VSS.SpaceServer(a, [k % 4 for k in range(p.n)])
    g = torch.Generator().manual_seed(2)
    d = (torch.rand(W, 4, 3, generator=g) - 0.5).to(p.dev).to(torch.float64)
    sf = torch.rand(W, 4, generator=g).to(p.dev).to(torch.float64)
    ids = a.ids_of_world
    mask = torch.ones(W, dtype=torch.bool, device=p.dev)

    def tick():
        for k in range(4):
            server.push("GotoDirection", mask, ids[:, k], f=d[:, k])
            server.push("SetSpeedFraction", mask, ids[:, k], f=sf[:, k:k + 1])
        server.pre_tick(snapshot=False)
        server.evolve()
        server.post_tick()
    return server, tick


def timed(fn, seconds: float, dev) -> float:
    fn()
    torch.cuda.synchronize(dev)
    n, t0 = 0, time.perf_counter()
    while True:
        fn()
        n += 1
        if n % 10 == 0:
            torch.cuda.synchronize(dev)
            if time.perf_counter() - t0 > seconds:
                break
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t0) / n * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="256,2048,16384,65536", help="worlds (4 ships each)")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--server-widths", default="256,2048,16384", help="worlds (4 ships each)")
    a = ap.parse_args()
    dev = torch.device("cuda", torch.cuda.current_device())
    out = {"device": torch.cuda.get_device_name(dev), "rows": []}
    for W in [int(x) for x in a.widths.split(",")]:
        p, space = build(W, "cuda")
        park_ms = timed(p.evolve, a.seconds, dev)
        p, space = build(W, "cuda")
        eager_ms = timed(lambda: space.tick(p), a.seconds, dev)
        bubbles = int(space.st["bubble_id"].float().mean().item())
        p, space = build(W, "cuda")
        graph = V.SpaceGraph(space, p, warm=2)
        graph_ms = timed(graph.tick, a.seconds, dev)
        del graph
        p2, space2 = build(W, "cuda", sensors=False)
        graph2 = V.SpaceGraph(space2, p2, warm=2)
        partition_ms = timed(graph2.tick, a.seconds, dev)
        del p2, space2, graph2
        row = {"worlds": W, "ships": 4 * W, "park_ms": round(park_ms, 4), "layer_eager_ms": round(eager_ms, 4),
               "layer_graph_ms": round(graph_ms, 4), "partition_only_graph_ms": round(partition_ms, 4),
               "mean_bubble_ids": bubbles,
               "events_last": int(space.st["evn"].sum().item()), "flags": int((space.st["flags"] != 0).sum().item()),
               "peak_mib": round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1)}
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(out, f, indent=1)
        del p, space
        torch.cuda.empty_cache()
    out["server_rows"] = []
    for W in [int(x) for x in a.server_widths.split(",") if x]:
        server, tick = build_server(W, "cuda")
        ms = timed(tick, a.seconds, dev)
        row = {"worlds": W, "ships": 4 * W, "server_torch_eager_ms": round(ms, 3),
               "peak_mib": round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1)}
        out["server_rows"].append(row)
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(out, f, indent=1)
        del server, tick
        torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

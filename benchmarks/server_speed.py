# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Engineering probe: the batched server layer's cost on the GPU (exact_ballpark.vec_server), against the
parks alone, at a few widths. Never a scientific result; the login card is shared, so every timing is provisional.

Worlds of four ships, each flown by a pilot (a game's shape): every tick each pilot sends GotoDirection and
SetSpeedFraction (8 Actions calls a world a tick, Destiny's order), with directions and speed fractions drawn once.
Per park (VecModesPark, its kernel; VecOrientPark with dynamical orientation on its kernel; VecCollidePark on its
kernel with contact_slots) and width, the modes:
- "park": the park's tick alone (no command);
- "direct": the same commands applied straight to the park (two calls for every ship at once),
  then its tick: no queue, no stamps, no routing, no update batches;
- "server": the server layer, eager: the 8 pushes (TIME-710's duplicate test), the pre-tick (the commands in queue
  order, the routing), the park's tick, the post-tick;
- "server_graph": the same as one replay of a ServerGraph;
- "server_k" and "server_k_graph" (VecModesPark only): the same on the server kernel (exact_ballpark.vec_server_cuda,
  one launch a pre-tick), eager and as a ServerGraph.
Each: a warm-up, then ticks for about `--seconds`, timed between device syncs; ms a tick and ship-ticks a second;
torch's peak allocation.

The units. A width is a number of ships, in worlds of four (K): n ships are n / 4 worlds. Each row reports `ships`,
`worlds` (the server's own count), `world_ticks_per_s` (worlds x ticks / wall), `ship_ticks_per_s` (ships x ticks /
wall; `mships_per_s` is the same in millions) and `ms_per_tick` (the wall time of one batch tick, every world stepped
once).

    python3 benchmarks/server_speed.py OUT.json --widths 1024,8192,65536,262144 --seconds 3
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time

import torch

from exact_ballpark import vec_server as VS
from exact_ballpark.vec_collide import VecCollidePark
from exact_ballpark.vec_modes import VecModesPark
from exact_ballpark.vec_orient import VecOrientPark

F64, I64 = torch.float64, torch.int64
K = 4                                                       # ships a world


def _ships(n: int, seed: int, spread: float) -> list:
    r = random.Random(seed)
    out = []
    for i in range(n):
        if i % K == 0:
            centre = [r.uniform(-1e5, 1e5) for _ in range(3)]
        out.append({"id": i % K + 1, "world": i // K, "mass": 10 ** r.uniform(6.0, 8.0), "radius": r.uniform(20.0, 80.0),
                    "max_vel": r.uniform(100.0, 400.0), "pos": [c + r.uniform(-spread, spread) for c in centre],
                    "vel": [0.0, 0.0, 0.0], "agility": r.uniform(0.3, 1.5), "speed_fraction": 1.0})
    return out


def build(park: str, n: int, dev, server_kernel: str = "torch"):
    """(park, adapter, server) for n ships in worlds of K, every ship interactive and piloted."""
    sh = _ships(n, 7, 4e3 if park != "collide" else 600.0)
    pilots = [s["id"] - 1 for s in sh]
    if park == "modes":
        p = VecModesPark(n, device=dev)
        p.load([dict(s, time=0, free=True, massive=False, interactive=True, cloaked=0, bubble=0, moribund=False)
                for s in sh])
        a = VS.ModesAdapter(p)
    elif park == "orient":
        p = VecOrientPark(n, device=dev, dynamical_orientation=True, kernel="cuda")
        p.load(sh)
        a = VS.OrientAdapter(p, world=[s["world"] for s in sh], ids=[s["id"] for s in sh])
    else:
        W = n // K
        p = VecCollidePark(W, K, device=dev, contact_slots=8, kernel="cuda")
        p.load([[dict(s, free=True, massive=True) for s in sh[w * K:(w + 1) * K]] for w in range(W)])
        a = VS.CollideAdapter(p)
    return p, a, VS.VecServer(a, pilots, kernel=server_kernel)


def inputs(server, seed: int):
    """Per ship k of a world: (mask, ball id, direction [W, 3], speed fraction [W, 1])."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    W, d = server.W, server.dev
    out = []
    for k in range(K):
        out.append((torch.ones(W, dtype=torch.bool, device=d), torch.full((W,), k + 1, dtype=I64, device=d),
                    (torch.rand(W, 3, generator=g, dtype=F64) * 2 - 1).to(d),
                    torch.rand(W, 1, generator=g, dtype=F64).to(d)))
    return out


def direct_inputs(park_kind: str, p, ins):
    """The same commands as row tensors for one call each (every ship at once)."""
    dirs = torch.stack([x[2] for x in ins], dim=1).reshape(-1, 3)     # world-major: row = w K + k
    sf = torch.stack([x[3][:, 0] for x in ins], dim=1).reshape(-1)
    if park_kind == "collide":
        return torch.ones(p.W, p.N, dtype=torch.bool, device=dirs.device), dirs.reshape(p.W, p.N, 3), \
            sf.reshape(p.W, p.N)
    return torch.ones(dirs.shape[0], dtype=torch.bool, device=dirs.device), dirs, sf


def run_mode(park_kind: str, mode: str, n: int, seconds: float, dev) -> dict:
    p, a, server = build(park_kind, n, dev, "cuda" if mode.startswith("server_k") else "torch")
    mode = {"server_k": "server", "server_k_graph": "server_graph"}.get(mode, mode)
    ins = inputs(server, 11)
    if mode == "park":
        def tick():
            server.a.evolve()
    elif mode == "direct":
        m, dirs, sf = direct_inputs(park_kind, p, ins)

        def tick():
            p.goto_direction(m, dirs)
            p.set_speed_fraction(m, sf)
            server.a.evolve()
    elif mode == "server":
        def tick():
            for mask, ball, d, s in ins:
                server.push("GotoDirection", mask, ball, f=d)
                server.push("SetSpeedFraction", mask, ball, f=s)
            server.pre_tick(snapshot=False)
            server.evolve()
            server.post_tick()
    else:
        sched = []
        for k in range(K):
            sched += ["GotoDirection", "SetSpeedFraction"]
        g = VS.ServerGraph(server, sched)
        for j, (mask, ball, d, s) in enumerate(ins):
            for slot, f in ((g.inputs[2 * j], d), (g.inputs[2 * j + 1], s)):
                slot["mask"].copy_(mask)
                slot["ball"].copy_(ball)
                slot["f"][:, :f.shape[1]].copy_(f)

        def tick():
            g.tick()
    cuda = torch.device(dev).type == "cuda"
    sync = (lambda: torch.cuda.synchronize(dev)) if cuda else (lambda: None)
    for _ in range(3):
        tick()
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    ticks, t0 = 0, time.perf_counter()
    while True:
        for _ in range(5):
            tick()
        ticks += 5
        sync()
        el = time.perf_counter() - t0
        if el >= seconds:
            break
    return {"park": park_kind, "mode": mode, "ships": n, "worlds": server.W, "ticks": ticks, "seconds": el,
            "ms_per_tick": 1e3 * el / ticks, "world_ticks_per_s": server.W * ticks / el,
            "ship_ticks_per_s": n * ticks / el, "mships_per_s": n * ticks / el / 1e6,
            "peak_mib": torch.cuda.max_memory_allocated(dev) / 2 ** 20 if cuda else None}


def card() -> str:
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip().replace("\n", "; ")
    return f"{q} | apps: {a}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--widths", default="1024,8192,65536,262144")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--parks", default="modes,orient,collide")
    ap.add_argument("--modes", default="park,direct,server,server_graph")
    args = ap.parse_args()
    dev = torch.device("cuda")
    rows = []
    for park_kind in args.parks.split(","):
        for n in [int(x) for x in args.widths.split(",")]:
            for mode in args.modes.split(","):
                if mode.startswith("server_k") and park_kind != "modes":
                    continue
                before = card()
                row = run_mode(park_kind, mode, n, args.seconds, dev)
                row["mode"] = mode
                row.update(card_before=before, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                rows.append(row)
                print(f"{park_kind:8s} {mode:13s} {row['worlds']:7d} worlds {n:7d} ships {row['ms_per_tick']:8.3f} "
                      f"ms/tick {row['world_ticks_per_s'] / 1e6:8.2f} M world-ticks/s {row['mships_per_s']:8.2f} M "
                      f"ship-ticks/s peak {row['peak_mib']:7.1f} MiB", flush=True)
                torch.cuda.empty_cache()
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

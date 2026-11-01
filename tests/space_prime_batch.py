# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""the batched layer (exact_ballpark.vec_space) against the scalar reference on crowded worlds
(tests/space_prime_cases.py's worlds: up to 64 balls a world, proximity maps and sensor lists past 13, 29 and 59), with
tests/space_batch.py's runners and two additions: each map's bucket count and next resize compared beside its order
(the proximity map's per world, each ball's sensor list), and the scalar side run first and kept (`expected`), so a
GPU item runs only the batched side. The volumes count what the comparison saw: the longest list and map, the bucket
counts reached, the world-steps compared.
"""

from __future__ import annotations

import copy
from collections import Counter

import torch

import space_batch as SB
from exact_ballpark import vec_space as VS


def scalar_buckets(sw: SB.ScalarWorld) -> dict:
    p = sw.park
    return {"proximity": (p.proximity_balls.bucket_count, p.proximity_balls.next_resize),
            "sensors": {b.id: (b.sensor_balls.bucket_count, b.sensor_balls.next_resize) for b in p.balls.values()}}


def batched_buckets(bw: SB.BatchedWorlds, wi: int) -> dict:
    sp = bw.space
    ids = sp.slot_ids(wi)
    return {"proximity": sp.proximity_buckets(wi),
            "sensors": {bid: sp.sensor_buckets(wi, j) for j, bid in ids.items() if bool(sp.st["in_park"][wi, j])}}


class _HostView:
    def __init__(self, tensors: dict):
        self._t = tensors

    def tensors(self) -> dict:
        return self._t


def host_copy(bw: SB.BatchedWorlds) -> SB.BatchedWorlds:
    """The batched worlds' layer and park read back once (a transfer a tensor), for the readers: on a CUDA device
    tests/space_batch.py's readers would synchronise world by world and ball by ball."""
    if bw.dev.type == "cpu":
        return bw
    sp = copy.copy(bw.space)
    sp.st = {k: (v.cpu() if v is not None else None) for k, v in bw.space.st.items()}
    sp.rows = bw.space.rows.cpu()
    sp.view = _HostView({k: v.cpu() for k, v in bw.view.tensors().items()})
    snap = copy.copy(bw)
    snap.space, snap.view = sp, sp.view
    return snap


def expected(cs: list, steps: int | None = None) -> list:
    """The reference on every world, step by step: [step][world] = (state, buckets, events, answers)."""
    T = len(cs[0]["steps"]) if steps is None else int(steps)
    scal = [SB.ScalarWorld(w) for w in cs]
    for sw in scal:
        SB.scalar_events(sw)
    out = []
    for t in range(T):
        for wi, w in enumerate(cs):
            for rec in w["steps"][t]:
                scal[wi].apply(rec)
        for sw in scal:
            sw.evolve()
        row = []
        for sw in scal:
            row.append((SB.scalar_state(sw), scalar_buckets(sw), SB.scalar_events(sw), sw.answers))
            sw.answers = []
        out.append(row)
    return out


EVENTS = 1024                       # a tick's events a world: crowds post more than the layer's default 256


def compare(cs: list, device: str, park: str = "orient", want: list | None = None, lib: dict | None = None,
            umap_cap: int | None = None, hits: bool = True) -> dict:
    """The batched layer on cs against the reference (`want`, from `expected`, computed here when None): the
    differences (step, world, what, want, got), the volumes, the flagged worlds (left out of the comparison)."""
    want = expected(cs) if want is None else want
    T = len(want)
    bw = SB.BatchedWorlds(cs, device, park, hits=hits, lib=lib, umap_cap=umap_cap, events=EVENTS)
    diffs, vol = [], Counter()
    lists, maps, buckets = Counter(), Counter(), Counter()
    for t in range(T):
        bw.space.clear_events()
        bw.step(t)
        hb = host_copy(bw)
        flags = hb.space.st["flags"].tolist()
        for wi in range(len(cs)):
            if flags[wi]:
                continue
            state, bk, ev, ans = want[t][wi]
            got = SB.batched_state(hb, wi)
            if set(state["balls"]) != set(got["balls"]):
                diffs.append((t, wi, "balls", sorted(set(state["balls"]) ^ set(got["balls"]))))
            for bid, fields in state["balls"].items():
                if bid not in got["balls"]:
                    continue
                for key, val in fields.items():
                    vol["ball_fields"] += 1
                    if got["balls"][bid][key] != val:
                        diffs.append((t, wi, bid, key, val, got["balls"][bid][key]))
                lists[len(fields["sensor"][-1])] += 1
            for key in ("dicts", "interactives", "keep_alives", "bubble_id", "tops", "proximity"):
                vol["world_fields"] += 1
                if state["world"][key] != got["world"][key]:
                    diffs.append((t, wi, key, state["world"][key], got["world"][key]))
            maps[len(state["world"]["proximity"])] += 1
            gb = batched_buckets(hb, wi)
            vol["bucket_states"] += 1 + len(bk["sensors"])
            if gb["proximity"] != bk["proximity"]:
                diffs.append((t, wi, "proximity buckets", bk["proximity"], gb["proximity"]))
            for bid, b in bk["sensors"].items():
                if gb["sensors"].get(bid) != b:
                    diffs.append((t, wi, bid, "sensor buckets", b, gb["sensors"].get(bid)))
            buckets[bk["proximity"][0]] += 1
            for b in bk["sensors"].values():
                buckets[b[0]] += 1
            evg = hb.space.events(wi)
            vol["events"] += len(ev)
            if ev != evg:
                diffs.append((t, wi, "events", ev, evg))
            vol["answers"] += len(ans)
            if ans != bw.answers[wi]:
                diffs.append((t, wi, "answers", ans, bw.answers[wi]))
            bw.answers[wi] = []
            vol["world_steps"] += 1
    hb = host_copy(bw)
    flagged = {wi: [n for n, bit in VS.FLAG.items() if f & bit]
               for wi, f in enumerate(hb.space.st["flags"].tolist()) if f}
    return {"diffs": diffs, "volumes": dict(vol), "flagged": flagged, "hits": hb.space.hit_counts(),
            "longest_list": max(lists) if lists else 0, "longest_map": max(maps) if maps else 0,
            "lists_over": {k: sum(v for n, v in lists.items() if n > k) for k in (13, 29, 59)},
            "maps_over": {k: sum(v for n, v in maps.items() if n > k) for k in (13, 29, 59)},
            "bucket_counts": dict(sorted(buckets.items())), "steps": T, "worlds": len(cs)}


def same_state(a: SB.BatchedWorlds, b: SB.BatchedWorlds) -> list:
    """Every tensor of two layers and their parks equal (lists and maps over their lengths when the capacities
    differ): the names that differ."""
    bad = []
    sa, sb = a.space.st, b.space.st
    for k, v in sa.items():
        w = sb.get(k)
        if v is None or w is None:
            if (v is None) != (w is None):
                bad.append(k)
            continue
        if k in ("slist", "pmap") and v.shape != w.shape:
            S = min(v.shape[-1], w.shape[-1])
            n = sa["slen"] if k == "slist" else sa["pn"]
            if not torch.equal(n, (sb["slen"] if k == "slist" else sb["pn"])) or int(n.max()) > S:
                bad.append(k)
                continue
            idx = torch.arange(S, device=v.device)
            live = idx < n.unsqueeze(-1)
            if not torch.equal(torch.where(live, v[..., :S], 0), torch.where(live, w[..., :S].to(v.device), 0)):
                bad.append(k)
            continue
        if not torch.equal(v, w.to(v.device)):
            bad.append(k)
    for k, v in a.view.tensors().items():
        w = b.view.tensors()[k].to(v.device)
        eq = torch.equal(v.view(torch.int64), w.view(torch.int64)) if v.dtype == torch.float64 else torch.equal(v, w)
        if not eq:
            bad.append("park." + k)
    return bad

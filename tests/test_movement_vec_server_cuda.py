# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's server layer for VecModesPark as one CUDA kernel (exact_ballpark.vec_server_cuda) against the torch path
(exact_ballpark.vec_server, kernel="torch", itself held to the scalar server layer by tests/test_movement_vec_server.py),
bit for bit, every tick: the park's state of every ball, every field of the tick's Batch (the queued actions' rows,
duplicates and CloakBall's bubbles, the routes, the uncloaks' notices, the bubble update's lists, the stamps, the send
flags) and the update batches decoded; and directly against the scalar server layer. On the CPU: the generated header,
the source scan, the build and its PTX check, the refusals. On the GPU (skipped without one): the generated command
streams of tests/server_cases.py (every flight mode, every setter, cloaks, warps, turning on and off; development seed 1
and held-out seed 2), a CUDA graph against eager kernel ticks, 10,000 worlds, block sizes, and planted faults (each
kernel fault built into its own library, each caught).
"""

from __future__ import annotations

import logging
import re

import pytest
import torch

import server_cases as S
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_cuda as V

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
logging.getLogger("exact_ballpark.server").setLevel(logging.CRITICAL)


# ==================================================================================================== on the CPU

def test_the_header_holds_the_codes_and_the_literals():
    """Every kind's code, the routes and the normal cloak from exact_ballpark.vec_server; every literal found in the torch
    path's source with its spelling; a perturbed value changes exactly one line."""
    h = V.tables_header()
    for name, code in VS.CODE.items():
        assert f"#define K_{V._snake(name)} {code}\n" in h
    for name, val in (("SV_NONE", VS.NONE), ("SV_BUBBLE", VS.BUBBLE), ("SV_CHARACTER", VS.CHARACTER),
                      ("SV_NORMAL_CLOAK", VS.DSTNORMALCLOAK), ("SV_WARPIN_BACK", 5)):
        assert f"#define {name} {val}\n" in h
    for name, (val, _, _) in V.LITERALS.items():
        assert f"#define {name} 0x{V._dbits(val):016x}ULL\n" in h
    assert all(V.literals_in_source().values()), V.literals_in_source()
    p = V.tables_header({"SV_SNAP_UPRIGHT": 0.98})
    assert sum(a != b for a, b in zip(h.splitlines(), p.splitlines())) == 1


def test_the_sources_call_no_cuda_math():
    """The source scan finds nothing in the three sources the build compiles, and flags a planted call."""
    from exact_ballpark import vec_modes_cuda as M
    from exact_ballpark import vec_orient_cuda as K
    for src in (V.SOURCE, M.SOURCE, K.SOURCE):
        assert K.scan_source(V.active_text(src.read_text())) == [], src.name
    text = V.SOURCE.read_text().replace("return x <= 0.0 ? 0.0 : g_exp(", "return x <= 0.0 ? 0.0 : exp(")
    assert any(name == "exp" for name, _ in K.scan_source(text))


def test_the_build_pins_every_rounding(tmp_path):
    """The library builds with the pinned flags and its PTX passes the check (every add, sub, mul, div, sqrt and rcp
    rounded to nearest; no mad, flush or approximation), with the kernel's resources recorded."""
    m = V.build(tmp_path, variant="sm90")
    assert m["ptx_check"]["ok"] and not m["source_scan"]
    assert "-fmad=false" in m["flags"] and "-prec-div=true" in m["flags"] and "-ftz=false" in m["flags"]
    assert any("server_kernel" in ln for ln in m["ptxas"])
    assert not any(re.search(r"[1-9]\d* bytes spill", ln) for ln in m["ptxas"])


def test_refusals():
    """The kernel runs a VecModesPark's server layer on a CUDA device only."""
    cs = S.cases(1, 2, "orient", ticks=1)
    with pytest.raises(ValueError, match="VecModesPark on a CUDA device"):
        S.build(cs, "orient", "cpu", server_kernel="cuda")
    cs = S.cases(1, 2, "modes", ticks=1)
    with pytest.raises(ValueError, match="VecModesPark on a CUDA device"):
        S.build(cs, "modes", "cpu", server_kernel="cuda")
    with pytest.raises(ValueError, match="'torch' or 'cuda'"):
        S.build(cs, "modes", "cpu", server_kernel="numba")


# ==================================================================================================== on the GPU

def _batch_view(server) -> dict:
    """Every field of the last Batch, on the host, the slots' data per slot."""
    b = server.batch
    out = {"stamp": b.stamp.tolist(), "clock": b.clock.tolist(), "route": b.route.tolist(),
           "has": b.has_character.tolist(), "gets": b.gets_bubble.tolist(),
           "fresh": b.additions[0].tolist(), "listed": b.additions[1].tolist(),
           "notices": [(j, rec.tolist()) for j, rec, _ in b.notices]}
    for j, s in enumerate(b.slots):
        okw = s.ok.tolist()
        rows = [r if o else None for r, o in zip(s.row.tolist(), okw)]
        trows = [r if o else None for r, o in zip(s.trow.tolist(), okw)]
        locs = [r if o else None for r, o in zip(s.loc.tolist(), okw)]
        out[f"slot{j}"] = (s.kind, okw, rows, trows, locs)
    return out


def _kernel_against_torch(cs: list, do: bool, snapshot: bool = True, block: int | None = None,
                          ticks: int | None = None) -> list:
    """Both servers on their own VecModesPark, the same pushes every tick: [(tick, what differs)]."""
    T = len(cs[0]["ticks"]) if ticks is None else ticks
    ts, index = S.build(cs, "modes", "cuda", do=do)
    ks, _ = S.build(cs, "modes", "cuda", do=do, server_kernel="cuda")
    if block is not None:
        V.BLOCK_SAVED, V.BLOCK = V.BLOCK, block
    bad = []
    try:
        for t in range(T):
            for kind, *data in S.pushes(cs, t, "cuda"):
                ts.push(kind, *data)
                ks.push(kind, *data)
            for srv in (ts, ks):
                srv.pre_tick(snapshot=snapshot)
                srv.evolve()
                srv.post_tick()
            a, b = S.batched_rows(ts, cs, index, "modes", do), S.batched_rows(ks, cs, index, "modes", do)
            rows = [k for k in a if not S.same_rows(a[k], b[k])]
            if rows:
                bad.append((t, "rows", rows[:3]))
            for name in ("sensor", "refiled", "cloaked"):
                if not torch.equal(getattr(ts.a.park, name), getattr(ks.a.park, name)):
                    bad.append((t, name))
            va, vb = _batch_view(ts), _batch_view(ks)
            diff = [k for k in va if va[k] != vb[k]]
            if diff:
                bad.append((t, "batch", diff[:5]))
            for k in ("stamp_for_system", "settled", "_fresh", "_listed", "_bubble_time"):
                if not torch.equal(getattr(ts, k), getattr(ks, k)):
                    bad.append((t, k))
            if snapshot and S.canon([S._listify(x) for x in VS.decode(ts)]) != \
                    S.canon([S._listify(x) for x in VS.decode(ks)]):
                bad.append((t, "decode"))
    finally:
        if block is not None:
            V.BLOCK = V.BLOCK_SAVED
    return bad


@CUDA
@pytest.mark.parametrize("do", [False, True])
@pytest.mark.parametrize("seed", [1, 2])
def test_the_kernel_equals_the_torch_path(do, seed):
    """Every tick bit for bit: the park, the Batch, the server's state, the decoded updates (snapshots on)."""
    bad = _kernel_against_torch(S.cases(seed, 40, "modes", do=do, ticks=14), do)
    assert bad == [], bad[:4]


@CUDA
def test_the_kernel_in_one_launch_equals_the_torch_path():
    """Without snapshots the pre-tick is one launch: the same states and Batch."""
    bad = _kernel_against_torch(S.cases(1, 40, "modes", ticks=14), False, snapshot=False)
    assert bad == [], bad[:4]


@CUDA
@pytest.mark.parametrize("do", [False, True])
def test_the_kernel_equals_the_scalar_server_layer(do):
    """Directly against exact_ballpark.server (no torch path between): states and update batches."""
    res = S.run(S.cases(2, 40, "modes", do=do, ticks=14), "modes", "cuda", do=do, server_kernel="cuda")
    assert res["rows"] == [] and res["sends"] == [] and res["compared"] > 0, (res["rows"][:2], res["sends"][:1])


@CUDA
@pytest.mark.parametrize("block", [32, 384])
def test_block_sizes(block):
    bad = _kernel_against_torch(S.cases(3, 40, "modes", ticks=8), False, block=block)
    assert bad == [], bad[:4]


@CUDA
def test_a_block_past_the_kernels_limit_is_refused():
    """A block larger than the kernel's registers allow (maxThreadsPerBlock) is refused before the launch."""
    cs = S.cases(1, 4, "modes", ticks=1)
    server, _ = S.build(cs, "modes", "cuda", server_kernel="cuda")
    limit = server._lib["info"]["max_threads_per_block"]
    assert 128 <= limit < 1024
    old, V.BLOCK = V.BLOCK, 1024
    try:
        with pytest.raises(ValueError, match="threads a block"):
            server.tick()
    finally:
        V.BLOCK = old


@CUDA
def test_ten_thousand_worlds():
    """40 worlds tiled to 10,000 in one park: kernel and torch path equal every tick (no snapshots)."""
    cs = S.cases(1, 40, "modes", ticks=6)
    tiled = [dict(cs[k % len(cs)], id=k) for k in range(10000)]
    bad = _kernel_against_torch(tiled, False, snapshot=False)
    assert bad == [], bad[:4]


@CUDA
@pytest.mark.parametrize("do", [False, True])
def test_a_graph_of_kernel_ticks_equals_eager_kernel_ticks(do):
    """The fixed schedule of tests/test_movement_vec_server.py ticked eagerly on the kernel and replayed from a
    ServerGraph on the kernel: the same states, routes and flags every tick."""
    import test_movement_vec_server as T
    cs, sched, inputs = T._fixed_schedule_case(1, 64, "modes", do, 10)
    eager, index = S.build(cs, "modes", "cuda", do=do, server_kernel="cuda")
    graphed, _ = S.build(cs, "modes", "cuda", do=do, server_kernel="cuda")
    g = VS.ServerGraph(graphed, sched)
    for t, per in enumerate(inputs):
        for kind, *data in per:
            eager.push(kind, *(x.cuda() for x in data))
        eager.pre_tick(snapshot=False)
        eager.evolve()
        eager.post_tick()
        for slot_in, (kind, *data) in zip(g.inputs, per):
            for key, x in zip(("mask", "ball", "target", "f", "n", "loc"), data):
                slot_in[key].copy_(x.cuda())
        out = g.tick()
        torch.cuda.synchronize()
        a, b = S.batched_rows(eager, cs, index, "modes", do), S.batched_rows(graphed, cs, index, "modes", do)
        assert all(S.same_rows(a[k], b[k]) for k in a), t
        assert torch.equal(eager.batch.route, out.route), t
        assert torch.equal(eager.batch.has_character, out.has_character), t
        assert torch.equal(eager.batch.gets_bubble, out.gets_bubble), t


@CUDA
@pytest.mark.parametrize("fault", sorted(V.FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    """Each planted fault, built into its own library (checks recorded, not enforced), makes the kernel differ from
    the torch path on the development streams."""
    lib = V.load_library(variant="sm90", fault=fault, checks=False)
    real = V.load_library

    def faulty(*args, **kw):
        return lib
    monkeypatch.setattr(V, "load_library", faulty)
    bad = _kernel_against_torch(S.cases(1, 60, "modes", ticks=14), False)
    monkeypatch.setattr(V, "load_library", real)
    assert bad, f"the planted fault {fault} went unnoticed"

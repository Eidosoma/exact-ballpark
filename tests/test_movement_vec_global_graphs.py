# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The space server's kernel with global balls at AddBall as a CUDA graph: CudaSpaceServer over
VecModesPark with each row's AddBall flag held by the partition layer's view (tests/global_edge_cases.py's worlds,
`build_space`), its tick captured by exact_ballpark.vec_server_space_cuda.SpaceServerGraph and replayed, against eager
kernel ticks with the same pushes (tests/test_movement_vec_server_space_cuda.py's fixed schedule): every tensor of the
park, the partition and the server and the batch equal after every tick, bit for bit; the flags unchanged (the kernel
refuses make_ball_global and make_ball_local). Needs a GPU."""

import pytest
import torch

import global_edge_cases as E
import test_movement_vec_server_space_cuda as K
from exact_ballpark import vec_server as VS

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")


@CUDA
def test_a_graph_of_kernel_ticks_with_global_balls_equals_eager_kernel_ticks():
    from exact_ballpark import vec_server_space_cuda as Q
    cs = E.space_cases(4, 64, "modes", 1)
    eager, _ = E.build_space(cs, "modes", "cuda", server_cls=Q.CudaSpaceServer)
    graphed, _ = E.build_space(cs, "modes", "cuda", server_cls=Q.CudaSpaceServer)
    flags = eager.space.view.global_flags().clone()
    assert bool(flags.any()) and torch.equal(flags, graphed.space.view.global_flags())
    g = Q.SpaceServerGraph(graphed, K.SCHEDULE)
    W, dev = eager.W, eager.dev
    for t in range(10):
        args = K._schedule_inputs(cs, t, W, dev)
        for k, a in zip(K.SCHEDULE, args):
            eager.push(k, a["mask"], a["ball"], a["target"], a["f"], a["n"], a["loc"])
        eager.pre_tick(snapshot=False)
        eager.evolve()
        eager.post_tick()
        for inp, a in zip(g.inputs, args):
            for k in inp:
                inp[k].copy_(a[k])
        b = g.tick()
        diff = [n for n in ("pos", "vel", "mode", "bubble", "cloaked", "massive", "sensor", "follow_id")
                if not K._same(getattr(eager.a.park, n), getattr(graphed.a.park, n))]
        diff += [n for n, v in eager.space.st.items() if torch.is_tensor(v) and not K._same(v, graphed.space.st[n])]
        diff += [n for n in ("stamp_for_system", "_ad_time", "cleanup_total")
                 if not K._same(getattr(eager, n), getattr(graphed, n))]
        e = eager.batch
        for n in ("stamp", "clock", "route", "rbubble", "has_character", "gets_bubble", "narrow"):
            if not K._same(getattr(e, n), getattr(b, n)):
                diff.append(f"batch.{n}")
        for n in ("kind", "adds", "addn", "dels", "deln"):
            if not torch.equal(e.ad[n], b.ad[n]):
                diff.append(f"ad.{n}")
        for j, (sa, sb) in enumerate(zip(e.slots, b.slots)):
            if not (torch.equal(sa.ok, sb.ok) and torch.equal(sa.row, sb.row) and torch.equal(sa.ball, sb.ball)):
                diff.append(f"slot{j}")
        assert not diff, (t, diff[:6])
    assert int(eager.batch.route.ne(VS.NONE).sum()) > 0
    assert torch.equal(graphed.space.view.global_flags(), flags) and torch.equal(eager.space.view.global_flags(), flags)

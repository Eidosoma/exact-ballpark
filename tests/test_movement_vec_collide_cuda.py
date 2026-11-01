# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's gradient collision step as one CUDA kernel (exact_ballpark.vec_collide_cuda, selected by
VecCollidePark(kernel="cuda")) held to the torch module (exact_ballpark.vec_collide, kernel="torch", the exactness
reference) bit for bit (doubles by their bits, NaNs equal):

- the generated worlds (tests/collide_cases.py: approaches, overlaps and the recursion, light balls, fixed
  spheres, several contacts of one ball in a tick, near misses, box faces, junk, racers; seeds 1 to 3), every park
  tensor after every tick, the kernel on the card against the torch path on the CPU (pruned and without host syncs,
  rows, impact slots) and on the card, and against the scalar reference (exact_ballpark.space);
- a fixed contact_slots exceeded is the torch path's overflow event, with the same truncated results; impact_slots,
  the torch path's cap on the impact branch, does not bind the kernel; inactive worlds keep their state and a world's
  bits do not depend on the batch around it;
- the kernel captured in a CUDA graph against eager: a park's ticks;
- the kernel's exp and its native fused multiply-add against vec_exp's (the emulation) and the C library's;
- the build's pins on the CPU: the header generated from vec_exp and vec_collide, the source calling no math library,
  the PTX check (every float64 operation rounded to nearest, the fused multiply-adds exp's own);
- planted faults the comparisons must catch: a contracted multiply-add, CUDA's own exp, a wrong table entry, the
  contact order reversed.

The card's cases are skipped where no GPU is visible; the build's need nvcc. Engineering checks only."""

from __future__ import annotations

import math
import os
import re
import shutil
from collections import defaultdict

import pytest
import torch

import collide_cases as cc
from exact_ballpark import vec_collide_cuda as K
from exact_ballpark import vec_exp
from exact_ballpark.vec_collide import VecCollidePark
from test_movement_vec_exp import exp_args, fma_triples, libm_exp, u64

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
NVCC = pytest.mark.skipif(not (K.NVCC.exists() or shutil.which("nvcc")), reason="no nvcc")
FIELDS = ("pos", "vel", "old_pos", "old_vel", "goto", "last_g", "last_c", "last_collision", "mode", "speed_fraction",
          "level", "key", "mod", "hits", "hits_n", "imp_slot", "imp_s", "imp_closing", "imp_n", "overflow")
FAULT_WORD = 255          # the table word of the planted fault: 2^(127/128)'s scale, reached by 985 of seed 1's exps


@pytest.fixture(autouse=True)
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int64) if t.dtype == torch.float64 else t


# ---- the worlds, every park tensor after every tick ----

def run_park(cs: list, device: str, rows=None, **kw) -> dict:
    """The scenarios batched as tests/collide_cases.batched groups them (one park per friction, tick and, with
    rows="free", free-ball count), every park tensor after every tick: {(group, step): {field: tensor}}; no check()."""
    groups = defaultdict(list)
    nfree = lambda c: sum(1 for rec in c["records"] if rec[0] == "B" and rec[2]["free"])      # noqa: E731
    for c in cs:
        groups[(c["friction"], c["tick"], nfree(c) if rows == "free" else None)].append(c)
    out, parks = {}, []
    for gi, ((friction, tick, free), g) in enumerate(groups.items()):
        n = max(sum(1 for rec in c["records"] if rec[0] == "B") for c in g)
        extra = dict(kw)
        if rows is not None:
            extra["rows"] = max(1, free) if free is not None else rows
        park = VecCollidePark(len(g), n, friction=friction, tick_interval_ms=tick, device=device, **extra)
        worlds = []
        for c in g:
            balls = {rec[2]["id"]: cc._ball_dict(rec[2]) for rec in c["records"] if rec[0] == "B"}
            for rec in c["records"]:
                if rec[0] == "K" and rec[1] == 0 and rec[2] == cc.CLOAK_OP:
                    balls[rec[3]]["cloaked"] = rec[5]
            worlds.append(list(balls.values()))
        park.load(worlds)
        slot = [{int(i): j for j, i in enumerate(ids) if i} for ids in park.id.cpu().tolist()]
        by_step = defaultdict(list)
        for w, c in enumerate(g):
            for rec in c["records"]:
                if rec[0] == "K" and rec[1] > 0:
                    by_step[rec[1]].append((w, rec))
        for s in range(1, max(c["steps"] for c in g) + 1):
            cc._commands(park, slot, by_step.get(s, []))
            park.evolve()
            out[(gi, s)] = {f: getattr(park, f).detach().cpu().clone() for f in FIELDS}
        parks.append(park)
    out["parks"] = parks
    return out


def differences(a: dict, b: dict) -> list:
    """(group, step, field, values differing) for every park tensor that differs."""
    keys = [k for k in a if k != "parks"]
    if set(keys) != {k for k in b if k != "parks"}:
        return [("ticks", len(keys), len(b) - 1)]
    bad = []
    for k in keys:
        for f in FIELDS:
            x, y = a[k][f], b[k][f]
            if x.shape != y.shape or not torch.equal(bits(x), bits(y)):
                bad.append((*k, f, int((bits(x) != bits(y)).sum()) if x.shape == y.shape else -1))
    return bad


_CPU: dict = {}


def torch_cpu(seed: int, n: int = 240, **kw) -> dict:
    """The torch path on the CPU (the exactness reference), cached."""
    key = (seed, n, tuple(sorted(kw.items())))
    if key not in _CPU:
        _CPU[key] = run_park(cc.cases(seed, n=n), "cpu", **kw)
    return _CPU[key]


CONFIGS = [(1, {}), (1, {"contact_slots": 7}), (1, {"rows": "free"}), (1, {"rows": "free", "contact_slots": 7}),
           (2, {}), (2, {"rows": "free", "contact_slots": 7}), (3, {"contact_slots": 7, "impact_slots": 4096})]


@CUDA
@pytest.mark.parametrize("seed,kw", CONFIGS, ids=lambda x: str(x).replace(" ", ""))
def test_the_kernel_equals_the_torch_module_on_the_cpu_bit_for_bit(seed, kw):
    """the worlds on the kernel against the torch path on the CPU, every park tensor after every tick (the state, the
    filing, the collision lists, the impact record, the overflow flag), in each host mode and layout."""
    want = torch_cpu(seed, **kw)
    got = run_park(cc.cases(seed, n=240), "cuda", kernel="cuda", **kw)
    bad = differences(want, got)
    assert not bad, f"{len(bad)} park tensors differ, first {bad[:5]}"
    hits = sum(int(v["hits_n"].sum()) for k, v in want.items() if k != "parks")
    assert hits > 1000                                                  # contacts in every configuration


@CUDA
@pytest.mark.parametrize("seed", [1, 2])
def test_the_iterative_methods_worlds_under_the_gradient_on_the_kernel(seed):
    """the worlds generated for the iterative method (their own draw) run with the gradient method: the kernel as the
    torch path on the CPU, every park tensor after every tick."""
    cs = cc.cases(seed, n=40, iterative=True)
    bad = differences(run_park(cs, "cpu", contact_slots=7), run_park(cs, "cuda", kernel="cuda", contact_slots=7))
    assert not bad, f"{len(bad)} park tensors differ, first {bad[:5]}"


@CUDA
def test_the_kernel_equals_the_torch_module_on_the_card_bit_for_bit():
    cs = cc.cases(1, n=240)
    for kw in ({}, {"contact_slots": 7}):
        bad = differences(run_park(cs, "cuda", **kw), run_park(cs, "cuda", kernel="cuda", **kw))
        assert not bad, f"{kw}: {len(bad)} park tensors differ, first {bad[:5]}"


@CUDA
@pytest.mark.parametrize("seed", [1, 2])
def test_the_kernel_matches_the_scalar_reference(seed):
    """The kernel's balls against exact_ballpark.space.SpaceBallpark's (which compiled Destiny's own code equals), every
    tick: positions, velocities, mLastG, mLastC, mLastCollision, the filings and the collision lists."""
    cs = cc.cases(seed, n=240)
    want, und = cc.reference_rows(cs)
    got = cc.batched(cs, "cuda", kernel="cuda")
    bad = cc.differences(want, {k: v for k, v in got.items() if k[0] not in und})
    assert not bad, f"{len(bad)} values differ, first {bad[:5]}"


@CUDA
def test_an_exceeded_contact_slots_is_the_same_event_with_the_same_bits():
    """contact_slots=1 on worlds with several contacts a tick: the kernel applies each ball's first contact in order, as
    the torch path does, and check() raises the same overflow event."""
    kw = {"contact_slots": 1}
    want = torch_cpu(1, **kw)
    got = run_park(cc.cases(1, n=240), "cuda", kernel="cuda", **kw)
    assert not differences(want, got)
    assert any(bool(p.overflow.any()) for p in want["parks"])
    for p in got["parks"]:
        if bool(p.overflow.any()):
            with pytest.raises(RuntimeError, match="contact_slots"):
                p.check()


@CUDA
def test_impact_slots_do_not_bind_the_kernel():
    """The torch path's impact_slots=1 is exceeded (an overflow event: its results past the cap are not Destiny's); the
    kernel computes every impact, so with the same settings it equals the torch path without the cap and raises
    nothing."""
    cs = cc.cases(1, n=240)
    capped = run_park(cs, "cpu", contact_slots=7, impact_slots=1)
    with pytest.raises(RuntimeError, match="impact_slots"):
        for p in capped["parks"]:
            p.check()
    got = run_park(cs, "cuda", kernel="cuda", contact_slots=7, impact_slots=1)
    assert not differences(torch_cpu(1, contact_slots=7), got)
    for p in got["parks"]:
        p.check()


@CUDA
def test_inactive_worlds_and_the_batch_around_a_world():
    """evolve(active) on the kernel: an inactive world's state unchanged bit for bit, the active ones as a park of their
    own; and a world alone gives the bits it gives among others, in either order."""
    worlds = [[cc._ball_dict(r[2]) for r in c["records"] if r[0] == "B"] for c in cc.cases(1, n=24)]
    n = max(len(w) for w in worlds)

    def park(ws):
        p = VecCollidePark(len(ws), n, device="cuda", kernel="cuda", contact_slots=7)
        p.load(ws)
        p.goto_point(p.exists & p.free, torch.zeros(len(ws), n, 3, dtype=torch.float64, device="cuda"))
        return p
    mixed, alone = park(worlds), park(worlds[::2])
    active = (torch.arange(len(worlds)) % 2 == 0).cuda()
    names = FIELDS[:-1] + ("cap", "has_prop")
    before = {k: getattr(mixed, k).clone() for k in names}
    for _ in range(6):
        mixed.evolve(active)
        alone.evolve()
    for k in names:
        assert torch.equal(bits(getattr(mixed, k)[~active]), bits(before[k][~active])), k
        assert torch.equal(bits(getattr(mixed, k)[active]), bits(getattr(alone, k))), k
    cs = cc.cases(1, n=60)
    together = cc.batched(cs, "cuda", kernel="cuda")
    assert not cc.differences(together, cc.batched(list(reversed(cs)), "cuda", kernel="cuda"))
    each = {}
    for c in cs[:8]:
        each.update(cc.batched([c], "cuda", kernel="cuda"))
    assert not cc.differences({k: together[k] for k in each}, each)


@CUDA
def test_the_kernel_in_a_cuda_graph_equals_eager():
    """A park's ticks on the kernel recorded in one CUDA graph (graph_capture's capture and pins: every tensor the
    tick rebinds copied back into a fixed one) and replayed, against the same ticks run eagerly: every tensor after
    every tick, bit for bit; no host synchronisation in the recorded tick."""
    from graph_capture import Pins, capture
    cs = [c for c in cc.cases(1, n=240) if c["friction"] == 1.0e6 and c["tick"] == 1000]
    n = max(sum(1 for rec in c["records"] if rec[0] == "B") for c in cs)
    parks = []
    for _ in range(2):
        p = VecCollidePark(len(cs), n, device="cuda", kernel="cuda", contact_slots=7)
        p.load([list({rec[2]["id"]: cc._ball_dict(rec[2]) for rec in c["records"] if rec[0] == "B"}.values())
                for c in cs])
        slot = [{int(i): j for j, i in enumerate(ids) if i} for ids in p.id.cpu().tolist()]
        cc._commands(p, slot, [(w, rec) for w, c in enumerate(cs) for rec in c["records"]
                               if rec[0] == "K" and rec[1] == 1])          # each world's first commands, then coast
        parks.append(p)
    eager, graphed = parks
    pins = Pins([graphed])

    def body():
        graphed.evolve()
        pins.pin(True)
    eager.evolve()
    body()                                         # one eager tick: every lazy constant made
    torch.cuda.synchronize()
    g = capture(body, torch.cuda.graph_pool_handle())
    names = FIELDS + ("impact_overflow",)
    for k in range(12):
        eager.evolve()
        g.replay()
        torch.cuda.synchronize()
        bad = [f for f in names if not torch.equal(bits(getattr(eager, f)), bits(getattr(graphed, f)))]
        assert not bad, (k, bad)
    assert int(eager.hits_n.sum()) + int(eager.imp_n.sum()) > 0


# ---- exp and the fused multiply-add ----

@CUDA
@pytest.mark.parametrize("seed", [1, 2])
def test_the_kernels_exp_is_vec_exps_and_the_c_librarys_bit_for_bit(seed):
    """xbpc_exp (the transcription of vec_exp.exp, __fma_rn where vec_exp emulates its fused multiply-adds) on every
    path: the main path, |x| below 2^-54, the special case from 512 with subnormal and near-overflow results, 1024 and
    beyond, infinities, NaN, every bit pattern."""
    xs = exp_args(seed, 400_000)
    got = K.exp(torch.tensor(xs, dtype=torch.float64, device="cuda")).cpu()
    ref = vec_exp.exp(torch.tensor(xs, dtype=torch.float64))
    same = (bits(got) == bits(ref)) | (torch.isnan(got) & torch.isnan(ref))
    assert bool(same.all()), f"{int((~same).sum())} differ from vec_exp"
    bad = [x for x, g in zip(xs, got.tolist()) if not (u64(g) == u64(libm_exp(x)) or (g != g and x != x))]
    assert not bad, f"{len(bad)} differ from the C library, first {bad[:3]}"


@CUDA
def test_native_fma_is_vec_exps_emulated_fma_bit_for_bit():
    """__fma_rn, the IEEE fused multiply-add rounded once, is the operation vec_exp.fma emulates (Boldo and Melquiond's
    algorithm, exact without overflow or underflow): equal to the emulation on the card and on the CPU, and to
    math.fma, on exact ties, cancellations, a product's exact error, exp's own operand scales, random bit patterns and
    near ties."""
    tr = fma_triples(5, 400_000)
    cols = [torch.tensor([x[i] for x in tr], dtype=torch.float64) for i in range(3)]
    native = K.fma(*(c.cuda() for c in cols)).cpu()
    assert torch.equal(bits(native), bits(vec_exp.fma(*(c.cuda() for c in cols)).cpu()))
    assert torch.equal(bits(native), bits(vec_exp.fma(*cols)))
    assert all(u64(g) == u64(math.fma(*x)) for x, g in zip(tr, native.tolist()))


# ---- planted faults ----

@CUDA
@pytest.mark.parametrize("fault", ["contracted_fma", "libm_exp", "wrong_table_entry", "reversed_order"])
def test_planted_faults_are_caught(monkeypatch, fault):
    """Each fault, built into the kernel, changes what the comparison with the torch module sees on the development
    worlds: a contracted multiply-add (the integration's last sum fused), CUDA's own exp in place of the transcription,
    one bit of one generated table word, a ball's contacts in reverse order."""
    if fault == "wrong_table_entry":
        lib = K.load_library("sm90", perturb={"T_EXP": FAULT_WORD})
    else:
        lib = K.load_library("sm90", fault=fault, checks=fault == "reversed_order")
    monkeypatch.setattr(K, "load_library", lambda *a, **k: lib)
    bad = differences(torch_cpu(1), run_park(cc.cases(1, n=240), "cuda", kernel="cuda"))
    assert len(bad) > 100, bad[:5]


# ---- the build's pins, on the CPU ----

def test_the_header_holds_vec_exps_and_vec_collides_values():
    """The generated header's doubles are vec_exp's and vec_collide's (exact hexadecimal literals), its table is
    vec_exp's 256 words (glibc's, read from the library), its integers vec_collide's sort-key constants; vec_collide
    still writes the literals the header carries; a perturbed header flips exactly one bit of one word."""
    h = K.tables_header()
    vals = K.table_values()
    for name, (kind, v) in vals.items():
        if isinstance(v, tuple) and kind != "u64":
            body = re.search(rf"{name}\[{len(v)}\] = \{{(.*?)\}};", h).group(1).split(",")
            got = [float.fromhex(x) for x in body] if kind == "d" else [int(x.strip()[:-2]) for x in body]
            assert got == list(v), name
        elif kind == "d":
            m = re.search(rf"constexpr double {name} = (\S+);", h)
            assert m and float.fromhex(m.group(1)) == v, name
        elif kind == "i64":
            assert f"constexpr long long {name} = {int(v)}LL;" in h, name
    words = [int(w, 16) for w in re.search(r"T_EXP\[256\] = \{(.*?)\}", h).group(1).replace("ULL", "").split(",")]
    assert tuple(words) == vec_exp.TABLE
    assert vals["K_EXP_C3"] == ("d", vec_exp.C3) and vals["XBPC_CODE_BASE"][1] == 65
    assert all(K.literals_in_source().values())
    p = K.tables_header({"T_EXP": FAULT_WORD})
    pw = [int(w, 16) for w in re.search(r"T_EXP\[256\] = \{(.*?)\}", p).group(1).replace("ULL", "").split(",")]
    assert [i for i, (a, b) in enumerate(zip(words, pw)) if a != b] == [FAULT_WORD]
    assert bin(words[FAULT_WORD] ^ pw[FAULT_WORD]).count("1") == 1


def test_the_source_calls_no_math_library():
    """The kernel's source as compiled calls no function of CUDA's math library nor an approximate or directed
    intrinsic, and holds exactly vec_exp.exp's fused multiply-adds (plus the test kernel's one); the planted fault that
    calls CUDA's exp is what the scan finds."""
    text = K.SOURCE.read_text()
    assert K.scan_source(K.active_text(text, 0)) == []
    assert [n for n, _ in K.scan_source(K.active_text(text, K.FAULTS["libm_exp"]))] == ["exp"]
    body = K._strip_comments(K.active_text(text, 0))
    exp_fn = body[body.index("f64 xbpc_exp(f64 x)"):body.index("__device__ __forceinline__ f64 exp_of")]
    assert exp_fn.count("__fma_rn(") == K.EXP_FMAS
    assert body.count("__fma_rn(") == K.EXP_FMAS + 1


@NVCC
def test_the_build_pins_every_rounding():
    """The library as built: the PTX's float64 operations each rounded to nearest (no contraction stage can fuse
    them), the fused multiply-adds exactly the source's own, nothing flushed, approximate or transcendental; a build
    with a contracted multiply-add is refused by the PTX check, one calling CUDA's exp by the source scan."""
    m = K.build(variant="sm90")
    assert m["ptx_check"]["ok"] and m["source_scan"] == []
    assert m["ptx_check"]["fp_ops"]["fma.rn.f64"] == K.EXP_FMAS + 1
    assert set(m["ptx_check"]["fp_ops"]) <= set(K.PTX_ALLOWED)
    with pytest.raises(RuntimeError, match="unpinned"):
        K.build(variant="sm90", fault="contracted_fma")
    with pytest.raises(RuntimeError, match="math library"):
        K.build(variant="sm90", fault="libm_exp")
    assert K.ptx_check("mul.f64 %fd1, %fd2, %fd3;\n")["refused"] == {"mul.f64": 1}
    assert K.ptx_check("fma.rn.f64 %fd1, %fd2, %fd3, %fd4;\n", fmas=0)["ok"] is False


def test_the_switch():
    """kernel: 'torch' (the default) or 'cuda'; the kernel needs a CUDA device and the gradient method."""
    assert VecCollidePark(2, 3).kernel == "torch"
    with pytest.raises(ValueError, match="kernel"):
        VecCollidePark(2, 3, kernel="fused")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA device"):
            VecCollidePark(2, 3, kernel="cuda")
    else:
        with pytest.raises(ValueError, match="CUDA device"):
            VecCollidePark(2, 3, kernel="cuda", device="cpu")
        with pytest.raises(ValueError, match="iterative"):
            VecCollidePark(2, 3, kernel="cuda", device="cuda", iterative=True)
        p = VecCollidePark(2, 3, device="cuda")
        p.use_kernel("cuda")
        assert p.kernel == "cuda" and K.info()["registers"] > 0

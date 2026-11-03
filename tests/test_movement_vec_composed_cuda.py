# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The composed park's kernel path (exact_ballpark.vec_composed_cuda: VecComposedPark.evolve with
collisions on, thirteen launches a tick) against its torch path (VecComposedPark.evolve_torch, the
exactness reference), bit for bit, every tensor of the park at every tick (two parks in lockstep on the same commands:
every state tensor, the collision lists, the filings, the flags, the dispatch counts, and the DoCollision events and
the ReserveFormationSlot answers), on generated worlds (tests/park_cases.py's seeds 1 and 2, the worlds
of tests/destiny_combat_cases.py, tests/park_rest_cases.py's development seed 11 and held-out seed 3) and the
hand-built drop-out worlds; and, through the torch path's own tests, against
the scalar reference and Destiny. A tick captured in a CUDA graph equals eager ticks. The planted faults of the earlier
kernels (a contracted multiply-add, CUDA's own exp, the contact order reversed: the switches, in the composed
build) and this park's own (in this library and in the composed candidates) are each caught. On the CPU: the default
build of the kernel is the one it was (its header and defines), the sources call no math library, the build pins
every rounding.
"""

from __future__ import annotations

import shutil
from collections import Counter, defaultdict

import pytest
import torch

import destiny_combat_cases as dc
import park_cases as pc
import park_rest_cases as prc
from exact_ballpark import vec_collide_cuda as KD
from exact_ballpark import vec_composed as vp
from exact_ballpark import vec_composed_cuda as KP

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device visible")
NVCC = pytest.mark.skipif(not (KD.NVCC.exists() or shutil.which("nvcc")), reason="nvcc not found")
SCRATCH = {"act", "pos0", "vel0", "vy_after"}       # the prepass records: the kernels' scratch, not the state


# ---- two parks in lockstep: the torch path and the kernel path ----

def _park(cfg: dict, rows: list, device: str, cls=None, **kw):
    cls = cls or vp.VecComposedPark
    p = cls(len(rows), cfg["friction"], cfg["tick"], device, cfg["do"], cfg["new_orbit"], master=cfg["master"],
            disable_do_for_missiles=cfg["do_off"], **kw)
    p.load(rows)
    p.load_formations(pc.FORMATIONS)
    return p


def state_names(p) -> list:
    """Every tensor the park holds (its state: rows, lists, filings, flags, counts, the view's index), less the
    prepass records and the kernel path's own buffers."""
    return sorted(k for k, v in vars(p).items() if isinstance(v, torch.Tensor) and k not in SCRATCH
                  and not k.startswith("_"))


def _differ(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elements whose bits differ (two NaNs equal), as a bool tensor of a's shape."""
    if a.is_floating_point():
        ia = a.contiguous().view(torch.int64 if a.element_size() == 8 else torch.int32)
        ib = b.contiguous().view(torch.int64 if b.element_size() == 8 else torch.int32)
        return (ia != ib) & ~(torch.isnan(a) & torch.isnan(b))
    return a != b


def compare(A, B, out: Counter, examples: list, tag) -> None:
    """Every state tensor of A against B's, bit for bit; counts in `out`, the first differences in `examples`."""
    na, nb = state_names(A), state_names(B)
    if na != nb:
        out["names_differ"] += 1
        examples.append((tag, "names", sorted(set(na) ^ set(nb))))
    for k in na:
        if k not in nb:
            continue
        a, b = getattr(A, k), getattr(B, k)
        out["tensors"] += 1
        if a.shape != b.shape or a.dtype != b.dtype:
            out["differ"] += 1
            examples.append((tag, k, "shape or dtype", tuple(a.shape), tuple(b.shape), a.dtype, b.dtype))
            continue
        out["values"] += a.numel()
        bad = _differ(a, b.to(a.device))
        nbad = int(bad.sum())
        if nbad:
            out["differ"] += 1
            out["values_differ"] += nbad
            out[f"differ:{k}"] += nbad
            if len(examples) < 12:
                idx = bad.nonzero()[0].tolist()
                examples.append((tag, k, idx, a[tuple(idx)].item(), b[tuple(idx)].item()))
    if A.current_time != B.current_time:
        out["differ"] += 1
        examples.append((tag, "current_time", A.current_time, B.current_time))


def lockstep(cfg: dict, device: str = "cuda", kernel_cls=None, libs=None, graph: bool = False, torch_cls=None,
             compare_every: bool = True, a_kernel: bool = False) -> tuple:
    """The batch on two parks, the torch path (A; with a_kernel the kernel path's eager ticks) and the kernel path (B,
    or a CUDA graph of it), the same commands on both, every state tensor compared after the load and after every
    step's commands and tick; the events and the answers compared. Returns (counts, examples, A, B)."""
    rows, where = dc.rows_for(cfg)
    A = _park(cfg, rows, device, torch_cls)
    B = _park(cfg, rows, device, kernel_cls)
    g = KP.ComposedGraph(B, libs=libs) if graph else None
    W = len(cfg["worlds"])
    outs = {k: [{"S": {}, "E": [], "A": [], "B": [], "X": None} for _ in range(W)] for k in "AB"}
    by_step = [defaultdict(list) for _ in range(W)]
    for wi, w in enumerate(cfg["worlds"]):
        for rec in w["records"]:
            by_step[wi][rec[0]].append(rec)
    n = len(rows)
    out, examples = Counter(), []
    compare(A, B, out, examples, ("load",))
    for s in range(cfg["steps"] + 1):
        slots = max(len(by_step[wi].get(s, ())) for wi in range(W))
        for j in range(slots):
            recs = {wi: by_step[wi][s][j] for wi in range(W) if j < len(by_step[wi].get(s, ()))}
            pc.apply_batch(A, recs, where, n, outs["A"])
            pc.apply_batch(B, recs, where, n, outs["B"])
        if s > 0:
            if a_kernel:
                A.evolve(libs)
            else:
                A.evolve_torch()
            if g is not None:
                B.watch()
                g.tick()
                B.watch()
                B.after_tick()
            else:
                B.evolve(libs)
            ea, eb = A.events(), B.events()
            out["events"] += len(ea)
            if ea != eb:
                out["events_differ"] += 1
                examples.append((s, "events", ea[:3], eb[:3]))
            out["ticks"] += 1
            out["ball_steps"] += int(A.exists.sum())
            out["contacts"] += int((A.hits_n > 0).sum())
        if compare_every or s == cfg["steps"]:
            compare(A, B, out, examples, (s,))
    if [o["A"] for o in outs["A"]] != [o["A"] for o in outs["B"]]:
        out["answers_differ"] += 1
    out["worlds"] += W
    return out, examples, A, B


def run_batches(cfgs: list, **kw) -> tuple:
    total, ex = Counter(), []
    for bi, cfg in enumerate(cfgs):
        out, examples, _, _ = lockstep(cfg, **kw)
        total.update(out)
        ex.extend((bi, e) for e in examples[:3])
    return total, ex


def _clean(out: Counter, ex: list) -> None:
    bad = {k: v for k, v in out.items() if "differ" in k}
    assert not bad, (bad, ex[:6])


# ---- the hand-built worlds ----

def hand_built() -> list:
    import test_movement_vec_composed as tc
    return [tc._drop_world(0.0), tc._drop_world(150.0), tc._overrun_world()]


def edge_world() -> dict:
    """Two ships near the partition's lower x face, ship 2 thrusting outward (GOTO a point beyond the face): it
    reaches the edge box within the steps (a filing VecComposedPark refuses: FLAG_PARTITION), though not at its
    load. Ship 2 starts a fifth of a box inside the second box of the level it is filed at."""
    from exact_ballpark.space import LEVEL_WIDTH
    from exact_ballpark.vec_collide import HALF
    common = {"free": True, "massive": True, "agility": 1.0, "speed_fraction": 1.0, "max_ang_vel": 1.0,
              "rot_agility": 0.9, "vel": [0.0, 0.0, 0.0], "max_vel": 300.0, "mass": 1e7, "radius": 40.0}

    def cfg_at(x0):
        ships = [{**common, "id": 1, "interactive": True, "pos": [x0 + 4000.0, 1000.0, -2000.0]},
                 {**common, "id": 2, "interactive": False, "pos": [x0, 1000.0, -2000.0]}]
        w = {"ships": ships, "pool": list(pc.POOL), "records": [(0, "goto", 2, x0 - 1.0e6, 1000.0, -2000.0)],
             "start": 10}
        return {"do": False, "do_off": False, "master": False, "tick": 2000, "friction": 1.0e6,
                "new_orbit": False, "steps": 12, "worlds": [w]}
    probe = cfg_at(-HALF[0] + 1.0e5)
    rows, where = dc.rows_for(probe)
    p = _park(probe, rows, "cpu")
    level = int(p.level[where[(0, 2)]])
    return cfg_at(-HALF[0] + 1.2 * LEVEL_WIDTH[level])


# ---- the kernel path against the torch path, on the card ----

@CUDA
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_kernel_equals_the_torch_path_every_tensor_every_tick(seed):
    """The first four batches of each seed (64 worlds; both orientations, the server's view and a client's), every
    state tensor after every tick: no bit differs, the events and the answers are the same."""
    out, ex = run_batches(pc.batches(seed, 16, 16)[:4])
    _clean(out, ex)
    assert out["worlds"] == 64 and out["ticks"] > 60 and out["contacts"] > 500 and out["events"] > 50, dict(out)


@CUDA
@pytest.mark.parametrize("seed", [11, 3], ids=["development", "held_out"])
def test_the_kernel_on_the_rest_worlds(seed):
    """tests/park_rest_cases.py's worlds (long ticks, short warp-ins: drop-outs at rest re-filed far from their last
    filing), the first two batches of the development seed (11) and the held-out seed (3), every tensor every tick."""
    out, ex = run_batches(prc.batches(seed, 2, 16))
    _clean(out, ex)
    assert out["worlds"] == 32 and out["contacts"] > 100, dict(out)


@CUDA
def test_a_ball_reaching_the_partitions_edge_is_flagged_by_the_kernel_too():
    """A ball re-filed into the partition's edge box after a tick: the kernel's check flags it as the torch path's
    `_file` does (FLAG_PARTITION), every tensor every tick."""
    cfg = edge_world()
    out, ex, A, B = lockstep(cfg)
    _clean(out, ex)
    assert B.world_flags()[0] & vp.FLAG_PARTITION and A.world_flags() == B.world_flags()


def test_the_edge_world_reaches_the_edge_in_a_tick_not_at_its_load():
    """The edge world's precondition, on the CPU's torch path: no flag after the load, FLAG_PARTITION after the ticks."""
    cfg = edge_world()
    rows, where = dc.rows_for(cfg)
    p = _park(cfg, rows, "cpu")
    assert p.world_flags() == [0]
    got, park = pc.batched(cfg)
    assert park.world_flags()[0] & vp.FLAG_PARTITION


def test_the_rest_scenarios_show_the_refiling_at_rest():
    """The rest generator's development worlds (seed 11, the first two batches) on the CPU: the torch path equals
    the reference, and the planted fault (the drop-out's re-filing at rest left out), which changes no
    world of park_cases, changes many here."""
    import test_movement_vec_composed as tc
    plain = tc.run(11, "cpu", batches=range(2), gen=prc)
    tc._clean(plain, 25)
    fault = tc.run(11, "cpu", park_cls=tc.FAULTS["drop_not_filed_at_rest"], batches=range(2), gen=prc)
    assert fault["differ"] >= 10, dict(fault)


@CUDA
def test_the_kernel_on_the_hand_built_and_combat_worlds():
    """The drop-out met by the higher id only (at rest and moving), the overrun filed where it rests, and the worlds (missiles, formations, cloaks, warps; two batches), kernel against torch path."""
    out, ex = run_batches(hand_built())
    _clean(out, ex)
    out2, ex2 = run_batches(dc.batches(1, 6, 24)[:2])
    _clean(out2, ex2)
    assert out["contacts"] > 10 and out2["worlds"] == 48, (dict(out), dict(out2))


class _KernelPark(vp.VecComposedPark):
    """VecComposedPark whose evolve_torch is the kernel path (park_cases.batched ticks by evolve_torch)."""
    def evolve_torch(self):
        self.evolve()


@CUDA
def test_the_kernel_matches_fullballpark():
    """The kernel path against the scalar reference, every field of every ball at every step and the events (the torch
    path's own comparison, tests/test_movement_vec_composed.py's `run`), on two batches of each seed and the
    hand-built worlds."""
    import test_movement_vec_composed as tc
    for seed in (1, 2):
        out = tc.run(seed, "cuda", park_cls=_KernelPark, batches=range(2))
        tc._clean(out, 28)
    for cfg in hand_built():
        got, park = pc.batched(cfg, "cuda", _KernelPark)
        assert park.world_flags() == [0]
        assert dc.differences(pc.reference(cfg, cfg["worlds"][0]), got[0]) == []


@CUDA
def test_a_cuda_graph_equals_eager_ticks():
    """With dynamical orientation (batches 1 and 3 of the development seed): the tick captured once as a CUDA graph
    and replayed, against the torch path and against eager kernel ticks, every tensor every tick; a world's commands
    between replays (AddBall, launches, setters) reach the graph's buffers."""
    cfgs = [c for c in pc.batches(1, 16, 16)[:4] if c["do"]]
    assert len(cfgs) == 2
    out, ex = run_batches(cfgs, graph=True)
    _clean(out, ex)
    assert out["ticks"] > 20, dict(out)
    out, ex = run_batches(cfgs, graph=True, a_kernel=True)          # the graph against eager kernel ticks
    _clean(out, ex)
    with pytest.raises(ValueError, match="host"):
        cfg = pc.batches(1, 1, 2)[0]
        rows, _ = dc.rows_for(cfg)
        KP.ComposedGraph(_park(cfg, rows, "cuda"))


@CUDA
def test_collisions_off_is_vec_combat_parks_kernel_path():
    """With collisions off, evolve is VecCombatPark's four launches (the kernels) on the worlds:
    the torch path's bits (its two-massive-balls refusal kept; the worlds it refuses are not compared)."""
    import test_movement_vec_composed as tc
    out = tc.run(1, "cuda", park_cls=_KernelPark, nb=2, nw=24, gen=dc, collisions=False)
    from exact_ballpark import vec_combat as vc
    tc._clean(out, 40, allowed=vc.FLAG_MASSIVE)


@CUDA
def test_a_massive_warper_is_refused_by_the_kernel_too():
    """SetBallMassive during a warp (the refusal test's world): the kernel path flags the world
    FLAG_WARP_MASSIVE as the torch path does (check() raises); the values of a refused world are not compared."""
    import test_movement_vec_composed as tc
    cfg = tc._drop_world(0.0)
    rows, where = dc.rows_for(cfg)
    flags = []
    for path in ("torch", "kernel"):
        p = _park(cfg, rows, "cuda")
        n = len(rows)
        one = torch.zeros(n, dtype=torch.bool, device="cuda")
        one[where[(0, 2)]] = True
        p.entity_warp_in(one, torch.tensor([[1000.0, -2000.0, 500.0]] * n, dtype=torch.float64, device="cuda"),
                         torch.full((n,), 6000, dtype=torch.int64, device="cuda"))
        tick = p.evolve_torch if path == "torch" else p.evolve
        tick()
        p.massive = torch.where(one, True, p.massive)
        tick()
        flags.append(p.world_flags())
        with pytest.raises(RuntimeError, match="warp proper"):
            p.check()
    assert flags[0] == flags[1] and flags[1][0] & vp.FLAG_WARP_MASSIVE


def test_the_kernel_path_needs_a_card(monkeypatch):
    """The kernels and ComposedGraph refuse a park on the CPU (they need a CUDA device; evolve_torch is the CPU path),
    before any library is loaded; on the CPU `evolve` is the tick on tensors, a kernel library
    refused."""
    monkeypatch.setattr(KP, "libraries", lambda *a, **k: pytest.fail("the libraries were loaded for a CPU park"))
    cfg = pc.batches(1, 1, 2)[0]
    rows, _ = dc.rows_for(cfg)
    p = _park(cfg, rows, "cpu")
    for fn in (KP.tick, KP.after):
        with pytest.raises(ValueError, match="CUDA device"):
            fn(p)
    p.dynamical_orientation = True
    with pytest.raises(ValueError, match="CUDA device"):
        KP.ComposedGraph(p)
    p.dynamical_orientation = bool(cfg["do"])
    with pytest.raises(ValueError, match="kernel library"):
        p.evolve(lib={})
    q = _park(cfg, rows, "cpu")
    p.evolve()
    q.evolve_torch()
    for name in p.row_tensors():
        x, y = getattr(p, name), getattr(q, name)
        assert torch.equal(x.view(torch.int64) if x.dtype == torch.float64 else x,
                           y.view(torch.int64) if y.dtype == torch.float64 else y), name


# ---- with collisions off, the collision lists after the kernels ----

class _CardLists(vp.VecComposedPark):
    """VecComposedPark with collisions off whose tick on tensors is checked, each tick, against the card's bookkeeping:
    after the tick its lists are made junk and hits_n is put back to the kernels' count (its own less ev_self), and
    `_lists_from_counts` from the follow ids at the tick's start must give back the tick's hits and hits_n bit for bit
    (then the tick's own are put back). `evolve_torch` is replaced on the instance, so the class's own stays the park's
    (`_g11` reads it)."""
    seen: Counter = Counter()

    def __init__(self, *a, **kw):
        kw["collisions"] = False
        super().__init__(*a, **kw)
        self.evolve_torch = self._checked_tick

    def _checked_tick(self, active=None):
        assert active is None
        follow, ptr = self.follow_id.clone(), self.follow_ptr.clone()
        vp.VecComposedPark.evolve_torch(self)
        want_h, want_n = self.hits.clone(), self.hits_n.clone()
        count = want_n - self.ev_self.to(torch.int64)
        self.hits, self.hits_n = torch.full_like(want_h, -7), count.clone()
        self._lists_from_counts(follow)
        s = _CardLists.seen
        s["ticks"] += 1
        s["differ"] += int(not (torch.equal(self.hits, want_h) and torch.equal(self.hits_n, want_n)))
        s["contacts"] += int((count > 0).sum())
        s["self_collisions"] += int(self.ev_self.sum())
        t = ptr.clamp(min=0)
        s["released_after_a_contact"] += int(((count > 0) & (ptr >= 0) & self.dying.bool()[t]
                                              & (self.follow_ptr < 0)).sum())
        self.hits, self.hits_n = want_h, want_n


def test_with_collisions_off_the_lists_follow_from_the_kernels_counts():
    """With collisions off, `evolve` on a card ticks with VecCombatPark's kernels, which
    count each row's contacts with its target (hits_n) and the master's self-collision (ev_self) but keep no list, then
    makes the lists from those counts and the follow ids at the tick's start (`_lists_from_counts`). Here on the CPU,
    every tick of the worlds (development seed 1), the released missiles' worlds of
    tests/test_movement_vec_defenders.py and the mushroom worlds (seed 1, missiles released after a
    contact with a mushroom removed that tick): from the tick on tensors' own counts, `_lists_from_counts` gives back
    its hits and hits_n bit for bit, every row. (The card's item compares the kernels' own counts.)"""
    import composed_leftover_cases as clc
    import test_movement_vec_defenders as td
    _CardLists.seen = Counter()
    for cfg in dc.batches(1, 6, 24) + td.owner_cfgs()[:4]:
        pc.batched(cfg, "cpu", park_cls=_CardLists)
    for cfg in clc.batches(1, 12, 12):
        clc.batched(cfg, "cpu", park_cls=_CardLists)
    s = _CardLists.seen
    assert s["differ"] == 0, dict(s)
    assert (s["ticks"] >= 373 and s["contacts"] >= 887 and s["self_collisions"] >= 88
            and s["released_after_a_contact"] >= 6), dict(s)       # the run: these, exactly


def test_a_combat_graph_refuses_the_composed_park(monkeypatch):
    """CombatGraph captures VecCombatPark's kernels alone, so it refuses a VecComposedPark (with
    collisions off its lists are made after the kernels, by `evolve`; with them on its tick is ComposedGraph's), before
    any library is loaded; ComposedGraph's refusal of collisions off names `evolve`."""
    from exact_ballpark import vec_combat_cuda as KC
    monkeypatch.setattr(KC, "load_library", lambda *a, **k: pytest.fail("the library was loaded"))
    monkeypatch.setattr(KP, "libraries", lambda *a, **k: pytest.fail("the libraries were loaded"))
    cfg = pc.batches(1, 1, 2)[0]
    rows, _ = dc.rows_for(cfg)
    for collisions in (False, True):
        p = _park(cfg, rows, "cpu", collisions=collisions)
        p.dynamical_orientation = True
        with pytest.raises(ValueError, match="a VecComposedPark ticks by ComposedGraph"):
            KC.CombatGraph(p)
    q = _park(cfg, rows, "cpu", collisions=False)
    q.dynamical_orientation = True
    q.dev = torch.device("cuda")                    # past the device check, as a card's park (nothing runs)
    with pytest.raises(ValueError, match="with collisions off the park ticks by evolve"):
        KP.ComposedGraph(q)


def _lockstep_off_masked(cfg: dict, cls, masked: bool, seed: int = 5) -> tuple:
    """Two collisions-off parks on the card, the tick on tensors (A) and the kernel path (B), the same commands, every
    state tensor compared after every step; with `masked` each tick under OnTick's per-world mask (a seeded half of the
    worlds, a world's rows together)."""
    rows, where = dc.rows_for(cfg)
    A, B = _park(cfg, rows, "cuda", cls), _park(cfg, rows, "cuda", cls)
    W = len(cfg["worlds"])
    outs = {k: [{"S": {}, "E": [], "A": [], "B": [], "X": None} for _ in range(W)] for k in "AB"}
    by_step = [defaultdict(list) for _ in range(W)]
    for wi, w in enumerate(cfg["worlds"]):
        for rec in w["records"]:
            by_step[wi][rec[0]].append(rec)
    g = torch.Generator().manual_seed(seed)
    out, examples = Counter(), []
    compare(A, B, out, examples, ("load",))
    for s in range(cfg["steps"] + 1):
        slots = max(len(by_step[wi].get(s, ())) for wi in range(W))
        for j in range(slots):
            recs = {wi: by_step[wi][s][j] for wi in range(W) if j < len(by_step[wi].get(s, ()))}
            pc.apply_batch(A, recs, where, len(rows), outs["A"])
            pc.apply_batch(B, recs, where, len(rows), outs["B"])
        if s > 0:
            act = (torch.rand(W, generator=g) < 0.5)[A.world_index.cpu()].to("cuda") if masked else None
            A.evolve_torch(act)
            B.evolve(None, act)
            ea, eb = A.events(), B.events()
            out["events"] += len(ea)
            if ea != eb:
                out["events_differ"] += 1
                examples.append((s, "events", ea[:3], eb[:3]))
            out["ticks"] += 1
            out["lists"] += int((A.hits_n > 0).sum())
        compare(A, B, out, examples, (s,))
    return out, examples


@CUDA
def test_with_collisions_off_the_kernel_path_keeps_the_tensor_ticks_lists():
    """On the card, with collisions off, the kernel path (VecCombatPark's kernels, then
    `_lists_from_counts`) equals the tick on tensors on the same card, every state tensor every tick, `hits` and
    `hits_n` among them: the worlds (two batches), every tick unmasked and under OnTick's per-world mask;
    and the mushroom worlds (seed 1, four batches): every ball's state and collision list at every step
    and the events."""
    import composed_leftover_cases as clc
    import test_movement_vec_defenders as td
    out, ex = Counter(), []
    for cfg in dc.batches(1, 6, 24)[:2]:
        for masked in (False, True):
            o, e = _lockstep_off_masked(cfg, td._Off, masked)
            out.update(o)
            ex.extend(e[:3])
    _clean(out, ex)
    assert out["ticks"] > 0 and out["lists"] > 0 and out["events"] > 0, dict(out)
    lists = 0
    for cfg in clc.batches(1, 12, 12)[:4]:
        a, _ = clc.batched(cfg, "cuda", path="torch", park_cls=td._Off)
        b, _ = clc.batched(cfg, "cuda", path="kernel", park_cls=td._Off)
        assert all(not clc.differences(x, y) for x, y in zip(a, b))
        lists += sum(1 for w in a for st in w["S"].values() for row in st.values() if row["collisions"])
    assert lists > 0


# ---- planted faults: each must change the comparison ----

# the composed candidates (vec_collide_cuda.cu under XBPC_COMPOSED): (old text, new text), each found once
COLLIDE_FAULTS = {
    "drop_seen_by_all": ("(P.dropped[j] && P.id[j] < P.id[j0])", "(P.dropped[j] && P.id[j] != P.id[j0])"),
    "drop_seen_by_none": ("P.massive[j] || (P.dropped[j] && P.id[j] < P.id[j0])", "P.massive[j]"),
    "missiles_seen_by_others": ("(me_defender || !P.missile[j])", "true"),
    "no_missile_exemption": ("const bool kept = !(st.missile && other == st.follow);", "const bool kept = true;"),
}
DC2_FAULTS = ("contracted_fma", "libm_exp", "reversed_order")     # the own switches, in the composed build
# the development batches and the hand-built worlds each fault is looked for in
FAULT_BATCHES = {"contracted_fma": [0], "libm_exp": [0, 1], "reversed_order": [0], "drop_seen_by_all": [0, 1],
                 "drop_seen_by_none": [0, 1], "missiles_seen_by_others": [0], "no_missile_exemption": [0],
                 "not_filed_at_rest": [], "no_last_c": [0], "no_self_collision": [0, 1, 2, 3],
                 "drop_at_warp_position": [0, 1]}


def fault_libs(fault: str, tmp_path) -> dict:
    """The two libraries with one planted fault."""
    if fault in DC2_FAULTS:
        col = KD.load_library("sm90", fault=fault, checks=fault == "reversed_order", composed=True)
        return KP.libraries(collide_lib=col)
    if fault in COLLIDE_FAULTS:
        old, new = COLLIDE_FAULTS[fault]
        text = KD.SOURCE.read_text()
        assert text.count(old) == 1, (fault, text.count(old))
        src = tmp_path / KD.SOURCE.name
        src.write_text(text.replace(old, new))
        return KP.libraries(collide_lib=KD.load_library("native", composed=True, source=src))
    return KP.libraries(composed_lib=KP.load_library(fault=fault))


@CUDA
@pytest.mark.parametrize("fault", sorted(set(DC2_FAULTS) | set(COLLIDE_FAULTS) | set(KP.FAULTS)))
def test_planted_faults_are_caught(fault, tmp_path):
    libs = fault_libs(fault, tmp_path)
    cfgs = [pc.batches(1, 16, 16)[b] for b in FAULT_BATCHES[fault]] + hand_built()
    if fault == "not_filed_at_rest":
        cfgs.append(prc.batches(11, 1, 16)[0])
    out, ex = run_batches(cfgs, libs=libs)
    assert out["differ"] + out["events_differ"] > 0, (fault, dict(out))


# ---- the build, on the CPU ----

def test_the_default_build_of_dc2s_kernel_is_unchanged():
    """The composed switch adds nothing to the default build: its header and defines are the default build's; the composed build's header adds
    the modes its candidates read; every composed block is outside the planted-fault blocks."""
    h0 = KD.tables_header()
    assert "XBPC_MISSILE" not in h0 and "XBPC_MINIBALL" not in h0
    h1 = KD.tables_header(composed=True)
    from exact_ballpark import modes as MO
    for name in KD.COMPOSED_MODES:
        assert f"constexpr long long XBPC_{name} = {getattr(MO, name)}LL;" in h1
    assert h1.startswith(h0[:-1])                      # the default header, then the modes
    text = KD.SOURCE.read_text()
    body = KD.active_text(text, 0)
    assert body.count("#if XBPC_COMPOSED") == text.count("#if XBPC_COMPOSED") >= 8
    for k in (1, 2, 3):                                  # each fault block intact with the composed blocks present
        assert KD.active_text(text, k).count("#if XBPC_COMPOSED") == text.count("#if XBPC_COMPOSED")
    assert KD.scan_source(body) == []


def test_the_sources_call_no_math_library():
    """vec_composed_cuda.cu as compiled with each planted fault calls no function of CUDA's math library; each fault's
    block is there; the header's values are the reference modules' (never typed)."""
    from exact_ballpark import space as SP
    text = KP.SOURCE.read_text()
    for level in [0, *KP.FAULTS.values()]:
        assert KP.K.scan_source(KP.active_text(text, level)) == [], level
    for name, level in KP.FAULTS.items():
        assert f"XBPP_FAULT == {level}" in text or f"XBPP_FAULT != {level}" in text, name
    vals = KP.table_values()
    assert vals["P_REST_SQ"] == ("d", vp.REST_SQ) and vals["P_FLAG_PARTITION"] == ("int", vp.FLAG_PARTITION)
    assert vals["P_LEVEL_GRID"] == ("i64", tuple(SP.LEVEL_GRID)) and vals["P_LEVEL_WIDTH0"][1] == SP.LEVEL_WIDTH[0]
    for name, text_ in COLLIDE_FAULTS.items():
        assert KD.SOURCE.read_text().count(text_[0]) == 1, name


@NVCC
def test_the_build_pins_every_rounding():
    """The composed library as built: the PTX check of vec_orient_cuda (every float operation rounded to nearest, the
    fused multiply-adds only the C library's transcriptions'), no source-scan hit; the composed build: its PTX check
    with exactly its own fused multiply-adds."""
    m = KP.build()
    assert m["ptx_check"]["ok"] and m["source_scan"] == []
    c = KD.build(composed=True)
    assert c["ptx_check"]["ok"] and c["ptx_check"]["fp_ops"]["fma.rn.f64"] == KD.EXP_FMAS + 1
    assert "-DXBPC_COMPOSED=1" in c["defines"] and "_composed_" in c["library"]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""OnTick's per-world intervals in the one-bubble server layers: `VecServer.on_tick` (the layer over
VecOrientPark, VecCollidePark, VecModesPark) and the combat server's (the CombatServer over VecCombatPark, the
same method) against the scalar server layer driven through FullBallpark.on_tick (the Ticker at DoPreTick, send_batch
at DoPostTick), bit for bit after every call: each world's sends in order, every ball, the clocks (mTime, mFirstTime,
mCurrentTime); on the CPU (the torch paths) and the card (the parks' kernels and the server kernels);
the server kernels against their torch paths under OnTick, every tensor after every call; a masked tick of every world
equal to the unmasked tick, an empty mask changing nothing; a server that never calls on_tick holding no new tensor;
planted faults caught. Generators and runners: tests/server_ontick_cases.py."""

from __future__ import annotations

import pytest
import torch

import server_cases as S
import server_combat_kernel_cases as SK
import server_ontick_cases as O
import server_space_cases as SS
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_space as SP

CUDA = torch.cuda.is_available()
need_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
DEV_SEEDS = (1, 7)
HELD_OUT = 31                                    # (19 ran before the freeze by a slip: now development)
N, CALLS = 16, 14
PARKS_CPU = [("orient", False), ("orient", True), ("collide", False)]


def _clean(r: dict) -> None:
    assert not r["sends"], r["sends"][:6]
    assert not r["rows"], r["rows"][:6]
    assert not r["clocks"], r["clocks"][:6]
    assert r["compared"] > 0 and r["updates"] > 0
    assert len(r["out"]) <= r["worlds"] // 4, r["out"]       # most worlds stay in the layer's domain to the end
    cov = r["coverage"]
    for k in ("worlds too early", "first calls", "resynchronisations", "remainders carried", "masked rounds",
              "calls kept by a world without a tick"):
        assert cov.get(k, 0) > 0, (k, cov)


def _dirty(r: dict) -> bool:
    return bool(r["sends"] or r["rows"] or r["clocks"])


# ===================================================================================== the CPU (torch paths)

@pytest.mark.parametrize("park,do", PARKS_CPU)
@pytest.mark.parametrize("seed", DEV_SEEDS, ids=[f"development_{s}" for s in DEV_SEEDS])
def test_on_tick_matches_the_scalar_server_layer_on_the_cpu(seed, park, do):
    _clean(O.run(O.server_cases(seed, N, park, do, CALLS), park, "cpu", do=do))


@pytest.mark.parametrize("seed", DEV_SEEDS, ids=[f"development_{s}" for s in DEV_SEEDS])
def test_the_combat_server_on_tick_matches_the_scalar_server_layer_on_the_cpu(seed):
    _clean(O.run(O.combat_cases(seed, N, "cpu", CALLS), "combat", "cpu"))


@pytest.mark.parametrize("park,do", PARKS_CPU + [("combat", False)])
def test_held_out_on_tick_on_the_cpu(park, do):
    cs = (O.combat_cases(HELD_OUT, N, "cpu", CALLS) if park == "combat" else
          O.server_cases(HELD_OUT, N, park, do, CALLS))
    _clean(O.run(cs, park, "cpu", do=do))


def test_the_scenarios_reach_every_branch_of_the_schedule():
    """Coverage, seed 1, every CPU park: too early, first calls, the master's resynchronisation, catch-ups of two or
    more intervals, remainders carried, masked rounds and rounds of every world, calls kept by a world without a tick,
    and a call pushed again across a kept queue (TIME-710's duplicate test against the kept last action)."""
    total = {}
    for park, do in PARKS_CPU + [("combat", False)]:
        cs = O.combat_cases(1, N, "cpu", CALLS) if park == "combat" else O.server_cases(1, N, park, do, CALLS)
        r = O.run(cs, park, "cpu", do=do)
        for k, v in r["coverage"].items():
            total[k] = total.get(k, 0) + v
    for k in ("worlds too early", "first calls", "resynchronisations", "intervals run", "remainders carried",
              "masked rounds", "rounds of every world", "calls kept by a world without a tick",
              "repeats across a kept queue"):
        assert total.get(k, 0) > 0, (k, total)
    assert total["intervals run"] > total["masked rounds"] + total["rounds of every world"]     # catch-ups


# ============================================================================ the mask's identities (CPU)

def _state(server) -> dict:
    """Every tensor of the park, the adapter and the server (OnTick's own bookkeeping apart), on the host."""
    out = {}
    for tag, obj in (("park", server.a.park), ("adapter", server.a), ("server", server)):
        for k, v in vars(obj).items():
            if torch.is_tensor(v) and k not in ("_seen_w",):
                out[f"{tag}.{k}"] = v.detach().cpu().clone()
    return out


def _same_state(a: dict, b: dict) -> list:
    return sorted(k for k in set(a) | set(b) if k not in a or k not in b or not SK._bits_equal(a[k], b[k]))


def _view(b) -> dict:
    v = O._batch_view(b) if b.ticked is not None else None
    if v is None:
        b.ticked = torch.ones_like(b.stamp, dtype=torch.bool)
        v = O._batch_view(b)
        b.ticked = None
    v.pop("ticked")
    return v


@pytest.mark.parametrize("park,do", PARKS_CPU)
def test_a_masked_tick_of_every_world_is_the_unmasked_tick(park, do):
    """Twin servers, the same pushes: one ticks unmasked, the other with a mask of every world: every tensor, every
    field of the batch and the decoded updates equal, every tick."""
    cs = S.cases(3, 8, park, do, ticks=6)
    a, _ = S.build(cs, park, "cpu", do=do)
    b, _ = S.build(cs, park, "cpu", do=do)
    ones = torch.ones(len(cs), dtype=torch.bool)
    for t in range(6):
        for srv in (a, b):
            for kind, *data in S.pushes(cs, t, "cpu"):
                srv.push(kind, *data)
        a.pre_tick(snapshot=True)
        a.evolve()
        a.post_tick()
        b.pre_tick(snapshot=True, mask=ones)
        b.evolve(mask=ones)
        b.post_tick(mask=ones)
        assert not _same_state(_state(a), _state(b)), (t, _same_state(_state(a), _state(b)))
        assert _view(a.batch) == _view(b.batch), t
        assert S.canon([S._listify(x) for x in VS.decode(a)]) == S.canon([S._listify(x) for x in VS.decode(b)])


@pytest.mark.parametrize("park,do", PARKS_CPU)
def test_an_empty_mask_changes_nothing_and_keeps_the_queue(park, do):
    """After pushes, a pre-tick, tick and post-tick under an empty mask: every tensor as before, the queue kept with
    each call's queued flag, the batch routing and sending nothing; then an unmasked tick equals a twin's unmasked tick
    that never saw the empty round."""
    cs = S.cases(5, 8, park, do, ticks=4)
    a, _ = S.build(cs, park, "cpu", do=do)
    b, _ = S.build(cs, park, "cpu", do=do)
    none = torch.zeros(len(cs), dtype=torch.bool)
    for t in range(4):
        for srv in (a, b):
            for kind, *data in S.pushes(cs, t, "cpu"):
                srv.push(kind, *data)
        before, queued = _state(b), [q.clone() for q in b._queued()]
        b.pre_tick(snapshot=True, mask=none)
        b.evolve(mask=none)
        b.post_tick(mask=none)
        assert not _same_state(before, _state(b)), (t, _same_state(before, _state(b)))
        want, kept = [q for q in queued if bool(q.any())], b._queued()
        assert len(want) == len(kept) and all(torch.equal(x, y) for x, y in zip(want, kept)), t
        assert not bool(b.batch.has_character.any()) and not bool(b.batch.gets_bubble.any())
        assert not any(VS.decode(b))
        a.tick()
        b.tick()
        assert not _same_state(_state(a), _state(b)), (t, _same_state(_state(a), _state(b)))
        assert S.canon([S._listify(x) for x in VS.decode(a)]) == S.canon([S._listify(x) for x in VS.decode(b)])


def test_a_server_without_on_tick_holds_no_new_tensor():
    """the ticks (no on_tick): no OnTick clock, no per-world first pre-tick, no removal flag; the batches' `ticked`
    None."""
    cs = S.cases(2, 6, "orient", False, ticks=3)
    server, _ = S.build(cs, "orient", "cpu")
    for t in range(3):
        for kind, *data in S.pushes(cs, t, "cpu"):
            server.push(kind, *data)
        server.tick()
        assert server.batch.ticked is None
    for k in ("time", "first_time", "_seen_w", "removed", "tick_counts"):
        assert k not in vars(server), k


def test_a_removal_by_bring_out_the_deads_is_flagged():
    """A combat world whose moribund ball is due at an OnTick: BringOutTheDeads removes it there (every world, before
    the interval check), the world is flagged in `removed` and `check()` refuses it (the one-bubble layer sends no
    RemoveBalls)."""
    cs = O.combat_cases(1, 4, "cpu", 3)
    server, index = O.build(cs, "combat", "cpu")
    p = server.a.park
    r = index[(cs[2]["id"], cs[2]["balls"][-1]["id"])]
    p.moribund[r] = True
    p.effect_stamp[r] = p.clock[r] - 1
    server.on_tick(torch.zeros(len(cs), dtype=torch.int64))         # too early everywhere: BringOutTheDeads still runs
    assert server.removed.tolist() == [False, False, True, False]
    assert not bool(p.exists[r])
    with pytest.raises(RuntimeError, match="BringOutTheDeads"):
        server.check()


# ===================================================================================== planted faults (CPU)

def _fault_queue_dropped(monkeypatch):
    def f(self, on, flag):
        slots, self.slots = self.slots, []
        return [VS._with_flag(s, flag, getattr(s, flag) & on) for s in slots]
    monkeypatch.setattr(VS.VecServer, "_split_queue", f)


def _fault_last_reset(monkeypatch):
    monkeypatch.setattr(VS.VecServer, "_keep_last", lambda self, last, on: self._fresh_last())


def _fault_pre_tick_unmasked(monkeypatch):
    orig = VS.VecServer.pre_tick
    monkeypatch.setattr(VS.VecServer, "pre_tick", lambda self, snapshot=None, mask=None: orig(self, True, None))


def _fault_evolve_unmasked(monkeypatch):
    orig = VS.VecServer.evolve
    monkeypatch.setattr(VS.VecServer, "evolve", lambda self, mask=None: orig(self, None))


def _fault_first_time_ignored(monkeypatch):
    orig = VS.VecServer.on_tick

    def f(self, sim_time):
        if getattr(self, "time", None) is None:
            self.time = torch.zeros(self.W, dtype=torch.int64, device=self.dev)
            self.first_time = torch.zeros(self.W, dtype=torch.bool, device=self.dev)
        return orig(self, sim_time)
    monkeypatch.setattr(VS.VecServer, "on_tick", f)


def _fault_remainder_dropped(monkeypatch):
    orig = VS.VecServer.on_tick

    def f(self, sim_time):
        old = self.time.clone() if getattr(self, "time", None) is not None else None
        out = orig(self, sim_time)
        sim = torch.as_tensor(sim_time, dtype=torch.int64).to(self.dev).reshape(self.W)
        went = self.time != (old if old is not None else torch.zeros_like(self.time))
        self.time = torch.where(went, sim, self.time)
        return out
    monkeypatch.setattr(VS.VecServer, "on_tick", f)


def _fault_no_resync(monkeypatch):
    orig = VS.VecServer.on_tick

    def f(self, sim_time):
        self.a.master = False                   # (a collision world: no normal cloak, so no sensor reads it)
        return orig(self, sim_time)
    monkeypatch.setattr(VS.VecServer, "on_tick", f)


CPU_FAULTS = {"queue_dropped": (_fault_queue_dropped, "orient"), "last_reset": (_fault_last_reset, "orient"),
              "pre_tick_unmasked": (_fault_pre_tick_unmasked, "orient"),
              "evolve_unmasked": (_fault_evolve_unmasked, "orient"),
              "first_time_ignored": (_fault_first_time_ignored, "orient"),
              "remainder_dropped": (_fault_remainder_dropped, "orient"),
              "no_resync": (_fault_no_resync, "collide")}


@pytest.mark.parametrize("fault", sorted(CPU_FAULTS))
def test_planted_faults_are_caught_on_the_cpu(fault, monkeypatch):
    plant, park = CPU_FAULTS[fault]
    plant(monkeypatch)
    r = O.run(O.server_cases(1, N, park, False, CALLS), park, "cpu")
    assert _dirty(r), fault


# ================================================================================================ the card

GPU_CASES = [("modes", False, "torch", "torch"), ("modes", True, "torch", "torch"), ("modes", False, "torch", "cuda"),
             ("modes", True, "torch", "cuda"), ("orient", True, "cuda", "torch"), ("collide", False, "cuda", "torch")]


@need_cuda
@pytest.mark.parametrize("park,do,kernel,server_kernel", GPU_CASES,
                         ids=[f"{p}-{int(d)}-park_{k}-server_{s}" for p, d, k, s in GPU_CASES])
@pytest.mark.parametrize("seed", DEV_SEEDS, ids=[f"development_{s}" for s in DEV_SEEDS])
def test_on_tick_matches_the_scalar_server_layer_on_the_gpu(seed, park, do, kernel, server_kernel):
    _clean(O.run(O.server_cases(seed, N, park, do, CALLS), park, "cuda", do=do, kernel=kernel,
                 server_kernel=server_kernel))


@need_cuda
@pytest.mark.parametrize("server_kernel", ["torch", "cuda"])
@pytest.mark.parametrize("seed", DEV_SEEDS, ids=[f"development_{s}" for s in DEV_SEEDS])
def test_the_combat_server_on_tick_matches_the_scalar_server_layer_on_the_gpu(seed, server_kernel):
    _clean(O.run(O.combat_cases(seed, N, "cuda", CALLS), "combat", "cuda", server_kernel=server_kernel))


@need_cuda
@pytest.mark.parametrize("park,do", [("modes", False), ("modes", True), ("combat_kernel", False),
                                     ("combat_kernel", True)])
@pytest.mark.parametrize("seed", DEV_SEEDS, ids=[f"development_{s}" for s in DEV_SEEDS])
def test_the_server_kernels_equal_their_torch_paths_under_on_tick(seed, park, do):
    cs = (O.combat_kernel_cases(seed, N, do, CALLS) if park == "combat_kernel" else
          O.server_cases(seed, N, park, do, CALLS))
    bad, vol = O.kernel_against_torch(cs, "combat" if park == "combat_kernel" else park, do=do)
    assert not bad, bad[:8]
    assert vol["masked rounds"] > 0 and vol["kept calls"] > 0 and vol["decoded updates"] > 0, vol


@need_cuda
@pytest.mark.parametrize("case", ["modes", "combat"])
def test_held_out_on_tick_on_the_gpu(case):
    if case == "modes":
        for do in (False, True):
            _clean(O.run(O.server_cases(HELD_OUT, N, "modes", do, CALLS), "modes", "cuda", do=do,
                         server_kernel="cuda"))
        bad, _ = O.kernel_against_torch(O.server_cases(HELD_OUT, N, "modes", True, CALLS), "modes", do=True)
    else:
        _clean(O.run(O.combat_cases(HELD_OUT, N, "cuda", CALLS), "combat", "cuda", server_kernel="cuda"))
        bad, _ = O.kernel_against_torch(O.combat_kernel_cases(HELD_OUT, N, False, CALLS), "combat")
    assert not bad, bad[:8]


@need_cuda
@pytest.mark.parametrize("fault", ["no_put_back", "queue_dropped"])
def test_planted_faults_are_caught_on_the_gpu(fault, monkeypatch):
    """The kernel path's masked pre-tick: without the put-back of the server state (the stamp and the bubble update of
    a world without a tick), or with its queue dropped, the kernel no longer equals its torch path."""
    if fault == "no_put_back":
        monkeypatch.setattr(VS.VecServer, "_put_back_cuda", lambda self, on, saved, q: None)
    else:
        _fault_queue_dropped(monkeypatch)
        orig = VS.VecServer._split_queue

        def only_kernel(self, on, flag):
            if flag == "ok":                    # the torch path keeps its queue: the reference
                slots, self.slots = self.slots, []
                kept = [VS._with_flag(s, flag, getattr(s, flag) & ~on) for s in slots]
                self.slots = [s for s in kept if bool(getattr(s, flag).any())]
                return [VS._with_flag(s, flag, getattr(s, flag) & on) for s in slots]
            return orig(self, on, flag)
        monkeypatch.setattr(VS.VecServer, "_split_queue", only_kernel)
    bad, _ = O.kernel_against_torch(O.server_cases(1, N, "modes", False, CALLS), "modes")
    assert bad, fault


# ===================================================== the partition layer's masked tick (VecModesPark, VecOrientPark)

def _clean_space(r: dict) -> None:
    _clean(r)
    assert r["coverage"].get("world-calls with several bubbles", 0) > 0, r["coverage"]


@pytest.mark.parametrize("seed", DEV_SEEDS, ids=[f"development_{s}" for s in DEV_SEEDS])
def test_the_space_server_on_tick_over_the_orient_park_on_the_cpu(seed):
    """SpaceServer.on_tick over VecOrientPark (vec_space's masked tick: VecOrientPark.evolve(active) on the worlds'
    rows, then the dead bubbles' rows restored) against the scalar server layer through FullBallpark.on_tick."""
    _clean_space(O.run_space(O.space_cases(seed, N, "orient", CALLS), "orient", "cpu"))


def test_held_out_space_on_tick_on_the_cpu():
    _clean_space(O.run_space(O.space_cases(HELD_OUT, N, "orient", CALLS), "orient", "cpu"))


def _layer_state(server) -> dict:
    """Every tensor of the park, the adapter, the server and the layer's state, on the host (the AFTER_EVOLVE
    operation's input `dropped`, written for every world before each tick's operation, apart)."""
    out = _state(server)
    for k, v in server.space.st.items():
        if torch.is_tensor(v) and k != "dropped":
            out[f"layer.{k}"] = v.detach().cpu().clone()
    return out


def _space_masks(device: str, park: str) -> None:
    """Twin SpaceServers, the same pushes, pre-ticks and post-ticks: one evolves unmasked, the other under a mask of
    every world: every tensor equal after every tick; and a third evolve under an empty mask changes nothing."""
    cs = SS.cases(4, 8, park, ticks=8)
    a, _ = SS.build(cs, park, device)
    b, _ = SS.build(cs, park, device)
    W = len(cs)
    for t in range(8):
        for srv in (a, b):
            for kind, *data in S.pushes(cs, t, device):
                srv.push(kind, *data)
            srv.pre_tick(snapshot=True)
        a.evolve()
        b.evolve(mask=torch.ones(W, dtype=torch.bool, device=b.dev))
        for srv in (a, b):
            srv.post_tick()
        assert not _same_state(_layer_state(a), _layer_state(b)), (t, _same_state(_layer_state(a), _layer_state(b)))
        before = _layer_state(b)
        b.evolve(mask=torch.zeros(W, dtype=torch.bool, device=b.dev))
        assert not _same_state(before, _layer_state(b)), (t, _same_state(before, _layer_state(b)))


def test_the_layers_masked_tick_over_the_orient_park_is_the_unmasked_tick_or_nothing():
    _space_masks("cpu", "orient")


def test_a_planted_fault_in_the_layers_row_mask_is_caught(monkeypatch):
    """vec_space's row mask taken as every row (the worlds without a tick tick too): caught."""
    monkeypatch.setattr(SP.VecSpace, "row_mask", lambda self, worlds: torch.ones(self.view.R, dtype=torch.bool,
                                                                                    device=self.dev))
    assert _dirty(O.run_space(O.space_cases(1, N, "orient", CALLS), "orient", "cpu"))


def test_the_space_servers_kernel_refuses_on_tick():
    from exact_ballpark import vec_server_space_cuda as VSQ
    with pytest.raises(ValueError, match="not on the space server's kernel"):
        VSQ.CudaSpaceServer.on_tick(object(), [0])


@need_cuda
@pytest.mark.parametrize("seed", DEV_SEEDS, ids=[f"development_{s}" for s in DEV_SEEDS])
def test_the_space_server_on_tick_over_the_modes_park_on_the_gpu(seed):
    """SpaceServer.on_tick over VecModesPark (vec_space's masked tick: VecModesPark.evolve(active) on the worlds' rows
    inside the dead bubbles' moribund flags) against the scalar server layer: follows, orbits and warps too."""
    _clean_space(O.run_space(O.space_cases(seed, N, "modes", CALLS), "modes", "cuda"))


@need_cuda
def test_held_out_space_on_tick_on_the_gpu():
    _clean_space(O.run_space(O.space_cases(HELD_OUT, N, "modes", CALLS), "modes", "cuda"))


@need_cuda
def test_the_layers_masked_tick_over_the_modes_park_is_the_unmasked_tick_or_nothing():
    _space_masks("cuda", "modes")

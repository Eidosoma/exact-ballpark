# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's collisions on the master kept within bubbles: the partition layer
(exact_ballpark.vec_space.VecSpace on its CollideView) over the collision park (exact_ballpark.vec_collide.VecCollidePark:
its torch path, and the kernel on a GPU), and the server layer over both (exact_ballpark.vec_server_space.SpaceServer
over SpaceCollideAdapter), bit for bit against the master FullBallpark and the scalar server layer on generated worlds
(tests/space_collide_cases.py; development seeds 1 and 7, held-out seed 11, first run after the generator's freeze:
seed 7 was meant as the held-out seed and was run before the freeze by mistake, so it is a second
development seed); two
constructed worlds pin the section 9 (two massive ships across a top face between two bubbles collide on a
client's view, never on the master; across a face within one bubble they collide on both); the same worlds against
Destiny's own code compiled verbatim (the conformance kit's space harness, where it is built): the master reference on every row
kind, and the batched layer's ball rows directly; the planted faults must be caught. The CPU cases run everywhere; the
rest need a GPU."""

import struct
from pathlib import Path

import pytest
import torch

import collide_cases as cc
import space_collide_cases as C
from exact_ballpark import vec_collide as VC
from exact_ballpark import vec_space as SP
from exact_ballpark import vec_server_space as VSS
from exact_ballpark.space import LEVEL_WIDTH

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
DEV, DEV2, HELD_OUT = 1, 7, 11
TOP = LEVEL_WIDTH[0]
_RUNS: dict = {}

# what the development seed reached (60 worlds x 16 steps; the minimums below are about half of it)
LAYER_MIN = {"ball_steps_with_collisions": 190, "events": 110, "answers": 140}
REF_MIN = {"nearby.other_bubble": 1800, "tick.dead_bubble": 140, "potential.recurse": 480, "potential.free": 160,
           "potential.light": 90}
SERVER_MIN = {"ship_ticks_with_collisions": 200, "narrow:CloakBall": 70, "single:RemoveBalls (own lists)": 20,
              "ticks_with_several_bubbles_narrowcasting": 35, "single:PackagedAddBallsToPark": 120}


def layer(seed: int, n: int = 60, steps: int = 16, device: str = "cpu", **kw) -> dict:
    k = ("layer", seed, n, steps, device, tuple(sorted(kw.items())))
    if k not in _RUNS:
        _RUNS[k] = C.compare_layer(C.cases(seed, n, steps), device, **kw)
    return _RUNS[k]


def server(seed: int, n: int = 60, ticks: int = 16, device: str = "cpu", **kw) -> dict:
    k = ("server", seed, n, ticks, device, tuple(sorted(kw.items())))
    if k not in _RUNS:
        res = C.run_server(C.server_cases(seed, n, ticks), device, **kw)
        res.pop("server")
        _RUNS[k] = res
    return _RUNS[k]


def assert_layer_exact(res: dict, minimum: bool = True, worlds: int = 60) -> None:
    assert res["diffs"] == [], res["diffs"][:3]
    assert len(res["flagged"]) <= worlds // 25, res["flagged"]
    assert res["compared"]["ball_fields"] > 0 and res["compared"]["collision_fields"] > 0
    if minimum:
        for k, v in LAYER_MIN.items():
            assert res["compared"].get(k, 0) >= v, (k, res["compared"])
        for k, v in REF_MIN.items():
            assert res["ref_hits"].get(k, 0) >= v, (k, res["ref_hits"])


def assert_server_exact(res: dict, minimum: bool = True, worlds: int = 60) -> None:
    assert res["sends"] == [] and res["rows"] == [], (res["sends"][:2], res["rows"][:3])
    assert len(res["flagged"]) <= worlds // 25, res["flagged"]
    if minimum:
        assert res["worlds_with_several_bubbles"] >= worlds // 3, res["worlds_with_several_bubbles"]
        for k, v in SERVER_MIN.items():
            assert res["coverage"].get(k, 0) >= v, (k, res["coverage"])


# ---- the layer over the collision park, on the CPU (the host path, the park's torch path) ----

def test_the_layer_over_the_collision_park_matches_the_master_development_worlds():
    assert_layer_exact(layer(DEV))


def test_the_layer_over_the_collision_park_matches_the_master_second_development_worlds():
    assert_layer_exact(layer(DEV2))


def test_the_layer_over_the_collision_park_matches_the_master_held_out_worlds():
    assert_layer_exact(layer(HELD_OUT))


def test_the_iterative_method_keeps_within_bubbles_too():
    """Destiny's iterative method (behind its switch) on the park's torch path with the layer: its candidate lists
    skip other bubbles' balls as the gradient's do (the reference's iterative passes reached and counted)."""
    res = layer(DEV, n=30, steps=12, iterative=True)
    assert_layer_exact(res, minimum=False, worlds=30)
    assert res["ref_hits"].get("iterative.found", 0) >= 1000 and res["ref_hits"].get("nearby.other_bubble", 0) >= 500


# ---- the server layer over the collision park, on the CPU ----

def test_the_server_layer_over_the_collision_park_matches_the_reference_development_worlds():
    assert_server_exact(server(DEV))


def test_the_server_layer_over_the_collision_park_matches_the_reference_second_development_worlds():
    assert_server_exact(server(DEV2))


def test_the_server_layer_over_the_collision_park_matches_the_reference_held_out_worlds():
    assert_server_exact(server(HELD_OUT))


# ---- Destiny's compiled code, directly (the conformance kit's space harness; skipped where it is not built) ----

HARNESS = pytest.mark.skipif(not C.HARNESS_PATH().is_file(), reason="Destiny space harness not built")


def harness_rows(seed: int, n: int = 60, steps: int = 16) -> tuple:
    k = ("harness", seed, n, steps)
    if k not in _RUNS:
        import space_cases as sc
        cs = C.cases(seed, n, steps)
        scen = C.scenarios(cs)
        _RUNS[k] = (cs, scen, sc.oracle(scen))
    return _RUNS[k]


@HARNESS
@pytest.mark.parametrize("seed", [DEV, HELD_OUT])
def test_the_master_reference_on_these_worlds_matches_compiled_destiny(seed):
    """The worlds as space_cases scenarios: the master SpaceBallpark against Destiny's own code compiled verbatim, every
    row kind (ball states with collision lists and filings, partition checksums, bubble dictionaries and sets, query
    answers, events), as tests/test_movement_space_oracle.py compares them."""
    import space_cases as sc
    from test_movement_space_oracle import differences
    cs, scen, want = harness_rows(seed)
    got, undefined = sc.run_reference(scen)
    assert not undefined and differences(want, got, undefined) == {}
    assert len(want["S"]) > 3000 and len(want["Q"]) > 100


@HARNESS
@pytest.mark.parametrize("seed", [DEV, HELD_OUT])
def test_the_layer_over_the_collision_park_matches_compiled_destiny(seed):
    """The batched layer over VecCollidePark (the host path, the park's torch path) against the harness's ball rows
    directly: bubbles, filings, positions, velocities, mLastC, mLastCollision, the collision lists, radii, flags."""
    cs, scen, want = harness_rows(seed)
    res = C.compare_harness(cs, "cpu", want=want)
    assert res["diffs"] == [] and res["missing"] == [] and res["rows"] > 3000, res["diffs"][:3]


@HARNESS
@CUDA
@pytest.mark.parametrize("seed", [DEV, HELD_OUT])
def test_the_layers_kernel_over_the_parks_kernel_matches_compiled_destiny(seed):
    cs, scen, want = harness_rows(seed)
    res = C.compare_harness(cs, "cuda", kernel="cuda", want=want)
    assert res["diffs"] == [] and res["missing"] == [] and res["rows"] > 3000, res["diffs"][:3]


# ---- the section 9, constructed ----

def _across_the_face(seeds: bool) -> dict:
    """Two massive interactive ships 100 m apart across the face x = TOP (radii 100 m: overlapping), and, with
    `seeds`, a small ship in the left top and one in the right top added first, so that the middle top (the left
    ship's) takes bubble 0 and the right top keeps bubble 1: the two ships in different bubbles; without, the right
    top takes the middle top's bubble: one bubble."""
    y, z = 0.3 * TOP, -0.4 * TOP
    ship = lambda bid, x, r: {"id": bid, "mass": 1e8, "radius": r, "max_vel": 100.0, "agility": 1.0, "sf": 1.0,  # noqa
                              "pos": [x, y, z], "vel": [0.0, 0.0, 0.0], "interactive": True, "massive": True,
                              "free": True}
    balls = [ship(11, -0.5 * TOP, 10.0), ship(12, 1.5 * TOP, 10.0)] if seeds else []
    balls += [ship(21, TOP - 50.0, 100.0), ship(22, TOP + 50.0, 100.0)]
    return {"id": 0, "kind": "row", "park": "collide", "balls": balls, "anchors": [(TOP, y, z)], "row": (TOP, y, z),
            "pair": (21, 22), "steps": [[], []], "ticks": []}


@pytest.mark.parametrize("seeds", [True, False])
def test_two_ships_across_a_top_face_collide_on_a_clients_view_and_on_the_master_only_within_a_bubble(seeds):
    w = _across_the_face(seeds)
    sw = C.SB.ScalarWorld(w)
    sw.evolve()
    want = {b.id: (b.bubble, tuple(b.collisions)) for b in sw.park.balls.values()}
    bw = C.CollideWorlds([w], "cpu")
    bw.step(0)
    got = C.batched_collisions(bw, 0)
    bub = {bid: int(bw.space.st["bubble"][0, j]) for (wi, bid), (r, j) in bw.index.items()}
    client = VC.VecCollidePark(1, len(w["balls"]))                        # the park alone: a client's view
    client.load([[{"id": b["id"], "mass": b["mass"], "radius": b["radius"], "max_vel": b["max_vel"],
                   "agility": b["agility"], "speed_fraction": b["sf"], "pos": b["pos"], "vel": b["vel"],
                   "free": True, "massive": True} for b in w["balls"]]])
    client.evolve()
    ids = client.id[0].tolist()
    seen = {ids[s]: tuple(client.hits[0, s, :int(client.hits_n[0, s])].tolist()) for s in range(client.N)}
    assert seen[21] == (22,) and seen[22] == (21,)                       # the client's view collides them
    if seeds:
        assert want[21] == (0, ()) and want[22] == (1, ())               # the master: other bubbles, no collision
    else:
        assert want[21][0] == want[22][0] and want[21][1] == (22,) and want[22][1] == (21,)
    for bid in (21, 22):
        assert (bub[bid], got[bid][2]) == want[bid]


class _OneBubble(VC.VecCollidePark):
    """The collision park on the master's path with every ball in bubble 0 and an empty skip."""

    def load(self, worlds):
        super().load(worlds)
        self.bubble = torch.zeros(self.W, self.N, dtype=torch.int64, device=self.dev)

    def evolve(self, active=None, skip=None):
        none = torch.zeros(self.W, self.N, dtype=torch.bool, device=self.dev)
        super().evolve(active, none if skip is None else skip)


def _bits(x):
    if isinstance(x, float):
        return struct.pack("<d", x).hex()
    if isinstance(x, (list, tuple)):
        return tuple(_bits(v) for v in x)
    if isinstance(x, dict):
        return {k: _bits(v) for k, v in x.items()}
    return x


def test_one_bubble_and_no_skip_is_the_clients_view_bit_for_bit():
    """With every ball in one bubble and none skipped, the master's filter changes nothing: the development worlds
    (both collision methods) on a park with `bubble` all 0 and an empty skip equal the client's view's park (bubble
    None) in every ball's state after every tick (so a park that never sets `bubble` runs as before)."""
    cases = cc.cases(1, n=120)
    a = cc.batched(cases, "cpu")
    b = cc.batched(cases, "cpu", park_cls=_OneBubble)
    assert len(a) > 1000 and _bits(a) == _bits(b)


def test_refusals():
    w = _across_the_face(True)
    bw = C.CollideWorlds([w], "cpu")
    bw.space.ops([[("REMOVE_BALL", 2)]])                                # the layer removes a ball the park keeps
    with pytest.raises(RuntimeError, match="RemoveBall"):
        bw.space.check()
    other = C.CollideWorlds([_across_the_face(False)], "cpu")
    with pytest.raises(ValueError, match="CollideView"):
        bw.space.tick(other.real)                                         # another park's tick under this layer
    with pytest.raises(ValueError, match="CUDA device"):
        SP.SpaceGraph(other.space, other.real)
    with pytest.raises(ValueError, match="AddBall's order|added"):
        VSS.SpaceCollideAdapter(other.real, added=[[21]])


# ---- planted faults (each must make its comparison differ) ----

def _filter_off(monkeypatch):
    monkeypatch.setattr(VC.VecCollidePark, "_same_bubble",
                        lambda self: torch.ones(self.W, self.R, self.N, dtype=torch.bool, device=self.dev))


def _dead_evolved(monkeypatch):
    orig = SP.evolve_skipping

    def no_skip(park, skip_rows):
        if isinstance(park, VC.VecCollidePark):
            park.evolve()
            return
        orig(park, skip_rows)
    monkeypatch.setattr(SP, "evolve_skipping", no_skip)


def _filing_unsynced(monkeypatch):
    monkeypatch.setattr(SP.CollideView, "sync_filing", lambda self, space: None)


def _bubble_unwritten(monkeypatch):
    monkeypatch.setattr(SP.CollideView, "bubble_out", lambda self: None)


FAULTS = {"bubble_filter_off": _filter_off, "dead_bubbles_evolved": _dead_evolved,
          "park_filing_not_the_layers": _filing_unsynced, "park_bubble_field_unwritten": _bubble_unwritten}


@pytest.mark.parametrize("fault", sorted(FAULTS) + ["rows_in_id_order", "iterative_filter_off"])
def test_a_planted_fault_is_caught(fault, monkeypatch):
    if fault == "rows_in_id_order":                                     # the bubble dictionary's order lost
        res = C.run_server(C.server_cases(DEV, 30, 12), "cpu", added=False)
        assert len(res["sends"]) + len(res["rows"]) > 20
        return
    if fault == "iterative_filter_off":
        _filter_off(monkeypatch)
        assert len(C.compare_layer(C.cases(DEV, 20, 10), "cpu", iterative=True)["diffs"]) > 20
        return
    FAULTS[fault](monkeypatch)
    lay = C.compare_layer(C.cases(DEV, 30, 12), "cpu")
    srv = C.run_server(C.server_cases(DEV, 30, 12), "cpu")
    assert len(lay["diffs"]) > 20, len(lay["diffs"])
    assert len(srv["sends"]) + len(srv["rows"]) > 20, (len(srv["sends"]), len(srv["rows"]))


# ---- on the card ----

@CUDA
@pytest.mark.parametrize("kernel", ["torch", "cuda"])
@pytest.mark.parametrize("seed", [DEV, HELD_OUT])
def test_the_layers_kernel_over_the_collision_park_matches_the_master(kernel, seed):
    """The layer's kernel (one thread a world) over the park's torch path on the card and over the kernel."""
    assert_layer_exact(layer(seed, n=40, steps=12, device="cuda", kernel=kernel), minimum=False, worlds=40)


@CUDA
def test_the_parks_kernel_equals_its_torch_path_under_the_layer():
    """Every tensor of the park (the collision lists to their lengths) and of the layer, every step: the park's kernel
    against its torch path, both on the card under the layer's kernel."""
    res = C.lockstep(C.cases(DEV, 40, 12), "cuda")
    assert res["diffs"] == [] and res["tensors"] > 800, res["diffs"][:5]          # 78 tensors x 12 steps


@CUDA
@pytest.mark.parametrize("path", ["kernel", "contact_slots"])
def test_a_cuda_graph_of_the_layer_and_the_collision_park_matches_the_master(path):
    """The layer's tick over the park (the kernel, or its torch path with contact_slots) captured once as a CUDA
    graph after the first step and replayed, the records applied eagerly between replays (copied into the graph's
    buffers): every compared value equal to the master's (so to eager ticks)."""
    kw = {"kernel": "cuda"} if path == "kernel" else {"kernel": "torch", "contact_slots": 5}
    res = C.compare_layer(C.cases(DEV, 40, 12), "cuda", graph=True, **kw)
    assert res["graph"]
    assert_layer_exact(res, minimum=False, worlds=40)


@CUDA
@pytest.mark.parametrize("kernel", ["torch", "cuda"])
@pytest.mark.parametrize("seed", [DEV, HELD_OUT])
def test_the_server_layer_over_the_collision_park_on_the_card_matches_the_reference(kernel, seed):
    assert_server_exact(server(seed, n=40, ticks=12, device="cuda", kernel=kernel), minimum=False, worlds=40)


@CUDA
def test_the_kernels_planted_fault_is_caught(monkeypatch, tmp_path):
    """the kernel built from a copy of its source without the master's bubble test: the layer's comparison differs."""
    from exact_ballpark import vec_collide_cuda as K
    line = ("    if (fn.bubble != fb.bubble) return false;      // the master skips another bubble's "
            "ball\n")
    src = K.SOURCE.read_text()
    assert src.count(line) == 1
    bad = Path(tmp_path) / "vec_collide_cuda_no_bubble_test.cu"
    bad.write_text(src.replace(line, ""))
    lib = K.load_library("sm90", source=bad)
    monkeypatch.setattr(K, "load_library", lambda *a, **k: lib)
    assert len(C.compare_layer(C.cases(DEV, 40, 12), "cuda", kernel="cuda")["diffs"]) > 20

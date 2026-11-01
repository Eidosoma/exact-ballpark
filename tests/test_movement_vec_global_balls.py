# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Global balls on Destiny's master: GetNearbyBalls' asymmetric rule (a candidate in another
bubble is skipped unless it is global: Partition.cpp:344) in the collision park (exact_ballpark.vec_collide's torch path,
both collision methods, and the kernel, its statics build) and in the partition layer's sensors
(exact_ballpark.vec_space), global balls left out of the bubble lists (PopulateAddDelForBubble, AdditionsAndDeletions),
and the server's make_ball_global and make_ball_local (exact_ballpark.vec_server_space.SpaceServer: SetBallGlobal at the
call, AddBallsToPark and BallNotGlobal's RemoveBall to every bubble with an interactive ball, the streams' global
flag), bit for bit against the master FullBallpark, the scalar server layer and Destiny's own code compiled verbatim,
on generated worlds (tests/global_cases.py: development seeds 1 and 7; held-out seed 13, first run after the
generator's freeze); a constructed world pins the asymmetry; the planted faults must be caught. The CPU cases run
everywhere; the rest need a GPU."""

import pytest
import torch

import global_cases as G
import space_collide_cases as C
from exact_ballpark import vec_collide as VC
from exact_ballpark import vec_server as VS
from exact_ballpark import vec_server_space as VSS
from exact_ballpark import vec_space as SP
from exact_ballpark.space import LEVEL_WIDTH

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible GPU")
HARNESS = pytest.mark.skipif(not C.HARNESS_PATH().is_file(), reason="Destiny space harness not built")
DEV, DEV2, HELD_OUT = 1, 7, 13
TOP = LEVEL_WIDTH[0]
_RUNS: dict = {}

# what development seed 1 reached (60 worlds x 16 steps, 14 server ticks; the minimums below are about half of it)
LAYER_MIN = {"collision_entries_across_bubbles": 40, "sensor_entries_across_bubbles": 50,
             "nearby.global_other_bubble": 180}
SERVER_MIN = {"global: AddBallsToPark narrowcast entries": 140, "global: RemoveBall narrowcast entries": 40,
              "collision_entries_across_bubbles": 30}


def layer(seed: int, device: str = "cpu", n: int = 60, steps: int = 16, **kw) -> dict:
    k = ("layer", seed, device, n, steps, tuple(sorted(kw.items())))
    if k not in _RUNS:
        _RUNS[k] = G.compare_layer(G.cases(seed, n, steps), device, **kw)
    return _RUNS[k]


def server(seed: int, device: str = "cpu", n: int = 60, ticks: int = 14, **kw) -> dict:
    k = ("server", seed, device, n, ticks, tuple(sorted(kw.items())))
    if k not in _RUNS:
        res = G.run_server(G.server_cases(seed, n, ticks), device, **kw)
        res.pop("server")
        _RUNS[k] = res
    return _RUNS[k]


def assert_layer_exact(res: dict, minimum: bool = False, worlds: int = 60) -> None:
    assert res["diffs"] == [], res["diffs"][:3]
    assert len(res["flagged"]) <= worlds // 25, res["flagged"]
    assert res["compared"]["global_ball_steps"] > 0 and res["compared"]["ball_fields"] > 0
    assert res["reach"]["nearby.global_other_bubble"] > 0 and res["reach"]["sensor_entries_across_bubbles"] > 0
    if minimum:
        for key, low in LAYER_MIN.items():
            assert res["reach"][key] >= low, (key, res["reach"][key])


def assert_server_exact(res: dict, minimum: bool = False, worlds: int = 60) -> None:
    assert res["sends"] == [] and res["rows"] == [], (res["sends"][:2], res["rows"][:2])
    assert len(res["flagged"]) <= worlds // 25, res["flagged"]
    assert res["coverage"].get("global: AddBallsToPark narrowcast entries", 0) > 0
    assert res["coverage"].get("global: RemoveBall narrowcast entries", 0) > 0
    if minimum:
        for key, low in SERVER_MIN.items():
            assert res["coverage"].get(key, 0) >= low, (key, res["coverage"].get(key, 0))


# ---- the generated worlds on the CPU ----

def test_the_layer_over_the_collision_park_matches_the_master_development_worlds():
    assert_layer_exact(layer(DEV), minimum=True)


def test_the_layer_over_the_collision_park_matches_the_master_second_development_worlds():
    assert_layer_exact(layer(DEV2))


def test_the_iterative_method_and_fixed_contact_slots_keep_the_rule_too():
    assert_layer_exact(layer(DEV, iterative=True))
    res = layer(DEV, contact_slots=10)
    assert_layer_exact(res)
    assert res["reach"]["collision_entries_across_bubbles"] > 0


def test_the_server_layer_matches_the_reference_development_worlds():
    assert_server_exact(server(DEV), minimum=True)


def test_the_server_layer_matches_the_reference_second_development_worlds():
    assert_server_exact(server(DEV2))


def test_held_out_the_layer_and_the_server_layer_match_the_references():
    """Held-out seed 13 (its first run after the generator's freeze)."""
    res = layer(HELD_OUT)
    assert_layer_exact(res)
    assert res["reach"]["collision_entries_across_bubbles"] > 0
    assert_server_exact(server(HELD_OUT))


# ---- the asymmetry, constructed ----

def _across(global_id: int | None) -> dict:
    """the two massive ships 100 m apart across the face x = TOP between bubbles 0 and 1 (radii 100 m:
    overlapping; tests/test_movement_vec_space_collide.py's _across_the_face with its seeds), one of them global."""
    y, z = 0.3 * TOP, -0.4 * TOP
    ship = lambda bid, x, r: {"id": bid, "mass": 1e8, "radius": r, "max_vel": 100.0, "agility": 1.0, "sf": 1.0,  # noqa
                              "pos": [x, y, z], "vel": [0.0, 0.0, 0.0], "interactive": True, "massive": True,
                              "free": True, "global": bid == global_id}
    balls = [ship(11, -0.5 * TOP, 10.0), ship(12, 1.5 * TOP, 10.0), ship(21, TOP - 50.0, 100.0),
             ship(22, TOP + 50.0, 100.0)]
    return {"id": 0, "kind": "row", "park": "collide", "balls": balls, "anchors": [(TOP, y, z)], "row": (TOP, y, z),
            "pair": (21, 22), "steps": [[], []], "ticks": []}


@pytest.mark.parametrize("kernel", ["torch", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("global_id", [22, 21, None])
def test_a_global_ship_across_the_face_is_met_by_the_other_ship_and_meets_nothing(kernel, global_id):
    """On the master the ship beside a global ship of another bubble takes it as a candidate and collides with it;
    the global ship's own candidates are its bubble's balls, so it collides with nothing; without a global ship
    neither collides (the case). The batched layer over the park equals the master FullBallpark."""
    w = _across(global_id)
    sw = G.ScalarWorld(w)
    sw.evolve()
    want = {b.id: (b.bubble, tuple(b.collisions)) for b in sw.park.balls.values()}
    other = {22: 21, 21: 22}.get(global_id)
    if global_id is None:
        assert want[21] == (0, ()) and want[22] == (1, ())
    else:
        assert want[other] == (want[other][0], (global_id,)) and want[global_id][1] == ()
        assert want[21][0] != want[22][0]
    bw = G.GlobalWorlds([w], "cuda" if kernel == "cuda" else "cpu", kernel=kernel)
    bw.step(0)
    got = C.batched_collisions(bw, 0)
    bub = {bid: int(bw.space.st["bubble"][0, j]) for (wi, bid), (r, j) in bw.index.items()}
    for bid in (21, 22):
        assert (bub[bid], got[bid][2]) == want[bid]


def test_a_park_without_global_balls_holds_the_tensors_it_held_before():
    """A park made without the option has no global flag (every tensor as before: snapshots and graph pins
    enumerate them); one made with it holds exactly one more tensor, `glob`."""
    def tensors(p):
        return {k: (tuple(v.shape), v.dtype) for k, v in vars(p).items() if torch.is_tensor(v)}
    a, b = VC.VecCollidePark(3, 4), VC.VecCollidePark(3, 4, global_balls=True)
    ta, tb = tensors(a), tensors(b)
    assert "glob" not in ta and set(tb) - set(ta) == {"glob"} and tb["glob"] == ((3, 4), torch.bool)
    assert {k: v for k, v in tb.items() if k != "glob"} == ta
    assert not a.global_balls and SP.CollideView(a).global_flags() is None


def test_refusals():
    w = _across(22)
    plain = VC.VecCollidePark(1, 4)
    with pytest.raises(ValueError, match="global_balls"):
        plain.load([[{k: v for k, v in b.items() if k in ("id", "mass", "radius", "max_vel", "pos", "vel", "free",
                                                          "massive", "global")} for b in w["balls"]]])
    with pytest.raises(ValueError, match="global_balls"):
        plain.set_global(torch.ones(1, 4, dtype=torch.bool), True)
    cs = G.server_cases(DEV, 2, 1)
    srv, _ = C.build_server(cs, "cpu")                                  # a park made without global balls
    one = torch.tensor([True, True])
    ball = torch.tensor([cs[0]["balls"][0]["id"], cs[1]["balls"][0]["id"]])
    with pytest.raises(ValueError, match="global_balls"):
        srv.push("AddBallsToPark", one, ball)
    gs, _ = G.build_server(cs, "cpu")
    gs.push("AddBallsToPark", one, ball)        # (a ball not in the park, its empty stream, and the
    gs.pre_tick(snapshot=True)                   # client's view are modelled now: test_movement_vec_global_edges.py)
    gs.evolve()
    gs.post_tick()
    view = VSS.SpaceClientView(gs, 0)
    with pytest.raises(ValueError, match="set_state"):                  # global balls held: its SetState first
        view.apply_batch(gs.batch)
    assert not (VS.GLOBAL_KINDS & srv.a.kinds)                          # no adapter takes the kinds
    from exact_ballpark.vec_modes import VecModesPark
    p = VecModesPark(1, device="cpu")
    assert not (VS.GLOBAL_KINDS & VS.ModesAdapter(p).kinds)


# ---- planted faults (each must make its comparison differ) ----

def _rule_ignored(monkeypatch):
    monkeypatch.setattr(VC.VecCollidePark, "_same_bubble",
                        lambda self: self.bubble[:, :self.R, None] == self.bubble[:, None, :])


def _rule_symmetric(monkeypatch):
    monkeypatch.setattr(VC.VecCollidePark, "_same_bubble",
                        lambda self: ((self.bubble[:, :self.R, None] == self.bubble[:, None, :])
                                      | self.glob[:, None, :] | self.glob[:, :self.R, None]))


def _remove_to_its_own_bubble(monkeypatch):
    pre = VSS.SpaceServer.pre_tick

    def faulty(self, snapshot=None):
        pre(self, snapshot)
        for j, s in enumerate(self.batch.slots):
            if s.kind == VS.GLOBAL_OFF:
                self.batch.rbubble[:, j] = -1
    monkeypatch.setattr(VSS.SpaceServer, "pre_tick", faulty)


def _stream_without_flag(monkeypatch):
    pack = VS._pack_ball
    monkeypatch.setattr(VS, "_pack_ball", lambda fl, r, o: pack({k: v for k, v in fl.items() if k != "global"}, r, o))


def _no_duplicate_test(monkeypatch):
    push = VSS.SpaceServer._push_global

    def faulty(self, k, mask, ball):
        self._reset_last()
        return push(self, k, mask, ball)
    monkeypatch.setattr(VSS.SpaceServer, "_push_global", faulty)


FAULTS = {"rule_ignored": ("layer", _rule_ignored), "rule_symmetric": ("layer", _rule_symmetric),
          "remove_to_its_own_bubble": ("server", _remove_to_its_own_bubble),
          "stream_without_global_flag": ("server", _stream_without_flag),
          "no_duplicate_test": ("server", _no_duplicate_test)}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_a_planted_fault_is_caught(fault, monkeypatch):
    """The torch path's rule ignored or made symmetric (the responder's flag read too), BallNotGlobal's RemoveBall sent
    to the bubble it names as well, the streams without DSTBALL_ISGLOBAL, the global notices without TIME-710's
    duplicate test: each is seen (development seed 1)."""
    level, plant = FAULTS[fault]
    plant(monkeypatch)
    if level == "layer":
        res = G.compare_layer(G.cases(DEV, 60, 16), "cpu")
        assert len(res["diffs"]) > 0
    else:
        res = G.run_server(G.server_cases(DEV, 60, 14), "cpu")
        assert len(res["sends"]) + len(res["rows"]) > 0


SPACE_FAULTS = {
    "sensor_rule_ignored": ("if (c.s->bubble[k] != b_bubble && !is_global(c, r)) {",
                            "if (c.s->bubble[k] != b_bubble) {"),
    "bubble_additions_list_global": ("if (!here || c.p->cloaked[rj] || is_global(c, rj)) continue;",
                                     "if (!here || c.p->cloaked[rj]) continue;"),
    "mover_lists_list_global": ("if (cl || gl) continue;", "if (cl) continue;"),
}


@pytest.mark.parametrize("fault", sorted(SPACE_FAULTS))
def test_the_layers_planted_fault_is_caught(fault, monkeypatch, tmp_path):
    """Each fault, built into a copy of vec_space.cu (one change; the host path): the sensor's rule ignored (a global
    ball of another bubble not sensed), a global ball in a bubble's additions, or in a moving ship's own lists: each
    is seen by the layer's or the server's comparison (development seed 1)."""
    old, new = SPACE_FAULTS[fault]
    src = SP.SOURCE.read_text()
    assert src.count(old) == 1
    copy = tmp_path / f"{fault}.cu"
    copy.write_text(src.replace(old, new))
    lib = SP.load_library("ptx", source=copy)
    if fault == "sensor_rule_ignored":
        assert len(G.compare_layer(G.cases(DEV, 60, 16), "cpu", lib=lib)["diffs"]) > 0
    else:
        monkeypatch.setattr(SP, "load_library", lambda *a, **k: lib)
        res = G.run_server(G.server_cases(DEV, 60, 14), "cpu")
        assert len(res["sends"]) + len(res["rows"]) > 0


# ---- Destiny's compiled code, directly (the conformance kit's space harness; skipped where it is not built) ----

def harness_rows(seed: int, n: int = 60, steps: int = 16) -> tuple:
    k = ("harness", seed, n, steps)
    if k not in _RUNS:
        import space_cases as sc
        cs = G.cases(seed, n, steps)
        scen = G.scenarios(cs)
        _RUNS[k] = (cs, scen, sc.oracle(scen))
    return _RUNS[k]


@HARNESS
@pytest.mark.parametrize("seed", [DEV, HELD_OUT])
def test_the_master_reference_on_these_worlds_matches_compiled_destiny(seed):
    """The worlds as space_cases scenarios (AddBall's global flag, SetBallGlobal as GLOBAL_OP): the master
    SpaceBallpark against Destiny's own code compiled verbatim, every row kind."""
    import space_cases as sc
    from test_movement_space_oracle import differences
    cs, scen, want = harness_rows(seed)
    got, undefined = sc.run_reference(scen)
    assert not undefined and differences(want, got, undefined) == {}
    ops = sum(1 for s in scen for r in s["records"] if r[0] == "K" and r[2] == sc.GLOBAL_OP)
    assert len(want["S"]) > 3000 and ops > 30


@HARNESS
@pytest.mark.parametrize("seed", [DEV, HELD_OUT])
def test_the_layer_over_the_collision_park_matches_compiled_destiny(seed):
    """The batched layer over VecCollidePark made with global balls (the host path, the park's torch path) against
    the harness's ball rows directly, the global flag among the fields."""
    cs, scen, want = harness_rows(seed)
    res = G.compare_harness(cs, "cpu", want=want)
    assert res["diffs"] == [] and res["missing"] == [] and res["rows"] > 3000, res["diffs"][:3]


# ---- on the card: the kernel (the statics build), the torch path, CUDA graphs, the server ----

@CUDA
@pytest.mark.parametrize("seed", [DEV, DEV2])
def test_the_kernel_and_the_torch_path_on_the_card_match_the_master(seed):
    assert_layer_exact(layer(seed, "cuda", kernel="cuda"), minimum=seed == DEV)
    assert_layer_exact(layer(seed, "cuda", kernel="cuda", contact_slots=10))
    assert_layer_exact(layer(seed, "cuda", kernel="torch"))


@CUDA
def test_the_kernel_and_the_torch_path_in_lockstep():
    """Every compared tensor of the park (the global flags among them) and of the layer, after every step, equal
    between the kernel and the torch path on the card (float64 by their bits)."""
    res = G.lockstep(G.cases(DEV, 60, 16), "cuda")
    assert res["diffs"] == [] and res["tensors"] > 1000, res["diffs"][:5]


@CUDA
@pytest.mark.parametrize("path", ["kernel", "contact_slots"])
def test_a_cuda_graph_with_global_balls_matches_the_master(path):
    kw = {"kernel": "cuda"} if path == "kernel" else {"kernel": "torch", "contact_slots": 10}
    res = G.compare_layer(G.cases(DEV2, 40, 12), "cuda", graph=True, **kw)
    assert res["graph"]
    assert_layer_exact(res, worlds=40)


@CUDA
@pytest.mark.parametrize("kernel", ["torch", "cuda"])
def test_the_server_layer_on_the_card_matches_the_reference(kernel):
    assert_server_exact(server(DEV, "cuda", kernel=kernel))


KERNEL_FAULTS = {
    "kernel_rule_ignored": ("    if (fn.bubble != fb.bubble && !(P.glob != nullptr && P.glob[j])) return false;",
                            "    if (fn.bubble != fb.bubble) return false;"),
    "kernel_rule_xor": ("    if (fn.bubble != fb.bubble && !(P.glob != nullptr && P.glob[j])) return false;",
                        "    if ((fn.bubble != fb.bubble) != (P.glob != nullptr && P.glob[j] != 0)) return false;"),
}


@CUDA
@pytest.mark.parametrize("fault", sorted(KERNEL_FAULTS))
def test_planted_kernel_faults_are_caught(monkeypatch, tmp_path, fault):
    """Each fault, built into a copy of the statics kernel's source (one change), is seen by the comparison with the
    master (development seed 1): the global exception dropped, or the rule turned into an exclusive or (a global
    candidate of the responder's own bubble skipped)."""
    from exact_ballpark import vec_collide_cuda as K
    old, new = KERNEL_FAULTS[fault]
    src = K.SOURCE.read_text()
    assert src.count(old) == 1
    copy = tmp_path / f"{fault}.cu"
    copy.write_text(src.replace(old, new))
    lib = K.load_library("sm90", statics=True, source=copy)
    monkeypatch.setattr(K, "load_library", lambda *a, **k: lib)
    assert len(G.compare_layer(G.cases(DEV, 60, 16), "cuda", kernel="cuda")["diffs"]) > 0


@CUDA
def test_held_out_the_kernel_matches_the_master():
    """Held-out seed 13 on the card (its first card run after the generator's freeze): the kernel and the server."""
    assert_layer_exact(layer(HELD_OUT, "cuda", kernel="cuda"))
    assert_server_exact(server(HELD_OUT, "cuda", kernel="cuda"))


@HARNESS
@CUDA
@pytest.mark.parametrize("seed", [DEV, HELD_OUT])
def test_the_layers_kernel_over_the_parks_kernel_matches_compiled_destiny(seed):
    cs, scen, want = harness_rows(seed)
    res = G.compare_harness(cs, "cuda", kernel="cuda", want=want)
    assert res["diffs"] == [] and res["missing"] == [] and res["rows"] > 3000, res["diffs"][:3]

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""Destiny's flight modes on the GPU (exact_ballpark.vec_modes, VecModesPark: GOTO, STOP, FOLLOW, ORBIT under both laws
and WARP with their transitions, the setters and dynamical orientation, one CUDA kernel a tick, exact_ballpark.
vec_modes_cuda) against the scalar reference, bit for bit: doubles by their bits, two NaNs equal, integers and the
follower sets exactly. No tolerance anywhere.

- ModeBallpark (exact_ballpark.modes) on tests/mode_cases.py's scenarios: FOLLOW, ORBIT (both laws), WARP with
  followers, commands during warp, EntityWarpIn, the guards, mixed worlds; every ball every step (22 fields).
- FullBallpark (exact_ballpark.full) on tests/flight_cases.py's scenarios: the modes with dynamical orientation on and
  off and every setter in flight, and on mode_cases' scenarios with dynamical orientation (54 fields of full_cases'
  S rows). The reference rows are the ones tests/test_movement_modes_oracle.py, tests/test_movement_full_oracle.py and
  tests/test_movement_turn_oracle.py hold to Destiny's own code compiled verbatim; a sample is held to the compiled
  code here directly.
- Development cases are seed 1; held-out cases (seed 2) are separate tests, first run after the generator and the
  kernel were frozen.
- A world's result does not depend on the batch (alone, among others, tiled to 262,144 balls, any block size), and a
  tick replayed from a CUDA graph is the eager tick.
- The kernel's double sin, cos, log and exp equal exact_ballpark.dlibm, exact_ballpark.vec_exp and the C library.
- Planted faults must each be caught.

The CPU part (the generated header, the reference's literals, the source scan, the build, the park's argument checks,
the commands against the reference, the scenarios' coverage, and the tick on tensors, `evolve` on the
CPU, against ModeBallpark and FullBallpark on the same scenarios) runs everywhere; the rest skips where no GPU is
visible.
"""

from __future__ import annotations

import ast
import math
import random
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch

import flight_cases as flc
import full_cases as fc
import mode_cases as mc
import modes_batch as mb
from exact_ballpark import dlibm as D
from exact_ballpark import modes as MO
from exact_ballpark import ref as R
from exact_ballpark import vec_exp as E
from exact_ballpark import vec_modes as V
from exact_ballpark import vec_modes_cuda as K
from exact_ballpark import vec_orient_cuda as KO

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
try:
    KO._nvcc()
    NVCC = True
except RuntimeError:
    NVCC = False
needs_nvcc = pytest.mark.skipif(not NVCC, reason="nvcc not found")
F64 = torch.float64
_REF: dict = {}


def d64(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def same_bits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.view(torch.int64) == b.view(torch.int64)) | (torch.isnan(a) & torch.isnan(b))


# ============================================================================== the CPU part: header, scan, build

def parse_header(h: str) -> tuple:
    defs = {k: int(v, 16) for k, v in re.findall(r"#define (\w+) 0x([0-9a-f]+)(?:ULL|u)\n", h)}
    arrays = {k: [int(w, 16) for w in re.findall(r"0x([0-9a-f]+)", body)]
              for k, body in re.findall(r"(\w+)\[\d+\] = \{(.*?)\};", h)}
    ints = {k: int(v) for k, v in re.findall(r"#define (\w+) (\d+)\n", h)}
    return defs, arrays, ints


def test_the_generated_header_is_dlibms_and_the_references_constants():
    defs, arrays, ints = parse_header(K.tables_header())
    dd = lambda k: d64(defs[k])
    for name in ("TAYLOR_LIMIT", "BIG", "TOINT", "S1", "S2", "S3", "S4", "S5", "CS2", "CS4", "CS6", "SN3", "SN5", "HPINV",
                 "MP1", "MP2", "PP3", "PP4", "HP0", "HP1", "BR_TM600", "BR_SPLIT", "BR_TM24", "BR_BIG", "BR_BIG1",
                 "BR_TOVERP0", "BR_HP0", "BR_HP1", "BR_MP1", "BR_MP2", "LOG_TWO27", "LOG_TWO52", "LOG_LN2HI",
                 "LOG_LN2LO"):
        assert dd(f"DL_{name}") == getattr(D, name), name
    for name in ("K_TINY_SIN", "K_TINY_COS", "K_DO_SIN", "K_MID", "K_REDUCE", "K_HUGE", "BR_T576_HI", "LOG_NEG_LO",
                 "LOG_RANGE", "LOG_NEG_OFF", "LOG_NEG_52"):
        assert defs[f"DL_{name}"] == getattr(D, name), name
    assert (ints["DL_BR_EXP_LIMIT"], ints["DL_BR_EXP_BIAS"]) == (D.BR_EXP_LIMIT, D.BR_EXP_BIAS)
    assert [d64(w) for w in arrays["DL_SINCOSTAB"]] == list(D.SINCOSTAB)
    assert [d64(w) for w in arrays["DL_BR_TOVERP"]] == list(D.BR_TOVERP)
    assert [d64(w) for w in arrays["DL_LOG_A"]] == list(D.LOG_A)
    assert [d64(w) for w in arrays["DL_LOG_B"]] == list(D.LOG_B)
    assert [d64(w) for w in arrays["DL_LOG_INVC"]] == [c for c, _ in D.LOG_TAB]
    assert [d64(w) for w in arrays["DL_LOG_LOGC"]] == [c for _, c in D.LOG_TAB]
    assert (dd("M_AU"), dd("M_ORBITAL_PRECESSION"), dd("M_MIN_WARP_DISTANCE_SQ"), dd("M_STOP_DAMP")) == (
        MO.AU, MO.ORBITAL_PRECESSION, MO.MIN_WARP_DISTANCE_SQ, R.STOP_DAMP)
    assert (dd("M_WARP_AU_PER_SECOND"), dd("M_WARP_ACCELERATION"), dd("M_WARP_DECELERATION")) == (
        MO.WARP_FACTOR_TO_AU_PER_SECOND, MO.WARP_FACTOR_TO_ACCELERATION, MO.WARP_FACTOR_TO_DECELERATION)
    for name, v in K.MODE_LITERALS.items():
        assert dd(name) == v, name
    assert {k: v for k, v in ints.items() if k.startswith(("MODE_", "M_"))} == {
        "MODE_GOTO": R.GOTO, "MODE_STOP": R.STOP, "MODE_FOLLOW": MO.FOLLOW, "MODE_WARP": MO.WARP, "MODE_ORBIT": MO.ORBIT,
        "MODE_TROLL": MO.TROLL, "MODE_RIGID": MO.RIGID, "M_MAX_ALIGN_TICKS": MO.MAX_ALIGN_TICKS}   # TROLL, RIGID


def _float_literals(path: Path) -> set:
    """Every numeric literal of a source, a negated one (-0.999) with its sign."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            out.add(node.value)
        elif (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)
              and isinstance(node.operand.value, (int, float))):
            out.add(-node.operand.value)
    return out


def test_every_literal_of_the_header_is_one_of_the_references():
    """MODE_LITERALS are typed once in vec_modes_cuda.py: each must be a numeric literal of the reference's source."""
    here = Path(MO.__file__).parent
    found = _float_literals(here / "modes.py") | _float_literals(here / "full.py") | _float_literals(here / "ref.py")
    assert not [k for k, v in K.MODE_LITERALS.items() if v not in found]


def test_a_perturbed_header_differs_in_one_bit_of_one_entry():
    _, ta, _ = parse_header(K.tables_header())
    _, tb, _ = parse_header(K.tables_header({"DL_SINCOSTAB": 202}))
    diff = [(k, i) for k in ta for i, (x, y) in enumerate(zip(ta[k], tb[k])) if x != y]
    assert diff == [("DL_SINCOSTAB", 202)]
    with pytest.raises(KeyError):
        K.tables_header({"DL_NONE": 0})


def test_the_sources_call_no_function_of_cudas_math_library():
    assert KO.scan_source(K.SOURCE.read_text()) == []
    assert KO.scan_source(KO.SOURCE.read_text()) == []


@needs_nvcc
def test_the_build_pins_every_rounding():
    m = K.build()
    assert m["ptx_check"]["ok"] and not m["ptx_check"]["refused"] and not m["source_scan"]
    for flag in ("-fmad=false", "-prec-div=true", "-prec-sqrt=true", "-ftz=false"):
        assert flag in m["flags"]
    assert m["header_sha256"] == K._sha(K.tables_header().encode())
    assert m["orient_header_sha256"] == K._sha(KO.tables_header().encode())
    assert (m["source_sha256"], m["orient_source_sha256"]) == (K._sha(K.SOURCE.read_bytes()),
                                                              K._sha(KO.SOURCE.read_bytes()))
    ops = m["ptx_check"]["fp_ops"]
    assert ops.get("fma.rn.f64", 0) and ops.get("fma.rn.f32", 0)
    assert not any(k.startswith(("mad", "ex2", "lg2", "sin", "cos", "rsqrt")) or ".ftz" in k for k in ops)


def test_the_park_refuses_what_it_cannot_run():
    p = V.VecModesPark(2, device="cpu")
    with pytest.raises(ValueError):
        p.load([{"id": 1, "mass": 1e6, "radius": 1.0, "max_vel": 10.0, "pos": [0.0] * 3, "vel": [0.0] * 3,
                 "agility": 1.0, "speed_fraction": 1.0}])                               # two rows, one record
    rows = [{"id": i + 1, "mass": 1e6, "radius": 1.0, "max_vel": 10.0, "pos": [0.0] * 3, "vel": [0.0] * 3,
             "agility": 1.0, "speed_fraction": 1.0, "mode": mode} for i, mode in enumerate((R.STOP, MO.MISSILE))]
    with pytest.raises(ValueError):
        p.load(rows)                                                                 # MISSILE is not in this park
    rows[1]["mode"] = MO.BOID
    p.load(rows)
    with pytest.raises(ValueError, match="kernel library"):
        p.evolve(lib={})                                     # a kernel library for a park on the CPU


def test_the_kernel_and_its_graph_refuse_a_cpu_park_before_any_build(monkeypatch):
    """the kernel's tick and ModesGraph refuse a park on the CPU at once, naming the CPU path, and
    never reach the library's build (a missing nvcc is not what the user is told); on the CPU `evolve` is the tick on
    tensors, `evolve_torch`, bit for bit."""
    monkeypatch.setattr(K, "load_library", lambda *a, **k: pytest.fail("the library was loaded for a CPU park"))
    rows = [{"id": i + 1, "mass": 1e6, "radius": 1.0, "max_vel": 10.0, "pos": [100.0 * i, 0.0, 0.0],
             "vel": [1.0, 0.0, 0.0], "agility": 1.0, "speed_fraction": 1.0} for i in range(2)]
    p = V.VecModesPark(2, device="cpu")
    p.load(rows)
    with pytest.raises(ValueError, match="CUDA device.*evolve_torch"):
        K.tick(p)
    with pytest.raises(ValueError, match="CUDA device"):
        K.ModesGraph(p)
    q = V.VecModesPark(2, device="cpu")
    q.load(rows)
    one = torch.tensor([True, False])
    for park in (p, q):
        park.follow(one, torch.tensor([1, -1]), torch.tensor([50.0, 0.0], dtype=F64))
    for _ in range(3):
        p.evolve()
        q.evolve_torch()
    for name in p.row_tensors():
        assert bool(same_bits(getattr(p, name).to(F64), getattr(q, name).to(F64)).all()), name
    assert p.mode.tolist() == [MO.FOLLOW, R.STOP] and p.current_time == q.current_time == 3


# ---- the commands, on the CPU, against the reference (no tick) ----

def _command_state(park_rows: dict, ref: MO.ModeBallpark) -> tuple:
    out = []
    for bid in sorted(ref.balls):
        b = ref.balls[bid]
        out.append((bid, b.mode, b.effect_stamp, int(b.is_massive), int(b.sensor_active), b.follow_id, b.owner_id,
                    -1 if b.follow_ptr is None else b.follow_ptr, *b.pos, *b.vel, *b.goto, b.follow_range,
                    b.last_collision, b.speed_fraction, tuple(sorted(b.followers))))
    return tuple(out)


def test_the_commands_are_the_references_on_the_cpu():
    """Every mode_cases command of a scenario's first command step, applied by the park (torch, on the CPU) and by
    ModeBallpark, leaves the same state: modes, stamps, follow fields, goto points, speed fractions, followers."""
    cs = mc.cases(1)
    for c in cs[::3]:
        ref = MO.ModeBallpark(friction=c["friction"], tick_interval_ms=c["tick"], use_new_orbit=c["new_orbit"],
                              is_master=c["master"])
        ref.current_time = c["start"]
        for b in c["balls"]:
            ref.add_ball(b["id"], b["mass"], b["radius"], b["max_vel"], b["pos"], b["vel"], b["agility"], b["sf"],
                         is_free=b["free"], is_massive=b["massive"], is_interactive=b["interactive"],
                         is_cloaked=b["cloaked"], bubble=b["bubble"], is_moribund=b["moribund"], mode=b["mode"])
        park = V.VecModesPark(len(c["balls"]), friction=c["friction"], tick_interval_ms=c["tick"], device="cpu",
                              use_new_orbit=c["new_orbit"])
        park.load([{"id": b["id"], "world": 1, "time": c["start"], "mass": b["mass"], "radius": b["radius"],
                    "max_vel": b["max_vel"], "pos": b["pos"], "vel": b["vel"], "agility": b["agility"],
                    "speed_fraction": b["sf"], "free": b["free"], "massive": b["massive"],
                    "interactive": b["interactive"], "cloaked": b["cloaked"], "bubble": b["bubble"],
                    "moribund": b["moribund"], "mode": b["mode"]} for b in c["balls"]])
        index = {(c["id"], b["id"]): i for i, b in enumerate(c["balls"])}
        first = min(k[0] for k in c["cmds"])
        for k in (k for k in c["cmds"] if k[0] == first):
            _, op, a, bb, x, y, z, rr, w = k
            {mc.GOTO_OP: lambda: ref.goto_point(a, x, y, z), mc.DIR_OP: lambda: ref.goto_direction(a, x, y, z),
             mc.STOP_OP: lambda: ref.stop(a), mc.FOLLOW_OP: lambda: ref.follow_ball(a, bb, rr),
             mc.ORBIT_OP: lambda: ref.orbit(a, bb, rr), mc.WARP_OP: lambda: ref.warp_to(a, x, y, z, rr, w),
             mc.WARPIN_OP: lambda: ref.entity_warp_in(a, x, y, z, w)}[op]()
            mb._apply_modes(park, op, [(c["id"], k)], index)
        ids = park.ids.tolist()
        got = []
        for i in sorted(range(park.n), key=lambda i: ids[i]):
            ptr = int(park.follow_ptr[i])
            got.append((ids[i], int(park.mode[i]), int(park.effect_stamp[i]), int(park.massive[i]),
                        int(park.sensor[i]), int(park.follow_id[i]), int(park.owner_id[i]), -1 if ptr < 0 else ids[ptr],
                        *park.pos[i].tolist(), *park.vel[i].tolist(), *park.goto[i].tolist(),
                        float(park.follow_range[i]), float(park.last_collision[i]), float(park.speed_fraction[i]),
                        tuple(park.followers(i))))
        want = _command_state({}, ref)
        assert len(got) == len(want)
        for g, w_ in zip(got, want):
            assert all(fc.same(float(p), float(q)) if isinstance(q, float) else p == q for p, q in zip(g, w_)), (
                c["id"], g, w_)


# ---- the scenarios' coverage ----

MODE_BRANCHES = ["follow.coincident", "orbit_old.tangent", "orbit_old.inside", "orbit_old.real_part",
                 "orbit_new.on_target", "orbit_new.default_normal", "orbit_new.angle_clamp_low",
                 "orbit_new.angle_clamp_high", "orbit_new.dist_floor", "orbit_new.circle", "orbit_new.factor_floor",
                 "orbit_new.factor_log", "orbit_new.factor_linear", "gtf.orbit_thrust", "gtf.capped", "gtf.free",
                 "orbit_thrust.capped", "orbit_thrust.off", "warp.short_goto", "warp.repeat_noop", "warpin.warp",
                 "warpin.goto", "warp.align_ok", "warp.align_forced", "warp.align_wait", "release.goto_direction",
                 "release.stop", "warp.speed_limited", "warp.accel", "warp.speed_kept", "warp.cruise", "warp.decel",
                 "warp.drop_out", "warp.overrun"]
TURN_BRANCHES = ["orient.thrust", "orient.behind", "orient.coast", "torque.full", "roll.turn", "snap.at_rest",
                 "orbit_new.heading_normal"]


def _branches_by_scenario(run) -> Counter:
    seen = Counter()
    for c in run[0]:
        t = Counter()
        run[1](c, t)
        seen.update(set(t))
    return seen


def test_the_scenarios_reach_every_branch():
    """The reference's branch counters on the development scenarios: every branch of the modes reached by at least 3
    scenarios of mode_cases, and with dynamical orientation (flight_cases and mode_cases converted) by at least 3."""
    modes = _branches_by_scenario((mc.cases(1), lambda c, t: mc.reference(c, t)))
    assert not [b for b in MODE_BRANCHES if modes[b] < 3], {b: modes[b] for b in MODE_BRANCHES}
    flight = [c for c in flc.cases(1) if c["do"]] + [flc.from_mode_case(c, True) for c in mc.cases(1)[::4]]
    turned = _branches_by_scenario(([c for c in flight if c], lambda c, t: fc.reference(c, t)))
    low = [b for b in MODE_BRANCHES + TURN_BRANCHES if turned[b] < 3 and b not in ("warp.repeat_noop",)]
    assert not low, {b: turned[b] for b in low}


def test_the_flight_scenarios_set_every_value_in_flight():
    ops = Counter(rec[2] for c in flc.cases(1) for rec in c["records"] if rec[0] == "K")
    assert all(ops[op] >= 50 for op in flc.SETTER_OPS), ops
    kinds = Counter((c["kind"], c["do"], c["new_orbit"]) for c in flc.cases(1))
    assert all(kinds[(k, do, new)] >= 10 for k in ("follow", "orbit", "warp", "mixed") for do in (False, True)
               for new in (False, True)), kinds


# ======================================================================================= the reference, on the GPU

def reference(key: tuple, cs: list, full: bool) -> dict:
    if key not in _REF:
        _REF[key] = mb.reference_full(cs) if full else mb.reference_modes(cs)
    return _REF[key]


def flight_cases(seed: int, which: str) -> list:
    if which == "flight":
        return flc.cases(seed)
    return [c for c in (flc.from_mode_case(c, which == "mc_do") for c in mc.cases(seed)) if c]


@needs_cuda
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_kernel_matches_modeballpark(seed):
    cs = mc.cases(seed)
    bad = mb.differences(reference(("modes", seed), cs, False), mb.batched_modes(cs), mb.FIELDS_MODES)
    assert not bad, (len(bad), bad[:8])


@needs_cuda
@pytest.mark.parametrize("which", ["flight", "mc_do"])
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_kernel_matches_fullballpark(seed, which):
    cs = flight_cases(seed, which)
    bad = mb.differences(reference((which, seed), cs, True), mb.batched_full(cs), mb.FIELDS_FULL)
    assert not bad, (len(bad), bad[:8])


# ---- the tick on tensors (`evolve` on the CPU) against the same references, no tolerance ----

@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_tensor_tick_matches_modeballpark_on_the_cpu(seed):
    cs = mc.cases(seed)
    bad = mb.differences(reference(("modes", seed), cs, False), mb.batched_modes(cs, device="cpu"), mb.FIELDS_MODES)
    assert not bad, (len(bad), bad[:8])


@pytest.mark.parametrize("which", ["flight", "mc_do"])
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_tensor_tick_matches_fullballpark_on_the_cpu(seed, which):
    cs = flight_cases(seed, which)
    bad = mb.differences(reference((which, seed), cs, True), mb.batched_full(cs, device="cpu"), mb.FIELDS_FULL)
    assert not bad, (len(bad), bad[:8])


def test_a_tick_of_no_time_is_the_references_on_the_cpu():
    """with tick_interval_ms=0 Ballpark::Integrate returns at once (ref.py, the kernels); the tensor
    tick does too (fix_dt0): 40 development worlds of mode_cases with their tick set to 0 against ModeBallpark."""
    cs = [dict(c, tick=0) for c in mc.cases(1)[:40]]
    want = mb.reference_modes(cs)
    assert len(want) > 3000
    bad = mb.differences(want, mb.batched_modes(cs, device="cpu"), mb.FIELDS_MODES)
    assert not bad, (len(bad), bad[:8])


@needs_cuda
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_tensor_tick_on_the_card_matches_both_references(seed):
    """`evolve_torch` on a CUDA device (the cross-check of the kernel) against both references."""
    hook = lambda park: park.evolve_torch                                          # noqa: E731
    cs = mc.cases(seed)
    assert not mb.differences(reference(("modes", seed), cs, False), mb.batched_modes(cs, park_hook=hook),
                              mb.FIELDS_MODES)
    for which in ("flight", "mc_do"):
        fs = flight_cases(seed, which)
        assert not mb.differences(reference((which, seed), fs, True), mb.batched_full(fs, park_hook=hook),
                                  mb.FIELDS_FULL), which


# ===================================================== a world's result, alone, in any batch, at any width and block

def _sample(seed: int = 1) -> list:
    """Development worlds of one park group (Destiny's default friction and tick, the new law), every family."""
    return [c for c in mc.cases(seed) if c["friction"] == 1.0e6 and c["tick"] == 1000 and c["new_orbit"]]


@needs_cuda
def test_a_worlds_result_does_not_depend_on_the_batch_or_the_block():
    cs = _sample()[:60]
    together = mb.batched_modes(cs)
    assert together == mb.batched_modes(cs[::-1]) or not mb.differences(together, mb.batched_modes(cs[::-1]),
                                                                       mb.FIELDS_MODES)
    alone = {}
    for c in cs[:10]:
        alone.update(mb.batched_modes([c]))
    assert not mb.differences({k: v for k, v in together.items() if k in alone}, alone, mb.FIELDS_MODES)
    for block in (32, 256, 384):
        assert not mb.differences(together, mb.batched_modes(cs, block=block), mb.FIELDS_MODES), block


def _tiled(cs: list, reps: int) -> list:
    out = []
    for r_ in range(reps):
        for c in cs:
            out.append(dict(c, id=c["id"] + 100000 * r_))
    return out


@needs_cuda
@pytest.mark.parametrize("width", [100, 1000, 65536, 262144])
def test_a_worlds_result_is_the_same_at_every_width(width):
    """The development worlds tiled to about `width` balls in one park: every copy ends as the first (all its state, the
    last step), and the first as the reference."""
    cs = _sample()[:40]
    last = max(c["steps"] for c in cs)
    cs = [dict(c, steps=last) for c in cs]
    n_balls = sum(len(c["balls"]) for c in cs)
    reps = max(1, width // n_balls)
    got = mb.batched_modes(_tiled(cs, reps), record={last})
    first = {(cid, s, bid): v for (cid, s, bid), v in got.items() if cid < 100000}
    assert len(got) == reps * n_balls
    for (cid, s, bid), v in got.items():
        w_ = first[(cid % 100000, s, bid)]
        assert v == w_ or not mb.differences({0: w_}, {0: v}, mb.FIELDS_MODES), (cid, bid)
    want = {k: v for k, v in mb.reference_modes(cs).items() if k[1] == last}
    assert not mb.differences(want, first, mb.FIELDS_MODES)


# ================================================================================================ CUDA graphs

@needs_cuda
def test_a_tick_replayed_from_a_cuda_graph_is_the_eager_tick():
    """ModesGraph's replays against eager ticks, every command and setter applied between ticks, every field every
    step: mode_cases' worlds (the modes' bookkeeping) and flight_cases' (the setters, with dynamical orientation)."""
    cs = _sample()[:80]
    eager = mb.batched_modes(cs)
    graph = mb.batched_modes(cs, park_hook=lambda park: K.ModesGraph(park).tick)
    assert not mb.differences(eager, graph, mb.FIELDS_MODES)
    fs = [c for c in flc.cases(1) if c["friction"] == 1.0e6 and c["tick"] == 1000 and c["do"]][:60]
    eager = mb.batched_full(fs)
    graph = mb.batched_full(fs, park_hook=lambda park: K.ModesGraph(park).tick)
    assert not mb.differences(eager, graph, mb.FIELDS_FULL)


# ======================================================================= the double functions and the oracle

@needs_cuda
@pytest.mark.parametrize("name", ["sin", "cos", "log", "exp"])
def test_the_kernels_double_functions_are_dlibms_and_the_c_librarys(name):
    import test_movement_dlibm as TD
    xs = TD.arguments(11, 60000)
    x = torch.tensor(xs, dtype=F64, device="cuda")
    got = K.dlibm(name, x)
    if name == "exp":
        assert bool(same_bits(got, E.exp(x)).all())
        want = [TD_exp(v) for v in xs]
    else:
        assert bool(same_bits(got, torch.tensor([getattr(D, name)(v) for v in xs], dtype=F64, device="cuda")).all())
        want = [TD.c_value(name, v) for v in xs]
    assert bool(same_bits(got, torch.tensor(want, dtype=F64, device="cuda")).all())


def TD_exp(x: float) -> float:
    """The C library's exp: Python raises where C overflows to an infinity."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


ORACLE_FIELDS = mb.FIELDS_FULL


@needs_cuda
@pytest.mark.skipif(not fc.HARNESS.exists(), reason="the compiled full harness is not built")
def test_the_kernel_matches_destinys_compiled_code_on_a_sample():
    """A sample of the development scenarios run by Destiny's own code compiled verbatim (full_cases.HARNESS) and by
    the kernel, compared directly (no reference between them): every field the park holds, every ball every step."""
    cs = [c for c in flc.cases(1) if c["friction"] == 1.0e6][:150]
    cs += [c for c in flight_cases(1, "mc_do") if c["friction"] == 1.0e6][:150]
    idx = [fc.S_FIELDS.index(f) for f in ORACLE_FIELDS]
    want = {(row[0], row[1], row[2]): tuple(row[i] for i in idx) for row in fc.oracle(cs)["S"]}
    assert not mb.differences(want, mb.batched_full(cs), ORACLE_FIELDS)


@needs_cuda
@pytest.mark.skipif(not mc.HARNESS.exists(), reason="the compiled modes harness is not built")
def test_the_kernel_matches_destinys_compiled_modes_on_a_sample():
    cs = mc.cases(1)[::4]
    rows, _ = mc.oracle(cs)
    want = {(r_[0], r_[1], r_[2]): tuple(r_[3:]) for r_ in rows}
    assert not mb.differences(want, mb.batched_modes(cs), mb.FIELDS_MODES)


# ============================================================================================== planted faults

FAULTS = {          # name: (source edits (old, new, count), header perturbation, the checks that must catch it)
    # a multiply-add contracted (the integrator's first term fused, the x component)
    "contracted_fma": ([("pn = D3{dmul(dadd(dmul(m, dadd(dmul(dmul(m, a.x), tf1), dmul(k,",
                         "pn = D3{dmul(dadd(dmul(m, dfma(dmul(m, a.x), tf1, dmul(k,", 1)], None, ("modes",)),
    # CUDA's own sin for the new orbit's spiral (the source scan refuses it too; in the old orbit's precessing normal
    # its last bits would be truncated away to 7 decimals); since this port applied the sincos patch the
    # spiral's sine and cosine come from g_sincos, which the fault replaces
    "cuda_sin": ([("        const double dexp = dmul(dist, g_exp(dmul(angle, t)));\n        double st, ct;\n        g_sincos(t, &st, &ct);",
                   "        const double dexp = dmul(dist, g_exp(dmul(angle, t)));\n        double st, ct;\n        g_sincos(t, &st, &ct);\n        st = sin(t);", 1)],
                 None, ("modes",)),
    # CUDA's own log in the warp constants: refused by the source scan and by the PTX check (rcp.approx.ftz.f64); the
    # scenarios' warps reach no argument where CUDA's log and glibc's differ (one card run, 23 October 2026)
    "cuda_log": ([("const double t_a = ddiv(g_log(ddiv(s, acc)), acc);",
                   "const double t_a = ddiv(log(ddiv(s, acc)), acc);", 1)], None, ("build",)),
    # a warper's followers all released before their own turn, whatever their ids
    "release_order": ([("        if (s.ids[b.follow_ptr] < b.id) release_before = true;\n        else release_after = true;",
                        "        release_before = true;", 1)], None, ("modes",)),     # the lines as a later change left them
    # a new orbit reading its stopped target's velocity before the target's own damping, whatever their ids
    "target_damping": ([("                if (s.ids[t] < b.id) ov.y = s.vy_after[t];            // the target's turn came first\n",
                         "", 1)], None, ("modes",)),
    # the drop-out tick's extra integration from the drop-out point left out (TRV-714)
    "drop_out_step": ([("                b.vel = v;\n                integrate(&p, &v, a, m, s.friction, b.tf, s.dt);",
                        "                b.vel = v;", 1)], None, ("modes",)),
    # dynamical orientation's coast torque toward the warp's exit point left out
    "coast_torque": ([("    } else if (b.mode == MODE_WARP && b.effect_stamp >= 0) {",
                       "    } else if (false) {", 1)], None, ("turning",)),
    # one entry of the sine table wrong in its last bit (row 50's cosine: the device's sin and cos samples)
    "table_entry": ([], {"DL_SINCOSTAB": 202}, ("sincos",)),
    # exp's last fused step unfused (the device's exp samples, and the scenarios)
    "exp_unfused": ([("        return dfma(tmp, scale, scale);                               // scale + tmp scale",
                      "        return dadd(dmul(tmp, scale), scale);", 1)], None, ("exp", "modes")),
}


def faulted_library(name: str) -> dict:
    """The library built from the source with one planted fault (cached under the build directory)."""
    edits, perturb, _ = FAULTS[name]
    text = K.SOURCE.read_text()
    for old, new, count in edits:
        assert text.count(old) == count, (name, old)
        text = text.replace(old, new)
    d = K.build_dir() / "faults"
    d.mkdir(parents=True, exist_ok=True)
    src = d / f"{name}.cu"
    if not src.exists() or src.read_text() != text:
        src.write_text(text)
    return K.load_library(source=src, perturb=perturb, checks=False)


@needs_cuda
@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault):
    import test_movement_dlibm as TD
    lib = faulted_library(fault)
    if fault.startswith("cuda_"):
        assert lib["manifest"]["source_scan"], "the source scan missed it"
    for check in FAULTS[fault][2]:
        if check == "build":
            bad = [k for k, v in (("scan", lib["manifest"]["source_scan"]),
                                  ("ptx", not lib["manifest"]["ptx_check"]["ok"])) if v]
            assert len(bad) == 2, bad
        elif check == "modes":
            cs = [c for c in mc.cases(1) if c["friction"] == 1.0e6]
            bad = mb.differences(reference(("modes_fault", 1), cs, False), mb.batched_modes(cs, lib=lib),
                                 mb.FIELDS_MODES)
        elif check == "turning":
            cs = [c for c in flight_cases(1, "flight") if c["do"] and c["kind"] == "warp"][:120]
            bad = mb.differences(reference(("turning_fault", 1), cs, True), mb.batched_full(cs, lib=lib),
                                 mb.FIELDS_FULL)
        else:
            xs = TD.arguments(12, 40000) + [50 / 128 + i * 1e-6 for i in range(-2000, 2000)]
            x = torch.tensor(xs, dtype=F64, device="cuda")
            names = ("sin", "cos") if check == "sincos" else (check,)
            bad = [n for n in names if not bool(same_bits(K.dlibm(n, x, lib=lib), K.dlibm(n, x)).all())]
        assert bad, f"the planted fault {fault} went unnoticed by {check}"


@needs_cuda
def test_the_parks_warp_in_snap_is_caught_when_left_out(monkeypatch):
    """The park's EntityWarpIn without its SnapOrientation (the rotation kept through the call): the reference, which
    tests/test_movement_flight_oracle.py holds to the compiled code, differs."""
    whole = V.VecModesPark.entity_warp_in

    def no_snap(self, mask, point, warp_factor):
        keep = (self.rot, self.old_rot, self.ang_vel, self.old_ang_vel)
        whole(self, mask, point, warp_factor)
        self.rot, self.old_rot, self.ang_vel, self.old_ang_vel = keep

    monkeypatch.setattr(V.VecModesPark, "entity_warp_in", no_snap)
    cs = [c for c in flc.cases(1) if c["kind"] == "warpin" and c["do"]][:30]
    assert mb.differences(mb.reference_full(cs), mb.batched_full(cs), mb.FIELDS_FULL), "the omission went unnoticed"

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""the flight-mode and command leftovers on the GPU, and the sincos patch.

- SetBallFree, SetBallRigid, the Python SetBoid, SetBallTroll with PetrifyTroll and SetBallInteractive in the batched
  park (exact_ballpark.vec_modes.VecModesPark: its commands on tensors, its CUDA kernel a tick) against the scalar
  reference (exact_ballpark.full.FullBallpark) on tests/leftover_cases.py's scenarios: every field of every ball every
  step (tests/leftover_batch.py), the follower sets as StopAllFollowers acts on them (the reference's less its stale
  entries, counted), on a development seed and on held-out seeds drawn after the generator was frozen; a sample of the
  same scenarios run by Destiny's own code compiled verbatim (full_cases.HARNESS) against the reference.
- GetCenterDist and GetSurfaceDist batched (exact_ballpark.vec_distance) against the reference's queries, on the CPU and
  on the card, and captured in a CUDA graph.
- the sincos patch: compiled Destiny takes the sine and cosine of one argument from glibc's sincos at exactly the
  six call sites the reference and the kernel now take them from it (DefaultOrbitalNormal, EvolveOldStyleOrbit and
  EvolveNewStyleOrbit, two each, in every compiled harness), and nowhere else; the kernel's g_sincos equals glibc's
  sincos and exact_ballpark.dlibm.sincos bit for bit, at x = +-0x1.ad1fb54442d18p+0 too, where glibc's sin differs.
- Planted faults (a member's follow fields cleared at its release, the old follower test, the troll's petrification a
  tick early or never, SetBoid with a Stop, SetBallFree keeping mLastG, the centre distance's squares summed right to
  left) are caught; a CUDA graph replays the eager ticks.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch

import full_cases as fc
import leftover_batch as lb
import leftover_cases as lc
from exact_ballpark import dlibm as D
from exact_ballpark import full as FU
from exact_ballpark import modes as MO
from exact_ballpark import vec_distance as VD
from exact_ballpark import vec_modes as V
from exact_ballpark import vec_modes_cuda as K

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
F64, I64 = torch.float64, torch.int64
from kit import harness as kit_harness  # noqa: E402  (the conformance kit's paths: conformance/kit.toml)
HARNESSES = {"modes": kit_harness("modes"), "full": fc.HARNESS, "space": kit_harness("space")}
_REF: dict = {}


def reference(seed: int) -> tuple:
    if seed not in _REF:
        cs = lc.cases(seed)
        _REF[seed] = (cs, *lb.reference(cs))
    return _REF[seed]


def same_bits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.contiguous().view(I64) == b.contiguous().view(I64)) | (torch.isnan(a) & torch.isnan(b))


# =========================================================================================== on the CPU, no tick

def test_the_constants_are_the_references():
    assert V.TROLL_MIN_DELAY == FU.TROLL_MIN_DELAY
    defs = dict(re.findall(r"#define (MODE_\w+) (\d+)", K.tables_header()))
    assert (int(defs["MODE_TROLL"]), int(defs["MODE_RIGID"])) == (MO.TROLL, MO.RIGID)


def test_the_commands_and_the_distances_are_the_references_on_the_cpu():
    """Every record of the development scenarios, without a tick (the scenarios marked manual: the reference does not
    evolve), on a CPU park and on the reference: every field of every ball after every step's commands, and every
    GetCenterDist/GetSurfaceDist answer, bit for bit."""
    cs = [dict(c, manual=True) for c in lc.cases(1)]
    want, wq, _, undefined = lb.reference(cs)
    assert not undefined
    got, gq = lb.batched(cs, device="cpu", ticks=False)
    assert not fc_differences(want, got)
    assert sum(len(v) for v in wq.values()) > 5000
    assert not lb.query_differences(wq, gq)
    ops = Counter(rec[2] for c in cs for rec in c["records"] if rec[0] == "K")
    assert all(ops[op] >= 40 for op in lb.NEW_OPS + (fc.DIST_Q,)), ops


def fc_differences(want: dict, got: dict) -> list:
    import modes_batch as mb
    return mb.differences(want, got, lb.FIELDS)


def test_a_park_with_its_own_tick_refuses_the_commands():
    """A park with its own tick whose kernels lack the troll's petrification and StopAllFollowers' members refuses the
    commands: here a subclass of the composed park with yet another tick (this port batched them in VecCombatPark's
    tick, tests/test_movement_vec_combat_leftovers.py; in the composed park's,
    tests/test_movement_vec_composed_leftovers.py, and changed this test from the composed park's refusal)."""
    from exact_ballpark.vec_composed import VecComposedPark

    class Own(VecComposedPark):
        def evolve(self, lib=None) -> None:
            super().evolve(lib)

    p = Own(2, device="cpu")
    m = torch.ones(2, dtype=torch.bool)
    for call in (lambda: p.set_free(m, m), lambda: p.set_ball_rigid(m), lambda: p.set_boid(m, m),
                 lambda: p.set_ball_troll(m, torch.ones(2, dtype=I64))):
        with pytest.raises(NotImplementedError):
            call()


def test_the_scenarios_reach_every_new_path():
    """The development scenarios' reference reaches every path the batched park adds: petrified trolls, trolls never
    petrified, members outside the follow modes released (stopped) by their target's warp, stale entries, fixings
    and freeings in every mode, both orbit laws and dynamical orientation in every family."""
    cs, rows, _, stale, _ = reference(1)
    trace = Counter()
    for c in cs:
        fc.reference(c, trace)
    assert trace["troll.petrified"] >= 50 and trace["release.stop"] >= 50 and trace["release.goto_direction"] >= 10
    assert stale >= 20
    by = defaultdict(dict)
    for (cid, s, bid), row in rows.items():
        by[(cid, s)][bid] = row
    i = {f: k for k, f in enumerate(lb.FIELDS)}
    released, fixed_in, freed = Counter(), Counter(), 0
    for (cid, s), balls in by.items():
        before = by.get((cid, s - 1))
        if not before:
            continue
        for bid, row in balls.items():
            old = before.get(bid)
            if old is None:
                continue
            t = old[i["follow_ptr"]]
            if (old[i["mode"]] not in (MO.FOLLOW, MO.ORBIT, MO.STOP) and t == old[i["follow_id"]] and t != -1
                    and row[i["mode"]] == MO.STOP and t in balls and balls[t][i["mode"]] == MO.WARP
                    and balls[t][i["effect_stamp"]] >= 0):
                released[old[i["mode"]]] += 1
            if old[i["free"]] and not row[i["free"]]:
                fixed_in[old[i["mode"]]] += 1
            freed += int(not old[i["free"]] and row[i["free"]])
    assert sum(released.values()) >= 10, released
    assert all(fixed_in[m] >= 5 for m in (MO.FOLLOW, MO.ORBIT, MO.STOP, MO.WARP, MO.TROLL)), fixed_in
    assert freed >= 50
    kinds = Counter((c["kind"], c["do"], c["new_orbit"]) for c in cs)
    assert all(kinds[(k, do, new)] >= 5 for k in lc.FAMILY_SIZES for do in (False, True)
               for new in (False, True)), kinds


@pytest.mark.skipif(not fc.HARNESS.exists(), reason="the compiled full harness is not built")
@pytest.mark.parametrize("seed", [1, 2, 3], ids=["development", "held_out_2", "held_out_3"])
def test_the_reference_is_destinys_compiled_code_on_the_leftovers(seed):
    """The reference against Destiny's own code compiled verbatim on every scenario of a seed: every field of every
    ball-state row (the raw follower sets, stale entries included), every step, and every distance query, bit for
    bit."""
    cs = lc.cases(seed)
    bad, rows, queries = [], 0, 0
    for k in range(0, len(cs), 100):
        chunk = cs[k:k + 100]
        oracle = fc.oracle(chunk)
        want = {(r[0], r[1], r[2]): r for r in oracle["S"]}
        got, gq = {}, []
        for c in chunk:
            res = fc.reference(c)
            got.update({(r[0], r[1], r[2]): r for r in res["S"]})
            gq += res["Q"]
        assert set(want) == set(got)
        rows += len(want)
        bad += [(*key, f) for key, w in want.items() for f, a, b in zip(fc.S_FIELDS, w, got[key]) if not fc.same(a, b)]
        wq, gq = sorted(oracle["Q"]), sorted(gq)
        assert len(wq) == len(gq)
        queries += len(wq)
        bad += [a[:6] for a, b in zip(wq, gq)
                if a[:6] != b[:6] or len(a[6]) != len(b[6]) or not all(fc.same(x, y) for x, y in zip(a[6], b[6]))]
    assert rows > 20000 and queries > 5000
    assert not bad, (len(bad), bad[:8])


# ====================================================================== the sincos patch: the call sites

def _calls(harness: Path) -> dict:
    """{function: Counter of the C library's sin, cos and sincos calls} in a compiled harness (objdump)."""
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "-C", str(harness)], capture_output=True, text=True,
                         timeout=120, check=True).stdout
    calls, fn = defaultdict(Counter), None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if m:
            fn = m.group(1)
            continue
        m = re.search(r"\s(?:bl|b)\s+[0-9a-f]+ <(sin|cos|sincos)@plt>", line)
        if m and fn is not None:
            calls[fn.split("(")[0]][m.group(1)] += 1
    return calls


@pytest.mark.skipif(shutil.which("objdump") is None, reason="objdump not found")
def test_compiled_destiny_fuses_exactly_the_sites_the_reference_and_the_kernel_take_from_sincos():
    """Every compiled harness (g++ -O2) calls glibc's double sincos twice in each of DefaultOrbitalNormal,
    EvolveOldStyleOrbit and EvolveNewStyleOrbit, and sin or cos there never; its only other double sine is
    CalculateYawPitchRoll's lone sin (full.py's math.sin there). The reference's three functions call modes._sincos twice
    each and modes.py calls math.sin and math.cos nowhere; the kernel's three device functions call g_sincos twice
    each and g_sin and g_cos only in the test kernel."""
    sites = {"Ballpark::DefaultOrbitalNormal": "_default_orbital_normal",
             "Ballpark::EvolveOldStyleOrbit": "_evolve_old_orbit", "Ballpark::EvolveNewStyleOrbit": "_evolve_new_orbit"}
    built = [h for h in HARNESSES.values() if h.is_file()]
    if not built:
        pytest.skip("no compiled harness on this machine")
    for h in built:
        calls = _calls(h)
        for fn in sites:
            assert calls[fn] == Counter({"sincos": 2}), (h, fn, calls[fn])
        others = {fn: c for fn, c in calls.items() if fn not in sites}
        assert all(set(c) <= {"sin"} for c in others.values()), (h, others)
        assert sum(c["sin"] for c in others.values()) <= 1 and set(others) <= {"Ball::CalculateYawPitchRoll"}, others
    tree = ast.parse(Path(MO.__file__).read_text())
    methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in sites.values():
        n = sum(1 for x in ast.walk(methods[name]) if isinstance(x, ast.Call) and getattr(x.func, "id", "") == "_sincos")
        assert n == 2, name
    attrs = {(x.value.id, x.attr) for x in ast.walk(tree) if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)}
    assert not attrs & {("math", "sin"), ("math", "cos")}
    src = K.SOURCE.read_text()
    for fn in ("precessing_normal", "evolve_old_orbit", "evolve_new_orbit"):
        start = src.find(f" {fn}(")
        assert start > 0, fn
        body = src[start:src.index("\n}\n", start)]
        assert body.count("g_sincos(") == (2 if fn != "evolve_old_orbit" else 0), fn
        assert "g_sin(" not in body and "g_cos(" not in body, fn
    test_kernel = src[src.index("__global__ void dlibm_kernel"):]
    assert src.count("g_sin(") - test_kernel.count("g_sin(") == 1          # its definition
    assert src.count("g_cos(") - test_kernel.count("g_cos(") == 1


def test_the_kernels_sincos_is_the_ypr_kernels_copy():
    """vec_modes_cuda.cu's g_sincos (the patch) is vec_ypr_cuda.cu's (the device sincos, held to glibc on the
    card), comments aside, constants renamed (DL_ to YL_, KD to YKD)."""
    from exact_ballpark import vec_ypr_cuda as Y

    def block(text: str) -> str:
        start = text.index("static __device__ __noinline__ void g_sincos(")
        body = re.sub(r"//[^\n]*", "", text[start:text.index("\n}\n", start) + 3])
        return re.sub(r"\s+", " ", body).replace("( ", "(").replace(" )", ")")

    mine = re.sub(r"\bDL_", "YL_", block(K.SOURCE.read_text())).replace("KD(", "YKD(")
    assert mine == block(Y.SOURCE.read_text())


# ================================================================================================ on the card

def _run(seed: int, lib=None, park_hook=None, kinds: tuple = ()) -> tuple:
    """(row differences, query differences, rows, stale entries) of the park against the reference on a seed's
    scenarios (only the families in `kinds`, when given)."""
    cs, rows, queries, stale, undefined = reference(seed)
    assert not undefined
    if kinds:
        cs = [c for c in cs if c["kind"] in kinds]
        keep = {c["id"] for c in cs}
        rows = {k: v for k, v in rows.items() if k[0] in keep}
        queries = {k: v for k, v in queries.items() if k[0] in keep}
    got, gq = lb.batched(cs, lib=lib, park_hook=park_hook)
    return fc_differences(rows, got), lb.query_differences(queries, gq), len(rows), stale


@needs_cuda
@pytest.mark.parametrize("seed", [1, 2, 3], ids=["development", "held_out_2", "held_out_3"])
def test_the_park_matches_fullballpark_on_the_leftovers(seed):
    bad, qbad, n, stale = _run(seed)
    assert n > 20000 and stale > 0
    assert not bad, (len(bad), bad[:8])
    assert not qbad, (len(qbad), qbad[:8])


@needs_cuda
def test_a_tick_replayed_from_a_cuda_graph_is_the_eager_tick():
    """A ModesGraph's replays on the development scenarios (the commands rebinding free, interactive, mode and the
    rest between ticks) equal the eager ticks."""
    cs = lc.cases(1)[::2]
    eager, eq = lb.batched(cs)
    graph, gq = lb.batched(cs, park_hook=lambda park: K.ModesGraph(park).tick)
    import modes_batch as mb
    assert not mb.differences(eager, graph, lb.FIELDS) and not lb.query_differences(eq, gq)


@needs_cuda
def test_the_distances_captured_in_a_cuda_graph_are_the_eager_ones():
    g = torch.Generator().manual_seed(7)
    pos = (torch.rand(64, 3, generator=g, dtype=F64) * 2e5 - 1e5).cuda()
    radius = torch.rand(64, generator=g, dtype=F64).mul(500.0).to(torch.float32).to(F64).cuda()
    a = torch.randint(-1, 66, (4096,), generator=g).cuda()
    b = torch.randint(-1, 66, (4096,), generator=g).cuda()
    eager = VD.center_surface_dist(pos, radius, a, b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = VD.center_surface_dist(pos, radius, a, b)
    graph.replay()
    torch.cuda.synchronize()
    assert all(bool(same_bits(x.to(F64), y.to(F64)).all()) for x, y in zip(eager, out))
    ca, sa, fa = (t.cpu() for t in eager)
    p, r = pos.cpu().tolist(), radius.cpu().tolist()
    for k in range(0, 4096, 7):
        i, j = int(a[k]), int(b[k])
        if 0 <= i < 64 and 0 <= j < 64:
            d = MO._length([p[i][0] - p[j][0], p[i][1] - p[j][1], p[i][2] - p[j][2]])
            assert bool(fa[k]) and fc.same(float(ca[k]), d) and fc.same(float(sa[k]), d - r[i] - r[j])
        else:
            assert not bool(fa[k])


@needs_cuda
def test_the_kernels_sincos_is_glibcs():
    """The kernel's g_sincos (sincos_sin, sincos_cos) equals exact_ballpark.dlibm.sincos and glibc's sincos on the
    arguments (every branch; every argument within 2,048 patterns of each point where sincos's sine can part from
    sin's), and at +-0x1.ad1fb54442d18p+0 differs from the kernel's own g_sin as glibc's sin does."""
    import test_movement_dlibm as TD
    import test_movement_dlibm_sincos as TS
    xs = TD.arguments(41, 60000) + TS.windows(2048) + [TS.X0, -TS.X0]
    x = torch.tensor(xs, dtype=F64, device="cuda")
    s, c = K.dlibm("sincos_sin", x), K.dlibm("sincos_cos", x)
    want = [TS.c_sincos(v) for v in xs]
    mine = [D.sincos(v) for v in xs]
    for got, k in ((s, 0), (c, 1)):
        assert bool(same_bits(got, torch.tensor([w[k] for w in want], dtype=F64, device="cuda")).all())
        assert bool(same_bits(got, torch.tensor([m[k] for m in mine], dtype=F64, device="cuda")).all())
    x0 = torch.tensor([TS.X0, -TS.X0], dtype=F64, device="cuda")
    assert not bool(same_bits(K.dlibm("sincos_sin", x0), K.dlibm("sin", x0)).any())
    assert bool(same_bits(K.dlibm("sincos_cos", x0), K.dlibm("cos", x0)).all())


# ============================================================================================= planted faults

KFAULTS = {         # name: (source edits (old, new, count)): kernel faults, built apart
    # a member's follow fields cleared at its release whatever its mode (the code before the fix)
    "member_cleared": [("        if (b.mode == MODE_FOLLOW || b.mode == MODE_ORBIT) stop_following(b);\n        b.mode = MODE_STOP;",
                        "        stop_following(b);\n        b.mode = MODE_STOP;", 1)],
    # the follower test before the fix: FOLLOW and ORBIT balls only, in the prepass and in the main kernel
    "follow_modes_only": [
        ("        if (member_of(s, t, s.follow_id[i]) && s.ids[t] < s.ids[i] && activates(s, t)) {",
         "        if ((m == MODE_FOLLOW || m == MODE_ORBIT) && t >= 0 && t < s.n && s.ids[t] < s.ids[i] && activates(s, t)) {",
         1),
        ("    if (member_of(s, b.follow_ptr, b.follow_id) && s.act[b.follow_ptr]) {",
         "    if ((b.mode == MODE_FOLLOW || b.mode == MODE_ORBIT) && b.follow_ptr >= 0 && b.follow_ptr < s.n\n"
         "        && s.act[b.follow_ptr]) {", 1)],
    # a troll petrified a tick early
    "troll_early": [("    if (live && b.mode == MODE_TROLL && !(b.effect_stamp > b.clock)) {",
                     "    if (live && b.mode == MODE_TROLL && !(b.effect_stamp > b.clock + 1)) {", 1)],
    # a troll never petrified
    "troll_never": [("    if (live && b.mode == MODE_TROLL && !(b.effect_stamp > b.clock)) {",
                     "    if (false) {", 1)],
}


KFAULT_FAMILIES = {"member_cleared": ("boid",), "follow_modes_only": ("boid",), "troll_early": ("troll",),
                   "troll_never": ("troll",)}           # the families that reach each fault's path


def faulted_library(name: str) -> dict:
    text = K.SOURCE.read_text()
    for old, new, count in KFAULTS[name]:
        assert text.count(old) == count, (name, old)
        text = text.replace(old, new)
    d = K.build_dir() / "faults_modes2"
    d.mkdir(parents=True, exist_ok=True)
    src = d / f"{name}.cu"
    if not src.exists() or src.read_text() != text:
        src.write_text(text)
    return K.load_library(source=src, checks=False)


@needs_cuda
@pytest.mark.parametrize("fault", sorted(KFAULTS))
def test_planted_kernel_faults_are_caught(fault):
    bad, _, _, _ = _run(1, lib=faulted_library(fault), kinds=KFAULT_FAMILIES[fault])
    assert bad, f"the planted fault {fault} went unnoticed"


def _boid_with_stop(self, mask, on):
    on = on.to(self.dev) != 0
    self._stop_rows(mask)
    self.mode = torch.where(mask & on, MO.BOID, self.mode)


def _free_keeping_thrust(self, mask, flag):
    keep = self.last_g
    WHOLE["set_free"](self, mask, flag)
    self.last_g = keep


def _squares_reversed(pos, radius, a, b):
    """The centre distance's squares summed right to left (Vector3d::Length sums them left to right). The
    first distance fault, the surface distance with the radii summed first, went unnoticed on the card (item GF): a
    float radius r subtracts exactly from a centre distance below 2^29 times r's leading power of two (about 1e9 m
    for r = 1 m, 7e10 m for 100 m), so the two orders round alike there and part only for a ball far out in warp."""
    _, _, found = WHOLE["center_surface_dist"](pos, radius, a, b)
    sa, sb = torch.where(found, a.to(pos.device), 0), torch.where(found, b.to(pos.device), 0)
    d = pos[sa] - pos[sb]
    center = torch.sqrt(d[:, 2] * d[:, 2] + d[:, 1] * d[:, 1] + d[:, 0] * d[:, 0])
    nan = torch.full_like(center, float("nan"))
    return (torch.where(found, center, nan), torch.where(found, center - radius[sa] - radius[sb], nan), found)


WHOLE = {"set_free": V.VecModesPark.set_free, "center_surface_dist": VD.center_surface_dist}
TFAULT_FAMILIES = {"boid_with_stop": ("boid",), "free_keeps_thrust": ("free",), "squares_reversed": ("mixed",)}
TFAULTS = {"boid_with_stop": (V.VecModesPark, "set_boid", _boid_with_stop),
           "free_keeps_thrust": (V.VecModesPark, "set_free", _free_keeping_thrust),
           "squares_reversed": (VD, "center_surface_dist", _squares_reversed)}


@needs_cuda
@pytest.mark.parametrize("fault", sorted(TFAULTS))
def test_planted_command_faults_are_caught(fault, monkeypatch):
    owner, name, fn = TFAULTS[fault]
    monkeypatch.setattr(owner, name, fn)
    bad, qbad, _, _ = _run(1, kinds=TFAULT_FAMILIES[fault])
    assert bad or qbad, f"the planted fault {fault} went unnoticed"

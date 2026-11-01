# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The batched accuracy helpers (exact_ballpark.vec_accuracy: GetAccuracy, CMB-702; CalculateRelativeVelocities, CMB-701)
against the scalar reference (exact_ballpark.full.FullBallpark.get_accuracy and relative_velocities), bit for bit, on the
CPU and (when one is visible) the GPU, over generated pairs: random geometry and the arguments' edge values (each early
return, coincident and overlapping balls, the falloff branch, a negative speed2 from rounding, exponents whose power is
subnormal or zero, NaN and infinite arguments), with and without dynamical orientation. The reference is held to
Destiny's compiled code by tests/test_movement_full_oracle.py, so this is exact against Destiny. Development pairs are
seed 1, held-out pairs seed 2 (first run after the generator was written). Where the reference's Python raises and C
does not (pow overflowing, a zero divisor), the batch is held to the C library's pow directly. Planted faults must each
be caught.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import random
import struct

import pytest
import torch

import exact_ballpark.vec_accuracy as va
import exact_ballpark.vec_libm as L
from exact_ballpark.full import FullBallpark

CUDA = torch.cuda.is_available()
F64 = torch.float64
_LIBM = ctypes.CDLL(ctypes.util.find_library("m"))
_LIBM.pow.restype = ctypes.c_double
_LIBM.pow.argtypes = [ctypes.c_double, ctypes.c_double]


def bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def same(a: float, b: float) -> bool:
    return bits(a) == bits(b) or (a != a and b != b)


def _edge(r: random.Random, base: float) -> float:
    """An argument: mostly ordinary, sometimes an edge value."""
    u = r.random()
    if u < 0.04:
        return 0.0
    if u < 0.06:
        return -base
    if u < 0.07:
        return math.nan
    if u < 0.08:
        return math.inf
    if u < 0.10:
        return base * 10 ** r.uniform(-12.0, -4.0)
    return base * 10 ** r.uniform(-1.5, 1.5)


def scenarios(seed: int, n_parks: int = 60, balls: int = 6, calls: int = 40) -> list:
    """Parks of a few balls (FullBallpark, the Python AddBall) and calls on random pairs: (do, balls, calls) with each
    call (src, dst, optimal, falloff, tracking, signature, optimal_signature, flags)."""
    r = random.Random(seed)
    out = []
    for _ in range(n_parks):
        do = r.random() < 0.4
        recs = []
        aligned = []
        centre = [r.uniform(-1e5, 1e5) for _ in range(3)]
        spread = 10 ** r.uniform(0.0, 4.5)
        for k in range(balls):
            bid = 1 + k
            pos = [centre[i] + r.uniform(-spread, spread) for i in range(3)]
            if k > 0 and r.random() < 0.08:                 # coincident centres
                pos = list(recs[r.randrange(k)]["pos"])
            vel = [r.uniform(-400.0, 400.0) for _ in range(3)]
            if r.random() < 0.15:
                vel = [0.0, 0.0, 0.0]
            if k > 0 and r.random() < 0.15:                 # moving along the line of centres: speed2 near 0
                o = recs[r.randrange(k)]
                d = [pos[i] - o["pos"][i] for i in range(3)]
                s = r.uniform(-3.0, 3.0)
                vel = [o["vel"][i] + s * d[i] for i in range(3)]
                aligned.append((o["id"], bid))
            recs.append({"id": bid, "pos": pos, "vel": vel, "radius": r.choice([r.uniform(1.0, 60.0),
                                                                              r.uniform(60.0, 3000.0)]),
                         "mass": 10 ** r.uniform(5.0, 9.0), "max_vel": r.uniform(50.0, 500.0),
                         "ang_vel": [r.uniform(-0.5, 0.5) for _ in range(3)] if do and r.random() < 0.7 else None})
        calls_ = []
        for _ in range(calls):
            src = r.randint(1, balls + 1)                    # balls + 1: a missing ball
            dst = src if r.random() < 0.05 else r.randint(1, balls + 1)
            optimal = _edge(r, 1e4)
            falloff = _edge(r, 1e4)
            tracking = _edge(r, 0.05)
            signature = _edge(r, 100.0)
            osig = _edge(r, 40.0)
            if r.random() < 0.05:                           # far beyond optimal: tiny or subnormal powers
                optimal, falloff = 1.0, r.uniform(1e-2, 10.0)
            flags = r.choice([0, 1, 2, 3, 4, 7, 8, 15, r.randrange(16)])
            calls_.append((src, dst, optimal, falloff, tracking, signature, osig, flags))
        for a, b in aligned:                                # a rounding-negative speed2 times a huge k^2: E >= 1024
            for src, dst in ((a, b), (b, a)):
                calls_.append((src, dst, 1e30, 1.0, 1.0, 1.0, 10 ** r.uniform(8.0, 12.0), 15))
        out.append((do, recs, calls_))
    return out


def _park(do: bool, recs: list) -> FullBallpark:
    p = FullBallpark(dynamical_orientation=do)
    for b in recs:
        p.add_ball(b["id"], b["mass"], b["radius"], b["max_vel"], True, False, True, True, False, *b["pos"],
                   *b["vel"], 1.0, 1.0)
        if b["ang_vel"] is not None:
            p.set_ball_angular_velocity(b["id"], *b["ang_vel"])
    return p


def reference(cs: list) -> tuple:
    """The reference's answers: (accuracy rows, relative-velocity rows, the calls whose reference raised)."""
    acc, rel, raised = [], [], []
    for pi, (do, recs, calls) in enumerate(cs):
        p = _park(do, recs)
        for ci, (src, dst, opt, fall, trk, sig, osig, flags) in enumerate(calls):
            try:
                acc.append(((pi, ci), p.get_accuracy(src, dst, opt, fall, trk, sig, osig)))
            except (OverflowError, ZeroDivisionError) as exc:
                raised.append(((pi, ci), type(exc).__name__))
            r_ = p.relative_velocities(src, dst, flags)
            rel.append(((pi, ci), (tuple(r_[0]), *r_[1:])))
    return acc, rel, raised


def _tensors(cs: list, device: str) -> dict:
    """Every call's inputs as rows: the balls' fields as the park holds them."""
    rows = {k: [] for k in ("ps", "vs", "rs", "pd", "vd", "rd", "ws", "same", "present", "args", "flags", "do",
                            "key")}
    for pi, (do, recs, calls) in enumerate(cs):
        p = _park(do, recs)
        for ci, (src, dst, opt, fall, trk, sig, osig, flags) in enumerate(calls):
            s, d = p.balls.get(src), p.balls.get(dst)
            ok = s is not None and d is not None
            s = s if s is not None else next(iter(p.balls.values()))
            d = d if d is not None else next(iter(p.balls.values()))
            rows["ps"].append(list(s.pos))
            rows["vs"].append(list(s.vel))
            rows["rs"].append(float(s.radius))
            rows["pd"].append(list(d.pos))
            rows["vd"].append(list(d.vel))
            rows["rd"].append(float(d.radius))
            rows["ws"].append([float(c) for c in s.ang_vel])
            rows["same"].append(src == dst)
            rows["present"].append(ok)
            rows["args"].append([opt, fall, trk, sig, osig])
            rows["flags"].append(flags)
            rows["do"].append(do)
            rows["key"].append((pi, ci))
    t = {k: torch.tensor(v, dtype=F64, device=device) for k, v in rows.items()
         if k in ("ps", "vs", "rs", "pd", "vd", "rd", "ws", "args")}
    t["same"] = torch.tensor(rows["same"], device=device)
    t["present"] = torch.tensor(rows["present"], device=device)
    t["flags"] = torch.tensor(rows["flags"], dtype=torch.int64, device=device)
    t["do"] = torch.tensor(rows["do"], device=device)
    t["key"] = rows["key"]
    return t


def batched(cs: list, device: str) -> tuple:
    t = _tensors(cs, device)
    a = t["args"]
    acc = va.get_accuracy(t["ps"], t["vs"], t["rs"], t["pd"], t["vd"], t["rd"], a[:, 0], a[:, 1], a[:, 2], a[:, 3],
                          a[:, 4], same=t["same"], present=t["present"])
    plain = va.relative_velocities(t["ps"], t["vs"], t["rs"], t["pd"], t["vd"], t["rd"], t["flags"],
                                   present=t["present"])
    dyn = va.relative_velocities(t["ps"], t["vs"], t["rs"], t["pd"], t["vd"], t["rd"], t["flags"],
                                 ang_vel_s=t["ws"], dynamical=True, present=t["present"])
    do = t["do"]
    rel = tuple(torch.where(do[:, None] if x.dim() == 2 else do, y, x) for x, y in zip(plain, dyn))
    acc = [x.cpu().tolist() for x in acc]
    rel = [x.cpu().tolist() for x in rel]
    out_acc = {k: (acc[0][i], acc[1][i], acc[2][i]) for i, k in enumerate(t["key"])}
    out_rel = {k: (tuple(rel[0][i]), rel[1][i], rel[2][i], rel[3][i], rel[4][i]) for i, k in enumerate(t["key"])}
    return out_acc, out_rel, t


def _flat(v) -> list:
    out = []
    for x in v:
        out.extend(_flat(x) if isinstance(x, (tuple, list)) else [x])
    return out


def differences(cs: list, device: str) -> list:
    want_acc, want_rel, _ = reference(cs)
    got_acc, got_rel, _ = batched(cs, device)
    bad = []
    for k, v in want_acc:
        g = got_acc[k]
        if not all(same(float(x), float(y)) for x, y in zip(v, g)):
            bad.append(("accuracy", k, v, g))
    for k, v in want_rel:
        g = got_rel[k]
        if not all(same(float(x), float(y)) for x, y in zip(_flat(v), _flat(g))):
            bad.append(("relative", k, v, g))
    return bad


_CASES: dict = {}


def cases(seed: int) -> list:
    if seed not in _CASES:
        _CASES[seed] = scenarios(seed)
    return _CASES[seed]


def test_batched_accuracy_matches_reference_development_pairs():
    bad = differences(cases(1), "cpu")
    assert not bad, (len(bad), bad[:6])


def test_batched_accuracy_matches_reference_held_out_pairs():
    bad = differences(cases(2), "cpu")
    assert not bad, (len(bad), bad[:6])


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_batched_accuracy_matches_reference_on_the_gpu(seed):
    bad = differences(cases(seed), "cuda")
    assert not bad, (len(bad), bad[:6])


def test_the_pairs_reach_every_branch():
    """The reference's own counters and the pairs' kinds: every early return, the falloff branch, coincident centres,
    overlapping balls, NaN speeds (a negative speed2), subnormal and zero powers, and the dynamical branch."""
    want_acc, want_rel, raised = reference(cases(1))
    accs = [v[0] for _, v in want_acc]
    angs = [v[2] for _, v in want_acc]
    assert sum(1 for a in accs if a == 0.0) > 50
    assert sum(1 for a in accs if 0.0 < a < 2.2250738585072014e-308) >= 1          # subnormal
    assert sum(1 for a in accs if 0.0 < a < 1.0) > 300
    assert sum(1 for a in accs if a != a) > 5                                       # NaN arguments
    assert sum(1 for a in angs if a != a) >= 1                                      # sqrt of a negative speed2
    rels = [v for _, v in want_rel]
    assert sum(1 for v in rels if v[1] == va.UNSET and v[0][0] != va.UNSET) > 5    # coincident or REL only
    assert sum(1 for v in rels if v[2] == 0.0) > 5                                  # overlapping
    assert sum(1 for v in rels if v[4] not in (va.UNSET,)) > 300
    assert not raised, raised                                  # the reference's C semantics (the fix)
    assert sum(1 for a in accs if a == math.inf) >= 1          # E >= 1024: pow's +inf, where math.pow raised
    cs = cases(1)
    assert sum(1 for do, recs, calls in cs if do) >= 10


def test_the_power_is_the_c_librarys_through_its_overflow_and_underflow():
    """E of 1024 or more: Destiny's pow (the C library's) returns +inf, where Python's math.pow raises; the batch's
    power is held to the C library itself over a sweep of exponents through its overflow and underflow edges."""
    xs = [1023.5, 1024.0, 1024.5, 1100.0, 1e6, math.inf, -1022.0, -1074.0, -1074.5, -1075.0, -1075.5, -1200.0,
          -math.inf, math.nan, 0.0, -0.0, 5e-324, -5e-324, 1e-300]
    r = random.Random(7)
    xs += [r.uniform(-1080.0, 1030.0) for _ in range(20000)]
    t = torch.tensor(xs, dtype=F64)
    got = L.pow(torch.full_like(t, 2.0), t).tolist()
    for x, g in zip(xs, got):
        assert same(_LIBM.pow(2.0, x), g), (x, g)
    with pytest.raises(OverflowError):
        math.pow(2.0, 1024.0)


def test_lane_finding_the_reference_raised_where_destiny_returns_infinity():
    """The finding in the reference (not in Destiny): FullBallpark.get_accuracy took 2^E with Python's
    math.pow, which raises OverflowError for E of 1024 or more (a speed2 made negative by rounding, for a target moving
    along the line of centres, times a large k^2), where Destiny's compiled pow returns +inf. Pinned after the fix: a
    rounding-negative speed2 (its square root NaN) with k = 1e30 gives +inf, and math.pow would have raised."""
    p = FullBallpark()
    for bid, pos in ((1, (0.0, 0.0, 0.0)), (2, (3.0, 4.0, 12.0))):
        p.add_ball(bid, 1e6, 1.0, 100.0, True, False, True, True, False, *pos, 0.0, 0.0, 0.0, 1.0, 1.0)
    for i in range(1, 400):
        c = 0.01 * i
        p.set_velocity(2, 3.0 * c, 4.0 * c, 12.0 * c)
        if p.get_accuracy(1, 2, 1e30, 1.0, 1.0, 1.0, 1.0)[2] != p.get_accuracy(1, 2, 1e30, 1.0, 1.0, 1.0, 1.0)[2]:
            break
    else:
        pytest.fail("no velocity along the line of centres gave a rounding-negative speed2")
    acc = p.get_accuracy(1, 2, 1e30, 1.0, 1.0, 1.0, 1e30)[0]
    assert acc == math.inf, acc
    with pytest.raises(OverflowError):
        math.pow(2.0, 1e43)


def test_a_pairs_result_does_not_depend_on_the_batch():
    cs = cases(1)[:10]
    together, rel_t, _ = batched(cs, "cpu")
    for i in range(3):
        alone, rel_a, _ = batched([cs[i]], "cpu")
        for (pi, ci), v in alone.items():
            assert all(same(x, y) for x, y in zip(v, together[(i, ci)]))


# ---- planted faults: each must change the development comparison ----

def _pow_by_exp(x, y):
    """2^y as exp(y ln 2): a different rounding from glibc's pow."""
    return torch.exp(y * math.log(2.0))


def _fused_speed(orig):
    def f(pos_s, vel_s, rad_s, pos_d, vel_d, rad_d, optimal, falloff, tracking, signature, optimal_signature,
          same=None, present=None):
        a, s, w = orig(pos_s, vel_s, rad_s, pos_d, vel_d, rad_d, optimal, falloff, tracking, signature,
                       optimal_signature, same, present)
        return a, s, w * (1.0 + 2.0 ** -52)
    return f


def _short_from_centres(orig):
    def f(pos_s, vel_s, rad_s, pos_d, vel_d, rad_d, *args, **kw):
        return orig(pos_s, vel_s, rad_s * 0.0, pos_d, vel_d, rad_d, *args, **kw)
    return f


def _reciprocal_dist(orig):
    def f(pos_s, vel_s, rad_s, pos_t, vel_t, rad_t, flags, ang_vel_s=None, dynamical=False, present=None):
        rel, rad, short, trans, ang = orig(pos_s, vel_s, rad_s, pos_t, vel_t, rad_t, flags, ang_vel_s, dynamical,
                                           present)
        d = pos_t - pos_s
        dist = torch.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
        alt = trans * (torch.ones_like(dist) / torch.where(dist == 0.0, 1.0, dist))
        return rel, rad, short, trans, torch.where((ang != va.UNSET) & ~torch.as_tensor(dynamical), alt, ang)
    return f


def _no_spin(orig):
    def f(pos_s, vel_s, rad_s, pos_t, vel_t, rad_t, flags, ang_vel_s=None, dynamical=False, present=None):
        return orig(pos_s, vel_s, rad_s, pos_t, vel_t, rad_t, flags, None, False, present)
    return f


FAULTS = {
    "pow as exp(E ln 2)": ("L.pow", lambda m: _pow_by_exp),
    "the angular speed off by an ulp (a fused multiply-add)": ("get_accuracy", _fused_speed),
    "the surface distance without the shooter's radius": ("get_accuracy", _short_from_centres),
    "the angular speed by the reciprocal of the distance": ("relative_velocities", _reciprocal_dist),
    "the shooter's own rotation ignored with dynamical orientation": ("relative_velocities", _no_spin),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    target, patch = FAULTS[fault]
    if target == "L.pow":
        monkeypatch.setattr(va.L, "pow", patch(None))
    else:
        monkeypatch.setattr(va, target, patch(getattr(va, target)))
    assert differences(cases(1), "cpu"), f"the planted fault {fault} went unnoticed"


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_the_device_kernel_matches_the_reference(seed):
    """GetAccuracy as one CUDA kernel, one thread a pair (exact_ballpark.vec_combat_cuda.accuracy: glibc's pow
    transcribed in vec_orient_cuda.cu), against the reference, bit for bit."""
    from exact_ballpark import vec_combat_cuda as KC
    want_acc, _, _ = reference(cases(seed))
    t = _tensors(cases(seed), "cuda")
    out = KC.accuracy(t["ps"], t["vs"], t["rs"], t["pd"], t["vd"], t["rd"], t["args"], t["same"], t["present"])
    got = [x.cpu().tolist() for x in out]
    index = {k: i for i, k in enumerate(t["key"])}
    bad = [(k, v) for k, v in want_acc
           if not all(same(float(a), float(b)) for a, b in zip(v, (got[0][index[k]], got[1][index[k]],
                                                                    got[2][index[k]])))]
    assert not bad, (len(bad), bad[:6])

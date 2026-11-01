# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Eidosoma
"""The batched turning (exact_ballpark.vec_orient.VecOrientPark) against the scalar reference (exact_ballpark.full,
FullBallpark), bit for bit, every step, every ball-state field the batch holds (position, velocity and their old
values, goto point, thrust, mode, mass, radius, maximum speed, agility, speed fraction, decay factor, yaw, pitch, roll
and their old values, roll speed, yaw change, maximum angular speed, roll torque, rotational agility, rotation, angular
velocity and their old values, torque), on the CPU and (when one is visible) the GPU.

The scenarios are tests/turn_cases.py's batch-compatible families ("vec": without dynamical orientation, "vec_do":
with it): one to six free ships turning hard under GOTO, GotoDirection and STOP, every setter in flight, the refused
values among them. The reference rows are the ones tests/test_movement_turn_oracle.py holds to Destiny's own code
compiled with DirectXMath, so this is exact against Destiny. Development cases are seed 1; held-out cases (seed 2) are
a separate test, first run after the generator was frozen. Planted faults must each be caught.
"""

from __future__ import annotations

import pytest
import torch

import orient_batch as ob
import exact_ballpark.vec_orient as vo
import turn_cases as tc
from exact_ballpark.full import FullBallpark

CUDA = torch.cuda.is_available()
_REF: dict = {}


def cases(seed: int) -> list:
    return [c for c in tc.cases(seed) if c["kind"] in ob.BATCH_KINDS]


def reference(seed: int) -> dict:
    if seed not in _REF:
        _REF[seed] = ob.reference_rows(cases(seed))
    return _REF[seed]


def check(seed: int, device: str, cs=None) -> None:
    cs = cases(seed) if cs is None else cs
    ids = {c["id"] for c in cs}
    want = {k: v for k, v in reference(seed).items() if k[0] in ids}
    bad = ob.differences(want, ob.batched(cs, device))
    assert not bad, (len(bad), bad[:8])


def test_batched_matches_reference_development_cases():
    check(1, "cpu")


def test_batched_matches_reference_held_out_cases():
    check(2, "cpu")


@pytest.mark.skipif(not CUDA, reason="no CUDA device visible")
@pytest.mark.parametrize("seed", [1, 2], ids=["development", "held_out"])
def test_batched_matches_reference_on_the_gpu(seed):
    """On the GPU, the scenarios with Destiny's default friction (1e6: most of them, in a few large parks)."""
    check(seed, "cuda", [c for c in cases(seed) if c["friction"] == 1.0e6])


def test_a_balls_result_does_not_depend_on_the_batch():
    """Each ball alone, and the batch reversed, give the same rows as the batch as generated."""
    cs = [c for c in cases(1) if c["friction"] == 1.0e6][:60]
    together = ob.batched(cs, "cpu")
    reversed_ = ob.batched(cs[::-1], "cpu")
    alone = {}
    for c in cs[:12]:
        alone.update(ob.batched([c], "cpu"))
    assert together == reversed_ or not ob.differences(together, reversed_)
    assert not ob.differences({k: v for k, v in together.items() if k in alone}, alone)


TURNING = ["orient.thrust", "orient.behind", "orient.coast", "torque.full", "torque.brake_only", "torque.none",
           "roll.turn", "roll.brake_only", "roll.none", "snap.at_rest", "ypr.evolve", "ypr.snap"]


def test_the_scenarios_turn_and_set_in_every_way():
    """The reference's branch counters on the development scenarios, and the setters and refusals they contain."""
    from collections import Counter
    import full_cases as fc
    cs = cases(1)
    hits = Counter()
    for c in cs:
        t = Counter()
        fc.reference(c, t)
        hits.update(t.keys())
    thin = {b: hits[b] for b in TURNING if hits[b] < 5}
    assert not thin, thin
    ops = Counter(rec[2] for c in cs for rec in c["records"] if rec[0] == "K")
    for op in tc.VEC_OPS:
        assert ops[op] >= 50, (op, ops[op])
    refused = sum(1 for c in cs for rec in c["records"] if rec[0] == "K" and rec[2] in (fc.MASS_OP, fc.AGILITY_OP)
                  and not rec[7][3] > 0.0)
    assert refused >= 20, refused
    kinds = Counter(c["kind"] for c in cs)
    assert kinds["vec"] >= 200 and kinds["vec_do"] >= 300, kinds


# ---- planted faults: each must change the development comparison ----

def _kappas_by_pow(self, tf):
    """kappa1 = tf^n1 through pow, not Destiny's repeated float products."""
    n = torch.arange(1, vo.TORQUE_STEPS, dtype=torch.float64, device=tf.device)
    return vo.f32(vo.L.pow(tf[:, None].expand(-1, n.numel()).contiguous(), n[None, :].expand(tf.numel(), -1)
                           .contiguous()))


def _last_candidate(orig):
    def f(self, target, omega0, tau, kap, n2, kappa2, max_alpha, vector):
        n1, alpha1, kappa1 = orig(self, target, omega0, tau, kap, n2, kappa2, max_alpha, vector)
        return torch.where(n1 < vo.TORQUE_STEPS, n1 + 1, n1), alpha1, kappa1        # one step later than the first
    return f


def _stop_damped_with_do(orig):
    def f(self):
        if self.dynamical_orientation:
            vy = self.vel[:, 1]
            self.vel = self.vel.clone()
            self.vel[:, 1] = torch.where(self.mode == vo.STOP, (vy - 0.07 * vy) * vo.STOP_DAMP, vy)
        return orig(self)
    return f


def _mass_without_decay(orig):
    def f(self, mask, mass):
        tf = self.time_factor.clone()
        orig(self, mask, mass)
        self.time_factor = tf
    return f


def _velocity_without_snap(self, mask, v):
    v = v.to(torch.float64)
    self.vel = torch.where(mask[:, None], v, self.vel)
    self.old_vel = torch.where(mask[:, None], v, self.old_vel)


DO, NO_DO = ("vec_do",), ("vec",)
FAULTS = {
    # MOV-728: kappa1 by pow instead of Destiny's loop of float products
    "kappa_by_pow": (DO, lambda mp: mp.setattr(vo.VecOrientPark, "_kappas", _kappas_by_pow)),
    # MOV-728: the search's n1 one step after the first passing step
    "torque_last_candidate": (DO, lambda mp: mp.setattr(vo.VecOrientPark, "_search",
                                                        _last_candidate(vo.VecOrientPark._search))),
    # MOV-728: STOP's vertical damping kept with dynamical orientation
    "stop_damped_with_do": (DO, lambda mp: mp.setattr(vo.VecOrientPark, "evolve",
                                                      _stop_damped_with_do(vo.VecOrientPark.evolve))),
    # the float functions: torch's own sine (SLEEF on the CPU, CUDA's on the GPU) in place of the C library's
    "torch_sinf": (DO, lambda mp: mp.setattr(vo.L, "sinf", lambda x: torch.sin(x.to(torch.float32))
                                             .to(torch.float64))),
    # MOV-729: the roll decay -0.45
    "roll_decay": (NO_DO, lambda mp: mp.setattr(vo, "ROLL_DECAY", -0.45)),
    # MOV-729: Modulo2pi's 2 pi off by 1e-12
    "modulo_2pi": (NO_DO, lambda mp: mp.setattr(vo, "TWO_PI", 6.283185307179586 + 1e-12)),
    # MOV-704: SetBallMass without the decay factor's refresh
    "mass_without_decay": (DO + NO_DO, lambda mp: mp.setattr(vo.VecOrientPark, "set_mass",
                                                             _mass_without_decay(vo.VecOrientPark.set_mass))),
    # MOV-713: SetBallVelocity without the orientation's snap
    "velocity_without_snap": (DO + NO_DO, lambda mp: mp.setattr(vo.VecOrientPark, "set_velocity",
                                                                _velocity_without_snap)),
}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_planted_faults_are_caught(fault, monkeypatch):
    kinds, patch = FAULTS[fault]
    patch(monkeypatch)
    cs = [c for c in cases(1) if c["friction"] == 1.0e6 and c["kind"] in kinds][:80]
    ids = {c["id"] for c in cs}
    want = {k: v for k, v in reference(1).items() if k[0] in ids}
    assert ob.differences(want, ob.batched(cs, "cpu")), f"the planted fault {fault} went unnoticed"

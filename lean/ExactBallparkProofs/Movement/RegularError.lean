-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Movement.Integrator
public import ExactBallparkProofs.Movement.Reference
public import ExactBallparkProofs.Numerics.Product

@[expose] public noncomputable section
namespace ExactBallpark.Movement.RegularError
open ExactBallpark.Numerics.Compensated (rn roundBound)

/-- The value uses the source rounding order. The radius propagates the local
relative-rounding bounds even when large terms cancel in the ordinary position branch. -/
structure Estimate where
  value : ℝ
  ideal : ℝ
  radius : ℝ
  covers : |value-ideal| ≤ radius

def input (x : ℝ) : Estimate := ⟨x,x,0,by simp⟩

def rounded (raw target budget : ℝ) (bound : |raw-target| ≤ budget) : Estimate :=
  ⟨rn raw,target,|raw|/2^53+budget, (abs_sub_le ..).trans (add_le_add (roundBound raw) bound)⟩

def plus (x y : Estimate) : Estimate :=
  rounded (x.value+y.value) (x.ideal+y.ideal) (x.radius+y.radius) (by
    rw [show x.value+y.value-(x.ideal+y.ideal) = (x.value-x.ideal)+(y.value-y.ideal) by ring]
    exact (abs_add_le ..).trans (add_le_add x.covers y.covers))

def minus (x y : Estimate) : Estimate :=
  rounded (x.value-y.value) (x.ideal-y.ideal) (x.radius+y.radius) (by
    rw [show x.value-y.value-(x.ideal-y.ideal) = (x.value-x.ideal)-(y.value-y.ideal) by ring]
    exact (abs_sub _ _).trans (add_le_add x.covers y.covers))

def times (x y : Estimate) : Estimate :=
  rounded (x.value*y.value) (x.ideal*y.ideal) (x.radius*|y.value| + |x.ideal| * y.radius) (by
    rw [show x.value*y.value-x.ideal*y.ideal =
      (x.value-x.ideal)*y.value+x.ideal*(y.value-y.ideal) by ring]
    apply (abs_add_le ..).trans
    rw [abs_mul,abs_mul]
    exact add_le_add (mul_le_mul_of_nonneg_right x.covers (abs_nonneg _))
      (mul_le_mul_of_nonneg_left y.covers (abs_nonneg _)))

/-- Only k and the shared rounded k² are inverted. Their ideal values are known
from the input coefficient, so the reciprocal's input error can be measured exactly. -/
def reciprocal (x : Estimate) : Estimate :=
  rounded (1/x.value) (1/x.ideal) |1/x.value-1/x.ideal| le_rfl

def step (p v a m k q t : ℝ) : Estimate × Estimate :=
  let mass := input m
  let drag := input k
  let factor := input q
  let k2 := times drag drag
  let ma := times mass (input a)
  let av := minus (plus (times (input a) (input t)) (input v)) (times factor (input v))
  let inner := plus (times ma (minus factor (input 1))) (times drag av)
  let pn := times (plus (times mass inner) (times (input p) k2)) (reciprocal k2)
  let vn := times (minus ma (times (minus ma (times (input v) drag)) factor)) (reciprocal drag)
  (pn,vn)

/-- Separate the budget for the actual sequence of rounded operations from an
error in the cached or recomputed decay. The latter has an exact linear sensitivity. -/
theorem regular_step_error (p v a m k q t ε : ℝ) (hk : 0 < k)
    (cache : |q-decay m k t| ≤ ε) :
    let (pn,vn) := step p v a m k q t
    let (pos,vel) := Reference.ordinaryRounded rn p v a m k q t
    |pos-position p v a m k t| ≤ pn.radius + |m/k*(v-terminal m k a)| * ε ∧
    |vel-velocity v a m k t| ≤ vn.radius + |v-terminal m k a| * ε := by
  dsimp only
  let out := step p v a m k q t
  have pId : out.1.ideal - position p v a m k t =
      -(m/k*(v-terminal m k a))*(q-decay m k t) := by
    change regularPosition p v a m k q t - _ = _
    unfold regularPosition position terminal
    field_simp [hk.ne']; ring
  have vId : out.2.ideal - velocity v a m k t = (v-terminal m k a)*(q-decay m k t) := by
    change sourceVelocity v a m k q - _ = _
    unfold sourceVelocity velocity terminal
    field_simp [hk.ne']; ring
  constructor
  · apply (abs_sub_le out.1.value out.1.ideal _).trans
    apply add_le_add out.1.covers
    rw [pId,abs_mul,abs_neg]
    exact mul_le_mul_of_nonneg_left cache (abs_nonneg _)
  · apply (abs_sub_le out.2.value out.2.ideal _).trans
    apply add_le_add out.2.covers
    rw [vId,abs_mul]
    exact mul_le_mul_of_nonneg_left cache (abs_nonneg _)

end ExactBallpark.Movement.RegularError

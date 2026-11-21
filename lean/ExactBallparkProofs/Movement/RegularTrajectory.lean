-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Bounds
public import ExactBallparkProofs.Movement.RegularError

@[expose] public noncomputable section
namespace ExactBallpark.Movement.RegularTrajectory

open ExactBallpark.Numerics.Compensated (rn)

/-- One ordinary drag interval for one coordinate. Both paths receive this same
held acceleration. The supplied factor belongs to the rounded path. -/
structure Interval where
  mass : ℝ
  friction : ℝ
  duration : ℝ
  acceleration : ℝ
  factor : ℝ
  factorError : ℝ

def Valid (f : Interval) : Prop :=
  0 < f.mass ∧ 0 < f.friction ∧ 0 < f.duration ∧
    |f.factor - decay f.mass f.friction f.duration| ≤ f.factorError

abbrev Phase := ℝ × ℝ

def exactStep (f : Interval) (x : Phase) : Phase :=
  (position x.1 x.2 f.acceleration f.mass f.friction f.duration,
   velocity x.2 f.acceleration f.mass f.friction f.duration)

def roundedStep (f : Interval) (x : Phase) : Phase :=
  Reference.ordinaryRounded rn x.1 x.2 f.acceleration f.mass f.friction f.factor f.duration

/-- Carry error in metres and metres per second separately. The per-step radii are
evaluated on the incoming rounded path, where cancellation actually occurs. -/
def nextBudget (f : Interval) (x budget : Phase) : Phase :=
  let q := decay f.mass f.friction f.duration
  let gap := x.2 - terminal f.mass f.friction f.acceleration
  let (pn, vn) := RegularError.step x.1 x.2 f.acceleration f.mass f.friction f.factor f.duration
  (pn.radius + |f.mass / f.friction * gap| * f.factorError +
     (budget.1 + f.mass / f.friction * (1 - q) * budget.2),
   vn.radius + |gap| * f.factorError + q * budget.2)

def Within (observed reference budget : Phase) : Prop :=
  |observed.1 - reference.1| ≤ budget.1 ∧ |observed.2 - reference.2| ≤ budget.2

private theorem propagate (f : Interval) (valid : Valid f) (x y b : Phase) (h : Within x y b) :
    Within (roundedStep f x) (exactStep f y) (nextBudget f x b) := by
  obtain ⟨hm, hk, ht, hf⟩ := valid
  have hq := decay_bounds hm hk ht.le
  let q := decay f.mass f.friction f.duration
  let c := f.mass / f.friction * (1 - q)
  have hc : 0 ≤ c := mul_nonneg (div_nonneg hm.le hk.le) (sub_nonneg.mpr hq.2)
  have pd : (exactStep f x).1 - (exactStep f y).1 = (x.1-y.1) + c*(x.2-y.2) := by
    dsimp [exactStep, position, c, q]
    ring
  have vd : (exactStep f x).2 - (exactStep f y).2 = q*(x.2-y.2) := by
    dsimp [exactStep, velocity, q]
    ring
  obtain ⟨ep, ev⟩ := RegularError.regular_step_error x.1 x.2 f.acceleration f.mass f.friction
    f.factor f.duration f.factorError hk hf
  constructor
  · apply (abs_sub_le (roundedStep f x).1 (exactStep f x).1 (exactStep f y).1).trans
    apply add_le_add ep
    rw [pd]
    apply (abs_add_le ..).trans
    rw [abs_mul, abs_of_nonneg hc]
    exact add_le_add h.1 (mul_le_mul_of_nonneg_left h.2 hc)
  · apply (abs_sub_le (roundedStep f x).2 (exactStep f x).2 (exactStep f y).2).trans
    apply add_le_add ev
    rw [vd, abs_mul, abs_of_pos hq.1]
    exact mul_le_mul_of_nonneg_left h.2 hq.1.le

/-- The budget uses only the rounded path and the supplied interval certificates. -/
def withBudget (x : Phase × Phase) (f : Interval) : Phase × Phase :=
  (roundedStep f x.1, nextBudget f x.1 x.2)

theorem regular_trajectory_error (intervals : List Interval) (observed reference budget : Phase)
    (start : Within observed reference budget) (valid : ∀ f ∈ intervals, Valid f) :
    let (result, bound) := intervals.foldl withBudget (observed, budget)
    Within result (intervals.foldl (fun x f => exactStep f x) reference) bound := by
  induction intervals generalizing observed reference budget with
  | nil => exact start
  | cons first rest ih =>
    exact ih _ _ _ (propagate first (valid first List.mem_cons_self) _ _ _ start)
      (fun f hf => valid f (List.mem_cons_of_mem _ hf))

end ExactBallpark.Movement.RegularTrajectory

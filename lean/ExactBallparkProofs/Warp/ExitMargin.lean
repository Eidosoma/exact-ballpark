-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Warp.Boundaries

@[expose] public noncomputable section
namespace ExactBallpark.Warp

/-- A sample exits either on overrun, or strictly below the speed threshold in deceleration. -/
def exitsAt (ta tc td peak rate maxVel t : ℝ) : Bool := by
  classical
  exact match phase ta tc td t with
  | .overrun => true
  | .decel => decide (drop maxVel (peak * Real.exp (-rate*(t-tc-ta))))
  | _ => false

/-- Compare each sample in time units. The third boundary is the speed threshold
inverted by dropout_time_iff, so a margin on it controls exponential decay as well. -/
theorem cruise_exit_margin (samples : List ℝ) (ta tc td peak rate maxVel eps : ℝ)
    (hp : 0 < peak) (hr : 0 < rate) (hm : 0 < maxVel) (err : |tc| ≤ eps)
    (margin : ∀ t ∈ samples, ta ≤ t →
      ∀ boundary ∈ [0, td+1, Real.log (peak/min (maxVel/2) 100)/rate],
        eps < |(t-ta)-boundary|) :
    samples.find? (exitsAt ta tc td peak rate maxVel) =
      samples.find? (exitsAt ta 0 td peak rate maxVel) := by
  apply List.find?_congr
  intro t ht
  dsimp [exitsAt, phase]
  by_cases h : t < ta
  · simp [h]
  have bound := margin t ht (le_of_not_gt h)
  have eps0 := (abs_le.mp err)
  have away (c : ℝ) (hc : eps < |(t-ta)-c|) :
      ((t-tc-ta < c) ↔ (t-ta < c)) ∧ ((c < t-tc-ta) ↔ (c < t-ta)) := by
    rw [lt_abs] at hc
    rcases hc with h | h <;> constructor <;> constructor <;> intro <;> linarith
  obtain ⟨first,_⟩ := away 0 (bound 0 (by simp))
  obtain ⟨last,_⟩ := away (td+1) (bound (td+1) (by simp))
  obtain ⟨_,speed⟩ := away _ (bound (Real.log (peak/min (maxVel/2) 100)/rate) (by simp))
  have start : (t-ta < tc) ↔ (t-ta < 0) := by
    rw [← first]; constructor <;> intro <;> linarith
  have d : drop maxVel (peak * Real.exp (-rate*(t-tc-ta))) ↔
      drop maxVel (peak * Real.exp (-rate*(t-ta))) :=
    (dropout_time_iff hp hr hm _).trans (speed.trans (dropout_time_iff hp hr hm _).symm)
  by_cases c : t-ta < 0 <;> by_cases l : t-ta < td+1 <;>
    simp only [h, ite_false, sub_zero, start, last, c, l, ite_true];
    exact Bool.eq_iff_iff.mpr (by simpa only [decide_eq_true_eq, sub_zero] using d)

end ExactBallpark.Warp

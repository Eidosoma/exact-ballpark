-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Physics.Units
public import ExactBallparkProofs.Movement.Correspondence
public import ExactBallparkProofs.Collision.ImpactHistory

@[expose] public section
namespace RealPhysics.TimeGuards
open ExactBallpark ExactBallpark.Physics ExactBallpark.Movement
open Dimension

/-- Elapsed time since the start of the specific segment that produced the hit. -/
noncomputable def elapsed (u : Dimensionless ℝ) (window : Seconds ℝ) : Seconds ℝ :=
  (u * window).cast (one_mul _)

theorem elapsed_cutoff {window : Seconds ℝ} (hw : 0 < window.val) (u : Dimensionless ℝ)
    (tolerance : Seconds ℝ) :
    elapsed u window ≤ tolerance ↔ u.val ≤ tolerance.val / window.val :=
  (le_div_iff₀ hw).symm

#guard_msgs (drop info) in
#check_failure fun (u : Dimensionless ℝ) (tolerance : Seconds ℝ) => u ≤ tolerance

#guard_msgs (drop info) in
#check_failure fun (k : WithDim (M𝓭 / T𝓭) ℝ) (epsilon : Dimensionless ℝ)
    (m : WithDim M𝓭 ℝ) (dt : Seconds ℝ) => k < epsilon * m * dt

/-- The source expression has the units of friction if its coefficient is an inverse time squared. -/
noncomputable def frictionCutoff (c : WithDim (T𝓭 * T𝓭)⁻¹ ℝ) (m : WithDim M𝓭 ℝ)
    (t : Seconds ℝ) : WithDim (M𝓭 / T𝓭) ℝ :=
  (c * m * t).cast (by simp [div_eq_mul_inv, mul_comm, mul_left_comm])

/-- In second-based coordinates the source uses an exponent cutoff proportional to t².
The coefficient on the right of k < c m t would carry units of inverse seconds squared. -/
theorem special_guard {k : WithDim (M𝓭 / T𝓭) ℝ} {m : WithDim M𝓭 ℝ} {t : Seconds ℝ}
    (hm : 0 < m.val) (ht : 0 < t.val) :
    k < frictionCutoff ⟨threshold⟩ m t ↔
      -(dragExponent k t m).val < threshold * t.val ^ 2 := by
  change k.val < threshold * m.val * t.val ↔ -(-(k.val * t.val / m.val)) < _
  rw [neg_neg, div_lt_iff₀ hm]
  rw [show threshold * t.val ^ 2 * m.val = (threshold * m.val * t.val) * t.val by ring]
  exact (mul_lt_mul_iff_left₀ ht).symm

/-- With friction/mass equal to the source threshold, a half- and a two-second step
give opposite answers under the source guard and a fixed dimensionless exponent cutoff. -/
theorem opposite_drag_decisions :
    (¬ threshold < threshold * 1 * (1/2)) ∧ threshold * (1/2) < threshold ∧
    threshold < threshold * 1 * 2 ∧ ¬ threshold * 2 < threshold := by
  norm_num [threshold]

/-- Host exponential and translated position/velocity for each partial-tick replay. -/
def dragWitness (twoSeconds : Bool) : Float × (Float × Float) :=
  let t := if twoSeconds then 2 else Float.ofBits 0x3fe0000000000000
  let q := Float.ofBits (if twoSeconds then 0x3fefffffffe48320 else 0x3feffffffff920c8)
  (q, Reference.binary64 (fun _ => q) 0 0 1 1 (Float.ofBits 0x3ddb7cdfd9d7bdbb) 0 t 1)

set_option maxRecDepth 16384 in
theorem drag_step_words :
    let half := (dragWitness false).2
    let two := (dragWitness true).2
    (half.1.toBits, half.2.toBits) = (0xc079db3acc1cd9fd, 0x3fe000001635e000) ∧
      (two.1.toBits, two.2.toBits) = (0, 0x400000001635e000) := by
  decide +kernel

open ExactBallpark.Collision.History

/-- The actual history predicate on the same recent collider and the same sweep,
first with its raw fraction and then with an elapsed-time argument. -/
def repeatDecisions (duration u : Float) : Bool × Bool :=
  let location (time : Float) (id : Int) : Location := ⟨time, id, ⟨fun _ => 0, fun _ => 0⟩⟩
  let history := [location 0 7, location duration (-1)]
  (ignoreCollision history 7 u, ignoreCollision history 7 (u * duration))

theorem repeat_window_words :
    let u := Float.ofBits 0x3f14f8b588e368f1 -- nearest binary64 to 0.00008
    let v := Float.ofBits 0x3f23a92a30553261 -- nearest binary64 to 0.00015
    repeatDecisions 1 u = (true, true) ∧ repeatDecisions 2 u = (true, false) ∧
      repeatDecisions (Float.ofBits 0x3fe0000000000000) v = (false, true) := by
  decide +kernel

end RealPhysics.TimeGuards

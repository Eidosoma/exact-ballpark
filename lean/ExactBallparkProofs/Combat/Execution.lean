-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Combat.Accuracy
public import ExactBallparkProofs.Collision.FiniteSweep

@[expose] public section
namespace ExactBallpark.Combat
open ExactBallpark.Collision ExactBallpark.Movement

/-- Successful GetAccuracy lane, through the input to the host power function. -/
def trackingTerms (v radial : FloatPoint) (resolution tracking signature : Float) :=
  let d := (dot64 radial radial).sqrt
  let radialSpeed := dot64 v radial / d
  let speedSq := (dot64 v v - radialSpeed * radialSpeed) / (d*d)
  let k := resolution / (tracking * signature)
  (speedSq, k * (-k * speedSq))

/-- Binary64 inputs to sqrt and pow in the radial replay. -/
def radialInputWords : UInt64 × UInt64 := (0xbbd83c977ab2bedd, 0x484e2a40f9945fd6)

def radialSightline : FloatPoint := ![3, 4, 12]

def radialVelocity : FloatPoint := ![Float.ofBits 0x3f9eb851eb851eb8, Float.ofBits 0x3fa47ae147ae147b,
                                     Float.ofBits 0x3fbeb851eb851eb8]

set_option maxRecDepth 16384 in
theorem accuracy_overflow_input :
    let (speedSq, e) := trackingTerms radialVelocity radialSightline (Float.ofBits 0x46293e5939a08cea) 1 1
    speedSq.toBits = radialInputWords.1 ∧ speedSq.sqrt.isNaN = true ∧
      e.toBits = radialInputWords.2 ∧ 1024 ≤ e := by decide +kernel

abbrev AccuracyWords := UInt64 × UInt64 × UInt64

/-- Early branches return the original arguments, including their sign and NaN bits. -/
def earlyWords (optimal tracking : Float) : AccuracyWords := (0, optimal.toBits, tracking.toBits)

def scalarAccuracy (same present : Bool) (optimal tracking signature falloff dist : Float)
    (ordinary : AccuracyWords) : AccuracyWords := Id.run do
  if same then return earlyWords optimal tracking
  if !present || optimal < 0 then return earlyWords optimal tracking
  if tracking == 0 || signature == 0 || falloff == 0 then return earlyWords optimal tracking
  if dist == 0 then return earlyWords optimal tracking
  return ordinary

def laneMask (same present : Bool) (optimal tracking signature falloff dist : Float) :=
  same || !present || decide (optimal < 0) || tracking == 0 || signature == 0 || falloff == 0 || dist == 0

theorem accuracy_early_lanes (same present : Bool) (o k s f d : Float) (ordinary : AccuracyWords) :
    scalarAccuracy same present o k s f d ordinary =
      (if laneMask same present o k s f d then (0, o.toBits, k.toBits) else ordinary) := by
  simp only [scalarAccuracy, Id.run, laneMask, earlyWords, Bool.or_eq_true, decide_eq_true_eq]
  split_ifs <;> simp_all <;> rfl

/-- The cross product here is omega × unit line of sight, in the source order. -/
def observerCross (w n : Vec3) : Vec3 := !₂[
  w 1 * n 2 - w 2 * n 1, w 2 * n 0 - w 0 * n 2, w 0 * n 1 - w 1 * n 0]

noncomputable def observerSpeed (tv w n : Vec3) (d : ℝ) := ‖tv - d • observerCross w n‖ / d

theorem rotating_observer (tv w n : Vec3) {d : ℝ} (hd : 0 < d) :
    observerSpeed tv w n d = ‖(1/d) • tv - observerCross w n‖ ∧
    (observerSpeed tv w n d = 0 ↔ tv = d • observerCross w n) := by
  constructor
  · have h : (1/d) • (tv - d • observerCross w n) =
        (1/d) • tv - observerCross w n := by
      rw [smul_sub, smul_smul, one_div, inv_mul_cancel₀ hd.ne', one_smul]
    rw [← h, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity), observerSpeed]
    ring
  · simp [observerSpeed, hd.ne', sub_eq_zero]

end ExactBallpark.Combat

-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Bounds
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Tactic.Positivity

@[expose] public noncomputable section
namespace ExactBallpark.Combat
open ExactBallpark.Movement
open scoped InnerProductSpace

/-- RelativeVelocities without dynamical orientation, with a unit line of sight. -/
def transversal (v n : Vec3) := v - ⟪v,n⟫_ℝ • n

/-- GetAccuracy independently computes this quantity without forming the transverse vector. -/
def angularSq (v radial : Vec3) :=
  (‖v‖^2 - (⟪v,radial⟫_ℝ / ‖radial‖)^2) / ‖radial‖^2

theorem angular_agrees (v radial : Vec3) (hr : radial ≠ 0) :
    angularSq v radial = (‖transversal v ((1 / ‖radial‖) • radial)‖ / ‖radial‖)^2 := by
  let n := (1 / ‖radial‖) • radial
  have h := norm_ne_zero_iff.mpr hr
  have hn : ‖n‖ = 1 := by simp [n, norm_smul, h]
  have proj : ⟪v,n⟫_ℝ = ⟪v,radial⟫_ℝ / ‖radial‖ := by rw [real_inner_smul_right]; ring
  change angularSq v radial = (‖transversal v n‖ / ‖radial‖)^2
  rw [angularSq, div_pow ‖transversal v n‖, transversal, norm_sub_sq_real, real_inner_smul_right,
    norm_smul, Real.norm_eq_abs, hn, mul_one, sq_abs, proj]
  ring

/-- The nondegenerate branch of FullBallpark.get_accuracy. -/
def exponent (short optimal falloff tracking signature optimalSignature speedSq : ℝ) :=
  let ratio := optimalSignature / (tracking * signature)
  let base := ratio * (-ratio * speedSq)
  if short > optimal then base - (short - optimal)^2 / falloff^2 else base

theorem accuracy_bounds (v radial : Vec3) (hr : radial ≠ 0)
    (short optimal falloff tracking signature optimalSignature : ℝ) :
    let e := exponent short optimal falloff tracking signature optimalSignature (angularSq v radial)
    0 < (2 : ℝ)^e ∧ (2 : ℝ)^e ≤ 1 := by
  dsimp only
  refine ⟨Real.rpow_pos_of_pos (by norm_num) _, Real.rpow_le_one_of_one_le_of_nonpos (by norm_num) ?_⟩
  rw [angular_agrees v radial hr]
  unfold exponent
  have penalty (a x : ℝ) : a * (-a * x^2) ≤ 0 := by nlinarith [sq_nonneg (a*x)]
  split
  · exact sub_nonpos_of_le (le_trans (penalty ..) (div_nonneg (sq_nonneg _) (sq_nonneg _)))
  · exact penalty ..

theorem optimal_stationary (d optimal falloff tracking signature optimalSignature : ℝ) (h : d ≤ optimal) :
    (2 : ℝ) ^ exponent d optimal falloff tracking signature optimalSignature 0 = 1 := by
  simp [exponent, not_lt.mpr h]

end ExactBallpark.Combat

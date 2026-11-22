-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Physics.FlightLimits
public import ExactBallparkProofs.Movement.StopDistance

@[expose] public noncomputable section
namespace RealPhysics.FrameAndOrbit
open ExactBallpark.Movement
open scoped InnerProductSpace

/-- A medium moving at b contributes (k/m)b to the still-medium acceleration
input. The ordinary flow then translates exactly with the medium. -/
theorem moving_medium (p v a t b : ℝ) {m k : ℝ} (hm : 0 < m) (hk : 0 < k) :
    position p (v+b) (a+k/m*b) m k t = position p v a m k t + b*t ∧
    velocity (v+b) (a+k/m*b) m k t = velocity v a m k t + b := by
  have eq : terminal m k (a+k/m*b) = terminal m k a + b := by
    unfold terminal
    field_simp
  unfold position velocity
  rw [eq]
  constructor <;> ring

/-- The extra vertical STOP multiplier cannot be the same constant-rate decay
at both half a second and one second. The ordinary exponential drag factors out. -/
theorem stop_has_no_fixed_rate :
    ¬ ∃ γ : ℝ, Real.exp (-γ/2) = StopDistance.damping true ∧
      Real.exp (-γ) = StopDistance.damping true := by
  rintro ⟨γ, half, full⟩
  have square := Real.exp_add (-γ/2) (-γ/2)
  rw [show -γ/2 + -γ/2 = -γ by ring, full, half] at square
  norm_num [StopDistance.damping, stopVertical] at square

def requiredSq (radius rate speed : ℝ) := (speed^2/radius)^2 + (rate*speed)^2

/-- The inward unit normal and the unit tangent are perpendicular. Thrust must
supply both the centripetal acceleration and the speed lost to drag. -/
theorem circle_thrust (inward tangent : Vec3) (hn : ‖inward‖ = 1) (ht : ‖tangent‖ = 1)
    (orth : ⟪inward,tangent⟫_ℝ = 0) (radius rate speed : ℝ) :
    let thrust := (speed^2/radius) • inward + (rate*speed) • tangent
    thrust - rate • (speed • tangent) = (speed^2/radius) • inward ∧
      ‖thrust‖^2 = requiredSq radius rate speed := by
  dsimp only
  constructor
  · rw [smul_smul, add_sub_cancel_right]
  · simp only [norm_add_sq_real, real_inner_smul_left, real_inner_smul_right, orth,
      mul_zero, add_zero, norm_smul, Real.norm_eq_abs, hn, ht, mul_one, sq_abs, requiredSq]

/-- Exact squared-speed ceiling for an ideal steady circle with a fixed thrust budget. -/
theorem circle_budget_iff {radius : ℝ} (hr : 0 < radius) (rate speed budget : ℝ) :
    requiredSq radius rate speed ≤ budget^2 ↔
      speed^2 ≤ radius/2 * (Real.sqrt ((radius*rate^2)^2 + 4*budget^2) - radius*rate^2) := by
  let x := 2*speed^2/radius + radius*rate^2
  have hx : 0 ≤ x := by dsimp [x]; positivity
  have hrhs : 0 ≤ (radius*rate^2)^2 + 4*budget^2 := by positivity
  have complete : x^2 = (radius*rate^2)^2 + 4*requiredSq radius rate speed := by
    dsimp [x, requiredSq]
    field_simp
    ring
  calc
    _ ↔ x^2 ≤ (radius*rate^2)^2 + 4*budget^2 := by rw [complete]; constructor <;> intro h <;> linarith
    _ ↔ x ≤ Real.sqrt ((radius*rate^2)^2 + 4*budget^2) := by
      rw [← Real.sq_sqrt hrhs, sq_le_sq₀ hx (Real.sqrt_nonneg _), Real.sqrt_sq (Real.sqrt_nonneg _)]
    _ ↔ _ := by
      dsimp [x]
      rw [← le_sub_iff_add_le, div_le_iff₀ hr]
      constructor <;> intro h <;> linarith only [h]

/-- When curvature sets the engine's reference speed, it spends the entire
budget on centripetal acceleration. Positive linear drag then makes the steady
circle require strictly more thrust. -/
theorem curvature_target_exceeds_budget {radius rate budget : ℝ}
    (hr : 0 < radius) (hβ : 0 < rate) (hA : 0 < budget) :
    budget^2 < requiredSq radius rate (Real.sqrt (radius*budget)) := by
  have prod := mul_pos hr hA
  have tangential := sq_pos_of_pos (mul_pos hβ (Real.sqrt_pos.mpr prod))
  rw [requiredSq, Real.sq_sqrt prod.le, mul_div_cancel_left₀ _ hr.ne']
  linarith

end RealPhysics.FrameAndOrbit

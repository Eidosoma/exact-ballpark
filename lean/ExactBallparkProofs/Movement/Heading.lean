-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Flight

@[expose] public noncomputable section
namespace ExactBallpark.Movement

/-- Row two of m_rotation_q is the image of the forward vector (0,0,1). -/
def forwardRow (x y z w : ℝ) : Vec3 := !₂[
  2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)]

/-- The last two operations of ApplyTorque are q_len and, for a positive norm, q_div. -/
theorem normalized_heading (x y z w accel cosine : ℝ)
    (nonzero : 0 < x*x + y*y + z*z + w*w) (ha : 0 ≤ accel) :
    let n := Real.sqrt (x*x + y*y + z*z + w*w)
    let heading := forwardRow (x/n) (y/n) (z/n) (w/n)
    ‖heading‖ = 1 ∧ ‖orientedThrust heading accel cosine‖ ≤ accel := by
  let n := Real.sqrt (x*x + y*y + z*z + w*w)
  have hn : n ≠ 0 := (Real.sqrt_pos.2 nonzero).ne'
  have unit : (x/n)^2 + (y/n)^2 + (z/n)^2 + (w/n)^2 = 1 := by
    rw [div_pow, div_pow, div_pow, div_pow]
    field_simp
    nlinarith [Real.sq_sqrt nonzero.le]
  have row (a b c d : ℝ) (hu : a^2+b^2+c^2+d^2 = 1) : ‖forwardRow a b c d‖ = 1 := by
    rw [EuclideanSpace.norm_eq]
    simp [forwardRow, Fin.sum_univ_succ, Real.norm_eq_abs, sq_abs]
    linear_combination 4*(a^2+b^2)*hu
  exact ⟨row _ _ _ _ unit, orientation_bound _ (row _ _ _ _ unit) _ _ ha⟩

end ExactBallpark.Movement

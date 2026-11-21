-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Flight
public import Mathlib.Tactic.LinearCombination

@[expose] public noncomputable section
namespace ExactBallpark.Movement

/-- Row-vector multiplication by m_rotation_q. The homogeneous divisor in
transform_coord is one for this matrix. Coordinates are in source order x,y,z,w. -/
def rotateSlot (x y z w : ℝ) (v : Vec3) : Vec3 := !₂[
  v 0*(1-2*(y*y+z*z)) + v 1*(2*(x*y-z*w)) + v 2*(2*(x*z+y*w)),
  v 0*(2*(x*y+z*w)) + v 1*(1-2*(x*x+z*z)) + v 2*(2*(y*z-x*w)),
  v 0*(2*(x*z-y*w)) + v 1*(2*(y*z+x*w)) + v 2*(1-2*(x*x+y*y))]

theorem rotateSlot_norm (x y z w : ℝ) (unit : x^2+y^2+z^2+w^2 = 1) (v : Vec3) :
    ‖rotateSlot x y z w v‖ = ‖v‖ := by
  rw [← sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _), EuclideanSpace.real_norm_sq_eq,
    EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three, Fin.sum_univ_three]
  change (_ + _ + _) ^ 2 + (_ + _ + _) ^ 2 + (_ + _ + _) ^ 2 = _
  dsimp [rotateSlot]
  linear_combination 4 * ((y*v 2-z*v 1)^2 + (z*v 0-x*v 2)^2 + (x*v 1-y*v 0)^2) * unit

theorem rotated_formation_distance (x y z w : ℝ) (unit : x^2+y^2+z^2+w^2 = 1)
    (leader slot : Vec3) (r₁ r₂ : ℝ) :
    ‖formationTarget (rotateSlot x y z w) leader slot r₁ r₂ - leader‖ =
      if slot = 0 then 0 else |r₁ + r₂ + ‖slot‖| :=
  formation_distance _ (rotateSlot_norm x y z w unit) leader slot r₁ r₂

end ExactBallpark.Movement

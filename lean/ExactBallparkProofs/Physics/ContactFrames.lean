-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.LimitedPair
public import ExactBallparkProofs.Combat.Accuracy

@[expose] public noncomputable section
namespace ExactBallpark.Physics.ContactFrames
open Collision.LimitedPair Movement Combat

/-- With equal and opposite normal impulses, energy conservation and separation
pick exactly the raw pair impulse, before either body's cap. -/
theorem elastic_impulse_iff {m n u v : ℝ} (hm : 0 < m) (hn : 0 < n) (closing : u < v) (j : ℝ) :
    (energy m n (u+j/m) (v-j/n) = energy m n u v ∧ v-j/n ≤ u+j/m) ↔ j = impulse m n u v := by
  have loss : energy m n (u+j/m) (v-j/n) - energy m n u v =
      j * (j - impulse m n u v) * ((m+n)/(2*m*n)) := by
    unfold energy impulse
    field_simp
    ring
  have pos : 0 < (m+n)/(2*m*n) := by positivity
  constructor
  · rintro ⟨h, sep⟩
    rw [h, sub_self, eq_comm, mul_eq_zero, or_iff_left pos.ne', mul_eq_zero] at loss
    rcases loss with hj | hj
    · simp [hj] at sep
      exact (closing.not_ge sep).elim
    · exact sub_eq_zero.mp hj
  · rintro rfl
    constructor
    · apply sub_eq_zero.mp
      simp [loss]
    · have swap : u + impulse m n u v / m - (v - impulse m n u v / n) = v-u := by
        unfold impulse
        field_simp
        ring
      linarith

/-- Kinetic-energy *changes* of a given pair seen with every velocity shifted by b. -/
theorem boost_energy_change (m n u v w z b : ℝ) :
    energy m n (w+b) (z+b) - energy m n (u+b) (v+b) =
      energy m n w z - energy m n u v + b*(m*(w-u)+n*(z-v)) := by
  unfold energy
  ring

/-- Internal kinetic energy measured about each state's own centre of mass. -/
theorem internal_energy {m n : ℝ} (hm : 0 < m) (hn : 0 < n) (u v : ℝ) :
    let centre := (m*u+n*v)/(m+n)
    energy m n (u-centre) (v-centre) = m*n/(2*(m+n))*(u-v)^2 := by
  dsimp [energy]
  field_simp
  ring

/-- Re-express the velocities actually produced by the unequal-cap witness.
The +100 m/s observer and the two instantaneous centres of mass are distinct. -/
theorem limited_energy_frames :
    let (w,z) := limited 1 1 (-100) (-90) 1 2
    energy 1 1 (w+100) (z+100) - energy 1 1 0 10 = -(375/8) ∧
      (w-z)^2/4 - ((-100 : ℝ)-(-90))^2/4 = -(375/16) := by
  norm_num [limited, applyFractions, retain, impulse, energy]

/-- The angular-speed square in the regular accuracy query under a common
translation, velocity boost and orthogonal change of axes. -/
theorem angular_frame (R : Vec3 ≃ₗᵢ[ℝ] Vec3) (p q u v offset boost : Vec3) :
    angularSq ((R v + boost) - (R u + boost)) ((R q + offset) - (R p + offset)) =
      angularSq (v-u) (q-p) := by
  simp only [add_sub_add_right_eq_sub, ← R.map_sub, angularSq, R.norm_map, R.inner_map_map]

end ExactBallpark.Physics.ContactFrames

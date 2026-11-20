-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.Response

/-! Velocity immediately after the iterative impulse, before AddImpact limiting and
subsequent integration. The overlap correction is from the separate gradient path. -/
@[expose] public noncomputable section
namespace ExactBallpark.Collision
open ExactBallpark.Movement
open scoped InnerProductSpace

def staticVelocity (v n : Vec3) :=
  let impulse := ⟪(-2 : ℝ) • v, n⟫_ℝ
  if impulse < 0 then v else v + impulse • n

/-- Outward motion is kept, inward motion is reflected, and the tangent plane is unaffected. -/
theorem static_velocity_components (v n : Vec3) (unit : ‖n‖ = 1) :
    ⟪staticVelocity v n, n⟫_ℝ = |⟪v,n⟫_ℝ| ∧
    ∀ w : Vec3, ⟪n,w⟫_ℝ = 0 → ⟪staticVelocity v n,w⟫_ℝ = ⟪v,w⟫_ℝ := by
  simp only [staticVelocity, real_inner_smul_left]
  split
  next h => simp [abs_of_nonneg (by linarith : 0 ≤ ⟪v,n⟫_ℝ)]
  next h =>
    rw [inner_add_left, real_inner_smul_left, real_inner_self_eq_norm_sq, unit,
      abs_of_nonpos (by linarith : ⟪v,n⟫_ℝ ≤ 0)]
    exact ⟨by ring, fun w hw => by simp [inner_add_left, real_inner_smul_left, hw]⟩

/-- At a start contact the replacement collision acceleration cancels the
existing normal acceleration and substitutes the configured thrust per inertia. -/
theorem start_contact_acceleration (g n : Vec3) (speed k mass agility : ℝ) (unit : ‖n‖ = 1) :
    let desired := speed*k/(mass*agility)
    let correction := (desired - ⟪g,n⟫_ℝ) • n
    ⟪g + correction,n⟫_ℝ = desired ∧
      ∀ w : Vec3, ⟪n,w⟫_ℝ = 0 → ⟪g + correction,w⟫_ℝ = ⟪g,w⟫_ℝ := by
  dsimp only
  rw [inner_add_left, real_inner_smul_left, real_inner_self_eq_norm_sq, unit]
  exact ⟨by ring, fun w hw => by simp [inner_add_left, real_inner_smul_left, hw]⟩

/-- Ballpark::_impact_free uses the second ball's opposite normal to compute the guard. -/
def pairVelocity (m₁ m₂ : ℝ) (u v n : Vec3) : Vec3 × Vec3 :=
  let impulse := -2 * ⟪v-u,-n⟫_ℝ
  if impulse < 0 then (u,v) else
    let j := m₁*m₂*impulse/(m₁+m₂)
    (u + (j/m₁) • n, v + (j/m₂) • (-n))

/-- The post-impact relative normal speed is outward or zero, including the skip branch. -/
theorem pair_separates {m₁ m₂ : ℝ} (h₁ : 0 < m₁) (h₂ : 0 < m₂)
    (u v n : Vec3) (unit : ‖n‖ = 1) :
    let (u',v') := pairVelocity m₁ m₂ u v n
    ⟪u'-v',n⟫_ℝ = |⟪u-v,n⟫_ℝ| := by
  simp only [pairVelocity, inner_neg_right, inner_sub_left]
  split
  next h => rw [abs_of_nonneg (by linarith : 0 ≤ ⟪u,n⟫_ℝ - ⟪v,n⟫_ℝ)]
  next h =>
    simp only [inner_add_left, real_inner_smul_left, inner_neg_left, real_inner_self_eq_norm_sq, unit]
    rw [abs_of_nonpos (by linarith : ⟪u,n⟫_ℝ - ⟪v,n⟫_ℝ ≤ 0)]
    field_simp
    ring

theorem pair_preserves_tangent (m₁ m₂ : ℝ) (u v n w : Vec3) (hw : ⟪n,w⟫_ℝ = 0) :
    let (u',v') := pairVelocity m₁ m₂ u v n
    ⟪u',w⟫_ℝ = ⟪u,w⟫_ℝ ∧ ⟪v',w⟫_ℝ = ⟪v,w⟫_ℝ := by
  by_cases skip : -2 * ⟪v-u,-n⟫_ℝ < 0
  · simp only [pairVelocity, skip, ↓reduceIte, and_self]
  · simp only [pairVelocity, skip, ↓reduceIte]
    simp [inner_add_left, real_inner_smul_left, hw]

theorem pair_momentum {m₁ m₂ : ℝ} (h₁ : 0 < m₁) (h₂ : 0 < m₂) (u v n : Vec3) :
    let (u',v') := pairVelocity m₁ m₂ u v n
    m₁ • u' + m₂ • v' = m₁ • u + m₂ • v := by
  by_cases skip : -2 * ⟪v-u,-n⟫_ℝ < 0
  · simp only [pairVelocity, skip, ↓reduceIte]
  · simp only [pairVelocity, skip, ↓reduceIte, smul_add, smul_smul, mul_div_cancel₀ _ h₁.ne', mul_div_cancel₀ _ h₂.ne', smul_neg]
    abel

theorem pair_kinetic_energy {m₁ m₂ : ℝ} (h₁ : 0 < m₁) (h₂ : 0 < m₂)
    (u v n : Vec3) (unit : ‖n‖ = 1) :
    let (u',v') := pairVelocity m₁ m₂ u v n
    m₁ * ‖u'‖^2 + m₂ * ‖v'‖^2 = m₁ * ‖u‖^2 + m₂ * ‖v‖^2 := by
  by_cases skip : -2 * ⟪v-u,-n⟫_ℝ < 0
  · simp only [pairVelocity, skip, ↓reduceIte]
  · simp only [pairVelocity, skip, ↓reduceIte]
    simp only [inner_neg_right, inner_sub_left, norm_add_sq_real, real_inner_smul_right,
      norm_smul, Real.norm_eq_abs, norm_neg, unit, mul_one, sq_abs]
    field_simp
    ring

/-- Temporary positions used to re-test a free-free overlap. The inertias include agility. -/
def splitPositions (p q n : Vec3) (m₁ m₂ radius distance : ℝ) :=
  let gap := radius - distance + 1
  (p - (m₂/(m₁+m₂)*gap) • n, q + (m₁/(m₁+m₂)*gap) • n)

theorem overlap_clearance {m₁ m₂ radius distance : ℝ} (h₁ : 0 < m₁) (h₂ : 0 < m₂)
    (hr : 0 ≤ radius) (p q n : Vec3) (unit : ‖n‖ = 1) (direction : q-p = distance • n) :
    let (p',q') := splitPositions p q n m₁ m₂ radius distance
    ‖q'-p'‖ = radius + 1 := by
  dsimp [splitPositions]
  rw [show ∀ a b : ℝ, q + a • n - (p - b • n) = (distance + a + b) • n by
    intro a b; rw [add_smul, add_smul, ← direction]; abel]
  rw [norm_smul, Real.norm_eq_abs, unit, mul_one]
  have h := (overlap_split h₁ h₂ radius distance).1
  rw [show distance + m₁/(m₁+m₂)*(radius-distance+1) + m₂/(m₁+m₂)*(radius-distance+1) =
    radius+1 by linarith]
  exact abs_of_nonneg (by linarith)

theorem overlap_inertia_center {m₁ m₂ : ℝ} (h₁ : 0 < m₁) (h₂ : 0 < m₂)
    (p q n : Vec3) (radius distance : ℝ) :
    let (p',q') := splitPositions p q n m₁ m₂ radius distance
    (m₁+m₂)⁻¹ • (m₁ • p' + m₂ • q') = (m₁+m₂)⁻¹ • (m₁ • p + m₂ • q) := by
  dsimp [splitPositions]
  apply congrArg (fun x : Vec3 => (m₁+m₂)⁻¹ • x)
  simp only [smul_sub, smul_add, smul_smul]
  rw [(overlap_split h₁ h₂ radius distance).2]
  abel

theorem limited_speed_bound (dv : Vec3) {maxVel : ℝ} (hm : 0 ≤ maxVel) :
    ‖limitImpact dv maxVel‖ ≤ min ‖dv‖ (5*maxVel) := by
  by_cases zero : dv = 0
  · simp [zero, limitImpact, hm]
  rw [limitImpact_norm dv maxVel zero hm]
  have pos := norm_pos_iff.mpr zero
  have nonneg : 0 ≤ 5*maxVel := by positivity
  split
  next active =>
    have bound : (5*maxVel)^2 / ‖dv‖ ≤ 5*maxVel := by
      rw [div_le_iff₀ pos, sq]
      exact mul_le_mul_of_nonneg_left active.le nonneg
    exact le_min (bound.trans active.le) bound
  next inactive => exact le_min le_rfl (le_of_not_gt inactive)

end ExactBallpark.Collision

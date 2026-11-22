-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Combat.Accuracy

@[expose] public noncomputable section
namespace ExactBallpark.Combat.Falloff
open Movement
open scoped InnerProductSpace

def surface (distance ownRadius otherRadius : ℝ) := max 0 ((distance-otherRadius)-ownRadius)

def rangeCost (short optimal falloff : ℝ) := (max 0 (short-optimal) / falloff)^2
def trackingCost (v r : Vec3) (tracking sig res : ℝ) :=
  (res/(tracking*sig))^2 * angularSq v r

/-- Adding any component along the line of sight leaves the real angular term unchanged. -/
theorem radial_invariance (v r : Vec3) (hr : r ≠ 0) (c : ℝ) :
    angularSq (v + c • r) r = angularSq v r := by
  have nz := norm_ne_zero_iff.mpr hr
  rw [angularSq,angularSq,norm_add_sq_real,inner_add_left,real_inner_smul_left,
    real_inner_smul_right,real_inner_self_eq_norm_sq,norm_smul, Real.norm_eq_abs,mul_pow,sq_abs]
  field_simp
  ring

private theorem exponent_cost (v r : Vec3) (short opt fall tr sig res : ℝ) :
    exponent short opt fall tr sig res (angularSq v r) =
      -trackingCost v r tr sig res - rangeCost short opt fall := by
  by_cases beyond : opt < short
  · simp [exponent,beyond,rangeCost,trackingCost,max_eq_right (sub_nonneg.mpr beyond.le),div_pow]; ring
  · simp [exponent,beyond,rangeCost,trackingCost,max_eq_left (sub_nonpos.mpr (le_of_not_gt beyond))]; ring

theorem chance_factors (v r : Vec3) (d r₁ r₂ opt fall tr sig res : ℝ) :
    (2:ℝ)^exponent (surface d r₁ r₂) opt fall tr sig res (angularSq v r) =
      (2:ℝ)^(-trackingCost v r tr sig res) * (2:ℝ)^(-rangeCost (surface d r₁ r₂) opt fall) := by
  rw [exponent_cost,sub_eq_add_neg,Real.rpow_add (by norm_num : (0:ℝ)<2)]

/-- With the angular term held fixed, range is strictly worse once the farther
surface distance exceeds optimal and the surface distances differ. -/
theorem range_order (v r : Vec3) {x y : ℝ} (xy : x ≤ y) (opt : ℝ) {fall : ℝ} (hf : fall ≠ 0)
    (tr sig res : ℝ) :
    let chance := fun s => (2:ℝ)^exponent s opt fall tr sig res (angularSq v r)
    chance y ≤ chance x ∧ (chance y < chance x ↔ x < y ∧ opt < y) := by
  have hx : 0 ≤ max 0 (x-opt) := le_max_left ..
  have hy : 0 ≤ max 0 (y-opt) := le_max_left ..
  have ord : max 0 (x-opt) ≤ max 0 (y-opt) := max_le_max_left _ (by linarith)
  have strict : max 0 (x-opt) < max 0 (y-opt) ↔ x < y ∧ opt < y := by
    by_cases h : opt < y
    · rw [max_eq_right (by linarith : 0 ≤ y-opt),max_lt_iff]
      constructor <;> rintro ⟨a,b⟩ <;> constructor <;> linarith
    · simp [max_eq_left (by linarith : y-opt ≤ 0),h,not_lt_of_ge hx]
  dsimp only
  simp only [exponent_cost,Real.rpow_le_rpow_left_iff (by norm_num : (1:ℝ)<2),
    Real.rpow_lt_rpow_left_iff (by norm_num : (1:ℝ)<2),sub_le_sub_iff_left,sub_lt_sub_iff_left,
    rangeCost,div_pow,div_le_div_iff_of_pos_right (sq_pos_of_ne_zero hf),
    div_lt_div_iff_of_pos_right (sq_pos_of_ne_zero hf), sq_le_sq₀ hx hy, sq_lt_sq₀ hx hy]
  exact ⟨ord,strict⟩

theorem half_chance_iff (v r : Vec3) (s opt fall tr sig res : ℝ) :
    1/2 ≤ (2:ℝ)^exponent s opt fall tr sig res (angularSq v r) ↔
      trackingCost v r tr sig res + rangeCost s opt fall ≤ 1 := by
  rw [show (1/2:ℝ) = (2:ℝ)^(-1:ℝ) by norm_num,exponent_cost,
    Real.rpow_le_rpow_left_iff (by norm_num : (1:ℝ)<2)]
  constructor <;> intro h <;> linarith

/-- A radial relative velocity contributes zero angular penalty. One and two
falloff lengths beyond optimal are measured from the surfaces. -/
theorem falloff_landmarks (r : Vec3) (hr : r ≠ 0) (c opt r₁ r₂ tr sig res : ℝ)
    (ho : 0 ≤ opt) {fall : ℝ} (hf : 0 < fall) :
    let chance := fun d => (2:ℝ)^exponent (surface d r₁ r₂) opt fall tr sig res (angularSq (c • r) r)
    chance (r₁+r₂+opt+fall) = 1/2 ∧ chance (r₁+r₂+opt+2*fall) = 1/16 := by
  have radial := radial_invariance (0:Vec3) r hr c
  simp [angularSq] at radial
  have one : surface (r₁+r₂+opt+fall) r₁ r₂ = opt+fall := by
    dsimp [surface]; rw [max_eq_right (by linarith)]; ring
  have two : surface (r₁+r₂+opt+2*fall) r₁ r₂ = opt+2*fall := by
    dsimp [surface]; rw [max_eq_right (by linarith)]; ring
  have zero : angularSq (c • r) r = 0 := by simpa [angularSq] using radial
  dsimp only
  rw [one,two,zero]
  have e (s : ℝ) (h : 0 < s) : exponent (opt+s*fall) opt fall tr sig res 0 = -s^2 := by
    simp [exponent, show opt < opt+s*fall by linarith [mul_pos h hf],mul_pow,hf.ne']
  rw [show opt+fall=opt+1*fall by ring, e 1 (by norm_num),e 2 (by norm_num)]
  norm_num [show (-(2:ℝ)^2) = ((-4:ℤ):ℝ) by norm_num,Real.rpow_intCast]

end ExactBallpark.Combat.Falloff

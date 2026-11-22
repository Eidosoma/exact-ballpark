-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Guidance
public import Mathlib.Analysis.SpecialFunctions.Log.Basic

@[expose] public noncomputable section
namespace ExactBallpark.Movement.OrbitSteering
open scoped InnerProductSpace

/-- Both inverses used by GotoThrustFollow exist for the exact positive-duration decay. -/
theorem guidance_denominators {f t : ℝ} (hf : 0 < f) (ht : 0 < t) :
    0 < 1 - Real.exp (-(f*t)) ∧ 0 < t*f - (1-Real.exp (-(f*t))) := by
  have prod := mul_pos hf ht
  constructor
  · exact sub_pos.mpr (Real.exp_lt_one_iff.mpr (neg_lt_zero.mpr prod))
  · linarith [Real.add_one_lt_exp (show -(f*t) ≠ 0 from neg_ne_zero.mpr prod.ne')]

def blend (p v dst dstV f q t weight : ℝ) :=
  (reachAccel p v dst f q t + matchAccel v dstV f q * weight) * (1 / (1+weight))

/-- Before clipping, each endpoint error is the corresponding error of the
other controller, multiplied by its share of the blend. -/
theorem blend_residuals (p v dst dstV f q t w : ℝ) (hf : f ≠ 0)
    (hq : 1-q ≠ 0) (ht : t*f-(1-q) ≠ 0) (hw : 0 ≤ w) :
    let a := blend p v dst dstV f q t w
    regularPosition p v a 1 f q t - dst =
      w/(1+w)*(regularPosition p v (matchAccel v dstV f q) 1 f q t - dst) ∧
    sourceVelocity v a 1 f q - dstV =
      (sourceVelocity v (reachAccel p v dst f q t) 1 f q - dstV)/(1+w) := by
  obtain ⟨hp,hv⟩ := solve_guidance p v dst dstV f q t hf hq ht
  dsimp only
  unfold blend
  generalize reachAccel p v dst f q t = ar at hp ⊢
  generalize matchAccel v dstV f q = av at hv ⊢
  have hw' : 1+w ≠ 0 := by positivity
  constructor
  · rw [← hp]
    unfold regularPosition
    field_simp
    ring
  · rw [← hv]
    unfold sourceVelocity
    field_simp
    ring

def clip (a : Vec3) (limit : ℝ) : Vec3 :=
  if ‖a‖^2 > limit^2 then (limit / Real.sqrt (‖a‖^2)) • a else a

theorem clipped_orbit_speed (a v : Vec3) {m k speed t : ℝ} (hm : 0 < m) (hk : 0 < k)
    (hs : 0 ≤ speed) (ht : 0 ≤ t) (hv : ‖v‖ ≤ speed) :
    let limit := k/m*speed
    ‖clip a limit‖ = min ‖a‖ limit ∧
    ‖velocity3 v (clip a limit) m k t‖ ≤ speed := by
  let lim := k/m*speed
  have hl : 0 ≤ lim := mul_nonneg (div_pos hk hm).le hs
  have norm : ‖clip a lim‖ = min ‖a‖ lim := by
    unfold clip
    simp only [gt_iff_lt,sq_lt_sq₀ hl (norm_nonneg a), Real.sqrt_sq (norm_nonneg a)]
    split
    next h =>
      rw [min_eq_right h.le,norm_smul,Real.norm_eq_abs,abs_of_nonneg (div_nonneg hl (norm_nonneg a)),
        div_mul_cancel₀ _ (hl.trans_lt h).ne']
    next h => rw [min_eq_left (le_of_not_gt h)]
  refine ⟨norm,speed_envelope hm hk ht hv ?_⟩
  change ‖(m/k) • clip a lim‖ ≤ speed
  rw [norm_smul, Real.norm_eq_abs,abs_of_pos (div_pos hm hk),norm]
  calc _ ≤ m/k*lim := mul_le_mul_of_nonneg_left (min_le_right ..) (div_pos hm hk).le
       _ = speed := by dsimp [lim]; field_simp

theorem weight_order {r d e : ℝ} (closer : |r-d| ≤ |r-e|) : orbitWeight r e ≤ orbitWeight r d := by
  have sq := (sq_le_sq).2 closer
  unfold orbitWeight
  apply div_le_div_of_nonneg_right _ (by norm_num)
  exact_mod_cast Int.floor_le_floor (mul_le_mul_of_nonneg_right
    (Real.exp_le_exp.mpr (by nlinarith)) (by norm_num : (0:ℝ) ≤ 10000000))

/-- The positive Gaussian becomes exactly zero after seven-place truncation. -/
theorem weight_cutoff (r d : ℝ) :
    orbitWeight r d = 0 ↔ 40000 * Real.log 10000000 < (r-d)^2 := by
  let x := -(r-d)*(r-d)/40000
  have pos : 0 ≤ Real.exp x * 10000000 := by positivity
  have floor0 : ⌊Real.exp x * 10000000⌋ = 0 ↔ Real.exp x * 10000000 < 1 := by
    rw [Int.floor_eq_iff]; simp [pos]
  change (⌊Real.exp x * 10000000⌋ : ℝ) / 10000000 = 0 ↔ _
  rw [div_eq_zero_iff, or_iff_left (by norm_num),Int.cast_eq_zero,floor0]
  have thresh : Real.exp (-Real.log 10000000) = (10000000:ℝ)⁻¹ := by
    rw [Real.exp_neg, Real.exp_log (by norm_num)]
  rw [← lt_div_iff₀ (by norm_num : (0:ℝ)<10000000),one_div,← thresh, Real.exp_lt_exp]
  dsimp [x]; constructor <;> intro h <;> linarith [show -(r-d)*(r-d) = -(r-d)^2 by ring]

/-- Old-style orbit, outer branch, after both steering directions have been normalized. -/
theorem outer_thrust_norm (w toward : Vec3) (hw : ‖w‖ = 1) (ht : ‖toward‖ = 1)
    {rf : ℝ} (lo : 0 ≤ rf) (hi : rf ≤ 1) :
    let phi := ⟪-toward,w⟫_ℝ
    let rad := 1+rf^2*(phi^2-1)
    let tf := if 0 < rad then rf*phi + Real.sqrt rad else rf*phi
    ‖rf • w + tf • toward‖ = 1 := by
  let phi := ⟪-toward,w⟫_ℝ
  let rad := 1+rf^2*(phi^2-1)
  have nonneg : 0 ≤ rad := by dsimp [rad]; nlinarith [sq_nonneg (rf*phi),mul_nonneg (sub_nonneg.mpr hi) (by linarith : 0 ≤ 1+rf)]
  have coeff : (if 0 < rad then rf*phi+Real.sqrt rad else rf*phi) = rf*phi+Real.sqrt rad := by
    split
    · rfl
    next h => simp [le_antisymm (le_of_not_gt h) nonneg]
  change ‖rf • w + (if 0 < rad then rf*phi+Real.sqrt rad else rf*phi) • toward‖ = 1
  rw [coeff]
  apply (sq_eq_sq₀ (norm_nonneg _) (by norm_num : (0:ℝ) ≤ 1)).mp
  rw [norm_add_sq_real, real_inner_smul_left,real_inner_smul_right,norm_smul,norm_smul,
    hw,ht,mul_one,mul_one,Real.norm_eq_abs,Real.norm_eq_abs,sq_abs,sq_abs]
  have inner : ⟪w,toward⟫_ℝ = -phi := by simp [phi,real_inner_comm]
  rw [inner]
  nlinarith only [Real.sq_sqrt nonneg, show rad = 1+rf^2*(phi^2-1) from rfl]

end ExactBallpark.Movement.OrbitSteering

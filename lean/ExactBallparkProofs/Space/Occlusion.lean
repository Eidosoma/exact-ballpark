-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Space.Queries
public import ExactBallparkProofs.Collision.Response

@[expose] public noncomputable section
namespace ExactBallpark.Space
open ExactBallpark.Movement

/-- Source and destination are distinct. A strict interior entry is the first
sphere-surface intersection on the entire oriented line. This handles tangency
and rejects a sphere already surrounding the origin even if it has an exit on
the segment. Distance t is measured along the unit ray from the source. -/
theorem ray_first_entry (src dst center : Vec3) (radius : ℝ) (hr : 0 ≤ radius)
    (different : dst ≠ src) :
    let ray := dst-src
    let d := ‖ray‖
    let n := d⁻¹ • ray
    let c := center-src
    let o : Occluder := ⟨0,true,inner ℝ n c,‖c‖^2,radius⟩
    blocks o d ↔ ∃ t : ℝ, 0 < t ∧ t < d ∧ ‖t • n - c‖ = radius ∧
      ∀ earlier < t, ‖earlier • n - c‖ ≠ radius := by
  let ray := dst-src
  let d := ‖ray‖
  let n := d⁻¹ • ray
  let c := center-src
  let v := inner ℝ n c
  let disc := radius^2-(‖c‖^2-v^2)
  have hd : 0 < d := norm_pos_iff.mpr (sub_ne_zero.mpr different)
  have unit : ‖n‖ = 1 := by simp [n, norm_smul, d, hd.ne']
  have root (t : ℝ) : ‖t • n - c‖ = radius ↔ (t-v)^2 = disc := by
    have ident : ‖t • n - c‖^2 - radius^2 = (t-v)^2-disc := by
      rw [norm_sub_sq_real, real_inner_smul_left, norm_smul, Real.norm_eq_abs, unit, mul_one, sq_abs]
      dsimp [v,disc]; ring
    constructor <;> intro eq
    · rw [eq] at ident; linarith
    · nlinarith [norm_nonneg (t • n - c)]
  change blocks ⟨0,true,v,‖c‖^2,radius⟩ d ↔ _
  rw [visibility_entry]
  change (true = true ∧ 0 ≤ disc ∧ 0 < v-Real.sqrt disc ∧ v-Real.sqrt disc < d) ↔ _
  constructor
  · rintro ⟨_,ok,lo,hi⟩
    refine ⟨v-Real.sqrt disc,lo,hi,(root _).2 (by nlinarith [Real.sq_sqrt ok]),?_⟩
    intro e he eq
    have heq := (root e).1 eq
    nlinarith [Real.sq_sqrt ok,Real.sqrt_nonneg disc,
      mul_pos (show 0 < v-e-Real.sqrt disc by linarith) (show 0 < v-e+Real.sqrt disc by linarith [Real.sqrt_nonneg disc])]
  · rintro ⟨t,lo,hi,contact,first⟩
    have ht := (root t).1 contact
    have ok : 0 ≤ disc := ht ▸ sq_nonneg _
    have he : ‖(v-Real.sqrt disc) • n - c‖ = radius := (root _).2 (by nlinarith [Real.sq_sqrt ok])
    have ordered : t ≤ v-Real.sqrt disc := le_of_not_gt (fun lt => first _ lt he)
    have eq : t = v-Real.sqrt disc := by
      have bound : -(Real.sqrt disc) ≤ t-v := by
        by_contra! h
        nlinarith [Real.sq_sqrt ok,Real.sqrt_nonneg disc,
          mul_pos (show 0 < v-t-Real.sqrt disc by linarith) (show 0 < v-t+Real.sqrt disc by linarith [Real.sqrt_nonneg disc])]
      linarith
    exact ⟨rfl,ok,eq ▸ lo,eq ▸ hi⟩

end ExactBallpark.Space

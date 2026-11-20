-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.NormNum
public import Mathlib.Tactic.Positivity

@[expose] public noncomputable section
namespace ExactBallpark.Space

/-- Inspect one finer level at a time, starting from an already selected level.
The top box is the fallback even for an object too large for the top width. -/
def descend (radius width : ℝ) : ℕ → ℕ
  | 0 => 0
  | n + 1 => if 3*radius > width/4 then 0 else 1 + descend radius (width/4) n

def levelFor (radius : ℝ) := descend radius 7864320 7

private theorem descend_le (r w : ℝ) (n : ℕ) : descend r w n ≤ n := by
  induction n generalizing w with
  | zero => rfl
  | succ n ih => simp only [descend]; split <;> first | omega | have := ih (w/4); omega

private theorem descends_iff (r w : ℝ) (hw : 0 ≤ w) (n j : ℕ) (hj : j ≤ n) :
    j ≤ descend r w n ↔ j = 0 ∨ 3*r ≤ w / 4^j := by
  induction n generalizing w j with
  | zero =>
    have hj0 : j = 0 := by omega
    simp [hj0, descend]
  | succ n ih =>
    cases j with
    | zero => simp
    | succ j =>
      have hpow : (1 : ℝ) ≤ 4^j := one_le_pow₀ (by norm_num)
      have shrink : w / 4 ^ (j+1) ≤ w/4 := by
        rw [pow_succ]
        apply div_le_div_of_nonneg_left hw (by norm_num)
        linarith
      simp only [descend]
      split
      next h => simp; linarith
      next h =>
        have hjn : j ≤ n := by omega
        rw [Nat.add_comm 1, Nat.succ_le_succ_iff, ih (w/4) (by positivity) j hjn]
        simp only [Nat.succ_ne_zero, false_or, pow_succ, div_div, mul_comm (4:ℝ)]
        constructor
        · rintro (rfl | hfit)
          · simpa using le_of_not_gt h
          · exact hfit
        · exact Or.inr

theorem finest_level (radius : ℝ) (i : Fin 8) :
    i.val ≤ levelFor radius ↔ i.val = 0 ∨ 3*radius ≤ 7864320 / (4:ℝ)^i.val :=
  descends_iff radius 7864320 (by norm_num) 7 i.val (by omega)

theorem finest_width (radius : ℝ) :
    levelFor radius = 7 ↔ 3*radius ≤ 480 := by
  have h := finest_level radius (7 : Fin 8)
  have bound := descend_le radius 7864320 7
  change levelFor radius ≤ 7 at bound
  change 7 ≤ levelFor radius ↔ (7 : ℕ) = 0 ∨ 3 * radius ≤ 7864320 / (4 : ℝ)^7 at h
  norm_num at h
  exact ⟨fun eq => h.mp eq.ge, fun fit => le_antisymm bound (h.mpr fit)⟩

end ExactBallpark.Space

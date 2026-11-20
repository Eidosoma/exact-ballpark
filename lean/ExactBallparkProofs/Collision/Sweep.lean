-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Positivity
public import Mathlib.Tactic.Ring

/-! Straight endpoint chords in space.SpaceBallpark.collide_two_spheres and full.AABB.can_exclude. -/
@[expose] public section

namespace ExactBallpark.Collision

abbrev Point := Fin 3 → ℝ

def chord (p₀ p₁ : Point) (t : ℝ) : Point := fun i => p₀ i + t * (p₁ i - p₀ i)

def dot (u v : Point) : ℝ := u 0 * v 0 + u 1 * v 1 + u 2 * v 2

/-- A positive gap means disjoint spheres. The radius is the combined radius. -/
def gap (p₀ p₁ q₀ q₁ : Point) (r t : ℝ) :=
  let d := chord p₀ p₁ t - chord q₀ q₁ t
  dot d d - r * r

theorem gap_quadratic (p₀ p₁ q₀ q₁ : Point) (r t : ℝ) :
    let d := p₀ - q₀
    let w := (p₁ - p₀) - (q₁ - q₀)
    gap p₀ p₁ q₀ q₁ r t = dot w w * t ^ 2 + 2 * dot d w * t + (dot d d - r * r) := by
  dsimp [gap, dot, chord]
  ring

noncomputable def quadraticTime (a b c : ℝ) : ℝ :=
  if c ≤ 0 then 0 else
  if a = 0 ∧ b = 0 then -1 else
  let det := b * b - 4 * a * c
  if det < 0 then -1 else
  let s := (-b - Real.sqrt det) * (1/2) / a
  if 0 ≤ s ∧ s ≤ 1 then s else -1

noncomputable def contactTime (p₀ p₁ q₀ q₁ : Point) (r : ℝ) :=
  let d := p₀ - q₀
  let w := (p₁ - p₀) - (q₁ - q₀)
  quadraticTime (dot w w) (2 * dot d w) (dot d d - r * r)

/-- Either every instant is clear, or the reported time is the first touch/overlap.
At a positive first time there is exact contact. -/
def FirstContact (g : ℝ → ℝ) (s : ℝ) : Prop :=
  (s = -1 ∧ ∀ t, 0 ≤ t → t ≤ 1 → 0 < g t) ∨
  (0 ≤ s ∧ s ≤ 1 ∧ g s ≤ 0 ∧ (0 < s → g s = 0) ∧ ∀ t, 0 ≤ t → t < s → 0 < g t)

private theorem quadratic_correct {a b c : ℝ} (ha : 0 ≤ a) (hstill : a = 0 → b = 0) :
    FirstContact (fun t => a * t ^ 2 + b * t + c) (quadraticTime a b c) := by
  unfold quadraticTime
  split
  next hc => right; simp [hc]; intro t ht ht'; linarith
  next hc =>
    have hc' : 0 < c := lt_of_not_ge hc
    split
    next h => left; simp [h.1, h.2, hc']
    next h =>
      have ha' : 0 < a := lt_of_le_of_ne ha (fun he => h ⟨he.symm, hstill he.symm⟩)
      let D := b * b - 4 * a * c
      have complete (t : ℝ) :
          4 * a * (a * t ^ 2 + b * t + c) = (2 * a * t + b) ^ 2 - D := by dsimp [D]; ring
      dsimp only
      split
      next hD =>
        left
        refine ⟨rfl, fun t _ _ => ?_⟩
        have hp : 0 < 4 * a * (a * t ^ 2 + b * t + c) := by
          rw [complete]; exact sub_pos_of_lt (lt_of_lt_of_le hD (sq_nonneg _))
        exact pos_of_mul_pos_right hp (by positivity)
      next hD =>
        let L := (-b - Real.sqrt D) * (1/2) / a
        let U := (-b + Real.sqrt D) * (1/2) / a
        have hs := Real.sq_sqrt (le_of_not_gt hD : 0 ≤ D)
        have hLU : L ≤ U := by
          dsimp [L, U]
          exact div_le_div_of_nonneg_right (by linarith [Real.sqrt_nonneg D]) ha
        have factor (t : ℝ) : a * t ^ 2 + b * t + c = a * (t - L) * (t - U) := by
          dsimp [L, U]
          field_simp
          dsimp [D] at hs
          nlinarith [hs]
        have before (t : ℝ) (ht : t < L) : 0 < a * t ^ 2 + b * t + c := by
          rw [factor]
          exact mul_pos_of_neg_of_neg (mul_neg_of_pos_of_neg ha' (sub_neg.mpr ht))
            (sub_neg.mpr (ht.trans_le hLU))
        split
        next hL =>
          right
          have hit : a * L ^ 2 + b * L + c = 0 := by rw [factor]; simp
          exact ⟨hL.1, hL.2, hit.le, fun _ => hit, fun t _ ht => before t ht⟩
        next hL =>
          left
          refine ⟨rfl, fun t ht ht' => ?_⟩
          by_cases behind : L < 0
          · have prodPos : 0 < a * (0 - L) * (0 - U) := by
              rw [← factor]; simpa using hc'
            have past : U < 0 := by
              have := pos_of_mul_pos_right prodPos (mul_pos ha' (sub_pos.mpr behind)).le
              linarith
            change 0 < a * t ^ 2 + b * t + c
            rw [factor]
            exact mul_pos (mul_pos ha' (sub_pos.mpr (behind.trans_le ht)))
              (sub_pos.mpr (past.trans_le ht))
          · exact before t (ht'.trans_lt (lt_of_not_ge (fun h => hL ⟨le_of_not_gt behind, h⟩)))

theorem contact_correct (p₀ p₁ q₀ q₁ : Point) (r : ℝ) :
    FirstContact (gap p₀ p₁ q₀ q₁ r) (contactTime p₀ p₁ q₀ q₁ r) := by
  let d := p₀ - q₀
  let w := (p₁ - p₀) - (q₁ - q₀)
  have eqn : gap p₀ p₁ q₀ q₁ r = fun t =>
    dot w w * t ^ 2 + 2 * dot d w * t + (dot d d - r * r) := funext (gap_quadratic p₀ p₁ q₀ q₁ r)
  rw [eqn]
  apply quadratic_correct
  · simp only [dot, ← sq]
    positivity
  · intro hw
    change dot w w = 0 at hw
    dsimp [dot] at hw
    have zeros : w 0 = 0 ∧ w 1 = 0 ∧ w 2 = 0 := by
      refine ⟨?_, ?_, ?_⟩ <;> nlinarith [sq_nonneg (w 0), sq_nonneg (w 1), sq_nonneg (w 2)]
    change 2 * dot d w = 0
    simp [dot, zeros.1, zeros.2.1, zeros.2.2]

/-- Each disjunct in the six-way source test chooses an axis and a strict separating side. -/
def canExclude (lo hi p₀ p₁ : Point) (r : ℝ) : Prop :=
  ∃ i, (hi i < p₀ i - r ∧ hi i < p₁ i - r) ∨ (p₀ i + r < lo i ∧ p₁ i + r < lo i)

theorem excluded_chord (lo hi p₀ p₁ : Point) (r : ℝ) (h : canExclude lo hi p₀ p₁ r) :
    ∃ i, ∀ t : ℝ, 0 ≤ t → t ≤ 1 →
      hi i < chord p₀ p₁ t i - r ∨ chord p₀ p₁ t i + r < lo i := by
  obtain ⟨i, side⟩ := h
  refine ⟨i, fun t ht ht' => ?_⟩
  have mix (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 0 < (1-t)*a + t*b := by
    by_cases h₀ : t = 0
    · simpa [h₀] using ha
    · exact add_pos_of_nonneg_of_pos (mul_nonneg (sub_nonneg.mpr ht') ha.le)
        (mul_pos (lt_of_le_of_ne ht (Ne.symm h₀)) hb)
  dsimp [chord]
  rcases side with ⟨h₀, h₁⟩ | ⟨h₀, h₁⟩
  · left; nlinarith [mix _ _ (sub_pos.mpr h₀) (sub_pos.mpr h₁)]
  · right; nlinarith [mix _ _ (sub_pos.mpr h₀) (sub_pos.mpr h₁)]

end ExactBallpark.Collision

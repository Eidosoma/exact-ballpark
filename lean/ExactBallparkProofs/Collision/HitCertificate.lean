-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Collision.Timing

@[expose] public section
namespace ExactBallpark.Collision

/-- Exact rational bisection of the first root, using the descending side and
the sign of the quadratic. The final bracket is independently validated. -/
def bisectEntry (a b c : ℚ) : Nat → ℚ → ℚ → ℚ × ℚ
  | 0, l, h => (l,h)
  | n+1, l, h =>
    let mid := (l+h)/2
    if beforeEntry a b c mid then bisectEntry a b c n mid h
    else bisectEntry a b c n l mid

private theorem bisect_width (a b c : ℚ) (n : Nat) (l h : ℚ) :
    (bisectEntry a b c n l h).2 - (bisectEntry a b c n l h).1 = (h-l)/2^n := by
  induction n generalizing l h with
  | zero => simp [bisectEntry]
  | succ n ih =>
    simp only [bisectEntry,pow_succ]
    split <;> rw [ih] <;> ring

structure HitBracket where
  lower : ℚ
  upper : ℚ
  reported : ℚ
  deriving DecidableEq, Repr

def certifyHit (p q u v : FloatPoint) (r : Float) (depth : Nat) : Option HitBracket := do
  let s ← decodeChord p q u v r
  if s.radius < 0 then none else
  let (a,b,c) := exactCoefficients s
  let (lo,hi) := bisectEntry a b c depth 0 1
  let result ← inspectTiming p q u v r lo hi
  if result.inTick ∧ 0 < lo ∧ hi < 1 ∧ lo ≤ result.roundedTime ∧ result.roundedTime ≤ hi then
    some ⟨lo,hi,result.roundedTime⟩
  else none

/-- A successful certificate identifies a hit by both solvers and bounds the
difference in tick fractions by 2⁻ᵈᵉᵖᵗʰ. All geometry comes from the endpoint words. -/
theorem certified_chord_hit (p q u v : FloatPoint) (r : Float) (depth : Nat) (h : HitBracket)
    (accepted : certifyHit p q u v r depth = some h) :
    finiteValue (contact64 p q u v r) = some h.reported ∧
    ∃ s, decodeChord p q u v r = some s ∧
      let entry := contactTime (realPoint s.p₀) (realPoint s.p₁) (realPoint s.q₀) (realPoint s.q₁) s.radius
      (0 < h.lower ∧ h.upper < 1 ∧ h.lower ≤ h.reported ∧ h.reported ≤ h.upper) ∧
      (h.lower : ℝ) ≤ entry ∧ entry ≤ h.upper ∧ |(h.reported : ℝ)-entry| ≤ 1/(2:ℝ)^depth := by
  unfold certifyHit at accepted
  obtain ⟨s,ds,accepted⟩ := Option.bind_eq_some_iff.mp accepted
  split at accepted
  next => contradiction
  next =>
    generalize hc : exactCoefficients s = abc at accepted
    obtain ⟨a,b,c⟩ := abc
    let l := (bisectEntry a b c depth 0 1).1
    let h' := (bisectEntry a b c depth 0 1).2
    obtain ⟨result,check,accepted⟩ := Option.bind_eq_some_iff.mp accepted
    split at accepted
    next conditions =>
      obtain ⟨inside,lo,hi,lb,ub⟩ := conditions
      cases accepted
      obtain ⟨seen,s',ds',th⟩ := timing_sound p q u v r l h' result check
      have eq : s'=s := Option.some.inj (ds'.symm.trans ds)
      subst s'
      rw [hc] at th
      obtain ⟨el,eu,_,entry⟩ := th
      simp only [inside, ↓reduceIte] at entry
      refine ⟨seen,s,ds,⟨lo,hi,lb,ub⟩,entry.symm ▸ el,entry.symm ▸ eu,?_⟩
      have width := bisect_width a b c depth 0 1
      change h' - l = (1-0)/2^depth at width
      simp only [sub_zero] at width
      have realWidth : (h':ℝ)-l = 1/(2:ℝ)^depth := by
        have cast := congrArg (fun x : ℚ => (x : ℝ)) width
        simpa using cast
      change |(result.roundedTime:ℝ)-_| ≤ _
      rw [entry, abs_le]
      have lb' : (l:ℝ) ≤ result.roundedTime := by exact_mod_cast lb
      have ub' : (result.roundedTime:ℝ) ≤ h' := by exact_mod_cast ub
      constructor <;> linarith
    next => contradiction

end ExactBallpark.Collision

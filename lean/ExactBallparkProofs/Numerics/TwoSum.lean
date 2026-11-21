-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import FloatLib.Floats.Formats.Flocq.Theory.Error.Addition
public import FloatLib.Floats.Formats.Flocq.Theory.Error.Multiplication
public import FloatLib.Floats.Formats.Flocq.Theory.Analysis.Sterbenz
public import Mathlib.Tactic.Linarith
public import Mathlib.Data.List.TakeWhile

/-! The six rounded additions/subtractions of vec_exp.two_sum on a binary
53-bit significand with unbounded exponents. This models value arithmetic on
IEEE operations when their rounding grids agree, including a zero result. -/
@[expose] public noncomputable section
namespace ExactBallpark.Numerics.Compensated

open FloatLib FloatLib.Floats.Formats.Flocq
local instance : ValidExp (flxExp 53) := validExp_flx 53 (by decide)
local instance : MonotoneExp (flxExp 53) := monotoneExp_flx 53

abbrev Rep (x : ℝ) := GenericRepresentable Numerics.binaryRadix (flxExp 53) x
def rn (x : ℝ) := round (β := Numerics.binaryRadix) (fexp := flxExp 53) nearestEven x

def twoSum (r : ℝ → ℝ) (a b : ℝ) :=
  let s := r (a+b)
  let z := r (s-a)
  (s, r (r (a-r (s-z)) + r (b-z)))

private structure Laws (r : ℝ → ℝ) : Prop where
  mono : Monotone r
  fmt : ∀ x, Rep (r x)
  exactOn : ∀ {x}, Rep x → r x = x
  error : ∀ {x y}, Rep x → Rep y → Rep (r (x+y)-(x+y))

private theorem flxLaws : Laws rn where
  mono _ _ h := round_mono nearestEven h
  fmt := genericRepresentable_round nearestEven
  exactOn := round_preserves_generic nearestEven _
  error := add_round_error_generic

private theorem negFmt {x : ℝ} (h : Rep x) : Rep (-x) := genericRepresentable_neg _ h

private theorem halfFmt {x : ℝ} (h : Rep x) : Rep (x/2) := by
  simpa [bpow_binaryRadix, div_eq_mul_inv] using genericRepresentable_flx_mul_bpow 53 (by decide) h (-1)

private theorem doubleFmt {x : ℝ} (h : Rep x) : Rep (2*x) := by
  simpa [bpow_binaryRadix, mul_comm] using genericRepresentable_flx_mul_bpow 53 (by decide) h 1

private theorem sterbenz {x y : ℝ} (hx : Rep x) (hy : Rep y)
    (p : x ≤ 2*y) (q : y ≤ 2*x) : Rep (x-y) :=
  genericRepresentable_flx_sterbenz 53 (by decide) p q hx hy

private theorem mirror {r : ℝ → ℝ} (L : Laws r) : Laws (fun x => -r (-x)) where
  mono _ _ h := neg_le_neg (L.mono (neg_le_neg h))
  fmt x := negFmt (L.fmt (-x))
  exactOn h := by rw [L.exactOn (negFmt h), neg_neg]
  error {x y} hx hy := by
    have hn := negFmt (L.error (negFmt hx) (negFmt hy))
    rw [← neg_add] at hn
    convert hn using 1; ring

private theorem boundedRound {r : ℝ → ℝ} (L : Laws r) {x l h : ℝ}
    (hl : Rep l) (hh : Rep h) (p : l ≤ x) (q : x ≤ h) : l ≤ r x ∧ r x ≤ h := by
  simpa only [L.exactOn hl,L.exactOn hh] using And.intro (L.mono p) (L.mono q)

private theorem nonnegativeSum {r : ℝ → ℝ} (L : Laws r) {a b : ℝ}
    (ha : Rep a) (hb : Rep b) (hn : 0 ≤ a+b) :
    (twoSum r a b).2 = a+b-r (a+b) := by
  have rz : r 0 = 0 := L.exactOn genericRepresentable_zero
  by_cases exactSum : Rep (a+b)
  · simp [twoSum, L.exactOn exactSum, L.exactOn ha, L.exactOn hb, rz]
  let s := r (a+b)
  let z := r (s-a)
  have hs : Rep s := L.fmt _
  have hz : Rep z := L.fmt _
  have he : Rep (a+b-s) := by simpa only [neg_sub] using negFmt (L.error ha hb)
  change r (r (a-r (s-z)) + r (b-z)) = a+b-s
  rcases le_total b a with dom | dom
  · have bound : -(a/2) ≤ b := by
      by_contra! h
      apply exactSum
      simpa using sterbenz ha (negFmt hb) (by linarith) (by linarith)
    obtain ⟨sl,su⟩ := boundedRound L (halfFmt ha) (doubleFmt ha)
      (show a/2 ≤ a+b by linarith) (by linarith : a+b ≤ 2*a)
    change a/2 ≤ s at sl
    change s ≤ 2*a at su
    have zz : z = s-a := L.exactOn (sterbenz hs ha su (by linarith))
    have back : s-z = a := by rw [zz]; ring
    have tail : b-z = a+b-s := by rw [zz]; ring
    rw [back, L.exactOn ha, sub_self, rz, tail, L.exactOn he, zero_add, L.exactOn he]
  · have bounds : (s/2 ≤ z ∧ z ≤ 2*s) ∧ (b/2 ≤ z ∧ z ≤ 2*b) := by
      by_cases sign : 0 ≤ a
      · obtain ⟨sl,su⟩ := boundedRound L hb (doubleFmt hb) (show b ≤ a+b by linarith)
          (show a+b ≤ 2*b by linarith)
        have sa := (L.mono (show 2*a ≤ a+b by linarith))
        rw [L.exactOn (doubleFmt ha)] at sa
        change 2*a ≤ s at sa
        change b ≤ s at sl
        change s ≤ 2*b at su
        obtain ⟨zl,zu⟩ := boundedRound L (halfFmt hs) hs
          (show s/2 ≤ s-a by linarith) (show s-a ≤ s by linarith)
        change s/2 ≤ z at zl
        change z ≤ s at zu
        constructor <;> constructor <;> linarith
      · have bound : -(b/2) ≤ a := by
          by_contra! h
          apply exactSum
          simpa [add_comm] using sterbenz hb (negFmt ha) (by linarith) (by linarith)
        obtain ⟨sl,su⟩ := boundedRound L (halfFmt hb) hb
          (show b/2 ≤ a+b by linarith) (show a+b ≤ b by linarith)
        change b/2 ≤ s at sl
        change s ≤ b at su
        obtain ⟨zl,zu⟩ := boundedRound L hs (doubleFmt hs)
          (show s ≤ s-a by linarith) (show s-a ≤ 2*s by linarith)
        change s ≤ z at zl
        change z ≤ 2*s at zu
        constructor <;> constructor <;> linarith
    have back := sterbenz hs hz (by linarith [bounds.1.1]) bounds.1.2
    have tail := sterbenz hb hz (by linarith [bounds.2.1]) bounds.2.2
    have small : Rep (a-(s-z)) := by
      convert L.error hs (negFmt ha) using 1
      rw [← sub_eq_add_neg]; dsimp [z]; ring
    rw [L.exactOn back, L.exactOn tail, L.exactOn small,
      show a-(s-z)+(b-z) = a+b-s by ring, L.exactOn he]

/-- Knuth's source-order correction recovers the complete sum error, for either
input ordering and sign, with binary nearest-even rounding at each operation. -/
theorem two_sum_exact {a b : ℝ} (ha : Rep a) (hb : Rep b) :
    (twoSum rn a b).1 = rn (a+b) ∧ (twoSum rn a b).1 + (twoSum rn a b).2 = a+b := by
  refine ⟨rfl, ?_⟩
  change rn (a+b) + (twoSum rn a b).2 = a+b
  rcases le_total 0 (a+b) with pos | neg
  · rw [nonnegativeSum flxLaws ha hb pos]; ring
  · have h := nonnegativeSum (mirror flxLaws) (negFmt ha) (negFmt hb) (by linarith)
    have na (x y : ℝ) : -(-x + -y) = x+y := by ring
    have ns (x y : ℝ) : -(-x - -y) = x-y := by ring
    simp only [twoSum, na, ns] at h ⊢
    linarith

/-- The proposed history cutoff advances strictly past the *exact* sum of
stored time and tolerance. The same first stopping index follows for a whole
history, including repeated times and queries equal to the rounded sum. -/
theorem compensated_history_cutoff (times : List ℝ) (delta q : ℝ)
    (ht : ∀ t ∈ times, Rep t) (hd : Rep delta) (hq : Rep q) :
    times.takeWhile (fun t => let (s,e) := twoSum rn t delta
      decide (s < q ∨ (s=q ∧ 0 < -e))) =
    times.takeWhile (fun t => decide (t+delta < q)) := by
  induction times with
  | nil => rfl
  | cons t ts ih =>
    have ht0 := ht t (by simp)
    obtain ⟨_,sum⟩ := two_sum_exact ht0 hd
    let s := rn (t+delta)
    let e := (twoSum rn t delta).2
    change s+e = t+delta at sum
    have guard : (s < q ∨ (s=q ∧ 0 < -e)) ↔ t+delta < q := by
      constructor
      · rintro (lt | ⟨eq,lt⟩)
        · by_contra! ge
          have le := flxLaws.mono ge
          rw [flxLaws.exactOn hq] at le
          exact (not_lt_of_ge le) lt
        · linarith
      · intro lt
        have le := flxLaws.mono lt.le
        rw [flxLaws.exactOn hq] at le
        rcases le.lt_or_eq with h | h
        · exact Or.inl h
        · exact Or.inr ⟨h, by change s = q at h; linarith⟩
    simp only [List.takeWhile_cons]
    change (if decide (s < q ∨ (s=q ∧ 0 < -e)) = true then _ else _) = _
    simp only [decide_eq_true_eq, guard]
    split <;> simp_all

end ExactBallpark.Numerics.Compensated

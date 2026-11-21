-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Numerics.TwoSum
public import FloatLib.Floats.Formats.Flocq.Theory.Error.Relative

@[expose] public noncomputable section
namespace ExactBallpark.Numerics.Compensated
open FloatLib FloatLib.Floats.Formats.Flocq

local instance : ValidExp (flxExp 53) := validExp_flx 53 (by decide)

def dyadic (m e : Int) : ℝ := (m:ℝ) * (2:ℝ)^e

private theorem repDyadic (m e : Int) (h : m.natAbs < 2^53) : Rep (dyadic m e) := by
  apply generic_of_flxFormat 53 (by decide)
  exact ⟨by decide, ⟨m,e⟩, by simp [dyadic,toReal,bpow_binaryRadix], h⟩

def fits (x : ℝ) := ∃ m e : Int, x = dyadic m e ∧ m.natAbs < 2^53

private theorem roundFits {x : ℝ} (h : fits x) : rn x = x := by
  obtain ⟨m,e,rfl,width⟩ := h
  exact round_preserves_generic nearestEven _ (repDyadic _ _ width)

private theorem exactValue {x : ℝ} (h : Rep x) : rn x = x :=
  round_preserves_generic nearestEven _ h

private theorem rnFmt (x : ℝ) : Rep (rn x) := genericRepresentable_round nearestEven x

/-- Finite binary32 values, including subnormals, have a signed significand
below 2^24 and a stored dyadic exponent between -149 and 104. Both promoted
products fit the 53-bit binary64 significand, and their dyadic exponents stay
well inside the range for finite normal doubles whenever the result is nonzero. -/
theorem promoted_limiter_exact (m e : Int) (width : m.natAbs < 2^24)
    (low : -149 ≤ e) (high : e ≤ 104) :
    let x := dyadic m e
    let a := 25*x
    rn a = a ∧ rn (rn a*x) = 25*x*x ∧
      (25*m*m).natAbs < 2^53 ∧ -298 ≤ e+e ∧ e+e ≤ 208 := by
  have w1 : (25*m).natAbs < 2^53 := by rw [Int.natAbs_mul]; norm_num at *; omega
  have w2 : (25*m*m).natAbs < 2^53 := by
    simp only [Int.natAbs_mul]
    have sq := Nat.mul_le_mul (Nat.le_of_lt_succ width) (Nat.le_of_lt_succ width)
    norm_num at *; nlinarith only [sq]
  have a : 25*dyadic m e = dyadic (25*m) e := by simp [dyadic]; ring
  have p : 25*dyadic m e*dyadic m e = dyadic (25*m*m) (e+e) := by
    simp [dyadic,zpow_add₀]; ring
  have r1 : rn (25*dyadic m e) = 25*dyadic m e := by rw [a]; exact exactValue (repDyadic _ _ w1)
  refine ⟨r1, ?_,w2, by omega, by omega⟩
  rw [r1,p]; exact exactValue (repDyadic _ _ w2)

theorem roundBound (x : ℝ) : |rn x-x| ≤ |x| / 2^53 := by
  obtain ⟨δ,bound,eq⟩ := round_relative_error_flx (β := Numerics.binaryRadix) 53 (by decide) nearestEven x
  have hδ : |δ| ≤ 1/2^53 := by norm_num [bpow_binaryRadix] at bound ⊢; exact bound
  change |round (β := Numerics.binaryRadix) (fexp := flxExp 53) nearestEven x-x| ≤ _
  rw [eq,show x*(1+δ)-x = x*δ by ring,abs_mul,div_eq_mul_inv]
  exact mul_le_mul_of_nonneg_left (by simpa only [one_div] using hδ) (abs_nonneg x)

private theorem closeDiff {x y : ℝ} (hx : Rep x) (hy : Rep y) (near : |x-y| ≤ |x|/2) :
    Rep (x-y) := by
  rcases le_total 0 x with pos | neg
  · rw [abs_of_nonneg pos, abs_le] at near
    exact genericRepresentable_flx_sterbenz 53 (by decide) (by linarith) (by linarith) hx hy
  · rw [abs_of_nonpos neg, abs_le] at near
    have h := genericRepresentable_flx_sterbenz 53 (by decide)
      (show -x ≤ 2*(-y) by linarith) (show -y ≤ 2*(-x) by linarith)
      (genericRepresentable_neg _ hx) (genericRepresentable_neg _ hy)
    simpa [sub_eq_add_neg,add_comm] using genericRepresentable_neg _ h

def split53 (a : ℝ) :=
  let c := rn (134217729*a)
  let hi := rn (c-rn (c-a))
  (hi,rn (a-hi))

/-- Veltkamp reconstruction in the 53-bit nearest-even model. The residual
bound and the splitter's upper range follow from the operand magnitude. -/
theorem split_reconstructs {a : ℝ} (ha : Rep a) (safe : |a| < 2^996) :
    (split53 a).1 + (split53 a).2 = a ∧
    |(split53 a).2| ≤ (134217731 / 2^53)*|a| ∧ |rn (134217729*a)| < 2^1024 := by
  let c := rn (134217729*a)
  let z := rn (c-a)
  have e1 := roundBound (134217729*a)
  change |c-134217729*a| ≤ _ at e1
  have absProd : |134217729*a| = 134217729*|a| := by rw [abs_mul,abs_of_nonneg (by norm_num : (0:ℝ) ≤ 134217729)]
  rw [absProd] at e1
  have lower : 134217728*|a| ≤ |c| := by
    have tri := abs_sub_abs_le_abs_sub (134217729*a) c
    rw [absProd,abs_sub_comm (134217729*a) c] at tri
    linarith [abs_nonneg a]
  have upper : |c| ≤ 134217730*|a| := by
    have tri := abs_sub_abs_le_abs_sub c (134217729*a)
    rw [absProd] at tri
    linarith [abs_nonneg a]
  have e2 := roundBound (c-a)
  change |z-(c-a)| ≤ _ at e2
  have short : |z-(c-a)| ≤ (134217731 / 2^53)*|a| := by
    have tri : |c-a| ≤ |c|+|a| := by simpa only [sub_eq_add_neg, abs_neg] using abs_add_le c (-a)
    linarith
  have pair : |c-z| ≤ |c|/2 := by
    have tri := abs_add_le a (-(z-(c-a)))
    rw [abs_neg,show a + -(z-(c-a)) = c-z by ring] at tri
    linarith [abs_nonneg a]
  have hi : rn (c-z) = c-z := exactValue (closeDiff (rnFmt _) (rnFmt _) pair)
  have low : |a-(c-z)| ≤ |a|/2 := by
    rw [show a-(c-z) = z-(c-a) by ring]; linarith [abs_nonneg a]
  have repHi : Rep (c-z) := closeDiff (rnFmt _) (rnFmt _) pair
  have lo := exactValue (closeDiff ha repHi low)
  change rn (c-z) + rn (a-rn (c-z)) = a ∧ |rn (a-rn (c-z))| ≤ _ ∧ |c| < _
  rw [hi,lo,show a-(c-z) = z-(c-a) by ring]
  exact ⟨by ring,short, upper.trans_lt ((mul_lt_mul_of_pos_left safe (by norm_num)).trans_le (by
    rw [show (1024 : Nat) = 28+996 by decide, pow_add]; gcongr; norm_num))⟩

def product53 (a b : ℝ) :=
  let p := rn (a*b)
  let (ah,al) := split53 a
  let (bh,bl) := split53 b
  (p, rn (rn (al*bl) - rn (rn (rn (p-rn (ah*bh))-rn (al*bh))-rn (ah*bl))))

/-- Independent dyadic width certificates for the partial products and the
three differences in Dekker's order. The final correction needs no certificate:
the exact residual of a rounded product is itself representable. -/
def productWidths (a b : ℝ) :=
  let p := rn (a*b)
  let (ah,al) := split53 a
  let (bh,bl) := split53 b
  let d1 := p-ah*bh
  let d2 := d1-al*bh
  fits (ah*bh) ∧ fits (al*bh) ∧ fits (ah*bl) ∧ fits (al*bl) ∧
    fits d1 ∧ fits d2 ∧ fits (d2-ah*bl)

theorem dekker_product_exact {a b : ℝ} (ha : Rep a) (hb : Rep b)
    (sa : |a| < 2^996) (sb : |b| < 2^996) (widths : productWidths a b) :
    (product53 a b).1 = rn (a*b) ∧ (product53 a b).1 + (product53 a b).2 = a*b := by
  simp only [productWidths] at widths
  obtain ⟨w1,w2,w3,w4,w5,w6,w7⟩ := widths
  have aeq := (split_reconstructs ha sa).1
  have beq := (split_reconstructs hb sb).1
  have err := mul_round_error_flx 53 (by decide) nearestEven ha hb
  have er : Rep (a*b-rn (a*b)) := by simpa only [rn,neg_sub] using genericRepresentable_neg _ err
  refine ⟨rfl,?_⟩
  simp only [product53]
  rw [roundFits w1,roundFits w2,roundFits w3,roundFits w4,roundFits w5,roundFits w6,roundFits w7]
  have residual : (split53 a).2*(split53 b).2 -
      (rn (a*b)-(split53 a).1*(split53 b).1-(split53 a).2*(split53 b).1-(split53 a).1*(split53 b).2)
      = a*b-rn (a*b) := by
    linear_combination (split53 b).1*aeq + (split53 b).2*aeq + a*beq
  rw [residual,exactValue er]; ring

end ExactBallpark.Numerics.Compensated

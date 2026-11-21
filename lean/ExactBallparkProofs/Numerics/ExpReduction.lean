-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Numerics.Product
public import Mathlib.Analysis.Complex.Exponential
public import Mathlib.Data.Nat.Bitwise

@[expose] public noncomputable section
namespace ExactBallpark.Numerics.ExpReduction
open FloatLib FloatLib.Floats.Formats.Flocq Compensated

def address (k : Int) := ((BitVec.ofInt 64 k &&& 127).toNat) * 2

/-- Negative reduction indices wrap into the same 128-pair table. Both the
tail and scale loads are in bounds for every signed index. -/
theorem table_address (k : Int) :
    address k = 2*(k%128).toNat ∧ Even (address k) ∧ address k+1 < 256 := by
  have low : (BitVec.ofInt 64 k &&& 127).toNat = (k%128).toNat := by
    rw [BitVec.toNat_and,BitVec.toNat_ofInt]
    change (k % 18446744073709551616).toNat &&& (2^7-1) = _
    rw [Nat.and_two_pow_sub_one_eq_mod]
    omega
  rw [address,low]
  exact ⟨Nat.mul_comm ..,⟨(k%128).toNat,by omega⟩,by omega⟩

def scaleWord (m k : Int) : BitVec 64 :=
  BitVec.ofInt 64 (1023*2^52+m-(k%128)*2^45) + BitVec.ofInt 64 (k*2^45)

/-- The low seven bits index the table. The remaining quotient changes
only the exponent of a normal positive scale word, including negative k. -/
theorem scale_reconstruction (m k : Int) (lo : 0 ≤ m) (hi : m < 2^52)
    (normalLo : 1 ≤ 1023+k/128) (normalHi : 1023+k/128 ≤ 2046) :
    let bits := scaleWord m k
    bits.toNat / 2^52 = (1023+k/128).toNat ∧ bits.toNat % 2^52 = m.toNat := by
  have eq : scaleWord m k = BitVec.ofInt 64 ((1023+k/128)*2^52+m) := by
    rw [scaleWord,← BitVec.ofInt_add]
    congr 1
    norm_num at *; omega
  dsimp only
  rw [eq,BitVec.toNat_ofInt]
  norm_num at *; omega

/-- On either special arm the int64 adjustment moves the scale exponent
without altering its significand. Undoing that shift recovers the ideal
scale, before the polynomial, subnormal compensation or final rounding. -/
theorem special_rescaling (m k : Int) (bound : |k| < 2^18) :
    let high := (BitVec.ofInt 64 k &&& 0x80000000) = 0
    let shift : Int := if high then -1009 else 1022
    let exponent := k/128
    let adjusted := BitVec.ofInt 64 (1023*2^52+m-(k%128)*2^45) +
      (BitVec.ofInt 64 (k*2^45) + BitVec.ofInt 64 (shift*2^52))
    (high ↔ 0 ≤ k) ∧ adjusted = BitVec.ofInt 64 ((1023+exponent+shift)*2^52+m) ∧
    ((1+(m:ℝ)/2^52)*2^(exponent+shift))*(2:ℝ)^(-shift) =
      (1+(m:ℝ)/2^52)*2^exponent := by
  dsimp only
  have sign : ((BitVec.ofInt 64 k &&& 0x80000000) = 0) ↔ 0 ≤ k := by
    rw [← BitVec.toNat_inj,BitVec.toNat_and,BitVec.toNat_ofInt]
    change ((k % 18446744073709551616).toNat &&& 2^31) = 0 ↔ _
    rw [Nat.and_two_pow,Nat.testBit_eq_decide_div_mod_eq]
    obtain ⟨lo,hi⟩ := abs_lt.mp bound
    norm_num at lo hi ⊢
    omega
  refine ⟨sign,?_,?_⟩
  · rw [← BitVec.ofInt_add,← BitVec.ofInt_add]
    congr 1
    split <;> norm_num <;> omega
  · rw [mul_assoc,← zpow_add₀ (by norm_num : (2:ℝ) ≠ 0)]; simp

local instance : ValidExp (flxExp 53) := validExp_flx 53 (by decide)
private theorem integer_rn (n : Int) (small : n.natAbs < 2^53) : rn (n:ℝ) = n := by
  apply round_preserves_generic nearestEven
  apply generic_of_flxFormat 53 (by decide)
  exact ⟨by decide,⟨n,0⟩,by simp [toReal,bpow_binaryRadix],small⟩

def index (x : ℝ) : Int :=
  let z := rn (x * (3248660424278399/17592186044416))
  truncRound (rn (z + if z < 0 then -(1/2) else 1/2))

/-- xs is the input after the tiny/huge lane substitution. The multiply
and signed half addition each round to nearest-even before truncation. -/
theorem reduction_index_bound (x : ℝ) (h : |x| ≤ 1024) :
    |index x| < 2^18 ∧ -(2^63) ≤ index x * 2^45 ∧ index x * 2^45 < 2^63 := by
  let c : ℝ := 3248660424278399/17592186044416
  have mono : Monotone rn := fun _ _ h => round_mono nearestEven h
  have fixed (n : Int) (hn : n.natAbs < 2^53) := integer_rn n hn
  obtain ⟨l,u⟩ := abs_le.mp h
  have lower := mono (show ((-200000:Int):ℝ) ≤ x*c by dsimp [c]; linarith)
  have upper := mono (show x*c ≤ ((200000:Int):ℝ) by dsimp [c]; linarith)
  rw [fixed _ (by decide)] at lower upper
  let z := rn (x*c)
  let biased := z + if z < 0 then -(1/2:ℝ) else 1/2
  have lo : ((-200001:Int):ℝ) ≤ biased := by
    norm_num at lower ⊢
    change (-200000:ℝ) ≤ z at lower
    change (-200001:ℝ) ≤ z + _
    split <;> linarith
  have hi : biased ≤ ((200001:Int):ℝ) := by
    norm_num at upper ⊢
    change z ≤ (200000:ℝ) at upper
    change z + _ ≤ (200001:ℝ)
    split <;> linarith
  have p := mono lo
  have q := mono hi
  rw [fixed _ (by decide)] at p q
  have lt := ValidRnd.monotone (rnd := truncRound) _ _ p
  have ut := ValidRnd.monotone (rnd := truncRound) _ _ q
  rw [ValidRnd.id (rnd := truncRound)] at lt ut
  change -200001 ≤ index x at lt
  change index x ≤ 200001 at ut
  rw [abs_lt]; norm_num; omega

private theorem midpoint_lo : rn (1-1/2^54) = 1 := by
  have mag := magnitude_eq_of_bpow_bounds Numerics.binaryRadix (1-1/2^54) 0
    (by norm_num [bpow_binaryRadix]) (by norm_num [bpow_binaryRadix])
  simp only [rn,Floats.Formats.Flocq.round,scaledMantissa,cexp,mag,flxExp,toReal,bpow_binaryRadix]
  norm_num [nearestEven]
  exact ⟨4503599627370495,by norm_num⟩

private theorem midpoint_hi : rn (1+1/2^53) = 1 := by
  have mag := magnitude_eq_of_bpow_bounds Numerics.binaryRadix (1+1/2^53) 1
    (by norm_num [bpow_binaryRadix]) (by norm_num [bpow_binaryRadix])
  simp only [rn,Floats.Formats.Flocq.round,scaledMantissa,cexp,mag,flxExp,toReal,bpow_binaryRadix]
  norm_num [nearestEven]
  exact ⟨2251799813685248,by norm_num⟩

/-- In the tiny exponent bin the actual 1+x expression and the exact
exponential round to the same normal binary64 value for either sign. -/
theorem tiny_correct_round (x : ℝ) (tiny : |x| < 1/2^54) :
    rn (x+1) = 1 ∧ rn (Real.exp x) = 1 := by
  have inCell (y : ℝ) (lo : 1-1/2^54 ≤ y) (hi : y ≤ 1+1/2^53) : rn y = 1 := by
    have h₁ : rn (1-1/2^54) ≤ rn y := round_mono nearestEven lo
    have h₂ : rn y ≤ rn (1+1/2^53) := round_mono nearestEven hi
    rw [midpoint_lo] at h₁; rw [midpoint_hi] at h₂; exact le_antisymm h₂ h₁
  obtain ⟨lo,hi⟩ := abs_lt.mp tiny
  refine ⟨inCell _ (by linarith) (by linarith),inCell _ (by linarith [Real.add_one_le_exp x]) ?_⟩
  exact ((Real.exp_le_exp.mpr hi.le).trans
    (Real.exp_bound_div_one_sub_of_interval (by norm_num) (by norm_num))).trans (by norm_num)

end ExactBallpark.Numerics.ExpReduction

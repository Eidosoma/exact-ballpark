-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Init.Data.BitVec.Lemmas
import Lean.Elab.Tactic.Omega

/-! Integer dispatch fragments, quantified over all input encodings. Floating
operations and polynomial/table evaluations after dispatch are separate obligations. -/
@[expose] public section
namespace ExactBallpark.Numerics

/-- Two's-complement arithmetic shift in the Torch expression. -/
def expTop (bits : BitVec 64) := (bits.sshiftRight 52) &&& 0x7ff

private theorem magnitude_nat (x : BitVec 64) :
    (x &&& 0x7fffffffffffffff).toNat = x.toNat % 2^63 := by
  exact Nat.and_two_pow_sub_one_eq_mod x.toNat 63

private theorem top_nat (x : BitVec 64) :
    (expTop x).toNat = (x.toNat % 2^63) / 2^52 := by
  change ((x.sshiftRight 52).toNat &&& (2^11-1)) = _
  rw [Nat.and_two_pow_sub_one_eq_mod, BitVec.toNat_sshiftRight]
  have hx := x.isLt
  simp only [Nat.shiftRight_eq_div_pow]
  split <;> omega

theorem exp_tiny_iff (x : BitVec 64) :
    expTop x < 0x3c9 ↔ (x &&& 0x7fffffffffffffff) < 0x3c90000000000000 := by
  simp only [BitVec.lt_def, top_nat, magnitude_nat]
  change _ / 2^52 < 0x3c9 ↔ _ < (0x3c90000000000000 : Nat)
  omega

theorem exp_large_iff (x : BitVec 64) :
    expTop x = 0x408 ↔
    0x4080000000000000 ≤ (x &&& 0x7fffffffffffffff) ∧
      (x &&& 0x7fffffffffffffff) < 0x4090000000000000 := by
  simp only [BitVec.lt_def, BitVec.le_def, BitVec.toNat_eq, top_nat, magnitude_nat]
  change _ / 2^52 = 0x408 ↔ (0x4080000000000000 : Nat) ≤ _ ∧ _ < (0x4090000000000000 : Nat)
  omega

theorem powf_signaling_iff (x : BitVec 32) :
    0x7fc00000 < ((x ^^^ 0x400000) &&& 0x7fffffff) ↔
    x.extractLsb' 23 8 = 255 ∧ x.extractLsb' 22 1 = 0 ∧ x.extractLsb' 0 22 ≠ 0 := by
  let y := (x ^^^ 0x400000) &&& 0x7fffffff
  have expEq : y.extractLsb' 23 8 = x.extractLsb' 23 8 := by
    simp only [y, BitVec.extractLsb'_and, BitVec.extractLsb'_xor]
    change (_ ^^^ 0#8) &&& BitVec.allOnes 8 = _
    rw [BitVec.and_allOnes, BitVec.xor_zero]
  have quietEq : y.extractLsb' 22 1 = (x.extractLsb' 22 1 ^^^ 1) := by
    simp only [y, BitVec.extractLsb'_and, BitVec.extractLsb'_xor]
    change (_ ^^^ 1#1) &&& BitVec.allOnes 1 = _
    rw [BitVec.and_allOnes]
    rfl
  have payloadEq : y.extractLsb' 0 22 = x.extractLsb' 0 22 := by
    simp only [y, BitVec.extractLsb'_and, BitVec.extractLsb'_xor]
    change (_ ^^^ 0#22) &&& BitVec.allOnes 22 = _
    rw [BitVec.and_allOnes, BitVec.xor_zero]
  have bound : y.toNat < 2^31 := by
    change ((x ^^^ 0x400000).toNat &&& (2^31-1)) < _
    rw [Nat.and_two_pow_sub_one_eq_mod]
    exact Nat.mod_lt _ (by decide)
  have flip : y.extractLsb' 22 1 = 1 ↔ x.extractLsb' 22 1 = 0 := by
    rw [quietEq]
    generalize x.extractLsb' 22 1 = q
    exact (by decide : ∀ q : BitVec 1, q ^^^ 1 = 1 ↔ q = 0) q
  change 0x7fc00000 < y ↔ _
  rw [← expEq, ← flip, ← payloadEq]
  rw [← BitVec.toNat_ne_iff_ne]
  simp only [BitVec.toNat_eq, BitVec.lt_def]
  change 0x7fc00000 < y.toNat ↔ (y.toNat >>> 23) % 256 = 255 ∧
    (y.toNat >>> 22) % 2 = 1 ∧ (y.toNat >>> 0) % 4194304 ≠ 0
  simp only [Nat.shiftRight_eq_div_pow]
  omega

private theorem wrap_range (x offset modulus lower upper width : Nat)
    (hx : x < modulus) (hoff : offset + lower = modulus) (hwidth : lower + width = upper)
    (hdisjoint : width < offset) :
    (x + offset) % modulus ≤ width ↔ lower ≤ x ∧ x ≤ upper := by
  by_cases h : x + offset < modulus
  · rw [Nat.mod_eq_of_lt h]
    omega
  · have hw : modulus ≤ x + offset := by omega
    have bounded : x + offset - modulus < modulus := by omega
    rw [Nat.mod_eq_sub_mod hw, Nat.mod_eq_of_lt bounded]
    omega

private theorem sumOff : 0xc012000000000000 + 0x3fee000000000000 = (2^64 : Nat) := by decide
private theorem sumWidth : 0x3fee000000000000 + 0x308ffffffffff = (0x3ff108ffffffffff : Nat) := by decide
private theorem disjWidth : (0x308ffffffffff : Nat) < 0xc012000000000000 := by decide
private theorem wordNum (x : BitVec 64) :
    (x.toNat + 0xc012000000000000) % (2^64) ≤ 0x308ffffffffff ↔
    0x3fee000000000000 ≤ x.toNat ∧ x.toNat ≤ 0x3ff108ffffffffff :=
  wrap_range _ _ _ _ _ _ x.isLt sumOff sumWidth disjWidth
private theorem addLE (x : BitVec 64) : x + 0xc012000000000000 ≤ 0x308ffffffffff ↔ (x.toNat + 0xc012000000000000) % 2^64 ≤ 0x308ffffffffff := by rw [BitVec.le_def, BitVec.toNat_add]; rfl
private theorem lowerLE (x : BitVec 64) : 0x3fee000000000000 ≤ x ↔ 0x3fee000000000000 ≤ x.toNat := by rw [BitVec.le_def]; rfl
private theorem upperLE (x : BitVec 64) : x ≤ 0x3ff108ffffffffff ↔ x.toNat ≤ 0x3ff108ffffffffff := by rw [BitVec.le_def]; rfl
/-- dlibm.log uses a wrapping addition and an inclusive upper bound. -/
theorem log_near_one_iff (x : BitVec 64) :
    x + 0xc012000000000000 ≤ 0x308ffffffffff ↔
    0x3fee000000000000 ≤ x ∧ x ≤ 0x3ff108ffffffffff := by
  rw [addLE, lowerLE, upperLE]
  exact wordNum x

def withSign (z y : BitVec 64) := (y &&& ((1 : BitVec 64) <<< 63)) ||| (z &&& ~~~((1 : BitVec 64) <<< 63))

theorem atan2_sign (magnitude sign : BitVec 64) :
    (withSign magnitude sign).extractLsb' 0 63 = magnitude.extractLsb' 0 63 ∧
    (withSign magnitude sign).extractLsb' 63 1 = sign.extractLsb' 63 1 := by
  simp only [withSign, BitVec.extractLsb'_or, BitVec.extractLsb'_and]
  change (_ &&& 0#63) ||| (_ &&& BitVec.allOnes 63) = _ ∧
    (_ &&& BitVec.allOnes 1) ||| (_ &&& 0#1) = _
  simp only [BitVec.and_allOnes, BitVec.and_zero, BitVec.zero_or, BitVec.or_zero, and_self]

end ExactBallpark.Numerics

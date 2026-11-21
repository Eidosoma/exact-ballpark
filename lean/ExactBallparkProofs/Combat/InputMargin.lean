-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Combat.TrackingMargin

@[expose] public section
namespace ExactBallpark.Combat.InputMargin
open ExactBallpark.Collision Float.Model Float.Model.UnpackedFloat

/-- Preserve the sign through rounding, allowing underflow to +0 or overflow to +∞. -/
private def positiveShape : UnpackedFloat → Prop
  | .zero .positive | .finite .positive .. | .infinity .positive => True
  | _ => False

private theorem sign_components (e : BitVec 11) (m : BitVec 52) :
    unpackSign (packComponents .binary64 .positive e m) = 0 := by
  ext i hi
  obtain rfl : i = 0 := by omega
  simp [unpackSign, packComponents, Format.binary64, Sign.toBitVec]
  rw [BitVec.getLsbD_append (x := (0#1 ++ e)) (y := m), ite_eq_right (by decide : ¬63 < 52),
    BitVec.getLsbD_append (x := 0#1) (y := e), ite_eq_right (by decide : ¬63-52 < 11)]
  rfl

private theorem unpack_components (e : BitVec 11) (m : BitVec 52) (finite : e ≠ -1#11) :
    positiveShape (unpack .binary64 (packComponents .binary64 .positive e m)) := by
  simp only [UnpackedFloat.unpack,unpackMantissa_packComponents,unpackExponent_packComponents,
    sign_components,finite,ite_false]
  repeat' first | trivial | split

private theorem repack (x : UnpackedFloat) (h : positiveShape x) :
    positiveShape (Float.Model.pack x).unpack := by
  change positiveShape (unpack .binary64 (pack .binary64 x))
  cases x with
  | zero s | infinity s => cases s <;> first | contradiction | trivial
  | notANumber => contradiction
  | finite s m e hm =>
    cases s with
    | negative => contradiction
    | positive =>
      dsimp only [UnpackedFloat.pack]
      split
      · trivial
      next bound =>
        split
        · apply unpack_components
          intro eq
          have n := congrArg BitVec.toNat eq
          simp only [BitVec.toNat_ofNat] at n
          change _ % 2048 = 2047 at n
          change ¬ 2048 ≤ _ + 1 at bound
          omega
        · exact unpack_components _ _ (by decide)

private theorem rounded_positive (m : Nat) (e : Int) (a : Accuracy) :
    positiveShape (roundWithAccuracy .binary64 .positive m e a) := by
  unfold roundWithAccuracy
  split
  dsimp only
  split <;> trivial

/-- Exact integer margin before the subtraction, measured in units of the
smaller of the two binary exponents. Both upstream operands must be finite. -/
def enough : UnpackedFloat → UnpackedFloat → Bool
  | .zero .positive, .zero .positive => true
  | .finite .positive .., .zero .positive => true
  | .finite .positive a e _, .finite .positive b f _ =>
    let scale := min e f
    decide ((decreaseExponent b f scale).1 ≤ (decreaseExponent a e scale).1)
  | _, _ => false

private theorem subtract_sign (x y : UnpackedFloat) (h : enough x y = true) :
    positiveShape (sub .binary64 x y) := by
  fun_cases enough x y with
  | case1 | case2 => trivial
  | case3 a e ha b f hb =>
    change decide (_ ≤ _) = true at h
    have hbnd := of_decide_eq_true h
    dsimp only [UnpackedFloat.sub, UnpackedFloat.normalize]
    dsimp only [Sign.apply]
    split
    next neg => simp only [Int.compare_eq_lt] at neg; omega
    · trivial
    · apply rounded_positive
  | case4 => simp [enough] at h

private theorem quotient_sign (n : UnpackedFloat) (m : Nat) (e : Int) (hm : 0 < m)
    (hn : positiveShape n) : positiveShape (div .binary64 n (.finite .positive m e hm)) := by
  cases n with
  | notANumber => contradiction
  | infinity s | zero s => cases s <;> first | contradiction | trivial
  | finite s _ _ _ =>
    cases s with
    | negative => contradiction
    | positive => apply rounded_positive

def positiveFinite : UnpackedFloat → Bool
  | .finite .positive .. => true
  | _ => false

/-- Only the rounded dot products, radial projection and squared distance are
needed. The final subtraction, division and square root are absent from the guard. -/
def safe (v r : FloatPoint) : Bool :=
  let d := (dot64 r r).sqrt
  let radial := dot64 v r / d
  positiveFinite (d*d).toModel.unpack && enough (dot64 v v).toModel.unpack (radial*radial).toModel.unpack

theorem tracking_input_nonnegative (v r : FloatPoint) (res track sig : Float) (pass : safe v r = true) :
    (0 : Float) ≤ (trackingTerms v r res track sig).1 := by
  let d := (dot64 r r).sqrt
  let radial := dot64 v r / d
  let a := dot64 v v
  let b := radial*radial
  let den := d*d
  obtain ⟨hd,hs⟩ := Bool.and_eq_true_iff.mp pass
  change positiveFinite den.toModel.unpack = true at hd
  have num : positiveShape (a-b).toModel.unpack := repack _ (subtract_sign _ _ hs)
  have q : positiveShape ((a-b)/den).toModel.unpack := by
    apply repack
    cases h : den.toModel.unpack with
    | finite s m e hm =>
      cases s <;> first | exact quotient_sign _ _ _ _ num | simp [h,positiveFinite] at hd
    | _ => simp [h,positiveFinite] at hd
  change UnpackedFloat.le (.zero .positive) ((a-b)/den).toModel.unpack = true
  generalize ((a-b)/den).toModel.unpack = u at q ⊢
  cases u with
  | notANumber => contradiction
  | zero s | finite s _ _ _ | infinity s =>
    cases s <;> first | contradiction | rfl

set_option maxRecDepth 16384 in
theorem tracking_margin_controls :
    let r : FloatPoint := ![3,4,12]
    let radial : FloatPoint := ![Float.ofBits 0x3f9eb851eb851eb8, Float.ofBits 0x3fa47ae147ae147b,
      Float.ofBits 0x3fbeb851eb851eb8]
    safe radial r = false ∧ safe (![4,-3,0]) r = true ∧ safe (![3,0,0]) (![4,0,0]) = true := by
  decide +kernel

end ExactBallpark.Combat.InputMargin

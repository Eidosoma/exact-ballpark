-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.Discriminant

/-! An offline certificate from encoded endpoints. Decode the input geometry exactly,
measure the coefficient errors, then check the discriminant margin. -/
@[expose] public section
namespace ExactBallpark.Collision

abbrev RatPoint := Fin 3 → ℚ

def decodePoint (p : FloatPoint) : Option RatPoint := do
  let x ← finiteValue (p 0)
  let y ← finiteValue (p 1)
  let z ← finiteValue (p 2)
  return fun i => if i = 0 then x else if i = 1 then y else z

def realPoint (p : RatPoint) : Point := fun i => (p i : ℝ)

structure ExactChord where
  p₀ : RatPoint
  p₁ : RatPoint
  q₀ : RatPoint
  q₁ : RatPoint
  radius : ℚ

def decodeChord (p₀ p₁ q₀ q₁ : FloatPoint) (r : Float) : Option ExactChord := do
  return ⟨← decodePoint p₀, ← decodePoint p₁, ← decodePoint q₀, ← decodePoint q₁, ← finiteValue r⟩

def exactCoefficients (s : ExactChord) : ℚ × ℚ × ℚ :=
  let dotQ := fun (x y : RatPoint) => x 0*y 0 + x 1*y 1 + x 2*y 2
  let d := s.p₀ - s.q₀
  let w := (s.p₁ - s.p₀) - (s.q₁ - s.q₀)
  (dotQ w w, 2*dotQ d w, dotQ d d - s.radius*s.radius)

theorem decoded_gap (s : ExactChord) (t : ℝ) :
    let (a,b,c) := exactCoefficients s
    gap (realPoint s.p₀) (realPoint s.p₁) (realPoint s.q₀) (realPoint s.q₁) s.radius t =
      (a : ℝ)*t^2 + b*t + c := by
  rw [gap_quadratic]
  dsimp [exactCoefficients, dot, realPoint]
  push_cast
  ring

/-- Use exact input bits to establish the three error budgets. An early overlap,
nonfinite data or insufficient separation from zero gives no certificate. -/
def certifySweep (p₀ p₁ q₀ q₁ : FloatPoint) (r : Float) : Option DiscSign := do
  let (af,bf,cf) ← sweepQuadratic64 p₀ p₁ q₀ q₁ r
  let s ← decodeChord p₀ p₁ q₀ q₁ r
  let (a,b,c) := exactCoefficients s
  let a₀ ← finiteValue af
  let b₀ ← finiteValue bf
  let c₀ ← finiteValue cf
  certifyDiscriminant af bf cf |a-a₀| |b-b₀| |c-c₀|

/-- No external finiteness, error or velocity hypotheses. A certified negative
from the endpoints rules out contact along the real chords and in the binary64 solver. -/
theorem certified_chord_miss (p₀ p₁ q₀ q₁ : FloatPoint) (r : Float)
    (cert : certifySweep p₀ p₁ q₀ q₁ r = some .negative) :
    (contact64 p₀ p₁ q₀ q₁ r).toBits = 0xbff0000000000000 ∧
    ∃ s, decodeChord p₀ p₁ q₀ q₁ r = some s ∧
      contactTime (realPoint s.p₀) (realPoint s.p₁) (realPoint s.q₀) (realPoint s.q₁) s.radius = -1 := by
  unfold certifySweep at cert
  obtain ⟨⟨af,bf,cf⟩, formed, cert⟩ := Option.bind_eq_some_iff.mp cert
  obtain ⟨s, decoded, cert⟩ := Option.bind_eq_some_iff.mp cert
  generalize coefficients : exactCoefficients s = abc at cert
  obtain ⟨a,b,c⟩ := abc
  obtain ⟨a₀, da, cert⟩ := Option.bind_eq_some_iff.mp cert
  obtain ⟨b₀, db, cert⟩ := Option.bind_eq_some_iff.mp cert
  obtain ⟨c₀, dc, cert⟩ := Option.bind_eq_some_iff.mp cert
  obtain ⟨_, hneg⟩ := certified_sign da db dc (le_refl _) (le_refl _) (le_refl _) cert
  have nonneg : 0 ≤ a := by
    have eqn := congrArg Prod.fst coefficients
    dsimp [exactCoefficients] at eqn
    rw [← eqn]
    simp only [← sq]
    positivity
  have moving : 0 < a := by
    rcases nonneg.eq_or_lt with zero | pos
    · rw [← zero] at hneg; nlinarith [sq_nonneg b]
    · exact pos
  obtain ⟨bits, clear⟩ := certified_miss _ _ _ _ _ _ _ _ formed da db dc
    (le_refl _) (le_refl _) (le_refl _) moving cert
  refine ⟨bits, s, decoded, ?_⟩
  rcases contact_correct (realPoint s.p₀) (realPoint s.p₁) (realPoint s.q₀)
    (realPoint s.q₁) s.radius with miss | hit
  · exact miss.1
  · have eqn := decoded_gap s (contactTime (realPoint s.p₀) (realPoint s.p₁)
      (realPoint s.q₀) (realPoint s.q₁) s.radius)
    rw [coefficients] at eqn
    exact absurd (eqn ▸ hit.2.2.1) (not_le.mpr (clear _))

def horizontalPoint (x y : Float) : FloatPoint := fun i => if i = 0 then x else if i = 1 then y else 0

set_option maxRecDepth 8192 in
theorem horizontal_margin_examples :
    let inspect := fun y => certifySweep (horizontalPoint (Float.ofBits 0xc1b0000000000000) y)
      (horizontalPoint (Float.ofBits 0x41b0000000000000) y) (fun _ => 0) (fun _ => 0) 1
    inspect 5 = some .negative ∧ inspect 2 = none := by decide +kernel

set_option maxRecDepth 8192 in
/-- A positive discriminant is compatible with both roots outside the tick interval. -/
theorem positive_outside_tick :
    let z : FloatPoint := fun _ => 0
    certifySweep (horizontalPoint 2 0) (horizontalPoint 3 0) z z 1 = some .positive ∧
    (contact64 (horizontalPoint 2 0) (horizontalPoint 3 0) z z 1).toBits = 0xbff0000000000000 := by
  decide +kernel

end ExactBallpark.Collision

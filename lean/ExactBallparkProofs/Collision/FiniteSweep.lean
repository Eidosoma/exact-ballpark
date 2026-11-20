-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.Sweep
public import Init.Data.Float

@[expose] public section
namespace ExactBallpark.Collision

abbrev FloatPoint := Fin 3 → Float

def diff64 (p q : FloatPoint) : FloatPoint := fun i => p i - q i

def dot64 (p q : FloatPoint) := p 0 * q 0 + p 1 * q 1 + p 2 * q 2

/-- Keep both products and the final subtraction in the scalar source order. -/
def discriminant64 (a b c : Float) := b*b - (4*a)*c

/-- The scalar solver only forms these coefficients when the initial overlap test falls through. -/
def sweepQuadratic64 (p₀ p₁ q₀ q₁ : FloatPoint) (r : Float) : Option (Float × Float × Float) :=
  let d := diff64 p₀ q₀
  let d₂ := dot64 d d
  if d₂ ≤ r*r then none else
  let w := diff64 (diff64 p₁ p₀) (diff64 q₁ q₀)
  let a := dot64 w w
  let b := 2 * dot64 d w
  let c := d₂ - r*r
  some (a, b, c)

/-- Binary64 scalar transcription of SpaceBallpark.collide_two_spheres. -/
def contact64 (p₀ p₁ q₀ q₁ : FloatPoint) (r : Float) : Float :=
  match sweepQuadratic64 p₀ p₁ q₀ q₁ r with
  | none => 0
  | some (a, b, c) =>
    if a == 0 && b == 0 then -1 else
    let det := discriminant64 a b c
    if det < 0 then -1 else
    let s := ((-b - det.sqrt) * Float.ofBits 0x3fe0000000000000) / a
    if 0 ≤ s ∧ s ≤ 1 then s else -1

def sweepPoint (x : ℝ) : Point := fun i => if i = 0 then x else if i = 1 then 2 else 0
def sweepFloatPoint (x : Float) : FloatPoint := fun i => if i = 0 then x else if i = 1 then 2 else 0

/-- Along the whole real sweep, squared distance minus radius squared is at least three. -/
theorem long_chord_clear (t : ℝ) :
    3 ≤ gap (sweepPoint (-(2^28))) (sweepPoint (2^28)) 0 0 1 t := by
  norm_num [gap, chord, dot, sweepPoint]
  nlinarith [sq_nonneg (-268435456 + t * 536870912)]

theorem long_chord_real_miss :
    contactTime (sweepPoint (-(2^28))) (sweepPoint (2^28)) 0 0 1 = -1 := by
  rcases contact_correct (sweepPoint (-(2^28))) (sweepPoint (2^28)) 0 0 1 with clear | hit
  · exact clear.1
  · linarith [hit.2.2.1, long_chord_clear (contactTime (sweepPoint (-(2^28))) (sweepPoint (2^28)) 0 0 1)]

set_option maxRecDepth 8192 in
/-- Rounded coefficients cancel in the discriminant, producing a spurious contact at t = 1/2. -/
theorem long_chord_rounded_hit :
    (contact64 (sweepFloatPoint (Float.ofBits 0xc1b0000000000000))
      (sweepFloatPoint (Float.ofBits 0x41b0000000000000)) (fun _ => 0) (fun _ => 0) 1).toBits =
        0x3fe0000000000000 := by decide

end ExactBallpark.Collision

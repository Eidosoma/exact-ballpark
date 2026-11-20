-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Init.Data.Float

@[expose] public section

namespace ExactBallpark.Numerics

/-- Lean 4.36's native narrowing primitive is opaque. Use the core logical
binary32 rounding and packing operations with the unpacked binary64 input. -/
def narrow (x : Float) : Float32 :=
  let rounded := match x.toModel.unpack with
    | .finite sign mantissa exponent _ =>
      Float.Model.UnpackedFloat.round .binary32 sign mantissa exponent
    | special => special
  .ofModel (Float32.Model.pack rounded)

/-- A successful write also triggers refresh of the cached decay in ref.set_agility. -/
def agilityWrite (x : Float) : Option Float32 :=
  let a := narrow x
  if a ≤ Float32.ofBits 0 then none else some a

theorem agility_accepts_iff (x : Float) :
    (agilityWrite x).isSome = true ↔
    match (narrow x).toModel.unpack with
    | .finite .positive .. | .infinity .positive | .notANumber => True
    | _ => False := by
  unfold agilityWrite
  change (if (narrow x).toModel.unpack.le (.zero .positive) = true then none else some (narrow x)).isSome = true ↔ _
  cases (narrow x).toModel.unpack with
  | infinity s | zero s => cases s <;> simp [Float.Model.UnpackedFloat.le, Float.Model.UnpackedFloat.compare]
  | finite s _ _ _ => cases s <;> simp [Float.Model.UnpackedFloat.le, Float.Model.UnpackedFloat.compare]
  | notANumber => exact ⟨fun _ => trivial, fun _ => rfl⟩

theorem agility_half_subnormal :
    (narrow (Float.ofBits 0x3690000000000000)).toBits = 0 ∧
    (narrow (Float.ofBits 0xb690000000000000)).toBits = 0x80000000 ∧
    agilityWrite (Float.ofBits 0x3690000000000000) = none ∧
    agilityWrite (Float.ofBits 0xb690000000000000) = none := by decide

theorem agility_above_half_subnormal :
    ((agilityWrite (Float.ofBits 0x3690000000000001)).map Float32.toBits) = some 1 ∧
    agilityWrite (Float.ofBits 0xb690000000000001) = none := by decide

/-- Promote a stored binary32 field to binary64 through the logical encodings. -/
def widen (x : Float32) : Float :=
  let converted := match x.toModel.unpack with
    | .finite sign mantissa exponent _ =>
      Float.Model.UnpackedFloat.round .binary64 sign mantissa exponent
    | special => special
  .ofModel (Float.Model.pack converted)

/-- MissileFollow narrows the summed radii, whereas EvolveMissile adds the two
promoted radius fields separately. The ideal cancellation requires an exact sum. -/
def pursuitOffset (r₁ r₂ : Float32) :=
  let stored := -(narrow (widen r₁ + widen r₂))
  (widen stored + widen r₁) + widen r₂

set_option maxRecDepth 8192 in
theorem pursuit_offset_witness :
    (pursuitOffset (Float32.ofBits 0x3f800000) (Float32.ofBits 0x33800000)).toBits =
      0x3e70000000000000 := by decide

end ExactBallpark.Numerics

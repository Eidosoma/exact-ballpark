-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Init.Data.Float

@[expose] public section

namespace ExactBallpark.Numerics

/-- The binary64 mass clamp in Ballpark.add_ball. -/
def clampMass (x : Float) : Float := if x < Float.ofBits 0 then Float.ofBits 0 else x

/-- Equality convention of test_movement_m2_corners.same. -/
def same (x y : Float) : Prop := x.toBits = y.toBits ∨ (x.isNaN = true ∧ y.isNaN = true)

theorem clamp_negative {x : Float} (h : x < Float.ofBits 0) :
    (clampMass x).toBits = 0 := by rw [clampMass, ite_eq_left h]; rfl

theorem clamp_other {x : Float} (h : ¬ x < Float.ofBits 0) :
    (clampMass x).toBits = x.toBits := by rw [clampMass, ite_eq_right h]

theorem clamp_nan {x : Float} (h : x.isNaN = true) : same (clampMass x) Float.nan := by
  have hn : ¬ x < Float.ofBits 0 := by
    change x.toModel.unpack.isNaN = true at h
    change ¬ x.toModel.unpack.lt (Float.ofBits 0).toModel.unpack = true
    cases hx : x.toModel.unpack <;> rw [hx] at h <;> first | contradiction | decide
  exact Or.inr ⟨by simpa [clampMass, hn] using h, rfl⟩

theorem clamp_negative_zero : (clampMass (Float.ofBits 0x8000000000000000)).toBits =
    0x8000000000000000 := by decide

theorem signed_zeros_distinct : ¬ same (Float.ofBits 0) (Float.ofBits 0x8000000000000000) := by
  unfold same
  decide

end ExactBallpark.Numerics

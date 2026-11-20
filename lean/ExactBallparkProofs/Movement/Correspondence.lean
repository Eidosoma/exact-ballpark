-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Reference
public import ExactBallparkProofs.Movement.SpecialBranch

@[expose] public section
namespace ExactBallpark.Movement

/-- No branch or cache invariant is required for the two real programs to agree. -/
theorem extracted_integrator (p v a m k dt cached t : ℝ) :
    Reference.real p v a m k cached t dt = integrate p v a m k dt cached t := rfl

/-- For any supplied cached bits, every nonzero full tick takes the same binary64
velocity operations in both position branches. expFn need not implement any particular library. -/
theorem full_tick_velocity_bits (expFn : Float → Float) (p v a m k cached dt : Float)
    (nonzero : (dt == Float.ofBits 0) = false) (comparable : (dt != dt) = false) :
    (Reference.binary64 expFn p v a m k cached dt dt).2.toBits =
      ((m*a - (m*a - v*k)*cached) * (Float.ofBits 0x3ff0000000000000 / k)).toBits := by
  simp only [Reference.binary64, nonzero, comparable, Bool.false_eq_true, ↓reduceIte]

theorem zero_duration_bits (expFn : Float → Float) (p v a m k cache t dt : Float)
    (zero : (t == Float.ofBits 0) = true) :
    let result := Reference.binary64 expFn p v a m k cache t dt
    (result.1.toBits, result.2.toBits) = (p.toBits, v.toBits) := by
  simp [Reference.binary64, zero]

set_option maxRecDepth 8192 in
/-- The prescribed acceleration gives terminal speed 2^60 on this finite MOV-703
stress input. Rounding erases the metre predicted by the first-order position branch.
The stored decay is the scalar host's full-tick value. -/
theorem special_rounding_witness :
    let one := Float.ofBits 0x3ff0000000000000
    let p := Float.ofBits 0
    let result := Reference.binary64 id p one (Float.ofBits 0x4130000000000000)
      (Float.ofBits 0x4270000000000000) one (Float.ofBits 0x3fefffffffffe000) one one
    result.1.toBits = 0 ∧ (p + one*one).toBits = 0x3ff0000000000000 := by decide

end ExactBallpark.Movement

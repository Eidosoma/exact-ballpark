-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.FiniteSweep
public import ExactBallparkProofs.Numerics.Agility

@[expose] public section
namespace ExactBallpark.Collision
open ExactBallpark.Numerics

/-- AddImpact computes the numerator at field precision. Speed squared, division
and component scaling use binary64. The input dv is the unmodified impulse. -/
def limitImpact64 (dv : FloatPoint) (maxVel : Float32) : FloatPoint :=
  let numerator := (25 : Float32) * maxVel * maxVel
  let fraction := widen numerator / dot64 dv dv
  if fraction < 1 then fun i => dv i * fraction else dv

set_option maxRecDepth 8192 in
/-- For maxVel = 4097 the positive x impulse 20485 + 2⁻¹³ exceeds the nominal
bound. The rounded numerator makes AddImpact accept it at its original bits. -/
theorem rounded_limiter_exceeds_bound :
    let x := Float.ofBits 0x40d4014002000000
    let dv : FloatPoint := fun i => if i = 0 then x else 0
    let maxVel : Float32 := 4097
    (5 : Float)*widen maxVel < x ∧
      ((25 : Float32)*maxVel*maxVel).toBits = 0x4dc81901 ∧
      ((limitImpact64 dv maxVel) 0).toBits = 0x40d4014002000000 := by decide +kernel

end ExactBallpark.Collision

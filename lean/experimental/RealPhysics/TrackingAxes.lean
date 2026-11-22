-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Physics.ContactFrames
public import ExactBallparkProofs.Combat.Execution

@[expose] public section
namespace RealPhysics.TrackingAxes
open ExactBallpark.Combat ExactBallpark.Collision

def cyclic (v : FloatPoint) : FloatPoint := ![v 1, v 2, v 0]

/-- Rotate both stored inputs from (x,y,z) to (y,z,x) with no coordinate arithmetic. -/
def radialProjection (rotated : Bool) : Float :=
  let rotate := if rotated then cyclic else id
  (trackingTerms (rotate radialVelocity) (rotate radialSightline) 1 1 1).1

set_option maxRecDepth 16384 in
theorem cyclic_projection_words :
    (radialProjection false).toBits = radialInputWords.1 ∧
    (radialProjection false).sqrt.isNaN = true ∧ (radialProjection true).toBits = 0 := by
  decide +kernel

end RealPhysics.TrackingAxes

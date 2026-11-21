-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Collision.ReplayTick
import ExactBallparkProofs.Combat.Execution

open ExactBallpark.Collision.ReplayTick

-- Emit the word constants certified by the kernel. The CPU replay independently computes each pair.
private def emit (label : String) (words : UInt64 × UInt64) : IO Unit :=
  IO.println s!"{label} {words.1} {words.2}"

def main : IO Unit := do
  emit "engine-endpoint" (expectedEndpoint false)
  emit "control-endpoint" (expectedEndpoint true)
  emit "return-times" expectedReturnTimes
  emit "radial-inputs" ExactBallpark.Combat.radialInputWords

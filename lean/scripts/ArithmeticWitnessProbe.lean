-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Collision.ReplayTick
import ExactBallparkProofs.Collision.ReboundCutoff
import ExactBallparkProofs.Collision.SearchWindow
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
  for ms in [250,500,1000,2000,4000] do
    for (label,seconds) in [("engine",false),("control",true)] do
      let hits := ExactBallpark.Collision.SearchWindow.fixedPairOrders 0 (ms.toFloat / 1000) seconds
      for ((id,time),order) in hits.zipIdx do
        emit s!"fixed-{ms}-{label}-{order}" (id.toUInt64,time)
  for (run, control) in [("engine",false),("control",true)] do
    let (hits,endpoint) := ExactBallpark.Collision.ReboundCutoff.expected control
    emit s!"detected-{run}-endpoint" endpoint
    for ((id,time,index,kick),i) in hits.zipIdx do
      emit s!"detected-{run}-hit-{i}" (time,kick)
      emit s!"detected-{run}-record-{i}" (id.toNat.toUInt64,index.toUInt64)

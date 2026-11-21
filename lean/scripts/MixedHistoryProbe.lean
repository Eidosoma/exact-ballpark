-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Time.MixedHistory

open ExactBallpark.Time.MixedHistory

private def entries (mask : Nat) (packet : Bool) : List Entry :=
  ((List.range 3).filter (mask.testBit ·)).map fun (i : Nat) =>
    if packet then ⟨i,[100+i,200+i],false⟩ else ⟨i,[10+i],i == 1⟩

def main : IO Unit := do
  for old in [0:8] do
    for new in [1:8] do
      for w in [false,true] do
        let rows := merge (entries old false) (entries new true) w
        let encoded := rows.map fun r =>
          s!"{r.stamp}:{if r.waits then 1 else 0}:{String.intercalate "," (r.events.map toString)}"
        IO.println (String.intercalate "|" encoded)

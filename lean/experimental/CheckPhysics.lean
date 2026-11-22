-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ProofAudit
import RealPhysics

run_cmd do
  ProofAudit.checkLibrary `RealPhysics "experimental/RealPhysics"
  let rows ← ProofAudit.registered `RealPhysics "experimental/claims.json"
  IO.println s!"registered {rows.size}"

private def readRat (word : String) : IO ℚ := do
  let [n, d] := word.splitOn "/" | throw <| IO.userError "expected numerator/denominator"
  let some n := n.toInt? | throw <| IO.userError "invalid numerator"
  let some d := d.toNat? | throw <| IO.userError "invalid denominator"
  if d == 0 then throw <| IO.userError "zero denominator"
  return (n : ℚ) / d

/-- Evaluate the history predicate on input bits and the settling formula on rational SI data. -/
def main (args : List String) : IO Unit := do
  for row in args do
    match row.splitOn "," with
    | ["A", frame] =>
      if frame != "xyz" && frame != "yzx" then throw <| IO.userError "unknown tracking frame"
      IO.println (RealPhysics.TrackingAxes.radialProjection (frame == "yzx")).toBits
    | ["C", dt, u] =>
      let some dt := dt.toNat? | throw <| IO.userError "invalid duration bits"
      let some u := u.toNat? | throw <| IO.userError "invalid fraction bits"
      let (raw, timed) := RealPhysics.TimeGuards.repeatDecisions
        (Float.ofBits dt.toUInt64) (Float.ofBits u.toUInt64)
      IO.println s!"{raw} {timed}"
    | ["S", r, g, solid, fluid, eta] =>
      let v := RealPhysics.FallingSphere.stokesSpeed (← readRat r) (← readRat g)
        (← readRat solid) (← readRat fluid) (← readRat eta)
      IO.println s!"{v.num}/{v.den}"
    | ["D", duration] =>
      if duration != "0.5" && duration != "2" then throw <| IO.userError "unknown drag witness"
      let (q, p, v) := RealPhysics.TimeGuards.dragWitness (duration == "2")
      IO.println s!"{q.toBits} {p.toBits} {v.toBits}"
    | _ => throw <| IO.userError "unknown physical probe or wrong number of inputs"

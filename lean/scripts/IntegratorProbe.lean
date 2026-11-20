-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Movement.Reference

/-- Input per row: p,v,a,m,k,cache,t,dt and the supplied partial-tick exp result. -/
def main (args : List String) : IO Unit := do
  for row in args do
    let bits ← (row.splitOn ",").mapM fun word => do
      let some n := word.toNat? | throw <| IO.userError "invalid binary64 encoding"
      if n ≥ 2^64 then throw <| IO.userError "encoding out of range"
      pure (Float.ofBits n.toUInt64)
    let [p,v,a,m,k,cache,t,dt,expVal] := bits | throw <| IO.userError "expected nine inputs"
    let (pn, vn) := ExactBallpark.Movement.Reference.binary64 (fun _ => expVal) p v a m k cache t dt
    IO.println s!"{pn.toBits} {vn.toBits}"

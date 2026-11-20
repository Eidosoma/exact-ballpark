-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Numerics.Clamp
import ExactBallparkProofs.Numerics.Agility

open ExactBallpark.Numerics

/-- Emit the mass bits and optional agility write for the same binary64 input. -/
def main (args : List String) : IO Unit := do
  for arg in args do
    let some bits := arg.toNat? | throw <| IO.userError s!"invalid bits {arg}"
    if bits ≥ 2^64 then throw <| IO.userError s!"outside binary64 {arg}"
    let input := Float.ofBits bits.toUInt64
    let write := ((agilityWrite input).map (fun a => toString a.toBits)).getD "-"
    IO.println s!"{(clampMass input).toBits} {write}"

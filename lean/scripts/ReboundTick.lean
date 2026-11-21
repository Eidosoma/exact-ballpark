-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import Lean.Data.Json
import ExactBallparkProofs.Collision.ReboundTick

open Lean ExactBallpark.Collision.ReboundFlight ExactBallpark.Collision.ReboundTick

private def vector (v : PointWords) := toJson [v.1.toNat,v.2.1.toNat,v.2.2.toNat]

private def row (name : String) (id : Nat) (s : SideWords) := Json.mkObj [
  ("case", toJson name), ("id", toJson id), ("cache", toJson s.cache.toNat), ("time", toJson s.time.toNat),
  ("thrust", vector s.thrust), ("incomingPos", vector s.incomingPos), ("incomingVel", vector s.incomingVel),
  ("raw", vector s.raw), ("applied", vector s.applied), ("contactPos", vector s.contactPos),
  ("outgoingVel", vector s.outgoingVel), ("pos", vector s.pos), ("vel", vector s.vel)]

def main : IO UInt32 := do
  for (name,c) in [("equal", Scenario.equalMass), ("unequal", .unequalMass), ("limited", .limited)] do
    let some (a,b) := replay c | IO.eprintln s!"Unresolved one-impact tick for {name}"; return 1
    IO.println (row name 1 a).compress
    IO.println (row name 2 b).compress
  return 0

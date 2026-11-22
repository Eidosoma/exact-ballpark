-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Time.MixedHistory

open ExactBallpark.Time.MixedHistory

private def entries (mask : Nat) (packet : Bool) : List Entry :=
  ((List.range 3).filter (mask.testBit ·)).map fun (i : Nat) =>
    if packet then ⟨i,[100+i,200+i],false⟩ else ⟨i,[10+i],i == 1⟩

private def encode (rows : List Entry) := String.intercalate "|" (rows.map fun r =>
  s!"{r.stamp}:{if r.waits then 1 else 0}:{String.intercalate "," (r.events.map toString)}")

private def queueTraces : IO Unit := do
  for count in [2,3,5] do
    let groups : List Entry := (List.range count).map fun i => ⟨(3+i : Nat),[10+i,20+i],false⟩
    let last : Int := (2+count : Nat)
    for complete in [false,true] do
      let history := if complete then groups.foldl (fun h row =>
        let events := if row.stamp == last then row.events ++ [99] else row.events
        merge h [{row with events}] false) []
        else merge (merge [] groups true) [⟨last,[99],false⟩] false
      for now in [2,12,100] do
        let (applied,clock,pending) := drain (fun c _ => c) now history
        IO.println s!"{count} {complete} {now} {clock} [{encode applied}] [{encode pending}]"

def main (args : List String) : IO Unit := do
  if args == ["drain"] then return ← queueTraces
  unless args.isEmpty do throw <| IO.userError "expected no arguments or drain"
  for old in [0:8] do
    for new in [1:8] do
      for w in [false,true] do
        let rows := merge (entries old false) (entries new true) w
        IO.println (encode rows)

-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Collision.ImpactHistory
import ExactBallparkProofs.Collision.SearchWindow

open ExactBallpark.Collision.History

-- Zero impulses on a stationary ball make the integrator an identity in this trace.
private def motion : Motion := ⟨fun _ => 0, fun _ => 0⟩
private def step : Integrate := fun state _ => state
private def event (t : Float) (id : Int) : Impact := ⟨t, id, fun _ => 0⟩

private def completed (p : Path) : IO Path :=
  match finish step 1 22 p with
  | some result => pure result
  | none => throw (IO.userError "unexpected end-of-tick failure")

private def stamp (t : Float) (id : Int) := s!"{t.toBits}/{id}"

private def emit (p : Path) (removed : List (Float × Int)) : IO Unit := do
  let locs := p.locations.map (fun loc => stamp loc.time loc.collider)
  let events := removed.map (fun (t,id) => stamp t id)
  IO.println (String.intercalate "|" [toString p.startTime.toBits, toString p.endTime.toBits,
    String.intercalate "," locs, String.intercalate "," events])

def main : IO Unit := do
  let mut path : Path := ⟨[⟨0,-1,motion⟩, ⟨1,-1,motion⟩], 0, 1⟩
  for (t,id) in [(0.25,10), (0.5,11), (0.75,12)] do
    path ← completed (insert step motion path (event t id)).1
  let (cut, removed) := insert step motion path (event 0.25 13)
  emit cut removed
  let (retry, repeated) := insert step motion cut (event 0.25 14)
  emit retry repeated
  for id in [10,99] do
    let (result, dropped) := remove step 1 22 motion path 0.25 id
    let some result := result | throw (IO.userError "unexpected removal failure")
    emit result dropped
  IO.println s!"{ignoreCollision path.locations 12 delta} {ignoreCollision path.locations 11 delta}"
  for order in [[(3,3),(4,5)],[(4,5),(3,3)]] do
    let (id,t) := ExactBallpark.Collision.SearchWindow.fixedReplay order
    IO.println s!"{id} {t}"

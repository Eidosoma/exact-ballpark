-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Space.SensorDelivery
import ExactBallparkProofs.Space.FieldQuery
import ExactBallparkProofs.Time.CommandFlight
import ExactBallparkProofs.Warp.Journey
import ExactBallparkProofs.Movement.WorldMask
import ExactBallparkProofs.Combat.InputMargin

open ExactBallpark ExactBallpark.Movement.Scalar

private def motion (b : Ball) : String :=
  String.intercalate " " (([b.pos.x,b.pos.y,b.pos.z,b.vel.x,b.vel.y,b.vel.z]).map
    (fun v => toString v.toBits))

private def mode (b : Ball) := if b.mode = .goto then 0 else 2

def main : IO Unit := do
  let states := Space.SensorDelivery.crossingStates.drop 1
  for ((stamp,x),s) in Space.SensorDelivery.crossingTrace.zip states do
    let flags := s.callbacks.filterMap fun (n,flag) => if n == stamp then some flag.toNat else none
    IO.println s!"sensor {stamp} {x.toBits} {s.member.toNat} {s.elapsed.toBits} {flags.head?.getD 2}"
  for known in [true,false] do
    for live in [true,false] do
      for forward in [true,false] do
        let audience := fun (id : Int) => if id = 3 then ({10000} : Finset Int) else {10001}
        for i in [10000,10001,10002] do
          let sends := Space.SensorDelivery.packets (if forward then some 1 else none) 3 7
            (fun _ => known) (fun _ => live) {3,4} audience i
          IO.println s!"route {known.toNat} {live.toNat} {forward.toNat} {i} {sends.length}"
  let tags : Time.ServerLaws.Tags := ⟨7,8,9⟩
  let old : Time.ServerLaws.Field := ⟨false,.stop,⟨-1,-1,-1⟩⟩
  for ship in ([⟨7,-1,-1⟩,⟨-1,8,-1⟩,⟨-1,-1,9⟩,⟨-2,-1,-1⟩,⟨6,7,8⟩] : List Time.ServerLaws.Tags) do
    let ids := Space.FieldQuery.configuredQuery old tags ship 9 [6,9,9,6]
    IO.println s!"field {ship.harmonic} {ship.corp} {ship.alliance} {String.intercalate " " (ids.map toString)}"
  for (name,table) in [("exp",Warp.Journey.expWords),("log",Warp.Journey.logWords)] do
    for (arg,result) in table do IO.println s!"{name} {arg} {result}"
  for (s,i) in Warp.Journey.committed.zipIdx do
    IO.println s!"warp {i+1} {if s.phase = .stop then 2 else 3} {s.x} {s.vx}"
  if let some (_,x,v) := Warp.Journey.sample 9 (Float.ofBits 0x40fffb7acd76a807) (Float.ofBits 0x4058710d872a84f1) then
    IO.println s!"drop {x.toBits} {v.toBits}"
  for i in List.range 6 do
    let frame := Time.CommandFlight.sessionFrame i
    IO.println s!"goto {3+i} {frame.sent.getLast?.getD (-1)} {mode frame.ball} {motion frame.ball}"
  let r : Collision.FloatPoint := ![3,4,12]
  let radial : Collision.FloatPoint := ![Float.ofBits 0x3f9eb851eb851eb8, Float.ofBits 0x3fa47ae147ae147b,
    Float.ofBits 0x3fbeb851eb851eb8]
  for (v,p) in [(radial,r),(![4,-3,0],r),(![3,0,0],![4,0,0])] do
    let speedSq := (Combat.trackingTerms v p 1 1 1).1
    IO.println s!"combat {(Combat.InputMargin.safe v p).toNat} {speedSq.toBits} {speedSq.sqrt.isNaN.toNat}"
  let first := Time.CommandFlight.command Time.CommandFlight.sessionBall (vec3 1 0 0)
  let other := {Time.CommandFlight.sessionBall with
    pos := vec3 20 0 (Float.ofBits 0x8000000000000000), vel := vec3 0 (-8) 3}
  let mut rows : Bool → Movement.WorldMask.Row := fun w => ⟨if w then other else first,0⟩
  for (left,right) in [(false,true),(true,false),(true,false),(false,true),(true,true)] do
    rows := Movement.WorldMask.masked ⟨1000000,1⟩ id rows (fun w => if w then right else left)
    for w in [false,true] do
      let row := rows w
      IO.println s!"world {w.toNat} {row.clock} {mode row.ball} {motion row.ball}"

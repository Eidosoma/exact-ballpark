-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.ImpactHistory
public import ExactBallparkProofs.Collision.SearchWindow
public import ExactBallparkProofs.Movement.Reference

/-! One-axis projection of the four-ball CPU replay. There is no thrust or spin.
Only the nine host exponential results are supplied. Candidate order is the scalar
filing order. History insertion, fuzzy lookup, and duplicate suppression use the
same definitions as the previous impact-history proofs. -/
@[expose] public section
namespace ExactBallpark.Collision.ReplayTick
open History
open SearchWindow (axis)

def decayWord (x : Float) : Float := Float.ofBits (match x.toBits with
  | 0xbf740e86dc6084c8 => 0x3fefd7fc0c277278
  | 0xbf9081554fa51cfd => 0x3fef7d044c52660f
  | 0xbf9530132eb1d60b => 0x3fef583d3e28d113
  | 0xbf954af5ca90ebc6 => 0x3fef576a939fc61b
  | 0xbf9625a21050166c => 0x3fef50b9e882bdb9
  | 0xbf995908a00acc97 => 0x3fef37b4f8d2250a
  | 0xbf9a4e9781a90cf8 => 0x3fef30396f61568e
  | 0xbfa7bf55582d7182 => 0x3fee8cb7f25b4b2e
  | 0xbfb0000000000000 => 0x3fee0fabfbc702a4
  | _ => 0x7ff8000000000000)

def advance (a : Float) (m : Motion) (dt : Float) : Motion :=
  let (p,v) := Movement.Reference.binary64 decayWord (m.pos 0) (m.vel 0) a 16 1
    (Float.ofBits 0x3fee0fabfbc702a4) dt 1
  ⟨axis p, axis v⟩

def initial : Motion := ⟨axis 105, axis (-16)⟩

structure FixedHit where
  id : Int
  time : Float
  normal : Float

def fixedHit (absolute : Bool) (path : Path) (p q : Motion) : Option FixedHit := Id.run do
  let mut best : Option FixedHit := none
  for (id, x) in ([(2,100), (3,107), (4,108)] : List (Int × Float)) do
    let u := contact64 p.pos q.pos (axis x) (axis x) 1
    if 0 ≤ u && !ignoreCollision path.locations id u then
      let t := path.startTime + u * (path.endTime-path.startTime)
      if best.all (fun hit => decide ((if absolute then t else u) < hit.time)) then
        let dx := (p.pos 0 + (q.pos 0 - p.pos 0) * u) - x
        let length := (dx*dx).sqrt
        best := some ⟨id,t,if length == 0 then dx else dx * (1/length)⟩
    if best.any (fun hit => hit.time == path.startTime) then break
  return best

/-- A separating contact returns the value-initialised impact with collider zero.
At a start contact the acceleration has already been written at that point. -/
def passes (absolute : Bool) : Nat → Path → Float → Option Motion
  | 0, path, a => position (advance a) path.locations 1
  | n+1, path, a => do
    let p ← position (advance a) path.locations path.startTime
    let q ← position (advance a) path.locations path.endTime
    let some hit := fixedHit absolute path p q | return q
    let a' := if path.startTime == hit.time then hit.normal * (16*1/16) else a
    let atHit ← position (advance a') path.locations hit.time
    let impulse := ((atHit.vel 0 * (-2))*hit.normal) + 0 + 0
    let (id,dv) := if impulse < 0 then (0,0) else (hit.id,hit.normal * impulse)
    let fraction := 6400 / (dv*dv)
    let kick := if fraction < 1 then dv*fraction else dv
    let path' := (insert (advance a') initial path ⟨hit.time,id,axis kick⟩).1
    let next ← finish (advance a') 1 22 path'
    passes absolute n next a'

def endpoint (absolute : Bool) := do
  let path : Path := ⟨[⟨0,-1,initial⟩, ⟨1,-1,advance 0 initial 1⟩],0,1⟩
  let m ← passes absolute 20 path 0
  return ((m.pos 0).toBits, (m.vel 0).toBits)

/-- End-of-tick x and vx, using the engine comparison or the absolute-time control. -/
def expectedEndpoint (absolute : Bool) : UInt64 × UInt64 :=
  if absolute then (0x40598ac26d594d72, 0x402e0fabfbc702a3) else (0x40597782a556ba6c, 0xc02d6065e449c05c)

set_option maxRecDepth 131072 in
set_option maxHeartbeats 8000000 in
theorem four_ball_endpoint :
    endpoint false = some (expectedEndpoint false) ∧
    endpoint true = some (expectedEndpoint true) := by decide +kernel

/-- Two consecutive equal-duration chords of the supplied free-ball history.
At time zero the two centers are one combined radius apart. -/
def repeatSearch (includeFirst : Bool) : Float := Id.run do
  let half := Float.ofBits 0x3fe0000000000000
  let endpoints : List (Float × Float) := [(0,1), (half, Float.ofBits 0x4004974d039069f6),
    (1, Float.ofBits 0xbff317244b8f6afc)]
  let path := if includeFirst then endpoints else endpoints.drop 1
  let mut x := path.head!.2
  for (left,right) in path.zip path.tail do
    let u := contact64 (axis 0) (axis 0) (axis x) (axis right.2) 1
    if 0 ≤ u then
      if u ≤ delta then continue
      return left.1 + (right.1-left.1)*u
    x := right.2
  return -1

def expectedReturnTimes : UInt64 × UInt64 := (0xbff0000000000000, 0x3fe6af52c7e6490c)

set_option maxRecDepth 16384 in
theorem returning_ball_words :
    (repeatSearch true).toBits = expectedReturnTimes.1 ∧
    (repeatSearch false).toBits = expectedReturnTimes.2 := by decide +kernel

end ExactBallpark.Collision.ReplayTick

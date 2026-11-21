-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.TensorLane

@[expose] public section
namespace ExactBallpark.Warp.Journey

/-- The host elementary-function words used by this journey. An unlisted
argument fails the replay, including one reached by an unexpected phase. -/
def expWords : List (UInt64 × UInt64) := [
  (0xbff0000000000000,0x3fd78b56362cef38), (0,0x3ff0000000000000),
  (0x4010000000000000,0x404b4c902e273a58), (0x4020000000000000,0x40a749ea7d470c6e),
  (0xbfe118abafc8de55,0x3fe2c158d859ab9a), (0xbffde1ab2d39c480,0x3fc3c67c4106e595),
  (0xc0099b8041478cea,0x3fa4d9d39d407091), (0xc012231575f91bca,0x3f85fc24ae150c06),
  (0xc017786acb4e7120,0x3f672e3ff5ffe890), (0xc01ccdc020a3c675,0x3f4871014ea9dd9c),
  (0xc021118abafc8de5,0x3f29c55086cfb9d7)]

def logWords : List (UInt64 × UInt64) :=
  [(0x40e0000800000000,0x4024cb5fcf0a5650),(0x40f8000c00000000,0x4026fddd1e5ff6b2)]

def host (table : List (UInt64 × UInt64)) (x : Float) :=
  (table.lookup x.toBits).map Float.ofBits

inductive Phase where | align | warp | stop deriving DecidableEq, Repr

structure Ship where
  phase : Phase
  x : UInt64
  vx : UInt64
  deriving DecidableEq, Repr

def ship (phase : Phase) (x vx : Float) : Ship := ⟨phase,x.toBits,vx.toBits⟩

def lengthX (x : Float) := (x*x + 0 + 0).sqrt
def unitX (x : Float) := if lengthX x == 0 then x else x * (1 / lengthX x)

def destination : Float := 131072

/-- The accepted long-distance branch of a fresh WarpTo with minRange zero. -/
def begin : Option Ship :=
  let x : Float := 0
  let delta := x-destination
  if !destination.isFinite || delta*delta < 10000000000 then none else some (ship .align x 80)

/-- Full one-second ordinary integration, on the x-axis with zero acceleration.
STOP's vertical damping leaves the axial input unchanged. -/
def coast (x vx : Float) : Option Ship := do
  let q ← host expWords (-1)
  let (p,v) := Movement.TensorLane.atFactor x vx 0 1 1 q 1
  return ship .stop p v

/-- _warp_distance on this axial flight, including the short-cruise and overrun
guards. The elapsed stamp is measured from activation at zero. -/
def sample (stamp : Nat) (x vx : Float) : Option (Bool × Float × Float) := do
  let t := Float.ofNat stamp
  let a := 4000 * Float.ofBits 0x3f50624dd2f1a9fc
  let b := min (4000 * Float.ofBits 0x3f35d867c3ece2a5) 2
  let s := min (a*149597870700) ((destination+1)*a*b/(a+b))
  let ta := (← host logWords (s/a))/a
  let tc := destination/s - 1/a - 1/b + 1/s
  let td := (← host logWords (s/b))/b
  let da := s/a
  let dc := s*tc
  let dd := s/b-1
  let dir := unitX (destination-x)
  let place := fun progress => destination-dir*(da+dc+dd-progress)
  if t < ta then
    let e ← host expWords (a*t)
    let speed := a*e
    return (false,place e,dir*(if lengthX vx > speed then lengthX vx else speed))
  if t-ta < tc then return (false,place (s*(t-ta)+da),dir*s)
  if t-tc-ta < td+1 then
    let e ← host expWords (-b*(t-tc-ta))
    let speed := s*e
    return (decide (speed < min ((100:Float)/2) 100),place (s/b-(s/b)*e+da+dc),dir*speed)
  return (true,destination,0)

def tick (stamp : Nat) (state : Ship) : Option Ship := do
  let x := Float.ofBits state.x
  let vx := Float.ofBits state.vx
  if state.phase = .stop then return ← coast x vx
  if state.phase = .align then
    let aligned := (1-unitX vx*unitX (destination-x)).abs < Float.ofBits 0x3f847ae147ae147b
    if !(decide aligned && decide (vx*vx > (0.5625:Float)*100*100)) then none
  let (dropped,p,v) ← sample stamp x vx
  if dropped then coast p v else return ship .warp p v

def committed : List Ship := [
  ⟨.warp,0x3ff0000000000000,0x4054000000000000⟩, ⟨.warp,0x404b4c902e273800,0x406b4c902e273a58⟩,
  ⟨.warp,0x40a749ea7d470c80,0x40c749ea7d470c6d⟩, ⟨.warp,0x40f1ef06553b6e2c,0x40f2c162390617c6⟩,
  ⟨.warp,0x40fc4ad6d9330edb,0x40d3c68624450618⟩, ⟨.warp,0x40ff05d99786050a,0x40b4d9de0a2a3f31⟩,
  ⟨.warp,0x40ffbe1b70fb89d7,0x4095fc2fac276311⟩, ⟨.warp,0x40ffeead47562815,0x40772e4b8d1fe390⟩,
  ⟨.warp,0x40fffb7acd76a807,0x4058710d872a84f1⟩, ⟨.stop,0x40ffffdf64e6a8fb,0x4022f611df9ca855⟩,
  ⟨.stop,0x4100001fa3e13b64,0x400be6d9978ba15c⟩]

/-- An edge is accepted only if tick computes the next stored state. -/
def follows (stamp : Nat) (previous : Option Ship) : List Ship → Bool
  | [] => true
  | next :: rest => decide (previous.bind (tick stamp) = some next) && follows (stamp+1) (some next) rest

set_option maxRecDepth 32768 in
set_option maxHeartbeats 1200000 in
theorem accepted_warp_commits : follows 0 begin committed = true := by decide +kernel

set_option maxRecDepth 16384 in
/-- At dropout the sampled state passes through ordinary drag before commit.
The following STOP tick integrates from that committed pair. -/
theorem dropout_handoff :
    let p := Float.ofBits 0x40fffedabf9f0e60
    let v := Float.ofBits 0x4039c55d6977fd3f
    let exitState : Ship := ⟨.stop,0x40ffffdf64e6a8fb,0x4022f611df9ca855⟩
    (sample 9 (Float.ofBits 0x40fffb7acd76a807) (Float.ofBits 0x4058710d872a84f1)).map
      (fun (drop,x,vx) => (drop,x.toBits,vx.toBits)) = some (true,p.toBits,v.toBits) ∧
    coast p v = some exitState ∧ tick 10 exitState =
      some ⟨.stop,0x4100001fa3e13b64,0x400be6d9978ba15c⟩ := by decide +kernel

end ExactBallpark.Warp.Journey

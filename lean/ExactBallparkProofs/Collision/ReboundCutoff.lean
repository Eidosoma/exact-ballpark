-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Collision.ReplayTick
public import ExactBallparkProofs.Collision.Discriminant

@[expose] public section
namespace ExactBallpark.Collision.ReboundCutoff
open History
open SearchWindow (axis)
open ReplayTick (fixedCandidates)

def dt := Float.ofBits 0x3fb999999999999a
def initial : Motion := ⟨axis (Float.ofBits 0xc089821b123583d1),axis 16384⟩
def walls : List (Int × Float) := [(2,Float.ofBits 0xbff6cffb2ca84a00),(3,Float.ofBits 0xbfd738a574389bfd)]

/-- Supplied host exp values for the two runs. An unlisted argument poisons its lane. -/
def decay (x : Float) := Float.ofBits (match x.toBits with
  | 0xbff999999999999a => 0x3fc9d7bebefb4e15
  | 0xbff985f06f695459 => 0x3fc9f793b729d6da
  | 0xbf73a92a30454100 => 0x3fefd8c5ca7df6ef
  | 0xbff98c7e28241b6a => 0x3fc9ecf30b88ab28
  | 0xbf5a36e2eb1c4400 => 0x3feff2e73d5e848c
  | 0xbf6a36e2eafc6000 => 0x3fefe5d3d6ee74a9
  | 0xbf5a3992055e2800 => 0x3feff2e5e65e05f4
  | 0xbf5a3433d09a9800 => 0x3feff2e8945f315e
  | _ => 0x7ff8000000000000)

def advance (motion : Motion) (t : Float) : Motion :=
  let (p,v) := Movement.Reference.binary64 decay (motion.pos 0) (motion.vel 0) 0
    (Float.ofBits 0x3fb0000000000000) 1 (Float.ofBits 0x3fc9d7bebefb4e15) t dt
  ⟨axis p,axis v⟩

def index (control : Bool) (xs : List Location) (t : Float) : Option Nat := do
  let ts := xs.map Location.time
  if !control then return indexAt ts t
  let times ← ts.mapM finiteValue
  let q ← finiteValue t
  let d ← finiteValue delta
  return (times.takeWhile (fun s => decide (s+d < q))).length - 1

def sample (control : Bool) (xs : List Location) (t : Float) : Option Motion := do
  let loc ← xs[← index control xs t]?
  return if (loc.time-t).abs ≥ delta then advance loc.motion (t-loc.time) else loc.motion

abbrev Contact := Int × UInt64 × Nat × UInt64
abbrev Trace := List Contact × (UInt64 × UInt64)

/-- Ordinary non-start contacts, with the source limiter and 20-pass cap. Both
runs have zero thrust. A start contact would require a new acceleration and is
rejected by this fixture. Collision times come entirely from the sphere sweeps. -/
def passes (control : Bool) : Nat → Path → Option Trace
  | 0, path => do
    let m ← sample control path.locations dt
    return ([],(m.pos 0 |>.toBits,m.vel 0 |>.toBits))
  | n+1, path => do
    let p ← sample control path.locations path.startTime
    let q ← sample control path.locations path.endTime
    let some hit := fixedCandidates walls (Float.ofBits 0x3fa0000000000000) false path p q
      | return ([],(q.pos 0 |>.toBits,q.vel 0 |>.toBits))
    if hit.time == path.startTime then none else
    let i ← index control path.locations hit.time
    let state ← sample control path.locations hit.time
    let impulse := state.vel 0 * (-2) * hit.normal + 0 + 0
    let (id,dv) := if impulse < 0 then (0,0) else (hit.id,hit.normal*impulse)
    let limit := 26843545600 / (dv*dv)
    let kick := if limit < 1 then dv*limit else dv
    let path' := (insert advance initial path ⟨hit.time,id,axis kick⟩).1
    let next ← finish advance dt 22 path'
    let (later,endpoint) ← passes control n next
    return ((hit.id,hit.time.toBits,i,kick.toBits)::later,endpoint)

def replay (control : Bool) : Option Trace :=
  passes control 20 ⟨[⟨0,-1,initial⟩,⟨dt,-1,advance initial dt⟩],0,dt⟩

def expected (control : Bool) : Trace :=
  let first : Contact := (2,0x3fb985f06f695459,0,0xc0b9f793b729d6da)
  if control then ([first,(3,0x3fb98c7e28241b6a,1,0x40b9ecf30b88ab28)],
                   (0x3fd52741d8968a00,0x40a9d7bebefb4e15))
  else ([first,(3,0x3fb98c7e28241b6a,0,0),(3,0x3fb9930c8ca572f4,2,0x40b9e255a38229a7)],
        (0xbfefd4f82e6ee100,0xc0a9d7bebefb4e15))

set_option maxRecDepth 131072 in
set_option maxHeartbeats 8000000 in
theorem detected_rebound :
    replay false = some (expected false) ∧ replay true = some (expected true) := by
  constructor <;> decide +kernel

end ExactBallpark.Collision.ReboundCutoff

-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Numerics.Agility
public import Std.Data.TreeMap.Lemmas

@[expose] public section
namespace ExactBallpark.Movement.Scalar

structure Vec where
  x : Float
  y : Float
  z : Float

instance : CoeFun Vec (fun _ => Fin 3 → Float) where
  coe v i := if i == 0 then v.x else if i == 1 then v.y else v.z

abbrev vec3 := Vec.mk

def Vec.ofFn (f : Fin 3 → Float) : Vec := ⟨f 0, f 1, f 2⟩

def zeros : Vec := vec3 (Float.ofBits 0) (Float.ofBits 0) (Float.ofBits 0)

structure Motion where
  pos : Vec
  vel : Vec

def Motion.ofFn (f : Fin 3 → Float × Float) : Motion :=
  let (px, vx) := f 0
  let (py, vy) := f 1
  let (pz, vz) := f 2
  ⟨vec3 px py pz, vec3 vx vy vz⟩

inductive Mode where
  | stop
  | goto
  deriving DecidableEq

/-- dt is the binary64 duration already calculated by Ballpark.__init__. -/
structure Config where
  friction : Float
  dt : Float

/-- Arguments of add_ball, before the binary32 parameter conversions. -/
structure Spawn where
  mass : Float
  radius : Float
  maxVel : Float
  pos : Vec
  vel : Vec
  agility : Float
  speedFraction : Float

/-- A ball's ID is the key in the park. Scalar float members hold the widened binary32 values. -/
structure Ball extends Spawn where
  mode : Mode := .stop
  target : Vec := zeros
  timeFactor : Float := Float.ofBits 0
  lastG : Vec := zeros

structure Park where
  balls : Std.TreeMap Int Ball := ∅
  clock : Nat := 0

inductive Edit where
  | add (args : Spawn)
  | mass (x : Float)
  | agility (x : Float)
  | maxVelocity (x : Float)
  | radius (x : Float)
  | speedFraction (x : Float)
  | stop
  | gotoPoint (v : Vec)
  | gotoDirection (v : Vec)

inductive Action where
  | edit (id : Int) (command : Edit)
  | tick

def stored (x : Float) : Float := Numerics.widen (Numerics.narrow x)

end ExactBallpark.Movement.Scalar

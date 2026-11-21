-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Movement.CoastWitness

open ExactBallpark.Movement.Scalar
open ExactBallpark.Movement.CoastWitness (endpoint coastExp)

private def word (s : String) : IO Float := do
  let some n := s.toNat? | throw (IO.userError "expected a binary64 word")
  if n ≥ 2^64 then throw (IO.userError "binary64 word out of range")
  return Float.ofBits n.toUInt64

private def action (tokens : List String) : IO Action := do
  if tokens == ["tick"] then return .tick
  let op :: key :: inputs := tokens | throw (IO.userError "expected a command and ball ID")
  let some id := key.toInt? | throw (IO.userError "invalid ball ID")
  let values ← inputs.mapM word
  let cmd : Edit ← match op, values with
    | "add", [m,r,vm,px,py,pz,vx,vy,vz,a,s] =>
      pure (.add ⟨m,r,vm,vec3 px py pz,vec3 vx vy vz,a,s⟩)
    | "mass", [x] => pure (.mass x)
    | "agility", [x] => pure (.agility x)
    | "max_velocity", [x] => pure (.maxVelocity x)
    | "speed_fraction", [x] => pure (.speedFraction x)
    | "radius", [x] => pure (.radius x)
    | "stop", [] => pure .stop
    | "point", [x,y,z] => pure (.gotoPoint (vec3 x y z))
    | "direction", [x,y,z] => pure (.gotoDirection (vec3 x y z))
    | _, _ => throw (IO.userError s!"unknown command or argument count for {op}")
  return .edit id cmd

private def vector (v : Vec) : List Float := [v 0, v 1, v 2]

private def snapshot (s : Park) : String :=
  let rows := s.balls.toList.map fun (id, b) =>
    let mode := if b.mode = .stop then "2" else "0"
    let values := [b.mass, b.radius, b.maxVel, b.agility, b.speedFraction, b.timeFactor] ++
      vector b.pos ++ vector b.vel ++ vector b.target ++ vector b.lastG
    String.intercalate "," ([toString id, mode] ++ values.map (fun x => toString x.toBits))
  String.intercalate "|" (toString s.clock :: rows)

def main (args : List String) : IO Unit := do
  let mut cfg : Config := ⟨0, 0⟩
  let mut state : Option Park := none
  for row in args do
    match row.splitOn "," with
    | ["coast"] =>
      let some (p, v) := endpoint coastExp
        | throw (IO.userError "five-second coast has no finite endpoint")
      let q := coastExp (-1)
      IO.println s!"{q.toBits},{p.num},{p.den},{v.num},{v.den}"
      continue
    | ["park", k, t] =>
      cfg := ⟨← word k, ← word t⟩
      state := run cfg Float.exp {} []
    | tokens =>
      let some s := state | throw (IO.userError "start a new park after a rejected trace")
      state := run cfg Float.exp s [← action tokens]
    IO.println (state.elim "outside" snapshot)

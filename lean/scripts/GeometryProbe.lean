-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Collision.HitCertificate
import ExactBallparkProofs.Combat.TrackingMargin

open ExactBallpark.Collision ExactBallpark.Combat

private def word (s : String) : IO Float := do
  let some n := s.toNat? | throw <| IO.userError "expected a binary64 word"
  if n ≥ 2^64 then throw <| IO.userError "binary64 word out of range"
  return Float.ofBits n.toUInt64

def main (args : List String) : IO Unit := do
  for row in args do
    match row.splitOn "," with
    | "hit" :: precision :: args =>
      let some depth := precision.toNat? | throw <| IO.userError "expected a bisection depth"
      let [x,y,z,u,v,w,r] ← args.mapM word | throw <| IO.userError "expected two endpoints and radius"
      let p : FloatPoint := ![x,y,z]
      let q : FloatPoint := ![u,v,w]
      let origin : FloatPoint := fun _ => 0
      let reported := (contact64 p q origin origin r).toBits
      match certifyHit p q origin origin r depth with
      | none => IO.println s!"hit {reported} uncertain"
      | some cert => IO.println s!"hit {reported} {cert.lower} {cert.upper}"
    | "track" :: args =>
      let [vx,vy,vz,x,y,z] ← args.mapM word | throw <| IO.userError "expected velocity and sightline"
      let v : FloatPoint := ![vx,vy,vz]
      let r : FloatPoint := ![x,y,z]
      let speed := ((trackingTerms v r 1 1 1).1).toBits
      match TrackingMargin.inspect v r with
      | none => IO.println s!"track {speed} uncertain"
      | some cert => IO.println s!"track {speed} {cert.sidewaysSq} {cert.threshold}"
    | _ => throw <| IO.userError "expected hit or track"

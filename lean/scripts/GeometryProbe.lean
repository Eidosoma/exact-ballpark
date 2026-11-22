-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Collision.HitCertificate
import ExactBallparkProofs.Combat.TrackingMargin

open ExactBallpark.Collision ExactBallpark.Combat

private def word (s : String) : IO Float := do
  let some n := s.toNat? | throw <| IO.userError "expected a binary64 word"
  if n ≥ 2^64 then throw <| IO.userError "binary64 word out of range"
  return Float.ofBits n.toUInt64

private def signLabel : Option DiscSign → String
  | some .negative => "negative"
  | some .positive => "positive"
  | none => "uncertain"

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
    | "disc" :: args =>
      let [x,u,y,r] ← args.mapM word | throw <| IO.userError "expected a horizontal chord and radius"
      let p := horizontalPoint x y
      let q := horizontalPoint u y
      let origin : FloatPoint := fun _ => 0
      let some (a,b,c) := sweepQuadratic64 p q origin origin r | throw <| IO.userError "expected a sweep from outside"
      let lastOnly := certifyDiscriminant a b c 0 0 0
      let fromInputs := certifySweep p q origin origin r
      IO.println s!"disc {(contact64 p q origin origin r).toBits} {signLabel lastOnly} {signLabel fromInputs}"
    | kind :: args =>
      unless kind == "track" || kind == "cross" do throw <| IO.userError "expected hit, disc, track or cross"
      let [vx,vy,vz,x,y,z] ← args.mapM word | throw <| IO.userError "expected velocity and sightline"
      let v : FloatPoint := ![vx,vy,vz]
      let r : FloatPoint := ![x,y,z]
      let speed := ((trackingTerms v r 1 1 1).1).toBits
      if kind == "cross" then
        let some vq := decodePoint v | throw <| IO.userError "expected finite velocity"
        let some rq := decodePoint r | throw <| IO.userError "expected finite sightline"
        let plain := TrackingMargin.crossWith (fun a b c d => a*b-c*d) v r
        let fused := TrackingMargin.crossWith TrackingMargin.compensatedDet v r
        let words := [plain 0,plain 1,plain 2,fused 0,fused 1,fused 2]
        IO.println s!"cross {speed} {String.intercalate " " (words.map (toString ·.toBits))} {TrackingMargin.crossAngularSq vq rq}"
      else
        match TrackingMargin.inspect v r with
        | none => IO.println s!"track {speed} uncertain"
        | some cert => IO.println s!"track {speed} {cert.sidewaysSq} {cert.threshold}"
    | _ => throw <| IO.userError "expected hit, disc, track or cross"

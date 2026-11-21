-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import Lean.Data.Json
import ExactBallparkProofs.Collision.ContactExamples

open Lean ExactBallpark.Collision CurveClearance

private def metres (j : Json) : Except String ℚ := do
  let n ← j.getNum?
  return (n.mantissa : ℚ) / 10^n.exponent

/-- A JSON number is converted to binary64. A bits object preserves the original word. -/
private def binary64 (j : Json) : Except String Float := do
  if let .num n := j then return n.toFloat
  let n ← (← j.getObjVal? "bits").getNat?
  if n ≥ 2^64 then throw "binary64 word exceeds 64 bits"
  return Float.ofBits n.toUInt64

private def point (j : Json) : Except String FloatPoint := do
  let #[x,y,z] ← j.getArr? | throw "a position must contain three coordinates"
  return ![← binary64 x, ← binary64 y, ← binary64 z]

private def body (j : Json) : Except String (FloatPoint × FloatPoint × Budget) := do
  let budget : Budget := ⟨← metres (← j.getObjVal? "accelerationBound"),
    ← metres (← j.getObjVal? "startError"), ← metres (← j.getObjVal? "endError")⟩
  return (← point (← j.getObjVal? "start"), ← point (← j.getObjVal? "end"), budget)

private def report (a b c d : FloatPoint) (radius : Float) (duration : ℚ) (p q : Budget) :
    List (String × Json) :=
  let time := ("sweepTimeBits", toJson (contact64 a b c d radius).toBits.toNat)
  match check a b c d radius duration p q with
  | some margin => [time, ("status", "certified clear"), ("marginMetres", toJson (toString margin))]
  | none => [time, ("status", "insufficient clearance")]

private def clearanceRequest (j : Json) : Except String Json := do
  let (a,b,p) ← body (← j.getObjVal? "first")
  let (c,d,q) ← body (← j.getObjVal? "second")
  return Json.mkObj (report a b c d (← binary64 (← j.getObjVal? "radius"))
    (← metres (← j.getObjVal? "duration")) p q)

open ExactBallpark.Movement.PositionEnclosure ContactCertificate

/-- The curved path starts from the exact decoded state. The supplied end belongs
to the engine chord and is used only by that solver. -/
private def flightBody (j : Json) : Except String (FloatPoint × FloatPoint × Option Body) := do
  let start ← point (← j.getObjVal? "start")
  let finish ← point (← j.getObjVal? "end")
  let v ← point (← j.getObjVal? "velocity")
  let a ← point (← j.getObjVal? "applied")
  let m ← binary64 (← j.getObjVal? "effectiveMass")
  let k ← binary64 (← j.getObjVal? "friction")
  let state : Option Body := do
    return ⟨← decodePoint start, ← decodePoint v, ← decodePoint a, ← finiteValue m, ← finiteValue k⟩
  return (start, finish, state)

private def contactReport (a b c d : FloatPoint) (r : Float) (scene : Option Scene) (cells : Nat) :
    List (String × Json) := Id.run do
  let time := contact64 a b c d r
  let decision := if (decodeChord a b c d r).isNone || !(time.isFinite) || r < 0 then "undefined"
    else if time.toBits == (Float.ofBits 0xbff0000000000000).toBits then "miss" else "hit"
  let chord := [("sweepTimeBits", toJson time.toBits.toNat), ("sweepDecision", toJson decision)]
  let mut fields := [("status", Json.str "unresolved")]
  if let some s := scene then
    match classify s cells with
    | .window l h => fields := [("status", "first contact in window"),
        ("loSeconds", toJson (toString l)), ("hiSeconds", toJson (toString h))]
    | .clear =>
      fields := [("status", "certified clear")]
      if let some margin := s.margin then fields := ("marginMetres", toJson (toString margin)) :: fields
    | .unresolved => pure ()
  return chord ++ fields

private def contactRequest (j : Json) : Except String Json := do
  let (a,b,p) ← flightBody (← j.getObjVal? "first")
  let (c,d,q) ← flightBody (← j.getObjVal? "second")
  let r ← binary64 (← j.getObjVal? "radius")
  let duration ← metres (← j.getObjVal? "duration")
  let cells ← match j.getObjVal? "cells" with
    | .ok value => value.getNat?
    | .error _ => pure 1000000
  if cells > 1000000 then throw "cells must be at most 1000000"
  let scene : Option Scene := do return ⟨← p, ← q, ← finiteValue r, duration⟩
  return Json.mkObj (contactReport a b c d r scene cells)

private def inspectRequest (text : String) : Except String Json := do
  let j ← Json.parse text
  match j.getObjVal? "mode" with
  | .error _ => clearanceRequest j
  | .ok value =>
    if (← value.getStr?) = "firstContact" then contactRequest j
    else throw "unknown collision checker mode"

open TurningFlight in
def main (args : List String) : IO UInt32 := do
  if args == ["--contact-demo"] then
    for (name, y, scene) in [("turn", (1:Float), ContactExamples.turning 1),
        ("clear control", -4, ContactExamples.turning (-4))] do
      IO.println (Json.mkObj (("case", toJson name) ::
        contactReport origin finish (obstacle y) (obstacle y) 0.5 (some scene) 1000000)).compress
    return 0
  if args == ["--demo"] then
    for (name, y) in [("turn", 1), ("clear control", -4)] do
      let fields := report origin finish (obstacle y) (obstacle y) 0.5 1 movingBudget fixedBudget
      IO.println (Json.mkObj ([("case", toJson name), ("cacheBits", toJson cached.toBits.toNat),
        ("endBits", toJson (List.ofFn (fun i : Fin 3 => (finish i).toBits.toNat)))] ++ fields)).compress
    return 0
  let inputs ← match args with
    | ["--file", path] => pure [← IO.FS.readFile path]
    | [] => IO.eprintln "Use --demo, --contact-demo, --file request.json, or one JSON argument per sweep."; return 1
    | inputs => pure inputs
  for input in inputs do
    match inspectRequest input with
    | .ok result => IO.println result.compress
    | .error reason => IO.eprintln s!"Collision clearance input: {reason}"; return 1
  return 0

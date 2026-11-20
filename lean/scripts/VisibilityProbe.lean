-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Space.Distribution
import ExactBallparkProofs.Space.SensorScan

open ExactBallpark.Space.Distribution

/-- Emit the inputs alongside the results so the Python replay exercises the same snapshot. -/
private def snapshot (id : Int) (code : Nat) : Option Bubble :=
  match code with
  | 0 => none
  | 1 => some (fun _ => none)
  | n+2 => some (fun key => if key = id then some ((n : Int) - 1) else none)

def main : IO Unit := do
  let mut member := false
  for x in [(12 : Nat), 12, 13, 12, 1073741824, 12] do
    let d := Float.ofNat x
    let nearby : List ExactBallpark.Space.SensorScan.Candidate :=
      if x = 1073741824 then [] else [⟨4, false, 1, d*d⟩]
    let (current, events) := ExactBallpark.Space.SensorScan.scan 3 4 10 1 member nearby
    let event := match events with
      | [] => "quiet"
      | [e] => toString e.toNat
      | _ => "multiple"
    IO.println s!"sensor {x} {current.toNat} {event}"
    member := current
  for id in [(-1073741825 : Int), -1073741824, 17] do
    for flags in List.range 5 do
      let ball : Option BallFlags := if flags = 0 then none else some ⟨flags == 3 || flags == 4, flags == 2 || flags == 4⟩
      for previous in List.range 5 do
        for current in List.range 5 do
          let (added, removed) := pilotDelta id ball (snapshot id previous) (snapshot id current)
          IO.println s!"pilot {id} {flags} {previous} {current} {added.toNat} {removed.toNat}"
  for id in [(-1 : Int), 0, 17] do
    for flags in List.range 3 do
      let ball : Option EventBall := if flags = 0 then none else some ⟨12, flags == 2⟩
      for known in [false,true] do
        for interactive in [false,true] do
          for character in [none, some (0 : Int), some 37] do
            -- Bubble 11 is available as the RemoveBall fallback even when 12 is invalid.
            let destination := route id ball (some 11) character
              (fun b => b == 11 || (b == 12 && known)) (fun b => b == 11 || (b == 12 && interactive))
            let result := match destination with
              | .drop => "drop"
              | .bubble b => s!"bubble/{b}"
              | .character c => s!"character/{c}"
            IO.println s!"route {id} {flags} {known.toNat} {interactive.toNat} {character.getD (-1)} {result}"

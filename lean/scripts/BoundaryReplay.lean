-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Collision.LimitedPair
import ExactBallparkProofs.Space.RangeAlerts
import ExactBallparkProofs.Time.CleanupWindow

open ExactBallpark
open Space.RangeAlerts

private def alertCode : Alert → Int
  | .range true => 1 | .range false => -1
  | .track true => 2 | .track false => -2

private def numbers {α : Type} [ToString α] (values : List α) :=
  String.intercalate " " (values.map toString)

def main : IO Unit := do
  let pair := Collision.LimitedPair.witness
  IO.println s!"impact {numbers [pair.1,pair.2.1,pair.2.2.1,pair.2.2.2]}"
  let mut flags : Flags := (false,false)
  for s in witnessSamples do
    let (next,events) := poll flags s
    IO.println s!"alerts {next.1.toNat} {next.2.toNat} {numbers (events.map alertCode)}"
    flags := next
  let ids := (List.range 17).map (·+3)
  for left in ([-1,0,1,2] : List Int) do
    IO.println s!"cohort {left} {numbers (Time.CleanupWindow.survivors left ids)}"
  let mut pending := ids
  for _ in List.range 3 do
    pending := Time.CleanupWindow.survivors 1 pending
    IO.println s!"poll {numbers pending}"
  let rows : List Time.RemovalMasks.Row := (List.range 19).map (fun i =>
    ⟨i != 0, i != 2, if i = 16 then -1 else if i % 4 = 3 then 2 else (i%2 : Nat)⟩)
  for (clear,kill) in Time.RemovalMasks.scalar 0 rows do
    IO.println s!"mask {clear.toNat} {kill.toNat}"

-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import Lean
import Mathlib.Data.Finset.Sort
import ExactBallparkProofs.Space.Batches
import ExactBallparkProofs.Time.Snapshots

open Lean ExactBallpark.Space.Batches ExactBallpark.Time.Snapshots

private structure BatchCase where
  interests : List (Int × List Int)
  bubbles : List (Option (List Int) × List Nat)
  chars : List (Option Int × List Nat)
  deriving FromJson

private structure Step where
  op : String
  running : Bool
  flag : Bool
  now : Int
  stamp : Int
  deriving FromJson

private structure Fixtures where
  batches : List BatchCase
  traces : List (List Step)
  deriving FromJson

private def replayBatch (b : BatchCase) : Json :=
  let interests := fun id => ((b.interests.lookup id).getD []).toFinset
  let entries := b.bubbles.map (fun (ids, events) => ⟨(ids.getD []).toFinset, events⟩)
  let chars := b.chars.map (fun (client, events) => ⟨client, events⟩)
  toJson ((send interests entries chars).map (fun (ids, update) =>
    Json.arr #[toJson (ids.sort (· ≤ ·)), toJson update.events, toJson update.waitForBubble, toJson update.count]))

private def replayCache (steps : List Step) : IO Json := do
  let mut cache : Cache := ⟨[], -1⟩
  let mut rows := #[]
  for (s, token) in steps.zipIdx do
    let mut restore : Option Nat := none
    let mut success := true
    let mut ticks := 0
    match s.op with
    | "store" => cache := store cache s.running ⟨token, s.now, s.flag⟩
    | "rebase" => cache := rebase cache s.running s.flag s.now token
    | "post" => cache := (post cache s.flag s.running s.stamp s.now token).1
    | "sync" =>
      let result := synchronize cache.states s.now s.stamp
      restore := (result.bind Prod.fst).map Snapshot.token
      success := result.isSome
      ticks := (result.map Prod.snd).getD 0
    | _ => throw (IO.userError s!"Unknown operation {s.op}")
    let saved := cache.states.map (fun x => Json.arr #[toJson x.token, toJson x.stamp, toJson x.midTick])
    rows := rows.push (.arr #[toJson saved, toJson cache.lastStamp, toJson restore, toJson success, toJson ticks])
  return .arr rows

def main (args : List String) : IO Unit := do
  let [path] := args | throw (IO.userError "Expected a fixture file")
  let parsed ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
  let fixtures : Fixtures ← IO.ofExcept (fromJson? parsed)
  let traces ← fixtures.traces.mapM replayCache
  IO.println (Json.arr #[toJson (fixtures.batches.map replayBatch), toJson traces]).compress

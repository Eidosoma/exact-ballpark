-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
import ExactBallparkProofs.Movement.Formation

open ExactBallpark.Formation ExactBallpark.Control

private def unpack (word : Nat) : Mask := (Finset.range 16).filter (fun i => word.testBit i = true)
private def pack (mask : Mask) : Nat := ((List.range 16).map (fun i => if i ∈ mask then 2^i else 0)).sum
private def index (result : Option Nat) : Int := (result.map Int.ofNat).getD (-1)

private def allocation : IO Unit := do
  for n in [0, 1, 2, 7, 15, 16, 17] do
    let masks := [0, 1, 21845, 43690, 65535] ++ (List.range 16).map (fun i => 65535-2^i)
    for word in masks do
      let (i, bits) := reserve n (unpack word)
      IO.println s!"reserve {n} {word} {index i} {pack bits}"
    for raw in ([-1, 0, 1, 6, 7, 14, 15, 16, 17] : List Int) do
      let slot := Fin.ofNat (2^64) (raw % (2^64)).toNat
      let result := release n slot (unpack 65535)
      IO.println s!"free {n} {raw} {index (result.map pack)}"

private def joins : IO Unit := do
  let initial : Following := ⟨.formation, some 3, 0, 0⟩
  -- Cases 0..4: ordinary, self, missing source, missing destination, no formation.
  for scenario in [0, 1, 2, 3, 4] do
    let src := if scenario = 2 then 9 else 1
    let dst := if scenario = 1 then 1 else if scenario = 3 then 9 else 2
    for n in [0, 2] do
      let size := if scenario ≥ 3 then none else some n
      for raw in ([-1, 0, 1, 2, 128, 255, 256] : List Int) do
        let slot := (raw + 128) % 256 - 128
        for (same, cloak) in [(true, false), (false, false), (true, true), (false, true)] do
          let how := action src dst (scenario != 2) same cloak size slot
          let ship := follow how initial dst slot
          let mode := if ship.mode == .stop then "stop" else "formation"
          IO.println s!"join {scenario} {n} {raw} {same.toNat} {cloak.toNat} {mode} {ship.target.getD 0} {ship.stamp}"

private def resizing : IO Unit := do
  -- Two ships can claim slot 3, a third holds slot 0, and a plain follower
  -- has no formation slot. The word can also include reservations with no ship.
  let fs : List Follower := [⟨2,.formation,3⟩, ⟨3,.formation,0⟩,
                              ⟨4,.formation,3⟩, ⟨5,.follow,0⟩]
  for master in [false, true] do
    for n in [0, 1, 3, 4, 5] do
      for word in [0, 7, 8, 15, 65535] do
        let stops := stoppedOnResize 4 n fs
        let stoppedIds := pack ((stops.map (fun f => f.id.toNat)).toFinset)
        let bits := resizeMask master 4 n fs (unpack word)
        let remaining := fs.filter (fun f => !stops.contains f)
        let back := resizeMask master n 4 remaining bits
        let (allocated, final) := reserve 4 back
        IO.println s!"resize {master.toNat} {n} {word} {stoppedIds} {pack bits} {pack back} {index allocated} {pack final}"

def main : IO Unit := do
  allocation
  joins
  resizing

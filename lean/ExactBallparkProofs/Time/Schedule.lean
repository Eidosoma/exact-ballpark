-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Control
public import Init.Data.Float
public import Mathlib.Data.List.Sort

@[expose] public section
namespace ExactBallpark.Time
open ExactBallpark.Control

/-- ref.Ballpark multiplies by the rounded binary64 literal 0.001. -/
theorem default_tick_seconds :
    ((Float.ofBits 0x408f400000000000) * Float.ofBits 0x3f50624dd2f1a9fc).toBits =
      0x3ff0000000000000 := by decide

inductive Phase where | thrust | responses | calculate | commit | proximity | advance
  deriving DecidableEq, Repr

def tickSchedule (free : List ℤ) (live : ℤ → Bool) (iterative : Bool) : List (Phase × Option ℤ) :=
  let order := (free.mergeSort).filter live
  let visit := fun phase => order.map (fun id => (phase, some id))
  visit .thrust ++ (if iterative then [(.responses, none)] else []) ++
    visit .calculate ++ visit .commit ++ [(.proximity, none), (.advance, none)]

def idsAt (phase : Phase) (trace : List (Phase × Option ℤ)) :=
  trace.filterMap (fun (p, id) => if p = phase then id else none)

theorem phase_ids (free : List ℤ) (live : ℤ → Bool) (iterative : Bool) :
    let trace := tickSchedule free live iterative
    let ids := free.mergeSort.filter live
    idsAt .thrust trace = ids ∧ idsAt .calculate trace = ids ∧ idsAt .commit trace = ids := by
  cases iterative <;> simp [idsAt, tickSchedule, List.filterMap_map]

theorem phase_ordered (free : List ℤ) (live : ℤ → Bool) :
    (free.mergeSort.filter live).Pairwise (· ≤ ·) :=
  (List.pairwise_mergeSort' (· ≤ ·) free).filter _

def phaseNumber : Phase → ℕ
  | .thrust => 0 | .responses => 1 | .calculate => 2 | .commit => 3 | .proximity => 4 | .advance => 5

/-- Across all IDs a calculate call precedes every commit, even when some phases are empty. -/
theorem schedule_monotone (free : List ℤ) (live : ℤ → Bool) (iterative : Bool) :
    (tickSchedule free live iterative).Pairwise (fun a b => phaseNumber a.1 ≤ phaseNumber b.1) := by
  let rank := fun a : Phase × Option ℤ => phaseNumber a.1
  change (tickSchedule free live iterative).Pairwise (fun a b => rank a ≤ rank b)
  apply (List.pairwise_map (f := rank)).mp
  cases iterative <;> simp [tickSchedule, rank, List.map_append, List.map_map,
    Function.comp_def, phaseNumber, List.pairwise_append, List.pairwise_cons] <;> grind

def deadBubble (master : Bool) (mode : Mode) (interactives keepAlives : ℕ) :=
  if !master then false else if mode == .warp then false else
    if interactives != 0 then false else keepAlives == 0

theorem dead_iff (master : Bool) (mode : Mode) (i k : ℕ) :
    deadBubble master mode i k ↔ master = true ∧ mode ≠ .warp ∧ i = 0 ∧ k = 0 := by
  cases master <;> simp [deadBubble]

def remoteFollowers (free : List ℤ) (mode : ℤ → Mode) (bubble : ℤ → ℤ) (targetBubble : ℤ → Option ℤ) :=
  free.mergeSort.filter (fun i => hasFollow (mode i) && targetBubble i != some (bubble i))

theorem remote_iff (free : List ℤ) (mode : ℤ → Mode) (bubble : ℤ → ℤ) (dst : ℤ → Option ℤ) (i : ℤ) :
    i ∈ remoteFollowers free mode bubble dst ↔ i ∈ free ∧ hasFollow (mode i) ∧ dst i ≠ some (bubble i) := by
  simp [remoteFollowers]

/-- For an actual moribund entry outside Evolve. The counter counts only early deletions. -/
def due (stamp now : ℤ) (early : ℕ) : Bool × ℕ :=
  if stamp - now < 0 then (true, early) else
  if stamp - now - 2 < 0 ∧ early < 7 then (true, early+1) else (false, early)

theorem cleanup_budget (stamp now : ℤ) (early : ℕ) (h : early ≤ 7) :
    (due stamp now early).2 ≤ 7 ∧
    ((due stamp now early).1 ↔ stamp < now ∨ (stamp < now+2 ∧ early < 7)) ∧
    (stamp < now → (due stamp now early).2 = early) := by
  unfold due; split_ifs <;> simp_all <;> omega

end ExactBallpark.Time

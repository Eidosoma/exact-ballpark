-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.ScalarTrace

@[expose] public section
namespace ExactBallpark.Movement.WorldMask
open Scalar

/-- The per-row movement state and clock. The scalar host launch counter is shared. -/
structure Row where
  ball : Ball
  clock : Nat

def single (cfg : Config) (r : Row) : Row := ⟨localTick cfg r.ball,r.clock+1⟩

/-- The unmasked GOTO/STOP tick has no reads from another row. masked_tick
saves the old tensors, runs the full tick, and selects by the row's world. -/
def masked {ι ω : Type} (cfg : Config) (world : ι → ω) (rows : ι → Row)
    (due : ω → Bool) (i : ι) : Row :=
  let proposed := single cfg (rows i)
  if due (world i) then proposed else rows i

/-- Every row replays exactly its own due rounds. Equality of the stored Ball
covers the encodings of its three positions and velocities, including signed zeros. -/
theorem masked_world_trace {ι ω : Type} (cfg : Config) (world : ι → ω) (rows : ι → Row)
    (rounds : List (ω → Bool)) (i : ι) :
    rounds.foldl (masked cfg world) rows i =
      (single cfg)^[(rounds.filter (fun due => due (world i))).length] (rows i) := by
  induction rounds generalizing rows with
  | nil => rfl
  | cons due rest ih =>
    simp only [List.foldl_cons,ih,List.filter_cons]
    cases h : due (world i) <;> simp [h,masked,Function.iterate_succ_apply]

/-- VecServer.on_tick runs the non-first worlds with count greater than the
zero-based round number, up to the largest count in the batch. -/
def pollRounds {ω : Type} (counts : ω → Nat) (total : Nat) : List (ω → Bool) :=
  (List.range total).map (fun k w => decide (k < counts w))

private theorem selected_rounds {ω : Type} (counts : ω → Nat) (w : ω) (total : Nat) :
    ((pollRounds counts total).filter (fun due => due w)).length = min total (counts w) := by
  induction total with
  | zero => simp [pollRounds]
  | succ n ih =>
    rw [pollRounds, List.range_succ, List.map_append, List.filter_append, List.length_append]
    change ((pollRounds counts n).filter _).length + _ = _
    rw [ih]
    by_cases h : n < counts w <;> simp [h] <;> omega

theorem poll_replays_single {ι ω : Type} (cfg : Config) (world : ι → ω) (rows : ι → Row)
    (counts : ω → Nat) (total : Nat) (i : ι) (covers : counts (world i) ≤ total) :
    (pollRounds counts total).foldl (masked cfg world) rows i =
      (single cfg)^[counts (world i)] (rows i) := by
  rw [masked_world_trace, selected_rounds, Nat.min_eq_right covers]

end ExactBallpark.Movement.WorldMask

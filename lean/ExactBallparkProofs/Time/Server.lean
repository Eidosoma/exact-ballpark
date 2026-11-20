-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Time.ServerReference
public import Mathlib.Data.Set.Disjoint

@[expose] public section
namespace ExactBallpark.Time

theorem removal_calls (g : Bool) :
    (Reference.removeBall g).getLast? = some "self._park._parent_RemoveBall" ∧
    ("self._notify_remove_global_ball" ∈ Reference.removeBall g ↔ g = true) := by
  cases g <;> decide

theorem pretick_calls : Reference.preTick =
    ["self._finalize_tick", "self._increment_timestamp", "self._flush_history",
     "self._pre_update_bubbles", "self._update_bubbles", "self._post_update_bubbles"] := rfl

/-- Union of interests across the bubble's interactive balls before the single/dual split. -/
def recipients (interactive : Set Int) (interests : Int → Set Int) : Set Int :=
  {c | ∃ b ∈ interactive, c ∈ interests b}

theorem recipient_split (balls : Set Int) (interests : Int → Set Int) (history : Set Int) :
    let clients := recipients balls interests
    (clients \ history) ∪ (clients ∩ history) = clients ∧
    Disjoint (clients \ history) (clients ∩ history) := by
  dsimp
  constructor
  · ext c
    simp only [Set.mem_union, Set.mem_sdiff, Set.mem_inter_iff]
    by_cases h : c ∈ history <;> simp [h]
  · exact Set.disjoint_left.mpr fun _ h₁ h₂ => h₁.2 h₂.2

end ExactBallpark.Time

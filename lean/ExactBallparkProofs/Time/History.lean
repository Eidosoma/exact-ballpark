-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Mathlib.Data.Int.Basic
public import Mathlib.Data.List.Basic
public import Mathlib.Tactic.SplitIfs

@[expose] public section
namespace ExactBallpark.Time

/-- The payload includes ball id, stamp and full event, including any local args. -/
def enqueue {α : Type} [DecidableEq α] (history : List α) (action : α) :=
  if history.getLast? = some action then history else history ++ [action]

theorem last_enqueued {α : Type} [DecidableEq α] (h : List α) (a : α) :
    (enqueue h a).getLast? = some a := by unfold enqueue; split <;> simp_all

theorem duplicate_idempotent {α : Type} [DecidableEq α] (h : List α) (a : α) :
    enqueue (enqueue h a) a = enqueue h a := by
  change (if (enqueue h a).getLast? = some a then _ else _) = _
  rw [ite_eq_left (last_enqueued h a)]

theorem distinct_actions_kept {α : Type} [DecidableEq α] (h : List α) (a b : α) (hne : a ≠ b) :
    enqueue (enqueue h a) b = enqueue h a ++ [b] := by
  change (if (enqueue h a).getLast? = some b then _ else _) = _
  rw [last_enqueued]; simp [hne]

/-- Project the Python snapshot loop onto stamps. Only truthy, usable streams are
represented. Fold order matters when several snapshots share a stamp. -/
def rewindBase (desired : ℤ) (snapshots : List ℤ) : Option ℤ :=
  snapshots.foldl (fun prev stamp => if stamp ≤ desired then some stamp else prev) none

private theorem base_le (desired : ℤ) (ss : List ℤ) (acc : Option ℤ)
    (hacc : ∀ x ∈ acc, x ≤ desired) :
    ∀ x ∈ ss.foldl (fun prev stamp => if stamp ≤ desired then some stamp else prev) acc,
      x ≤ desired := by
  induction ss generalizing acc with
  | nil => exact hacc
  | cons x xs ih =>
    apply ih
    dsimp only
    split
    next h => simpa using h
    next => exact hacc

def sync (now desired : ℤ) (ss : List ℤ) : Option (ℤ × ℕ) := do
  let base ← if desired < now then rewindBase desired ss else some now
  return (base, (desired - base).toNat)

theorem sync_reaches_stamp (now desired : ℤ) (ss : List ℤ) (base : ℤ) (ticks : ℕ)
    (success : sync now desired ss = some (base, ticks)) : base + ticks = desired := by
  unfold sync at success
  by_cases behind : desired < now
  · rw [ite_eq_left behind] at success
    cases h : rewindBase desired ss with
    | none => simp [h] at success
    | some t =>
      have bound := base_le desired ss none (by simp) t (show t ∈ rewindBase desired ss by simp [h])
      simp [h] at success
      omega
  · simp [behind] at success
    omega

def waitForUpdate (waitingBubble : Bool) (stamp now : ℤ) :=
  waitingBubble || (decide (stamp > now) && decide (stamp-now < 3))

theorem wait_iff (w : Bool) (stamp now : ℤ) :
    waitForUpdate w stamp now = true ↔ w = true ∨ stamp = now+1 ∨ stamp = now+2 := by
  cases w <;> simp [waitForUpdate]; omega

end ExactBallpark.Time

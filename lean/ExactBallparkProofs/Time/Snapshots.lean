-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Time.History

@[expose] public section
namespace ExactBallpark.Time.Snapshots

/-- A token identifies the stored stream, even when another snapshot shares its stamp. -/
structure Snapshot where
  token : Nat
  stamp : Int
  midTick : Bool
  deriving DecidableEq, Repr

structure Cache where
  states : List Snapshot
  lastStamp : Int
  deriving DecidableEq, Repr

def store (c : Cache) (running : Bool) (s : Snapshot) : Cache :=
  if !running then c else
    let appended := c.states ++ [s]
    -- The middle Python slice [3:3] is empty for every length.
    let states := if appended.length > 10 then appended.take 1 ++ appended.drop (appended.length - 5)
      else appended
    ⟨states, s.stamp⟩

/-- At capacity the next write preserves the original base, the four most recent
old streams, and the new stream, in that order. -/
theorem eleventh_store (base s : Snapshot) (old : List Snapshot) (h : old.length = 9) (last : Int) :
    store ⟨base :: old, last⟩ true s = ⟨base :: (old.drop 5 ++ [s]), s.stamp⟩ := by
  simp [store, h, List.drop_append]

private theorem store_bound (c : Cache) (running : Bool) (s : Snapshot) (h : c.states.length ≤ 10) :
    (store c running s).states.length ≤ 10 := by
  cases running <;> simp [store, h]
  split <;> simp <;> omega

theorem stores_bounded (c : Cache) (writes : List (Bool × Snapshot)) (h : c.states.length ≤ 10) :
    (writes.foldl (fun cache w => store cache w.1 w.2) c).states.length ≤ 10 := by
  induction writes generalizing c with
  | nil => exact h
  | cons w ws ih => exact ih _ (store_bound _ _ _ h)

/-- A rebase may reuse only the final old stream if it was made mid-tick one tick ago. -/
def rebase (c : Cache) (running newBase : Bool) (now : Int) (token : Nat) : Cache :=
  let lastMid := c.states.getLast?.filter (fun s => s.midTick && s.stamp == now-1)
  let cleared : Cache := ⟨[], c.lastStamp⟩
  if newBase then store {cleared with states := lastMid.toList} running ⟨token, now, false⟩
  else cleared

theorem rebase_contents (c : Cache) (running newBase : Bool) (now : Int) (token : Nat) :
    rebase c running newBase now token =
      let retained := match c.states.getLast? with
        | some s => if s.midTick ∧ s.stamp = now-1 then [s] else []
        | none => []
      ⟨if newBase then retained ++ (if running then [⟨token, now, false⟩] else []) else [],
       if newBase && running then now else c.lastStamp⟩ := by
  cases last : c.states.getLast? with
  | none => cases newBase <;> cases running <;> simp [rebase, store, last]
  | some s =>
    cases newBase <;> cases running <;> by_cases keep : s.midTick ∧ s.stamp = now-1 <;>
      simp [rebase, store, last, Option.filter, keep]

def post (c : Cache) (needsRebase running : Bool) (stamp now : Int) (token : Nat) : Cache × Bool :=
  if needsRebase then (rebase c running true now token, false) else
  if stamp > c.lastStamp+10 then (store c running ⟨token,now,false⟩, false) else (c, false)

/-- When there is no rebase, the first periodic save is eleven stamp units after the last one.
The rebase flag takes precedence even at the ten-unit boundary. -/
theorem post_deadline (c : Cache) (running : Bool) (now : Int) (token : Nat) :
    (∀ stamp ≤ c.lastStamp+10, post c false running stamp now token = (c,false)) ∧
    post c false running (c.lastStamp+11) now token = (store c running ⟨token,now,false⟩,false) ∧
    (∀ stamp, post c true running stamp now token = (rebase c running true now token,false)) := by
  simp [post]
  omega

def pick (desired : Int) (previous : Option Snapshot) (s : Snapshot) :=
  if s.stamp ≤ desired then some s else previous

/-- Optional stream to restore, followed by the exact number of evolve calls.
All stored tokens stand for readable, truthy streams. -/
def synchronize (ss : List Snapshot) (now desired : Int) : Option (Option Snapshot × Nat) :=
  if desired < now then do
    let s ← ss.foldl (pick desired) none
    return (some s, (desired-s.stamp).toNat)
  else some (none, (desired-now).toNat)

private theorem skip_future (desired : Int) (later : List Snapshot) (acc : Option Snapshot)
    (h : ∀ s ∈ later, desired < s.stamp) : later.foldl (pick desired) acc = acc := by
  induction later generalizing acc with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldl_cons, pick, ite_eq_right (Int.not_le.mpr (h x (by simp)))]
    exact ih acc (fun s hs => h s (by simp [hs]))

/-- Rewind restores the last qualifying stream in list order, also for equal stamps. -/
theorem rewind_stream (before after : List Snapshot) (s : Snapshot) (now desired : Int)
    (rewinding : desired < now) (qualifies : s.stamp ≤ desired) (later : ∀ t ∈ after, desired < t.stamp) :
    synchronize (before ++ s :: after) now desired = some (some s,(desired-s.stamp).toNat) ∧
    sync now desired ((before ++ s :: after).map Snapshot.stamp) = some (s.stamp,(desired-s.stamp).toNat) := by
  have finish (acc : Option Int) :
      (after.map Snapshot.stamp).foldl (fun p t => if t ≤ desired then some t else p) acc = acc := by
    induction after generalizing acc with
    | nil => rfl
    | cons x xs ih =>
      simp [Int.not_le.mpr (later x (by simp)), ih (fun t ht => later t (by simp [ht])) acc]
  simp [synchronize, rewinding, List.foldl_append, pick, qualifies, skip_future _ _ _ later, sync, rewindBase, finish]

private theorem none_iff (ss : List Snapshot) (desired : Int) (acc : Option Snapshot) :
    ss.foldl (pick desired) acc = none ↔ acc = none ∧ ∀ s ∈ ss, desired < s.stamp := by
  induction ss generalizing acc with
  | nil => simp
  | cons s ss ih => by_cases h : s.stamp ≤ desired <;> simp [pick, ih, h] <;> omega

theorem sync_failure (ss : List Snapshot) (now desired : Int) :
    synchronize ss now desired = none ↔ desired < now ∧ ∀ s ∈ ss, desired < s.stamp := by
  by_cases h : desired < now
  · have empty := none_iff ss desired none
    cases eq : ss.foldl (pick desired) none with
    | none => simpa [synchronize, h, eq] using (empty.mp eq).2
    | some t =>
      have found : ¬ ∀ s ∈ ss, desired < s.stamp := by
        intro late
        have impossible := eq.symm.trans (empty.mpr ⟨rfl, late⟩)
        contradiction
      simp [synchronize, h, eq, found]
  · simp [synchronize, h]

end ExactBallpark.Time.Snapshots

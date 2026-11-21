-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Time.History
public import ExactBallparkProofs.Time.Schedule
public import ExactBallparkProofs.Space.Queries

@[expose] public section
namespace ExactBallpark.Time.ServerLaws
open ExactBallpark.Control

abbrev Event := Int × Int × String
def stopEvent (stamp id : Int) : Event := (id, stamp, "Stop")

private theorem append_stops (ids : List Int) (stamp : Int) (old : List Event)
    (unique : ids.Nodup) (fresh : ∀ i ∈ ids, old.getLast? ≠ some (stopEvent stamp i)) :
    ids.foldl (fun h i => enqueue h (stopEvent stamp i)) old =
    old ++ ids.map (stopEvent stamp) := by
  induction ids generalizing old with
  | nil => simp
  | cons i ids ih =>
    obtain ⟨hi,rest⟩ := List.nodup_cons.mp unique
    simp only [List.foldl_cons]
    rw [ih _ rest (by
      intro j hj
      rw [last_enqueued]
      intro eq
      have : i=j := congrArg (fun p : Event => p.1) (Option.some.inj eq)
      exact hi (this ▸ hj))]
    simp [enqueue, fresh i (by simp)]

/-- At the cleanup hook each eligible ID queues a Stop at the current system
stamp. Only the first one can duplicate the action already at the tail. -/
theorem remote_pending_stops (free : List Int) (mode : Int → Mode)
    (bubble : Int → Int) (dst : Int → Option Int) (old : List Event) (stamp : Int)
    (unique : free.Nodup) :
    let ids := remoteFollowers free mode bubble dst
    (∀ i, i ∈ ids ↔ i ∈ free ∧ hasFollow (mode i) ∧ dst i ≠ some (bubble i)) ∧
    (∀ i ∈ ids, ∀ state : Following, state.mode = mode i → stop state = ⟨.stop,none,0,0⟩) ∧
    ids.foldl (fun h i => enqueue h (stopEvent stamp i)) old =
      match ids with
      | [] => old
      | i :: rest => enqueue old (stopEvent stamp i) ++ rest.map (stopEvent stamp) := by
  refine ⟨remote_iff free mode bubble dst, ?_, ?_⟩
  · intro i selected state modeEq
    have following := ((remote_iff free mode bubble dst i).1 selected).2.1
    simp [stop,modeEq,following]
  have nd : (remoteFollowers free mode bubble dst).Nodup :=
    ((List.mergeSort_perm free _).nodup_iff.mpr unique).filter _
  cases e : remoteFollowers free mode bubble dst with
  | nil => rfl
  | cons i rest =>
    rw [e] at nd
    obtain ⟨hi,tail⟩ := List.nodup_cons.mp nd
    apply append_stops rest stamp _ tail
    intro j hj
    rw [last_enqueued]
    intro eq
    have : i=j := congrArg (fun p : Event => p.1) (Option.some.inj eq)
    exact hi (this ▸ hj)

structure Tags where
  harmonic : Int
  corp : Int
  alliance : Int
  deriving DecidableEq
inductive FieldAction where
  | massive (on : Bool)
  | harmonic (tags : Tags)
  deriving DecidableEq
structure Field where
  massive : Bool
  mode : Mode
  tags : Tags

def enabled (v : Tags) := decide (v.harmonic ≠ -1 ∨ v.corp ≠ -1 ∨ v.alliance ≠ -1)
def configure (v : Tags) : List FieldAction := [.massive (enabled v), .harmonic v]
def applyField (s : Field) : FieldAction → Field
  | .massive m => {s with massive := m}
  | .harmonic t => {s with tags := t, mode := .field}

def kept (ball : Tags) (field : Field) :=
  field.massive && !(field.mode == .field && ExactBallpark.Space.fieldPass
    ball.harmonic ball.corp ball.alliance field.tags.harmonic field.tags.corp field.tags.alliance)

/-- Both queued parent calls have run for a present force field. The candidate
query excludes cloaked/non-massive objects, enables harmonic checks and has
already admitted this uncloaked field through the other spatial filters. -/
theorem configured_field_filter (s : Field) (tags ship : Tags) :
    let f := (configure tags).foldl applyField s
    f.mode = .field ∧ f.tags = tags ∧ f.massive = enabled tags ∧
    (kept ship f ↔ enabled tags ∧ ship.harmonic ≠ -2 ∧
      (tags.harmonic = -1 ∨ tags.harmonic ≠ ship.harmonic) ∧
      (tags.corp = -1 ∨ tags.corp ≠ ship.corp) ∧
      (tags.alliance = -1 ∨ tags.alliance ≠ ship.alliance)) := by
  simp only [configure,List.foldl_cons,List.foldl_nil,applyField,kept,beq_self_eq_true,
    Bool.true_and, true_and, Bool.and_eq_true, Bool.not_eq_true']
  rw [Bool.eq_false_iff]
  have pass := ExactBallpark.Space.field_iff ship.harmonic ship.corp ship.alliance tags.harmonic tags.corp tags.alliance
  grind

end ExactBallpark.Time.ServerLaws

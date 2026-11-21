-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Time.ServerLaws
public import ExactBallparkProofs.Time.Server

@[expose] public section
namespace ExactBallpark.Time.FollowerTick
open ExactBallpark.Control
open ServerLaws (Event stopEvent)

/-- An ordinary master follower and its current leader's follower set. -/
structure Ship where
  flight : Following
  pointer : Option Int
  followRange : ℚ
  followersOfLeader : Finset Int

def parentStop (id : Int) (s : Ship) : Ship :=
  if hasFollow s.flight.mode then
    ⟨⟨.stop,none,0,0⟩,none,0,s.followersOfLeader.erase id⟩
  else {s with flight := {s.flight with mode := .stop}}

def applyEvent (id : Int) (other : Ship → Event → Ship) (s : Ship) (e : Event) :=
  if e.2.2 = "Stop" then (if e.1 = id then parentStop id s else s) else other s e

private theorem stop_twice (id : Int) (s : Ship) : parentStop id (parentStop id s) = parentStop id s := by
  unfold parentStop
  split <;> simp [hasFollow]

private theorem queue_one (id follower stamp : Int) (other : Ship → Event → Ship)
    (old : List Event) (s : Ship) :
    (enqueue old (stopEvent stamp follower)).foldl (applyEvent id other) s =
    applyEvent id other (old.foldl (applyEvent id other) s) (stopEvent stamp follower) := by
  unfold enqueue
  split
  next dup =>
    obtain ⟨pre,rfl⟩ := List.getLast?_eq_some_iff.mp dup
    simp only [List.foldl_append,List.foldl_cons,List.foldl_nil,applyEvent,stopEvent,↓reduceIte]
    split <;> simp [stop_twice]
  next => simp

private theorem apply_cleanup (id stamp : Int) (other : Ship → Event → Ship)
    (ids : List Int) (old : List Event) (s : Ship) :
    (ids.foldl (fun q i => enqueue q (stopEvent stamp i)) old).foldl (applyEvent id other) s =
    let after := old.foldl (applyEvent id other) s
    if id ∈ ids then parentStop id after else after := by
  induction ids generalizing old with
  | nil => simp
  | cons i rest ih =>
    rw [List.foldl_cons,ih,queue_one]
    by_cases same : i=id <;> by_cases again : id ∈ rest <;>
      simp [same,again,applyEvent,stopEvent,stop_twice]
    exact fun h => (same h.symm).elim

structure AtBubbles where
  stampForSystem : Int
  pending : List Event
  flushed : List Event
  ship : Ship

/-- Cleanup uses the previous actions stamp. Incrementing the producer stamp
does not restamp the flushed entries. Non-cloak parent calls run before bubbles. -/
def preTick (id now stamp : Int) (free : List Int) (mode : Int → Mode) (bubble : Int → Int)
    (target : Int → Option Int) (old : List Event) (s : Ship) (other : Ship → Event → Ship) : AtBubbles :=
  let ids := remoteFollowers free mode bubble target
  let batch := ids.foldl (fun q i => enqueue q (stopEvent stamp i)) old
  ⟨now+1,[],batch,batch.foldl (applyEvent id other) s⟩

/-- An eligible ship gets the effect of a final parent Stop, even when its
queued Stop duplicated the old tail. Other queued pre-bubble actions may act
first. A still-following ship releases its leader's membership and all fields. -/
theorem stopped_before_bubbles (id now stamp : Int) (free : List Int) (mode : Int → Mode)
    (bubble : Int → Int) (target : Int → Option Int) (old : List Event) (s : Ship)
    (other : Ship → Event → Ship) (listed : id ∈ free) (following : hasFollow (mode id))
    (remote : target id ≠ some (bubble id)) :
    let after := old.foldl (applyEvent id other) s
    let atBubble := preTick id now stamp free mode bubble target old s other
    atBubble.stampForSystem = now+1 ∧ atBubble.pending = [] ∧ atBubble.ship = parentStop id after ∧
    (hasFollow after.flight.mode →
      atBubble.ship = ⟨⟨.stop,none,0,0⟩,none,0,after.followersOfLeader.erase id⟩) := by
  have mem := (remote_iff free mode bubble target id).mpr ⟨listed,following,remote⟩
  simp only [preTick,apply_cleanup,mem,↓reduceIte,true_and]
  intro h
  simp [parentStop,h]

end ExactBallpark.Time.FollowerTick

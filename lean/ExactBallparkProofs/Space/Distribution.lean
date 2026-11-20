-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Mathlib.Data.Set.Disjoint

@[expose] public section
namespace ExactBallpark.Space.Distribution

structure EventBall where
  bubble : Int
  cloaked : Bool

inductive Destination where
  | drop
  | bubble (id : Int)
  | character (id : Int)
  deriving DecidableEq

/-- Default path of Ticker._apply_history after any specialized handler has returned.
removedFrom is the bubble argument for RemoveBall, and none for other actions. -/
def route (id : Int) (ball : Option EventBall) (removedFrom character : Option Int)
    (known interactive : Int → Bool) : Destination :=
  let bubble := (ball.map EventBall.bubble).getD (removedFrom.getD (-1))
  if !known bubble || !interactive bubble || id ≤ 0 || ball.isNone then .drop else
  match ball with
  | none => .drop
  | some b => if b.cloaked then (character.map Destination.character).getD .drop else .bubble bubble

theorem bubble_destination (id dest : Int) (ball : Option EventBall) (removedFrom character : Option Int)
    (known interactive : Int → Bool) :
    route id ball removedFrom character known interactive = .bubble dest ↔
      0 < id ∧ ∃ b, ball = some b ∧ b.bubble = dest ∧ known dest ∧ interactive dest ∧ b.cloaked = false := by
  cases ball with
  | none => simp [route]
  | some b =>
    cases character <;> cases hc : b.cloaked <;> simp [route, hc] <;> grind

theorem character_destination (id dest : Int) (ball : Option EventBall) (removedFrom character : Option Int)
    (known interactive : Int → Bool) :
    route id ball removedFrom character known interactive = .character dest ↔
      0 < id ∧ character = some dest ∧ ∃ b, ball = some b ∧ known b.bubble ∧ interactive b.bubble ∧ b.cloaked := by
  cases ball with
  | none => simp [route]
  | some b =>
    cases character <;> cases hc : b.cloaked <;> simp [route, hc] <;> grind

structure BallFlags where
  globalBall : Bool
  cloaked : Bool

abbrev Bubble := Int → Option Int

/-- Per-ID projection of the two dictionary loops for a pilot changing bubbles.
The option distinguishes an unavailable old bubble from a present, empty one. -/
def pilotDelta (id : Int) (ball : Option BallFlags) (old new : Option Bubble) : Bool × Bool :=
  let previous := old.bind (· id)
  let current := new.bind (· id)
  let hidden := ball.any (fun b => b.globalBall || b.cloaked)
  if id < -1073741824 then (false, false) else
    let (add, removeNew) := match current with
      | none => (false, false)
      | some action =>
        if hidden then (false, false) else
        if 0 ≤ action && previous.isNone then (true, false) else
        (false, action == -1 && (previous.isSome || old.isNone))
    let removeOld := match previous with
      | none => false
      | some action =>
        if action != -1 && (ball.isNone || hidden) then false else
        if action == -1 && ball.any BallFlags.cloaked then false else
        current.isNone && decide (action ≤ 0)
    (add, removeNew || removeOld)

theorem pilot_add_iff (id : Int) (ball : Option BallFlags) (old new : Option Bubble) :
    (pilotDelta id ball old new).1 ↔
      -1073741824 ≤ id ∧ (∀ b ∈ ball, b.globalBall = false ∧ b.cloaked = false) ∧
      (old.bind (· id)) = none ∧ ∃ a ∈ new.bind (· id), 0 ≤ a := by
  cases hp : old.bind (· id) <;> cases hc : new.bind (· id) <;>
    cases ball <;> simp [pilotDelta, hp, hc] <;> grind

theorem pilot_remove_iff (id : Int) (ball : Option BallFlags) (old new : Option Bubble) :
    (pilotDelta id ball old new).2 ↔
      -1073741824 ≤ id ∧ (
        ((new.bind (· id)) = some (-1) ∧ ((old.bind (· id)).isSome ∨ old.isNone) ∧
          ∀ b ∈ ball, b.globalBall = false ∧ b.cloaked = false) ∨
        ((new.bind (· id)) = none ∧ ∃ a ∈ old.bind (· id), a ≤ 0 ∧
          (if a = -1 then ∀ b ∈ ball, b.cloaked = false
           else ∃ b, ball = some b ∧ b.globalBall = false ∧ b.cloaked = false))) := by
  cases hp : old.bind (· id) <;> cases hc : new.bind (· id) <;>
    cases ball <;> simp [pilotDelta, hp, hc] <;> split_ifs <;> simp_all <;> grind

/-- The same pilot can never receive both an addition and a deletion of an ID
in this transition, including stale records and the missing-old-bubble branch. -/
theorem pilot_disjoint (balls : Int → Option BallFlags) (old new : Option Bubble) :
    Disjoint {id | (pilotDelta id (balls id) old new).1 = true}
             {id | (pilotDelta id (balls id) old new).2 = true} := by
  apply Set.disjoint_left.mpr
  intro id ha hr
  obtain ⟨_, _, prev, a, current, hpos⟩ := (pilot_add_iff ..).mp ha
  obtain ⟨_, del | del⟩ := (pilot_remove_iff ..).mp hr
  · have h := del.1
    simp_all
    omega
  · have h := del.1
    simp_all

end ExactBallpark.Space.Distribution

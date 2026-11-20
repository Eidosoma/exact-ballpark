-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Mathlib.Data.Finset.Union
public import Mathlib.Data.Finset.Lattice.Basic

@[expose] public section
namespace ExactBallpark.Space.Batches

/-- A payload is an ordered list of event tokens. Missing interactive-bubble keys have no balls. -/
structure BubbleEntry where
  interactive : Finset Int
  events : List Nat
structure CharacterEntry where
  client : Option Int
  events : List Nat

structure Update where
  events : List Nat
  waitForBubble : Bool
  count : Nat
  deriving DecidableEq, Repr

abbrev Packet := Finset Int × Update

def audience (interest : Int → Finset Int) (b : BubbleEntry) := b.interactive.biUnion interest

def privateClients (chars : List CharacterEntry) : Finset Int :=
  (chars.filterMap (fun e => if e.events.isEmpty then none else e.client)).toFinset

def narrow (recipients : Finset Int) (payload : List Nat) (count : Nat) : Option Packet :=
  if recipients = ∅ then none else some (recipients, ⟨payload, false, count⟩)

def singleGroup (ids privateIds : Finset Int) (payload : List Nat) :=
  narrow (ids \ privateIds) payload 1

def dualGroup (ids privateIds : Finset Int) (payload : List Nat) :=
  narrow (ids ∩ privateIds) payload 2

def receive (client : Int) (packet : Packet) : Option Update :=
  if client ∈ packet.1 then some packet.2 else none

private theorem narrow_receive (ids : Finset Int) (payload : List Nat) (n : Nat) (c : Int) :
    (narrow ids payload n).bind (receive c) =
      if c ∈ ids then some ⟨payload, false, n⟩ else none := by
  by_cases empty : ids = ∅ <;> simp [narrow, receive, empty]

private theorem private_mem (chars : List CharacterEntry) (c : Int) :
    c ∈ privateClients chars ↔ ∃ e ∈ chars, e.events ≠ [] ∧ e.client = some c := by
  simp only [privateClients, List.mem_toFinset, List.mem_filterMap]
  apply exists_congr
  intro e
  by_cases h : e.events = [] <;> simp [h, List.isEmpty_iff]

theorem single_recipient (interest : Int → Finset Int) (b : BubbleEntry) (chars : List CharacterEntry) (c : Int) :
    ((singleGroup (audience interest b) (privateClients chars) b.events).bind (receive c) =
      some ⟨b.events, false, 1⟩) ↔
      (∃ ball ∈ b.interactive, c ∈ interest ball) ∧
      ¬ ∃ e ∈ chars, e.events ≠ [] ∧ e.client = some c := by
  simp [singleGroup, narrow_receive, audience, private_mem]
  grind

theorem dual_recipient (interest : Int → Finset Int) (b : BubbleEntry) (chars : List CharacterEntry) (c : Int) :
    ((dualGroup (audience interest b) (privateClients chars) b.events).bind (receive c) =
      some ⟨b.events, false, 2⟩) ↔
      (∃ ball ∈ b.interactive, c ∈ interest ball) ∧
      ∃ e ∈ chars, e.events ≠ [] ∧ e.client = some c := by
  simp [dualGroup, narrow_receive, audience, private_mem]
  grind

def waitingClients (interest : Int → Finset Int) (bubbles : List BubbleEntry) :=
  bubbles.foldl (fun ids b => ids ∪ audience interest b) ∅

private theorem waiting_mem (interest : Int → Finset Int) (bs : List BubbleEntry) (c : Int) :
    c ∈ waitingClients interest bs ↔ ∃ b ∈ bs, c ∈ audience interest b := by
  suffices ∀ ids : Finset Int, c ∈ bs.foldl (fun acc b => acc ∪ audience interest b) ids ↔
      c ∈ ids ∨ ∃ b ∈ bs, c ∈ audience interest b by simpa [waitingClients] using this ∅
  induction bs with
  | nil => simp
  | cons b bs ih => intro ids; simp [ih, or_assoc]

def personal (waiting : Finset Int) (e : CharacterEntry) : Option Packet := do
  let id ← e.client
  if e.events.isEmpty then none else
    let waits := decide (id ∈ waiting)
    some ({id}, ⟨e.events, waits, if waits then 2 else 1⟩)

theorem character_wait (interest : Int → Finset Int) (bs : List BubbleEntry)
    (e : CharacterEntry) (c : Int) (msg : Update) :
    (personal (waitingClients interest bs) e).bind (receive c) = some msg ↔
      e.client = some c ∧ e.events ≠ [] ∧
      let waits := decide (∃ b ∈ bs, c ∈ audience interest b)
      msg = ⟨e.events, waits, if waits then 2 else 1⟩ := by
  cases h : e.client with
  | none => simp [personal, h]
  | some id =>
    by_cases empty : e.events = []
    · simp [personal, h, empty]
    · by_cases eq : c = id
      · subst id
        simp [personal, receive, h, empty, List.isEmpty_iff, waiting_mem, eq_comm (a := msg)]
      · simp [personal, receive, h, empty, List.isEmpty_iff, eq, Ne.symm eq]

/-- Flatten the three successful interface calls in send order. -/
def send (interest : Int → Finset Int) (bs : List BubbleEntry) (cs : List CharacterEntry) : List Packet :=
  cs.filterMap (personal (waitingClients interest bs)) ++
    bs.filterMap (fun b => singleGroup (audience interest b) (privateClients cs) b.events) ++
    bs.filterMap (fun b => dualGroup (audience interest b) (privateClients cs) b.events)

theorem client_trace (interest : Int → Finset Int) (bs : List BubbleEntry) (cs : List CharacterEntry) (c : Int) :
    (send interest bs cs).filterMap (receive c) =
      cs.filterMap (fun e => (personal (waitingClients interest bs) e).bind (receive c)) ++
      (bs.filter (fun b => c ∈ audience interest b)).map
        (fun b => ⟨b.events, false, if c ∈ privateClients cs then 2 else 1⟩) := by
  simp only [send, List.filterMap_append, List.filterMap_filterMap, singleGroup, dualGroup, narrow_receive]
  by_cases h : c ∈ privateClients cs <;>
    simp [h] <;> induction bs with
    | nil => rfl
    | cons b bs ih => by_cases hb : c ∈ audience interest b <;> simp [hb, ih]

end ExactBallpark.Space.Batches

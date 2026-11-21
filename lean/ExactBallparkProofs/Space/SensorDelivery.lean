-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Space.SensorScan
public import ExactBallparkProofs.Space.Distribution
public import ExactBallparkProofs.Space.Batches

@[expose] public section
namespace ExactBallpark.Space.SensorDelivery
open SensorScan Batches Distribution

structure Poll where
  elapsed : Float
  member : Bool
  callbacks : List (Nat × Bool)

/-- One-second polls of an owner and target of radius one, with a 10 m range
and a two-second period. The target remains in the collected boxes throughout. -/
def poll (s : Poll) (stamp : Nat) (x : Float) : Poll :=
  let elapsed := s.elapsed + 1
  if elapsed < 2 then {s with elapsed} else
    let (member, events) := scan 3 4 10 1 s.member [⟨4,false,1,x*x⟩]
    ⟨0,member,s.callbacks ++ events.map (stamp, ·)⟩

def crossingTrace : List (Nat × Float) :=
  [(1,12),(2,13),(3,13),(4,12),(5,13),(6,11),(7,12),(8,13)]

def states (s : Poll) : List (Nat × Float) → List Poll
  | [] => [s]
  | (stamp,x) :: rest => s :: states (poll s stamp x) rest

def crossingStates := states ⟨0,false,[]⟩ crossingTrace

set_option maxRecDepth 16384 in
theorem timed_crossing :
    crossingStates.map Poll.member = [false,false,false,false,true,true,true,true,false] ∧
    (crossingStates.getLast?.map Poll.callbacks) = some [(4,true),(8,false)] := by
  decide +kernel

/-- DoProximity is delivered to the ball. 'action' is an optional ordinary action
queued by the application for that ball. Without it the server has no history entry.
The projection considers one visible owner, one bubble history and no private traffic. -/
def packets (action : Option Nat) (owner bubble : Int) (known live : Int → Bool)
    (ships : Finset Int) (interest : Int → Finset Int) (client : Int) : List Update :=
  let histories : List BubbleEntry := match action with
    | none => []
    | some event =>
      match route owner (some ⟨bubble,false⟩) none none known live with
      | .bubble _ => [⟨ships,[event]⟩]
      | _ => []
  (send interest histories []).filterMap (receive client)

theorem callback_recipients (action : Option Nat) (owner bubble : Int) (positive : 0 < owner)
    (known live : Int → Bool) (ships : Finset Int) (interest : Int → Finset Int) (client : Int) :
    packets action owner bubble known live ships interest client =
      if known bubble ∧ live bubble ∧ (∃ s ∈ ships, client ∈ interest s) then
        action.toList.map (fun e => ⟨[e],false,1⟩) else [] := by
  cases action <;> cases hk : known bubble <;> cases hl : live bubble <;>
    simp [packets, route, positive.not_ge, hk,hl,client_trace,audience,privateClients]
  split <;> simp_all

/-- A fixture application queues a radius of one on enter and two on leave.
The next server tick sends the queued action. With the default callback there
are no queued responses, even though the sensor recorded both edges. -/
def crossingHistory (forward : Bool) (known live : Int → Bool) (ships : Finset Int)
    (interest : Int → Finset Int) (client : Int) : List (Nat × Nat) :=
  let events := (crossingStates.getLast?.map Poll.callbacks).getD []
  events.flatMap fun (stamp,entered) =>
    let response := if forward then some (if entered then 1 else 2) else none
    (packets response 3 7 known live ships interest client).flatMap
      (fun update => update.events.map fun radius => (stamp+1,radius))

theorem crossing_delivered (forward : Bool) (known live : Int → Bool) (ships : Finset Int)
    (interest : Int → Finset Int) (client : Int) :
    crossingHistory forward known live ships interest client =
      if forward ∧ known 7 ∧ live 7 ∧ (∃ s ∈ ships, client ∈ interest s) then [(5,1),(9,2)] else [] := by
  by_cases active : known 7 ∧ live 7 ∧ (∃ s ∈ ships, client ∈ interest s) <;>
    cases forward <;> simp [crossingHistory, timed_crossing.2, callback_recipients _ _ _ (by decide : 0 < (3:Int)), active]

end ExactBallpark.Space.SensorDelivery

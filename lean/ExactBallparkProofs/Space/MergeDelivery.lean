-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Space.Batches

@[expose] public section
namespace ExactBallpark.Space.Batches

structure HistoryEntry where
  stamp : Int
  events : List Nat
  waits : Bool
  deriving DecidableEq

/-- Projection of merge_state_into_history for a nonempty packet at one stamp.
Passing an older history entry clears the incoming wait flag. -/
def mergePacket (stamp : Int) (events : List Nat) (waits : Bool) :
    List HistoryEntry → List HistoryEntry
  | [] => [⟨stamp,events,waits⟩]
  | entry::rest =>
    if entry.stamp < stamp then entry :: mergePacket stamp events false rest
    else if entry.stamp = stamp then {entry with events := entry.events ++ events, waits := false} :: rest
    else ⟨stamp,events,waits⟩ :: entry :: rest

private theorem prefix_merge (before after : List HistoryEntry) (stamp : Int) (es : List Nat) (w : Bool)
    (early : ∀ e ∈ before, e.stamp < stamp) :
    mergePacket stamp es w (before ++ after) =
      before ++ mergePacket stamp es (w && before.isEmpty) after := by
  induction before generalizing w with
  | nil => simp
  | cons e bs ih =>
    simp only [List.cons_append, mergePacket, early e (by simp), ite_true, List.isEmpty_cons,
      Bool.and_false]
    simpa using ih false (fun x hx => early x (by simp [hx]))

private theorem fresh_merge (after : List HistoryEntry) (stamp : Int) (es : List Nat) (w : Bool)
    (late : ∀ e ∈ after, stamp < e.stamp) :
    mergePacket stamp es w after = ⟨stamp,es,w⟩ :: after := by
  cases after with
  | nil => rfl
  | cons e es =>
    have h := late e (by simp)
    simp [mergePacket, not_lt.mpr h.le, ne_of_gt h]

/-- A private update followed by a bubble update at one stamp. Earlier and
later entries retain their order, payloads and flags. The send trace supplies
the private flag and payload order. A pre-existing group keeps its earlier events. -/
theorem private_bubble_history (before after : List HistoryEntry) (stamp client : Int)
    (prior : Option (List Nat × Bool)) (privateEvents bubbleEvents : List Nat) (hp : privateEvents ≠ []) (hb : bubbleEvents ≠ [])
    (early : ∀ e ∈ before, e.stamp < stamp) (late : ∀ e ∈ after, stamp < e.stamp) :
    let interest := fun (_ : Int) => ({client} : Finset Int)
    let bs : List BubbleEntry := [⟨{0},bubbleEvents⟩]
    let cs : List CharacterEntry := [⟨some client,privateEvents⟩]
    let packets := (send interest bs cs).filterMap (receive client)
    let initial := before ++ (prior.map (fun p => (⟨stamp,p.1,p.2⟩ : HistoryEntry))).toList ++ after
    let oldEvents := (prior.map Prod.fst).getD []
    packets = [⟨privateEvents,true,2⟩,⟨bubbleEvents,false,2⟩] ∧
    packets.foldl (fun hist p => if p.events = [] then hist else mergePacket stamp p.events p.waitForBubble hist) initial =
      before ++ ⟨stamp,oldEvents++privateEvents++bubbleEvents,false⟩ :: after := by
  dsimp only
  have trace : ((send (fun _ => {client}) [⟨{0},bubbleEvents⟩] [⟨some client,privateEvents⟩]).filterMap
      (receive client)) = [⟨privateEvents,true,2⟩,⟨bubbleEvents,false,2⟩] := by
    rw [client_trace]
    simp [personal,waitingClients,audience,privateClients,receive,hp,List.isEmpty_iff]
  refine ⟨trace, ?_⟩
  rw [trace]
  simp only [List.foldl_cons,List.foldl_nil,hp,hb,ite_false]
  cases prior with
  | none =>
    simp only [Option.map_none,Option.toList_none,List.append_nil,Option.getD_none,List.nil_append]
    rw [prefix_merge _ _ _ _ _ early, fresh_merge _ _ _ _ late, prefix_merge _ _ _ _ _ early]
    simp [mergePacket]
  | some entry =>
    simp only [Option.map_some,Option.toList_some,List.append_assoc,List.cons_append,List.nil_append,
      Option.getD_some]
    rw [prefix_merge _ _ _ _ _ early]
    simp only [mergePacket,lt_self_iff_false,ite_false,ite_true]
    rw [prefix_merge _ _ _ _ _ early]
    simp [mergePacket]

end ExactBallpark.Space.Batches

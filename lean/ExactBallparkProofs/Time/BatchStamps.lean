-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Time.MixedHistory

@[expose] public section
namespace ExactBallpark.Time.BatchStamps
open ExactBallpark.Space.Batches
open MixedHistory (Entry merge)

/-- Group raw tokens by increasing stamp, retaining input order within each
group, as the client's dictionary and sorted timestamp list do. -/
def insertEvent (stampOf : Nat → Int) (event : Nat) : List Entry → List Entry
  | [] => [⟨stampOf event,[event],false⟩]
  | e::rest =>
    if stampOf event < e.stamp then ⟨stampOf event,[event],false⟩ :: e :: rest
    else if stampOf event = e.stamp then {e with events := event::e.events} :: rest
    else e :: insertEvent stampOf event rest

def group (stampOf : Nat → Int) (events : List Nat) := events.foldr (insertEvent stampOf) []

private theorem block (stampOf : Nat → Int) (s : Int) (xs : List Nat) (tail : List Entry)
    (oneStamp : ∀ x ∈ xs, stampOf x = s) (later : ∀ e ∈ tail, s < e.stamp) :
    xs.foldr (insertEvent stampOf) tail =
      (if xs=[] then tail else ⟨s,xs,false⟩::tail) := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    have hx := oneStamp x (by simp)
    rw [List.foldr_cons,ih (fun y hy => oneStamp y (by simp [hy]))]
    cases xs with
    | nil =>
      cases tail with
      | nil => simp [insertEvent,hx]
      | cons a rest => simp [insertEvent,hx,later a (by simp)]
    | cons y ys => simp [insertEvent,hx]

def delivered (client interactiveBall : Int) (stampOf : Nat → Int) (personalEvents bubbleEvents : List Nat) :=
  let interest := fun (_ : Int) => ({client} : Finset Int)
  let packets := send interest [⟨{interactiveBall},bubbleEvents⟩] [⟨some client,personalEvents⟩]
  (packets.filterMap (receive client)).foldl (fun hist p =>
    merge hist (group stampOf p.events) p.waitForBubble) []

/-- Empty starting client queue, one recipient of both packets, and ordinary
admitted events. Flushing once per uniform stamp releases the entire group.
Holding the private history across a second stamp leaves that group blocked
when only the earlier stamp is present in the bubble history. -/
theorem delayed_private_flush (client interactiveBall s t : Int) (stampOf : Nat → Int) (early late bubble : List Nat)
    (ordered : s < t) (hp : early ≠ []) (hb : bubble ≠ [])
    (sp : ∀ x ∈ early, stampOf x = s) (sl : ∀ x ∈ late, stampOf x = t)
    (sb : ∀ x ∈ bubble, stampOf x = s) :
    delivered client interactiveBall stampOf (early++late) bubble =
      ⟨s,early++bubble,false⟩ :: (if late=[] then [] else [⟨t,late,true⟩]) := by
  have hp' : early++late ≠ [] := by simp [hp]
  have trace : ((send (fun _ => {client}) [⟨{interactiveBall},bubble⟩] [⟨some client,early++late⟩]).filterMap
      (receive client)) = [⟨early++late,true,2⟩,⟨bubble,false,2⟩] := by
    rw [client_trace]
    simp [personal,waitingClients,audience,privateClients,receive,hp',List.isEmpty_iff]
  have bgroup : group stampOf bubble = [⟨s,bubble,false⟩] := by
    simpa [group,hb] using block stampOf s bubble [] sb (by simp)
  have lgroup := block stampOf t late [] sl (by simp)
  simp only [delivered,trace]
  simp only [List.foldl_cons,List.foldl_nil,bgroup]
  simp only [group,List.foldr_append]
  rw [lgroup]
  by_cases empty : late=[]
  · simp [empty,hp,block stampOf s early [] sp (by simp),merge]
  · simp only [empty,↓reduceIte]
    rw [block stampOf s early [⟨t,late,false⟩] sp (by simpa using ordered)]
    simp [hp,merge]

end ExactBallpark.Time.BatchStamps

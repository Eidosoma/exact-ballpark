-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Space.MergeDelivery
public import ExactBallparkProofs.Time.History

/-! The client groups incoming events by stamp, in their original payload
order. While inserting before an existing entry the Python iterator visits
that existing entry again. Appending the final tail copies the current flag. -/
@[expose] public section
namespace ExactBallpark.Time.MixedHistory
open ExactBallpark.Space.Batches (HistoryEntry)

abbrev Entry := HistoryEntry

def earlier (a b : Entry) := a.stamp < b.stamp

def merge : List Entry → List Entry → Bool → List Entry
  | [], incoming, w => incoming.map fun e => {e with waits := w}
  | e::es, [], _ => e::es
  | e::es, n::ns, w =>
    if e.stamp < n.stamp then e :: merge es (n::ns) false
    else if e.stamp = n.stamp then
      {e with events := e.events++n.events, waits := false} :: merge es ns w
    else {n with waits := w} :: merge (e::es) ns false
termination_by old incoming _ => old.length+incoming.length

private theorem stamp_mem (k : Int) (old inc : List Entry) (w : Bool) :
    k ∈ (merge old inc w).map HistoryEntry.stamp ↔
    k ∈ old.map HistoryEntry.stamp ∨ k ∈ inc.map HistoryEntry.stamp := by
  fun_induction merge <;> simp_all [Function.comp_def] <;> grind

private theorem lower_bound (k : Int) (old inc : List Entry) (w : Bool)
    (h₁ : ∀ e ∈ old, k < e.stamp) (h₂ : ∀ e ∈ inc, k < e.stamp) :
    ∀ e ∈ merge old inc w, k < e.stamp := by
  intro e he
  have mem := (stamp_mem e.stamp old inc w).mp (List.mem_map_of_mem he)
  rcases mem with h | h <;> obtain ⟨j,hj,eq⟩ := List.mem_map.mp h <;> grind

/-- Every stamp occurs exactly once in the sorted union of the two histories. -/
theorem merged_stamps (old inc : List Entry) (w : Bool)
    (ho : old.Pairwise earlier) (hn : inc.Pairwise earlier) :
    (merge old inc w).Pairwise earlier ∧
    ∀ k, k ∈ (merge old inc w).map HistoryEntry.stamp ↔
      k ∈ old.map HistoryEntry.stamp ∨ k ∈ inc.map HistoryEntry.stamp := by
  refine ⟨?_, fun k => stamp_mem k old inc w⟩
  fun_induction merge with
  | case1 inc w => exact (List.pairwise_map.mpr hn)
  | case2 => exact ho
  | case3 e es n ns w h ih =>
    obtain ⟨allOld,rest⟩ := List.pairwise_cons.mp ho
    apply List.pairwise_cons.mpr
    refine ⟨lower_bound e.stamp es (n::ns) false allOld ?_,ih rest hn⟩
    intro j hj
    rcases List.mem_cons.mp hj with rfl | later
    · exact h
    · exact lt_trans h ((List.pairwise_cons.mp hn).1 j later)
  | case4 e es n ns w _ eq ih =>
    obtain ⟨allOld,restOld⟩ := List.pairwise_cons.mp ho
    obtain ⟨allNew,restNew⟩ := List.pairwise_cons.mp hn
    exact List.pairwise_cons.mpr ⟨lower_bound e.stamp es ns w allOld (eq ▸ allNew),ih restOld restNew⟩
  | case5 e es n ns w h ne ih =>
    obtain ⟨allNew,restNew⟩ := List.pairwise_cons.mp hn
    have lt : n.stamp < e.stamp := by omega
    apply List.pairwise_cons.mpr
    refine ⟨lower_bound n.stamp (e::es) ns false ?_ allNew,ih ho restNew⟩
    intro j hj
    rcases List.mem_cons.mp hj with rfl | later
    · exact lt
    · exact lt_trans lt ((List.pairwise_cons.mp ho).1 j later)


def payload (stamp : Int) (rows : List Entry) : List Nat :=
  rows.flatMap fun e => if e.stamp = stamp then e.events else []

private theorem absent (k : Int) (xs : List Entry) (h : ∀ e ∈ xs, k < e.stamp) : payload k xs = [] := by
  induction xs with
  | nil => rfl
  | cons a xs ih =>
    have lt := h a (by simp)
    simpa [payload,lt.ne'] using ih (fun e he => h e (by simp [he]))

/-- Equal-stamp payloads keep every old event followed by every incoming
event, including duplicates; timestamp sorting preserves the order inside a group. -/
theorem merged_payload (k : Int) (old inc : List Entry) (w : Bool)
    (ho : old.Pairwise earlier) (hn : inc.Pairwise earlier) :
    payload k (merge old inc w) = payload k old ++ payload k inc := by
  have cons (e : Entry) (es : List Entry) : payload k (e::es) =
      (if e.stamp=k then e.events else []) ++ payload k es := rfl
  fun_induction merge with
  | case1 inc w => simp [payload,List.flatMap_map]
  | case2 => simp [payload]
  | case3 e es n ns w h ih =>
    rw [cons,cons,ih (List.pairwise_cons.mp ho).2 hn,List.append_assoc]
  | case4 e es n ns w h eq ih =>
    obtain ⟨he,hes⟩ := List.pairwise_cons.mp ho
    obtain ⟨_,hns⟩ := List.pairwise_cons.mp hn
    rw [cons,cons,cons,ih hes hns]
    by_cases key : e.stamp=k
    · have empty := absent k es (by simpa [earlier,key] using he)
      simp [key,← eq,empty]
    · simp [key,← eq]
  | case5 e es n ns w h ne ih =>
    obtain ⟨hn',hns⟩ := List.pairwise_cons.mp hn
    rw [cons,cons n ns,ih ho hns]
    by_cases key : n.stamp=k
    · have empty := absent k (e::es) (by
        obtain ⟨he,_⟩ := List.pairwise_cons.mp ho
        simp [earlier] at he; simp; grind)
      simp [key,empty]
    · simp [key]

/-- If the initial history is empty, every incoming stamp inherits the wait
flag. One bubble reply clears its matching group; each later group still
blocks the tick for any value of the client's clock. -/
theorem waiting_tail (a : Entry) (later : List Entry) (reply : List Nat)
    (h : ∀ e ∈ later, a.stamp < e.stamp) :
    let held := merge [] (a::later) true
    let after := merge held [⟨a.stamp,reply,false⟩] false
    after = {a with events := a.events++reply,waits := false} :: later.map (fun e => {e with waits := true}) ∧
      ∀ e ∈ after.tail, a.stamp < e.stamp ∧ ∀ now, waitForUpdate e.waits e.stamp now := by
  cases later <;> simp_all [merge,waitForUpdate]

/-- After applying a group, the real flush may synchronize the clock. The
extra parent evolve runs only when at least two groups remain and the next
one is ready for its bubble. A successful parent evolve advances one tick. -/
def drain (flushClock : Int → Entry → Int) (now : Int) : List Entry → List Entry × Int × List Entry
  | [] => ([],now,[])
  | e::rest =>
    if waitForUpdate e.waits e.stamp now then ([],now,e::rest) else
    let after := flushClock now e
    let next := match rest with
      | n::_::_ => if n.waits then after else after+1
      | _ => after
    let (applied,clock,pending) := drain flushClock next rest
    (e::applied,clock,pending)

/-- On successful handlers the tick consumes a prefix exactly once and stops
at the first blocking head for the clock actually reached. Retrying with
that clock and unchanged pending groups cannot consume another action. -/
theorem drain_frontier (flush : Int → Entry → Int) (now : Int) (history : List Entry) :
    let (done,clock,rest) := drain flush now history
    done++rest = history ∧ (∀ e ∈ rest.head?, waitForUpdate e.waits e.stamp clock) ∧
      drain flush clock rest = ([],clock,rest) := by
  induction history generalizing now with
  | nil => simp [drain]
  | cons e es ih =>
    by_cases block : waitForUpdate e.waits e.stamp now
    · simp [drain,block]
    · simp only [drain,block,Bool.false_eq_true,ite_false,List.cons_append,List.cons.injEq,true_and]
      apply ih

def admitPacket (reset : Bool) (headStamp lastReset : Int) (history : List Entry) : Option (Int × List Entry) :=
  if reset then some (headStamp,history.filter (fun e => headStamp ≤ e.stamp))
  else if headStamp < lastReset then none else some (lastReset,history)

/-- Reset trims existing history at its own stamp. Ordinary admission checks
only the first raw event, before grouping; a stale event later in the same
packet can therefore re-enter history, and reversing event order drops both. -/
theorem reset_packet_order (old fresh cut : Int) (stale : old < cut) (ok : cut ≤ fresh)
    (history : List Entry) (last : Int) :
    (∀ e, e ∈ ((admitPacket true cut last history).get!).2 ↔ e ∈ history ∧ cut ≤ e.stamp) ∧
    admitPacket false old cut history = none ∧
    ((admitPacket false fresh cut []).map (fun h =>
      merge h.2 [⟨old,[1],false⟩,⟨fresh,[2],false⟩] false)).map (List.map HistoryEntry.stamp)
      = some [old,fresh] := by
  simp [admitPacket,stale,not_lt.mpr ok,merge]

end ExactBallpark.Time.MixedHistory

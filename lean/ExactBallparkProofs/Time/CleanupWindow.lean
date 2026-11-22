-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Time.RemovalMasks
public import ExactBallparkProofs.Time.Tick

@[expose] public section
namespace ExactBallpark.Time.CleanupWindow
open RemovalMasks

def overdue (r : Row) := r.pending && r.moribund && decide (r.left < 0)
def stale (r : Row) := r.pending && !r.moribund
def kill (seen : Nat) (r : Row) := overdue r || (candidate r && decide (seen < 7))

private theorem tensor_cons (seen : Nat) (r : Row) (rs : List Row) :
    tensor seen (r::rs) = (stale r || kill seen r, kill seen r) ::
      tensor (seen + if candidate r then 1 else 0) rs := by
  obtain ⟨p,m,l⟩ := r
  cases p <;> cases m <;> simp [tensor,candidate,kill,overdue,stale]; grind

private theorem suffix (seen : Nat) (pre rest : List Row) :
    (tensor seen (pre ++ rest)).drop pre.length = tensor (seen+pre.countP candidate) rest := by
  induction pre generalizing seen with
  | nil => simp
  | cons r rs ih =>
    simp only [List.cons_append,List.length_cons,tensor_cons,List.drop_succ_cons,ih,List.countP_cons]
    congr 1
    cases candidate r <;> simp; omega

/-- Characterize a particular stored row without evaluating any later row. -/
theorem row_decision (pre : List Row) (r : Row) (post : List Row) :
    ((scalar 0 (pre ++ r::post)).drop pre.length).head? =
      some (stale r || kill (pre.countP candidate) r, kill (pre.countP candidate) r) := by
  rw [show 0 = min 0 7 by decide,removal_prefix_masks,suffix,tensor_cons]
  simp

/-- Retain only IDs that both met the early condition and actually got a removal mask. -/
def selected {α : Type} (rows : List (α × Row)) (masks : List (Bool × Bool)) : List α :=
  (rows.zip masks).filterMap (fun ((id,r),mask) => if candidate r && mask.2 then some id else none)

private theorem candidate_not_overdue (r : Row) (h : candidate r = true) : overdue r = false := by
  simp [candidate] at h
  simp [overdue,not_lt_of_ge h.2.1]

private theorem early_prefix {α : Type} (rows : List (α × Row)) (seen : Nat) :
    selected rows (tensor seen (rows.map Prod.snd)) =
      ((rows.filter (fun s => candidate s.2)).map Prod.fst).take (7-seen) := by
  induction rows generalizing seen with
  | nil => simp [selected]
  | cons row rs ih =>
    obtain ⟨id,r⟩ := row
    simp only [List.map_cons,tensor_cons]
    by_cases early : candidate r = true
    · have no := candidate_not_overdue r early
      by_cases space : seen < 7
      · have slots : 7-seen = (7-(seen+1))+1 := by omega
        simp [selected,kill,early,no,space,slots] at ih ⊢
        simpa using ih (seen+1)
      · have full : 7-seen = 0 := by omega
        have full' : 7-(seen+1) = 0 := by omega
        simpa [selected,kill,early,no,space,full,full',List.filterMap_cons] using ih (seen+1)
    · simpa [selected,early,List.filterMap_cons] using ih seen

theorem early_ids {α : Type} (rows : List (α × Row)) :
    selected rows (scalar 0 (rows.map Prod.snd)) =
      ((rows.filter (fun s => candidate s.2)).map Prod.fst).take 7 := by
  rw [show 0 = min 0 7 by decide,removal_prefix_masks]
  exact early_prefix rows 0

private theorem count_tensor (rs : List Row) (seen : Nat) :
    (tensor seen rs).countP Prod.snd = rs.countP overdue + min (7-seen) (rs.countP candidate) := by
  induction rs generalizing seen with
  | nil => simp [tensor]
  | cons r rs ih =>
    simp only [tensor_cons,List.countP_cons,ih]
    cases c : candidate r with
    | false => cases d : overdue r <;> simp [kill,c,d]; omega
    | true =>
      have notDue := candidate_not_overdue r c
      by_cases space : seen < 7 <;> simp [kill,c,notDue,space] <;> omega

/-- Traversal order can affect identities. The total depends only on the
counts of overdue and early candidates. Stale entries do not add removals. -/
theorem removal_count (rows : List Row) :
    (scalar 0 rows).countP Prod.snd = rows.countP overdue + min 7 (rows.countP candidate) := by
  rw [show 0 = min 0 7 by decide,removal_prefix_masks,count_tensor]

def cohort (left : Int) : Row := ⟨true,true,left⟩
def survivors {α : Type} (left : Int) (ids : List α) : List α :=
  (ids.zip (scalar 0 (ids.map (fun _ => cohort left)))).filterMap
    (fun (id,mask) => if mask.2 then none else some id)

private theorem cohort_tensor {α : Type} (ids : List α) (left : Int) (seen : Nat) :
    (ids.zip (tensor seen (ids.map (fun _ => cohort left)))).filterMap
      (fun (id,mask) => if mask.2 then none else some id) =
      if left < 0 then [] else if left < 2 then ids.drop (7-seen) else ids := by
  induction ids generalizing seen with
  | nil => simp [tensor]
  | cons id ids ih =>
    simp only [List.map_cons,tensor_cons,List.zip_cons_cons,List.filterMap_cons,ih]
    by_cases past : left < 0
    · simp [kill,overdue,candidate,cohort,past,not_le.mpr past]
    · have nonneg := le_of_not_gt past
      by_cases near : left < 2
      · by_cases room : seen < 7
        · have slots : 7-seen = (7-(seen+1))+1 := by omega
          simp [kill,overdue,candidate,cohort,past,near,room,slots,nonneg]
        · simp [kill,overdue,candidate,cohort,past,near,room,nonneg,show 7-seen=0 by omega,
            show 7-(seen+1)=0 by omega]
      · simp [kill,overdue,candidate,cohort,past,near]

/-- A group sharing a deadline, with no other entries using the quota. At two
ticks out all survive. At one and zero the first seven go. Past due all go. -/
theorem cohort_deadline {α : Type} (ids : List α) (left : Int) :
    survivors left ids = if left < 0 then [] else if left < 2 then ids.drop 7 else ids := by
  unfold survivors
  rw [show 0 = min 0 7 by decide,removal_prefix_masks,cohort_tensor]

/-- Cleanup runs even on early polls. Repeated polls at one stored time can
drain more than seven early entries without advancing the simulation clock. -/
theorem repeat_early_poll {α : Type} (ids : List α) (left : Int) (dueSoon : 0 ≤ left ∧ left < 2)
    {i : Nat} (_hi : 0 < i) (last now : Int) (waiting : now-last < i) (first master : Bool) (n : Nat) :
    onTick i now last first master = ⟨[.clean],last,first⟩ ∧
      (survivors (α := α) left)^[n] ids = ids.drop (7*n) := by
  refine ⟨early i now last first master waiting,?_⟩
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply',ih,cohort_deadline,ite_eq_right (not_lt.mpr dueSoon.1),ite_eq_left dueSoon.2,
      List.drop_drop,Nat.mul_succ]

end ExactBallpark.Time.CleanupWindow

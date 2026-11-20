-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Control
public import Mathlib.Data.Finset.Basic
public import Mathlib.Data.Finset.Range

/-! Bookkeeping around the rotated slot in MOV-726. Membership represents a set
bit. Sizes refer to resolved formation definitions. Command slots are signed
after char narrowing. Release arguments are the unsigned size_t on x86-64. -/
@[expose] public section
namespace ExactBallpark.Formation

open ExactBallpark.Control

abbrev Mask := Finset Nat

/-- Scan in the same order as ReserveFormationSlot's loop. -/
def firstFree (n : Nat) (mask : Mask) :=
  (List.range n).find? (fun i => decide (i ∉ mask))

private theorem scan_first (n : Nat) (mask : Mask) (i : Nat) :
    firstFree n mask = some i ↔ i < n ∧ i ∉ mask ∧ ∀ j < i, j ∈ mask := by
  induction n generalizing i with
  | zero => simp [firstFree]
  | succ n ih =>
    simp only [firstFree, List.range_succ, List.find?_append]
    cases scan : firstFree n mask with
    | none =>
      have full : ∀ j < n, j ∈ mask := by simpa [firstFree] using scan
      unfold firstFree at scan
      rw [scan]
      by_cases busy : n ∈ mask <;> simp [busy] <;> grind
    | some j =>
      have prior := (ih j).mp scan
      unfold firstFree at scan
      rw [scan]
      change (some j : Option Nat) = some i ↔ _
      grind

def reserve (n : Nat) (mask : Mask) : Option Nat × Mask :=
  if n > 16 then (none, mask) else
  match firstFree n mask with
  | none => (none, mask)
  | some i => (some i, insert i mask)

/-- A successful call sets just the first vacant bit in the defined slots. -/
theorem reserve_first (n i : Nat) (mask after : Mask) :
    reserve n mask = (some i, after) ↔
      n ≤ 16 ∧ i < n ∧ i ∉ mask ∧ (∀ j < i, j ∈ mask) ∧ after = insert i mask := by
  by_cases capacity : n ≤ 16
  · simp only [reserve, Nat.not_lt.mpr capacity, ite_false, capacity, true_and]
    cases scan : firstFree n mask with
    | none => grind [scan_first]
    | some j =>
      have least := (scan_first n mask j).mp scan
      simp only [Prod.mk.injEq, Option.some.injEq]
      grind [scan_first]
  · simp [reserve, Nat.lt_of_not_ge capacity, capacity]

theorem reserve_failure (n : Nat) (mask after : Mask) :
    reserve n mask = (none, after) ↔
      (16 < n ∨ ∀ i < n, i ∈ mask) ∧ after = mask := by
  by_cases oversized : 16 < n
  · simp [reserve, oversized, eq_comm]
  · simp only [reserve, oversized, ite_false, false_or]
    cases scan : firstFree n mask with
    | none =>
      have full : ∀ j < n, j ∈ mask := by simpa [firstFree] using scan
      exact ⟨fun h => ⟨full, (Prod.mk.inj h).2.symm⟩, fun ⟨_, eq⟩ => by rw [eq]⟩
    | some i =>
      have least := (scan_first n mask i).mp scan
      simp only [Prod.mk.injEq, reduceCtorEq, false_and, false_iff]
      exact fun ⟨full, _⟩ => least.2.1 (full i least.1)

/-- none denotes the out-of-range bitset reset. The caller supplies the converted size_t. -/
def release (n : Nat) (slot : Fin (2^64)) (mask : Mask) : Option Mask :=
  if slot.val > (UInt64.ofNat n - 1).toNat then some mask
  else if slot.val < 16 then some (mask.erase slot.val) else none

theorem reserve_release (n : Nat) (slot : Fin (2^64)) (mask after : Mask)
    (ok : reserve n mask = (some slot.val, after)) :
    release n slot after = some mask := by
  obtain ⟨cap, bound, vacantBit, _, rfl⟩ := (reserve_first ..).mp ok
  have encoded : (UInt64.ofNat n).toNat = n := by
    change n % (2^64) = n
    exact Nat.mod_eq_of_lt (by omega)
  have positive : (1 : UInt64) ≤ UInt64.ofNat n := by
    rw [UInt64.le_iff_toNat_le, encoded]
    change 1 ≤ n
    omega
  have small : slot.val < 16 := by omega
  have within : ¬slot.val > n-1 := by omega
  unfold release
  rw [UInt64.toNat_sub_of_le _ _ positive, encoded]
  change (if slot.val > n-1 then _ else _) = _
  simp [within, small, Finset.erase_insert vacantBit]

theorem release_empty (slot : Fin (2^64)) (mask : Mask) :
    release 0 slot mask = if slot.val < 16 then some (mask.erase slot.val) else none := by
  have limit := slot.isLt
  have h : ¬slot.val > 2^64-1 := by omega
  change (if slot.val > 2^64-1 then some mask else _) = _
  rw [ite_eq_right h]

/-- The state and call sequence observable on the joining ship. A missing size
covers a missing destination or an unassigned or stale destination formation. -/
inductive JoinAction where | ignore | stopOnly | attach
  deriving DecidableEq, Repr

def action (src dst : Int) (present sameBubble cloak : Bool) (size : Option Nat)
    (slot : Int) : JoinAction :=
  if !present || src == dst then .ignore else
  match size with
  | none => .ignore
  | some n => if slot > (n : Int) - 1 then .ignore else
              if !sameBubble || cloak then .stopOnly else .attach

def follow (a : JoinAction) (s : Following) (dst slot : Int) : Following :=
  match a with
  | .ignore => s
  | .stopOnly => stop s
  | .attach => {stop s with mode := .formation, target := some dst, stamp := slot}

theorem ignored_iff (src dst slot : Int) (present sameBubble cloak : Bool) (size : Option Nat) :
    action src dst present sameBubble cloak size slot = .ignore ↔
      present = false ∨ src = dst ∨ ∀ n ∈ size, (n : Int) ≤ slot := by
  cases size <;> cases present <;> simp [action]
  split_ifs <;> simp_all <;> omega

theorem refused_stops (src dst slot : Int) (n : Nat) (sameBubble cloak : Bool)
    (s : Following) (distinct : src ≠ dst) (fits : slot < (n : Int)) :
    (action src dst true sameBubble cloak (some n) slot = .stopOnly ∧
      follow (action src dst true sameBubble cloak (some n) slot) s dst slot = stop s) ↔
      sameBubble = false ∨ cloak = true := by
  have guard : ¬slot > (n : Int)-1 := by omega
  cases sameBubble <;> cases cloak <;> simp [action, distinct, guard, follow]

theorem joined_link (src dst slot : Int) (n : Nat) (s : Following)
    (distinct : src ≠ dst) (fits : slot < (n : Int)) :
    follow (action src dst true true false (some n) slot) s dst slot =
      {stop s with mode := .formation, target := some dst, stamp := slot} := by
  have guard : ¬slot > (n : Int)-1 := by omega
  simp [follow, action, distinct, guard]

structure Follower where
  id : Int
  mode : Mode
  slot : Int
  deriving DecidableEq

def stoppedOnResize (old new : Nat) (followers : List Follower) :=
  if old > new then followers.filter (fun f => f.mode == .formation && f.slot > (new : Int)-1)
  else []

theorem stopped_iff (old new : Nat) (fs : List Follower) (f : Follower) :
    f ∈ stoppedOnResize old new fs ↔
      new < old ∧ f ∈ fs ∧ f.mode = .formation ∧ (new : Int) ≤ f.slot := by
  simp only [stoppedOnResize]
  split <;> simp_all
  omega

/-- For live followers pointing at this leader with valid assigned slots, each
master stop clears its slot before the leader changes formation ID. The C++
loop from old to new has no iterations on a shrink. -/
def resizeMask (master : Bool) (old new : Nat) (fs : List Follower) (mask : Mask) : Mask :=
  if master then (stoppedOnResize old new fs).foldl (fun bits f => bits.erase f.slot.toNat) mask
  else mask

private theorem fold_erase (fs : List Follower) (mask : Mask) (i : Nat) :
    i ∈ fs.foldl (fun bits f => bits.erase f.slot.toNat) mask ↔
      i ∈ mask ∧ ∀ f ∈ fs, f.slot.toNat ≠ i := by
  induction fs generalizing mask with
  | nil => simp
  | cons f fs ih =>
    simp [ih, Finset.mem_erase, ne_comm, and_assoc, and_left_comm]

theorem resize_bit (master : Bool) (old new i : Nat) (fs : List Follower) (mask : Mask) :
    i ∈ resizeMask master old new fs mask ↔
      i ∈ mask ∧ (master = true → ∀ f ∈ fs, new < old → f.mode = .formation →
        (new : Int) ≤ f.slot → f.slot.toNat ≠ i) := by
  cases master <;> simp [resizeMask, fold_erase, stopped_iff]
  grind

/-- An unclaimed reservation survives shrinking and growing back. Even if the
bit is visible again, allocation cannot return that slot. -/
theorem regrow_skips_reserved (master : Bool) (old new i : Nat) (fs remaining : List Follower)
    (mask : Mask) (shrink : new < old) (busy : i ∈ mask)
    (unclaimed : ∀ f ∈ fs, f.mode = .formation → f.slot.toNat ≠ i) :
    let shrunk := resizeMask master old new fs mask
    let grown := resizeMask master new old remaining shrunk
    i ∈ grown ∧ ∀ after, reserve old grown ≠ (some i, after) := by
  dsimp only
  have kept := (resize_bit master old new i fs mask).2
    ⟨busy, fun _ f mem _ mode _ => unclaimed f mem mode⟩
  have still := (resize_bit master new old i remaining _).2
    ⟨kept, fun _ _ _ impossible => by omega⟩
  exact ⟨still, fun _ h => (reserve_first .. |>.mp h).2.2.1 still⟩

end ExactBallpark.Formation

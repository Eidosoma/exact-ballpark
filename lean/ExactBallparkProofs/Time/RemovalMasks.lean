-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Time.Schedule

/-! A row is projected from a world's sorted by_id array. Padding has pending
false. Each mask pair is (clear pending, remove ball). -/
@[expose] public section
namespace ExactBallpark.Time.RemovalMasks

structure Row where
  pending : Bool
  moribund : Bool
  left : Int

def candidate (r : Row) := r.pending && r.moribund && decide (0 ≤ r.left ∧ r.left < 2)

def scalar (used : Nat) : List Row → List (Bool × Bool)
  | [] => []
  | r::rs =>
    let (clear,next) := if !r.pending then (false,used) else
      if !r.moribund then (true,used) else due r.left 0 used
    (clear,clear && r.moribund) :: scalar next rs

def tensor (seen : Nat) : List Row → List (Bool × Bool)
  | [] => []
  | r::rs =>
    let inclusive := seen + if candidate r then 1 else 0
    let clear := (r.pending && (!r.moribund || decide (r.left < 0))) ||
      (candidate r && decide (inclusive ≤ 7))
    (clear,clear && r.moribund) :: tensor inclusive rs

/-- The scalar counter saturates at seven while the tensor prefix count keeps
growing. Stale entries clear without removal and overdue balls consume no quota. -/
theorem removal_prefix_masks (rows : List Row) (seen : Nat) :
    scalar (min seen 7) rows = tensor seen rows := by
  induction rows generalizing seen with
  | nil => rfl
  | cons r rs ih =>
    simp only [scalar, tensor]
    rw [← ih]
    congr 1
    · obtain ⟨p,m,l⟩ := r
      cases p <;> cases m <;> simp [candidate,due]; split_ifs <;> simp_all <;> omega
    · congr 1
      obtain ⟨p,m,l⟩ := r
      cases p <;> cases m <;> simp [candidate,due]; split_ifs <;> simp_all <;> omega

end ExactBallpark.Time.RemovalMasks

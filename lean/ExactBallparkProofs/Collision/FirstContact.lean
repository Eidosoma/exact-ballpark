-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.DragChord
public import Mathlib.Topology.Order.IntermediateValue

@[expose] public section
namespace ExactBallpark.Collision.FirstContact
open Set

/-- Time is measured from the start of this interval. Equality counts as contact. -/
def First (distance : ℝ → ℝ) (radius duration t : ℝ) : Prop :=
  t ∈ Icc 0 duration ∧ distance t ≤ radius ∧ ∀ s ∈ Ico 0 t, radius < distance s

theorem exists_first {d : ℝ → ℝ} {r T : ℝ} (cont : ContinuousOn d (Icc 0 T))
    (hit : ∃ s ∈ Icc 0 T, d s ≤ r) : ∃ t, First d r T t := by
  let contacts := Icc 0 T ∩ d ⁻¹' Iic r
  obtain ⟨s, hs, hr⟩ := hit
  have closed : IsClosed contacts := cont.preimage_isClosed_of_isClosed isClosed_Icc isClosed_Iic
  have compact : IsCompact contacts := isCompact_Icc.of_isClosed_subset closed inter_subset_left
  have least := compact.isLeast_sInf (show contacts.Nonempty from ⟨s, hs, hr⟩)
  refine ⟨_, least.1.1, least.1.2, fun u hu => lt_of_not_ge fun h => ?_⟩
  exact hu.2.not_ge (least.2 ⟨⟨hu.1, hu.2.le.trans least.1.1.2⟩, h⟩)

theorem initial_contact {d : ℝ → ℝ} {r T : ℝ} (hT : 0 ≤ T) (h : d 0 ≤ r) :
    First d r T 0 := ⟨⟨le_rfl, hT⟩, h, fun _ hs => False.elim (hs.2.not_ge hs.1)⟩

theorem first_unique {d : ℝ → ℝ} {r T s t : ℝ} (hs : First d r T s) (ht : First d r T t) :
    s = t := le_antisymm (le_of_not_gt fun h => (hs.2.2 t ⟨ht.1.1, h⟩).not_ge ht.2.1)
      (le_of_not_gt fun h => (ht.2.2 s ⟨hs.1.1, h⟩).not_ge hs.2.1)

/-- A positive first contact of continuous paths lies exactly on the surface. -/
theorem first_on_surface {d : ℝ → ℝ} {r T t : ℝ} (cont : ContinuousOn d (Icc 0 T))
    (first : First d r T t) (ht : 0 < t) : d t = r := by
  have initial := first.2.2 0 ⟨le_rfl, ht⟩
  obtain ⟨s, hs, eq⟩ := intermediate_value_Icc' ht.le
    (cont.mono (Icc_subset_Icc_right first.1.2)) ⟨first.2.1, initial.le⟩
  have same : s = t := hs.2.eq_of_not_lt fun h => (first.2.2 s ⟨hs.1, h⟩).ne eq.symm
  simpa [same] using eq

theorem first_in_window {d : ℝ → ℝ} {r T lo hi : ℝ} (cont : ContinuousOn d (Icc 0 T))
    (hlo : 0 ≤ lo) (ordered : lo ≤ hi) (endTime : hi ≤ T)
    (prefixClear : ∀ s ∈ Ico 0 lo, r < d s) (hit : d hi ≤ r) :
    ∃ t ∈ Icc lo hi, First d r T t := by
  obtain ⟨t, first⟩ := exists_first cont ⟨hi, ⟨hlo.trans ordered, endTime⟩, hit⟩
  exact ⟨t, ⟨le_of_not_gt fun h => (prefixClear t ⟨first.1.1, h⟩).not_ge first.2.1,
    le_of_not_gt fun h => (first.2.2 hi ⟨hlo.trans ordered, h⟩).not_ge hit⟩, first⟩

/-- The two windows refer to eligible obstacles on the same uninterrupted trajectory
and clock. Every possible first hit against A precedes every first hit against B. -/
theorem disjoint_first_contacts {a b : ℝ → ℝ} {ra rb T al ah bl bh : ℝ}
    (wa : ∃ s ∈ Icc al ah, First a ra T s) (wb : ∃ t ∈ Icc bl bh, First b rb T t)
    (before : ah < bl) {s t : ℝ} (ha : First a ra T s) (hb : First b rb T t) : s < t := by
  obtain ⟨x, hx, fx⟩ := wa
  obtain ⟨y, hy, fy⟩ := wb
  rw [← first_unique fx ha, ← first_unique fy hb]
  exact (hx.2.trans_lt before).trans_le hy.1

end ExactBallpark.Collision.FirstContact

-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.SpecialBranch

/-! Cache-relevant fields of one scalar ball. Agility inputs are the values after the
source's float conversion. Friction and the tick interval stay fixed across a history. -/
@[expose] public section

namespace ExactBallpark.Movement

structure CacheState where
  mass : ℝ
  agility : ℝ
  factor : ℝ

inductive CacheUpdate where
  | setMass (mass : ℝ)
  | setAgility (agility : ℝ)

namespace CacheState

noncomputable section

def refreshed (k dt mass agility : ℝ) : CacheState :=
  ⟨mass, agility, refreshFactor mass agility k dt⟩

def addBall (k dt mass agility : ℝ) : CacheState :=
  refreshed k dt (if mass < 0 then 0 else mass) (if agility ≤ 0 then 1 else agility)

def update (k dt : ℝ) (s : CacheState) : CacheUpdate → CacheState
  | .setMass m => if m ≤ 0 then s else refreshed k dt m s.agility
  | .setAgility a => if a ≤ 0 then s else refreshed k dt s.mass a

/-- A valid cache stores zero for nonpositive effective mass and the one-tick decay otherwise. -/
def Valid (k dt : ℝ) (s : CacheState) : Prop :=
  let m := effectiveMass s.mass s.agility
  s.factor = if m ≤ 0 then 0 else decay m k dt

private theorem refreshed_valid (k dt mass agility : ℝ) : (refreshed k dt mass agility).Valid k dt := by
  change refreshFactor mass agility k dt = _
  by_cases h : effectiveMass mass agility ≤ 0
  · simp [refreshed, refreshFactor, h]
  · simpa [refreshed, h] using refresh_positive (lt_of_not_ge h) k dt

theorem addBall_valid (k dt mass agility : ℝ) : (addBall k dt mass agility).Valid k dt :=
  refreshed_valid ..

theorem update_valid {s : CacheState} {k dt : ℝ} (h : s.Valid k dt) (u : CacheUpdate) :
    (update k dt s u).Valid k dt := by
  cases u with
  | setMass m => by_cases hm : m ≤ 0 <;> simp [update, hm, h, refreshed_valid]
  | setAgility a => by_cases ha : a ≤ 0 <;> simp [update, ha, h, refreshed_valid]

theorem history_valid {k dt : ℝ} (updates : List CacheUpdate) {s : CacheState} (h : s.Valid k dt) :
    (updates.foldl (update k dt) s).Valid k dt := by
  induction updates generalizing s with
  | nil => exact h
  | cons u us ih => exact ih (update_valid h u)

/-- Discharge the integrator's cache premise after construction and any finite sequence of setters. -/
theorem velocity_after_updates (k dt mass agility : ℝ) (updates : List CacheUpdate)
    (p v a t : ℝ) (hk : k ≠ 0) :
    let s := updates.foldl (update k dt) (addBall k dt mass agility)
    let m := effectiveMass s.mass s.agility
    0 < m → (integrate p v a m k dt s.factor t).2 = velocity v a m k t := by
  dsimp only
  intro hm
  apply integrate_velocity hk
  intro _
  have valid := history_valid updates (addBall_valid k dt mass agility)
  simpa only [Valid, ite_eq_right (not_le.mpr hm)] using valid

end
end CacheState
end ExactBallpark.Movement

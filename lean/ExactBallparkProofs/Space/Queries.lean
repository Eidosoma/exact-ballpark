-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Flight

@[expose] public noncomputable section
namespace ExactBallpark.Space
open ExactBallpark.Movement

def offsets : List ℤ := [-1,0,1]

def neighbors : List (ℤ × ℤ × ℤ) :=
  ((offsets.flatMap fun i => offsets.flatMap fun j => offsets.map fun k => (i,j,k))).filter (· != (0,0,0))

private theorem offset_iff (i : ℤ) : i ∈ offsets ↔ -1 ≤ i ∧ i ≤ 1 := by
  simp [offsets]; omega

theorem neighbor_offsets : neighbors.length = 26 ∧ neighbors.Nodup ∧ (0,0,0) ∉ neighbors := by decide

theorem neighbor_iff (i j k : ℤ) :
    (i,j,k) ∈ neighbors ↔ -1 ≤ i ∧ i ≤ 1 ∧ -1 ≤ j ∧ j ≤ 1 ∧ -1 ≤ k ∧ k ≤ 1 ∧
      (i ≠ 0 ∨ j ≠ 0 ∨ k ≠ 0) := by
  simp only [neighbors, List.mem_filter, List.mem_flatMap, List.mem_map, offset_iff, bne_iff_ne]
  grind

def rangeQuery (d r radius : ℝ) := d*d ≤ (r+radius)*(r+radius)

theorem range_abs (d r radius : ℝ) (h : 0 ≤ d) :
    rangeQuery d r radius ↔ d ≤ |r+radius| := by
  rw [← sq_le_sq₀ h (abs_nonneg (r+radius)), sq_abs]
  simp [rangeQuery, pow_two]

structure Occluder where
  id : ℤ
  eligible : Bool
  projection : ℝ
  centerSq : ℝ
  radius : ℝ

def blocks (o : Occluder) (distance : ℝ) : Bool :=
  let dsq := o.radius^2 - (o.centerSq - o.projection^2)
  if !o.eligible || decide (dsq < 0) then false else
  let entry := o.projection - Real.sqrt dsq
  decide (0 < entry ∧ entry < distance)

def visibility (objects : List Occluder) (distance : ℝ) :=
  ((objects.find? (fun o => blocks o distance)).map Occluder.id).getD 0

theorem first_occluder (before after : List Occluder) (o : Occluder) (d : ℝ)
    (noneBefore : ∀ b ∈ before, blocks b d = false) (hit : blocks o d = true) :
    visibility (before ++ o :: after) d = o.id := by
  induction before with
  | nil => simp [visibility, hit]
  | cons b bs ih =>
    have hb := noneBefore b (by simp)
    simpa [visibility, hb] using ih (fun c hc => noneBefore c (by simp [hc]))

theorem visibility_entry (o : Occluder) (d : ℝ) :
    blocks o d ↔ o.eligible ∧
    let dsq := o.radius^2 - (o.centerSq - o.projection^2)
    0 ≤ dsq ∧ 0 < o.projection - Real.sqrt dsq ∧ o.projection - Real.sqrt dsq < d := by
  simp [blocks, and_assoc]

/-- Source sentinel −1 disables a field's matching tag. Harmonic −2 is the master pass. -/
def fieldPass (h corp alliance hf cf af : ℤ) :=
  if h = -2 then true else if hf ≠ -1 ∧ hf = h then true else
    if cf ≠ -1 ∧ cf = corp then true else decide (af ≠ -1 ∧ af = alliance)

theorem field_iff (h c a hf cf af : ℤ) :
    fieldPass h c a hf cf af ↔ h = -2 ∨ (hf ≠ -1 ∧ hf = h) ∨
    (cf ≠ -1 ∧ cf = c) ∨ (af ≠ -1 ∧ af = a) := by
  simp [fieldPass]

def capsuleEnds (free : Bool) (p a b : Vec3) :=
  if free then none else some (p+a, p+b)

theorem fixed_capsule (p a b : Vec3) :
    (capsuleEnds false p a b).map (fun (u,v) => v-u) = some (b-a) ∧
    capsuleEnds true p a b = none := by
  simp [capsuleEnds]

structure CloakState where
  mode : ℤ
  massive : Bool
  releaseCalls : ℕ
  sensorCalls : ℕ
  deriving DecidableEq

def cloak (s : CloakState) (master : Bool) (mode : ℤ) :=
  if mode ≤ 0 then s else
  ⟨mode % 256, false, s.releaseCalls+1, s.sensorCalls + if master ∧ mode = 1 then 1 else 0⟩

theorem cloak_effect (s : CloakState) (m : ℤ) (positive : 0 < m) (master : Bool) :
    let t := cloak s master m
    t.massive = false ∧ t.releaseCalls = s.releaseCalls+1 ∧
    (t.sensorCalls = s.sensorCalls+1 ↔ master ∧ m = 1) := by
  simp only [cloak, ite_eq_right (not_le.mpr positive), true_and]
  split_ifs <;> simp_all

end ExactBallpark.Space

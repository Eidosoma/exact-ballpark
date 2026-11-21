-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Space.Queries

@[expose] public noncomputable section
namespace ExactBallpark.Space.RangeScan
open ExactBallpark.Movement

structure Ball where
  pos : Vec3
  radius : ℝ
  moribund : Bool
  cloaked : Bool

/-- A resolved bubble supplies dictionary keys in iteration order. A key
identifies one ball object, as in a well-formed ballpark registry. -/
def accepts (center : Vec3) (rng : ℝ) (showCloaked : Bool) (b : Ball) : Prop :=
  b.moribund = false ∧ (b.cloaked = false ∨ showCloaked = true) ∧
    ‖b.pos-center‖^2 ≤ (rng+b.radius)^2

def visit {α : Type} (emit : ℝ → Int → α) (lookup : Int → Option Ball)
    (ref : Option Int) (center : Vec3) (rng : ℝ) (cloak : Bool) (id : Int) : Option α := by
  classical
  exact do
    let b ← lookup id
    if ref ≠ some id ∧ accepts center rng cloak b then some (emit (‖b.pos-center‖^2) id) else none

def scan {α : Type} (emit : ℝ → Int → α) (lookup : Int → Option Ball)
    (ref : Option Int) (center : Vec3) (rng : ℝ) (cloak : Bool) (ids : List Int) :=
  ids.filterMap (visit emit lookup ref center rng cloak)

abbrev withDist := fun (dsq : ℝ) (id : Int) => (dsq,id)
abbrev idsOnly := fun (_ : ℝ) (id : Int) => id

/-- The two output formats project the same ordered visits and retain the
squared center distance, including contact at the radius-expanded limit. -/
theorem ordered_scan (lookup : Int → Option Ball) (ref : Option Int) (p : Vec3)
    (r : ℝ) (cloak : Bool) (keys : List Int) :
    (scan withDist lookup ref p r cloak keys).map Prod.snd = scan idsOnly lookup ref p r cloak keys ∧
    ∀ d id, (d,id) ∈ scan withDist lookup ref p r cloak keys ↔
      id ∈ keys ∧ ref ≠ some id ∧ ∃ b, lookup id = some b ∧ accepts p r cloak b ∧ d = ‖b.pos-p‖^2 := by
  classical
  constructor
  · induction keys with
    | nil => rfl
    | cons k ks ih =>
      cases lk : lookup k with
      | none => simpa [scan,visit,lk] using ih
      | some b =>
        by_cases keep : ref ≠ some k ∧ accepts p r cloak b <;> simp_all [scan,visit]
  · intro d id
    simp only [scan,List.mem_filterMap]
    constructor
    · rintro ⟨k,hk,found⟩
      cases lk : lookup k with
      | none => simp [visit,lk] at found
      | some b =>
        by_cases keep : ref ≠ some k ∧ accepts p r cloak b <;> simp [visit,lk,keep] at found
        obtain ⟨rfl,rfl⟩ := found
        exact ⟨hk,keep.1,b,lk,keep.2,rfl⟩
    · rintro ⟨hk,hr,b,lk,hb,rfl⟩
      exact ⟨id,hk,by simp [visit,lk,hr,hb]⟩

/-- Once both overloads resolve the same bubble, querying by ball ID
removes exactly that object from the coordinate result, preserving order. -/
theorem query_by_id (lookup : Int → Option Ball) (id : Int) (source : Ball)
    (existsAt : lookup id = some source) (r : ℝ) (cloak : Bool) (keys : List Int) :
    scan withDist lookup (some id) source.pos r cloak keys =
      (scan withDist lookup none source.pos r cloak keys).filter (fun pair => pair.2 != id) := by
  classical
  induction keys with
  | nil => rfl
  | cons k ks ih =>
    by_cases same : k = id
    · subst k
      by_cases keep : accepts source.pos r cloak source <;> simp_all [scan,visit]
    · cases lk : lookup k with
      | none => simpa [scan,visit,lk] using ih
      | some b =>
        by_cases keep : accepts source.pos r cloak b <;> simp_all [scan,visit,Ne.symm same]

end ExactBallpark.Space.RangeScan

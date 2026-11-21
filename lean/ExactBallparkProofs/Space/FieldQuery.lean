-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Time.ServerLaws
public import ExactBallparkProofs.Space.CandidateOrder

@[expose] public section
namespace ExactBallpark.Space.FieldQuery
open ExactBallpark.Time.ServerLaws CandidateOrder

/-- Box visits after the common eligibility tests. Only the field's settings
change between the two queries. Repeated visits are resolved by the handled bit. -/
def configuredQuery (old : Field) (tags ship : Tags) (fieldId : Nat) (visits : List Nat) :=
  let f := (configure tags).foldl applyField old
  walk id ∅ (visits.filter (fun i => i != fieldId || kept ship f))

theorem configured_membership (old : Field) (tags ship : Tags) (fieldId : Nat) (visits : List Nat) :
    let found := configuredQuery old tags ship fieldId visits
    found.Nodup ∧ ∀ i, i ∈ found ↔ i ∈ visits ∧ (i ≠ fieldId ∨
      (enabled tags ∧ ship.harmonic ≠ -2 ∧
       (tags.harmonic = -1 ∨ tags.harmonic ≠ ship.harmonic) ∧
       (tags.corp = -1 ∨ tags.corp ≠ ship.corp) ∧
       (tags.alliance = -1 ∨ tags.alliance ≠ ship.alliance))) := by
  let f := (configure tags).foldl applyField old
  let eligible := visits.filter (fun i => i != fieldId || kept ship f)
  have walkFacts : (walk id ∅ eligible).Nodup ∧ ∀ i, i ∈ walk id ∅ eligible ↔ i ∈ eligible := by
    cases he : eligible with
    | nil => simp [walk]
    | cons a rest =>
      have h := first_visit_once id [] rest a (by simp)
      simpa using And.intro h.1 h.2.1
  refine ⟨walkFacts.1, fun i => (walkFacts.2 i).trans ?_⟩
  simp [eligible, f, (configured_field_filter old tags ship).2.2.2]

end ExactBallpark.Space.FieldQuery

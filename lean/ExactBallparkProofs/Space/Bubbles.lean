-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Space.Queries
public import Mathlib.Data.Finset.Card

/-! Finite top-level boxes that exist and still have bubble ID −1 are the initial
pool. Entering a box removes it from the pool before visiting its neighbors in
source loop order. Already labeled and missing boxes both return immediately. -/
@[expose] public section
namespace ExactBallpark.Space

abbrev BoxKey := ℤ × ℤ × ℤ

/-- LEVEL_GRID[0] has 2^23 cells per axis. CheckBox clamps before looking up the key. -/
def topIndex (i : ℤ) : ℤ := if i < 0 then 0 else if i ≥ 2^23 then 2^23-1 else i

def adjacentKeys (p : BoxKey) : List BoxKey :=
  neighbors.map (fun d => (topIndex (p.1 + d.1), topIndex (p.2.1 + d.2.1), topIndex (p.2.2 + d.2.2)))

theorem corner_neighbors :
    (adjacentKeys (0,0,0)).eraseDups.length = 8 ∧ (0,0,0) ∈ adjacentKeys (0,0,0) := by decide

/-- Return the boxes still unassigned when this seed's depth-first traversal finishes. -/
def unassigned (adj : BoxKey → List BoxKey) (pool : Finset BoxKey) (todo : List BoxKey) : Finset BoxKey :=
  match todo with
  | [] => pool
  | p :: rest =>
    if _h : p ∈ pool then unassigned adj (pool.erase p) (adj p ++ rest)
    else unassigned adj pool rest
termination_by (pool.card, todo.length)
decreasing_by
  · exact Prod.Lex.left _ _ (Finset.card_erase_lt_of_mem _h)
  · exact Prod.Lex.right _ (by simp)

private theorem closed_result (adj : BoxKey → List BoxKey) (pool : Finset BoxKey) (todo : List BoxKey) :
    let final := unassigned adj pool todo
    final ⊆ pool ∧ (∀ p ∈ todo, p ∉ final) ∧
    ∀ p ∈ pool, p ∉ final → ∀ q ∈ adj p, q ∉ final := by
  fun_induction unassigned with
  | case1 => simp; grind
  | case2 pool p rest hp ih =>
    obtain ⟨sub, visit, closed⟩ := ih
    refine ⟨fun _ h => Finset.mem_of_mem_erase (sub h), ?_, ?_⟩
    · intro x hx
      rcases List.mem_cons.mp hx with rfl | hr
      · exact fun h => Finset.notMem_erase _ pool (sub h)
      · exact visit x (List.mem_append_right _ hr)
    · intro x hx gone q hq
      by_cases eq : x = p
      · subst x; exact visit q (List.mem_append_left _ hq)
      · exact closed x (Finset.mem_erase.mpr ⟨eq,hx⟩) gone q hq
  | case3 pool p rest hp ih =>
    obtain ⟨sub, visit, closed⟩ := ih
    refine ⟨sub, ?_, closed⟩
    intro x hx
    rcases List.mem_cons.mp hx with rfl | hr
    · exact fun h => hp (sub h)
    · exact visit x hr

/-- Every removed box satisfies any invariant true for the seeds and preserved by an edge. -/
private theorem visited_invariant (adj : BoxKey → List BoxKey) (pool : Finset BoxKey)
    (todo : List BoxKey) (P : BoxKey → Prop) :
    (∀ x ∈ todo, x ∈ pool → P x) →
    (∀ x ∈ pool, P x → ∀ y ∈ adj x, y ∈ pool → P y) →
    ∀ z ∈ pool, z ∉ unassigned adj pool todo → P z := by
  fun_induction unassigned with
  | case1 => simp; grind
  | case2 pool p rest hp ih =>
    intro seeds step z hz gone
    have start := seeds p (by simp) hp
    by_cases eq : z = p
    · simpa [eq] using start
    · apply ih _ _ z (Finset.mem_erase.mpr ⟨eq,hz⟩) gone
      · intro x hx member
        rcases List.mem_append.mp hx with neighbor | pending
        · exact step p hp start x neighbor (Finset.mem_of_mem_erase member)
        · exact seeds x (List.mem_cons_of_mem _ pending) (Finset.mem_of_mem_erase member)
      · intro x hx px y hy member
        exact step x (Finset.mem_of_mem_erase hx) px y hy (Finset.mem_of_mem_erase member)
  | case3 pool p rest hp ih =>
    intro seeds step
    exact ih (fun x hx => seeds x (List.mem_cons_of_mem _ hx)) step

/-- Paths stay within the original unlabeled top-level boxes at every hop. -/
inductive Connected (pool : Finset BoxKey) (seed : BoxKey) : BoxKey → Prop
  | start (h : seed ∈ pool) : Connected pool seed seed
  | hop {p q : BoxKey} (path : Connected pool seed p) (edge : q ∈ adjacentKeys p)
      (present : q ∈ pool) : Connected pool seed q

theorem bubble_component (pool : Finset BoxKey) (seed x : BoxKey) :
    (x ∈ pool ∧ x ∉ unassigned adjacentKeys pool [seed]) ↔ Connected pool seed x := by
  constructor
  · rintro ⟨hx, gone⟩
    apply visited_invariant adjacentKeys pool [seed] (Connected pool seed) _ _ x hx gone
    · intro y hy present
      have : y = seed := by simpa using hy
      subst y; exact .start present
    · intro p _ path q edge present
      exact .hop path edge present
  · obtain ⟨_, visited, closed⟩ := closed_result adjacentKeys pool [seed]
    intro path
    induction path with
    | start present => exact ⟨present, visited seed (by simp)⟩
    | hop path edge present ih => exact ⟨present, closed _ ih.1 ih.2 _ edge⟩

end ExactBallpark.Space

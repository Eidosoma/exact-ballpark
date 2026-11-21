-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Collision.FiniteSweep
public import Mathlib.Data.List.MinMax

@[expose] public section
namespace ExactBallpark.Collision.SearchWindow

/-- Input records have passed the shape, sweep and repeat-contact guards. -/
abbrev Hit := Nat × ℝ

noncomputable def stamp (s e u : ℝ) := s + u * (e-s)

noncomputable def takeHit (s : ℝ) (best : Option Hit) (key : ℝ) (hit : Hit) : Option Hit :=
  match best with
  | none => some hit
  | some b => if b.2 = s then best else if key < b.2 then some hit else best

noncomputable def fixedScan (s e : ℝ) (xs : List Hit) (initial : Option Hit := none) :=
  xs.foldl (fun best x => takeHit s best x.2 (x.1, stamp s e x.2)) initial

noncomputable def shapeScan (s e : ℝ) (xs : List Hit) :=
  xs.foldl (fun best x =>
    let t := stamp s e x.2
    takeHit s best t (x.1,t)) none

private theorem absolute_fold (s : ℝ) (xs : List Hit) (h : ∀ x ∈ xs, s ≤ x.2) (b : Option Hit) :
    xs.foldl (fun best x => takeHit s best x.2 x) b =
      xs.foldl (List.argAux fun x y : Hit => x.2 < y.2) b := by
  induction xs generalizing b with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldl_cons]
    rw [ih (fun y hy => h y (by simp [hy]))]
    congr 1
    have hx := h x List.mem_cons_self
    cases b with
    | none => rfl
    | some b =>
      by_cases hb : b.2 = s
      · simp [takeHit, List.argAux, hb, not_lt.mpr hx]
      · simp [takeHit, List.argAux, hb]

theorem full_tick_first_min (xs : List Hit) (valid : ∀ x ∈ xs, 0 ≤ x.2) :
    fixedScan 0 1 xs = xs.argmin Prod.snd := by
  simpa [fixedScan, stamp, List.argmin] using absolute_fold 0 xs valid none

def axis (x : Float) : ExactBallpark.Collision.FloatPoint := fun i => if i = 0 then x else 0

/-- Finite half-tick fixture. The comparison receives the window, candidate fraction
and incumbent time. Geometry and stored timestamps are shared by both replays. -/
def fixedReplayBy (earlier : Float → Float → Float → Float → Bool)
    (obstacles : List (Nat × Float)) : Nat × UInt64 := Id.run do
  let half := Float.ofBits 0x3fe0000000000000
  let mut t : Float := -1
  let mut winner := 0
  for (id,x) in obstacles do
    let u := contact64 (axis 0) (axis 8) (axis x) (axis x) 1
    if u ≥ 0 && (t < 0 || earlier half 1 u t) then
      t := half + u * (1-half)
      winner := id
    if t == half then break
  return (winner,t.toBits)

/-- Destiny compares the candidate fraction directly with the stored time. -/
def fixedReplay := fixedReplayBy (fun _ _ fraction held => fraction < held)

set_option maxRecDepth 16384 in
/-- The two comparisons disagree on exactly the half-open interval between the
two operands. The finite witness computes the roots from the encoded geometry. -/
theorem partial_tick_selection (s e u held : ℝ) :
    (¬ ((u < held) ↔ (stamp s e u < held)) ↔
      (u < held ∧ held ≤ stamp s e u) ∨ (stamp s e u < held ∧ held ≤ u)) ∧
    fixedReplay [(3,3),(4,5)] = (4,0x3fe8000000000000) ∧
    fixedReplay [(4,5),(3,3)] = (3,0x3fe4000000000000) := by
  exact ⟨by grind, by decide +kernel, by decide +kernel⟩

noncomputable def laterPasses (s : ℝ) (shapes : Option Hit)
    (fixed free : Option Hit → Option Hit) :=
  if shapes.any (fun x => decide (x.2 = s)) then shapes else
    let b := fixed shapes
    if b.any (fun x => decide (x.2 = s)) then b else free b

theorem static_priority (s e : ℝ) (xs : List Hit) (valid : ∀ x ∈ xs, s ≤ stamp s e x.2)
    (fixed free : Option Hit → Option Hit) :
    let absolute := xs.map (fun x => (x.1, stamp s e x.2))
    shapeScan s e xs = absolute.argmin Prod.snd ∧
    ∀ id, shapeScan s e xs = some (id,s) →
      laterPasses s (shapeScan s e xs) fixed free = some (id,s) := by
  constructor
  · have valid' : ∀ x ∈ xs.map (fun x => (x.1, stamp s e x.2)), s ≤ x.2 := by
      intro x hx
      obtain ⟨y,hy,rfl⟩ := List.mem_map.mp hx
      exact valid y hy
    simpa [shapeScan, List.foldl_map, Function.comp_def, List.argmin] using
      absolute_fold s _ valid' none
  · intro id h
    simp [laterPasses, h]

def spans {α : Type} (s : α) (cuts : List α) (e : α) : List (α × α) :=
  match cuts with
  | [] => [(s,e)]
  | x :: xs => (s,x) :: spans x xs e

noncomputable def freeWindows (s e otherEnd : ℝ) (history : List ℝ) :=
  let stop := min e otherEnd
  if s < stop then spans s (history.filter (fun t => decide (s < t ∧ t < stop))) stop else []

private theorem span_cover (s e t : ℝ) (xs : List ℝ) (ordered : xs.Pairwise (· ≤ ·))
    (inside : ∀ x ∈ xs, s ≤ x ∧ x ≤ e) :
    ((spans s xs e).map Prod.snd = xs ++ [e]) ∧
    ((∃ w ∈ spans s xs e, w.1 ≤ t ∧ t ≤ w.2) ↔ s ≤ t ∧ t ≤ e) := by
  induction xs generalizing s with
  | nil => simp [spans]
  | cons x xs ih =>
    have head := inside x List.mem_cons_self
    have next := (List.pairwise_cons.mp ordered)
    have rest :=  ih x next.2 fun y hy => ⟨next.1 _ hy, (inside _ (by simp [hy])).2⟩
    simp only [spans, List.map_cons, rest.1, List.cons_append, true_and, List.mem_cons]
    constructor
    · rintro ⟨w, rfl | mem, h⟩
      · exact ⟨h.1,h.2.trans head.2⟩
      · have ht := rest.2.mp ⟨_,mem,h⟩
        exact ⟨head.1.trans ht.1,ht.2⟩
    · intro ht
      by_cases hx : t ≤ x
      · exact ⟨_, Or.inl rfl, ht.1,hx⟩
      · obtain ⟨w,hw,h⟩ := rest.2.mpr ⟨le_of_not_ge hx,ht.2⟩
        exact ⟨w,Or.inr hw,h⟩

theorem free_subintervals (s e otherEnd t : ℝ) (history : List ℝ) (ordered : history.Pairwise (· ≤ ·)) :
    let stop := min e otherEnd
    let cuts := history.filter (fun u => decide (s < u ∧ u < stop))
    (freeWindows s e otherEnd history).map Prod.snd = (if s < stop then cuts ++ [stop] else []) ∧
    ((∃ w ∈ freeWindows s e otherEnd history, w.1 ≤ t ∧ t ≤ w.2) ↔
      s < stop ∧ s ≤ t ∧ t ≤ stop) := by
  dsimp only
  unfold freeWindows
  split
  next h =>
    simp only [h, ite_true, true_and]
    apply span_cover _ _ _ _ (ordered.filter _)
    intro x hx
    have within : s < x ∧ x < min e otherEnd := by simpa using (List.mem_filter.mp hx).2
    exact ⟨within.1.le,within.2.le⟩
  next h => simp [h]

/-- Anchors hold both balls' start positions. On a continue the clock still
advances to the next boundary. On an ordinary multi-segment miss the anchors advance too. -/
structure Cursor (P : Type) where
  clock : ℝ
  anchors : P

noncomputable def freeStep {P : Type} (cursor : Cursor P) (next : ℝ) (points : P)
    (sweep : P → P → ℝ) (held : Option ℝ) (ignore : ℝ → Bool) : Sum ℝ (Cursor P) :=
  let u := sweep cursor.anchors points
  let time := stamp cursor.clock next u
  if 0 ≤ u ∧ held.all (fun t => decide (time < t)) then
    if ignore u then .inr ⟨next,cursor.anchors⟩ else .inl time
  else .inr ⟨next,points⟩

theorem ignored_segment {P : Type} (c : Cursor P) (next finish : ℝ) (p q : P)
    (sweep : P → P → ℝ) (held : Option ℝ) (ignore : ℝ → Bool)
    (hit : 0 ≤ sweep c.anchors p)
    (earlier : ∀ t ∈ held, stamp c.clock next (sweep c.anchors p) < t)
    (repeatHit : ignore (sweep c.anchors p) = true) :
    freeStep c next p sweep held ignore = .inr ⟨next,c.anchors⟩ ∧
    (match freeStep c next p sweep held ignore with
     | .inl t => Sum.inl t
     | .inr cursor => freeStep cursor finish q sweep none (fun _ => false)) =
       (if 0 ≤ sweep c.anchors q then .inl (stamp next finish (sweep c.anchors q))
        else .inr ⟨finish,q⟩) := by
  have step : freeStep c next p sweep held ignore = .inr ⟨next,c.anchors⟩ := by
    cases held <;> simp_all [freeStep]
  refine ⟨step, ?_⟩
  rw [step]
  simp [freeStep]

end ExactBallpark.Collision.SearchWindow

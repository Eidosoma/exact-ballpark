-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.Discriminant
public import Mathlib.Data.List.TakeWhile

/-! Translational impact history in space.CollisionBallProperties. Times and
velocity additions use logical binary64. Integration is a supplied operation. -/
@[expose] public section
namespace ExactBallpark.Collision.History

def delta : Float := Float.ofBits 0x3f1a36e2eb1c432d

structure Motion where
  pos : FloatPoint
  vel : FloatPoint

structure Location where
  time : Float
  collider : Int
  motion : Motion

structure Path where
  locations : List Location
  startTime : Float
  endTime : Float

structure Impact where
  time : Float
  collider : Int
  dv : FloatPoint

abbrev Integrate := Motion → Float → Motion

def contacts (xs : List Location) : List (Float × Int) :=
  (xs.filter (fun x => x.collider != -1)).map (fun x => (x.time, x.collider))

def kicked (base : Motion) (i : Impact) : Location :=
  ⟨i.time, i.collider, ⟨base.pos, fun axis => base.vel axis + i.dv axis⟩⟩

/-- Scan backwards to the first retained location. The removed suffix is already
in the order in which Python emits the invalidations. -/
def insert (step : Integrate) (initial : Motion) (p : Path) (i : Impact) :
    Path × List (Float × Int) :=
  if p.endTime ≤ i.time then (p, []) else
  let newest := p.locations.reverse
  let later := fun x : Location => decide (¬ x.time ≤ i.time)
  let keep := newest.dropWhile later
  let base := match keep.head? with
    | none => initial
    | some prev => step prev.motion (i.time - prev.time)
  ({p with locations := keep.reverse ++ [kicked base i], endTime := i.time},
    contacts (newest.takeWhile later))

/-- The end-time guard is inclusive. Once accepted, retrying the same comparable
timestamp also leaves the history intact and emits no more invalidations. -/
theorem insert_cutoff_and_retry (step : Integrate) (initial : Motion) (p : Path) (i : Impact)
    (comparable : i.time ≤ i.time) :
    (p.endTime ≤ i.time → insert step initial p i = (p, [])) ∧
    let after := (insert step initial p i).1
    insert step initial after i = (after, []) := by
  constructor
  · intro h; simp [insert, h]
  · by_cases h : p.endTime ≤ i.time
    · simp [insert, h]
    · simp [insert, h, comparable]

theorem insert_after_predecessor (step : Integrate) (initial : Motion) (p : Path) (i : Impact)
    (kept suffix : List Location) (prev : Location)
    (layout : p.locations = kept ++ prev :: suffix) (accept : ¬ p.endTime ≤ i.time)
    (atOrBefore : prev.time ≤ i.time) (later : ∀ x ∈ suffix, ¬ x.time ≤ i.time) :
    insert step initial p i =
      ({p with locations := kept ++ [prev, kicked (step prev.motion (i.time-prev.time)) i], endTime := i.time}, contacts suffix.reverse) := by
  have more : ∀ x ∈ suffix.reverse, decide (¬ x.time ≤ i.time) = true := by simpa using later
  simp only [insert, accept, ↓reduceIte, layout, List.reverse_append, List.reverse_cons,
    List.append_assoc, List.singleton_append]
  rw [List.takeWhile_append_of_pos more, List.dropWhile_append_of_pos more]
  simp [atOrBefore, List.append_assoc]

theorem insert_before_all (step : Integrate) (initial : Motion) (p : Path) (i : Impact)
    (accept : ¬ p.endTime ≤ i.time) (later : ∀ x ∈ p.locations, ¬ x.time ≤ i.time) :
    insert step initial p i =
      ({p with locations := [kicked initial i], endTime := i.time}, contacts p.locations.reverse) := by
  have more : ∀ x ∈ p.locations.reverse, decide (¬ x.time ≤ i.time) = true := by simpa using later
  simp only [insert, accept, ↓reduceIte, List.dropWhile_eq_nil_iff.mpr more, List.takeWhile_eq_self_iff.mpr more]
  rfl

/-- Empty paths have no last location. At full capacity Python signals
DestinyUndefined if extending the tick would invalidate the held C++ reference. -/
def finish (step : Integrate) (tickEnd : Float) (capacity : Nat) (p : Path) : Option Path := do
  let prev ← p.locations.getLast?
  if tickEnd > prev.time then
    if p.locations.length = capacity then none else
    let last : Location := ⟨tickEnd, -1, step prev.motion (tickEnd-prev.time)⟩
    return ⟨p.locations ++ [last], prev.time, tickEnd⟩
  else return p

theorem finish_window (step : Integrate) (tickEnd : Float) (capacity : Nat) (p : Path)
    (prev : Location) (last : p.locations.getLast? = some prev) :
    (¬ tickEnd > prev.time → finish step tickEnd capacity p = some p) ∧
    (tickEnd > prev.time → (finish step tickEnd capacity p = none ↔ p.locations.length = capacity) ∧
      (p.locations.length ≠ capacity → finish step tickEnd capacity p =
        some ⟨p.locations ++ [⟨tickEnd,-1,step prev.motion (tickEnd-prev.time)⟩], prev.time, tickEnd⟩)) := by
  constructor
  · intro h; simp [finish, last, h]
  · intro h; by_cases full : p.locations.length = capacity <;> simp [finish,last,h,full]

def isMatch (time : Float) (collider : Int) (x : Location) :=
  x.collider == collider && decide ((x.time-time).abs < delta)

/-- The source initializes its removal index to zero and resets the first
location both on a first-location match and on a search miss. -/
def remove (step : Integrate) (tickEnd : Float) (capacity : Nat) (initial : Motion)
    (p : Path) (time : Float) (collider : Int) : Option Path × List (Float × Int) :=
  let unmatched := fun x => !(isMatch time collider x)
  let kept := p.locations.takeWhile unmatched
  let rest := p.locations.dropWhile unmatched
  match kept, rest with
  | _ :: _, _ :: later => (finish step tickEnd capacity {p with locations := kept}, contacts later)
  | _, _ =>
    match p.locations with
    | [] => (none, [])
    | first :: later =>
      let reset := {first with collider := -1, motion := {first.motion with vel := initial.vel}}
      (finish step tickEnd capacity {p with locations := [reset]}, contacts later)

theorem remove_first_matching_impact (step : Integrate) (tickEnd : Float) (capacity : Nat)
    (initial : Motion) (p : Path) (time : Float) (collider : Int) (first hit : Location)
    (before later : List Location) (layout : p.locations = (first :: before) ++ hit :: later)
    (earlier : ∀ x ∈ first :: before, isMatch time collider x = false)
    (matchHit : isMatch time collider hit = true) :
    remove step tickEnd capacity initial p time collider =
      (finish step tickEnd capacity {p with locations := first :: before}, contacts later) := by
  have passed : ∀ x ∈ first :: before, (!isMatch time collider x) = true := by simpa using earlier
  simp only [remove]
  rw [layout, List.takeWhile_append_of_pos passed, List.dropWhile_append_of_pos passed]
  simp [matchHit]

theorem remove_reset_at_zero (step : Integrate) (tickEnd : Float) (capacity : Nat)
    (initial : Motion) (p : Path) (time : Float) (collider : Int) (first : Location)
    (later : List Location) (layout : p.locations = first :: later)
    (resetCase : isMatch time collider first = true ∨ ∀ x ∈ p.locations, isMatch time collider x = false) :
    let reset := {first with collider := -1, motion := {first.motion with vel := initial.vel}}
    remove step tickEnd capacity initial p time collider =
      (finish step tickEnd capacity {p with locations := [reset]}, contacts later) := by
  rcases resetCase with hit | miss
  · simp [remove, layout, hit]
  · have passed : ∀ x ∈ p.locations, (!isMatch time collider x) = true := by simpa using miss
    simp only [remove]
    rw [List.takeWhile_eq_self_iff.mpr passed, List.dropWhile_eq_nil_iff.mpr passed, layout]

def ignoreCollision (xs : List Location) (id : Int) (toi : Float) : Bool :=
  match xs.reverse with
  | _ :: anchor :: earlier =>
    if toi > delta then false else
    let lower := anchor.time - delta
    ((anchor :: earlier).takeWhile (fun x => decide (¬ x.time < lower))).any (fun x => x.collider == id)
  | _ => false

/-- The endpoint is skipped and the first stale location stops the scan, even
if the requested collider occurs farther back. The lower bound is inclusive. -/
theorem ignore_recent_colliders (xs : List Location) (id : Int) (toi : Float)
    (last anchor : Location) (recent older : List Location)
    (layout : xs.reverse = last :: anchor :: recent ++ older)
    (inside : ∀ x ∈ anchor :: recent, ¬ x.time < anchor.time - delta)
    (boundary : ∀ x ∈ older.head?, x.time < anchor.time - delta) :
    ignoreCollision xs id toi = true ↔ ¬ toi > delta ∧ ∃ x ∈ anchor :: recent, x.collider = id := by
  have passed : ∀ x ∈ anchor :: recent, decide (¬ x.time < anchor.time-delta) = true := by simpa using inside
  simp only [ignoreCollision, layout, List.cons_append]
  change (if toi > delta then false else
    (((anchor :: recent) ++ older).takeWhile _).any _) = true ↔ _
  rw [List.takeWhile_append_of_pos passed]
  cases older with
  | nil => simp
  | cons x tail =>
    have stop : x.time < anchor.time-delta := boundary x (by simp)
    simp [stop]

def indexAt (times : List Float) (query : Float) : Nat :=
  (times.takeWhile (fun t => decide (t + delta < query))).length - 1

/-- A strict scan stops at the first failed comparison, returning the previous
index or zero when the initial point already fails. -/
theorem index_prefix (passed pending : List Float) (query : Float)
    (before : ∀ t ∈ passed, t + delta < query)
    (stop : ∀ t ∈ pending.head?, ¬ t + delta < query) :
    indexAt (passed ++ pending) query = passed.length - 1 := by
  have h : ∀ t ∈ passed, decide (t+delta < query) = true := by simpa using before
  unfold indexAt
  rw [List.takeWhile_append_of_pos h]
  cases pending with
  | nil => simp
  | cons t tail =>
    have := stop t (by simp)
    simp [this]

def position (step : Integrate) (xs : List Location) (query : Float) : Option Motion := do
  let x ← xs[indexAt (xs.map Location.time) query]?
  return if (x.time-query).abs ≥ delta then step x.motion (query-x.time) else x.motion

theorem position_from_prefix (step : Integrate) (kept : List Location) (x : Location)
    (pending : List Location) (query : Float)
    (selected : ((∀ y ∈ kept ++ [x], y.time + delta < query) ∧
      (∀ y ∈ pending.head?, ¬ y.time + delta < query)) ∨ (kept = [] ∧ ¬ x.time + delta < query)) :
    position step (kept ++ x :: pending) query =
      some (if (x.time-query).abs ≥ delta then step x.motion (query-x.time) else x.motion) := by
  rcases selected with ⟨before,stop⟩ | ⟨rfl,first⟩
  swap
  · simp [position,indexAt,first]
  have cut := index_prefix ((kept ++ [x]).map Location.time) (pending.map Location.time) query
  have hs : ∀ t ∈ (pending.map Location.time).head?, ¬ t + delta < query := by
    cases pending <;> simp_all
  have idx := cut (by
    intro t ht
    obtain ⟨y, hy, rfl⟩ := List.mem_map.mp ht
    exact before y hy) hs
  unfold position
  simp only [List.map_append, List.map_cons, List.map_nil, List.append_assoc, List.cons_append,
    List.nil_append, List.length_map, List.length_append, List.length_cons, List.length_nil] at idx
  simp [List.map_append, idx]

set_option maxRecDepth 8192 in
/-- The rounded sum ties the query. Adding the exact decoded inputs instead
passes the strict guard and selects the newer collision record. -/
theorem rounded_history_lookup :
    let t : Float := 0.0625
    let query := Float.ofBits 0x3fb0068db8bac711
    let times := [0, t, 1]
    indexAt times query = 0 ∧ indexAt times (Float.ofBits 0x3fb0068db8bac712) = 1 ∧
    (do
      let ts ← times.mapM finiteValue
      let d ← finiteValue delta
      let q ← finiteValue query
      return (ts.takeWhile (fun u => decide (u+d < q))).length - 1) = some 1 := by decide +kernel

end ExactBallpark.Collision.History

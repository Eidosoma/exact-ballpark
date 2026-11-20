-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import Mathlib.Data.Finset.Prod
public import Mathlib.Data.List.Sort
public import Init.Data.Float

@[expose] public section
namespace ExactBallpark.Space.Filing

inductive Side where | low | stay | high
  deriving Repr, DecidableEq

def face (lo hi r p : Float) : Side :=
  if (lo+r)-p > 0 then .low else if (hi-r)-p < 0 then .high else .stay

abbrev Key := Nat × Nat × Nat

def shift (k : Nat) : Side → Nat
  | .low => k-1
  | .stay => k
  | .high => k+1

/-- The ordinary scalar insert calls, before converting the box keys to a set. -/
def inserts (k : Key) (x y z : Side) : List Key :=
  let a := shift k.1 x
  let b := shift k.2.1 y
  let c := shift k.2.2 z
  [k] ++ (if x != .stay then [(a,k.2)] else []) ++
    (if y != .stay then [(k.1,b,k.2.2)] else []) ++
    (if z != .stay then [(k.1,k.2.1,c)] else []) ++
    (if x != .stay && y != .stay then [(a,b,k.2.2)] else []) ++
    (if x != .stay && z != .stay then [(a,k.2.1,c)] else []) ++
    (if y != .stay && z != .stay then [(k.1,b,c)] else []) ++
    (if x != .stay && y != .stay && z != .stay then [(a,b,c)] else [])

def choices (k : Nat) : Side → List Nat
  | .low => [k-1,k]
  | .stay => [k]
  | .high => [k,k+1]

def grid (xs ys zs : List Nat) : List Key :=
  xs.flatMap (fun x => ys.flatMap (fun y => zs.map (fun z => (x,y,z))))

def boxes (k : Key) (x y z : Side) :=
  grid (choices k.1 x) (choices k.2.1 y) (choices k.2.2 z)

private theorem grid_mem (k : Key) (xs ys zs : List Nat) :
    k ∈ grid xs ys zs ↔ k.1 ∈ xs ∧ k.2.1 ∈ ys ∧ k.2.2 ∈ zs := by
  rcases k with ⟨x,y,z⟩
  simp [grid]

private theorem choice_mem (i k : Nat) (s : Side) :
    i ∈ choices k s ↔ i = k ∨ (s ≠ .stay ∧ i = shift k s) := by
  cases s <;> simp [choices,shift,or_comm]

private theorem choice_nodup (k : Nat) (s : Side) (hk : 0 < k) : (choices k s).Nodup := by
  cases k <;> cases s <;> simp_all [choices]

private theorem insert_cells (k : Key) (x y z : Side) :
    (inserts k x y z).toFinset = (boxes k x y z).toFinset := by
  obtain ⟨a,b,c⟩ := k
  ext ⟨i,j,l⟩
  simp only [List.mem_toFinset, boxes, grid_mem, choice_mem]
  by_cases hx : x = .stay <;> by_cases hy : y = .stay <;> by_cases hz : z = .stay <;>
    simp [inserts,hx,hy,hz] <;> tauto

/-- Membership, count, and the strict contact tests. Interior keys ensure the
requested neighbors exist as distinct cells, without the scalar clamp. -/
theorem face_edge_corner (k : Key) (x y z : Side) (lo hi r p : Float)
    (interior : 0 < k.1 ∧ 0 < k.2.1 ∧ 0 < k.2.2) :
    (inserts k x y z).toFinset = (boxes k x y z).toFinset ∧
    (inserts k x y z).toFinset.card =
      (choices k.1 x).length * (choices k.2.1 y).length * (choices k.2.2 z).length ∧
    (face lo hi r p = .low ↔ (lo+r)-p > 0) ∧
    (face lo hi r p = .high ↔ ¬ (lo+r)-p > 0 ∧ (hi-r)-p < 0) ∧
    (face lo hi r p = .stay ↔ ¬ (lo+r)-p > 0 ∧ ¬ (hi-r)-p < 0) := by
  have low : face lo hi r p = .low ↔ (lo+r)-p > 0 := by
    by_cases hl : (lo+r)-p > 0 <;> by_cases hh : (hi-r)-p < 0 <;> simp [face,hl,hh]
  have high : face lo hi r p = .high ↔ ¬ (lo+r)-p > 0 ∧ (hi-r)-p < 0 := by
    unfold face
    split <;> simp_all
  have guards : face lo hi r p = .stay ↔ ¬ (lo+r)-p > 0 ∧ ¬ (hi-r)-p < 0 := by
    unfold face
    split <;> simp_all
  have product : (boxes k x y z).toFinset =
      (choices k.1 x).toFinset ×ˢ ((choices k.2.1 y).toFinset ×ˢ (choices k.2.2 z).toFinset) := by
    ext pt
    simp [boxes,grid_mem]
  refine ⟨insert_cells .., ?_, low, high, guards⟩
  rw [insert_cells,product,Finset.card_product,Finset.card_product,
    List.toFinset_card_of_nodup (choice_nodup _ _ interior.1),
    List.toFinset_card_of_nodup (choice_nodup _ _ interior.2.1),
    List.toFinset_card_of_nodup (choice_nodup _ _ interior.2.2),Nat.mul_assoc]

structure Footprint where
  key : Key
  level : Nat
  x : Side
  y : Side
  z : Side
  deriving DecidableEq

def Footprint.cells (f : Footprint) := boxes f.key f.x f.y f.z

def ancestor (k : Key) (levels : Nat) : Key :=
  (k.1 >>> (2*levels), k.2.1 >>> (2*levels), k.2.2 >>> (2*levels))

def ends (k : Nat) (s : Side) :=
  (min k (shift k s), max k (shift k s))

def axisRelated (delta a : Nat) (x : Side) (b : Nat) (y : Side) : Bool :=
  let (lo,hi) := ends a x
  let (l,h) := ends b y
  let s := 2*delta
  (lo >>> s == l) || (lo >>> s == h) || (hi >>> s == l) || (hi >>> s == h)

def finerRelated (f c : Footprint) : Bool :=
  let d := f.level-c.level
  axisRelated d f.key.1 f.x c.key.1 c.x &&
  axisRelated d f.key.2.1 f.y c.key.2.1 c.y &&
  axisRelated d f.key.2.2 f.z c.key.2.2 c.z

def related (a b : Footprint) := if b.level ≤ a.level then finerRelated a b else finerRelated b a

def reaches (f c : Footprint) := ∃ k ∈ f.cells, ancestor k (f.level-c.level) ∈ c.cells

private theorem ends_mem (i k : Nat) (s : Side) :
    i ∈ choices k s ↔ i = (ends k s).1 ∨ i = (ends k s).2 := by
  cases s <;> simp [choices,ends,shift,Nat.sub_le]

private theorem axis_correct (d a : Nat) (x : Side) (b : Nat) (y : Side) :
    axisRelated d a x b y ↔ ∃ i ∈ choices a x, i >>> (2*d) ∈ choices b y := by
  simp only [axisRelated, Bool.or_eq_true, beq_iff_eq, ends_mem]
  grind

private theorem finer_correct (f c : Footprint) : finerRelated f c ↔ reaches f c := by
  simp only [finerRelated, Bool.and_eq_true, axis_correct, reaches, Footprint.cells, boxes,grid_mem, ancestor]
  constructor
  · rintro ⟨⟨⟨i,hi,hi'⟩,⟨j,hj,hj'⟩⟩,⟨k,hk,hk'⟩⟩
    exact ⟨(i,j,k),⟨hi,hj,hk⟩,hi',hj',hk'⟩
  · rintro ⟨⟨i,j,k⟩,⟨hi,hj,hk⟩,hi',hj',hk'⟩
    exact ⟨⟨⟨i,hi,hi'⟩,⟨j,hj,hj'⟩⟩,⟨k,hk,hk'⟩⟩

theorem related_iff (a b : Footprint) : related a b ↔
    (if b.level ≤ a.level then reaches a b else reaches b a) := by
  unfold related
  split <;> exact finer_correct ..

end ExactBallpark.Space.Filing

-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Space.Filing
public import Mathlib.Data.Finset.Sort
public import Mathlib.Order.Prod.Lex.Basic

@[expose] public section
namespace ExactBallpark.Space.CandidateOrder
open Filing

/-- The handled flag in GetNearbyBalls, after the eligibility filters. -/
def walk {α : Type} (id : α → Nat) (seen : Finset Nat) : List α → List α
  | [] => []
  | a :: xs => if id a ∈ seen then walk id seen xs
               else a :: walk id (insert (id a) seen) xs

private theorem walk_mem {α : Type} (id : α → Nat) (seen : Finset Nat) (xs : List α) (i : Nat) :
    i ∈ (walk id seen xs).map id ↔ i ∈ xs.map id ∧ i ∉ seen := by
  induction xs generalizing seen with
  | nil => simp [walk]
  | cons a xs ih =>
    by_cases h : id a ∈ seen <;> simp [walk,h,ih] <;> grind

private theorem walk_nodup {α : Type} (id : α → Nat) (seen : Finset Nat) (xs : List α) :
    ((walk id seen xs).map id).Nodup := by
  induction xs generalizing seen with
  | nil => simp [walk]
  | cons a xs ih =>
    by_cases h : id a ∈ seen <;> simp [walk,h,ih,walk_mem]

private theorem walk_append {α : Type} (id : α → Nat) (seen : Finset Nat) (xs ys : List α) :
    walk id seen (xs ++ ys) = walk id seen xs ++ walk id (seen ∪ (xs.map id).toFinset) ys := by
  induction xs generalizing seen with
  | nil => simp [walk]
  | cons a xs ih =>
    by_cases h : id a ∈ seen
    · have h' : id a ∈ seen ∪ (xs.map id).toFinset := Finset.mem_union_left _ h
      simp [walk,h,ih,Finset.union_insert,Finset.insert_eq_of_mem h']
    · simp [walk,h,ih,Finset.insert_union,Finset.union_insert]

theorem first_visit_once {α : Type} (id : α → Nat) (xs ys : List α) (a : α)
    (first : id a ∉ xs.map id) :
    ((walk id ∅ (xs ++ a::ys)).map id).Nodup ∧
    (∀ i, i ∈ (walk id ∅ (xs ++ a::ys)).map id ↔ i ∈ (xs ++ a::ys).map id) ∧
    walk id ∅ (xs ++ a::ys) = walk id ∅ xs ++ a :: walk id (insert (id a) (xs.map id).toFinset) ys := by
  exact ⟨walk_nodup .., fun i => by simp [walk_mem], by rw [walk_append]; simp [walk,first]⟩

/-- Scalar traversal address in nested lexicographic order: own box number,
ancestor/descendant/self, position in that part, ascending ball id. -/
abbrev Visit := Nat ×ₗ Fin 3 ×ₗ Fin (65^7) ×ₗ Nat
abbrev Packed := Nat ×ₗ Nat

def visit (seg : Nat) (part : Fin 3) (sub : Fin (65^7)) (id : Nat) : Visit :=
  toLex (seg, toLex (part, toLex (sub,id)))

def ballId (x : Visit) := (ofLex (ofLex (ofLex x).2).2).2

def pack (x : Visit) : Packed :=
  let (seg, rest) := ofLex x
  let (part, tail) := ofLex rest
  let (sub, id) := ofLex tail
  toLex ((seg*3+part.val)*65^7+sub.val, id)

private theorem pack_strict : StrictMono pack := by
  intro x y h
  have hp := (ofLex (ofLex x).2).1.isLt
  have hq := (ofLex (ofLex y).2).1.isLt
  have hk := (ofLex (ofLex (ofLex x).2).2).1.isLt
  have hl := (ofLex (ofLex (ofLex y).2).2).1.isLt
  simp only [pack,Prod.Lex.lt_iff,ofLex_toLex,Fin.lt_def,Fin.ext_iff] at h ⊢
  omega

/-- The highest tree level is zero. At each of the seven remaining levels,
64 follows all 64 children, so a subtree is emitted before its parent. -/
def pathDigit (own other : Nat) (box : Key) (level : Nat) : Fin 65 :=
  if level ≤ own then 0 else if other < level then 64 else
    let k := ancestor box (other-level)
    ⟨k.1 % 4 * 16 + k.2.1 % 4 * 4 + k.2.2 % 4, by omega⟩

def digitsValue : (ds : List (Fin 65)) → Fin (65^ds.length)
  | [] => 0
  | d :: ds =>
    let rest := digitsValue ds
    ⟨d.val * 65^ds.length + rest.val, by
      have bound := Nat.mul_le_mul_right (65^ds.length) (Nat.succ_le_of_lt d.isLt)
      simp only [List.length_cons, Nat.pow_succ, Nat.succ_mul] at *
      omega⟩

private theorem digits_compare (xs ys : List (Fin 65)) (size : xs.length = ys.length) :
    (digitsValue xs).val < (digitsValue ys).val ↔ xs < ys := by
  induction xs generalizing ys with
  | nil => cases ys <;> simp_all [digitsValue]
  | cons a xs ih =>
    cases ys with
    | nil => simp_all
    | cons b ys =>
      have same := Nat.succ.inj size
      have hx := (digitsValue xs).isLt
      have hy := (digitsValue ys).isLt
      have left := fun (h : a < b) =>
        Nat.mul_le_mul_right (65^xs.length) (Nat.succ_le_of_lt h : a.val+1 ≤ b.val)
      have right := fun (h : b < a) =>
        Nat.mul_le_mul_right (65^xs.length) (Nat.succ_le_of_lt h : b.val+1 ≤ a.val)
      simp only [Nat.succ_mul] at left right
      have weights := congrArg (fun n : Nat => 65^n) same
      have rightWeight := congrArg (fun n : Nat => b.val*65^n) same.symm
      change a.val * 65^xs.length + (digitsValue xs).val <
        b.val * 65^ys.length + (digitsValue ys).val ↔ _
      rw [rightWeight,List.cons_lt_cons_iff]
      rcases lt_trichotomy a b with h | rfl | h
      · exact iff_of_true (by have := left h; omega) (Or.inl h)
      · simp [ih _ same]
      · exact iff_of_false (by have := right h; omega) (by simp [h.ne',not_lt_of_gt h])

def postOrder (own other : Nat) (box : Key) : Fin (65^7) :=
  digitsValue ([1,2,3,4,5,6,7].map (pathDigit own other box))

def incidence (query b : Footprint) (valid : query.level < 8) (cell box : Key) (seg id : Nat) :
    Option Visit :=
  if b.level < query.level then
    if ancestor cell (query.level-b.level) = box then
      some (visit seg 0 ⟨query.level-b.level, by omega⟩ id) else none
  else if b.level = query.level then
    if cell = box then some (visit seg 2 0 id) else none
  else if ancestor box (b.level-query.level) = cell then
    some (visit seg 1 (postOrder query.level b.level box) id) else none

/-- All incidences, allowing a ball to be found in multiple cells. The caller
supplies the eligible other balls and ordinary filings at engine levels 0..7. -/
def occurrences (query : Footprint) (valid : query.level < 8) (others : List (Nat × Footprint)) : Finset Visit :=
  (query.cells.zipIdx.flatMap fun (cell,seg) => others.flatMap fun (id,b) =>
    b.cells.filterMap fun box => incidence query b valid cell box seg id).toFinset

private theorem incidence_id (q b : Footprint) (valid : q.level < 8) (cell box : Key) (seg id i : Nat) :
    (∃ v, incidence q b valid cell box seg id = some v ∧ ballId v = i) ↔ id = i ∧
    (if b.level ≤ q.level then ancestor cell (q.level-b.level) = box
     else ancestor box (b.level-q.level) = cell) := by
  rcases lt_trichotomy b.level q.level with h | h | h
  · by_cases meet : ancestor cell (q.level-b.level) = box <;> simp [incidence,h,h.le,meet,ballId,visit]
  · by_cases meet : cell = box <;> simp [incidence,h,ancestor,meet,ballId,visit]
  · by_cases meet : ancestor box (b.level-q.level) = cell <;>
      simp [incidence,h.ne',not_le_of_gt h,not_lt_of_gt h,meet,ballId,visit]

private theorem occurrence_ids (q : Footprint) (valid : q.level < 8) (others : List (Nat × Footprint)) (i : Nat) :
    (∃ v ∈ occurrences q valid others, ballId v = i) ↔ ∃ b, (i,b) ∈ others ∧ related q b := by
  have index {cell : Key} (hc : cell ∈ q.cells) : ∃ n, (cell,n) ∈ q.cells.zipIdx := by
    obtain ⟨n,hn⟩ := List.mem_iff_getElem?.mp hc
    exact ⟨n,List.mk_mem_zipIdx_iff_getElem?.mpr hn⟩
  simp only [occurrences,List.mem_toFinset,List.mem_flatMap,List.mem_filterMap]
  constructor
  · rintro ⟨v,⟨⟨cell,seg⟩,hseg,⟨id,b⟩,hb,box,hbox,found⟩,eq⟩
    obtain ⟨rfl,contact⟩ := (incidence_id q b valid cell box seg id i).mp ⟨v,found,eq⟩
    refine ⟨b,hb,(related_iff ..).mpr ?_⟩
    have hc : cell ∈ q.cells := List.fst_mem_of_mem_zipIdx hseg
    by_cases h : b.level ≤ q.level <;> simp only [h,ite_true,ite_false] at contact ⊢
    · exact ⟨cell,hc,contact ▸ hbox⟩
    · exact ⟨box,hbox,contact ▸ hc⟩
  · rintro ⟨b,hb,rel⟩
    have geom := (related_iff q b).mp rel
    split at geom
    next h =>
      obtain ⟨cell,hc,hit⟩ := geom
      obtain ⟨seg,hs⟩ := index hc
      obtain ⟨v,hv,hi⟩ := (incidence_id q b valid cell (ancestor cell (q.level-b.level)) seg i i).mpr
        ⟨rfl,by simp [h]⟩
      exact ⟨v,⟨(cell,seg),hs,(i,b),hb,_,hit,hv⟩,hi⟩
    next h =>
      obtain ⟨box,hc,hit⟩ := geom
      obtain ⟨seg,hs⟩ := index hit
      obtain ⟨v,hv,hi⟩ := (incidence_id q b valid (ancestor box (b.level-q.level)) box seg i i).mpr
        ⟨rfl,by simp [h]⟩
      exact ⟨v,⟨(_,seg),hs,(i,b),hb,box,hc,hv⟩,hi⟩

private theorem below_sentinel (q : Footprint) (valid : q.level < 8) (others : List (Nat × Footprint))
    (v : Visit) (member : v ∈ occurrences q valid others) : (ofLex (pack v)).1 < 2^62 := by
  have eight : q.cells.length ≤ 8 := by
    obtain ⟨k,lev,x,y,z⟩ := q
    cases x <;> cases y <;> cases z <;> simp [Footprint.cells,boxes,grid,choices]
  simp only [occurrences,List.mem_toFinset,List.mem_flatMap,List.mem_filterMap] at member
  obtain ⟨⟨cell,seg⟩,mem,⟨id,b⟩,_,box,_,found⟩ := member
  have segSmall := (List.mem_zipIdx' mem).1.trans_le eight
  have bound (p : Fin 3) (k : Fin (65^7)) :
      (ofLex (pack (visit seg p k id))).1 < 2^62 := by
    change (seg*3+p.val)*65^7+k.val < 2^62
    have := p.isLt
    have := k.isLt
    omega
  unfold incidence at found
  split_ifs at found <;> cases found <;> exact bound _ _

private def winners {α : Type} [LinearOrder α] (id : α → Nat) (s : Finset α) :=
  s.filter (fun x => ∀ y ∈ s, id y = id x → x ≤ y)

private theorem walk_minima {α : Type} [LinearOrder α] (id : α → Nat) (seen : Finset Nat)
    (xs : List α) (ordered : xs.Pairwise (· < ·)) :
    walk id seen xs = xs.filter (fun x => decide (id x ∉ seen ∧ ∀ y ∈ xs, id y = id x → x ≤ y)) := by
  induction xs generalizing seen with
  | nil => simp [walk]
  | cons a xs ih =>
    obtain ⟨before,tail⟩ := List.pairwise_cons.mp ordered
    have amin : ∀ y ∈ a::xs, id y = id a → a ≤ y := by
      intro y hy _
      rcases List.mem_cons.mp hy with rfl | h
      · exact le_rfl
      · exact (before _ h).le
    by_cases ha : id a ∈ seen
    · simp only [walk, ha, ite_true, List.filter_cons, not_true_eq_false, false_and,
        decide_false, Bool.false_eq_true, ite_false]
      rw [ih seen tail]
      apply List.filter_congr
      intro x hx
      simp only [List.forall_mem_cons]
      by_cases hi : id a = id x <;> simp_all
    · have test : decide (id a ∉ seen ∧ ∀ y ∈ a::xs, id y = id a → a ≤ y) = true :=
        decide_eq_true ⟨ha,amin⟩
      rw [List.filter_cons_of_pos (p := fun x => decide (id x ∉ seen ∧ ∀ y ∈ a::xs, id y = id x → x ≤ y)) test]
      simp only [walk,ha,ite_false,List.cons.injEq,true_and]
      rw [ih _ tail]
      apply List.filter_congr
      intro x hx
      have late := (before x hx).not_ge
      simp only [Finset.mem_insert,List.forall_mem_cons]
      by_cases hi : id a = id x
      · simp_all
      · simp_all [Ne.symm hi]

/-- Batched reduction: for each distinct eligible ID take its smallest packed
address, then sort the winning (address, id) rows. -/
def batchWinners (s : Finset Packed) : Finset Packed :=
  (s.image (fun x => (ofLex x).2)).attach.image fun i =>
    let rows := s.filter (fun r => (ofLex r).2 = i.val)
    rows.min' (by
      obtain ⟨x,hx,h⟩ := Finset.mem_image.mp i.property
      exact ⟨x,Finset.mem_filter.mpr ⟨hx,h⟩⟩)

private theorem batch_mem (s : Finset Packed) (x : Packed) :
    x ∈ batchWinners s ↔ x ∈ s ∧ ∀ y ∈ s, (ofLex y).2 = (ofLex x).2 → x ≤ y := by
  constructor
  · intro hx
    obtain ⟨i,_,rfl⟩ := Finset.mem_image.mp hx
    let rows := s.filter (fun r => (ofLex r).2 = i.val)
    have ne : rows.Nonempty := by
      obtain ⟨z,hz,h⟩ := Finset.mem_image.mp i.property
      exact ⟨z,Finset.mem_filter.mpr ⟨hz,h⟩⟩
    change rows.min' ne ∈ s ∧ _
    obtain ⟨hs,key⟩ := Finset.mem_filter.mp (rows.min'_mem ne)
    exact ⟨hs, fun y hy h => rows.min'_le y (Finset.mem_filter.mpr ⟨hy,h.trans key⟩)⟩
  · rintro ⟨hx,low⟩
    let i : {i // i ∈ s.image (fun x => (ofLex x).2)} :=
      ⟨(ofLex x).2,Finset.mem_image.mpr ⟨x,hx,rfl⟩⟩
    refine Finset.mem_image.mpr ⟨i,Finset.mem_attach _ _,?_⟩
    let rows := s.filter (fun r => (ofLex r).2 = i.val)
    have mem : x ∈ rows := Finset.mem_filter.mpr ⟨hx,rfl⟩
    change rows.min' ⟨x,mem⟩ = x
    obtain ⟨hs,key⟩ := Finset.mem_filter.mp (rows.min'_mem ⟨x,mem⟩)
    exact le_antisymm (rows.min'_le _ mem) (low _ hs key)

private theorem walk_sorted {α : Type} [LinearOrder α] (id : α → Nat) (s : Finset α) :
    walk id ∅ s.sort = (winners id s).sort := by
  rw [walk_minima _ _ _ s.sortedLT_sort.pairwise]
  let keep := fun (x : α) => decide (id x ∉ (∅ : Finset Nat) ∧ ∀ y ∈ s.sort (· ≤ ·), id y = id x → x ≤ y)
  change s.sort.filter keep = _
  apply List.Perm.eq_of_pairwise' (r := (· ≤ ·))
  · exact (s.pairwise_sort _).filter _
  · exact Finset.pairwise_sort _ _
  · apply (List.perm_ext_iff_of_nodup ((s.sort_nodup _).filter _) (Finset.sort_nodup _ _)).mpr
    intro x
    simp [keep,winners]

def scalarOrder (s : Finset Visit) := (walk ballId ∅ s.sort).map ballId
def batchOrder (s : Finset Visit) :=
  ((batchWinners (s.image pack)).sort).map (fun r => (ofLex r).2)

/-- Complete ordered lists on common ordinary filings. The scalar convention
orders segments, tree parts and box visits separately, suppressing previous IDs.
The batched convention minimizes the packed key per ID and breaks ties by ID. -/
theorem ordered_candidates (query : Footprint) (valid : query.level < 8) (others : List (Nat × Footprint)) :
    let s := occurrences query valid others
    scalarOrder s = batchOrder s ∧
    (∀ i, i ∈ scalarOrder s ↔ ∃ b, (i,b) ∈ others ∧ related query b) ∧
    (∀ v ∈ s, (ofLex (pack v)).1 < 2^62) ∧
    ∀ xs ys : List (Fin 65), xs.length = ys.length →
      ((digitsValue xs).val < (digitsValue ys).val ↔ xs < ys) := by
  dsimp only
  refine ⟨?_,?_,below_sentinel query valid others,digits_compare⟩
  swap
  · intro i
    simpa [scalarOrder,walk_mem] using occurrence_ids query valid others i
  generalize occurrences query valid others = s
  let emb : Visit ↪ Packed := ⟨pack,pack_strict.injective⟩
  have keys (x : Visit) : (ofLex (pack x)).2 = ballId x := rfl
  have same : (winners ballId s).map emb = batchWinners (s.image pack) := by
    ext y
    rw [batch_mem,Finset.mem_map]
    constructor
    · rintro ⟨x,hx,rfl⟩
      obtain ⟨mem,low⟩ := Finset.mem_filter.mp hx
      refine ⟨Finset.mem_image.mpr ⟨x,mem,rfl⟩,?_⟩
      rintro _ hz eq
      obtain ⟨z,member,rfl⟩ := Finset.mem_image.mp hz
      exact pack_strict.monotone (low _ member eq)
    · rintro ⟨hy,low⟩
      obtain ⟨x,hx,rfl⟩ := Finset.mem_image.mp hy
      refine ⟨x,Finset.mem_filter.mpr ⟨hx,?_⟩,rfl⟩
      intro z hz eq
      exact pack_strict.le_iff_le.mp (low _ (Finset.mem_image.mpr ⟨z,hz,rfl⟩) eq)
  have sorted := Finset.map_sort emb (winners ballId s) (· ≤ ·) (· ≤ ·)
    (fun _ _ _ _ => pack_strict.le_iff_le.symm)
  rw [scalarOrder,batchOrder,walk_sorted,← same,← sorted,List.map_map]
  rfl

end ExactBallpark.Space.CandidateOrder

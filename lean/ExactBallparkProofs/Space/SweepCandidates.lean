-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Space.CandidateOrder
public import ExactBallparkProofs.Space.FilingRadius

@[expose] public noncomputable section
namespace ExactBallpark.Space.SweepCandidates
open Filing CandidateOrder
open ExactBallpark.Movement

/-- Coordinates and signed face-test results in local cell-width units, with
the grid origin translated to zero. A level can be four times coarser. -/
structure Axis where
  home : Nat
  center : ℝ
  reach : ℝ
  error : ℝ
  lowTest : ℝ
  highTest : ℝ

def Axis.side (a : Axis) : Side :=
  if a.lowTest > 0 then .low else if a.highTest < 0 then .high else .stay

def Axis.valid (a : Axis) : Prop :=
  a.home ≤ a.center ∧ a.center < a.home+1 ∧ 0 ≤ a.error ∧ 3*a.reach ≤ 1 ∧
  |a.lowTest - (a.home+a.reach-a.center)| ≤ a.error ∧
  |a.highTest - (a.home+1-a.reach-a.center)| ≤ a.error

private theorem axis_member (a : Axis) (valid : a.valid) {j : Nat} {z : ℝ}
    (left : (j:ℝ) ≤ z) (right : z < (j:ℝ)+1) (inside : |z-a.center| + a.error < a.reach) :
    j ∈ choices a.home a.side := by
  obtain ⟨p0,p1,err,size,lo,hi⟩ := valid
  obtain ⟨l0,l1⟩ := abs_le.mp lo
  obtain ⟨h0,h1⟩ := abs_le.mp hi
  have near := abs_lt.mp (show |z-a.center| < a.reach-a.error by linarith)
  rcases lt_trichotomy j a.home with below | eq | above
  · have adj : a.home ≤ j+1 := by
      by_contra! far
      have gap : (j:ℝ)+2 ≤ a.home := by exact_mod_cast (show j+2 ≤ a.home by omega)
      linarith
    have before : (j:ℝ)+1 ≤ a.home := by exact_mod_cast below
    have low : 0 < a.lowTest := by linarith
    have cell : j = a.home-1 := by omega
    simp [Axis.side,choices,low,cell]
  · subst j; cases a.side <;> simp [choices]
  · have adj : j ≤ a.home+1 := by
      by_contra! far
      have gap : (a.home:ℝ)+2 ≤ j := by exact_mod_cast (show a.home+2 ≤ j by omega)
      linarith
    have afterFace : (a.home:ℝ)+1 ≤ j := by exact_mod_cast above
    have noLow : ¬ 0 < a.lowTest := by linarith
    have high : a.highTest < 0 := by linarith
    have cell : j = a.home+1 := by omega
    simp [Axis.side,choices,noLow,high,cell]

structure Body where
  axes : Fin 3 → Axis
  level : Nat

def key3 (k : Fin 3 → Nat) : Key := (k 0,k 1,k 2)
def Body.foot (b : Body) : Footprint :=
  ⟨key3 (fun i => (b.axes i).home),b.level,(b.axes 0).side,(b.axes 1).side,(b.axes 2).side⟩

/-- At the common time, the witness is inside the physical body with the
face-rounding margin. Its centre's displacement plus radius fits the filing
envelope. These are per-coordinate consequences of the Euclidean travel and
strict overlap bounds, in units of this body's own box width. -/
def Body.contains (b : Body) (z : Fin 3 → ℝ) : Prop :=
  ∃ location : Fin 3 → ℝ, ∃ radius : ℝ, 0 ≤ radius ∧ ∀ i,
    (b.axes i).valid ∧ |z i-location i| + (b.axes i).error < radius ∧
    |location i-(b.axes i).center| + radius ≤ (b.axes i).reach

/-- In this body's cell-width units, either a direct displacement envelope is
available (including a fixed sphere), or the ordinary drag trajectory has the
source filing radius and the speed and thrust bounds that cover this flight. -/
def Body.coversSweep (b : Body) (z : Fin 3 → ℝ) : Prop := b.contains z ∨
  ∃ (v a : Vec3) (m k S dt t radius q : ℝ),
    0 < m ∧ 0 < k ∧ 0 ≤ t ∧ t ≤ dt ∧ 0 ≤ radius ∧ q = decay m k dt ∧
    ‖v‖ ≤ S ∧ ‖terminal3 a m k‖ ≤ S ∧ ∀ i, (b.axes i).valid ∧
    |z i - ((b.axes i).center + traveled v a m k t i)| + (b.axes i).error < radius ∧
    filedRadius radius ‖v‖ m k S q dt ≤ (b.axes i).reach

private theorem contains_of_sweep {b : Body} {z : Fin 3 → ℝ} (sweep : b.coversSweep z) :
    b.contains z := by
  rcases sweep with direct | ⟨v,a,m,k,S,dt,t,r,q,hm,hk,t0,ht,hr,hq,hv,ha,axes⟩
  · exact direct
  let displacement := traveled v a m k t
  refine ⟨fun i => (b.axes i).center + displacement i,r,hr,fun i => ⟨(axes i).1,(axes i).2.1,?_⟩⟩
  rw [add_sub_cancel_left]
  have component : |displacement i| ≤ ‖displacement‖ := PiLp.norm_apply_le displacement i
  exact (add_le_add component (le_refl r)).trans
    ((radius_covers_trajectory v a m k S dt t r q hm hk t0 ht hq hv ha).trans (axes i).2.2)

private theorem in_boxes (b : Body) {z : Fin 3 → ℝ} {j : Fin 3 → Nat}
    (inside : b.contains z) (cell : ∀ i, (j i:ℝ) ≤ z i ∧ z i < (j i:ℝ)+1) :
    key3 j ∈ b.foot.cells := by
  obtain ⟨position,r,_,bounds⟩ := inside
  have axis (i : Fin 3) := axis_member (b.axes i) (bounds i).1 (cell i).1 (cell i).2 (by
    linarith [(bounds i).2.1,(bounds i).2.2,abs_sub_le (z i) (position i) (b.axes i).center])
  simpa [key3,Body.foot,Footprint.cells,boxes,grid] using And.intro (axis 0) ⟨axis 1,axis 2⟩

private theorem parent_cell {z : ℝ} {j f : Nat} (pos : 0 < f)
    (lo : (j:ℝ) ≤ z) (hi : z < (j:ℝ)+1) :
    ((j/f:Nat):ℝ) ≤ z/f ∧ z/f < ((j/f:Nat):ℝ)+1 := by
  have fp : (0:ℝ) < f := by exact_mod_cast pos
  have l : j / f * f ≤ j := Nat.div_mul_le_self ..
  have u : j+1 ≤ (j/f+1)*f := by
    have h := Nat.mod_lt j pos
    have e := Nat.div_add_mod j f
    rw [Nat.add_mul,Nat.one_mul,Nat.mul_comm (j/f)]
    omega
  rw [le_div_iff₀ fp,div_lt_iff₀ fp]
  exact ⟨(by exact_mod_cast l : ((j/f:Nat):ℝ)*f ≤ j).trans lo,
    hi.trans_le (by exact_mod_cast u)⟩

/-- The two bodies share a physical point at their respective resolutions. The
spatial search visits the other eligible body whether the query is finer or
coarser. No common filed cell or collision-list membership is assumed. -/
theorem sweep_candidate (fine coarse : Body) (hf : fine.level < 8) (hc : coarse.level < 8)
    (levels : coarse.level ≤ fine.level) (z : Fin 3 → ℝ) (interior : ∀ i, 0 ≤ z i)
    (physicalFine : fine.coversSweep z)
    (physicalCoarse : coarse.coversSweep (fun i => z i / (2:ℝ)^(2*(fine.level-coarse.level))))
    (others : List (Nat × Footprint)) (id : Nat) :
    ((id,coarse.foot) ∈ others → id ∈ scalarOrder (occurrences fine.foot hf others) ∧
      id ∈ batchOrder (occurrences fine.foot hf others)) ∧
    ((id,fine.foot) ∈ others → id ∈ scalarOrder (occurrences coarse.foot hc others) ∧
      id ∈ batchOrder (occurrences coarse.foot hc others)) := by
  let k := fun i => ⌊z i⌋₊
  have cell (i : Fin 3) : (k i:ℝ) ≤ z i ∧ z i < (k i:ℝ)+1 :=
    ⟨Nat.floor_le (interior i), Nat.lt_floor_add_one _⟩
  have fineBox := in_boxes fine (contains_of_sweep physicalFine) cell
  let d := fine.level-coarse.level
  let f := 2^(2*d)
  have positive : 0 < f := by dsimp [f]; positivity
  have coCell (i : Fin 3) := parent_cell positive (cell i).1 (cell i).2
  have coarseBox : ancestor (key3 k) d ∈ coarse.foot.cells := by
    have member := in_boxes coarse (contains_of_sweep physicalCoarse) (j := fun i => k i / f) (fun i => by
      simpa [f,d] using coCell i)
    simpa [ancestor,key3,Nat.shiftRight_eq_div_pow,f] using member
  have reach : reaches fine.foot coarse.foot := ⟨_,fineBox,coarseBox⟩
  have rel : related fine.foot coarse.foot := (related_iff ..).mpr (by
    simpa [Body.foot,levels] using reach)
  have reverse : related coarse.foot fine.foot := (related_iff ..).mpr (by
    by_cases eq : fine.level = coarse.level
    · have same : d=0 := by dsimp [d]; omega
      simp [same,ancestor,key3] at coarseBox
      simp only [Body.foot,eq,le_refl,↓reduceIte,reaches,Nat.sub_self,ancestor,Nat.mul_zero,
        Nat.shiftRight_zero]
      exact ⟨key3 k,coarseBox,fineBox⟩
    · simpa [Body.foot,show ¬fine.level ≤ coarse.level by omega] using reach)
  rw [← (ordered_candidates fine.foot hf others).1,← (ordered_candidates coarse.foot hc others).1]
  simp only [and_self]
  exact ⟨fun mem => ((ordered_candidates fine.foot hf others).2.1 id).mpr ⟨_,mem,rel⟩,
    fun mem => ((ordered_candidates coarse.foot hc others).2.1 id).mpr ⟨_,mem,reverse⟩⟩

end ExactBallpark.Space.SweepCandidates

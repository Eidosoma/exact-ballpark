-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Numerics.Product
public import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Odd

@[expose] public noncomputable section
namespace ExactBallpark.Numerics.Compensated
open FloatLib FloatLib.Floats.Formats.Flocq
local instance : ValidExp (flxExp 53) := validExp_flx 53 (by decide)

def gridAt (x : ℝ) := bpow Numerics.binaryRadix (cexp Numerics.binaryRadix (flxExp 53) x)

private theorem rnGrid (x h : ℝ) (pos : 0 < h) (scale : gridAt x = h) :
    rn x = roundAtScale nearestEven h pos x := by
  subst h
  simp only [rn,Floats.Formats.Flocq.round,toReal,scaledMantissa,roundAtScale,gridAt,bpow.neg_exp,div_eq_mul_inv]

/-- A sign-magnitude word for the rounded grid index k has an even offset
(base includes exponent and sign). Increasing its bits moves a negative value
down the real line. The return value is decoded on the chosen neighbor grid. -/
def wordIndex (base k : Int) (residual : ℝ) : Int :=
  let word := base + if 0 < k then k else -k
  let step : Int := if decide (0 < residual) == decide (0 < k) then 1 else -1
  if residual ≠ 0 ∧ word % 2 = 0 then (if 0 < k then k+step else k-step) else k

private theorem indexDirection (base k : Int) (e : ℝ) (hb : Even base) :
    wordIndex base k e = if e = 0 ∨ Odd k then k else if 0 < e then k+1 else k-1 := by
  obtain ⟨j,rfl⟩ := hb
  by_cases hk : 0 < k <;> by_cases plus : 0 < e <;> by_cases zero : e=0 <;>
    simp [wordIndex,hk,plus,zero,Int.odd_iff] <;> split_ifs <;> omega

private theorem roundedIndex (x h : ℝ) (pos : 0 < h) (base : Int) (hb : Even base) :
    wordIndex base (nearestEven x) (h * (x-(nearestEven x : ℝ))) = oddRound x := by
  rw [indexDirection _ _ _ hb]
  by_cases integral : x = (⌊x⌋:ℝ)
  · obtain ⟨k,hk⟩ : ∃ k : Int, x=(k:ℝ) := ⟨_,integral⟩
    subst x
    simp [ValidRnd.id (rnd := nearestEven)]
  have lo := Int.floor_le x
  have hi := Int.lt_floor_add_one x
  have left : (⌊x⌋:ℝ) < x := lt_of_le_of_ne lo (Ne.symm integral)
  obtain ⟨l,u⟩ := nearestEven_bounds x
  have cases : nearestEven x = ⌊x⌋ ∨ nearestEven x = ⌊x⌋+1 := by omega
  rcases cases with eq | eq <;> rw [eq]
  · have sign := mul_pos pos (sub_pos.mpr left)
    simp only [oddRound,integral,ite_false,sign.ne',false_or,sign,ite_true]
  · have sign : h * (x-((⌊x⌋+1:Int):ℝ)) < 0 := by
      apply mul_neg_of_pos_of_neg pos
      push_cast; linarith
    simp only [sign.ne, false_or, not_lt_of_ge sign.le, ite_false, oddRound, integral,
      odd_add_one, ite_false]
    by_cases parity : Odd ⌊x⌋ <;> simp_all [Int.odd_iff,Int.even_iff]

/-- Reproduce the parity test and signed word step of add_odd on an adjacent
binary grid, using the actual rounded sum and Knuth residual. At an exponent
transition the scale is the spacing on the side containing the exact result. -/
def addOddAt (base : Int) (h a b : ℝ) :=
  let (s,e) := twoSum rn a b
  (wordIndex base ⌊s/h⌋ e : ℝ)*h

theorem add_odd_grid (base : Int) (evenBase : Even base) {a b h : ℝ}
    (ha : Rep a) (hb : Rep b) (pos : 0 < h) (scale : gridAt (a+b) = h) :
    addOddAt base h a b = roundAtScale oddRound h pos (a+b) := by
  let x := (a+b)/h
  let k := nearestEven x
  obtain ⟨head,sum⟩ := two_sum_exact ha hb
  have s : (twoSum rn a b).1 = (k:ℝ)*h := head.trans (rnGrid _ _ pos scale)
  have idx : ⌊(twoSum rn a b).1/h⌋ = k := by rw [s,mul_div_cancel_right₀ _ pos.ne',Int.floor_intCast]
  have e : (twoSum rn a b).2 = h*(x-(k:ℝ)) := by
    rw [s] at sum
    dsimp [x]; field_simp; linarith
  simp only [addOddAt,idx,e,roundedIndex x h pos base evenBase,roundAtScale,x,k]

private theorem oddTranslate (k : Int) (evenK : Even k) (x : ℝ) :
    oddRound ((k:ℝ)+x) = k+oddRound x := by
  have floorEq : ⌊(k:ℝ)+x⌋ = k+⌊x⌋ := by simp [add_comm]
  have integral : (k:ℝ)+x = ((k+⌊x⌋:Int):ℝ) ↔ x=(⌊x⌋:ℝ) := by push_cast; simp
  simp only [oddRound,floorEq,integral]
  have par : Odd (k+⌊x⌋) ↔ Odd ⌊x⌋ := by rw [Int.odd_iff,Int.odd_iff]; obtain ⟨j,rfl⟩ := evenK; omega
  simp only [par]; split_ifs <;> omega

private theorem shiftFine (t y h : ℝ) (pos : 0 < h) (aligned : ∃ k : Int, Even k ∧ t=(k:ℝ)*h) :
    t + roundAtScale oddRound h pos y = roundAtScale oddRound h pos (t+y) := by
  obtain ⟨k,hk,rfl⟩ := aligned
  have shift : ((k:ℝ)*h+y)/h = (k:ℝ)+y/h := by field_simp
  simp only [roundAtScale,shift,oddTranslate k hk,Int.cast_add]
  ring

def fmaAt (base : Int) (fine a b c : ℝ) :=
  let (uh,ul) := product53 a b
  let (th,tl) := twoSum rn c uh
  rn (th+addOddAt base fine tl ul)

/-- Single rounding of the composed CPU algorithm in a checked pair of
binades. Widths certify the product intermediates. The low sum has at least
two finer bits, the head is an even multiple of that fine unit, and the
final rounded-add input and ideal result lie on the same coarse grid.
All conditions refer to widths, magnitudes or grids, independently of the
desired single-rounded value. -/
theorem fma_single_round (base : Int) (evenBase : Even base) (extra : Nat)
    {a b c coarse : ℝ} (ha : Rep a) (hb : Rep b) (hc : Rep c)
    (sa : |a| < 2^996) (sb : |b| < 2^996) (widths : productWidths a b)
    (positive : 0 < coarse) :
    let fine := coarse/(2:ℝ)^(extra+2)
    let (uh,ul) := product53 a b
    let (th,tl) := twoSum rn c uh
    gridAt (tl+ul) = fine →
    (∃ k : Int, Even k ∧ th=(k:ℝ)*fine) →
    gridAt (th+addOddAt base fine tl ul) = coarse →
    gridAt (a*b+c) = coarse →
    fmaAt base fine a b c = rn (a*b+c) := by
  intro fine
  dsimp only
  intro lowGrid aligned finalGrid idealGrid
  let uh := (product53 a b).1
  let ul := (product53 a b).2
  let th := (twoSum rn c uh).1
  let tl := (twoSum rn c uh).2
  have finePos : 0 < fine := by dsimp [fine]; positivity
  obtain ⟨_,product⟩ := dekker_product_exact ha hb sa sb widths
  have hu : Rep uh := by change Rep (rn (a*b)); exact genericRepresentable_round nearestEven _
  have hl : Rep ul := by
    change Rep (rn _); exact genericRepresentable_round nearestEven _
  have ht : Rep tl := by
    change Rep (rn _); exact genericRepresentable_round nearestEven _
  have sum := (two_sum_exact hc hu).2
  change uh+ul=a*b at product
  change th+tl=c+uh at sum
  have exactTotal : th+(tl+ul)=a*b+c := by linarith
  change rn (th+addOddAt base fine tl ul) = rn (a*b+c)
  rw [rnGrid _ _ positive finalGrid,add_odd_grid base evenBase ht hl finePos lowGrid,
    shiftFine th (tl+ul) fine finePos aligned,exactTotal,
    roundAtScale_nearestEven_after_odd_binary_extra extra coarse (a*b+c) positive,
    rnGrid _ _ positive idealGrid]

end ExactBallpark.Numerics.Compensated

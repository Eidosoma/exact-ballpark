-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Numerics.ExpReduction
public import FloatLib.Floats.Formats.Flocq.Theory.Analysis.SterbenzFLT

@[expose] public noncomputable section
namespace ExactBallpark.Numerics.ExpFirstFma
open FloatLib FloatLib.Floats.Formats.Flocq Compensated

local instance : ValidExp (fltExp (-1074) 53) := validExp_flt _ _ (by decide)
set_option exponentiation.threshold 2048
abbrev Stored (x : ℝ) := GenericRepresentable Numerics.binaryRadix (fltExp (-1074) 53) x
def rounded (x : ℝ) := round (β := Numerics.binaryRadix) (fexp := fltExp (-1074) 53) nearestEven x

def invLn := (3248660424278399/17592186044416 : ℝ)
def coeff := (-47632711549/2^43 : ℝ)
def highPart := (-1453635/2^28 : ℝ)
def lowPart := (131/2^43 : ℝ)

def reductionIndex (x : ℝ) : Int :=
  let z := rounded (x*invLn)
  truncRound (rounded (z + if z < 0 then -(1/2) else 1/2))

private theorem normalError {x : ℝ} (h : x ≤ -(1/2^54)) : |rounded x-x| ≤ -x/2^53 := by
  have hx : x < 0 := by linarith
  have hnormal : bpow Numerics.binaryRadix (-1074+53-1) ≤ |x| := by
    rw [abs_of_neg hx]; norm_num [bpow_binaryRadix] at *; linarith
  have e := relative_error_round_flt_normal (-1074) 53 (by decide) nearestEven x hx.ne hnormal
  change |rounded x-x|/|x| ≤ _ at e
  rw [abs_of_neg hx,div_le_iff₀ (neg_pos.mpr hx)] at e
  norm_num [bpow_binaryRadix] at e
  linarith

private theorem indexError (x : ℝ) (lo : -1 ≤ x) (hi : x ≤ -(1/2^54)) :
    -185 ≤ reductionIndex x ∧ reductionIndex x ≤ 0 ∧
    |(reductionIndex x:ℝ)-x*invLn| ≤ 1/2+1/2^43 := by
  let z := rounded (x*invLn)
  have raw : x*invLn ≤ -(1/2^54) := by dsimp [invLn]; linarith
  have e1 := abs_le.mp (normalError raw)
  change _ ≤ z-x*invLn ∧ z-x*invLn ≤ _ at e1
  have hz : z < 0 := by linarith
  let b := rounded (z - 1/2)
  have e2 := abs_le.mp (normalError (show z-1/2 ≤ -(1/2^54) by linarith))
  change _ ≤ b-(z-1/2) ∧ b-(z-1/2) ≤ _ at e2
  have hb : b ≤ 0 := by linarith
  have idx : reductionIndex x = ⌈b⌉ := by
    simp only [reductionIndex, show rounded (x*invLn) < 0 from hz, ite_true,← sub_eq_add_neg]
    exact truncRound_eq_ceil hb
  have lower := Int.le_ceil b
  have upper := Int.ceil_lt_add_one b
  rw [idx,abs_le]
  dsimp [invLn] at e1 ⊢
  refine ⟨?_,Int.ceil_nonpos.mpr hb,?_,?_⟩
  · rw [Int.le_ceil_iff]
    norm_num
    linarith
  · linarith
  · linarith

private theorem storeProduct (k a e : Int) (small : k.natAbs ≤ 186) (width : a.natAbs ≤ 2^38)
    (emin : -1074 ≤ e) : Stored ((k:ℝ)*((a:ℝ)*(2:ℝ)^e)) := by
  apply generic_of_fltFormat (-1074) 53 (by decide)
  refine ⟨by decide,⟨k*a,e⟩,?_,?_,emin⟩
  · simp [toReal,bpow_binaryRadix,mul_assoc]
  · change (k*a).natAbs < 2^53
    rw [Int.natAbs_mul]
    exact (Nat.mul_le_mul small width).trans_lt (by decide)

private theorem keep {x : ℝ} (hx : Stored x) : rounded x = x :=
  round_preserves_generic nearestEven x hx

/-- The constant arrives pre-split. The neighbor callback is the parity/word
step taken by add_odd on inexact sums; the theorem works for any such step
because its residual vanishes at this site. -/
def firstFma (neighbor : ℝ → ℝ → ℝ) (k x : ℝ) :=
  let c := rounded (k*134217729)
  let kh := rounded (c-rounded (c-k))
  let kl := rounded (k-kh)
  let p := rounded (k*coeff)
  let tail := rounded (rounded (kl*lowPart) -
    rounded (rounded (rounded (p-rounded (kh*highPart))-rounded (kl*highPart))-rounded (kh*lowPart)))
  let (th,tl) := twoSum rounded x p
  let (s,e) := twoSum rounded tl tail
  rounded (th + if e = 0 then s else neighbor s e)

/-- The first FMA of vec_exp.exp is exact throughout the ordinary drag
reduction interval. Index rounding, partial products and gradual underflow
use the 53-bit, minimum-exponent −1074 model. The claim is about values. -/
theorem first_reduction_exact (x : ℝ) (hx : Stored x) (lo : -1 ≤ x) (hi : x ≤ -(1/2^54))
    (neighbor : ℝ → ℝ → ℝ) :
    let k : ℝ := reductionIndex x
    (-185 ≤ reductionIndex x ∧ reductionIndex x ≤ 0) ∧ |k*134217729| < 2^35 ∧
    firstFma neighbor k x = k*coeff+x ∧ rounded (k*coeff+x) = k*coeff+x := by
  let i := reductionIndex x
  let k : ℝ := i
  obtain ⟨il,iu,err⟩ := indexError x lo hi
  change -185 ≤ i at il
  change i ≤ 0 at iu
  change |k-x*invLn| ≤ _ at err
  have width : i.natAbs ≤ 186 := by omega
  have safeSplit : |k*134217729| < 2^35 := by
    have lk : (-185:ℝ) ≤ (i:ℝ) := by exact_mod_cast il
    have uk : (i:ℝ) ≤ 0 := by exact_mod_cast iu
    dsimp only [k]; rw [abs_lt]; constructor <;> linarith
  refine ⟨⟨il,iu⟩,safeSplit,?_⟩
  have S := storeProduct i
  have sk : Stored k := by simpa [k] using S 1 0 width (by decide) (by decide)
  have sc : Stored (k*134217729) := by simpa [k] using S 134217729 0 width (by decide) (by decide)
  have ss : Stored (k*134217728) := by simpa [k] using S 134217728 0 width (by decide) (by decide)
  have sp : Stored (k*coeff) := by
    convert S (-47632711549) (-43) width (by decide) (by decide) using 1
    norm_num [coeff,k]
  have sh : Stored (k*highPart) := by
    convert S (-1453635) (-28) width (by decide) (by decide) using 1
    norm_num [highPart,k]
  have sl : Stored (k*lowPart) := by
    convert S 131 (-43) width (by decide) (by decide) using 1
    norm_num [lowPart,k]
  have fp : rounded (k*coeff) = k*coeff := keep sp
  have zero : rounded 0 = 0 := by simp [rounded]
  have sum : rounded (x+k*coeff) = x+k*coeff := by
    by_cases iz : i=0
    · simpa [k,iz] using keep hx
    have kr : k ≤ -1 := by dsimp only [k]; exact_mod_cast (show i ≤ -1 by omega)
    obtain ⟨el,eu⟩ := abs_le.mp err
    have diff := genericRepresentable_flt_sterbenz (-1074) 53 (by decide)
      (x := k*coeff) (y := -x)
      (by dsimp [coeff,invLn] at *; linarith)
      (by dsimp [coeff,invLn] at *; linarith) sp (genericRepresentable_neg x hx)
    apply keep
    simpa [sub_eq_add_neg,add_comm] using diff
  have splitDiff : k*134217729-k = k*134217728 := by ring
  have splitHi : k*134217729-k*134217728 = k := by ring
  have rem : k*coeff-k*highPart = k*lowPart := by
    norm_num [coeff,highPart,lowPart]; ring
  change firstFma neighbor k x = _ ∧ _
  refine ⟨?_,by simpa only [add_comm] using sum⟩
  simp only [firstFma,keep sc,splitDiff,keep ss,splitHi,keep sk,sub_self,zero,fp,
    keep sh,rem,keep sl,zero_mul,sub_zero,twoSum,sum,add_sub_cancel_left,
    add_sub_cancel_right,keep hx,add_zero,ite_true]
  exact add_comm ..

end ExactBallpark.Numerics.ExpFirstFma

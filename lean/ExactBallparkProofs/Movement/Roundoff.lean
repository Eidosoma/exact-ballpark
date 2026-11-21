-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Correspondence
public import ExactBallparkProofs.Numerics.TwoSum
public import FloatLib.Floats.Formats.Flocq.Theory.Error.Relative

/-! One relative rounding factor per binary64 operation in the taken MOV-703
position branch, for finite normal-or-zero exact intermediates. The same stored
k² is used in both places. Negation is exact. -/
@[expose] public noncomputable section
namespace ExactBallpark.Movement.Roundoff

abbrev Factors := Fin 16 → ℝ
def u : ℝ := 1 / 2^53
private def lo := 1-u
private def hi := 1+u

def position (p v a m k t : ℝ) (f : Factors) :=
  let k2 := (k*k)*f 0
  let ik2 := (1/k2)*f 1
  let im := (1/m)*f 2
  let neg := (a*((-k*t)*f 3))*f 4
  let drift := ((((v*k)*f 6*im)*f 7)*t)*f 8
  let pos := (k*((a*t*f 5+drift)*f 9))*f 10
  ((m*((neg+pos)*f 11)*f 12+p*k2*f 13)*f 14)*ik2*f 15

private def product (f : Factors) (xs : List (Fin 16)) := (xs.map f).prod

private theorem factors_bound (f : Factors) (hf : ∀ i, lo ≤ f i ∧ f i ≤ hi)
    (xs : List (Fin 16)) : lo^xs.length ≤ product f xs ∧ product f xs ≤ hi^xs.length := by
  have hl : 0 ≤ lo := by norm_num [lo,u]
  induction xs with
  | nil => simp [product]
  | cons i xs ih =>
    simp only [List.length_cons, pow_succ', product, List.map_cons, List.prod_cons] at *
    exact ⟨mul_le_mul (hf i).1 ih.1 (pow_nonneg hl _) (hl.trans (hf i).1),
      mul_le_mul (hf i).2 ih.2 ((pow_nonneg hl _).trans ih.1) (by norm_num [hi,u])⟩

private theorem perturbation_bound (p v a m k t : ℝ) (hm : 0 < m) (hk : 0 < k)
    (f : Factors) (rounded : ∀ i, |f i - 1| ≤ u) :
    |position p v a m k t f - (p+v*t)| ≤
      u*(6*|(m*a/k)*t| + 13*|v*t| + 5*|p|) := by
  have hf (i) : lo ≤ f i ∧ f i ≤ hi := by
    obtain ⟨h,j⟩ := abs_le.mp (rounded i)
    dsimp [lo,hi]; constructor <;> linarith
  have hl : 0 < lo := by norm_num [lo,u]
  have fpos := hl.trans_le (hf 0).1
  let A := product f [5,9,10]
  let B := product f [3,4]
  let C := product f [11,12,14,15,1]
  let D := product f [6,2,7,8,9,10,11,12,14,15,1]
  let P := product f [13,14,15,1]
  have hA : lo^3 ≤ A ∧ A ≤ hi^3 := factors_bound f hf [5,9,10]
  have hB : lo^2 ≤ B ∧ B ≤ hi^2 := factors_bound f hf [3,4]
  have hC : lo^5 ≤ C ∧ C ≤ hi^5 := factors_bound f hf [11,12,14,15,1]
  have hD : lo^11 ≤ D ∧ D ≤ hi^11 := factors_bound f hf [6,2,7,8,9,10,11,12,14,15,1]
  have hP : lo^4 ≤ P ∧ P ≤ hi^4 := factors_bound f hf [13,14,15,1]
  have ca : |C/f 0 * (A-B)| ≤ 6*u := by
    rw [abs_mul, abs_of_nonneg (div_nonneg ((by positivity : 0 ≤ lo^5).trans hC.1) fpos.le)]
    have ratio : C / f 0 ≤ hi^5 / lo :=
      div_le_div₀ (by norm_num [hi,u]) hC.2 hl (hf 0).1
    have gap : |A-B| ≤ max (hi^3-lo^2) (hi^2-lo^3) := by
      rw [abs_le]; constructor
      · linarith [le_max_right (hi^3-lo^2) (hi^2-lo^3)]
      · linarith [le_max_left (hi^3-lo^2) (hi^2-lo^3)]
    exact (mul_le_mul ratio gap (abs_nonneg _) (by norm_num [hi,lo,u])).trans
      (by norm_num [hi,lo,u, max_def])
  have cd : |D/f 0 - 1| ≤ 13*u := by
    have lower : lo^11 / hi ≤ D / f 0 := div_le_div₀ ((by positivity : 0 ≤ lo^11).trans hD.1) hD.1 fpos (hf 0).2
    have upper : D / f 0 ≤ hi^11 / lo := div_le_div₀ (by norm_num [hi,u]) hD.2 hl (hf 0).1
    rw [abs_le]; constructor
    · linarith only [lower, show 1-13*u ≤ lo^11/hi by norm_num [hi,lo,u]]
    · linarith only [upper, show hi^11/lo ≤ 1+13*u by norm_num [hi,lo,u]]
  have cp : |P-1| ≤ 5*u := by
    rw [abs_le]; constructor
    · linarith only [hP.1, show 1-5*u ≤ lo^4 by norm_num [lo,u]]
    · linarith only [hP.2, show hi^4 ≤ 1+5*u by norm_num [hi,u]]
  have expand : position p v a m k t f - (p+v*t) =
      C/f 0*(A-B)*((m*a/k)*t) + (D/f 0-1)*(v*t) + (P-1)*p := by
    simp [position,A,B,C,D,P,product]
    field_simp [hm.ne', hk.ne', fpos.ne']
    ring
  rw [expand]
  calc
    _ ≤ |C/f 0*(A-B)*((m*a/k)*t) + (D/f 0-1)*(v*t)| + |(P-1)*p| := abs_add_le ..
    _ ≤ (|C/f 0*(A-B)*((m*a/k)*t)| + |(D/f 0-1)*(v*t)|) + |(P-1)*p| :=
      add_le_add (abs_add_le ..) le_rfl
    _ ≤ _ := by
      rw [abs_mul (C / f 0 * (A - B)), abs_mul (D / f 0 - 1), abs_mul (P - 1)]
      linarith only [mul_le_mul_of_nonneg_right ca (abs_nonneg ((m*a/k)*t)),
        mul_le_mul_of_nonneg_right cd (abs_nonneg (v*t)), mul_le_mul_of_nonneg_right cp (abs_nonneg p)]

open ExactBallpark.Numerics.Compensated (rn)
open FloatLib FloatLib.Floats.Formats.Flocq

/-- Each input is the exact result immediately before one rounding. A shared
temporary is reused in the same places as the engine's vector operations. -/
def roundInputs (p v a m k t : ℝ) : Fin 16 → ℝ :=
  let k2 := rn (k*k)
  let reciprocal := rn (1/k2)
  let im := rn (1/m)
  let kt := rn (-k*t)
  let neg := rn (a*kt)
  let atime := rn (a*t)
  let vk := rn (v*k)
  let rate := rn (vk*im)
  let drift := rn (rate*t)
  let sum := rn (atime+drift)
  let pos := rn (k*sum)
  let net := rn (neg+pos)
  let scaled := rn (m*net)
  let origin := rn (p*k2)
  let total := rn (scaled+origin)
  ![k*k, 1/k2, 1/m, -k*t, a*kt, a*t, v*k, vk*im, rate*t, atime+drift, k*sum,
    neg+pos, m*net, p*k2, scaled+origin, total*reciprocal]

private def factor (x : ℝ) := if x=0 then 1 else rn x / x

private theorem times_factor (x : ℝ) : x * factor x = rn x := by
  by_cases z : x=0
  · simp [z,factor,rn]
  · simp [factor,z,mul_div_cancel₀]

private theorem factor_bound (x : ℝ) : |factor x-1| ≤ u := by
  by_cases z : x=0
  · simp [factor,z,u]
  obtain ⟨δ,bound,eq⟩ := round_relative_error_flx (β := Numerics.binaryRadix) 53 (by decide) nearestEven x
  change rn x = x*(1+δ) at eq
  have ratio : factor x-1 = δ := by rw [factor,ite_eq_right z,eq]; field_simp; ring
  rw [ratio]; norm_num [bpow_binaryRadix] at bound; norm_num [u]; exact bound

/-- On the taken special branch, all sixteen operations round to nearest in
the 53-bit model. The proof obtains each rounding factor from FloatLib. -/
theorem special_position_error (p v a m k t : ℝ) (hm : 0 < m) (hk : 0 < k) :
    |rn (roundInputs p v a m k t 15) - (p+v*t)| ≤
      u*(6*|(m*a/k)*t| + 13*|v*t| + 5*|p|) := by
  let f := fun i => factor (roundInputs p v a m k t i)
  have result := perturbation_bound p v a m k t hm hk f (fun i => factor_bound _)
  have eq : position p v a m k t f = rn (roundInputs p v a m k t 15) := by
    simp only [position,f,roundInputs,Matrix.cons_val,times_factor]
  rwa [eq] at result

end ExactBallpark.Movement.Roundoff

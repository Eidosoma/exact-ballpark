-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.CurveClearance
public import Mathlib.Analysis.Complex.Exponential

@[expose] public section
namespace ExactBallpark.Movement.PositionEnclosure
open Set DragChord ExactBallpark.Collision CurveClearance

structure Bounds where
  lo : ℚ
  hi : ℚ
  deriving Repr

def Bounds.Contains (b : Bounds) (x : ℝ) := (b.lo : ℝ) ≤ x ∧ x ≤ (b.hi : ℝ)

def Bounds.point (x : ℚ) : Bounds := ⟨x,x⟩
def Bounds.add (a b : Bounds) : Bounds := ⟨a.lo + b.lo, a.hi + b.hi⟩
def Bounds.sub (a b : Bounds) : Bounds := ⟨a.lo - b.hi, a.hi - b.lo⟩
def Bounds.scale (k : ℚ) (b : Bounds) : Bounds :=
  if 0 ≤ k then ⟨k*b.lo, k*b.hi⟩ else ⟨k*b.hi, k*b.lo⟩

theorem Bounds.point_contains (k : ℚ) : (Bounds.point k).Contains k := ⟨le_rfl, le_rfl⟩

theorem Bounds.add_contains {a b : Bounds} {x y : ℝ} (hx : a.Contains x) (hy : b.Contains y) :
    (a.add b).Contains (x+y) := by
  dsimp [Contains, add]; push_cast
  exact ⟨add_le_add hx.1 hy.1, add_le_add hx.2 hy.2⟩

theorem Bounds.sub_contains {a b : Bounds} {x y : ℝ} (hx : a.Contains x) (hy : b.Contains y) :
    (a.sub b).Contains (x-y) := by
  dsimp [Contains, sub]; push_cast
  exact ⟨sub_le_sub hx.1 hy.2, sub_le_sub hx.2 hy.1⟩

theorem Bounds.scale_contains (k : ℚ) {b : Bounds} {x : ℝ} (hx : b.Contains x) :
    (b.scale k).Contains ((k:ℝ)*x) := by
  by_cases h : 0 ≤ k
  · have hr : (0:ℝ) ≤ k := by exact_mod_cast h
    simp only [scale, ite_eq_left h, Contains, Rat.cast_mul]
    exact ⟨mul_le_mul_of_nonneg_left hx.1 hr, mul_le_mul_of_nonneg_left hx.2 hr⟩
  · have hr : (k:ℝ) ≤ 0 := by exact_mod_cast (le_of_not_ge h)
    simp only [scale, ite_eq_right h, Contains, Rat.cast_mul]
    exact ⟨mul_le_mul_of_nonpos_left hx.2 hr, mul_le_mul_of_nonpos_left hx.1 hr⟩

/-- Twelve terms and mathlib's bound for the Taylor remainder on the unit interval. -/
def unitBounds (z : ℚ) : Bounds :=
  let s := ∑ j ∈ Finset.range 12, (-z)^j / (j.factorial : ℚ)
  let e := z^12 * (13 / ((12 : ℕ).factorial * 12 : ℚ))
  ⟨max 0 (s-e), min 1 (s+e)⟩

private theorem unit_sound (z : ℚ) (hz : 0 ≤ z) (small : z ≤ 1) :
    (unitBounds z).Contains (Real.exp (-(z:ℝ))) := by
  have h0 : (0:ℝ) ≤ z := by exact_mod_cast hz
  have upper : Real.exp (-(z:ℝ)) ≤ 1 := Real.exp_le_one_iff.mpr (neg_nonpos.mpr h0)
  have h1 : (z:ℝ) ≤ 1 := by exact_mod_cast small
  have h := Real.exp_bound (x := -(z:ℝ)) (by simpa [abs_neg, abs_of_nonneg h0] using h1)
    (n := 12) (by decide)
  rw [abs_neg, abs_of_nonneg h0] at h
  obtain ⟨left, right⟩ := abs_le.mp h
  dsimp only [unitBounds, Bounds.Contains]
  push_cast
  constructor
  · apply max_le (Real.exp_pos _).le
    linarith
  · apply le_min upper
    linarith

def rescaled (z : ℚ) (n : ℕ) : Bounds :=
  let b := unitBounds (z/n)
  ⟨b.lo^n,b.hi^n⟩

private theorem rescaled_sound (z : ℚ) (n : ℕ) (hn : 0 < n) (hz : 0 ≤ z) (cap : z ≤ n) :
    (rescaled z n).Contains (Real.exp (-(z:ℝ))) := by
  have hp : (0:ℚ) < n := by exact_mod_cast hn
  have small := unit_sound (z/n) (div_nonneg hz hp.le) ((div_le_one hp).mpr cap)
  have lower : (0:ℝ) ≤ (unitBounds (z/n)).lo := by
    exact Rat.cast_nonneg.mpr (le_max_left 0 _)
  have eq : Real.exp (-(z:ℝ)) = (Real.exp (-((z/n:ℚ):ℝ)))^n := by
    rw [← Real.exp_nat_mul]
    push_cast; congr 1; field_simp
  rw [eq]
  simp only [Bounds.Contains, rescaled, Rat.cast_pow]
  exact ⟨pow_le_pow_left₀ lower small.1 _, pow_le_pow_left₀ (Real.exp_pos _).le small.2 _⟩

/-- Reduce the argument on [1,64]. Beyond 64, monotonicity and exp(1) ≥ 2
give a uniform upper bound without constructing very large rational powers. -/
def expBounds (z : ℚ) : Bounds :=
  if z ≤ 1 then unitBounds z else if z ≤ 8 then rescaled z 8
  else if z ≤ 64 then rescaled z 64 else ⟨0,(1/2)^64⟩

theorem decay_enclosed (z : ℚ) (hz : 0 ≤ z) :
    (expBounds z).Contains (Real.exp (-(z:ℝ))) := by
  unfold expBounds
  split
  next h => exact unit_sound z hz h
  next =>
    split
    next h => exact rescaled_sound z 8 (by decide) hz h
    next =>
      split
      next h => exact rescaled_sound z 64 (by decide) hz h
      next h =>
        have base : Real.exp (-1) ≤ 1/2 := by
          rw [Real.exp_neg, inv_eq_one_div]
          exact one_div_le_one_div_of_le (by norm_num) (by linarith [Real.add_one_le_exp 1])
        have cap : (64:ℝ) ≤ z := by exact_mod_cast (le_of_not_ge h)
        have shrink := Real.exp_le_exp.mpr (neg_le_neg cap)
        rw [show -(64:ℝ) = (64:ℕ)*(-1:ℝ) by norm_num, Real.exp_nat_mul] at shrink
        exact ⟨by simpa using (Real.exp_pos (-(z:ℝ))).le,
          by push_cast; exact shrink.trans (pow_le_pow_left₀ (Real.exp_pos _).le base _)⟩

/-- A rational initial state, effective mass, friction and constant applied acceleration. -/
structure Body where
  start : RatPoint
  velocity : RatPoint
  applied : RatPoint
  mass : ℚ
  friction : ℚ

def Body.Valid (b : Body) := 0 < b.mass ∧ 0 < b.friction
instance (b : Body) : Decidable b.Valid := inferInstanceAs (Decidable (_ ∧ _))

noncomputable def Body.flight (b : Body) : Flight :=
  ⟨asVec b.start, asVec b.velocity, asVec b.applied, b.mass, b.friction⟩

theorem Body.valid_flight (b : Body) (valid : b.Valid) : b.flight.Valid := by
  change (0:ℝ) < b.mass ∧ (0:ℝ) < b.friction
  exact ⟨by exact_mod_cast valid.1, by exact_mod_cast valid.2⟩

/-- The coefficient of initial velocity is F(t) = (m/k)(1−exp(−kt/m)).
The coefficient of terminal velocity is t−F(t). Both increase for t ≥ 0. -/
noncomputable def weight (m k t : ℝ) := m/k*(1-decay m k t)

theorem weights_monotone {m k : ℝ} (hm : 0 < m) (hk : 0 < k) :
    MonotoneOn (weight m k) (Ici 0) ∧ MonotoneOn (fun t => t - weight m k t) (Ici 0) := by
  have derivF (t : ℝ) : HasDerivAt (weight m k) (decay m k t) t := by
    have h := ((decay_hasDerivAt m k t).const_sub 1).const_mul (m/k)
    change HasDerivAt (fun s => m/k * (1 - decay m k s)) _ t
    convert h using 1
    field_simp
  have derivG (t : ℝ) : HasDerivAt (fun s => s - weight m k s) (1 - decay m k t) t :=
    (hasDerivAt_id t).sub (derivF t)
  constructor
  · apply monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici 0)
      (fun t _ => (derivF t).continuousAt.continuousWithinAt)
      (fun t _ => (derivF t).hasDerivWithinAt)
    exact fun t _ => (Real.exp_pos _).le
  · apply monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici 0)
      (fun t _ => (derivG t).continuousAt.continuousWithinAt)
      (fun t _ => (derivG t).hasDerivWithinAt)
    intro t ht
    exact sub_nonneg.mpr (decay_bounds hm hk (interior_subset ht)).2

def Body.weightsAt (b : Body) (t : ℚ) : Bounds × Bounds :=
  let q := expBounds (b.friction / b.mass * t)
  let f := ((Bounds.point 1).sub q).scale (b.mass / b.friction)
  (f, (Bounds.point t).sub f)

private theorem weightsAt_sound (b : Body) (hb : b.Valid) {t : ℚ} (ht : 0 ≤ t) :
    let f := weight b.mass b.friction t
    (b.weightsAt t).1.Contains f ∧ (b.weightsAt t).2.Contains ((t:ℝ)-f) := by
  have he := decay_enclosed (b.friction / b.mass * t)
    (mul_nonneg (div_nonneg hb.2.le hb.1.le) ht)
  have hq : (expBounds (b.friction / b.mass * t)).Contains (decay b.mass b.friction t) := by
    simpa [decay, neg_div] using he
  have hf := Bounds.scale_contains (b.mass/b.friction)
    (Bounds.sub_contains (Bounds.point_contains 1) hq)
  have hw : (b.weightsAt t).1.Contains (weight b.mass b.friction t) := by
    simpa [Body.weightsAt, weight] using hf
  exact ⟨hw, Bounds.sub_contains (Bounds.point_contains t) hw⟩

/-- This enclosure covers every intermediate time, including intervals where a
velocity component reverses. Equal initial and terminal velocities are exactly linear. -/
def Body.box (b : Body) (l h : ℚ) : Fin 3 → Bounds :=
  let (fl, gl) := b.weightsAt l
  let (fh, gh) := b.weightsAt h
  fun i =>
    let v := b.velocity i
    let w := b.mass / b.friction * b.applied i
    let shift := if v = w then (⟨l,h⟩ : Bounds).scale v
      else ((⟨fl.lo, fh.hi⟩ : Bounds).scale v).add ((⟨gl.lo, gh.hi⟩ : Bounds).scale w)
    (Bounds.point (b.start i)).add shift

theorem position_enclosed (b : Body) (valid : b.Valid) {l h : ℚ} (hl : 0 ≤ l)
    {t : ℝ} (ht : t ∈ Icc (l:ℝ) h) (i : Fin 3) : (b.box l h i).Contains (b.flight.pos t i) := by
  have hl' : (0:ℝ) ≤ l := by exact_mod_cast hl
  have h0 := hl'.trans ht.1
  have hh : 0 ≤ h := by exact_mod_cast h0.trans ht.2
  obtain ⟨fl, gl⟩ := weightsAt_sound b valid hl
  obtain ⟨fh, gh⟩ := weightsAt_sound b valid hh
  obtain ⟨mf, mg⟩ := weights_monotone (b.valid_flight valid).1 (b.valid_flight valid).2
  have leF := mf hl' h0 ht.1
  have geF := mf h0 (h0.trans ht.2) ht.2
  have leG := mg hl' h0 ht.1
  have geG := mg h0 (h0.trans ht.2) ht.2
  let w := b.mass/b.friction*b.applied i
  have value : b.flight.pos t i = (b.start i : ℝ) + ((b.velocity i:ℝ)*weight b.mass b.friction t
      + (w:ℝ)*(t-weight b.mass b.friction t)) := by
    change position _ _ _ _ _ t = _
    dsimp [position, terminal, Body.flight, asVec, realPoint, weight, w]; push_cast; ring
  rw [value]
  apply Bounds.add_contains (Bounds.point_contains _)
  by_cases same : b.velocity i = w
  · have eqv : (b.velocity i:ℝ) = (w:ℝ) := by exact_mod_cast same
    rw [show (b.velocity i:ℝ)*weight b.mass b.friction t + (w:ℝ)*(t-weight b.mass b.friction t) =
      (b.velocity i:ℝ)*t by rw [← eqv]; ring]
    simpa [Body.box, w, same] using Bounds.scale_contains (b.velocity i)
      (show (⟨l,h⟩ : Bounds).Contains t from ht)
  · change Bounds.Contains (if b.velocity i = w then _ else _) _
    rw [ite_eq_right same]
    exact Bounds.add_contains (Bounds.scale_contains _ ⟨fl.1.trans leF, geF.trans fh.2⟩)
      (Bounds.scale_contains _ ⟨gl.1.trans leG, geG.trans gh.2⟩)

end ExactBallpark.Movement.PositionEnclosure

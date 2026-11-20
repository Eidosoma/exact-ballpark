-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Warp.Reference
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Positivity
public import Mathlib.Tactic.Ring

/-! Real interpretation of the scalar warp coordinate in ModeBallpark.warp_constants
and _warp_distance. Rates use the binary64 values read by the reference. -/
@[expose] public noncomputable section

namespace ExactBallpark.Warp

def milli : ℝ := 1152921504606847 / 1152921504606846976
def acceleration (factor : ℝ) := factor * milli
def deceleration (factor : ℝ) := min (factor * (6148914691236517 / 18446744073709551616)) 2
def requestedSpeed (factor : ℝ) := factor * milli * 149597870700

theorem rate_bounds {f : ℝ} (hf : 0 < f) :
    0 < acceleration f ∧ requestedSpeed f = acceleration f * 149597870700 ∧
    0 < deceleration f ∧ deceleration f ≤ 2 := by
  refine ⟨?_, rfl, lt_min ?_ (by norm_num), min_le_right ..⟩
  · unfold acceleration milli; positivity
  · positivity

def shortPeak (d a b : ℝ) := (d + 1) * a * b / (a + b)
def peak (d a b requested : ℝ) := min requested (shortPeak d a b)
def cruiseTime (d a b s : ℝ) := d / s - 1 / a - 1 / b + 1 / s
def accelTime (a s : ℝ) := Real.log (s / a) / a
def decelTime (b s : ℝ) := Real.log (s / b) / b

theorem source_constants (factor distance : ℝ) :
    let a := acceleration factor
    let b := deceleration factor
    let s := peak distance a b (requestedSpeed factor)
    let tc := cruiseTime distance a b s
    Reference.constants factor distance =
      (accelTime a s, tc, decelTime b s, s/a, s*tc, s/b-1, a, b, s) := by rfl

theorem short_cruise_zero {d a b requested : ℝ} (hd : 0 ≤ d) (ha : 0 < a) (hb : 0 < b)
    (h : shortPeak d a b ≤ requested) : cruiseTime d a b (peak d a b requested) = 0 := by
  rw [peak, min_eq_right h]
  unfold cruiseTime shortPeak
  field_simp
  ring

theorem cruise_nonnegative {d a b requested : ℝ} (hd : 0 ≤ d) (ha : 0 < a) (hb : 0 < b)
    (hr : 0 < requested) : 0 ≤ cruiseTime d a b (peak d a b requested) := by
  let s := peak d a b requested
  have hs : 0 < s := lt_min hr (by unfold shortPeak; positivity)
  have cap : s * (a + b) ≤ (d + 1) * a * b :=
    (le_div_iff₀ (add_pos ha hb)).mp (min_le_right ..)
  change 0 ≤ d / s - 1 / a - 1 / b + 1 / s
  field_simp
  linarith only [cap]

theorem phase_distances_sum {s : ℝ} (hs : s ≠ 0) (d a b : ℝ) :
    s / a + s * cruiseTime d a b s + (s / b - 1) = d := by
  unfold cruiseTime
  field_simp [hs]; ring

def accelDistance (a t : ℝ) := Real.exp (a * t)
def cruiseDistance (a s u : ℝ) := s * u + s / a
def decelDistance (a b s tc u : ℝ) :=
  s / b - s / b * Real.exp (-b * u) + s / a + s * tc

/-- The two joins and the endpoint of nominal deceleration, with local phase times. -/
theorem phase_joins {a b s : ℝ} (ha : 0 < a) (hb : 0 < b) (hs : 0 < s) (d : ℝ) :
    accelDistance a (accelTime a s) = cruiseDistance a s 0 ∧
    cruiseDistance a s (cruiseTime d a b s) = decelDistance a b s (cruiseTime d a b s) 0 ∧
    decelDistance a b s (cruiseTime d a b s) (decelTime b s) = d := by
  have hA : a * accelTime a s = Real.log (s / a) := by unfold accelTime; field_simp
  have hD : -b * decelTime b s = -Real.log (s / b) := by unfold decelTime; field_simp
  refine ⟨?_, ?_, ?_⟩
  · simp [accelDistance, cruiseDistance, hA, Real.exp_log (div_pos hs ha)]
  · simp [cruiseDistance, decelDistance, add_comm]
  · rw [decelDistance, hD, Real.exp_neg, Real.exp_log (div_pos hs hb)]
    have sum := phase_distances_sum hs.ne' d a b
    field_simp at sum ⊢
    linarith only [sum]

inductive Phase where | accel | cruise | decel | overrun
  deriving DecidableEq

def phase (ta tc td t : ℝ) :=
  if t < ta then Phase.accel else if t - ta < tc then .cruise
  else if t - tc - ta < td + 1 then .decel else .overrun

theorem phase_decel_iff (ta tc td t : ℝ) :
    phase ta tc td t = .decel ↔ ta ≤ t ∧ ta + tc ≤ t ∧ t < ta + tc + td + 1 := by
  have h₁ : ta + tc ≤ t ↔ ¬ t - ta < tc := by constructor <;> intro h <;> linarith
  have h₂ : t < ta + tc + td + 1 ↔ t - tc - ta < td + 1 := by constructor <;> intro h <;> linarith
  rw [h₁, h₂, ← not_lt]
  unfold phase
  split_ifs <;> simp_all

def drop (maxVel speed : ℝ) : Prop := speed < min (maxVel / 2) 100

theorem dropout_iff (maxVel speed : ℝ) :
    drop maxVel speed ↔ 2 * speed < maxVel ∧ speed < 100 := by
  rw [drop, lt_min_iff]; constructor <;> rintro ⟨h₁, h₂⟩ <;> exact ⟨by linarith, h₂⟩

theorem decel_velocity {b : ℝ} (hb : b ≠ 0) (a s tc u : ℝ) :
    HasDerivAt (decelDistance a b s tc) (s * Real.exp (-b * u)) u := by
  have he := ((hasDerivAt_id u).const_mul (-b)).exp
  have h := ((((he.const_mul (s / b)).const_sub (s / b)).add_const (s / a)).add_const (s * tc))
  convert h using 1 <;> try rfl
  simp only [id_eq, mul_one]
  field_simp

end ExactBallpark.Warp

-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Warp.Profile

@[expose] public noncomputable section
namespace ExactBallpark.Warp

theorem positive_cruise_iff {d a b requested : ℝ} (hd : 0 ≤ d) (ha : 0 < a) (hb : 0 < b)
    (hr : 0 < requested) :
    0 < cruiseTime d a b (peak d a b requested) ↔ requested*(a+b) < (d+1)*a*b := by
  by_cases h : shortPeak d a b ≤ requested
  · have cap := (div_le_iff₀ (add_pos ha hb)).mp h
    rw [short_cruise_zero hd ha hb h]
    exact iff_of_false (lt_irrefl _) (not_lt.mpr cap)
  · have cap := (lt_div_iff₀ (add_pos ha hb)).mp (lt_of_not_ge h : requested < shortPeak d a b)
    rw [peak, min_eq_left (lt_of_not_ge h).le, cruiseTime]
    refine iff_of_true ?_ cap
    field_simp
    linarith only [cap]

theorem dropout_time_iff {s b maxVel : ℝ} (hs : 0 < s) (hb : 0 < b) (hm : 0 < maxVel) (u : ℝ) :
    drop maxVel (s * Real.exp (-b*u)) ↔ Real.log (s / min (maxVel/2) 100) / b < u := by
  let c := min (maxVel/2) 100
  have hc : 0 < c := lt_min (by positivity) (by norm_num)
  change s * Real.exp (-b*u) < c ↔ Real.log (s/c)/b < u
  rw [neg_mul, Real.exp_neg, ← div_eq_mul_inv, div_lt_iff₀ (Real.exp_pos _),
    div_lt_iff₀ hb, mul_comm u b, Real.log_lt_iff_lt_exp (div_pos hs hc), div_lt_iff₀ hc, mul_comm c]

/-- Binary64 projection of SetupWarpConstants through the cruise duration field. -/
def cruise64 (factor distance : Float) : Float :=
  let a := factor * Float.ofBits 0x3f50624dd2f1a9fc
  let b₀ := factor * Float.ofBits 0x3f35d867c3ece2a5
  let b := if (2 : Float) < b₀ then 2 else b₀
  let requested := a * 149597870700
  let candidate := (distance+1)*a*b/(a+b)
  let s := if candidate < requested then candidate else requested
  distance/s - 1/a - 1/b + 1/s

set_option maxRecDepth 8192 in
theorem short_cruise_words :
    (∀ d : ℝ, 100000 ≤ d → d ≤ 100001 →
      cruiseTime d (acceleration 1000) (deceleration 1000)
        (peak d (acceleration 1000) (deceleration 1000) (requestedSpeed 1000)) = 0) ∧
    (cruise64 1000 100000).toBits = 0x3caf75c000000000 ∧
    (cruise64 1000 100001).toBits = 0xbcaa6dc000000000 := by
  constructor
  · intro d lo hi
    obtain ⟨ha, _, hb, _⟩ := rate_bounds (by norm_num : (0 : ℝ) < 1000)
    apply short_cruise_zero (by linarith) ha hb
    rw [shortPeak, div_le_iff₀ (add_pos ha hb)]
    norm_num [acceleration, deceleration, milli, requestedSpeed] at *
    linarith
  · decide +kernel

end ExactBallpark.Warp

-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Mathlib.Analysis.SpecialFunctions.ExpDeriv
public import Mathlib.Tactic.Linarith

@[expose] public section
namespace ExactBallpark.Space

/-- State and optional event for a live, eligible candidate. Guard order mirrors the
enter/leave branches used by sensors, tracking and notification range. -/
def edge (old inside : Bool) : Bool × Option Bool :=
  if inside then
    if old then (old, none) else (true, some true)
  else if old then (false, some false) else (old, none)

theorem crossing (old inside : Bool) :
    edge old inside = (inside, if old = inside then none else some inside) := by
  cases old <;> cases inside <;> decide

theorem repeated_sample (old inside : Bool) : edge (edge old inside).1 inside = (inside, none) := by
  rw [crossing, crossing]; simp

noncomputable def sensor (old : Bool) (d threshold : ℝ) :=
  edge old (decide (¬d^2 > threshold^2))

theorem sensor_inside (old : Bool) {d threshold : ℝ} (hd : 0 ≤ d) (ht : 0 ≤ threshold) :
    (sensor old d threshold).1 = true ↔ d ≤ threshold := by simp [sensor, crossing, sq_le_sq₀ hd ht]

noncomputable def tracking (old hasTarget : Bool) (d range radius : ℝ) : Bool × Option Bool :=
  if 5902958103587057 / 590295810358705651712 ≤ range ∧ hasTarget then
    edge old (decide (d^2 < (range + radius)^2)) else (old, none)

theorem tracking_boundary (old : Bool) (range radius : ℝ)
    (hr : 5902958103587057 / 590295810358705651712 ≤ range) :
    tracking old true (range + radius) range radius = (false, if old then some false else none) := by
  cases old <;> simp [tracking, hr, edge]

noncomputable def followBand (d hi tolerance : ℝ) : Prop :=
  d^2 < hi^2 ∧ (max 0 (hi - 2*tolerance))^2 < d^2

theorem follow_band_iff (d hi tolerance : ℝ) (hd : 0 ≤ d) (hh : 0 ≤ hi) :
    followBand d hi tolerance ↔ max 0 (hi - 2*tolerance) < d ∧ d < hi := by
  rw [followBand, sq_lt_sq₀ hd hh, sq_lt_sq₀ (le_max_left 0 (hi - 2*tolerance)) hd, and_comm]

noncomputable def sensorClock (elapsed dt period : ℝ) : Bool × ℝ :=
  let next := elapsed + dt
  if next < period then (false, next) else (true, 0)

theorem sensor_schedule (elapsed dt period : ℝ) :
    (sensorClock elapsed dt period).1 = true ↔ period ≤ elapsed + dt := by
  unfold sensorClock; dsimp only; split <;> simp_all

end ExactBallpark.Space

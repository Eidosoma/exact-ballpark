-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Modes
public import ExactBallparkProofs.Movement.SpecialBranch

@[expose] public noncomputable section
namespace ExactBallpark.Movement

/-- A target at x=1 and requested one-tick travel of 1: no taper. Unit acceleration
holds terminal speed at 1. Starting at speed 2 gives a genuine ordinary-branch overshoot. -/
theorem goto_overshoot :
    taper 1 1 = 1 ∧ 1 < (integrate 0 2 1 1 1 1 (decay 1 1 1) 1).1 := by
  constructor
  · norm_num [taper]
  · rw [integrate_regular (by norm_num) (by norm_num [threshold]) (fun _ => rfl)]
    norm_num [position, terminal, decay]

def unit (v : Vec3) := if ‖v‖ = 0 then v else (1 / ‖v‖) • v

theorem unit_norm (v : Vec3) (h : v ≠ 0) : ‖unit v‖ = 1 := by
  simp [unit, norm_smul, norm_ne_zero_iff.mpr h]

def followTarget (target offset fallback : Vec3) (r : ℝ) :=
  if ‖offset‖ = 0 then target + r • fallback else target + (r * (1/‖offset‖)) • offset

theorem follow_range (target offset fallback : Vec3) (haxis : ‖fallback‖ = 1) (r : ℝ) :
    ‖followTarget target offset fallback r - target‖ = |r| := by
  unfold followTarget
  split
  · simp [norm_smul, Real.norm_eq_abs, haxis]
  next h =>
    simp [norm_smul, Real.norm_eq_abs, h]

/-- Inverse of the one-coordinate regular position and velocity updates, used by
GotoThrustFollow before blending and capping. f is friction / effective mass. -/
def reachAccel (p v dst f q t : ℝ) :=
  ((dst-p)*f - v*(1-q))*f * (1 / (t*f - (1-q)))

def matchAccel (v dstV f q : ℝ) := (dstV - v*q)*f * (1 / (1-q))

theorem solve_guidance (p v dst dstV f q t : ℝ) (hf : f ≠ 0) (hq : 1 - q ≠ 0)
    (ht : t*f - (1-q) ≠ 0) :
    regularPosition p v (reachAccel p v dst f q t) 1 f q t = dst ∧
    sourceVelocity v (matchAccel v dstV f q) 1 f q = dstV := by
  generalize hden : t * f - (1-q) = den at ht
  constructor
  · unfold reachAccel regularPosition
    rw [hden]
    field_simp
    rw [← hden]
    ring
  · unfold sourceVelocity matchAccel
    field_simp
    ring

def orbitBrake (v : Vec3) (limit : ℝ) : Vec3 :=
  if limit ≤ 0 then 0 else
  let ratio := ‖v‖ / limit
  let capped := if ratio > 1 then (1/ratio) • v else v
  if ratio < 3602879701896397 / 36028797018963968 then 0 else capped

theorem orbit_brake_cap (v : Vec3) {limit : ℝ} (hl : 0 < limit) :
    ‖orbitBrake v limit‖ ≤ limit := by
  unfold orbitBrake
  simp only [ite_eq_right (not_le.mpr hl)]
  split
  · simpa using hl.le
  split
  next h =>
    have hv : 0 < ‖v‖ := hl.trans ((one_lt_div hl).mp h)
    simp [norm_smul, div_div_eq_mul_div, hv.ne', Real.norm_eq_abs, abs_of_pos hl]
  next h => exact (div_le_one hl).mp (le_of_not_gt h)

def phaseAngles (id tick : ℕ) (step : ℝ) : ℝ × ℝ :=
  (tick * step, (id % 65536 : ℕ) + tick * step)

theorem orbit_id_period (id tick : ℕ) (step : ℝ) :
    phaseAngles (id + 65536) tick step = phaseAngles id tick step := by simp [phaseAngles]

/-- The Gaussian is positive, so _trunc7 uses the floor branch of truncation. -/
def orbitWeight (r dist : ℝ) :=
  (⌊Real.exp (-(r-dist)*(r-dist)/40000) * 10000000⌋ : ℝ) / 10000000

theorem orbit_weight_bounds (r dist : ℝ) :
    0 ≤ orbitWeight r dist ∧ orbitWeight r dist ≤ 1 := by
  have he : Real.exp (-(r-dist)*(r-dist)/40000) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith [sq_nonneg (r-dist)])
  have hp : 0 ≤ Real.exp (-(r-dist)*(r-dist)/40000) * 10000000 := by positivity
  constructor
  · unfold orbitWeight
    have hf := Int.floor_nonneg.mpr hp
    positivity
  · unfold orbitWeight
    have := Int.floor_le (Real.exp (-(r-dist)*(r-dist)/40000) * 10000000)
    linarith

/-- Local orbit-plane components of EvolveNewStyleOrbit's target displacement. -/
def spiralOffset (dist angle t : ℝ) : ℝ × ℝ :=
  let radius := dist * Real.exp (angle*t)
  (dist - radius * Real.cos t, radius * Real.sin t)

theorem spiral_radius (dist angle t : ℝ) :
    let gp := spiralOffset dist angle t
    (dist - gp.1)^2 + gp.2^2 = (dist * Real.exp (angle*t))^2 := by
  dsimp [spiralOffset]
  nlinarith only [show (dist * Real.exp (angle*t))^2 * ((Real.sin t)^2 + (Real.cos t)^2) =
    (dist * Real.exp (angle*t))^2 by rw [Real.sin_sq_add_cos_sq]; ring]

end ExactBallpark.Movement

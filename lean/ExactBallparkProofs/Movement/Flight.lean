-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Guidance
public import ExactBallparkProofs.Movement.Control

@[expose] public noncomputable section
namespace ExactBallpark.Movement
open ExactBallpark.Control

inductive Behavior where | thrust (m : Mode) | damp | inert deriving DecidableEq

def behavior (dynamic : Bool) : Mode → Behavior
  | m@.warp | m@.goto | m@.missile | m@.formation | m@.follow | m@.orbit => .thrust m
  | .stop => if dynamic then .inert else .damp
  | _ => .inert

theorem thrust_modes (dynamic : Bool) (m : Mode) :
    behavior dynamic m = .thrust m ↔ m ∈ [.warp, .goto, .missile, .formation, .follow, .orbit] := by
  cases m <;> cases dynamic <;> decide

def orbitLaw {α : Type} (useNew : Bool) (old new : α) := if useNew then new else old

theorem orbit_switch {α : Type} (old new : α) :
    orbitLaw false old new = old ∧ orbitLaw true old new = new := ⟨rfl, rfl⟩

def predictedOffset (pos target targetV : Vec3) (dt : ℝ) := target - pos + dt • targetV

theorem predicted_direction (p q v : Vec3) (dt : ℝ) (h : predictedOffset p q v dt ≠ 0) :
    ‖predictedOffset p q v dt‖ • unit (predictedOffset p q v dt) = q + dt • v - p := by
  rw [unit, ite_eq_right (norm_ne_zero_iff.mpr h), smul_smul, one_div, mul_inv_cancel₀ (norm_ne_zero_iff.mpr h), one_smul]
  unfold predictedOffset
  abel

def flightTarget (elapsedMs : ℤ) (stored selfPos target targetV axis : Vec3) (range r₁ r₂ speed : ℝ) :=
  if elapsedMs ≤ 800 then (stored, false) else
    let d := ‖selfPos - target‖
    let offset := range + r₁ + r₂
    (if d = 0 then target + offset • axis
     else target + (d / speed) • targetV + (offset / d) • (selfPos - target), true)

theorem pursuit_center (ms : ℤ) (stored p q v axis : Vec3) (r₁ r₂ speed : ℝ) :
    flightTarget ms stored p q v axis (-(r₁ + r₂)) r₁ r₂ speed =
      if ms ≤ 800 then (stored, false) else (q + (‖p - q‖ / speed) • v, true) := by
  simp [flightTarget]
  split <;> simp_all

def launchVelocity (aimed : Bool) (p target velocity stationaryDirection : Vec3) (maxVel : ℝ) :=
  if aimed then unit (target - p) else
  velocity + max 150 maxVel • (if unit velocity = 0 then stationaryDirection else unit velocity)

theorem launch_speeds (p q v axis : Vec3) (speed : ℝ) (ht : q ≠ p) (hv : v ≠ 0) :
    ‖launchVelocity true p q v axis speed‖ = 1 ∧
    ‖launchVelocity false p q v axis speed - v‖ = max 150 speed := by
  have u := unit_norm v hv
  have hn : unit v ≠ 0 := by intro h; simp [h] at u
  have hp : 0 ≤ max 150 speed := le_trans (by norm_num) (le_max_left _ _)
  simp [launchVelocity, hn, unit_norm _ (sub_ne_zero.mpr ht), norm_smul, u, Real.norm_eq_abs, abs_of_nonneg hp]

def formationTarget (rotate : Vec3 → Vec3) (leader slot : Vec3) (r₁ r₂ : ℝ) :=
  let d := if ‖slot‖ = 0 then 1 else ‖slot‖
  leader + rotate (((r₁ + r₂ + d) / d) • slot)

theorem formation_distance (rotate : Vec3 → Vec3) (iso : ∀ v, ‖rotate v‖ = ‖v‖)
    (p slot : Vec3) (r₁ r₂ : ℝ) :
    ‖formationTarget rotate p slot r₁ r₂ - p‖ =
      if slot = 0 then 0 else |r₁ + r₂ + ‖slot‖| := by
  by_cases h : slot = 0
  · simp [formationTarget, h, iso]
  · simp [formationTarget, h, norm_ne_zero_iff.mpr h, iso, norm_smul, Real.norm_eq_abs]

def orientedThrust (heading : Vec3) (accel cosine : ℝ) :=
  let c := min 1 cosine
  if c ≤ 0 then 0 else (c*c*accel) • heading

theorem orientation_bound (heading : Vec3) (axis : ‖heading‖ = 1) (cosine accel : ℝ) (ha : 0 ≤ accel) :
    ‖orientedThrust heading accel cosine‖ ≤ accel := by
  dsimp only [orientedThrust]
  split
  · simpa using ha
  next h =>
    have hc : 0 ≤ min 1 cosine := le_of_not_ge h
    have h1 : min 1 cosine ≤ 1 := min_le_left _ _
    rw [norm_smul, axis, mul_one, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    exact mul_le_of_le_one_left ha ((mul_le_of_le_one_left hc h1).trans h1)

def capAcceleration (a : Vec3) := a

theorem cap_preserves (a : Vec3) : capAcceleration a = a := rfl

def smoothAngle (old target mm : ℝ) := (old + mm * (1/2) * target) * (1 / (1 + mm * (1/2)))

theorem smooth_error (old target mm : ℝ) (h : 0 ≤ mm) :
    smoothAngle old target mm - target = (old - target) / (1 + mm / 2) := by
  unfold smoothAngle
  have hd : 0 < 1 + mm / 2 := by linarith
  rw [show mm * (1/2) = mm / 2 by ring]
  field_simp
  ring

def trollReady (mode : Mode) (start delay now : ℤ) :=
  mode = .troll ∧ ¬ start + max 1 delay > now

theorem troll_due (start delay elapsed : ℤ) :
    trollReady .troll start delay (start + elapsed) ↔ 1 ≤ elapsed ∧ delay ≤ elapsed := by
  simp [trollReady]

end ExactBallpark.Movement

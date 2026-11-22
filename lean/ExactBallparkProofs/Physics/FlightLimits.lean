-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.DragChord
public import ExactBallparkProofs.Movement.Modes

@[expose] public noncomputable section
namespace ExactBallpark.Physics.FlightLimits
open Movement Movement.DragChord Set

/-- The GOTO command is held for the sweep. The requested speed includes speed fraction. -/
theorem goto_chord (f : Flight) (valid : f.Valid) (offset : Vec3) {speed dt T u : ℝ}
    (hs : 0 ≤ speed) (cmd : f.applied = gotoAcceleration offset f.mass f.friction speed dt)
    (ht : 0 ≤ T) (hu : u ∈ Icc 0 1) :
    dist (f.pos (u*T)) (lerp f.start (f.pos T) u) ≤
      (f.friction / f.mass) * (speed + ‖f.initialVelocity‖) * T^2 / 8 := by
  have hterm : ‖terminal3 f.applied f.mass f.friction‖ ≤ speed := by
    rw [cmd]
    unfold gotoAcceleration
    apply goto_terminal_bound _ (sq_nonneg _) valid.1 valid.2 hs
    split
    next h => simp [h]
    next h => simp [norm_smul, h]
  have rate := div_pos valid.2 valid.1
  have cancel : f.friction / f.mass * (f.mass / f.friction) = 1 := by
    field_simp [valid.1.ne', valid.2.ne']
  have hforce : ‖f.applied‖ ≤ f.friction / f.mass * speed := by
    calc
      _ = ‖(f.friction / f.mass) • terminal3 f.applied f.mass f.friction‖ := by
        rw [terminal3, smul_smul, cancel, one_smul]
      _ ≤ _ := by
        rw [norm_smul, Real.norm_eq_abs, abs_of_pos rate]
        exact mul_le_mul_of_nonneg_left hterm rate.le
  have ha : f.initialAccel ≤ f.friction / f.mass * (speed + ‖f.initialVelocity‖) := by
    unfold Flight.initialAccel Flight.forceField
    grw [norm_sub_le, hforce, norm_smul, Real.norm_eq_abs, abs_of_pos rate]
    rw [mul_add]
  exact (flight_chord_deviation f valid ht hu).trans (by gcongr)

/-- On two ordinary constant-input flights, a time tolerance limits the change in
relative position by the sum of the two speed envelopes times the elapsed time. -/
theorem relative_travel (f g : Flight) (hf : f.Valid) (hg : g.Valid) {s t F G : ℝ}
    (hs : 0 ≤ s) (ht : s ≤ t)
    (fv : ‖f.initialVelocity‖ ≤ F) (ft : ‖terminal3 f.applied f.mass f.friction‖ ≤ F)
    (gv : ‖g.initialVelocity‖ ≤ G) (gt : ‖terminal3 g.applied g.mass g.friction‖ ≤ G) :
    ‖(f.pos t - g.pos t) - (f.pos s - g.pos s)‖ ≤ (F + G) * (t-s) := by
  apply norm_image_sub_le_of_norm_deriv_le_segment'
    (f := fun x => f.pos x - g.pos x) (f' := fun x => f.vel x - g.vel x)
    (b := t) _ _ t (right_mem_Icc.mpr ht)
  · exact fun x _ => ((position_derivative f hf x).sub (position_derivative g hg x)).hasDerivWithinAt
  · intro x hx
    exact (norm_sub_le ..).trans (add_le_add (speed_envelope hf.1 hf.2 (hs.trans hx.1) fv ft)
      (speed_envelope hg.1 hg.2 (hs.trans hx.1) gv gt))

/-- Adding a common initial velocity in a fixed drag medium leaves a decaying offset. -/
theorem fixed_medium_boost (p v a m k t b : ℝ) :
    velocity (v+b) a m k t - velocity v a m k t = decay m k t * b ∧
    position p (v+b) a m k t - position p v a m k t = m/k * (1-decay m k t) * b := by
  unfold velocity position
  constructor <;> ring

end ExactBallpark.Physics.FlightLimits

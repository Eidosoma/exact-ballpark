-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Physlib.Units.WithDim.Basic
public import ExactBallparkProofs.Collision.SearchWindow
public import ExactBallparkProofs.Movement.Integrator

@[expose] public section

namespace ExactBallpark.Physics
open Dimension

abbrev Seconds (α : Type) := WithDim T𝓭 α
abbrev Dimensionless (α : Type) := WithDim (1 : Dimension LTMCTDimensionBase) α

/-- Convert a fraction of the current sweep into an absolute time in the tick. -/
noncomputable def contactTime (start finish : Seconds ℝ) (u : Dimensionless ℝ) : Seconds ℝ :=
  start + (u * (finish - start)).cast (one_mul _)

theorem contact_time_value (start finish : Seconds ℝ) (u : Dimensionless ℝ) :
    (contactTime start finish u).val =
      Collision.SearchWindow.stamp start.val finish.val u.val := rfl

/-- On a positive-length window, stamping preserves and reflects strict contact order. -/
theorem contact_time_lt_iff {start finish : Seconds ℝ} (window : start < finish)
    (u v : Dimensionless ℝ) :
    contactTime start finish u < contactTime start finish v ↔ u < v := by
  have duration : 0 < finish.val - start.val := sub_pos.mpr (show start.val < finish.val from window)
  change start.val + u.val * (finish.val - start.val) < start.val + v.val * (finish.val - start.val)
    ↔ u.val < v.val
  rw [add_lt_add_iff_left, mul_lt_mul_iff_left₀ duration]

-- The raw comparison in the fixed-obstacle loop must fail to typecheck.
#guard_msgs (drop info) in
#check_failure fun (fraction : Dimensionless ℝ) (incumbent : Seconds ℝ) => fraction < incumbent

/-- The binary64 control computes the offset with the engine's multiplication order. -/
def earlier64 (start finish : Seconds Float) (fraction : Dimensionless Float)
    (held : Seconds Float) : Bool :=
  let offset : Seconds Float := ⟨fraction.val * (finish - start).val⟩
  let candidate := start + offset
  letI : Decidable (candidate < held) := inferInstanceAs (Decidable (candidate.val < held.val))
  decide (candidate < held)

#guard_msgs (drop info) in
#check_failure fun (fraction : Dimensionless Float) (incumbent : Seconds Float) => fraction < incumbent

theorem earlier64_eq (s e u held : Float) :
    earlier64 ⟨s⟩ ⟨e⟩ ⟨u⟩ ⟨held⟩ = decide (s + u * (e - s) < held) := rfl

def fixedReplaySeconds := Collision.SearchWindow.fixedReplayBy
  (fun s e u held => earlier64 ⟨s⟩ ⟨e⟩ ⟨u⟩ ⟨held⟩)

open Collision.SearchWindow in
/-- The two absolute contact times and both visiting orders in the COL-715 half-tick fixture. -/
theorem half_tick_absolute_selection :
    let start : Seconds ℝ := ⟨1 / 2⟩
    let finish : Seconds ℝ := ⟨1⟩
    let near := contactTime start finish ⟨1 / 4⟩
    let far := contactTime start finish ⟨1 / 2⟩
    near.val = 5 / 8 ∧ far.val = 3 / 4 ∧ near < far ∧
      shapeScan start.val finish.val [(3, 1 / 4), (4, 1 / 2)] = some (3, near.val) ∧
      shapeScan start.val finish.val [(4, 1 / 2), (3, 1 / 4)] = some (3, near.val) := by
  dsimp only
  norm_num [WithDim.lt_def, contact_time_value, shapeScan, takeHit, stamp]

open Collision.SearchWindow in
set_option maxRecDepth 16384 in
/-- Binary64 model of the same sweeps, with engine and seconds-based comparisons. -/
theorem half_tick_selection_words :
    fixedReplay [(3, 3), (4, 5)] = (4, 0x3fe8000000000000) ∧
    fixedReplay [(4, 5), (3, 3)] = (3, 0x3fe4000000000000) ∧
    fixedReplaySeconds [(3, 3), (4, 5)] = (3, 0x3fe4000000000000) ∧
    fixedReplaySeconds [(4, 5), (3, 3)] = (3, 0x3fe4000000000000) := by
  obtain ⟨_, forward, reverse⟩ := partial_tick_selection 0 1 0 0
  exact ⟨forward, reverse, by decide +kernel, by decide +kernel⟩

/-- For effective mass m, friction k has dimension mass/time, so −k Δt / m has dimension one. -/
noncomputable def dragExponent (k : WithDim (M𝓭 / T𝓭) ℝ) (dt : Seconds ℝ) (m : WithDim M𝓭 ℝ) :
    Dimensionless ℝ := (-(k * dt / m)).cast (by simp)

theorem drag_exponent_decay (k : WithDim (M𝓭 / T𝓭) ℝ) (dt : Seconds ℝ) (m : WithDim M𝓭 ℝ) :
    Real.exp (dragExponent k dt m).val = Movement.decay m.val k.val dt.val := by
  unfold Movement.decay
  congr 1
  change -(k.val * dt.val / m.val) = -k.val / m.val * dt.val
  ring

/-- The engine's positive-mass cache refresh uses this same dimensionless exponent. -/
theorem refresh_factor_units (mass : WithDim M𝓭 ℝ) (agility : Dimensionless ℝ)
    (k : WithDim (M𝓭 / T𝓭) ℝ) (dt : Seconds ℝ)
    (positive : 0 < Movement.effectiveMass mass.val agility.val) :
    Movement.refreshFactor mass.val agility.val k.val dt.val =
      Real.exp (dragExponent k dt ((mass * agility).cast (mul_one _))).val := by
  rw [Movement.refresh_positive positive, drag_exponent_decay]
  rfl

end ExactBallpark.Physics

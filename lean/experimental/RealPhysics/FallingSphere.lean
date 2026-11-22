-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Physics.NewtonDrag

@[expose] public section
namespace RealPhysics.FallingSphere
open ExactBallpark.Movement ExactBallpark.Physics.NewtonDrag

/-- The same SI formula can evaluate tabulated rational inputs exactly. -/
def stokesSpeed {K : Type} [Field K] (radius gravity solid fluid viscosity : K) : K :=
  2 * radius^2 * gravity * (solid - fluid) / (9 * viscosity)

theorem rational_speed (r g solid fluid eta : ℚ) :
    ((stokesSpeed (K := ℚ) r g solid fluid eta : ℚ) : ℝ) =
      stokesSpeed (r : ℝ) (g : ℝ) (solid : ℝ) (fluid : ℝ) (eta : ℝ) := by
  unfold stokesSpeed
  norm_cast

noncomputable section

/-- Downward-positive SI coordinates, a sphere in a quiescent Newtonian liquid,
with gravity, buoyancy and an unbounded-fluid Stokes drag law. -/
structure Setup where
  radius : ℝ
  solidDensity : ℝ
  fluidDensity : ℝ
  viscosity : ℝ
  gravity : ℝ
  radius_pos : 0 < radius
  solid_pos : 0 < solidDensity
  fluid_pos : 0 < fluidDensity
  viscosity_pos : 0 < viscosity
  gravity_pos : 0 < gravity

namespace Setup
variable (s : Setup)

def mass := (4/3 : ℝ) * Real.pi * s.radius^3 * s.solidDensity
def friction := 6 * Real.pi * s.viscosity * s.radius
def applied := s.gravity * (s.solidDensity - s.fluidDensity) / s.solidDensity
def speed := stokesSpeed s.radius s.gravity s.solidDensity s.fluidDensity s.viscosity

private theorem mass_pos : 0 < s.mass := by
  unfold mass
  positivity [s.radius_pos, s.solid_pos, Real.pi_pos]

private theorem friction_pos : 0 < s.friction := by
  unfold friction
  positivity [s.radius_pos, s.viscosity_pos, Real.pi_pos]

theorem terminal_speed : terminal s.mass s.friction s.applied = s.speed := by
  unfold terminal mass friction applied speed stokesSpeed
  field_simp [s.radius_pos.ne', s.solid_pos.ne', s.viscosity_pos.ne', Real.pi_ne_zero]
  ring

/-- If the bead is already settling at terminal speed, both source branches give
the same uniform motion. The stored full-tick factor may be arbitrary over the reals. -/
theorem terminal_step (p cached t dt : ℝ) :
    Reference.real p s.speed s.applied s.mass s.friction cached t dt =
      (p + s.speed * t, s.speed) := by
  apply balanced_source_step s.mass_pos s.friction_pos
  have h := s.terminal_speed
  unfold terminal at h
  field_simp [s.friction_pos.ne'] at h
  linarith

open Filter Topology in
/-- From rest, the real velocity update tends to the predicted settling speed.
Partial steps recompute the decay and full steps here have the matching cache. -/
theorem settling_from_rest (p dt : ℝ) :
    Tendsto (fun t => (Reference.real p 0 s.applied s.mass s.friction
      (decay s.mass s.friction dt) t dt).2) atTop (𝓝 s.speed) := by
  simp_rw [extracted_integrator, integrate_velocity s.friction_pos.ne' (fun _ => rfl)]
  rw [← s.terminal_speed]
  exact velocity_tends_to_terminal s.mass_pos s.friction_pos 0 s.applied

/-- Holding gravity, densities and viscosity fixed, scaling radius by q scales speed by q². -/
theorem speed_ratio (q : ℝ) (hq : 0 < q) :
    ({s with radius := q * s.radius, radius_pos := mul_pos hq s.radius_pos}).speed = q^2 * s.speed := by
  unfold speed stokesSpeed
  dsimp
  ring

end Setup
end
end RealPhysics.FallingSphere

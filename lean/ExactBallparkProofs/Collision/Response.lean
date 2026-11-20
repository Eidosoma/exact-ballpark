-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Bounds
public import Mathlib.Tactic.Positivity

@[expose] public noncomputable section
namespace ExactBallpark.Collision
open ExactBallpark.Movement
open scoped InnerProductSpace

def reflected (v n : Vec3) := v - (2 * ⟪v, n⟫_ℝ) • n

theorem reflection (v n : Vec3) (hn : ‖n‖ = 1) :
    ‖reflected v n‖ = ‖v‖ ∧ ⟪reflected v n, n⟫_ℝ = -⟪v, n⟫_ℝ := by
  constructor
  · apply (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
    rw [reflected, norm_sub_sq_real, real_inner_smul_right, norm_smul, Real.norm_eq_abs, hn,
      mul_one, sq_abs]; ring
  · rw [reflected, inner_sub_left, real_inner_smul_left, real_inner_self_eq_norm_sq, hn]; ring

/-- One direction of the gradient update, using the two base masses. -/
def exchanged (m n u v : ℝ) := (m*u - n*u + 2*n*v) / (m+n)

theorem elastic_exchange {m n : ℝ} (hm : 0 < m) (hn : 0 < n) (u v : ℝ) :
    let u' := exchanged m n u v
    let v' := exchanged n m v u
    m*u' + n*v' = m*u + n*v ∧ m*u'^2 + n*v'^2 = m*u^2 + n*v^2 ∧ u' - v' = v-u := by
  dsimp [exchanged]
  rw [add_comm n m]
  refine ⟨?_, ?_, ?_⟩ <;> field_simp <;> ring

/-- Iterative impact before AddImpact's limiter. The input speeds are projections
on the first ball's normal; the second ball receives the opposite impulse. -/
theorem iterative_exchange {m n : ℝ} (hm : 0 < m) (hn : 0 < n) (u v : ℝ) :
    let impulse := -2 * (u-v)
    let j := m * n * impulse / (m+n)
    (u + j / m, v - j / n) = (exchanged m n u v, exchanged n m v u) := by
  dsimp [exchanged]
  rw [add_comm n m]
  apply Prod.ext <;> dsimp <;> field_simp <;> ring

theorem overlap_split {m n : ℝ} (hm : 0 < m) (hn : 0 < n) (radius distance : ℝ) :
    let gap := radius - distance + 1
    let d₁ := n / (m+n) * gap
    let d₂ := m / (m+n) * gap
    distance + d₁ + d₂ = radius + 1 ∧ m*d₁ = n*d₂ := by
  dsimp
  constructor <;> field_simp; ring

/-- Nonzero dv, before rounding the binary32 numerator of AddImpact. -/
def limitImpact (v : Vec3) (maxVel : ℝ) :=
  let fraction := 25 * maxVel * maxVel / ‖v‖^2
  if fraction < 1 then fraction • v else v

theorem limitImpact_norm (v : Vec3) (maxVel : ℝ) (hv : v ≠ 0) (hm : 0 ≤ maxVel) :
    ‖limitImpact v maxVel‖ =
      if 5 * maxVel < ‖v‖ then (5 * maxVel)^2 / ‖v‖ else ‖v‖ := by
  have hn := norm_pos_iff.mpr hv
  have h : 25 * maxVel * maxVel / ‖v‖^2 < 1 ↔ 5 * maxVel < ‖v‖ := by
    rw [div_lt_one (sq_pos_of_pos hn)]
    convert sq_lt_sq₀ (show 0 ≤ 5 * maxVel by positivity) hn.le using 2; ring
  dsimp only [limitImpact]
  simp only [h]
  split_ifs
  · rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    field_simp; ring
  · rfl

/-- One component of FullBallpark._reflect after the two substeps have produced endV. -/
def collisionAccel (v g m k q endV : ℝ) :=
  -(((g * -m + g * (q*m)) - (v*q)*k) + endV*k) * (1/m) * (1/(q-1))

/-- Applying a fraction of that acceleration interpolates between the free tick
and the velocity obtained by reflecting at the contact and integrating the remainder. -/
theorem scaled_reflection (v g m k q endV scale : ℝ) (hm : m ≠ 0) (hk : k ≠ 0) (hq : q ≠ 1) :
    sourceVelocity v (g + scale * collisionAccel v g m k q endV) m k q =
      (1-scale) * sourceVelocity v g m k q + scale*endV := by
  unfold sourceVelocity collisionAccel
  have hden : q - 1 ≠ 0 := sub_ne_zero.mpr hq
  field_simp
  ring

inductive StaticAction where | skip | reflect | eject deriving DecidableEq

def staticAction (capsule : Bool) (time outward : ℝ) : StaticAction :=
  if time = -1 then .skip else if time > 0 then
    if capsule ∧ 0 < outward then .skip else .reflect
  else .eject

theorem static_branches (cap : Bool) (t outward : ℝ) (ht : t = -1 ∨ 0 ≤ t) :
    (staticAction cap t outward = .eject ↔ t = 0) ∧
    (staticAction cap t outward = .reflect ↔ 0 < t ∧ (cap = false ∨ outward ≤ 0)) := by
  cases cap <;> unfold staticAction <;> split_ifs <;> simp_all <;> grind

end ExactBallpark.Collision

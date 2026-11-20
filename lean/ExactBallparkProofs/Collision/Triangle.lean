-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.Response

@[expose] public noncomputable section
namespace ExactBallpark.Collision.Triangle
open ExactBallpark.Movement
open scoped InnerProductSpace

def support (centre n : Vec3) (r : ℝ) := centre - r • n

theorem support_geometry (centre n : Vec3) (r : ℝ) (unit : ‖n‖ = 1) (hr : 0 ≤ r) :
    ‖support centre n r - centre‖ = r ∧
      ⟪n, support centre n r⟫_ℝ = ⟪n, centre⟫_ℝ - r := by
  simp [support, sub_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_nonneg hr,
    inner_sub_right, real_inner_smul_right, unit]

def chord (p q : Vec3) (t : ℝ) := p + t • (q - p)

/-- Project the returned time and point. The source writes a time even for an out-of-range root. -/
def segmentPlane (p q n : Vec3) (d : ℝ) : Option ℝ × Option Vec3 :=
  let rate := ⟪n, q-p⟫_ℝ
  if rate = 0 then (none, none) else
  let t := (d - ⟪n,p⟫_ℝ) / rate
  (some t, if 0 ≤ t ∧ t ≤ 1 then some (chord p q t) else none)

theorem crossing_iff (p q n : Vec3) (d t : ℝ) (h : ⟪n,q-p⟫_ℝ ≠ 0) :
    segmentPlane p q n d = (some t, some (chord p q t)) ↔
      0 ≤ t ∧ t ≤ 1 ∧ ⟪n,chord p q t⟫_ℝ = d := by
  have root : (d - ⟪n,p⟫_ℝ) / ⟪n,q-p⟫_ℝ = t ↔ ⟪n,chord p q t⟫_ℝ = d := by
    rw [div_eq_iff h, chord, inner_add_right, real_inner_smul_right]
    constructor <;> intro eq <;> linarith
  simp only [segmentPlane, ite_eq_right h, Prod.mk.injEq, Option.some.injEq]
  constructor
  · rintro ⟨ht, point⟩
    rw [ht] at point
    split at point
    · exact ⟨by tauto, by tauto, root.mp ht⟩
    · contradiction
  · rintro ⟨lo, hi, eq⟩
    rw [root.mpr eq, ite_eq_left ⟨lo, hi⟩]
    exact ⟨rfl, rfl⟩

theorem parallel_sweep (p q n : Vec3) (d : ℝ) (h : ⟪n,q-p⟫_ℝ = 0) :
    segmentPlane p q n d = (none, none) ∧
      ∀ t : ℝ, ⟪n,chord p q t⟫_ℝ = d ↔ ⟪n,p⟫_ℝ = d := by
  simp [segmentPlane, chord, h, inner_add_right, real_inner_smul_right]

/-- Triangle.contains with e = c-a, f = b-a, x = point-a. -/
def containsOffsets (e f x : Vec3) : Prop :=
  let ee := ⟪e,e⟫_ℝ
  let ef := ⟪e,f⟫_ℝ
  let ff := ⟪f,f⟫_ℝ
  let inv := 1 / (ee*ff-ef*ef)
  let u := (ff*⟪e,x⟫_ℝ-ef*⟪f,x⟫_ℝ)*inv
  let v := (ee*⟪f,x⟫_ℝ-ef*⟪e,x⟫_ℝ)*inv
  0 ≤ u ∧ 0 ≤ v ∧ u+v < 1

theorem face_membership (e f : Vec3) (s t : ℝ)
    (det : ⟪e,e⟫_ℝ*⟪f,f⟫_ℝ-⟪e,f⟫_ℝ*⟪e,f⟫_ℝ ≠ 0) :
    containsOffsets e f (s • e + t • f) ↔ 0 ≤ s ∧ 0 ≤ t ∧ s+t < 1 := by
  simp only [containsOffsets, inner_add_right, real_inner_smul_right]
  rw [real_inner_comm e f]
  generalize ⟪e,e⟫_ℝ = a, ⟪e,f⟫_ℝ = b, ⟪f,f⟫_ℝ = c at *
  have u : (c*(s*a+t*b)-b*(s*b+t*c))*(1/(a*c-b*b)) = s := by rw [one_div, ← div_eq_mul_inv, div_eq_iff det]; ring
  have v : (a*(s*b+t*c)-b*(s*a+t*b))*(1/(a*c-b*b)) = t := by rw [one_div, ← div_eq_mul_inv, div_eq_iff det]; ring
  rw [u,v]

end ExactBallpark.Collision.Triangle

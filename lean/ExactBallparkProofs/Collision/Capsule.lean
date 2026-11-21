-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.Response
public import ExactBallparkProofs.Collision.Sweep
public import Init.Data.Float

/-! The iterative Capsule::CheckCollision path. Vectors are offsets from the
first end cap. The two radii have already been added. -/
@[expose] public noncomputable section
namespace ExactBallpark.Collision.Capsule
open ExactBallpark.Movement
open scoped InnerProductSpace

def along (axis x : Vec3) := ⟪x,axis⟫_ℝ / ‖axis‖^2
def perp (axis x : Vec3) := x - along axis x • axis

/-- The engine's perpendicular is the shortest offset to the axis line. The
squared distance to any other axial point includes exactly this axial penalty. -/
theorem axis_distance (axis x : Vec3) (hw : axis ≠ 0) (s : ℝ) :
    ⟪perp axis x,axis⟫_ℝ = 0 ∧
    ‖x-s • axis‖^2 = ‖perp axis x‖^2 + (along axis x-s)^2 * ‖axis‖^2 := by
  have nonzero : ‖axis‖^2 ≠ 0 := pow_ne_zero _ (norm_ne_zero_iff.mpr hw)
  have h : along axis x * ‖axis‖^2 = ⟪x,axis⟫_ℝ := div_mul_cancel₀ _ nonzero
  constructor
  · rw [perp,inner_sub_left,real_inner_smul_left,real_inner_self_eq_norm_sq,h,sub_self]
  · simp only [perp,norm_sub_sq_real,real_inner_smul_right,norm_smul,Real.norm_eq_abs,mul_pow,sq_abs]
    rw [← h]; ring

/-- Bounding-sphere prefilter before the short-axis branch. -/
def reject (sep path : Vec3) (r : ℝ) : Prop :=
  let h := ⟪sep,path⟫_ℝ
  (h < 0 ∧ r^2 < ‖sep‖^2) ∨ (‖path‖^2 < h ∧ r^2 < ‖sep-path‖^2) ∨
    ‖path‖^2 * ‖sep‖^2 - h*h > ‖path‖^2 * r^2

/-- Every excluded chord is strictly outside the supplied bounding sphere.
For an actual capsule this needs a radius enclosing the capsule and ball. -/
theorem bounding_rejection (sep path : Vec3) (r t : ℝ) (h : reject sep path r)
    (lo : 0 ≤ t) (hi : t ≤ 1) : r^2 < ‖sep-t • path‖^2 := by
  let b := ⟪sep,path⟫_ℝ
  let a := ‖path‖^2
  have az : 0 ≤ a := sq_nonneg _
  have expand (u : ℝ) : ‖sep-u • path‖^2 = ‖sep‖^2 - 2*u*b + u^2*a := by
    simp only [b,a,norm_sub_sq_real,real_inner_smul_right,norm_smul,Real.norm_eq_abs,mul_pow,sq_abs]
    ring
  change (b < 0 ∧ _) ∨ (a < b ∧ _) ∨ a*‖sep‖^2-b*b > a*r^2 at h
  rw [expand]
  rcases h with ⟨toward,far⟩ | ⟨past,far⟩ | line
  · nlinarith [mul_nonpos_of_nonneg_of_nonpos lo toward.le, mul_nonneg (sq_nonneg t) az]
  · have endSq := expand 1
    simp only [one_smul, mul_one,one_pow,one_mul] at endSq
    rw [endSq] at far
    have delta : 0 ≤ (1-t)*(2*b-(1+t)*a) :=
      mul_nonneg (by linarith) (by nlinarith [mul_nonneg (sub_nonneg.mpr hi) az])
    nlinarith
  · by_contra! near
    nlinarith [mul_le_mul_of_nonneg_left near az,sq_nonneg (t*a-b)]

private theorem perp_affine (axis x v : Vec3) (t : ℝ) :
    perp axis (x+t • v) = perp axis x+t • perp axis v := by
  have coeff : along axis (x+t • v) = along axis x+t*along axis v := by
    simp only [along,inner_add_left,real_inner_smul_left]; ring
  simp only [perp,coeff,add_smul,smul_sub,smul_smul]; abel

def radialGap (axis x v : Vec3) (radius t : ℝ) := ‖perp axis (x+t • v)‖^2-radius^2

/-- The source's quadratic measures squared distance to the infinite axis
minus the squared combined radius at every point on the chord. -/
theorem cylinder_quadratic (axis x v : Vec3) (r t : ℝ) :
    radialGap axis x v r t = ‖perp axis v‖^2*t^2 +
      2*⟪perp axis v,perp axis x⟫_ℝ*t + (‖perp axis x‖^2-r^2) := by
  rw [radialGap,perp_affine,norm_add_sq_real,real_inner_smul_right,norm_smul,Real.norm_eq_abs,
    mul_pow,sq_abs,real_inner_comm (perp axis x)]; ring

def barrelTime (axis x v : Vec3) (r : ℝ) :=
  quadraticTime (‖perp axis v‖^2) (2*⟪perp axis v,perp axis x⟫_ℝ) (‖perp axis x‖^2-r^2)

/-- On the admitted exterior quadratic branch, a root whose foot is on the
finite axis is first contact with the whole capsule. Every axial point,
including both hemispherical cap centers, was clear at earlier fractions. -/
theorem first_barrel_contact (axis x v : Vec3) (r : ℝ) (hw : axis ≠ 0) (hr : 0 ≤ r)
    (outside : r^2 < ‖perp axis x‖^2) (moving : 0 < ‖perp axis v‖^2)
    (accepted : 0 ≤ barrelTime axis x v r) (foot : along axis (x+barrelTime axis x v r • v) ∈ Set.Icc 0 1) :
    let t := barrelTime axis x v r
    t ∈ Set.Ioc 0 1 ∧ (∃ s ∈ Set.Icc (0:ℝ) 1, ‖x+t • v-s • axis‖ = r) ∧
    ∀ u, 0 ≤ u → u < t → ∀ s : ℝ, r < ‖x+u • v-s • axis‖ := by
  let t := barrelTime axis x v r
  have hc : FirstContact (radialGap axis x v r) t := by
    have eq := funext (cylinder_quadratic axis x v r)
    rw [eq]
    exact quadratic_correct moving.le (fun h => False.elim (moving.ne' h))
  rcases hc with ⟨miss,_⟩ | ⟨lo,hi,nonpos,zero,earlier⟩
  · change 0 ≤ t at accepted; linarith
  have tp : 0 < t := by
    rcases lo.eq_or_lt with eq | pos
    · rw [← eq] at nonpos; simp [radialGap] at nonpos; linarith
    · exact pos
  refine ⟨⟨tp,hi⟩, ⟨_,foot,?_⟩, ?_⟩
  · have z := zero tp
    change ‖perp axis (x+t • v)‖ = r
    unfold radialGap at z
    nlinarith [norm_nonneg (perp axis (x+t • v))]
  · intro u hu before s
    have d := (axis_distance axis (x+u • v) hw s).2
    have g := earlier u hu before
    unfold radialGap at g
    nlinarith [mul_nonneg (sq_nonneg (along axis (x+u • v)-s)) (sq_nonneg ‖axis‖),
      norm_nonneg (x+u • v-s • axis)]

/-- Projection of the real source's binary64 guards for a radius-1 combined
barrel around the z axis and a chord at y=z=0. Its broad radius is two. -/
def shallowGate (x y : Float) : Bool :=
  let v := y-x
  let h := (-x)*v
  let a := v*v
  !(h < 0 && 4 < x*x || a < h && 4 < y*y || a*(x*x)-h*h > a*4) &&
    x*x-1 > 0 && a < Float.ofBits 0x3ddb7cdfd9d7bdbb

/-- A true half-tick barrel crossing is discarded by the finite parallel
guard. The total normal displacement is exactly 2⁻¹⁹ m. -/
theorem shallow_crossing_missed :
    shallowGate (Float.ofBits 0x3ff0000100000000) (Float.ofBits 0x3feffffe00000000) = true ∧
    ∀ t : ℝ, 0 ≤ t → t ≤ 1 →
      (|1+1/2^20 - t/2^19| ≤ 1 ↔ 1/2 ≤ t) := by
  constructor
  · decide +kernel
  · intro t lo hi; rw [abs_le]; constructor <;> intro h
    · linarith [h.2]
    · constructor <;> linarith

end ExactBallpark.Collision.Capsule

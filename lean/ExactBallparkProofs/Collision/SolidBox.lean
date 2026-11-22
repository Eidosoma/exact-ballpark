-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Collision.Sweep
public import ExactBallparkProofs.Movement.Bounds

@[expose] public noncomputable section
namespace ExactBallpark.Collision.SolidBox
open Movement
open scoped InnerProductSpace

abbrev Interval := ℝ × ℝ

/-- One coordinate of AABB.update, including the else-if. -/
def grow (b : Interval) (x : ℝ) : Interval :=
  if x > b.2 then (b.1,x) else if x < b.1 then (x,b.2) else b

theorem grown_bounds (b : Interval) (valid : b.1 ≤ b.2) (xs : List ℝ) (l h : ℝ) :
    let out := xs.foldl grow b
    l ≤ out.1 ∧ out.2 ≤ h ↔ (l ≤ b.1 ∧ b.2 ≤ h) ∧ ∀ x ∈ xs, l ≤ x ∧ x ≤ h := by
  induction xs generalizing b with
  | nil => simp
  | cons x xs ih =>
    have ok : (grow b x).1 ≤ (grow b x).2 := by unfold grow; split_ifs <;> simp_all; linarith
    simp only [List.foldl_cons,ih _ ok,List.forall_mem_cons]
    unfold grow
    split_ifs <;> grind

/-- The seven corners visited after initializing both extrema to A. -/
def corners (a x y z : ℝ) := [a+x,a+y,a+z,a+x+y,a+y+z,a+x+z,a+x+y+z]
def box (a x y z : ℝ) := (corners a x y z).foldl grow (a,a)

theorem axis_extents (a x y z : ℝ) :
    box a x y z = (a+min 0 x+min 0 y+min 0 z,a+max 0 x+max 0 y+max 0 z) := by
  let l := a+min 0 x+min 0 y+min 0 z
  let h := a+max 0 x+max 0 y+max 0 z
  have enclosed : l ≤ (box a x y z).1 ∧ (box a x y z).2 ≤ h := by
    apply (grown_bounds (a,a) le_rfl (corners a x y z) l h).2
    dsimp [l,h,corners]; simp only [List.forall_mem_cons,List.not_mem_nil,IsEmpty.forall_iff,implies_true,and_true]
    repeat' constructor
    all_goals linarith [min_le_left 0 x,min_le_right 0 x,min_le_left 0 y,min_le_right 0 y,
      min_le_left 0 z,min_le_right 0 z,le_max_left 0 x,le_max_right 0 x,le_max_left 0 y,
      le_max_right 0 y,le_max_left 0 z,le_max_right 0 z]
  have C := (grown_bounds (a,a) le_rfl (corners a x y z) (box a x y z).1 (box a x y z).2).1 ⟨le_rfl,le_rfl⟩
  have ends : l ∈ a :: corners a x y z ∧ h ∈ a :: corners a x y z := by
    by_cases hx : 0 ≤ x <;> by_cases hy : 0 ≤ y <;> by_cases hz : 0 ≤ z <;>
      simp [l,h,corners,min_def,max_def,hx,hy,hz]
  have member (k : ℝ) (hk : k ∈ a :: corners a x y z) :
      (box a x y z).1 ≤ k ∧ k ≤ (box a x y z).2 := by
    rcases List.mem_cons.mp hk with rfl | hmem
    · exact C.1
    · exact C.2 k hmem
  exact Prod.ext (le_antisymm (member l ends.1).1 enclosed.1) (le_antisymm enclosed.2 (member h ends.2).2)

def point (a x y z : Vec3) (s t u : ℝ) := a + s • x + t • y + u • z

private theorem arm (x s : ℝ) (hs : s ∈ Set.Icc (0:ℝ) 1) : min 0 x ≤ s*x ∧ s*x ≤ max 0 x := by
  rcases hs with ⟨lo,hi⟩
  by_cases nonneg : 0 ≤ x
  · simp [min_eq_left nonneg,max_eq_right nonneg,mul_nonneg lo nonneg,mul_le_of_le_one_left nonneg hi]
  · have neg := le_of_not_ge nonneg
    simp [min_eq_right neg,max_eq_left neg,mul_nonpos_of_nonneg_of_nonpos lo neg]
    nlinarith [mul_nonneg (sub_nonneg.mpr hi) (neg_nonneg.mpr neg)]

theorem solid_enclosed (a x y z : Vec3) {s t u : ℝ} (hs : s ∈ Set.Icc (0:ℝ) 1)
    (ht : t ∈ Set.Icc (0:ℝ) 1) (hu : u ∈ Set.Icc (0:ℝ) 1) (i : Fin 3) :
    let b := box (a i) (x i) (y i) (z i)
    (point a x y z s t u) i ∈ Set.Icc b.1 b.2 := by
  rw [axis_extents]
  change _ ≤ a i+s*x i+t*y i+u*z i ∧ a i+s*x i+t*y i+u*z i ≤ _
  obtain ⟨xl,xh⟩ := arm (x i) s hs
  obtain ⟨yl,yh⟩ := arm (y i) t ht
  obtain ⟨zl,zh⟩ := arm (z i) u hu
  constructor <;> dsimp <;> linarith

/-- If the broad phase excludes a straight sphere sweep, even the closed solid
is out of reach for the entire interval. No orthogonality is needed here. -/
theorem excluded_solid (a x y z p q : Vec3) {r : ℝ} (_radius : 0 ≤ r)
    (reject : canExclude (fun i => (box (a i) (x i) (y i) (z i)).1)
      (fun i => (box (a i) (x i) (y i) (z i)).2) (fun i => p i) (fun i => q i) r)
    {s t u f : ℝ} (hs : s ∈ Set.Icc (0:ℝ) 1) (ht : t ∈ Set.Icc (0:ℝ) 1)
    (hu : u ∈ Set.Icc (0:ℝ) 1) (hf : f ∈ Set.Icc (0:ℝ) 1) :
    r < ‖(p + f • (q-p)) - point a x y z s t u‖ := by
  obtain ⟨i,side⟩ := excluded_chord _ _ _ _ r reject
  have inBox := solid_enclosed a x y z hs ht hu i
  let d := (p + f • (q-p)) - point a x y z s t u
  have bound : |d i| ≤ ‖d‖ := PiLp.norm_apply_le d i
  apply lt_of_lt_of_le _ bound
  change r < |p i+f*(q i-p i) - (point a x y z s t u) i|
  obtain ⟨lo,hi⟩ := inBox
  rcases side f hf.1 hf.2 with above | below
  · exact lt_of_lt_of_le (by dsimp [chord] at above; linarith) (le_abs_self _)
  · exact lt_of_lt_of_le (by dsimp [chord] at below; linarith) (neg_le_abs _)

/-- BoxShape.center and half the A-to-H diagonal bound a rectangular solid
with pairwise perpendicular arms, before the bounding-radius cast to binary32. -/
theorem diagonal_radius (a x y z : Vec3) (xy : ⟪x,y⟫_ℝ = 0) (xz : ⟪x,z⟫_ℝ = 0)
    (yz : ⟪y,z⟫_ℝ = 0) {s t u : ℝ} (hs : s ∈ Set.Icc (0:ℝ) 1)
    (ht : t ∈ Set.Icc (0:ℝ) 1) (hu : u ∈ Set.Icc (0:ℝ) 1) :
    ‖point a x y z s t u - (a+(1/2:ℝ) • (x+y+z))‖ ≤ ‖x+y+z‖/2 := by
  have diff : point a x y z s t u - (a+(1/2:ℝ) • (x+y+z)) =
      (s-1/2) • x + (t-1/2) • y + (u-1/2) • z := by simp only [point,sub_smul,smul_add]; abel
  have weight (d : ℝ) (hd : d ∈ Set.Icc (0:ℝ) 1) : (d-1/2)^2 ≤ 1/4 := by
    nlinarith [mul_nonneg hd.1 (sub_nonneg.mpr hd.2)]
  rw [diff, ← sq_le_sq₀ (norm_nonneg _) (by positivity : 0 ≤ ‖x+y+z‖/2),div_pow]
  simp only [norm_add_sq_real,inner_add_left,real_inner_smul_left,real_inner_smul_right,xy,xz,yz,
    mul_zero,add_zero,norm_smul,Real.norm_eq_abs,mul_pow,sq_abs]
  linarith [mul_le_mul_of_nonneg_right (weight s hs) (sq_nonneg ‖x‖),
    mul_le_mul_of_nonneg_right (weight t ht) (sq_nonneg ‖y‖),
    mul_le_mul_of_nonneg_right (weight u hu) (sq_nonneg ‖z‖)]

end ExactBallpark.Collision.SolidBox

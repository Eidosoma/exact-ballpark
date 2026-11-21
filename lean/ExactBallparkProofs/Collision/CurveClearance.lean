-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.DragChord
public import ExactBallparkProofs.Movement.RegularTrajectory
public import ExactBallparkProofs.Collision.SweepCertificate

@[expose] public section
namespace ExactBallpark.Collision.CurveClearance
open ExactBallpark.Movement ExactBallpark.Movement.DragChord Set

/-- Euclidean endpoint errors and an upper bound on initial net acceleration, in SI units. -/
structure Budget where
  accel : ℚ
  startError : ℚ
  endError : ℚ
  deriving Repr

def Budget.valid (b : Budget) : Bool := 0 ≤ b.accel && 0 ≤ b.startError && 0 ≤ b.endError

def Budget.allowance (b : Budget) (duration : ℚ) :=
  b.accel * duration^2 / 8 + max b.startError b.endError

noncomputable def Budget.covers (b : Budget) (f : Flight) (start finish : Vec3) (T : ℝ) : Prop :=
  f.initialAccel ≤ b.accel ∧ dist f.start start ≤ b.startError ∧ dist (f.pos T) finish ≤ b.endError

noncomputable def asVec (p : RatPoint) : Vec3 := WithLp.toLp 2 (realPoint p)

/-- Strict clearance on both exact endpoint chords excludes even a grazing touch
of the continuous spheres, for any nonnegative real duration. -/
theorem pair_no_contact (f g : Flight) (vf : f.Valid) (vg : g.Valid) {T radius : ℝ}
    (hT : 0 ≤ T) (clear : ∀ u ∈ Icc 0 1, radius + (f.initialAccel + g.initialAccel)*T^2/8 <
      dist (lerp f.start (f.pos T) u) (lerp g.start (g.pos T) u)) :
    ∀ u ∈ Icc 0 1, radius < dist (f.pos (u*T)) (g.pos (u*T)) := by
  intro u hu
  have df := flight_chord_deviation f vf hT hu
  have dg := flight_chord_deviation g vg hT hu
  have h := dist_triangle4 (lerp f.start (f.pos T) u) (f.pos (u*T))
    (g.pos (u*T)) (lerp g.start (g.pos T) u)
  rw [dist_comm (lerp f.start (f.pos T) u) (f.pos (u*T))] at h
  linarith [clear u hu]

private theorem lerp_error {x y x' y' : Vec3} {a b u : ℝ} (hu : u ∈ Icc 0 1)
    (hx : dist x x' ≤ a) (hy : dist y y' ≤ b) :
    dist (lerp x y u) (lerp x' y' u) ≤ max a b := by
  rw [dist_eq_norm] at hx hy
  rw [dist_eq_norm, show lerp x y u - lerp x' y' u =
    (1-u) • (x-x') + u • (y-y') by unfold lerp; module]
  have hn := norm_add_le ((1-u) • (x-x')) (u • (y-y'))
  rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
    abs_of_nonneg hu.1, abs_of_nonneg (sub_nonneg.mpr hu.2)] at hn
  apply hn.trans
  have left := mul_le_mul_of_nonneg_left (hx.trans (le_max_left a b)) (sub_nonneg.mpr hu.2)
  have right := mul_le_mul_of_nonneg_left (hy.trans (le_max_right a b)) hu.1
  linarith

private theorem body_envelope (f : Flight) (valid : f.Valid) {p q : Vec3} {T : ℚ}
    (hT : 0 ≤ T) (b : Budget) (h : b.covers f p q T) {u : ℝ} (hu : u ∈ Icc 0 1) :
    dist (f.pos (u*T)) (lerp p q u) ≤ (b.allowance T : ℝ) := by
  have curve := flight_chord_deviation f valid (by exact_mod_cast hT : (0:ℝ) ≤ T) hu
  have ends := lerp_error hu h.2.1 h.2.2
  have tri := dist_triangle (f.pos (u*T)) (lerp f.start (f.pos T) u) (lerp p q u)
  have accel := mul_le_mul_of_nonneg_right h.1 (sq_nonneg (T:ℝ))
  unfold Budget.allowance
  push_cast
  linarith

/-- A lower bound on the straight-chord separation survives both bodies' curvature
and the larger endpoint error for each body. Strict extra clearance excludes touching. -/
theorem pair_clearance (f g : Flight) (vf : f.Valid) (vg : g.Valid) {p₀ p₁ q₀ q₁ : Vec3}
    {T : ℚ} (hT : 0 ≤ T) (bp bq : Budget) (hp : bp.covers f p₀ p₁ T) (hq : bq.covers g q₀ q₁ T)
    {d : ℝ} (clear : ∀ u ∈ Icc 0 1, d + bp.allowance T + bq.allowance T ≤
      dist (lerp p₀ p₁ u) (lerp q₀ q₁ u)) :
    ∀ u ∈ Icc 0 1, d ≤ dist (f.pos (u*T)) (g.pos (u*T)) := by
  intro u hu
  have p := body_envelope f vf hT bp hp hu
  have q := body_envelope g vg hT bq hq hu
  have h := dist_triangle4 (lerp p₀ p₁ u) (f.pos (u*T)) (g.pos (u*T)) (lerp q₀ q₁ u)
  rw [dist_comm (lerp p₀ p₁ u) (f.pos (u*T))] at h
  linarith [clear u hu]

theorem position_error_from_axes (x y : Vec3) (b : Fin 3 → ℝ)
    (bounds : ∀ i, |x i - y i| ≤ b i) : dist x y ≤ b 0 + b 1 + b 2 := by
  have hb i := (abs_nonneg _).trans (bounds i)
  rw [dist_eq_norm, ← sq_le_sq₀ (norm_nonneg _) (by linarith [hb 0, hb 1, hb 2]),
    EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three]
  have hs i := sq_le_sq.mpr ((bounds i).trans (le_abs_self _))
  change (x 0-y 0)^2 + (x 1-y 1)^2 + (x 2-y 2)^2 ≤ _
  nlinarith [hs 0, hs 1, hs 2, mul_nonneg (hb 0) (hb 1), mul_nonneg (hb 0) (hb 2),
    mul_nonneg (hb 1) (hb 2)]

open RegularTrajectory in
/-- Lift the per-coordinate rounding budgets of the ordinary integrator to the Euclidean
endpoint error accepted by the clearance checker. Each list contains one coordinate's steps. -/
theorem rounded_endpoint_budget (steps : Fin 3 → List Interval) (x y budget : Fin 3 → Phase)
    (initial : ∀ i, Within (x i) (y i) (budget i)) (valid : ∀ i, ∀ s ∈ steps i, Valid s) :
    let result := fun i => (steps i).foldl withBudget (x i, budget i)
    let ideal := fun i => (steps i).foldl (fun state s => exactStep s state) (y i)
    dist (WithLp.toLp 2 (fun i => (result i).1.1) : Vec3) (WithLp.toLp 2 (fun i => (ideal i).1)) ≤
      (result 0).2.1 + (result 1).2.1 + (result 2).2.1 := by
  dsimp only
  apply position_error_from_axes _ _ (fun i => ((steps i).foldl withBudget (x i, budget i)).2.1)
  intro i
  exact (regular_trajectory_error (steps i) _ _ _ (initial i) (valid i)).1

/-- Minimum of a convex quadratic on [0,1]. Testing endpoint derivatives also covers
zero relative motion without dividing by zero. -/
def minimumSq (a b c : ℚ) : ℚ :=
  if 0 ≤ b then c else if 2*a+b ≤ 0 then a+b+c else c-b*b/(4*a)

private theorem minimum_le (a b c : ℚ) (ha : 0 ≤ a) {u : ℝ} (hu : u ∈ Icc 0 1) :
    (minimumSq a b c : ℝ) ≤ a*u^2+b*u+c := by
  have a₀ : (0:ℝ) ≤ a := by exact_mod_cast ha
  unfold minimumSq
  split
  next hb =>
    have b₀ : (0:ℝ) ≤ b := by exact_mod_cast hb
    linarith [mul_nonneg a₀ (sq_nonneg u), mul_nonneg b₀ hu.1]
  next hb =>
    split
    next h =>
      have right : 2*(a:ℝ)+b ≤ 0 := by exact_mod_cast h
      have au := mul_le_of_le_one_right a₀ hu.2
      push_cast
      nlinarith [mul_nonneg (sub_nonneg.mpr hu.2) (show 0 ≤ -(b:ℝ)-a*(1+u) by nlinarith)]
    next h =>
      have ap : 0 < a := by linarith
      have ap' : (0:ℝ) < a := by exact_mod_cast ap
      push_cast
      field_simp
      nlinarith [sq_nonneg (2*(a:ℝ)*u+b)]

/-- A metre value rounded downward to micrometres. Its square is checked before use. -/
def lowerMetres (sq : ℚ) : ℚ :=
  ((sq.num.toNat * 10^12 / sq.den).sqrt : ℚ) / 10^6

def inspect (s : ExactChord) (duration : ℚ) (p q : Budget) : Option ℚ :=
  let (a,b,c) := exactCoefficients {s with radius := 0}
  let sq := minimumSq a b c
  let lower := lowerMetres sq
  let required := s.radius + p.allowance duration + q.allowance duration
  if 0 ≤ s.radius ∧ 0 ≤ duration ∧ p.valid ∧ q.valid ∧ 0 ≤ a ∧ 0 ≤ lower ∧
      lower ^ 2 ≤ sq ∧ required < lower then some (lower-required) else none

/-- A nonfinite input, solver hit, invalid budget or insufficient margin gives no certificate. -/
def check (p₀ p₁ q₀ q₁ : FloatPoint) (r : Float) (duration : ℚ) (p q : Budget) : Option ℚ := do
  let s ← decodeChord p₀ p₁ q₀ q₁ r
  if (contact64 p₀ p₁ q₀ q₁ r).toBits = 0xbff0000000000000 then inspect s duration p q else none

private theorem squared_distance (s : ExactChord) (u : ℝ) :
    let (a,b,c) := exactCoefficients {s with radius := 0}
    dist (lerp (asVec s.p₀) (asVec s.p₁) u) (lerp (asVec s.q₀) (asVec s.q₁) u)^2 =
      a*u^2 + b*u + c := by
  have h := decoded_gap {s with radius := 0} u
  dsimp only at h ⊢
  rw [← h, dist_eq_norm, EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three]
  dsimp [lerp, asVec, gap, chord, dot]
  ring

theorem certified_clearance {p₀ p₁ q₀ q₁ : FloatPoint} {r : Float} {T margin : ℚ}
    {bp bq : Budget} (accepted : check p₀ p₁ q₀ q₁ r T bp bq = some margin) :
    (contact64 p₀ p₁ q₀ q₁ r).toBits = 0xbff0000000000000 ∧ 0 < margin ∧
    ∃ s, decodeChord p₀ p₁ q₀ q₁ r = some s ∧ ∀ (f g : Flight), f.Valid → g.Valid →
      bp.covers f (asVec s.p₀) (asVec s.p₁) T → bq.covers g (asVec s.q₀) (asVec s.q₁) T →
      ∀ u ∈ Icc 0 1, (s.radius + margin : ℚ) ≤ dist (f.pos (u*T)) (g.pos (u*T)) := by
  unfold check at accepted
  obtain ⟨s, decoded, accepted⟩ := Option.bind_eq_some_iff.mp accepted
  split at accepted
  next miss =>
    unfold inspect at accepted
    generalize coeff : exactCoefficients {s with radius := 0} = abc at accepted
    obtain ⟨a,b,c⟩ := abc
    dsimp only at accepted
    split at accepted
    next conditions =>
      obtain ⟨_, hT, _, _, ha, hl, hl2, sep⟩ := conditions
      cases accepted
      refine ⟨miss, sub_pos.mpr sep, s, decoded, ?_⟩
      intro f g vf vg hp hq
      apply pair_clearance f g vf vg hT bp bq hp hq
      intro u hu
      have hs := squared_distance s u
      rw [coeff] at hs
      have hm := minimum_le a b c ha hu
      have hsq : ((lowerMetres (minimumSq a b c) : ℚ) : ℝ)^2 ≤ (minimumSq a b c : ℝ) := by
        exact_mod_cast hl2
      have lo := (sq_le_sq₀ (by exact_mod_cast hl : (0:ℝ) ≤ lowerMetres (minimumSq a b c))
        (dist_nonneg : 0 ≤ dist _ _)).mp (hsq.trans (hm.trans_eq hs.symm))
      push_cast
      linarith
    next => contradiction
  next => contradiction

end ExactBallpark.Collision.CurveClearance

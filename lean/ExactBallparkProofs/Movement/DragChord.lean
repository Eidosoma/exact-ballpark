-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Energy
public import Mathlib.Analysis.ODE.ExistUnique
public import Mathlib.Tactic.Module

@[expose] public noncomputable section
namespace ExactBallpark.Movement.DragChord
open Set

/-- Initial state and constant inputs for one ordinary drag interval. -/
structure Flight where
  start : Vec3
  initialVelocity : Vec3
  applied : Vec3
  /-- Effective mass (mass times agility) passed to the movement integrator. -/
  mass : ℝ
  friction : ℝ

def Flight.Valid (f : Flight) := 0 < f.mass ∧ 0 < f.friction

def Flight.pos (f : Flight) (t : ℝ) : Vec3 :=
  WithLp.toLp 2 (fun i => position (f.start i) (f.initialVelocity i) (f.applied i) f.mass f.friction t)

def Flight.vel (f : Flight) := velocity3 f.initialVelocity f.applied f.mass f.friction

def Flight.forceField (f : Flight) (v : Vec3) := f.applied - (f.friction / f.mass) • v

def Flight.initialAccel (f : Flight) := ‖f.forceField f.initialVelocity‖

@[simp] theorem pos_zero (f : Flight) : f.pos 0 = f.start := by
  ext i
  exact congrArg Prod.fst (flow_zero (f.start i) (f.initialVelocity i) (f.applied i) f.mass f.friction)

@[simp] theorem vel_zero (f : Flight) : f.vel 0 = f.initialVelocity := by
  ext i
  exact congrArg Prod.snd (flow_zero 0 (f.initialVelocity i) (f.applied i) f.mass f.friction)

theorem position_derivative (f : Flight) (valid : f.Valid) (t : ℝ) :
    HasDerivAt f.pos (f.vel t) t := by
  have h := hasDerivAt_pi.mpr (fun i : Fin 3 =>
    position_hasDerivAt valid.1.ne' valid.2.ne' (f.start i) (f.initialVelocity i) (f.applied i) t)
  exact (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin 3 => ℝ)).symm.toContinuousLinearMap
    |>.hasFDerivAt.comp_hasDerivAt t h

theorem extracted_position (f : Flight) {dt cached t : ℝ} (valid : f.Valid)
    (ordinary : ¬ f.friction < threshold * f.mass * t)
    (cache : t = dt → cached = decay f.mass f.friction dt) :
    WithLp.toLp 2 (fun i : Fin 3 => (Reference.real (f.start i) (f.initialVelocity i)
      (f.applied i) f.mass f.friction cached t dt).1) = f.pos t := by
  ext i
  exact congrArg Prod.fst ((extracted_integrator ..).trans
    (integrate_regular valid.2.ne' ordinary cache ..))

/-- Uniqueness needs only continuity at the end and the right derivative before it. -/
theorem trajectory_unique (f : Flight) (valid : f.Valid) (T : ℝ) (p v : ℝ → Vec3)
    (pc : ContinuousOn p (Icc 0 T)) (vc : ContinuousOn v (Icc 0 T))
    (dp : ∀ t ∈ Ico 0 T, HasDerivWithinAt p (v t) (Ici t) t)
    (dv : ∀ t ∈ Ico 0 T, HasDerivWithinAt v (f.forceField (v t)) (Ici t) t)
    (p₀ : p 0 = f.start) (v₀ : v 0 = f.initialVelocity) :
    EqOn p f.pos (Icc 0 T) ∧ EqOn v f.vel (Icc 0 T) := by
  let c := f.friction / f.mass
  have lip : LipschitzWith ‖c‖₊ f.forceField := by
    apply lipschitzWith_iff_dist_le_mul.mpr
    intro x y
    rw [dist_eq_norm, dist_eq_norm, show f.forceField x - f.forceField y = -(c • (x-y)) by
      unfold Flight.forceField; dsimp [c]; module, norm_neg, norm_smul]
    rfl
  have vel' (t : ℝ) : HasDerivAt f.vel (f.forceField (f.vel t)) t :=
    velocity3_hasDerivAt _ _ valid.1 valid.2 t
  have heq : EqOn v f.vel (Icc 0 T) := ODE_solution_unique (v := fun _ => f.forceField)
    (fun _ => lip) vc dv (fun t _ => (vel' t).continuousAt.continuousWithinAt)
    (fun t _ => (vel' t).hasDerivWithinAt) (v₀.trans (vel_zero f).symm)
  refine ⟨?_, heq⟩
  apply ODE_solution_unique (v := fun t (_ : Vec3) => f.vel t) (K := 0)
    (fun t => LipschitzWith.const (f.vel t)) pc _ (fun t _ => (position_derivative f valid t).continuousAt.continuousWithinAt)
    (fun t _ => (position_derivative f valid t).hasDerivWithinAt) (p₀.trans (pos_zero f).symm)
  intro t ht
  rw [← heq (Ico_subset_Icc_self ht)]
  exact dp t ht

/-- Drag scales the entire initial net acceleration by the exact exponential decay. -/
theorem acceleration_bound (f : Flight) (valid : f.Valid) {t : ℝ} (ht : 0 ≤ t) :
    ‖f.forceField (f.vel t)‖ = decay f.mass f.friction t * f.initialAccel ∧
    ‖f.forceField (f.vel t)‖ ≤ f.initialAccel := by
  have eqv : f.forceField (f.vel t) = decay f.mass f.friction t • f.forceField f.initialVelocity := by
    ext i
    change f.applied i - f.friction / f.mass * velocity _ _ _ _ _ =
      decay _ _ _ * (f.applied i - f.friction / f.mass * f.initialVelocity i)
    unfold velocity terminal
    field_simp [valid.1.ne', valid.2.ne']
    ring
  obtain ⟨pos, one⟩ := decay_bounds valid.1 valid.2 ht
  rw [eqv, norm_smul, Real.norm_eq_abs, abs_of_pos pos]
  exact ⟨rfl, mul_le_of_le_one_left (norm_nonneg _) one⟩

/-- The same-time endpoint chord at tick fraction u. -/
def lerp (x y : Vec3) (u : ℝ) := (1-u) • x + u • y

/-- Grow both endpoints from the time of interest until they span the whole tick.
The derivative of their weighted mean is controlled by the difference in velocities. -/
theorem chord_deviation (p v acc : ℝ → Vec3) {T A u : ℝ} (hT : 0 ≤ T)
    (hu : u ∈ Icc 0 1) (dp : ∀ x ∈ Icc 0 T, HasDerivAt p (v x) x)
    (dv : ∀ x ∈ Icc 0 T, HasDerivAt v (acc x) x) (ha : ∀ x ∈ Icc 0 T, ‖acc x‖ ≤ A) :
    dist (p (u*T)) (lerp (p 0) (p T) u) ≤ A * T^2 / 8 := by
  let l (x : ℝ) := u*(T-x)
  let r (x : ℝ) := u*T+(1-u)*x
  let w := u*(1-u)
  have hw : 0 ≤ w := mul_nonneg hu.1 (sub_nonneg.mpr hu.2)
  have hl {x : ℝ} (hx : x ∈ Icc 0 T) : l x ∈ Icc 0 T :=
    ⟨mul_nonneg hu.1 (sub_nonneg.mpr hx.2),
     (mul_le_of_le_one_left (sub_nonneg.mpr hx.2) hu.2).trans (sub_le_self _ hx.1)⟩
  have hr {x : ℝ} (hx : x ∈ Icc 0 T) : r x ∈ Icc 0 T := by
    dsimp [r]
    exact ⟨add_nonneg (mul_nonneg hu.1 hT) (mul_nonneg (sub_nonneg.mpr hu.2) hx.1),
      by nlinarith [mul_le_mul_of_nonneg_left hx.2 (sub_nonneg.mpr hu.2)]⟩
  let d (x : ℝ) := lerp (p (l x)) (p (r x)) u - p (u*T)
  have hd (x : ℝ) (hx : x ∈ Icc 0 T) :
      HasDerivAt d (w • (v (r x) - v (l x))) x := by
    have dl : HasDerivAt l (-u) x := by
      simpa [l] using ((hasDerivAt_id x).const_sub T).const_mul u
    have dr : HasDerivAt r (1-u) x := by
      simpa [r] using (((hasDerivAt_id x).const_mul (1-u)).const_add (u*T))
    have h := (((dp _ (hl hx)).scomp x dl).const_smul (1-u)).add
      (((dp _ (hr hx)).scomp x dr).const_smul u)
    unfold d lerp
    convert h.sub_const (p (u*T)) using 1
    dsimp [w]; module
  let B (x : ℝ) := w*A*x^2/2
  have dB (x : ℝ) : HasDerivAt B (w*A*x) x := by
    convert (((hasDerivAt_id x).pow 2).const_mul (w*A)).div_const 2 using 1 <;> simp [B]
    ring
  have init : ‖d 0‖ ≤ B 0 := by
    simp [d, lerp, l, r, ← add_smul, B]
  have derivBound (x : ℝ) (hx : x ∈ Ico 0 T) : ‖w • (v (r x) - v (l x))‖ ≤ w*A*x := by
    have inside := Ico_subset_Icc_self hx
    have mv := (convex_Icc (0:ℝ) T).norm_image_sub_le_of_norm_hasDerivWithin_le
      (fun y hy => (dv y hy).hasDerivWithinAt) ha (hl inside) (hr inside)
    have dx : r x - l x = x := by dsimp [r,l]; ring
    rw [dx, Real.norm_eq_abs, abs_of_nonneg hx.1] at mv
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hw]
    simpa only [mul_assoc] using mul_le_mul_of_nonneg_left mv hw
  have bound := image_norm_le_of_norm_deriv_right_le_deriv_boundary
    (fun x hx => (hd x hx).continuousAt.continuousWithinAt)
    (fun x hx => (hd x (Ico_subset_Icc_self hx)).hasDerivWithinAt) init dB derivBound (x := T) ⟨hT,le_rfl⟩
  have finish : r T = T := by dsimp [r]; ring
  rw [dist_comm, dist_eq_norm]
  have first : ‖lerp (p 0) (p T) u - p (u*T)‖ ≤ w*A*T^2/2 := by
    simpa [d, l, finish, B] using bound
  apply first.trans
  have quarter : w ≤ 1/4 := by dsimp [w]; nlinarith [sq_nonneg (u-1/2)]
  have hA := (norm_nonneg _).trans (ha 0 ⟨le_rfl,hT⟩)
  nlinarith [mul_le_mul_of_nonneg_right quarter (mul_nonneg hA (sq_nonneg T))]

theorem flight_chord_deviation (f : Flight) (valid : f.Valid) {T u : ℝ} (hT : 0 ≤ T)
    (hu : u ∈ Icc 0 1) :
    dist (f.pos (u*T)) (lerp f.start (f.pos T) u) ≤ f.initialAccel * T^2/8 := by
  rw [← pos_zero f]
  exact chord_deviation f.pos f.vel (fun t => f.forceField (f.vel t)) hT hu
    (fun t _ => position_derivative f valid t) (fun t _ => velocity3_hasDerivAt _ _ valid.1 valid.2 t)
    (fun t ht => (acceleration_bound f valid ht.1).2)

end ExactBallpark.Movement.DragChord

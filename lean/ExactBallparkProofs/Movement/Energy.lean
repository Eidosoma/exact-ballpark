-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Bounds
public import ExactBallparkProofs.Movement.Correspondence
public import Mathlib.Analysis.Calculus.Deriv.MeanValue
public import Mathlib.Analysis.Calculus.Deriv.Prod
public import Mathlib.Analysis.InnerProductSpace.Calculus
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

@[expose] public noncomputable section

namespace ExactBallpark.Movement
open scoped InnerProductSpace

/-- Kinetic energy using the effective mass from the movement equation. -/
def kineticEnergy (m : ℝ) (v : Vec3) : ℝ := m / 2 * ‖v‖ ^ 2

/-- Both position branches return the same real velocity when the full-tick cache is valid. -/
theorem extracted_velocity3 (p v a : Vec3) {m k dt cached t : ℝ} (hk : k ≠ 0)
    (cache : t = dt → cached = decay m k dt) :
    WithLp.toLp 2 (fun i : Fin 3 => (Reference.real (p i) (v i) (a i) m k cached t dt).2) =
      velocity3 v a m k t := by
  ext i
  exact (congrArg Prod.snd (extracted_integrator ..)).trans (integrate_velocity hk cache ..)

theorem velocity3_hasDerivAt (v a : Vec3) {m k : ℝ} (hm : 0 < m) (hk : 0 < k) (t : ℝ) :
    HasDerivAt (velocity3 v a m k) (a - (k / m) • velocity3 v a m k t) t := by
  have coords : HasDerivAt (fun s (i : Fin 3) => velocity (v i) (a i) m k s)
      (fun i => a i - k / m * velocity (v i) (a i) m k t) t :=
    hasDerivAt_pi.mpr (fun i => velocity_hasDerivAt hm.ne' hk.ne' (v i) (a i) t)
  exact (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin 3 => ℝ)).symm.toContinuousLinearMap
    |>.hasFDerivAt.comp_hasDerivAt t coords

/-- Thrust supplies power m ⟪v,a⟫ and viscous drag removes power k ‖v‖². -/
theorem drag_energy_rate (v a : Vec3) {m k : ℝ} (hm : 0 < m) (hk : 0 < k) (t : ℝ) :
    HasDerivAt (fun s => kineticEnergy m (velocity3 v a m k s))
      (m * ⟪velocity3 v a m k t, a⟫_ℝ - k * ‖velocity3 v a m k t‖ ^ 2) t := by
  have h := (velocity3_hasDerivAt v a hm hk t).norm_sq.const_mul (m / 2)
  unfold kineticEnergy
  convert h using 1
  rw [inner_sub_right, real_inner_smul_right, real_inner_self_eq_norm_sq]
  field_simp

/-- Integrating the net power gives the change in kinetic energy over any interval. -/
theorem drag_work_energy (v a : Vec3) {m k : ℝ} (hm : 0 < m) (hk : 0 < k) (s e : ℝ) :
    kineticEnergy m (velocity3 v a m k e) - kineticEnergy m (velocity3 v a m k s) =
      ∫ t in s..e, m * ⟪velocity3 v a m k t, a⟫_ℝ - k * ‖velocity3 v a m k t‖ ^ 2 := by
  have hv : Continuous (velocity3 v a m k) :=
    continuous_iff_continuousAt.mpr (fun t => (velocity3_hasDerivAt v a hm hk t).continuousAt)
  have power : Continuous
      (fun t => m * ⟪velocity3 v a m k t, a⟫_ℝ - k * ‖velocity3 v a m k t‖ ^ 2) :=
    (continuous_const.mul (hv.inner continuous_const)).sub (continuous_const.mul (hv.norm.pow 2))
  exact (intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => drag_energy_rate v a hm hk t)
    (power.intervalIntegrable s e)).symm

/-- With zero applied acceleration, the kinetic energy is scaled by the squared decay. -/
theorem coast_energy (v : Vec3) (m k t : ℝ) :
    kineticEnergy m (velocity3 v 0 m k t) = decay m k t ^ 2 * kineticEnergy m v := by
  have speed := velocity_contraction v 0 m k t
  simp only [terminal3, smul_zero, sub_zero] at speed
  rw [kineticEnergy, speed, kineticEnergy]
  ring

/-- A moving, unforced ball loses energy strictly between any two increasing times. -/
theorem coast_energy_strictAnti {v : Vec3} (hv : v ≠ 0) {m k : ℝ} (hm : 0 < m) (hk : 0 < k) :
    StrictAnti (fun t => kineticEnergy m (velocity3 v 0 m k t)) := by
  apply strictAnti_of_deriv_neg
  intro t
  have speed := velocity_contraction v 0 m k t
  simp only [terminal3, smul_zero, sub_zero] at speed
  have pos : 0 < ‖velocity3 v 0 m k t‖ :=
    speed ▸ mul_pos (Real.exp_pos _) (norm_pos_iff.mpr hv)
  rw [(drag_energy_rate v 0 hm hk t).deriv, inner_zero_right, mul_zero, zero_sub]
  exact neg_lt_zero.mpr (mul_pos hk (sq_pos_of_pos pos))

end ExactBallpark.Movement

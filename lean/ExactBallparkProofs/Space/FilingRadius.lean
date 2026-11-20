-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module
public import ExactBallparkProofs.Movement.Bounds

@[expose] public noncomputable section
namespace ExactBallpark.Space
open ExactBallpark.Movement

/-- Free-ball branch of GetInflatedRadius. S is speedFraction times maxVel. -/
def filedRadius (radius speed m k S cached dt : ℝ) :=
  radius + dt * ((m * (k*S/m) - (m * (k*S/m) - speed*k)*cached)/k)

/-- Displacement of the regular trajectory, component by component. -/
def traveled (v a : Vec3) (m k t : ℝ) : Vec3 :=
  WithLp.toLp 2 (fun i => position 0 (v i) (a i) m k t)

theorem radius_covers_trajectory (v a : Vec3) (m k S dt t radius q : ℝ)
    (hm : 0 < m) (hk : 0 < k) (t₀ : 0 ≤ t) (within : t ≤ dt)
    (cache : q = decay m k dt) (speed : ‖v‖ ≤ S) (thrust : ‖terminal3 a m k‖ ≤ S) :
    ‖traveled v a m k t‖ + radius ≤ filedRadius radius ‖v‖ m k S q dt := by
  have dt₀ := t₀.trans within
  have ⟨qpos,qle⟩ := decay_bounds hm hk t₀
  have ⟨endpos,endle⟩ := decay_bounds hm hk dt₀
  let d := m/k * (1-decay m k t)
  have d₀ : 0 ≤ d := mul_nonneg (div_nonneg hm.le hk.le) (sub_nonneg.mpr qle)
  have d₁ : d ≤ t := by
    have expBound := Real.add_one_le_exp ((-k/m)*t)
    change m/k*(1-decay m k t) ≤ t
    unfold decay
    field_simp [hm.ne', hk.ne'] at expBound ⊢
    nlinarith
  have down : decay m k dt ≤ decay m k t := by
    apply Real.exp_le_exp.mpr
    exact mul_le_mul_of_nonpos_left within (div_nonpos_of_nonpos_of_nonneg (by linarith) hm.le)
  have d₂ : t * decay m k dt ≤ d := by
    apply le_trans (mul_le_mul_of_nonneg_left down t₀)
    have expBound := Real.add_one_le_exp ((k/m)*t)
    have cancel : Real.exp ((k/m)*t) * decay m k t = 1 := by
      rw [decay, ← Real.exp_add]
      simp only [Real.exp_eq_one_iff]
      ring
    have bound := mul_le_mul_of_nonneg_right expBound qpos.le
    rw [cancel] at bound
    change t * decay m k t ≤ m/k*(1-decay m k t)
    field_simp [hm.ne',hk.ne'] at bound ⊢
    linarith
  have form : traveled v a m k t = d • v + (t-d) • terminal3 a m k := by
    ext i
    change position 0 (v i) (a i) m k t = d * v i + (t-d)*(m/k*a i)
    unfold position terminal
    dsimp [d]
    ring
  have eff : filedRadius radius ‖v‖ m k S q dt =
      radius + dt*(decay m k dt * ‖v‖ + (1-decay m k dt)*S) := by
    unfold filedRadius
    rw [cache]
    field_simp
    ring
  rw [eff,form]
  have triangle := norm_add_le (d • v) ((t-d) • terminal3 a m k)
  rw [norm_smul,norm_smul,Real.norm_eq_abs,Real.norm_eq_abs,abs_of_nonneg d₀,
    abs_of_nonneg (sub_nonneg.mpr d₁)] at triangle
  have th := mul_le_mul_of_nonneg_left thrust (sub_nonneg.mpr d₁)
  have envelope := mul_nonneg (sub_nonneg.mpr d₂) (sub_nonneg.mpr speed)
  have vmax := add_nonneg (mul_nonneg endpos.le (norm_nonneg v))
    (mul_nonneg (sub_nonneg.mpr endle) ((norm_nonneg v).trans speed))
  have advance := mul_le_mul_of_nonneg_right within vmax
  nlinarith

end ExactBallpark.Space

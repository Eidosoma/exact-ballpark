-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import ExactBallparkProofs.Movement.Bounds
public import Mathlib.Tactic.Positivity
public import Mathlib.Tactic.NormNum

@[expose] public section

namespace ExactBallpark.Movement

noncomputable section

def taper (lengthSq dist : ℝ) :=
  if lengthSq < dist * dist then (lengthSq / (dist * dist)) ^ 2 else 1

theorem taper_bounds {lengthSq : ℝ} (hd : 0 ≤ lengthSq) (dist : ℝ) :
    0 ≤ taper lengthSq dist ∧ taper lengthSq dist ≤ 1 := by
  unfold taper
  split
  next h =>
    have hd2 : 0 < dist * dist := lt_of_le_of_lt hd h
    have h₁ := (div_lt_one hd2).mpr h
    have h₀ := div_nonneg hd hd2.le
    exact ⟨sq_nonneg _, by nlinarith [mul_nonneg h₀ (sub_nonneg.mpr h₁.le)]⟩
  next => norm_num

/-- `direction` is the normalized offset (or zero at the target). -/
theorem goto_terminal_bound {direction : Vec3} (hdir : ‖direction‖ ≤ 1)
    {lengthSq m k speed : ℝ} (hl : 0 ≤ lengthSq) (hm : 0 < m) (hk : 0 < k)
    (hs : 0 ≤ speed) (dt : ℝ) :
    ‖terminal3 ((k * speed / m * taper lengthSq (speed * dt)) • direction) m k‖ ≤ speed := by
  obtain ⟨h₀, h₁⟩ := taper_bounds hl (speed * dt)
  rw [terminal3, smul_smul, show m / k * (k * speed / m * taper lengthSq (speed * dt)) =
    speed * taper lengthSq (speed * dt) by field_simp, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (mul_nonneg hs h₀)]
  exact (mul_le_of_le_one_right (mul_nonneg hs h₀) hdir).trans (mul_le_of_le_one_right hs h₁)

/-- Real interpretation of the scalar reference's zero-guarded reciprocal normalization. -/
def gotoAcceleration (offset : Vec3) (m k speed dt : ℝ) : Vec3 :=
  let direction := if ‖offset‖ = 0 then offset else (1 / ‖offset‖) • offset
  (k * speed / m * taper (‖offset‖ ^ 2) (speed * dt)) • direction

theorem goto_speed_bound (offset v : Vec3) {m k speed dt : ℝ} (hm : 0 < m)
    (hk : 0 < k) (hs : 0 ≤ speed) (ht : 0 ≤ dt) (hv : ‖v‖ ≤ speed) :
    ‖velocity3 v (gotoAcceleration offset m k speed dt) m k dt‖ ≤ speed := by
  apply speed_envelope hm hk ht hv
  unfold gotoAcceleration
  apply goto_terminal_bound _ (sq_nonneg _) hm hk hs
  split
  next h => simp [h]
  next h => simp [norm_smul, h]

/-- Binary64 values of the two STOP literals interpreted in ℝ. -/
def stopVertical (y : ℝ) :=
  (y - (1261007895663739 / 18014398509481984) * y) * (8417943228729899 / 9007199254740992)

theorem stop_speed_sq {m k t : ℝ} (hm : 0 < m) (hk : 0 < k) (ht : 0 ≤ t) (x y z : ℝ) :
    velocity x 0 m k t ^ 2 + velocity (stopVertical y) 0 m k t ^ 2 + velocity z 0 m k t ^ 2 ≤
    x ^ 2 + y ^ 2 + z ^ 2 := by
  have hq := decay_bounds hm hk ht
  have hsq : decay m k t ^ 2 ≤ 1 := by nlinarith [mul_nonneg hq.1.le (sub_nonneg.mpr hq.2)]
  have hy : stopVertical y ^ 2 ≤ y ^ 2 := by
    dsimp [stopVertical]
    nlinarith [sq_nonneg y]
  simp only [velocity, terminal, mul_zero, sub_zero, zero_add, mul_pow]
  calc
    _ = decay m k t ^ 2 * (x ^ 2 + stopVertical y ^ 2 + z ^ 2) := by ring
    _ ≤ 1 * (x ^ 2 + stopVertical y ^ 2 + z ^ 2) := mul_le_mul_of_nonneg_right hsq (by positivity)
    _ ≤ _ := by linarith

end
end ExactBallpark.Movement

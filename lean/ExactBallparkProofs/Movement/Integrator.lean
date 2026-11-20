-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026 Eidosoma
module

public import Mathlib.Analysis.SpecialFunctions.ExpDeriv
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Ring

/-! Real interpretation of one Cartesian component of `ref.Ballpark.integrate`.
The operation order on the source side also records reciprocal multiplications. -/

@[expose] public section

namespace ExactBallpark.Movement

noncomputable section

def effectiveMass (mass agility : ℝ) := mass * agility

def decay (m k t : ℝ) := Real.exp ((-k / m) * t)

def refreshFactor (mass agility k dt : ℝ) :=
  let m := effectiveMass mass agility
  if m ≤ 0 then 0 else Real.exp (-k * dt / m)

def stepFactor (m k dt cached t : ℝ) := if t ≠ dt then decay m k t else cached

theorem refresh_positive {mass agility : ℝ} (hm : 0 < effectiveMass mass agility) (k dt : ℝ) :
    refreshFactor mass agility k dt = decay (effectiveMass mass agility) k dt := by
  simp only [refreshFactor, ite_eq_right (not_le.mpr hm), decay]
  congr 1
  ring

/-- Only a full tick needs a valid cached factor. -/
theorem stepFactor_eq {m k dt cached t : ℝ} (hcache : t = dt → cached = decay m k dt) :
    stepFactor m k dt cached t = decay m k t := by
  by_cases h : t = dt
  · simpa [stepFactor, h] using hcache h
  · simp [stepFactor, h]

def regularPosition (p v a m k q t : ℝ) :=
  let k2 := k * k
  (m * (m * a * (q - 1) + k * (a * t + v - q * v)) + p * k2) * (1 / k2)

def sourceVelocity (v a m k q : ℝ) := (m * a - (m * a - v * k) * q) * (1 / k)

def terminal (m k a : ℝ) := m / k * a

def velocity (v a m k t : ℝ) := terminal m k a + decay m k t * (v - terminal m k a)

def position (p v a m k t : ℝ) :=
  p + terminal m k a * t + m / k * (1 - decay m k t) * (v - terminal m k a)

theorem source_regular {k : ℝ} (hk : k ≠ 0) (p v a m t : ℝ) :
    (regularPosition p v a m k (decay m k t) t, sourceVelocity v a m k (decay m k t)) =
    (position p v a m k t, velocity v a m k t) := by
  apply Prod.ext <;> dsimp [regularPosition, sourceVelocity, position, velocity, terminal] <;>
    field_simp <;> ring

theorem flow_zero (p v a m k : ℝ) :
    (position p v a m k 0, velocity v a m k 0) = (p, v) := by
  simp [position, velocity, decay]

theorem decay_hasDerivAt (m k t : ℝ) :
    HasDerivAt (decay m k) (decay m k t * (-k / m)) t := by
  unfold decay
  simpa using (((hasDerivAt_id t).const_mul (-k / m)).exp)

theorem velocity_hasDerivAt {m k : ℝ} (hm : m ≠ 0) (hk : k ≠ 0) (v a t : ℝ) :
    HasDerivAt (velocity v a m k) (a - k / m * velocity v a m k t) t := by
  have h := ((decay_hasDerivAt m k t).mul_const (v - terminal m k a)).const_add (terminal m k a)
  convert h using 1 <;> try rfl
  dsimp [velocity, terminal]
  field_simp
  ring

theorem position_hasDerivAt {m k : ℝ} (hm : m ≠ 0) (hk : k ≠ 0) (p v a t : ℝ) :
    HasDerivAt (position p v a m k) (velocity v a m k t) t := by
  have linear := ((hasDerivAt_id t).const_mul (terminal m k a)).const_add p
  have correction := (((decay_hasDerivAt m k t).const_sub 1).const_mul (m / k)).mul_const
    (v - terminal m k a)
  convert linear.add correction using 1 <;> try rfl
  dsimp [velocity]
  field_simp

theorem flow_compose (p v a m k s t : ℝ) :
    (position (position p v a m k s) (velocity v a m k s) a m k t,
      velocity (velocity v a m k s) a m k t) =
    (position p v a m k (s + t), velocity v a m k (s + t)) := by
  apply Prod.ext <;> dsimp [position, velocity, decay] <;> rw [mul_add (-k / m), Real.exp_add] <;> ring

end
end ExactBallpark.Movement
